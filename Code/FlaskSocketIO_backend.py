from Models import *
import json
import os
import time

from flask import Flask, request
from flask_socketio import SocketIO
import threading
import logging
from langdetect import detect, DetectorFactory


main_path = os.path.dirname(os.path.realpath(__file__))
logs_path = main_path + "/logs/"
DetectorFactory.seed = 0

ABBREVIATION_MAP = {
    "z.B.": "zum Beispiel",
    "z. B.": "zum Beispiel",
    "u.a.": "unter anderem",
    "u. a.": "unter anderem",
    "etc.": "et cetera",
    "bzw.": "beziehungsweise",
    "i.d.R.": "in der Regel",
    "v.a.": "vor allem",
    "d.h.": "das heißt",
    "d. h.": "das heißt",
    "vgl.": "vergleiche",
    "ca. ": "circa ",  # extra white space to avoid end of sentence
    "OK,": "Okay,",
    "ok,": "okay,",
}


if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    

app = Flask(__name__)
sio = SocketIO(app, async_mode = "threading")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

entry_length = 300


def get_instance(name, args=None):
    Klass = globals()[name]
    if args is not None:
        instance = Klass(args)
    else:
        instance = Klass()
    return instance

# with open(main_path+'/params.json') as params_file:
#     params = json.load(params_file)
    
def load_models(params):
    """Loads all there models: Speech-to-Text, Text-to-Text, Text-to-Speech models.
    Returns:
        stt_model (STTStrategy): Speech-to-Text model, default = WhisperLargeV2.
        ttt_model (TTTStrategy): Text-to-Text model, default = FastChatModel.
        etts_model (TTSStrategy): English Text-to-Speech model default = XTTS_V2.
        gtts_model (TTSStrategy): German Text-to-Speech model default = ThorstenVits.
        tti_model (TTIStrategy): Text-to-Image deffusion model, default = StableDiffusion
    """
    # print("\nLoading Whisper ...\n")
    # stt_model = WhisperLargeV2()
    stt_model = get_instance(params["stt_model"])
    # print("\nLoading FastChat ...\n")
    # ttt_model = FastChatModel()
    ttt_model = get_instance(params["ttt_model"], params)
    
    # print("\nLoading StableDiffusion ...\n")
    # tts_model = StableDiffusion()
    tti_model = get_instance(params["tti_model"])
    
    
    # print("\nLoading XTTS_V2 ...\n")
    # tts_model = XTTS_V2()
    if params["conversation_language"]=="multi":
        gtts_model = get_instance(params["gtts_model"])
        etts_model = get_instance(params["etts_model"])
        default_language = params["default_language"]
        
    elif params["conversation_language"]=="de":
        gtts_model = get_instance(params["gtts_model"])
        etts_model = None
        default_language = "de"
        
    else:
        etts_model = get_instance(params["etts_model"])
        gtts_model = None
        default_language = "en"
    
    return stt_model, ttt_model, tti_model, gtts_model, etts_model, default_language


@sio.on("connect")
def connect():
    """Triggered when a client connect.
    Emits back a signal to the connected client.
    """
    print('Client connected')

@sio.on("disconnect")
def disconnect():
    """Triggered when a client disconnect.
    Clears chat history as well.
    """
    global ttt_model
    ttt_model.clear_history()
    print("Client disconnected")


@app.route("/reset", methods=["POST", "GET"])
def reset():
    """Triggered on reset request. Clears chat history.
    Args:
        sid (int): Clients id, assigned automatically by the socket server.
    """
    global ttt_model
    ttt_model.clear_history()
    print(" \nChat history is reset. \n#####")    
    return "Reset"

def generateAnswer(voice_request):
    """Takes audio bytes as input and processes it follows: 
        1- Transcribes it using the Speech-To-Text model.
        2- Query the Text-to-Text model on the transcribed text from 1.
        3- Generates speech on each generated chunck using the Text-to-Speech model.
        4- Emits the generated speech to the client.
        5- When generating is finished, a 'request' emit to notify the client to record a new voice request.
    Args:
        voice_request (bytes): Recorded voice request from client.
    """
    global params, stt_model, ttt_model, etts_model, gtts_model,tti_model
    first_response = True
    logs = {}
    start_time = time.time()
    current_time = start_time
    # print("########### generating ###########")
    # transcription = stt_model.run(voice_request)
    transcription, transcription_success = stt_model.run(voice_request, language=params["conversation_language"])

    if transcription_success:
        sio.emit("chat", {"message": transcription.strip(), "sender": "You"})
    else:
        sio.emit("chat", {"message": "[...]", "sender": "You"})

    transcription_time = time.time() - current_time
    print(transcription)
    # print(voice_query)
    entry = ""
    response = []
    ttt_times = []
    tts_times = []
    current_time = time.time()
    # image_generation_keywords_nouns = set(params["image_generation_keywords_nouns"])
    # image_generation_keywords_verbs = set(params["image_generation_keywords_verbs"])
    # prompt_words = set(transcription.split())

    
    # if image_generation_keywords_nouns & prompt_words and image_generation_keywords_verbs & prompt_words:
    plot_request = False
    for noun in params["image_generation_keywords_nouns"]:
        if noun in transcription:
            for verb in params["image_generation_keywords_verbs"]:
                if verb in transcription:
                    plot_request = True
                    break
            if plot_request:
                break

    def determine_language(text):
        if params["conversation_language"] == "multi":
            lang = detect(text)
            if lang not in ["de", "en"]:
                lang = default_language
        else:
            lang = params["conversation_language"]
        return lang
    def generate_voice_answer(text):
        lang = determine_language(text)

        if lang == "de":
            voice_answer = gtts_model.run(text)
        else:
            voice_answer = etts_model.run(text, language='en')
        return voice_answer

    def post_process_text(text):
        for abbr, full in ABBREVIATION_MAP.items():
            text = text.replace(abbr, full)
        return text

    if plot_request:
        print("Plotting Request")
        image_np = np.array(tti_model.run(transcription.strip()+ " " + params["image_generation_system_prompt"]))
        image_bytes = image_np.tobytes()

        lang = determine_language(transcription)
        
        if lang == "de":
            # print("#####\n",lang,"\n#####\n")
            sio.emit("chat", data={"message": "Hast du weitere Anfragen?", "sender": "AI"})
            voice_answer = gtts_model.run("Hast du weitere Anfragen?")
        else:
            # print("#####\n",lang,"\n#####\n")
            sio.emit("chat", data={"message": "Do you have another request?", "sender": "AI"})
            # voice_answer = etts_model.run("Do you have another request?")
            voice_answer = etts_model.run(text="Do you have another request?", language='en')
        sio.emit("image", data={"image_bytes": image_bytes})
        b_answer = voice_answer.tobytes()
        data = {"voice_answer": b_answer}
        sio.emit("voice reply", data)
    else:
        if transcription_success:
            generator = ttt_model.run(transcription.strip())
        else:
            generator = [transcription, "END"]

        for out in generator:
            ttt_times.append(time.time()-current_time)
            if len(out)<1 or out.isspace():
                continue
            if out == "END":
                if len(entry)<1:
                    entry = params["not_able_to_respond_message"]
                    sio.emit("chat", data={"message": entry, "sender": "AI"})
                    print(entry)
                    continue
                entry = post_process_text(entry)
                print("\n##############\n")
                print(entry)
                print("\n##############\n")
                current_time = time.time()

                sio.emit("chat", data={"message": entry, "sender": "AI"})
                # voice_answer = tts_model.run(entry)

                voice_answer = generate_voice_answer(entry)

                tts_times.append(time.time()-current_time)
                if first_response:
                    logs["first_response"] = sum(ttt_times) + sum(tts_times)
                    first_response = False
                b_answer = voice_answer.tobytes()
                response.append(entry)
                data = {"voice_answer": b_answer}
                sio.emit("voice reply", data)
            else:
                if (len(entry) + len(out) + 1) >= entry_length:
                    entry = post_process_text(entry)

                    print("\n##############\n")
                    print(entry)
                    print("\n##############\n")
                    current_time = time.time()
                    sio.emit("chat", data={'message': entry, 'sender': 'AI'})

                    voice_answer = generate_voice_answer(entry)
                    tts_times.append(time.time()-current_time)
                    
                    if first_response:
                        logs["first_response"] = sum(ttt_times) + sum(tts_times)
                        first_response = False
                        
                    b_answer = voice_answer.tobytes()
                    response.append(entry)
                    entry = out + " "
                    data = {"voice_answer": b_answer}
                    sio.emit("voice reply", data)
                else:
                    entry += out + " "
            current_time = time.time()    
        
    logs["stt_time"] = transcription_time
    logs["ttt_times"] = ttt_times
    logs["tts_times"] = tts_times
    logs["total_time"] = time.time() - start_time
    logs_bytes = json.dumps(logs, indent=2).encode('utf-8')
    sio.emit("request", data=logs_bytes)

@app.route("/request", methods=["POST", "GET"])
def receive():
    """Receives the voice request from client, creates a new thread to process it.
    Returns:
        (dict): Response message containing the emitting time.
    """
    received_time = time.time()
    print("\nRequest is received\n")
    bytes_data = request.get_data()
    voice_query = bytes_data
    t = threading.Thread(target=generateAnswer, daemon=True, args=[voice_query])
    t.start()
    return {"response": received_time}



@app.route("/request_welcome_message", methods=["POST", "GET"])
def generateWM():
    """Receives the text of the welcome message, generates an audio of it and save the audio in welcome_message.wav.
    Returns:
        (dict): Response message containing the emitting time.
    """
    global etts_model, gtts_model
    received_time = time.time()
    print("\nRequest is recieved\n")
    query = request.get_json()
    print(query)

    output_file = "welcome_message.wav"
    if params["conversation_language"]=="en":
        etts_model.run_to_file(query, language="en", filepath=output_file)
    else:
        gtts_model.run_to_file(query, filepath=output_file)
    return {"response": received_time}

if __name__ == '__main__':
    default_json_path = main_path + '/params.json'
    parser = argparse.ArgumentParser(description="Process a params file.")
    parser.add_argument(
        'json_file',
        type=str,
        nargs='?',  # Makes the argument optional
        default=default_json_path,  # Default value if no argument is passed
        help=f"Path to the JSON file (default: {main_path}/params.json)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get absolute path of the json file
    params_file_path = os.path.abspath(args.json_file)
    
    
    with open(params_file_path) as params_file:
        params = json.load(params_file)
        params_file.close()
    
    stt_model, ttt_model, tti_model, gtts_model, etts_model, default_language = load_models(params)
    sio.run(app, host="0.0.0.0", port="5000",debug=False, use_reloader=False, log_output=False)
