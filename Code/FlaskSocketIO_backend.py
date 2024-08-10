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


if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
tts_model = None
stt_model = None
ttt_model = None

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
        instance =Klass()
    return instance

with open(main_path+'/params.json') as params_file:
    params = json.load(params_file)
    
def load_models(params):
    """Loads all there models: Speech-to-Text, Text-to-Text, Text-to-Speech models.
    Returns:
        stt_model (STTStrategy): Speech-to-Text model, default = WhisperLargeV2.
        ttt_model (TTTStrategy): Text-to-Text model, default = FastChatModel.
        tts_model (TTSStrategy): Text-to-Speech model default = XTTS_V2.
        tti_model (TTIStrategy): Text-to-Image deffusion model, default = StableDiffusion
    """
    # print("\nLoading Whisper ...\n")
    # stt_model = WhisperLargeV2()
    stt_model = get_instance(params["stt_model"])
    # print("\nLoading FastChat ...\n")
    # ttt_model = FastChatModel()
    ttt_model = get_instance(params["ttt_model"], params["text_generation_system_prompt"])
    
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


stt_model, ttt_model, tti_model, gtts_model, etts_model, default_language = load_models(params)


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
    ttt_model.clear_history()
    print("Client disconnected")


@app.route("/reset", methods=["POST", "GET"])
def reset():
    """Triggered on reset request. Clears chat history.
    Args:
        sid (int): Clients id, assigned automatically by the socket server.
    """
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
    global params
    first_response = True
    logs = {}
    start_time = time.time()
    current_time = start_time
    # print("########### generating ###########")
    # transcribtion = stt_model.run(voice_request)
    transcribtion = stt_model.run(voice_request, language=params["conversation_language"])
    sio.emit("chat", "========================")
    sio.emit("chat", "USER  >>>  "+transcribtion)
    transcribtion_time = time.time() - current_time
    print(transcribtion)
    # print(voice_query)
    entry = ""
    response = []
    ttt_times = []
    tts_times = []
    current_time = time.time()
    # image_generation_keywords_nouns = set(params["image_generation_keywords_nouns"])
    # image_generation_keywords_verbs = set(params["image_generation_keywords_verbs"])
    # prompt_words = set(transcribtion.split())

    
    # if image_generation_keywords_nouns & prompt_words and image_generation_keywords_verbs & prompt_words:
    plot_request = False
    for noun in params["image_generation_keywords_nouns"]:
        if noun in transcribtion:
            for verb in params["image_generation_keywords_verbs"]:
                if verb in transcribtion:
                    plot_request = True
                    break
            if plot_request:
                break
    if plot_request:
        print("Plotting Request")
        image_np = np.array(tti_model.run(transcribtion.strip()+ " " + params["image_generation_system_prompt"]))
        image_bytes = image_np.tobytes()
        
        if params["conversation_language"] == "multi":
            lang = detect(transcribtion) 
            if lang not in ["de", "en"]:
                lang = default_language
        else:
            lang = params["conversation_language"]
        
        if lang == "de":
            # print("#####\n",lang,"\n#####\n")
            sio.emit("chat", data= "ALVI  >>>  Hast du weitere Anfragen?")
            # voice_answer = gtts_model.run(entry)
            voice_answer = gtts_model.run(entry, language='de')
        else:
            # print("#####\n",lang,"\n#####\n")
            sio.emit("chat", data= "ALVI  >>>  Do you have another request?")
            # voice_answer = etts_model.run("Do you have another request?")
            voice_answer = etts_model.run("Do you have another request?", language='en')
        sio.emit("image", data= {"image_bytes": image_bytes})
        b_answer = voice_answer.tobytes()
        data = {"voice_answer": b_answer}
        sio.emit("voice reply", data)
    else:
        for out in ttt_model.run(transcribtion.strip()):
            ttt_times.append(time.time()-current_time)
            if len(out)<1 or out.isspace():
                continue
            if out == "END":
                print("\n##############\n")
                print(entry)
                print("\n##############\n")
                current_time = time.time()
                if len(entry)<1:
                    entry = "Entschuldigung, ich konnte deine Anfrage nicht beantworten. "
                    continue
                sio.emit("chat", data= "ALVI  >>>  "+entry)
                # voice_answer = tts_model.run(entry)
                
                if params["conversation_language"] == "multi":
                    lang = detect(entry)
                    if lang not in ["de", "en"]:
                        lang = default_language
                else:
                    lang = params["conversation_language"]
                if lang == "de":
                    print("#####\n",lang,"\n#####\n")
                    voice_answer = gtts_model.run(entry)
                else:
                    print("#####\n",lang,"\n#####\n")
                    # voice_answer = etts_model.run(entry)
                    voice_answer = etts_model.run(entry, language='en')
                    
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
                    print("\n##############\n")
                    print(entry)
                    print("\n##############\n")
                    current_time = time.time()
                    sio.emit("chat", data= "ALVI >>> "+entry)
                    # voice_answer = tts_model.run(entry)
                    if params["conversation_language"] == "multi":
                        lang = detect(entry)
                        if lang not in ["de", "en"]:
                            lang = default_language
                    else:
                        lang = params["conversation_language"]
                        
                    if lang == "de":
                        # print("#####\n",lang,"\n#####\n")
                        voice_answer = gtts_model.run(entry)
                    else:
                        # print("#####\n",lang,"\n#####\n")
                        voice_answer = etts_model.run(entry, language='en')
                    tts_times.append(time.time()-current_time)
                    
                    if first_response:
                        logs["first_response"] = sum(ttt_times) + sum(tts_times)
                        first_response = False
                        
                    b_answer = voice_answer.tobytes()
                    data = {"voice_answer": b_answer}
                    response.append(entry)
                    entry = out + " "
                    data = {"voice_answer": b_answer}
                    sio.emit("voice reply", data)
                else:
                    entry += out + " "
            current_time = time.time()    
        
    logs["stt_time"] = transcribtion_time
    logs["ttt_times"] = ttt_times
    logs["tts_times"] = tts_times
    logs["total_time"] = time.time() - start_time
    logs_bytes = json.dumps(logs, indent=2).encode('utf-8')
    sio.emit("request", data = logs_bytes)

@app.route("/request", methods=["POST", "GET"])
def receive():
    """Receives the voice request from client, creates a new thread to process it.
    Returns:
        (dict): Resonse message containing the emitting time.
    """
    received_time = time.time()
    print("\nRequest is recieved\n")
    bytes_data = request.get_data()
    voice_query = bytes_data
    t = threading.Thread(target=generateAnswer, daemon=True, args=[voice_query])
    t.start()
    return {"response": received_time}

if __name__ == '__main__':
    
    sio.run(app, host="0.0.0.0", port="5000",debug=False, use_reloader=False, log_output=False)
