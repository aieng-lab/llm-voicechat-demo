from Models import *
import json
import os
import time

from flask import Flask, request
from flask_socketio import SocketIO
import threading
import logging
import re
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
    "*": "", # avoid markdown
    "§": "Paragraph",
    "§§": "Paragraphen",
    "Art.": "Artikel",
}


if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    

app = Flask(__name__)
sio = SocketIO(app, async_mode = "threading")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

entry_length = 400


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
    params = apply_api_model_selection(params)

    # print("\nLoading Whisper ...\n")
    # stt_model = WhisperLargeV2()
    stt_model = get_instance(params["stt_model"]["active"])
    # print("\nLoading FastChat ...\n")
    # ttt_model = FastChatModel()
    ttt_model = get_instance(params["ttt_model"]["active"], params)
    
    # print("\nLoading StableDiffusion ...\n")
    # tts_model = StableDiffusion()
    tti_model = get_instance(params["tti_model"]["active"])
    
    
    # print("\nLoading XTTS_V2 ...\n")
    # tts_model = XTTS_V2()
    if params["conversation_language"]=="multi":
        gtts_model = get_instance(params["gtts_model"]["active"])
        etts_model = get_instance(params["etts_model"]["active"])
        default_language = params["default_language"]
        
    elif params["conversation_language"]=="de":
        gtts_model = get_instance(params["gtts_model"]["active"])
        etts_model = None
        default_language = "de"
        
    else:
        etts_model = get_instance(params["etts_model"]["active"])
        gtts_model = None
        default_language = "en"
    
    return stt_model, ttt_model, tti_model, gtts_model, etts_model, default_language



def is_anthropic_model(model_id):
    model_id = model_id.lower()
    return model_id.startswith("claude")


def is_openai_model(model_id):
    model_id = model_id.lower()
    return (
        model_id.startswith("gpt")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
    )


def apply_api_model_selection(params, model_id=None):
    """
    Selects the API backend automatically from a model id.

    If params["api_models"] exists, model_id is interpreted as an API model id,
    e.g. gpt-4o-mini or claude-3-5-sonnet-20241022.
    """

    api_models = params.get("api_models")

    if not api_models:
        return params

    if model_id is None:
        model_id = params.get("active_api_model")

    if model_id is None:
        model_id = next(iter(api_models.keys()))

    if model_id not in api_models:
        raise ValueError(f"Unknown API model id: {model_id}")

    params["active_api_model"] = model_id

    if is_anthropic_model(model_id):
        params["ttt_model"]["active"] = "ClaudeAPI"
        params["anthropic_model"] = model_id

    elif is_openai_model(model_id):
        params["ttt_model"]["active"] = "OpenAIAPI"
        params["openai_model"] = model_id

    else:
        raise ValueError(
            f"Could not determine API provider for model id: {model_id}. "
            "Expected OpenAI model id or Claude model id."
        )

    return params


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
    global params, stt_model, ttt_model, etts_model, gtts_model, tti_model

    logs = {}
    ttt_times = []
    tts_times = []

    first_response = True
    audio_id = 0

    start_time = time.time()
    current_time = start_time

    # ----------------------------
    # STT
    # ----------------------------
    transcription, transcription_success = stt_model.run(
        voice_request,
        language=params["conversation_language"]
    )

    transcription_time = time.time() - current_time
    current_time = time.time()

    user_message = transcription.strip() if transcription_success else "[...]"
    sio.emit("chat", {
        "message": user_message,
        "sender": "You"
    })

    # ----------------------------
    # Helpers
    # ----------------------------
    def determine_language(text):
        if params["conversation_language"] == "multi":
            try:
                lang = detect(text)
            except Exception:
                lang = params.get("default_language", "en")

            return lang if lang in ["de", "en"] else params.get("default_language", "en")

        return params["conversation_language"]

    def generate_voice_answer(text):
        lang = determine_language(text)

        if lang == "de":
            return gtts_model.run(text)

        return etts_model.run(text, language="en")

    def post_process_tts_text(text):
        for abbr, full in ABBREVIATION_MAP.items():
            text = text.replace(abbr, full)
        return text

    def post_process_chat_text(text):
        return text.replace("*", "").strip()
    
    def split_and_merge_chat_sentences(chat_text):
        raw_sentences = re.split(r'(?<=[.!?])\s+', chat_text.strip())
        raw_sentences = [s.strip() for s in raw_sentences if s.strip()]

        abbreviation_keys = sorted(
            ABBREVIATION_MAP.keys(),
            key=len,
            reverse=True
        )

        # 1. Re-merge false sentence splits after abbreviations, e.g. "Art."
        abbreviation_safe_sentences = []

        for part in raw_sentences:
            if (
                abbreviation_safe_sentences
                and any(abbreviation_safe_sentences[-1].endswith(abbr.strip()) for abbr in abbreviation_keys)
            ):
                abbreviation_safe_sentences[-1] += " " + part
            else:
                abbreviation_safe_sentences.append(part)

        # 2. Merge very short sentences with the following one
        merged = []
        i = 0

        while i < len(abbreviation_safe_sentences):
            current = abbreviation_safe_sentences[i]

            if len(current) < 20 and i + 1 < len(abbreviation_safe_sentences):
                merged.append(current + " " + abbreviation_safe_sentences[i + 1])
                i += 2
            else:
                merged.append(current)
                i += 1

        return merged
    
    def emit_sentence(chat_sentence):
        nonlocal first_response, audio_id, current_time

        chat_sentence = chat_sentence.strip()
        if not chat_sentence:
            return

        tts_sentence = post_process_tts_text(chat_sentence)

        tts_start = time.time()
        voice_answer = generate_voice_answer(tts_sentence)
        tts_times.append(time.time() - tts_start)

        sio.emit("voice reply", {
            "id": audio_id,
            "voice_answer": voice_answer.tobytes()
        })

        # Chat erst NACH der Audio-Message senden
        sio.emit("chat", {
            "message": chat_sentence,
            "sender": "AI"
        })

        audio_id += 1

        if first_response:
            logs["first_response"] = sum(ttt_times) + sum(tts_times)
            first_response = False

        current_time = time.time()

    def emit_entry(entry_text):
        entry_text = entry_text.strip()
        if not entry_text:
            return

        chat_text = post_process_chat_text(entry_text)
        if not chat_text:
            return

        chat_sentences = split_and_merge_chat_sentences(chat_text)

        for chat_sentence in chat_sentences:
            emit_sentence(chat_sentence)

    # ----------------------------
    # Image detection
    # ----------------------------
    plot_request = (
        transcription_success
        and any(noun in transcription for noun in params["image_generation_keywords_nouns"])
        and any(verb in transcription for verb in params["image_generation_keywords_verbs"])
    )

    # ----------------------------
    # IMAGE MODE
    # ----------------------------
    if plot_request:
        image_np = np.array(
            tti_model.run(
                transcription.strip()
                + " "
                + params["image_generation_system_prompt"]
            )
        )

        sio.emit("image", {
            "image_bytes": image_np.tobytes()
        })

        answer_text = (
            "Hast du weitere Anfragen?"
            if determine_language(transcription) == "de"
            else "Do you have another request?"
        )

        emit_sentence(answer_text)

    # ----------------------------
    # TEXT MODE
    # ----------------------------
    else:
        generator = (
            ttt_model.run(transcription.strip())
            if transcription_success
            else [params["not_able_to_respond_message"], "END"]
        )

        entry = ""

        for out in generator:
            ttt_times.append(time.time() - current_time)
            current_time = time.time()

            if not out or out.isspace():
                continue

            if out == "END":
                if not entry.strip():
                    entry = params["not_able_to_respond_message"]

                emit_entry(entry)
                entry = ""
                break

            next_entry = entry + out + " "

            if len(next_entry) >= entry_length:
                emit_entry(entry)
                entry = out + " "
            else:
                entry = next_entry

        # Safety fallback if generator ends without explicit END
        if entry.strip():
            emit_entry(entry)

    # ----------------------------
    # LOGS
    # ----------------------------
    if "first_response" not in logs:
        logs["first_response"] = None

    logs["stt_time"] = transcription_time
    logs["ttt_times"] = ttt_times
    logs["tts_times"] = tts_times
    logs["total_time"] = time.time() - start_time

    sio.emit("request", json.dumps(logs).encode("utf-8"))

@app.route("/api_models", methods=["GET"])
def api_models():
    global params

    return {
        "models": params.get("api_models", {}),
        "active": params.get("active_api_model")
    }

@app.route("/switch_api_model", methods=["POST"])
def switch_api_model():
    global params, ttt_model

    data = request.get_json()
    model_id = data["model_id"]

    apply_api_model_selection(params, model_id)

    ttt_model = get_instance(params["ttt_model"]["active"], params)
    ttt_model.clear_history()

    print(
        f"API model switched to {model_id} "
        f"via {params['ttt_model']['active']}"
    )

    sio.emit("chat", {
        "message": f"[Modell zu {params['api_models'][model_id]} gewechselt]",
        "sender": "System"
    })

    return {
        "status": "ok",
        "active": model_id,
        "pretty_name": params["api_models"][model_id],
        "backend": params["ttt_model"]["active"]
    }




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
    
@app.route("/switch_model", methods=["POST"])
def switch_model():
    global params, stt_model, ttt_model, tti_model, gtts_model, etts_model

    data = request.get_json()
    model_type = data["model_type"]
    model_name = data["model_name"]

    params[model_type]["active"] = model_name

    if model_type == "stt_model":
        stt_model = get_instance(model_name)
    elif model_type == "ttt_model":
        ttt_model = get_instance(model_name, params)
    elif model_type == "tti_model":
        tti_model = get_instance(model_name)
    elif model_type == "gtts_model":
        gtts_model = get_instance(model_name)
    elif model_type == "etts_model":
        etts_model = get_instance(model_name)

    print(f"{model_type} switched to {model_name}")

    return {"status": "ok"}

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
