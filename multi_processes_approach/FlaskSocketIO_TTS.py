from Models import *
import json
import os
import time

from flask import Flask, request
from flask_socketio import SocketIO
import threading
import logging
import queue
import requests
from langdetect import detect, DetectorFactory

main_path = os.path.dirname(os.path.realpath(__file__))
logs_path = main_path + "/logs/"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
tts_model = None

app = Flask(__name__)
sio = SocketIO(app, async_mode = "threading")

query_queue = queue.Queue()
active = True
ready = True

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
    """Loads Text-to-Speech model.
    Returns:
        etts_model (TTSStrategy): English Text-to-Speech model default = XTTS_V2.
        gtts_model (TTSStrategy): German Text-to-Speech model default = ThorstenVits.
    """
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
    
    return gtts_model, etts_model, default_language


# gtts_model, etts_model, default_language = load_models(params)


@sio.on("connect")
def connect():
    """Triggered when a client connect.
    Emits back a signal to the connected client.
    """
    t = threading.Thread(target=generateAnswer, daemon=True)
    t.start()
    print('Client connected')

@sio.on("disconnect")
def disconnect():
    """Triggered when a client disconnect.
    Clears chat history as well.
    """
    active = False
    url="http://127.0.0.1:5000/disconnect"
    response = requests.get(url)
    print("Client disconnected")


def generateAnswer():
    global ready, params, gtts_model, etts_model, default_language
    while active:
        first_response = True
        logs = {}
        start_time = time.time()
        entry = ""
        response = []
        ttt_times = []
        tts_times = []
        current_time = time.time()
        if not query_queue.empty():
            ready = False
            out = query_queue.get_nowait()
            
            if params["conversation_language"] == "multi":
                lang = detect(out)
                if lang not in ["de", "en"]:
                    lang = default_language
            else:
                lang = params["conversation_language"]
                
            if lang == "de":
                # print("#####\n",lang,"\n#####\n")
                voice_answer = gtts_model.run(out)
            else:
                # print("#####\n",lang,"\n#####\n")
                voice_answer = etts_model.run(out, language='en')
                
            b_answer = voice_answer.tobytes()
            data = {"voice_answer": b_answer}
            sio.emit("voice reply", data)
        elif not ready:
            ready = True
            sio.emit("request", data = {})



@app.route("/request_finished", methods=["POST", "GET"])
def requestAgain():
    sio.emit("request", data = {})
    return {"response": "response"}

@app.route("/request", methods=["POST", "GET"])
def receive():
    """Receives the voice request from client, creates a new thread to process it.
    Returns:
        (dict): Resonse message containing the emitting time.
    """
    received_time = time.time()
    print("\nRequest is recieved\n")
    text_query = request.json
    print(text_query)
    query_queue.put_nowait(text_query)
    return {"response": received_time}

@app.route("/request_image", methods=["POST", "GET"])
def receive_image():
    """Receives the voice request from client, creates a new thread to process it.
    Returns:
        (dict): Resonse message containing the emitting time.
    """
    received_time = time.time()
    print("\nRequest is recieved\n")
    image = request.data
    #print(image)
    sio.emit("image", data= {"image_bytes": image})
    query_queue.put_nowait("Hast du weitere Anfragen?")
    return {"response": received_time}

@app.route("/request_welcome_message", methods=["POST", "GET"])
def generateWM():
    """Receives the text of the welcome message, generates an audio of it and save the audio in welcome_message.wav.
    Returns:
        (dict): Resonse message containing the emitting time.
    """
    global gtts_model, etts_model
    received_time = time.time()
    print("\nRequest is recieved\n")
    query = request.json
    # query_queue.put_nowait(query)
    if params["conversation_language"]=="en":
        etts_model.run_to_file(query, language="en", filepath="welcome_message.wav")
    else:
        gtts_model.run_to_file(query, filepath="welcome_message.wav")
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
    
    gtts_model, etts_model, default_language = load_models(params)
    sio.run(app, host="0.0.0.0", port="8080",debug=False, use_reloader=False, log_output=False)
