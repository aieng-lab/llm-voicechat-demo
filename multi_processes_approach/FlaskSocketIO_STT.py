from Models import *
import json
import os
import time

from flask import Flask, request
# from flask_socketio import SocketIO
import threading
import logging
import requests

main_path = os.path.dirname(os.path.realpath(__file__))
logs_path = main_path + "/logs/"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
stt_model = None

app = Flask(__name__)


entry_length = 300


def get_instance(name, args=None):
    Klass = globals()[name]
    if args is not None:
        instance = Klass(args)
    else:
        instance =Klass()
    return instance

# with open(main_path+'/params.json') as params_file:
#     params = json.load(params_file)
    
def load_models(params):
    """Loads Speech-to-Text, Text-to-Text models.
    Returns:
        (WhisperLargeV2): Speech-to-Text model.
    """
    # print("\nLoading Whisper ...\n")
    # stt_model = WhisperLargeV2()
    stt_model = get_instance(params["stt_model"])
    return stt_model


# stt_model = load_models(params)



@app.route("/disconnect", methods=["POST", "GET"])
def disconnect():
    """Triggered when a client disconnect.
    Clears chat history as well.
    """
    
    print("Client disconnected")


def transcribe(voice_request):
    """Takes audio bytes as input and processes it follows: 
        1- Transcribes it using the Speech-To-Text model.
        2- Query the Text-to-Text model on the transcribed text from 1.
        3- Generates speech on each generated chunck using the Text-to-Speech model.
        4- Emits the generated speech to the client.
        5- When generating is finished, a 'request' emit to notify the client to record a new voice request.
    Args:
        voice_request (bytes): Recorded voice request from client.
    """
    global params, stt_model
    url_image="http://127.0.0.1:9090/request"
    url="http://127.0.0.1:7070/request"
    first_response = True
    # print("########### generating ###########")
    transcribtion = stt_model.run(voice_request, language=params["conversation_language"])
    print(transcribtion)
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
        resp = requests.get(url_image, json=json.dumps(transcribtion, ensure_ascii=False))
        plot_request = False
    else:
        resp = requests.get(url, json=json.dumps(transcribtion, ensure_ascii=False))

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
    t = threading.Thread(target=transcribe, daemon=True, args=[voice_query])
    t.start()
    return {"response": received_time}


@app.route("/request_welcome_message", methods=["POST", "GET"])
def generateWM():
    """Receives the text of the welcome message and send it to TTS process.
    Returns:
        (dict): Response message containing the emitting time.
    """
    url="http://127.0.0.1:8080/request_welcome_message"
    received_time = time.time()
    print("\nRequest is recieved\n")
    query = request.json
    query_json = json.dumps(query, ensure_ascii=False)
    response = requests.get(url, data=query_json, headers={"Content-Type": "application/json"})
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
    
    stt_model = load_models(params)
    app.run(host="0.0.0.0", port="5000",debug=False, use_reloader=False)
