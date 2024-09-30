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

main_path = os.path.dirname(os.path.realpath(__file__))
logs_path = main_path + "/logs/"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
tts_model = None

app = Flask(__name__)

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
        (StableDiffusion): Text-to-Speech models.
    """
    tti_model = get_instance(params["tti_model"])
    return tti_model


# tti_model = load_models(params)


@app.route("/disconnect", methods=["POST", "GET"])
def disconnect():
    """Triggered when a client disconnect.
    Clears chat history as well.
    """
    print("Client disconnected")


def generateImage(query):
    """Takes audio bytes as input and processes it follows: 
        1- Transcribes it using the Speech-To-Text model.
        2- Query the Text-to-Text model on the transcribed text from 1.
        3- Generates speech on each generated chunck using the Text-to-Speech model.
        4- Emits the generated speech to the client.
        5- When generating is finished, a 'request' emit to notify the client to record a new voice request.
    Args:
        voice_request (bytes): Recorded voice request from client.
    """
    global params, tti_model
    url="http://127.0.0.1:8080/request_image"
    image_np = np.array(tti_model.run(query+ " " + params["image_generation_system_prompt"]))
    image_bytes = image_np.tobytes()
    resp = requests.get(url, data=image_bytes)
    
@app.route("/request", methods=["POST", "GET"])
def receive():
    """Receives the generation query from tts api, creates a new thread to process it.
    Returns:
        (dict): Resonse message containing the emitting time.
    """
    received_time = time.time()
    print("\nRequest is recieved\n")
    query = request.json
    t = threading.Thread(target=generateImage, daemon=True, args=[query])
    t.start()
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
    
    tti_model = load_models(params) 
    app.run(host="0.0.0.0", port="9090",debug=False, use_reloader=False)
