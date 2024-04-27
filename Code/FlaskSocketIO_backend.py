from Models import *
import json
import os
import time

from flask import Flask, request
from flask_socketio import SocketIO, send, emit
import threading

# Get the path of this file.
main_path = os.path.dirname(os.path.realpath(__file__))

# Create a directory to save the recorded logs.
logs_path = main_path + "/logs/"
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
tts_model = None
stt_model = None
ttt_model = None

app = Flask(__name__)
sio = SocketIO(app, async_mode = "threading")

entry_length = 300

def load_models():
    """Loads all there models: Speech-to-Text, Text-to-Text, Text-to-Speech models.

    Returns:
        (WhisperLargeV2): Speech-to-Text model.
        (FastChatModel): Text-to-Text models.
        (XTTS_V2): Text-to-Speech models.
    """
    print("\nLoading Whisper ...\n")
    stt_model = WhisperLargeV2()
    print("\nLoading FastChat ...\n")
    ttt_model = FastChatModel()
    print("\nLoading XTTS_V2 ...\n")
    tts_model = XTTS_V2()
    return stt_model, ttt_model, tts_model

# Load the models
stt_model, ttt_model, tts_model = load_models()


@sio.on("connect")
def connect():
    """Triggered when a client connect.
    Emits back a signal to the connected client.
    """
    print('Client connected')
    sio.emit("client_Unlock", data={"Client": 1})

@sio.on("disconnect")
def disconnect():
    """Triggered when a client disconnect.
    Clears chat history as well.
    """
    ttt_model.clear_history()
    print("Client disconnected")


@sio.on('reset', namespace="/")
def reset(sid):
    """Triggered on reset emit. Clears chat history.

    Args:
        sid (int): Clients id, assigned automatically by the socket server.
    """
    ttt_model.clear_history()
    print("##### \nReset \n#####")
    

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
    first_response = True
    logs = {}
    start_time = time.time()
    current_time = start_time
    print("########### generating ###########")
    transcribtion = stt_model.run(voice_request)
    transcribtion_time = time.time() - current_time
    print(transcribtion)
    # print(voice_query)
    entry = ""
    response = []
    ttt_times = []
    tts_times = []
    current_time = time.time()
    
    for out in ttt_model.run(transcribtion.strip()):
        ttt_times.append(time.time()-current_time)
        if out == "END":
            print("\n##############\n")
            print(entry)
            print("\n##############\n")
            current_time = time.time()

            voice_answer = tts_model.run(entry)
            tts_times.append(time.time()-current_time)
            if first_response:
                logs["first_response"] = sum(ttt_times) + sum(tts_times)
                first_response = False
            b_answer = voice_answer.tobytes()
            response.append(entry)
            # jdata = json.dumps(data)
            data = {"voice_answer": b_answer}
            sio.emit("voice reply", data)
        else:
            if (len(entry) + len(out) + 1) >= entry_length:
                print("\n##############\n")
                print(entry)
                print("\n##############\n")
                current_time = time.time()

                voice_answer = tts_model.run(entry)
                tts_times.append(time.time()-current_time)
                
                if first_response:
                    logs["first_response"] = sum(ttt_times) + sum(tts_times)
                    first_response = False
                    
                b_answer = voice_answer.tobytes()
                data = {"voice_answer": b_answer}
                # jdata = json.dumps(data)
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
    # to_json = bytes_data.decode('utf-8').replace("'", '"')
    # json_data = json.loads(to_json)
    # voice_query = json_data["recorded"]
    voice_query = bytes_data
    # print(time.time()-json_data["start_time"])json.loads(logs_bytes.decode('utf-8'))
    t = threading.Thread(target=generateAnswer, daemon=True, args=[voice_query])
    t.start()
    return {"response": received_time}

if __name__ == '__main__':
    sio.run(app, debug=True, use_reloader=False)
