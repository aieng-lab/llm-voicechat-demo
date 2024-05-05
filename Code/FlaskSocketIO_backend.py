from Models import *
import json
import os
import time

from flask import Flask, request
from flask_socketio import SocketIO
import threading

main_path = os.path.dirname(os.path.realpath(__file__))
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
    # print("\nLoading Whisper ...\n")
    stt_model = WhisperLargeV2()
    # print("\nLoading FastChat ...\n")
    ttt_model = FastChatModel()
    # print("\nLoading XTTS_V2 ...\n")
    tts_model = XTTS_V2()
    return stt_model, ttt_model, tts_model


stt_model, ttt_model, tts_model = load_models()


@sio.on("connect")
def connect():
    print('Client connected')

@sio.on("disconnect")
def disconnect():
    ttt_model.clear_history()
    print("Client disconnected")


@sio.on('reset', namespace="/")
def reset(sid):
    ttt_model.clear_history()
    print(" \nChat history is reset. \n#####")
    

def generateAnswer(voice_request):
    first_response = True
    logs = {}
    start_time = time.time()
    current_time = start_time
    # print("########### generating ###########")
    transcribtion = stt_model.run(voice_request)
    transcribtion_time = time.time() - current_time
    # print(transcribtion)
    # print(voice_query)
    entry = ""
    response = []
    ttt_times = []
    tts_times = []
    current_time = time.time()
    for out in ttt_model.run(transcribtion.strip()):
        ttt_times.append(time.time()-current_time)
        if out == "END":
            # print("\n##############\n")
            # print(entry)
            # print("\n##############\n")
            current_time = time.time()

            voice_answer = tts_model.run(entry)
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
                # print("\n##############\n")
                # print(entry)
                # print("\n##############\n")
                current_time = time.time()
                if len(entry)<1:
                    entry = "Entschuldigung, ich konnte deine Anfrage nicht beantworten"
                    continue
                voice_answer = tts_model.run(entry)
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
    received_time = time.time()
    # print("\nRequest is recieved\n")
    bytes_data = request.get_data()
    voice_query = bytes_data
    t = threading.Thread(target=generateAnswer, daemon=True, args=[voice_query])
    t.start()
    return {"response": received_time}

if __name__ == '__main__':
    
    sio.run(app, debug=False, use_reloader=False, log_output=False)
