from Models import *
from flask import Flask, request, stream_with_context, jsonify
import json
import socket
import os
import pickle
import struct
import wave

from aiohttp import web
import socketio
import uvicorn
import engineio
import eventlet

print("\nLoading Whisper ...\n")
# stt_model = WhisperLargeV2()
stt_model = None
print("\nLoading FastChat ...\n")
# ttt_model = FastChatModel()
ttt_model = None
print("\nLoading Bark ...\n")
# tts_model = Bark(voice_preset = "v2/de_speaker_5")
tts_model = None

sio = socketio.AsyncServer(async_mode="asgi", ping_timeout=3000)

# app = web.Application()
# sio.attach(app)

app = socketio.ASGIApp(sio)

# sio = socketio.Server(async_mode='eventlet')
# app = socketio.WSGIApp(sio)

@sio.event
async def connect(sid, environ):
    print('connect ', sid)
    await sio.emit("client_Unlock", data={"Client": 1})


@sio.on('my_adsf_event')
# @sio.on('message', namespace='/')
async def any_event_any_namespace(sid, data):
    print(data)

# @sio.on('message', namespace='/')
# async def reply(sid, rdata):
#     voice_query = rdata["recorded"]
#     transcribtion = stt_model.run_flask(voice_query)
#     print(transcribtion)
#     entry = ""
#     response = []

#     for out in ttt_model.run(transcribtion):
#         if out == "END":
#             print("\n##############\n")
#             print(entry)
#             print("\n##############\n")
#             voice_answer = tts_model.run_ex(entry)
#             b_answer = voice_answer.tobytes()
#             data = {"voice_answer": b_answer}
#             response.append(entry)
#             # jdata = json.dumps(data)
#             await sio.emit("voice reply", data)
#         else:
#             if (len(entry) + len(out) + 1) >= 250:
#                 print("\n##############\n")
#                 print(entry)
#                 print("\n##############\n")
#                 voice_answer = tts_model.run_ex(entry)
#                 b_answer = voice_answer.tobytes()
#                 data = {"voice_answer": b_answer}
#                 # jdata = json.dumps(data)
#                 response.append(entry)
#                 entry = out + " "
#                 await sio.emit("voice reply", data)
#             else:
#                 entry += out + " "
#     await sio.emit("request")

if __name__ == '__main__':
    # web.run_app(app)
    uvicorn.run(app=app, host='127.0.0.1', port=8080)
    # eventlet.wsgi.server(eventlet.listen(('', 8080)), app)
