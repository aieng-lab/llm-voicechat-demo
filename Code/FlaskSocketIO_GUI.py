# # This is was based on part 16 of the PyQt5 learning series
# For details visit pyshine.com

import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
# Sounddevice is not used in the implementation.
# But it's needed for the system to work better with the sound card.
import sounddevice as sd
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream
import time
import os
import wave, pyaudio
import soundfile as sf
#Needed to display the GUI in dark mode.
#import BreezeStyleSheets.breeze_resources
import speech_recognition as sr
# transformers.pipeline helps initializing the microphone quicker.
from transformers import pipeline
import requests
import json
import socketio
import asyncio
from functools import cached_property
from main_ui import Ui_MainWindow
import PIL
import argparse



#Get this files path.
main_path = os.path.dirname(os.path.realpath(__file__))

#Create a logs folder in the same directory as this file.
logs_path = main_path + "/logs/"
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


class MplCanvas(FigureCanvas):
    """_summary_

    Args:
        width (float): Figure width in inches (default = 5.0).
        height (float): Figure height in inches (default = 4.0).
        dpi (float): Dots per inch (default = 100.0).
        facecolor (string): The figure patch facecolor (default = 'black').
    """
    def __init__(self, width=5, height=4, dpi=100, facecolor = 'black'):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor=facecolor)
        self.axes = fig.add_subplot(111)
        self.axes.axis("off")
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()


class Worker(QtCore.QRunnable):
    """QRunnable, an interface for representing a task or piece of code that needs to be executed.
    In this project, it's used to update the bot status.

    Args:
        function (Function): The task that needs to be executed (MainUI.updateStatus).
        *args: Other positional arguments passed to function.
        **kwargs: Other keyword arguments passed to function.
    """
    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Executes the assigned task to this worker.
        """
        self.function(*self.args, **self.kwargs)





class AudioOutputWorkerSignals(QtCore.QObject):
    """PyQt signals for AudioOutputWorker
    """
    finished = QtCore.pyqtSignal()
    waiting = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(str)

class AudioOutputWorker(QtCore.QRunnable):
    """QRunnable to handle the generated audio data.

    Args:
        function (Function): (MainUI.getAudio)
        audio_queue (queue.Queue): A queue of audio data.
        *args: Other positional arguments passed to function.
        **kwargs: Other keyword arguments passed to function.
    """
    def __init__(self, function, audio_queue, *args, **kwargs):
        super(AudioOutputWorker, self).__init__()
        self.function = function
        self.audio_queue = audio_queue
        self.args = args
        self.kwargs = kwargs
        self.signals = AudioOutputWorkerSignals()
        
        # Keeps the worker running as long as the conversation is not reset.
        self.running = True
        
        # We don't always have audio data to be handeled.
        # So we keep the worker running but idle.
        self.idle = True
        
        # The worker should wait until audio data is generated
        self.should_wait = True
        
        # The audio is being played
        self.speaking = True
    
    
    def run(self):
        """Run this worker and emits pyqtSignals based on the execution of the assigned function.
        """
        # print("AudioOutputWorker\n")
        # While this worker is running.
        while self.running:
            # If worker is not in idle state.
            if not self.idle:
                # If there is audio data in the queue.
                if not self.audio_queue.empty():
                    # Emit a status signal to change BOT Status
                    self.signals.status.emit("Ich spreche  ... ")	
                    self.function(*self.args, **self.kwargs)
                
                # If the audio queue is empty
                else:
                    # If we haven't sent a signal to generate new audio data.
                    # And there's no audio being played.
                    if not self.should_wait and not self.speaking:                        
                        
                        # Send a waiting signal to inform  the project, that we're
                        # waiting on new audio data.
                        self.signals.waiting.emit()
                        
                        #Wait for the audio data to be generated.
                        self.should_wait = True                
        self.signals.finished.emit()
    
    def stop_running(self):
        """Stop this worker.
        """
        self.running=False
    
    def dont_wait(self):
        """Stop waiting. This gives the worker the ability to ask for new audio data.
        """
        self.should_wait = False
    
    def wake(self):
        """Wake this worker from idle state to play and plot audio.
        """
        self.idle = False

class APISignals(QtCore.QObject):
    """PyQt signals for APIWorker.
    """
    start = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    result = QtCore.pyqtSignal(bytes)

# class APIWorker(QtCore.QRunnable):
class APIWorker(QtCore.QThread):
    """QRunnable to record audio from microphone and emit data to ClientWorker.
    """
    def __init__(self, params):
        super(APIWorker, self).__init__()
        self.r = sr.Recognizer()
        self.signals = APISignals()
        self.params=params
        self.mutex = QtCore.QMutex()
        self.activated = False
        
        
    def run(self):
        self.activated = True
        # print("STTWorker\n")
        if not self.params["type"]=="no_click":
            print("run")
            self.signals.start.emit("Ich höre zu  ... ")
            audio_bytes = self.pushToTalk()
            print("audio returtned")
        else:
            with sr.Microphone() as source:
                # print("Microphone is started ... \n")
                # Adjust the microphone to ignore unwanted noises.
                self.r.adjust_for_ambient_noise(source)
                
                # print("Microphone is listening ... \n")
                #Emit a status signal to change BOT Status
                self.signals.start.emit("Ich höre zu  ... ")
                #Start recording
                audio = self.r.listen(source)
                audio_bytes = audio.get_flac_data()
        #Emit a status signal to change BOT Status
        self.signals.start.emit("Ich überlege was ich antworte  ... ")
        # print(audio_bytes)
        #Emit the recorded data as bytes.
        self.signals.result.emit(audio_bytes)
        self.activated = False
        self.signals.finished.emit()
            
    def pushToTalk(self):
        print("pushToTalk")
        self.chunk = 2048  # Record in chunks of 1024 samples
        self.format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 2  # Stereo
        self.rate = 44100  # Record at 44100 samples per second
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False
        try:
            self.recording = True
            self.stream = self.p.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
            # self.update_label.emit("Recording...")
            self.stream.start_stream()
            while self.recording:
                # print("recording")
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            print(f"Error: {e}")
            # self.update_label.emit(f"Error: {e}")
        finally:
            if self.stream:
                print("stopped")
                self.stream.stop_stream()
                self.stream.close()
            audio_bytes = b''.join(self.frames)
            audio_data = sr.AudioData(audio_bytes, self.rate, 2)  # 2 is for 16-bit samples
        return audio_data.get_flac_data()
                
                
    def stop(self):
        print("try to stopped")
        with QtCore.QMutexLocker(self.mutex):
            self.recording = False
        self.quit()
        self.wait()  # Wait for the recording thread to finish
        
        
    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()


class ClientWorker(QtCore.QRunnable):
    """QRunnable to handle the client socketIO.

    Args:
        client (QtCore.QObject): The object containing client socketIO.
         
    """
    def __init__(self, client):
        super(ClientWorker, self).__init__()
        self.client = client
        self.loop = None
    
    def run(self):
        """Create an event loop to keep the asycronized connection alive.
        """
        if not self.loop is None:
            self.loop.stop()
        self.loop = asyncio.new_event_loop()
        self.loop.create_task(self.client.start())
        self.loop.run_forever()

class Client(QtCore.QObject):
    """QtCore.QObject that holds an asyncronized client socketIO.
    """
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal()
    error_ocurred = QtCore.pyqtSignal(object, name="errorOcurred")
    data_changed = QtCore.pyqtSignal(bytes, name="dataChanged")
    end_receive = QtCore.pyqtSignal()
    recorded_times = QtCore.pyqtSignal(bytes)
    chatReceived = QtCore.pyqtSignal(str)
    imageReceived = QtCore.pyqtSignal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.sio.on("connect", self._handle_connect, namespace=None)
        self.sio.on("connect_error", self._handle_connect_error, namespace=None)
        self.sio.on("disconnect", self._handle_disconnect, namespace=None)
        self.sio.on("voice reply", self.receiveAudio, namespace=None)
        self.sio.on("request", self.endReceiving, namespace=None)
        self.sio.on("chat", self.receiveChat, namespace=None)
        self.sio.on("image", self.receiveImage, namespace=None)

    @cached_property
    def sio(self):
        """Creates and caches an instance of socketio.AsyncClient
        """
        return socketio.AsyncClient(
            # reconnection=True,
            # reconnection_attempts=3,
            # reconnection_delay=5,
            # reconnection_delay_max=5,
            # logger=True,
        )

    async def start(self):
        """Connect to socketIO.AsyncServer
        """
        await self.sio.connect(url="http://127.0.0.1:5000", transports="polling")
        # await self.sio.connect(url="http://127.0.0.1:8080", transports="polling")
        await self.sio.wait()
        
    async def disconnect(self):
        """Disconnect from socketIO.AsyncServer
        """
        await self.sio.disconnect()
        
    def _handle_connect(self):
        """Emits a signal if the connection was successful.
        """
        self.connected.emit()

    def _handle_disconnect(self):
        """Emits a signal if the client disconnected.
        """
        self.disconnected.emit()

    def _handle_connect_error(self, data):
        """Emits unsuccessful connection details.

        Args:
            data (exception): The unsuccessful connection details sent from server.
        """
        self.error_ocurred.emit(data)


    def receiveAudio(self, data):
        """Receives audio data from server and emits it to be added to audio queue.

        Args:
            data (bytes): Generated audio sent by the server.
        """
        self.data_changed.emit(data["voice_answer"])
    
    def endReceiving(self, logs_bytes):
        """Emits a signal to inform the GUI that the server has finshed sending the generated audio.
        Also emits the logs of execution times sent from server.

        Args:
            logs_bytes (bytes): Execution times.
        """
        # print("endReceiving was reached")
        self.end_receive.emit()
        self.recorded_times.emit(logs_bytes)

    async def receiveFromGUI(self, data):
        """Receive the recorded audio from GUI and send it as a request to server.

        Args:
            data (bytes): audio data.
        """
        request = {"recorded": data,
                "start_time":time.time(),
                "request_length": len(data)}
        
        await self.sio.emit('request', data=request)
        
    def receiveChat(self, data):
        self.chatReceived.emit(data)
        
    def receiveImage(self, data):
        self.imageReceived.emit(data["image_bytes"])
        

    async def reset(self):
        """Emits to the server to reset the chat history.
        """
        await self.sio.emit('reset')


class GUISignals(QtCore.QObject):
    """PyQt signals for MainUI
    """
    request = QtCore.pyqtSignal()

        
        
class MainUI(QtWidgets.QMainWindow):
    """The main window of our GUI.

    Args:
        params (dict):
            'CHUNK': (int) length of data entry for the plot queue.
            'samplerate': (int) The samplerate for the audio.
    """
    def __init__(self, params):
        # print("Initializing ... \n")
        QtWidgets.QMainWindow.__init__(self)
        
        # Load a pre-designed GUI.
        # self.ui = uic.loadUi(main_path+'/main.ui',self)
        self.ui = Ui_MainWindow(params = params)
        self.ui.setupUi(self, params["window_size"])
        self.showMaximized()
        # self.showFullScreen()
        
        
        # QThreadPools are used to run QRunnable objects.
        self.threadpool = QtCore.QThreadPool()
        self.client_pool = QtCore.QThreadPool()		
        self.API_pool = QtCore.QThreadPool()
        
        self.params = params
        self.CHUNK = self.params["CHUNK"]
        
        self.plot_queue = queue.Queue(maxsize=self.CHUNK)
        self.audio_queue = queue.Queue()
        
        self.canvas = MplCanvas(width=5, height=4, dpi=100)
        self.ui.gridLayout_2.addWidget(self.canvas, 2, 1, 1, 1)
        self.reference_plot = None
        self.mic = False
        self.window_length = 1000
        self.downsample = 1
        self.channels = [1]
        self.samplerate = self.params["samplerate"]
        self.length  = int(self.window_length*self.samplerate/(1000*self.downsample))
        
        self.plotdata =  np.zeros((self.length,len(self.channels)))

        self.timer = QtCore.QTimer()
        self.timer.setInterval(30) #msec
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start()
        
        self.data=[0]
        self.record_allowed = True
        self.plotting = True
        self.stopped = False
        self.speaking_allowed = True
        self.stop_audio = False
        
        self.api_worker = APIWorker(self.params)
        self.api_worker.signals.start.connect(self.displayStatus)
        self.api_worker.signals.start.connect(self.changeColor)
        self.api_worker.signals.result.connect(self.request)
        self.api_worker.signals.finished.connect(self.changeColor)
    
        self.ui.startButton.setEnabled(False)
        self.ui.resetButton.setEnabled(False)
        
        self.status_worker = None
        self.r = sr.Recognizer()
        
        self.speaker_worker = AudioOutputWorker(function=self.getAudio, audio_queue=self.audio_queue)
        if self.params["type"] == "push_to_talk":
            self.speaker_worker.signals.waiting.connect(self.startPushToTalk)
            
        elif self.params["type"] == "two_clicks":
            self.speaker_worker.signals.waiting.connect(self.startPushToTalk)
        else:
            self.speaker_worker.signals.waiting.connect(self.startAPIWorker)
        
        self.speaker_worker.signals.status.connect(self.updateStatus)
        self.threadpool.start(self.speaker_worker)
        self.init = True
        self.client = Client()
        self.client_worker = None
        self.signals = GUISignals()
        self.logs={}
        self.ui.startButton.clicked.connect(self.start)
        self.ui.resetButton.clicked.connect(self.stop_answer)
        self.ui.chatButton.clicked.connect(self.showChatWindow)
        # self.ui.pushToTalk.clicked.connect(self.startAPIWorker)
        if self.params["type"] == "push_to_talk":
            self.ui.pushToTalk.setCheckable(True)
            self.ui.pushToTalk.pressed.connect(self.startAPIWorker)
            self.ui.pushToTalk.released.connect(self.stopPushToTalk)
        
        elif self.params["type"] == "two_clicks":
            self.ui.pushToTalk.setEnabled(False)
            self.ui.pushToTalk.clicked.connect(self.toggle)


        self.ui.startButton.setEnabled(True)
        self.ui.resetButton.setEnabled(True)
        self.ui.chatButton.setEnabled(True)
        self.ui.pushToTalk.setStyleSheet(f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: gray;')
        # self.ui.pushToTalk.setEnabled(False)
        # self.api_worker=None
        
    
    def startPushToTalk(self, update_status=True):
        #if update_status:
        #    self.updateStatus("Ich bin verfügbar ...")
        self.ui.pushToTalk.setEnabled(True)
        self.ui.pushToTalk.setStyleSheet(f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: white;')
        
    def stopPushToTalk(self):
        if self.api_worker is not None:
            self.api_worker.stop()
            self.ui.pushToTalk.setEnabled(False)
            self.ui.pushToTalk.setStyleSheet(f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: gray;')
    
    def toggle(self):
        if not self.api_worker.activated:
            self.ui.pushToTalk.setText("Klick nochmal, um zu stoppen")
            self.ui.pushToTalk.setStyleSheet(f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: white;')
            self.startAPIWorker()
        else:
            self.stopPushToTalk()
            self.ui.pushToTalk.setText("Klick zum Anfragen")
            self.ui.pushToTalk.setStyleSheet(f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: gray;')

            if self.stop_audio:
                self.stop_audio = False
                self.restart()

    def showChatWindow(self):
        self.ui.chatWindow.show()
        
        
    def stop_answer(self):
        """Stops the current answer, i.e., stops the audio speakers and the plotting.
        """
        self.ui.label_6.setText("Ich wurde gestoppt  ... ")
        self.stop_speaker()
        self.stop_audio = True

        self.speaker_worker = None
        
        with self.plot_queue.mutex:
            self.plot_queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        self.plot_queue = queue.Queue(maxsize=self.CHUNK)
        self.audio_queue = queue.Queue()
        self.plotdata =  np.zeros((self.length,len(self.channels)))

        self.startPushToTalk(update_status=False)

        
    def updateChat(self, data):
        self.ui.chatWindow.text.append(data)
        
    def updateImage(self, image):
        image_np = np.frombuffer(image, dtype=np.uint8).reshape((512,512,3))
        self.ui.imageWindow.setData(image_np)
        self.ui.imageWindow.plot()
        self.ui.imageWindow.show()
        # print(image_np.shape)
    
    def start(self):
        """Starts when Start button is clicked.
        Play a pre-generated welcome message.
        If the Reset is already clicked, re-initialize and start both client_worker and speaker_worker.
        """
        # print("Initialiazing is finished.\n")
        #if not self.audio_queue.empty():
        #    self.stop_answer()
        #    time.sleep(0.1)

        self.check_for_client_worker()
                        


        if self.params["welcome_message"]:
            self.requestWMGenerating(self.params["welcome_message"])
            wf = wave.open("welcome_message.wav")
            data = wf.readframes(-1)
            self.audio_queue.put_nowait({"data": data, "format": "2**16"})
        else:
            self.displayStatus("Ich bin verfügbar  ...")

        self.params["window_size"] = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.stop_audio = False

        self.check_for_speaker_worker()
        
        self.ui.chatWindow.text.setText("ALVI  >>>  " + self.params["welcome_message"])
            
        self.startSpeakerWorker()

    def restart(self):
        self.check_for_client_worker()
        self.check_for_speaker_worker()
        self.startSpeakerWorker()

    def check_for_client_worker(self):
        if self.client_worker is None:
            self.client_worker = ClientWorker(self.client)
            self.client.data_changed.connect(self.updateAudioQueue)
            self.client.chatReceived.connect(self.updateChat)
            self.client.imageReceived.connect(self.updateImage)
            self.client.end_receive.connect(self.startSpeakerWorker)
            # self.client.recorded_times.connect(self.saveLogs)
            self.client_pool.start(self.client_worker)

    def check_for_speaker_worker(self):
        if self.speaker_worker is None:
            self.stopped = False
            self.speaking_allowed = True
            self.plotting = True
            self.speaker_worker = AudioOutputWorker(function=self.getAudio, audio_queue=self.audio_queue)
            if self.params["type"] == "no_click":
                self.speaker_worker.signals.waiting.connect(self.startAPIWorker)
            else:
                self.speaker_worker.signals.waiting.connect(self.startPushToTalk)
            self.speaker_worker.signals.status.connect(self.updateStatus)
            self.threadpool.start(self.speaker_worker)
            self.init = True

    def requestWMGenerating(self, welcome_message):
        url = "http://localhost:5000/request_welcome_message"
        # query = {"content":welcome_message}
        response = requests.get(url, json=json.dumps(welcome_message, ensure_ascii=False))
        self.params["old_welcome_message"] = welcome_message
        self.params["window_size"] = None
        with open(main_path+'/params.json', "w") as params_file:
            json.dump(self.params, params_file)
            params_file.close()
    
    def displayStatus(self, text):
        """Display a text on GUI.

        Args:
            text (string): Message to be displayed.
        """
        
        self.ui.label_6.setText(text)
        self.status_size = int(0.03 * self.params["window_size"].width())
        if text == "Ich höre zu  ... ":
            self.ui.label_6.setStyleSheet(f''' font-size: {self.status_size}px; color: #00FF00;''')
        else:
            self.ui.label_6.setStyleSheet(f''' font-size: {self.status_size}px; color: Red;''')

    
    def updateStatus(self, text):
        """Change the displayed message.

        Args:
            text (string): The new message.
        """
        self.status_worker = Worker(self.displayStatus, text)
        self.threadpool.start(self.status_worker)
    
    def startAPIWorker(self):
        """Starts the api_worker.
        """
        # self.ui.pushToTalk.setStyleSheet(f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: gray;')
        # self.ui.pushToTalk.setEnabled(False)
        self.init=False
        # self.api_worker = APIWorker(self.params)
        # self.api_worker.signals.start.connect(self.displayStatus)
        # self.api_worker.signals.start.connect(self.changeColor)
        # self.api_worker.signals.result.connect(self.request)
        # self.api_worker.signals.finished.connect(self.changeColor)

        # self.threadpool.start(self.api_worker)
        self.api_worker.start()

    def changeColor(self):
        self.mic = not self.mic

    
    def request(self, data):
        """Send a GET request to FlaskAPI.

        Args:
            data (bytes): the recorded audio.
        """
        # print(data)
        url = "http://localhost:5000/request"
        starting_time = time.time()
        # print("Request is sent.\n")
        response = requests.get(url, data=data)
    
        a= response.content
        response_time = json.loads(a.decode('utf-8'))
        self.logs["sending_time"] = response_time["response"] - starting_time
    
    def saveLogs(self, logs_bytes):
        """Save time logs.

        Args:
            logs_bytes (bytes): Time logs sent from server in bytes.
        """
        self.logs = self.logs | json.loads(logs_bytes.decode('utf-8'))
        self.logs["first_response"] = self.logs["first_response"] + self.logs["sending_time"] * 2
        self.logs["total_time"] = self.logs["total_time"] + self.logs["sending_time"] * 2
        
        
        for dirname, _, filenames in os.walk(logs_path):
            file_name = "time_logs_{}.json".format(len(filenames))
            file_path = os.path.join(dirname, file_name)
            with open(file_path, "w") as file:
                json.dump(self.logs, file)
                # print(file_path)
                    
    def getAudio(self):
        """Process the data in audio queue to be played and plotted.
        """
        if not self.audio_queue.empty():
            # print("Getting Audio\n")
            CHUNK = self.CHUNK
            rate = self.samplerate
            audio_data = self.audio_queue.get_nowait()
            data = audio_data['data']
            format = audio_data['format']
            
            # Stream the audio using pyaudio
            p = pyaudio.PyAudio()
            
            # The welcome message is already recorded in the required format.
            # So it doesn't need extra processing as for audio from server.
            width = 2
            channels=1
            rate=24000
            OCHUNK = 2 * CHUNK
            #initialize stream
            stream = p.open(format=p.get_format_from_width(width),
                            channels=channels,
                            rate=rate,
                            output=True,
                            frames_per_buffer=CHUNK)
            if format == '2**16':
                max_int16 = 2**16        
                # print("Start audio\n")
                plotting = True
                while plotting:
                    if self.stop_audio:
                        if stream.is_active():
                            stream.stop_stream()
                        stream.close()
                        break

                    if len(data) > OCHUNK:
                        out = data[:OCHUNK]
                        data = data[OCHUNK:]
                    else:
                        out = data
                        plotting = False
                    try:
                        #Normalize audio data for a smoother plotting .
                        audio_as_np_int16 = np.frombuffer(out, dtype=np.int16)
                        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
                        audio_normalised = audio_as_np_float32 / max_int16
                        # print(audio_normalised.shape)
                        self.plot_queue.put_nowait(audio_normalised)
                        # Now that the data is ready for plotting,
                        # we start streaming so both happen synchronously.
                        stream.write(out)
                    except:
                        print("Broken\n")
                        break
                
            else:
                # If audio comes from the server, it needs to be saved to
                # a temporal file, to correct its format.
                arr = np.frombuffer(data, dtype=np.float32)
                sf.write("tmp.wav", arr, samplerate=24000)
                wf = wave.open('tmp.wav', 'rb')
                max_int16 = 2**15    
                out = wf.readframes(CHUNK)
                while out:
                    try:
                        if self.stop_audio:
                            if stream.is_active():
                                stream.stop_stream()
                            stream.close()
                            break

                        audio_as_np_int16 = np.frombuffer(out, dtype=np.int16)
                        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
                        audio_normalised = audio_as_np_float32 / max_int16
                        self.plot_queue.put_nowait(audio_normalised)
                        stream.write(out)
                        out = wf.readframes(CHUNK)
                    except:
                        print("Broken\n")
                        break

            self.updateStatus("Ich bin verfügbar ...")
                    
    def updateAudioQueue(self, data):
        """Add a new entry to audio queue.

        Args:
            data (bytes): audio data
        """
        self.audio_queue.put_nowait({'data': data, 'format': '2**15'})
    
    def generatePlotData(self):
        """Gets plotting data from plot_queue.
        if the queue is empty, it returns an array of zeros to display silence.

        Returns:
            numpy.ndarray: plotting data.
        """
        if not self.plot_queue.empty():
            audio_normalised = self.plot_queue.get_nowait()
            return audio_normalised
        else:
            if self.speaker_worker is not None:
                self.speaker_worker.speaking = False
            return np.zeros((self.CHUNK,len(self.channels)))
    
    def startSpeakerWorker(self):
        """Change both idle and waiting states of AudiooutputWorker.
        """
        try:
            if self.plotting:
                if self.speaker_worker is None:
                    self.restart()
                self.speaker_worker.dont_wait()
                self.speaker_worker.wake()
        except Exception as e:
            print(f"The model couldn't generate a speech from the text response\n{e}")
    
    def updatePlotQueue(self, plot_data):
        """Add new entry to plot_queue.

        Args:
            plot_data (numpy.ndarray): plotting data.
        """
        self.plot_queue.put_nowait(plot_data)
    
    def updatePlot(self):
        """Plot data from generatePlotData() as long as the project is running.
        """
        try:
            if  self.plotting is True:
                try: 
                    self.data = self.generatePlotData()
                except Exception as e:
                    pass
                shift = len(self.data)
                self.plotdata = np.roll(self.plotdata, -shift,axis = 0)
                self.plotdata = self.data
                self.ydata = self.plotdata[:]
                self.canvas.axes.set_facecolor((0,0,0))
                

            # To start plotting where the previous plot ended.
            if self.reference_plot is None:
                # plot_refs = self.canvas.axes.plot(self.ydata, color=(0,1,0.29), linewidth=self.params["linewidth"])
                plot_refs = self.canvas.axes.plot(self.ydata, color=self.params["color"], linewidth=self.params["linewidth"])
                # plot_refs = self.canvas.axes.plot(self.ydata, color=color, linewidth=self.params["linewidth"])
                self.reference_plot = plot_refs[0]	
            else:
                self.reference_plot.set_ydata(self.ydata)
                # self.reference_plot.set_color(color)
            
            self.canvas.axes.yaxis.grid(True,linestyle='--')
            start, end = self.canvas.axes.get_ylim()
            self.canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.1))
            self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.canvas.axes.set_ylim( ymin=-1, ymax=1)
            self.canvas.draw()
        except Exception as e:
            pass

    def closeEvent(self, event):
        self.stop_answer()
        self.record_allowed = False
        self.plotting = False
        self.stopped = False
        self.speaking_allowed =False
        url = url = "http://localhost:5000/close"
        response = requests.get(url, data={})
        
        self.stop_speaker()
        # Ask for confirmation before closing
        # confirmation = QtWidgets.QMessageBox.question(self, "Confirmation", "Are you sure you want to close the application?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)    

    def stop_speaker(self):
        if self.speaker_worker is not None:
            self.speaker_worker.stop_running()

def main():
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
        
    app = QtWidgets.QApplication(sys.argv)
    # screen = app.primaryScreen()
    # params["window_size"] = screen.size()
    
    params["window_size"] = QtWidgets.QDesktopWidget().screenGeometry(-1)
    mainWindow = MainUI(params=params)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

