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
from PyQt5.QtCore import QTimer
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

import signal



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
                    self.speaking = True	
                    self.idle = False
                    self.should_wait = False
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
            QtCore.QThread.msleep(10)           
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
        """
        Records audio in real-time using PyAudio and returns raw PCM waveform data.
        """
        print("pushToTalk")
        self.chunk = 2048  # Record in chunks of 2048 samples
        self.format = pyaudio.paInt16  # 16 bits per sample
        self.channels = 2  # Stereo
        self.rate = 44100  # Record at 44100 samples per second
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.recording = False

        try:
            self.recording = True
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.stream.start_stream()

            while True:
                with QtCore.QMutexLocker(self.mutex):
                    if not self.recording:
                        break

                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)

        except Exception as e:
            print(f"Error: {e}")

        finally:
            if self.stream:
                print("stopped")
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            audio_bytes = b''.join(self.frames)
            return audio_bytes

    def stop(self):
        print("try to stopped")

        with QtCore.QMutexLocker(self.mutex):
            self.recording = False

        # Do not close self.stream here.
        # pushToTalk() owns the stream and closes it in its own thread.
        self.wait(2000)

    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()


class ClientWorker(QtCore.QRunnable):
    """QRunnable to handle the client socketIO."""

    def __init__(self, client):
        super(ClientWorker, self).__init__()
        self.client = client
        self.loop = None
        self.running = True

    def run(self):
        if self.loop is not None:
            self.loop.stop()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.loop.create_task(self.client.start())

        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()

            try:
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            except Exception:
                pass

            self.loop.close()

    def stop(self):
        self.running = False

        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)

class Client(QtCore.QObject):
    """QtCore.QObject that holds an asyncronized client socketIO.
    """
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal()
    error_ocurred = QtCore.pyqtSignal(object, name="errorOcurred")
    data_changed = QtCore.pyqtSignal(bytes, name="dataChanged")
    end_receive = QtCore.pyqtSignal()
    recorded_times = QtCore.pyqtSignal(bytes)
    chatReceived = QtCore.pyqtSignal(dict)
    imageReceived = QtCore.pyqtSignal(bytes)
    start_speaker = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.sio.on("connect", self._handle_connect, namespace=None)
        self.sio.on("connect_error", self._handle_connect_error, namespace=None)
        self.sio.on("disconnect", self._handle_disconnect, namespace=None)
        self.sio.on("voice reply", self.receiveAudio, namespace=None)
        self.sio.on("request", self.endReceiving, namespace=None)
        self.sio.on("chat", self.receiveChat, namespace=None)
        self.sio.on("image", self.receiveImage, namespace=None)

        self.audio_buffer = {}
        self.next_audio_id = 0
        self.playing = False
        self.audio_queue = queue.Queue()

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
    

    def receiveAudio(self, data):
        audio = data["voice_answer"]

        # optional id support (safe fallback)
        audio_id = data.get("id", self.next_audio_id)

        print("RECV AUDIO ID:", audio_id)

        self.data_changed.emit(audio)


    def endReceiving(self, logs_bytes):
        """Emits a signal to inform the GUI that the server has finshed sending the generated audio.
        Also emits the logs of execution times sent from server.

        Args:
            logs_bytes (bytes): Execution times.
        """
        # print("endReceiving was reached")
        self.end_receive.emit()
        self.recorded_times.emit(logs_bytes)
        self.audio_stream_finished = True

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
        self.params = params
        self.ui = Ui_MainWindow(params=params)
        self.ui.setupUi(self, params["window_size"])

        combo_style = """
        QComboBox {
            background-color: #555555;
            color: white;
            border: 1px solid #777777;
            border-radius: 3px;
            padding: 2px 6px;
        }

        QComboBox::drop-down {
            background-color: #555555;
            border-left: 1px solid #777777;
            width: 28px;
        }

        QComboBox::down-arrow {
            image: none;
            border: none;
        }

        QComboBox QAbstractItemView {
            background-color: #222222;
            color: white;
            selection-background-color: #555555;
            selection-color: white;
        }
        """

        for combo in self.findChildren(QtWidgets.QComboBox):
            combo.setStyleSheet(combo_style)


        def keyPressEvent(event):
            key = event.key()
            if key == QtCore.Qt.Key.Key_PageDown:
                self.ui.pushToTalk.click()
            elif key == QtCore.Qt.Key.Key_PageUp:
                self.ui.resetButton.click()
            elif key == 46:
                self.ui.startButton.click()
            #elif key in {QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Return}:
            #    self.ui.chatButton.click()
            elif key in {
                QtCore.Qt.Key.Key_Escape,  # 16777216
                QtCore.Qt.Key.Key_F5,      # 16777268
            }:
                self.toggle_api_model()
            else:
                print(f'Unsupported Key was pressed {key}')
        self.keyPressEvent = keyPressEvent

        #self.showMaximized()
        # self.showFullScreen()
        self.show()
        
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
        self.client = Client()#
        self.client.start_speaker.connect(self.startSpeakerWorker)
        self.client_worker = None
        self.signals = GUISignals()
        self.logs={}
        self.ui.startButton.clicked.connect(self.start)
        self.ui.resetButton.clicked.connect(self.stop_answer)
        self.ui.chatButton.clicked.connect(self.showChatWindow)


        self.api_model_buttons = {}
        self.active_api_model = None
        self.last_model_switch_time = 0
        QtCore.QTimer.singleShot(100, self.load_api_model_buttons)

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

    def toggle_api_model(self):
        api_models = self.params.get("api_models")

        if not api_models:
            return

        # Debounce pointer remotes that emit multiple key events
        now = time.time()
        if now - self.last_model_switch_time < 0.5:
            return

        self.last_model_switch_time = now

        model_ids = list(api_models.keys())

        if not model_ids:
            return

        if self.active_api_model not in model_ids:
            next_model_id = model_ids[0]
        else:
            current_index = model_ids.index(self.active_api_model)
            next_model_id = model_ids[(current_index + 1) % len(model_ids)]

        self.switch_api_model(next_model_id)

    def load_api_model_buttons(self):
        api_models = self.params.get("api_models")

        if not api_models:
            print("No api_models found in params")
            return

        if self.api_model_buttons:
            return

        self.active_api_model = self.params.get("active_api_model")
        if self.active_api_model is None:
            self.active_api_model = next(iter(api_models.keys()))

        parent_layout = self.ui.gridLayout_1
        parent_layout.setVerticalSpacing(parent_layout.horizontalSpacing())

        spacing = parent_layout.horizontalSpacing()
        if spacing < 0:
            spacing = 6

        # Use actual button widths after layout is visible
        control_width = (
            self.ui.startButton.width()
            + self.ui.resetButton.width()
            + spacing
        )

        if control_width <= spacing:
            control_width = (
                self.ui.startButton.sizeHint().width()
                + self.ui.resetButton.sizeHint().width()
                + spacing
            )

        model_container = QtWidgets.QWidget()
        model_container.setFixedWidth(control_width)
        self.api_model_container = model_container

        model_grid = QtWidgets.QGridLayout(model_container)
        model_grid.setContentsMargins(0, 0, 0, 0)
        model_grid.setHorizontalSpacing(spacing)
        model_grid.setVerticalSpacing(0)

        self.api_model_buttons = {}

        for col, (model_id, pretty_name) in enumerate(api_models.items()):
            button = QtWidgets.QPushButton(pretty_name)
            button.setCheckable(True)
            button.setMinimumHeight(28)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Fixed
            )

            button.clicked.connect(
                lambda checked, mid=model_id: self.switch_api_model(mid)
            )

            model_grid.addWidget(button, 0, col)
            model_grid.setColumnStretch(col, 1)
            self.api_model_buttons[model_id] = button

        self.update_api_model_button_styles()


        wrapper = QtWidgets.QWidget()
        wrapper_layout = QtWidgets.QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addStretch(1)
        wrapper_layout.addWidget(model_container)
        wrapper_layout.addStretch(1)

        # Fixed row below the 2x2 control buttons
        self.api_model_wrapper = wrapper
        self.api_model_wrapper.setParent(self)
        self.api_model_wrapper.show()

        self.position_api_model_buttons()

        print("Added API model buttons:", list(api_models.keys()))
        print("control_width:", control_width)
    def position_api_model_buttons(self):
        if not hasattr(self, "api_model_wrapper"):
            return

        h_spacing = self.ui.gridLayout_1.horizontalSpacing()
        if h_spacing < 0:
            h_spacing = 6

        v_spacing = self.ui.gridLayout_1.verticalSpacing()
        if v_spacing < 0:
            v_spacing = h_spacing

        bottom_left = self.ui.pushToTalk.mapTo(
            self,
            QtCore.QPoint(0, self.ui.pushToTalk.height())
        )

        width = (
            self.ui.startButton.width()
            + h_spacing
            + self.ui.resetButton.width()
        )

        height = self.ui.startButton.height()

        if hasattr(self, "api_model_container"):
            self.api_model_container.setFixedWidth(width)

        self.api_model_wrapper.setGeometry(
            bottom_left.x(),
            bottom_left.y() + v_spacing,
            width,
            height
        )

        self.api_model_wrapper.raise_()

    def schedule_api_model_reposition(self):
        QtCore.QTimer.singleShot(50, self.position_api_model_buttons)
        QtCore.QTimer.singleShot(150, self.position_api_model_buttons)

    def update_api_model_button_styles(self):
        for model_id, button in self.api_model_buttons.items():
            selected = model_id == self.active_api_model
            button.setChecked(selected)

            if selected:
                button.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: #777777;
                        color: white;
                        border: 2px solid white;
                        padding: 6px;
                    }}
                    """
                )
            else:
                button.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: #555555;
                        color: white;
                        border: 1px solid #777777;
                        padding: 6px;
                    }}

                    QPushButton:hover {{
                        background-color: #666666;
                    }}
                    """
                )

    def switch_api_model(self, model_id):
        if model_id == self.active_api_model:
            return

        try:
            response = requests.post(
                "http://localhost:5000/switch_api_model",
                json={"model_id": model_id},
                timeout=10
            )

            response.raise_for_status()

            data = response.json()

            self.active_api_model = data["active"]
            self.params["active_api_model"] = self.active_api_model

            self.update_api_model_button_styles()

            self.ui.chatWindow.reset()
            self.reset_backend_chat()
            self.stop_answer()
            self.set_ui_state("ready")

            print(f"Switched API model to {data['pretty_name']}")

        except Exception as e:
            print("Could not switch API model:", e)

    
    def startPushToTalk(self, update_status=True):
        # Do not return to ready while the backend is still generating an answer.
        if self.awaiting_response:
            self.set_ui_state("thinking")
            return

        if update_status:
            self.set_ui_state("ready")
        else:
            self.ui.pushToTalk.setEnabled(True)
            self.ui.pushToTalk.setText("Klick zum Anfragen")
            self.ui.pushToTalk.setStyleSheet(
                f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: white;'
            )

    def stopPushToTalk(self):
        if self.api_worker is not None:
            self.api_worker.stop()
            self.ui.pushToTalk.setEnabled(False)
            self.ui.pushToTalk.setStyleSheet(f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: gray;')
    

    def stopPushToTalk(self):
        if self.api_worker is not None and self.api_worker.activated:
            self.api_worker.stop()

        self.set_ui_state("thinking")


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

    def toggle(self):
        if not self.api_worker.activated:
            self.set_ui_state("listening")
            self.startAPIWorker()
        else:
            self.stopPushToTalk()

            if self.stop_audio:
                self.stop_audio = False
                self.restart()

    def set_ui_state(self, state):
        """
        Central state handler for status display and Push-to-Talk button.
        States:
            ready
            listening
            thinking
            speaking
            stopped
        """

        if state == "ready":
            self.displayStatus("Ich bin verfügbar  ...")
            self.ui.pushToTalk.setEnabled(True)
            self.ui.pushToTalk.setText("Klick zum Anfragen")
            self.ui.pushToTalk.setStyleSheet(
                f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: white;'
            )

        elif state == "listening":
            self.displayStatus("Ich höre zu  ... ")
            self.ui.pushToTalk.setEnabled(True)

            if self.params["type"] == "two_clicks":
                self.ui.pushToTalk.setText("Klick nochmal, um zu stoppen")

            self.ui.pushToTalk.setStyleSheet(
                f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: white;'
            )
        elif state == "thinking":
            self.displayStatus("Ich überlege was ich antworte  ... ")
            self.ui.pushToTalk.setEnabled(False)
            self.ui.pushToTalk.setStyleSheet(
                f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: gray;'
            )

        elif state == "speaking":
            self.displayStatus("Ich spreche  ... ")
            self.ui.pushToTalk.setEnabled(False)
            self.ui.pushToTalk.setStyleSheet(
                f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: gray;'
            )

        elif state == "stopped":
            self.displayStatus("Ich wurde gestoppt  ... ")
            self.ui.pushToTalk.setEnabled(True)
            self.ui.pushToTalk.setText("Klick zum Anfragen")
            self.ui.pushToTalk.setStyleSheet(
                f'background-color: #555555; font-size: {self.ui.button_text_size}px; color: white;'
            )

    def showChatWindow(self):
        if self.params.get('chat_mode'):
            self.showNormal()

            # Get the list of available screens
            screens = QtWidgets.QApplication.screens()

            # Use the secondary screen if available; otherwise, use the primary screen
            if len(screens) > 1:
                screen = screens[1]  # Secondary screen
            else:
                screen = QtWidgets.QApplication.primaryScreen()  # Fallback to the primary screen
            screen = QtWidgets.QApplication.primaryScreen()

            # Get the available geometry of the chosen screen
            screen_geometry = screen.availableGeometry()

            chat_width = self.params.get('chat_width_px', 600)
            screen_height = screen_geometry.height()
            chat_height = self.params.get('chat_height_px', screen_height)

            # Resize and align the chat window
            self.ui.chatWindow.resize(chat_width, chat_height)
            self.ui.chatWindow.move(screen_geometry.x(),
                                    screen_geometry.y() + screen_geometry.height() // 2 - chat_height // 2)

            # Calculate the main window dimensions
            main_window_width = screen_geometry.width() - chat_width
            main_window_height = screen_geometry.height()

            # Resize and align the main window to the right of the chat window
            self.resize(main_window_width, main_window_height)
            self.move(screen_geometry.x() + chat_width, screen_geometry.y())

            # Force an update to apply changes
            self.update()
            self.position_api_model_buttons()

        self.ui.chatWindow.show()
        self.schedule_api_model_reposition()




        
    def stop_answer(self):
        """Stops the current answer, i.e., stops the audio speakers and the plotting.
        """
        self.awaiting_response = False
        self.set_ui_state("stopped")
        self.stop_audio = True

        if self.speaker_worker is not None:
            self.speaker_worker.speaking = False
            self.speaker_worker.idle = True
            self.speaker_worker.should_wait = True
        
        with self.plot_queue.mutex:
            self.plot_queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        #self.plot_queue = queue.Queue(maxsize=self.CHUNK)
        #self.audio_queue = queue.Queue()
        self.plotdata =  np.zeros((self.length,len(self.channels)))

        self.startPushToTalk(update_status=False)

        
    def updateChat(self, data):
        self.ui.chatWindow.add_message(**data)
        
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
        # print("Initializing is finished.\n")
        #if not self.audio_queue.empty():
        #    self.stop_answer()
        #    time.sleep(0.1)

        self.ui.chatWindow.reset()
        self.reset_backend_chat()

        self.check_for_client_worker()

        if self.params["welcome_message"]:
            self.requestWMGenerating(self.params["welcome_message"])
            wf = wave.open("welcome_message.wav")
            data = wf.readframes(-1)
            self.audio_queue.put_nowait({'data': data, 'format': '2**15'})

            # FORCE WAKE SPEAKER
            if self.speaker_worker is not None:
                self.speaker_worker.dont_wait()
                self.speaker_worker.wake()
        else:
            self.displayStatus("Ich bin verfügbar  ...")

        self.params["window_size"] = QtWidgets.QDesktopWidget().screenGeometry(-1)
        self.stop_audio = False
        self.set_ui_state("ready")

        self.check_for_speaker_worker()

        if self.params.get("welcome_message"):
            self.ui.chatWindow.add_ai_message(self.params["welcome_message"])

        self.startSpeakerWorker()

    def restart(self):
        self.check_for_client_worker()
        self.check_for_speaker_worker()
        self.startSpeakerWorker()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_api_model_buttons()

    def check_for_client_worker(self):
        if self.client_worker is None:
            self.client_worker = ClientWorker(self.client)
            self.client.data_changed.connect(self.updateAudioQueue)
            self.client.chatReceived.connect(self.updateChat)
            self.client.imageReceived.connect(self.updateImage)
            self.client.end_receive.connect(self.startSpeakerWorker)
            self.client.recorded_times.connect(self.saveLogs)
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
        #self.status_worker = Worker(self.displayStatus, text)
        #self.threadpool.start(self.status_worker)
        self.displayStatus(text)

    def reset_backend_chat(self):
        try:
            requests.get("http://localhost:5000/reset", timeout=2)
        except Exception as e:
            print("Could not reset backend chat:", e)

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

    def startAPIWorker(self):
        if self.api_worker is not None and self.api_worker.activated:
            return

        self.init = False
        self.set_ui_state("listening")
        self.api_worker.start()

    def changeColor(self):
        self.mic = not self.mic

    def request(self, data):
        """Send a GET request to FlaskAPI."""

        self.awaiting_response = True
        self.set_ui_state("thinking")

        url = "http://localhost:5000/request"
        starting_time = time.time()

        try:
            response = requests.get(url, data=data, timeout=10)

            a = response.content
            response_time = json.loads(a.decode("utf-8"))
            self.logs["sending_time"] = response_time["response"] - starting_time

        except Exception as e:
            print("Request failed:", e)
            self.awaiting_response = False
            self.set_ui_state("ready")


    def saveLogs(self, logs_bytes):
        try:
            if not logs_bytes:
                return

            decoded = logs_bytes.decode("utf-8").strip()
            if not decoded:
                return

            data = json.loads(decoded)

            if not isinstance(data, dict):
                return

            self.logs.update(data)

        except Exception as e:
            print("Logs parsing failed:", e)


    def clear_plot_queue(self):
        while not self.plot_queue.empty():
            try:
                self.plot_queue.get_nowait()
            except:
                break


    def getAudio(self):
        """Process the data in audio queue to be played and plotted.
        """
        CHUNK = self.CHUNK
        rate = self.samplerate
        audio_data = self.audio_queue.get_nowait()
        data = audio_data['data']
        format = audio_data['format']
        print("get audio")

        # Stream the audio using pyaudio
        p = pyaudio.PyAudio()
        
        # The welcome message is already recorded in the required format.
        # So it doesn't need extra processing as for audio from server.
        width = 2
        channels=1
        rate=48000    
        OCHUNK = 2 * CHUNK
        #initialize stream
        stream = p.open(format=p.get_format_from_width(width),
                        channels=channels,
                        rate=rate,
                        output=True,
                        frames_per_buffer=CHUNK)
        
        pause_ms = self.params.get("audio_reply_pause_ms", 180)
        pause_frames = int(rate * pause_ms / 1000)
        pause_bytes = b"\x00" * pause_frames * width * channels

        if format == '2**16':
            max_int16 = 2**16        
            print("Start audio\n")
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
                    #self.clear_plot_queue()
                    if not self.plot_queue.full():
                        self.plot_queue.put(audio_normalised)
                    else:
                        _ = self.plot_queue.get_nowait()
                        self.plot_queue.put(audio_normalised)
                    # Now that the data is ready for plotting,
                    # we start streaming so both happen synchronously.
                    stream.write(out)
                except Exception:
                    print("Broken\n")
                    break
            
        else:
            # If audio comes from the server, it needs to be saved to
            # a temporal file, to correct its format.
            print("temp audio")
            arr = np.frombuffer(data, dtype=np.float32)
            #sf.write("tmp.wav", arr, samplerate=24000)
            #wf = wave.open('tmp.wav', 'rb')
            #out = wf.readframes(CHUNK)

            audio_int16 = (arr * 32767).astype(np.int16)
            print("processed audio")
            #out = audio_int16.tobytes()

            max_int16 = 2**15   
            index = 0 #
            chunk_size = CHUNK
            while index < len(audio_int16):
                try:
                    if self.stop_audio:
                        if stream.is_active():
                            stream.stop_stream()
                        stream.close()
                        break

                    chunk = audio_int16[index:index + chunk_size]

                    out = chunk.tobytes()

                    audio_as_np_float32 = chunk.astype(np.float32)
                    audio_normalised = audio_as_np_float32 / 32768.0


                    stream.write(out)
                    #self.clear_plot_queue()
                    if not self.plot_queue.full():
                            self.plot_queue.put(audio_normalised)
                    else:
                        _ = self.plot_queue.get_nowait()
                        self.plot_queue.put(audio_normalised)
                    index += chunk_size

                    #audio_as_np_int16 = np.frombuffer(out, dtype=np.int16)
                    #audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
                    #audio_normalised = audio_as_np_float32 / max_int16
                    #self.clear_plot_queue()
                    #self.plot_queue.put_nowait(audio_normalised)
                    #stream.write(out)
                    #out = wf.readframes(CHUNK)
                except Exception as e:
                    print("Broken\n", e)
                    break

        #self.clear_plot_queue()
        #self.updateStatus("Ich bin verfügbar ...")

        if not self.stop_audio and pause_bytes:
            try:
                stream.write(pause_bytes)
            except Exception as e:
                print("Could not write pause:", e)

        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except Exception:
            pass

        try:
            p.terminate()
        except Exception:
            pass
        
        if self.speaker_worker is not None:
            self.speaker_worker.speaking = False
                    
    def updateAudioQueue(self, data):
        print("QUEUE FILLED:", time.time())

        self.awaiting_response = False

        self.audio_queue.put_nowait({
            'data': data,
            'format': '2**15'
        })

    def generatePlotData(self):
        """Gets plotting data from plot_queue.
        if the queue is empty, it returns an array of zeros to display silence.

        Returns:
            numpy.ndarray: plotting data.
        """

        if self.speaker_worker is None or not self.speaker_worker.speaking:
            return np.zeros((self.CHUNK, len(self.channels)))

        if not self.plot_queue.empty():
            audio_normalised = self.plot_queue.get_nowait()
            return audio_normalised
        else:
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
                    print("updatePlot", e)
                    pass

                if self.data.ndim == 1:
                    self.data = self.data[:, None]
                self.plotdata = self.data 

                self.ydata = self.plotdata
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
        except KeyboardInterrupt:
            QtWidgets.QApplication.quit()
        except Exception as e:
            print("draw", e)
            pass
               
    def speaking_finished(self):
        # Check if the worker itself exists and if its internal 'is_speaking' flag is False
        if self.speaker_worker is None:
            return True
        return not self.speaker_worker.speaking


    def closeEvent(self, event):
        self.stop_audio = True
        self.record_allowed = False
        self.plotting = False
        self.stopped = True
        self.speaking_allowed = False

        try:
            self.timer.stop()
        except Exception:
            pass

        try:
            if self.speaker_worker is not None:
                self.speaker_worker.stop_running()
                self.speaker_worker = None
        except Exception as e:
            print("Could not stop speaker worker:", e)

        try:
            if self.api_worker is not None and self.api_worker.isRunning():
                self.api_worker.requestInterruption()
                self.api_worker.recording = False

                if getattr(self.api_worker, "stream", None) is not None:
                    self.api_worker.stream.stop_stream()
                    self.api_worker.stream.close()

                self.api_worker.quit()
                self.api_worker.wait(1000)
        except Exception as e:
            print("Could not stop API worker:", e)

        try:
            if self.client_worker is not None:
                self.client_worker.stop()
                self.client_worker = None
        except Exception as e:
            print("Could not stop client worker:", e)

        try:
            requests.get("http://localhost:5000/close", data={}, timeout=1)
        except Exception as e:
            print("Could not notify server on close:", e)

        event.accept()

        QtCore.QTimer.singleShot(300, lambda: os._exit(0))

    def stop_speaker(self):
        if self.speaker_worker is not None:
            self.speaker_worker.stop_running()


def main():
    default_json_path = main_path + '/params.json'
    parser = argparse.ArgumentParser(description="Process a params file.")
    parser.add_argument(
        'json_file',
        type=str,
        nargs='?',
        default=default_json_path,
        help=f"Path to the JSON file (default: {main_path}/params.json)"
    )

    args = parser.parse_args()
    params_file_path = os.path.abspath(args.json_file)

    with open(params_file_path) as params_file:
        params = json.load(params_file)

    app = QtWidgets.QApplication(sys.argv)

    # Allows Python to process Ctrl+C while Qt event loop is running
    timer = QtCore.QTimer()
    timer.start(100)
    timer.timeout.connect(lambda: None)

    def handle_sigint(sig, frame):
        print("Ctrl+C received. Force exiting...")
        os._exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    params["window_size"] = QtWidgets.QDesktopWidget().screenGeometry(-1)
    mainWindow = MainUI(params=params)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

