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
import sounddevice as sd
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, QFile, QTextStream
import time
import os, socket
import wave, pyaudio
import traceback
import librosa
import soundfile as sf
import BreezeStyleSheets.breeze_resources
from Models import *
# import debugpy
import requests
import json
import socketio
import pickle
import struct
import asyncio
# from asyncqt import QEventLoop

from functools import cached_property


main_path = os.path.dirname(os.path.realpath(__file__))
logs_path = main_path + "/logs/"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)

class MplCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black')
		self.axes = fig.add_subplot(111)
		self.axes.axis("off")
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()


class WorkerSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal()

# www.pyshine.com
class Worker(QtCore.QRunnable):

	def __init__(self, function, *args, **kwargs):
		super(Worker, self).__init__()
		self.function = function
		self.args = args
		self.kwargs = kwargs
		self.signals = WorkerSignals()

	def run(self):
		self.function(*self.args, **self.kwargs)
		self.signals.finished.emit()		





class AudioOutputWorkerSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal()
	waiting = QtCore.pyqtSignal()
	status = QtCore.pyqtSignal(str)

# www.pyshine.com
class AudioOutputWorker(QtCore.QRunnable):
    def __init__(self, function, audio_queue, CHUNK, *args, **kwargs):
        super(AudioOutputWorker, self).__init__()
        self.function = function
        self.audio_queue = audio_queue
        self.active = True
        self.idle = True
        self.should_wait = True
        self.speaking = True
        self.CHUNK = CHUNK
        self.args = args
        self.kwargs = kwargs
        self.signals = AudioOutputWorkerSignals()
    
    def run(self):
        print("AudioOutputWorker\n")
        while self.active:
            if not self.idle:
                if not self.audio_queue.empty():
                    self.signals.status.emit("BOT Status:   SPEAKING  ... ")	
                    self.function(*self.args, **self.kwargs)
                else:
                    if not self.should_wait and not self.speaking:
                        self.signals.status.emit("BOT Status:   WAITING  ... ")
                        print("######### I'm waiting #########\n")
                        self.signals.waiting.emit()
                        self.should_wait = True
        self.signals.finished.emit()
    
    def stop_running(self):
        self.active=False
    
    def dont_wait(self):
        self.should_wait = False
    
    def wake(self):
        self.idle = False

class APISignals(QtCore.QObject):
    start = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    result = QtCore.pyqtSignal(bytes)
    # result = QtCore.pyqtSignal(str)

# www.pyshine.com
class APIWorker(QtCore.QRunnable):
    def __init__(self, *args, **kwargs):
        super(APIWorker, self).__init__()
        self.r = sr.Recognizer()
        self.args = args
        self.kwargs = kwargs
        self.signals = APISignals()
        
        
    def run(self):
        print("STTWorker\n")
        payload_size = struct.calcsize("Q")
        with sr.Microphone() as source:
            print("Microphone is started ... \n")
            self.r.adjust_for_ambient_noise(source)
            print("Microphone is listening ... \n")
            self.signals.start.emit("BOT Status:   lISTENING  ... ")
            audio = self.r.listen(source)
            self.signals.start.emit("BOT Status:   PROCESSING  ... ")
            self.signals.result.emit(audio.get_flac_data())
            self.signals.finished.emit()	
            



# www.pyshine.com
class ClientWorker(QtCore.QRunnable):
    def __init__(self, client, *args, **kwargs):
        super(ClientWorker, self).__init__()
        self.client = client
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        loop = asyncio.new_event_loop()
        loop.create_task(self.client.start())
        loop.run_forever()


class Client(QtCore.QObject):
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal()
    error_ocurred = QtCore.pyqtSignal(object, name="errorOcurred")
    data_changed = QtCore.pyqtSignal(bytes, name="dataChanged")
    end_receive = QtCore.pyqtSignal()
    recorded_times = QtCore.pyqtSignal(bytes)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.sio.on("connect", self._handle_connect, namespace=None)
        self.sio.on("connect_error", self._handle_connect_error, namespace=None)
        self.sio.on("disconnect", self._handle_disconnect, namespace=None)
        self.sio.on("client_Unlock", self.client_unlock_ack, namespace=None)
        self.sio.on("voice reply", self.receiveAudio, namespace=None)
        self.sio.on("request", self.endReceiving, namespace=None)

    @cached_property
    def sio(self):
        return socketio.AsyncClient(
            # reconnection=True,
            # reconnection_attempts=3,
            # reconnection_delay=5,
            # reconnection_delay_max=5,
            # logger=True,
        )

    async def start(self):
        await self.sio.connect(url="http://127.0.0.1:5000", transports="polling")
        # await self.sio.connect(url="http://127.0.0.1:8080", transports="websocket")
        # await self.sio.emit('/my_adsf_event', data={"recorded": "hello"})
        await self.sio.wait()

    def _handle_connect(self):
        self.connected.emit()

    def _handle_disconnect(self):
        self.disconnected.emit()

    def _handle_connect_error(self, data):
        self.error_ocurred.emit(data)

    def client_unlock_ack(self, data):
        print("client_unlock_ack was reached")
        # self.data_changed.emit(data)

    def receiveAudio(self, data):
        print("receiveAudio was reached")
        # print(data)
        self.data_changed.emit(data["voice_answer"])
    
    def endReceiving(self, logs_bytes):
        print("endReceiving was reached")
        self.end_receive.emit()
        self.recorded_times.emit(logs_bytes)

    async def receiveFromGUI(self, data):
        # print("here", data)
        # data = "Hallo Vikuna, ich heiße Salim. Wie geht's dir?"
        request = {"recorded": data,
                "start_time":time.time(),
                "request_length": len(data)}
        await self.sio.emit('request', data=request)
        # await self.sio.send(data={"recorded": "hello"})
        print("Request was sent")

    async def reset(self):
        await self.sio.emit('reset')


class GUISignals(QtCore.QObject):
    request = QtCore.pyqtSignal()

class MainUI(QtWidgets.QMainWindow):
    def __init__(self, params):
        print("Initializing ... \n")
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi(main_path+'/main.ui',self)
        self.resize(888, 600)
        icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.threadpool = QtCore.QThreadPool()
        self.client_pool = QtCore.QThreadPool()		
        self.API_pool = QtCore.QThreadPool()
        self.stt_model = WhisperLargeV2(device="cpu")
        self.params = params
        self.CHUNK = self.params["CHUNK"]
        
        self.plot_queue = queue.Queue(maxsize=self.CHUNK)
        self.audio_queue = queue.Queue()
        
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.ui.gridLayout_4.addWidget(self.canvas, 2, 1, 1, 1)
        self.reference_plot = None
        
        self.window_length = 1000
        self.downsample = 1
        self.channels = [1]
        # self.interval = 1
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
        self.speaking_allowed =True
        self.stopButton.setEnabled(False)
        self.startButton.setEnabled(False)
        self.resetButton.setEnabled(False)
        
        self.status_worker = None
        self.r = sr.Recognizer()
        self.audio_queue.put_nowait(self.plotdata)
        
        self.speaker_worker = AudioOutputWorker(function=self.getAudio, audio_queue=self.audio_queue, CHUNK=self.CHUNK)
        self.speaker_worker.signals.waiting.connect(self.startAPIWorker)
        # self.speaker_worker.signals.waiting.connect(self.request)
        self.speaker_worker.signals.status.connect(self.updateStatus)
        self.threadpool.start(self.speaker_worker)
        self.init = True
        self.client = Client()
        self.signals = GUISignals()
        self.logs={}

        
        # self.start()
        
        self.stopButton.clicked.connect(self.stopWorkers)
        self.startButton.clicked.connect(self.start)
        self.resetButton.clicked.connect(self.reset)

        self.stopButton.setEnabled(True)
        self.startButton.setEnabled(True)
        self.resetButton.setEnabled(True)
        
                    

    def reset(self):
        if self.stopped:
            # asyncio.run(self.client.reset())
            self.plot_queue = queue.Queue(maxsize=self.CHUNK)
            self.audio_queue = queue.Queue()
            self.plotdata =  np.zeros((self.length,len(self.channels)))
            self.record_allowed = True
            self.plotting = True
            self.stopped = False
            self.speaking_allowed =True
            wf = wave.open("welcome_message_2.wav")
            data = wf.readframes(-1)
            self.audio_queue.put_nowait(data)
            self.displayStatus("BOT Status:   IDLE  ... ")
        else:
            print("You need to stop the process first.")
    
    def start(self):
        wf = wave.open("welcome_message_2.wav")
        data = wf.readframes(-1)
        self.audio_queue.put_nowait(data)
        print("Initialiazing is finished.\n")
        client_worker = ClientWorker(self.client)
        self.client.data_changed.connect(self.updateAudioQueue)
        self.client.end_receive.connect(self.startSpeakerWorker)
        self.client.recorded_times.connect(self.saveLogs)
        self.client_pool.start(client_worker)
        self.startSpeakerWorker()
    
    def displayStatus(self, text):
        self.label_6.setText(text)
    
    def updateStatus(self, text):
        self.status_worker = Worker(self.displayStatus, text)
        self.threadpool.start(self.status_worker)
    
    def startAPIWorker(self):
        self.init=False
        api_worker = APIWorker()
        api_worker.signals.start.connect(self.displayStatus)
        api_worker.signals.result.connect(self.request)
        # api_worker.signals.finished.connect(self.startSpeakerWorker)
        self.threadpool.start(api_worker)

    # def request(self):
    #     self.init=False
    #     with sr.Microphone() as source:
    #         print("Microphone is started ... \n")
    #         self.r.adjust_for_ambient_noise(source)
    #         print("Microphone is listening ... \n")
    #         # self.updateStatus("BOT Status:   lISTENING  ... ")
    #         audio = self.r.listen(source)
    #         # self.updateStatus("BOT Status:   PROCESSING  ... ")
    #         # self.signals.result.emit(audio.get_flac_data())
    #         print("To Transcribion")
    #         transcribtion = self.stt_model.run(audio.get_flac_data())
    #         print(transcribtion)

    #     asyncio.run(self.client.receiveFromGUI(transcribtion))
    
    def request(self, data):
        url = "http://localhost:5000/request"
        # print(data)
        # dict_data = {"recorded": data.decode('latin-1').replace("'", '"'),
        #         "start_time":time.time(),
        #         "request_length": len(data)}
        
        # dict_data = {"recorded": "data",
        #         "start_time":"time.time()",
        #         "request_length": "len(data)"}
        starting_time = time.time()
        print("Request is sent.\n")
        response = requests.get(url, data=data)
    
        a= response.content
        response_time = json.loads(a.decode('utf-8'))
        self.logs["sending_time"] = response_time["response"] - starting_time
    
    def saveLogs(self, logs_bytes):
        self.logs = self.logs | json.loads(logs_bytes.decode('utf-8'))
        self.logs["first_response"] = self.logs["first_response"] + self.logs["sending_time"] * 2
        self.logs["total_time"] = self.logs["total_time"] + self.logs["sending_time"] * 2

        for dirname, _, filenames in os.walk(logs_path):
            file_name = "time_logs_{}.json".format(len(filenames))
            file_path = os.path.join(dirname, file_name)
            with open(file_path, "w") as file:
                json.dump(self.logs, file)
                print(file_path)

    def getAudio(self):
        if not self.audio_queue.empty():
            print("Getting Audio\n")
            CHUNK = self.CHUNK
            rate = self.samplerate
            data = self.audio_queue.get_nowait()

            p = pyaudio.PyAudio()
            if self.init:
                width = 2
                channels=1
                rate=24000
                max_int16 = 2**16
                OCHUNK = 2 * CHUNK
                stream = p.open(format=p.get_format_from_width(width),
                                channels=channels,
                                rate=rate,
                                output=True,
                                frames_per_buffer=CHUNK)
                print("Start audio\n")
                plotting = True
                while plotting:
                    if len(data) > OCHUNK:
                        out = data[:OCHUNK]
                        data = data[OCHUNK:]
                    else:
                        out = data
                        plotting = False
                    try:
                        audio_as_np_int16 = np.frombuffer(out, dtype=np.int16)
                        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
                        
                        audio_normalised = audio_as_np_float32 / max_int16
                        # print(audio_normalised.shape)
                        self.plot_queue.put_nowait(audio_normalised)
                        stream.write(out)
                    except:
                        print("Broken\n")
                        break
                
            else:
                arr = np.frombuffer(data, dtype=np.float32)
                sf.write("tmp.wav", arr, samplerate=24000)
                wf = wave.open('tmp.wav', 'rb')
                width =2
                channels=1
                rate = 24000
                max_int16 = 2**15
                OCHUNK = 2 * CHUNK
                stream = p.open(format=p.get_format_from_width(width),
                                channels=channels,
                                rate=rate,
                                output=True,
                                frames_per_buffer=CHUNK)

                out = wf.readframes(CHUNK)
                while out:
                    try:
                        audio_as_np_int16 = np.frombuffer(out, dtype=np.int16)
                        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
                        
                        audio_normalised = audio_as_np_float32 / max_int16
                        # print(audio_normalised.shape)
                        self.plot_queue.put_nowait(audio_normalised)
                        stream.write(out)
                        out = wf.readframes(CHUNK)
                    except:
                        print("Broken\n")
                        break
                    
    def updateAudioQueue(self, data):
        self.audio_queue.put_nowait(data)
    
    def generatePlotData(self):
        if not self.plot_queue.empty():
            audio_normalised = self.plot_queue.get_nowait()
            return audio_normalised
        else:
            self.speaker_worker.speaking=False
            return np.zeros((self.CHUNK,len(self.channels)))
    
    def startSpeakerWorker(self):
        if self.plotting and not self.stopped:
            self.speaker_worker.dont_wait()
            self.speaker_worker.wake()

        else:
            print("The model couldn't generate a speech from the text response\n")
    
    def updatePlotQueue(self, plot_data):
        self.plot_queue.put_nowait(plot_data)
    
    def stopWorkers(self):
        self.label_6.setText("BOT Status:   STOPPED  ... ")
        self.stopped = True
        self.speaking_allowed = False
        self.plotting = False
        self.speaker_worker.stop_running()
        print("You have stopped all threads!\n")
        with self.plot_queue.mutex:
            self.plot_queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
    
    def updatePlot(self):
        try:
            if  self.plotting is True:
                try: 
                    self.data = self.generatePlotData()
                except:
                    pass
                shift = len(self.data)
                self.plotdata = np.roll(self.plotdata, -shift,axis = 0)
                self.plotdata = self.data
                self.ydata = self.plotdata[:]
                self.canvas.axes.set_facecolor((0,0,0))
                
            if self.reference_plot is None:
                plot_refs = self.canvas.axes.plot( self.ydata, color=(0,1,0.29))
                self.reference_plot = plot_refs[0]	
            else:
                self.reference_plot.set_ydata(self.ydata)
            
            self.canvas.axes.yaxis.grid(True,linestyle='--')
            start, end = self.canvas.axes.get_ylim()
            self.canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.1))
            self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.canvas.axes.set_ylim( ymin=-1, ymax=1)
            self.canvas.draw()
        except Exception as e:
            pass

def main():
    params = {
	"CHUNK": 1024,
    "samplerate": 24000
	}
    app = QtWidgets.QApplication(sys.argv)
    file = QFile(":/dark/stylesheet.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())
    mainWindow = MainUI(params=params)
    mainWindow.showMaximized()
    app.exec_()


if __name__ == '__main__':
    main()

