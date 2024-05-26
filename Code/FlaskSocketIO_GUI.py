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
from PyQt5 import QtCore, QtWidgets
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

class APIWorker(QtCore.QRunnable):
    """QRunnable to record audio from microphone and emit data to ClientWorker.
    """
    def __init__(self):
        super(APIWorker, self).__init__()
        self.r = sr.Recognizer()
        self.signals = APISignals()
        
        
    def run(self):
        # print("STTWorker\n")

        with sr.Microphone() as source:
            # print("Microphone is started ... \n")
            # Adjust the microphone to ignore unwanted noises.
            self.r.adjust_for_ambient_noise(source)
            
            # print("Microphone is listening ... \n")
            #Emit a status signal to change BOT Status
            self.signals.start.emit("Ich höre zu  ... ")
            #Start recording
            audio = self.r.listen(source)
            #Emit a status signal to change BOT Status
            self.signals.start.emit("Ich überlege was ich antworte  ... ")
            
            #Emit the recorded data as bytes.
            self.signals.result.emit(audio.get_flac_data())
            self.signals.finished.emit()	
            


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

    def __init__(self, parent=None):
        super().__init__(parent)

        self.sio.on("connect", self._handle_connect, namespace=None)
        self.sio.on("connect_error", self._handle_connect_error, namespace=None)
        self.sio.on("disconnect", self._handle_disconnect, namespace=None)
        self.sio.on("voice reply", self.receiveAudio, namespace=None)
        self.sio.on("request", self.endReceiving, namespace=None)

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
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self, params["window_size"])
        # self.resize(888, 600)
        
        #Change the font size of BOT status
        # self.ui.label_6.setStyleSheet(''' font-size: 100px; color: Red;''')

        # self.label_6.setMargin(30)
        # self.ui.label_6.setIndent(60)
        #Left, above, right, under
        # self.ui.buttonsLayout.setContentsMargins(120, 500, 120, 0)

        # self.ui.startButton.setStyleSheet('QPushButton {background-color: #555555; font-size: 50px; color: white;}')
        # self.ui.resetButton.setStyleSheet('QPushButton {background-color: #555555; font-size: 50px; color: white;}')
        # self.gridLayout_5.setVerticalSpacing(5)
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
        self.speaking_allowed =True
    
        self.ui.startButton.setEnabled(False)
        self.ui.resetButton.setEnabled(False)
        
        self.status_worker = None
        self.r = sr.Recognizer()
        self.audio_queue.put_nowait(self.plotdata)
        
        self.speaker_worker = AudioOutputWorker(function=self.getAudio, audio_queue=self.audio_queue)
        self.speaker_worker.signals.waiting.connect(self.startAPIWorker)
        
        self.speaker_worker.signals.status.connect(self.updateStatus)
        self.threadpool.start(self.speaker_worker)
        self.init = True
        self.client = Client()
        self.client_worker = None
        self.signals = GUISignals()
        self.logs={}

        
 
        self.ui.startButton.clicked.connect(self.start)
        self.ui.resetButton.clicked.connect(self.reset)


        self.ui.startButton.setEnabled(True)
        self.ui.resetButton.setEnabled(True)
        
    def reset(self):
        """Reset the project back to the starting point.
        It doesn't terminate the current task, but it stops the all future ones.
        """
        if not self.speaker_worker is None:
            self.speaker_worker.stop_running()

        self.ui.label_6.setText("Ich würde gestoppt  ... ")
        self.stopped = True
        self.speaking_allowed = False
        self.speaker_worker = None
        
        
        with self.plot_queue.mutex:
            self.plot_queue.queue.clear()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        self.plot_queue = queue.Queue(maxsize=self.CHUNK)
        self.audio_queue = queue.Queue()
        self.plotdata =  np.zeros((self.length,len(self.channels)))
        #Change BOT Status
        self.displayStatus("Ich schlafe  ... ")
    
    def start(self):
        """Starts when Start button is clicked.
        Play a pre-generated welcome message.
        If the Reset is already clicked, re-initialize and start both client_worker and speaker_worker.
        """
        wf = wave.open("welcome_message.wav")
        data = wf.readframes(-1)
        self.audio_queue.put_nowait(data)
        # print("Initialiazing is finished.\n")
        if self.client_worker is None:
            self.client_worker = ClientWorker(self.client)
            self.client.data_changed.connect(self.updateAudioQueue)
            self.client.end_receive.connect(self.startSpeakerWorker)
            self.client.recorded_times.connect(self.saveLogs)
            self.client_pool.start(self.client_worker)
        if self.speaker_worker is None:
            self.stopped = False
            self.speaking_allowed = True
            self.plotting = True
            self.speaker_worker = AudioOutputWorker(function=self.getAudio, audio_queue=self.audio_queue)
            self.speaker_worker.signals.waiting.connect(self.startAPIWorker)
            self.speaker_worker.signals.status.connect(self.updateStatus)
            self.threadpool.start(self.speaker_worker)
            self.init = True
            
        self.startSpeakerWorker()
    
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
        self.init=False
        api_worker = APIWorker()
        api_worker.signals.start.connect(self.displayStatus)
        api_worker.signals.start.connect(self.changeColor)
        api_worker.signals.result.connect(self.request)
        api_worker.signals.finished.connect(self.changeColor)

        self.threadpool.start(api_worker)

    def changeColor(self):
        self.mic = not self.mic

    
    def request(self, data):
        """Send a GET request to FlaskAPI.

        Args:
            data (bytes): the recorded audio.
        """
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
            data = self.audio_queue.get_nowait()
            
            # Stream the audio using pyaudio
            p = pyaudio.PyAudio()
            
            # The welcome message is already recorded in the required format.
            # So it doesn't need extra processing as for audio from server.
            if self.init:
                width = 2
                channels=1
                rate=24000
                max_int16 = 2**16
                OCHUNK = 2 * CHUNK
                
                #initialize stream
                stream = p.open(format=p.get_format_from_width(width),
                                channels=channels,
                                rate=rate,
                                output=True,
                                frames_per_buffer=CHUNK)
                # print("Start audio\n")
                plotting = True
                
                while plotting:
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
                        self.plot_queue.put_nowait(audio_normalised)
                        stream.write(out)
                        out = wf.readframes(CHUNK)
                    except:
                        print("Broken\n")
                        break
                    
    def updateAudioQueue(self, data):
        """Add a new entry to audio queue.

        Args:
            data (bytes): audio data
        """
        self.audio_queue.put_nowait(data)
    
    def generatePlotData(self):
        """Gets plotting data from plot_queue.
        if the queue is empty, it returns an array of zeros to display silence.

        Returns:
            numpy.ndarray: plotting data.
        """
        if self.mic:
            color = (0,1,0.29)
        else:
            color='red'
        if not self.plot_queue.empty():
            audio_normalised = self.plot_queue.get_nowait()
            return audio_normalised, color
        else:
            self.speaker_worker.speaking=False
            return np.zeros((self.CHUNK,len(self.channels))), color
    
    def startSpeakerWorker(self):
        """Change both idle and waiting states of AudiooutputWorker.
        """
        if self.plotting and not self.stopped:
            self.speaker_worker.dont_wait()
            self.speaker_worker.wake()

        else:
            print("The model couldn't generate a speech from the text response\n")
    
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
                    self.data, color = self.generatePlotData()
                except:
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

def main():
    params = {
	"CHUNK": 1024,
    "samplerate": 24000,
    "linewidth": 8,
    "color": "#04d9ff"
    # "color": "#0BBAE2"
    # "color": "#2CC3FF"
    # "color": 'blue'
	}
    app = QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen()
    params["window_size"] = screen.size()
    # file = QFile(":/dark/stylesheet.qss")
    # file.open(QFile.ReadOnly | QFile.Text)
    # stream = QTextStream(file)
    # app.setStyleSheet(stream.readAll())
    mainWindow = MainUI(params=params)
    # mainWindow.showMaximized()
    # mainWindow.showFullScreen()
    app.exec_()


if __name__ == '__main__':
    main()

