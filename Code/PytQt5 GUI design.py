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
import os
import wave, pyaudio
import traceback
import librosa
import soundfile as sf
import BreezeStyleSheets.breeze_resources
from Models import *



params = {
	"CHUNK": 1024,
	"stt_model" : WhisperLargeV2(),
	"ttt_model" : FastChatModel(),
	"tts_model" : Bark(voice_preset = "v2/de_speaker_5"),
	"welcome_message" : "Hallo, Ich heiße Vicuna, wie kann ich dir helfen?",

}

class MplCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black')
		self.axes = fig.add_subplot(111)
		self.axes.axis("off")
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()

class PyShine_LIVE_PLOT_APP(QtWidgets.QMainWindow):
	def __init__(self, params=params):
		print("Initializing ... \n")
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('main.ui',self)
		self.resize(888, 600)
		self.tmpfile = 'temp.wav'

		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.setWindowIcon(icon)
		self.threadpool = QtCore.QThreadPool()	
		self.stt_pool = QtCore.QThreadPool()
		self.ttt_pool = QtCore.QThreadPool()
		self.tts_pool = QtCore.QThreadPool()
		self.params = params
		self.CHUNK = self.params["CHUNK"]
		self.q = queue.Queue(maxsize=self.CHUNK)
		
		self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
		self.ui.gridLayout_4.addWidget(self.canvas, 2, 1, 1, 1)
		self.reference_plot = None
		

		self.window_length = 1000
		self.downsample = 1
		self.channels = [1]
		self.interval = 1
		
		self.samplerate = 16000
		self.length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		
		self.plotdata =  np.zeros((self.length,len(self.channels)))
		self.timer = QtCore.QTimer()
		self.timer.setInterval(30) #msec
		self.timer.timeout.connect(self.update_plot)
		self.timer.start()
		self.data=[0]

		self.record_allowed = True
		self.query_allowed = False
		self.generating_allowed = False
		self.plotting = True
		self.stopped = False
		self.speaking_allowed =True

		self.stt_model = self.params["stt_model"]
		self.ttt_model = self.params["ttt_model"]
		self.tts_model = self.params["tts_model"]

		self.response=""
		self.query=""
		self.filepath=""


		self.label_6.setText("BOT Status:   IDLE  ... ")		
		self.welcome_message = self.params["welcome_message"]
		self.ttt_model.conv.append_message(self.ttt_model.conv.roles[1], self.welcome_message)
		self.filepath = self.generate_audio(self.welcome_message)
		
		print("Initialiazing is finished.\n")
		self.pushButton_2.clicked.connect(self.stop_worker)
		self.pushButton.clicked.connect(self.start_worker_4_speaker)

	

	def reset(self):
		self.q.queue.clear()
		self.record_allowed = True
		self.query_allowed = False
		self.generating_allowed = False
		self.plotting = True
		self.stopped = False
		self.speaking_allowed =True
		self.welcome_message = self.params["welcome_message"]
		self.ttt_model.conv.append_message(self.ttt_model.conv.roles[1], self.welcome_message)
		self.filepath = self.generate_audio(self.welcome_message)
		


	def display_status(self, text):
		self.label_6.setText(text)

	def update_status(self, text):
		status_worker = Worker(self.display_status, text)
		self.threadpool.start(status_worker)
		
	
	def record_speech(self):
		curr_time = time.time()
		print("Recording is started ...\n")
		self.query = self.stt_model.run()
		print(self.query)
		self.record_allowed = False
		self.query_allowed = True
		print("\nSpeech_to_text took {:.2f}s to finish\n".format(time.time()-curr_time))
		return self.query

	def start_worker_1_mic(self):
		print("start_worker_1 is reached\n")
		if self.record_allowed is True and self.stopped is False:
			print("Recording is allowed\n")
			mic_worker = STTWorker(self.record_speech)
			mic_worker.signals.start.connect(self.update_status)
			mic_worker.signals.finished.connect(self.start_worker_2_vicuna)
			self.stt_pool.start(mic_worker)	

		else:
			print("Recording isn't allowed\n")
		print("start_worker_1 is finished\n")
		
	
	def query_vicuna(self, query:str):
		curr_time = time.time()
		print("Querying Vicuna ...\n")
		self.response = self.ttt_model.generate_start(query)
		print(self.response)
		self.query_allowed = False
		self.generating_allowed = True
		print("\nText_to_text took {:.2f}s to finish\n".format(time.time()-curr_time))
		return self.response
	
	def start_worker_2_vicuna(self, emit_sig:int):
		
		if emit_sig==1 and self.query_allowed and self.stopped is False:
			vicuna_worker = VicunaWorker(self.query_vicuna, query=self.query)
			vicuna_worker.signals.start.connect(self.update_status)
			vicuna_worker.signals.finished.connect(self.start_worker_3_tts)
			self.ttt_pool.start(vicuna_worker)
		elif emit_sig!=1:
			print("The mic wasn't able to record!\n")
		
		elif not self.query_allowed:
			print("Vicuna isn't available!\n")
		else:
			print("Unknown problem while querying! \n")

	def generate_audio(self, response):
		curr_time=time.time()
		self.filepath = self.tts_model.run(response)
		self.generating_allowed = False
		self.speaking_allowed=True
		print("Text_to_Speech thread took {:.2f}s to finish\n".format(time.time()-curr_time))
		return self.filepath
	
	def start_worker_3_tts(self, emit_sig):
		if emit_sig ==1 and self.generating_allowed and self.stopped is False:
			tts_worker = TTSWorker(self.generate_audio, response=self.response)
			tts_worker.signals.start.connect(self.update_status)
			tts_worker.signals.finished.connect(self.start_worker_4_speaker)
			self.tts_pool.start(tts_worker)
		else:
			print("Vicuna wasn't able to process your query!\n")
	
	def generate_plotdata(self):
		if not self.q.empty():
			data = self.q.get_nowait()
			return data
		else:
			return np.zeros((self.CHUNK,len(self.channels)))
		
	def getAudio(self, filepath):
		curr_time=time.time()
		if self.speaking_allowed is True:	
			CHUNK = self.CHUNK
			x,_ = librosa.load(filepath, sr=16000)
			sf.write('tmp.wav', x, 16000)
			wf = wave.open('tmp.wav', 'rb')
			p = pyaudio.PyAudio()
			stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
							channels=wf.getnchannels(),
							rate=wf.getframerate(),
							output=True,
							frames_per_buffer=CHUNK)
			self.samplerate = wf.getframerate()
			sd.default.samplerate = self.samplerate
			data = wf.readframes(CHUNK)
			self.label_6.setText("BOT Status:   SPEAKING  ... ")
			while(self.speaking_allowed is True and data):
				audio_as_np_int16 = np.frombuffer(data, dtype=np.int16)
				audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
				# Normalise float32 array                                                   
				max_int16 = 2**15
				audio_normalised = audio_as_np_float32 / max_int16
				try:
					self.q.put_nowait(audio_normalised)
					stream.write(data)
					data = wf.readframes(CHUNK)
				except:
					self.speaking_allowed = False
			
		self.speaking_allowed=False
		self.record_allowed = True
		print("Playing and Plotting took {:.2f}s to finish\n".format(time.time()-curr_time))


	def start_worker_4_speaker(self, emit_sig):
		if self.speaking_allowed and self.stopped is False:
			speaker_worker = Worker(self.getAudio, filepath=self.filepath)
			speaker_worker.signals.finished.connect(self.start_worker_1_mic)
			self.threadpool.start(speaker_worker)
		else:
			print("The model couldn't generate a speech from the text response\n")

	def stop_worker(self):
		self.label_6.setText("BOT Status:   STOPPED  ... ")
		self.stopped = True
		self.record_allowed = False
		self.query_allowed = False
		self.generating_allowed = False
		self.speaking_allowed = False
		self.plotting = False
		self.timer.stop()
		print("You have stopped all threads!\n")
		with self.q.mutex:
			self.q.queue.clear()
		
			

	def update_plot(self):
		try:
			if self.timer.isActive():
				threads_num = self.threadpool.activeThreadCount() + self.stt_pool.activeThreadCount() + self.ttt_pool.activeThreadCount() + self.tts_pool.activeThreadCount() + 1
			else:
				threads_num = self.threadpool.activeThreadCount() + self.stt_pool.activeThreadCount() + self.ttt_pool.activeThreadCount() + self.tts_pool.activeThreadCount()
			print('ACTIVE THREADS:',threads_num,end=" \r")
			if  self.plotting is True:
				try: 
					self.data = self.generate_plotdata()
					
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

	@pyqtSlot()
	def run(self):
		self.function(*self.args, **self.kwargs)
		self.signals.finished.emit()			
	

class STTSignals(QtCore.QObject):
	start = QtCore.pyqtSignal(str)
	finished = QtCore.pyqtSignal(int)
	result = QtCore.pyqtSignal(str)

# www.pyshine.com
class STTWorker(QtCore.QRunnable):

	def __init__(self, function, *args, **kwargs):
		super(STTWorker, self).__init__()
		self.function = function
		self.args = args
		self.kwargs = kwargs
		self.signals = STTSignals()

	@pyqtSlot()
	def run(self):
		self.signals.start.emit("BOT Status:   lISTENING  ... ")
		try:
			result = self.function(*self.args, **self.kwargs)
		except:
			self.signals.finished.emit(0)
		else:
			self.signals.finished.emit(1)	


class VicunaSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal(int)
	error = QtCore.pyqtSignal(tuple)
	result = QtCore.pyqtSignal(str)
	progress = QtCore.pyqtSignal(str)
	start = QtCore.pyqtSignal(str)

# www.pyshine.com
class VicunaWorker(QtCore.QRunnable):

	def __init__(self, function, *args, **kwargs):
		super(VicunaWorker, self).__init__()
		self.function = function
		self.args = args
		self.kwargs = kwargs
		self.signals = VicunaSignals()

	@pyqtSlot()
	def run(self):
		self.signals.start.emit("BOT Status:   PROCESSING  ... ")
		# Retrieve args/kwargs here; and fire processing using them
		try:
			result = self.function(*self.args, **self.kwargs)
		except:
			traceback.print_exc()
			exctype, value = sys.exc_info()[:2]
			self.signals.finished.emit(0)
		else:
			self.signals.finished.emit(1)



class TTSSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal(int)
	# result = QtCore.pyqtSignal(str)
	start = QtCore.pyqtSignal(str)

# www.pyshine.com
class TTSWorker(QtCore.QRunnable):

	def __init__(self, function, *args, **kwargs):
		super(TTSWorker, self).__init__()
		self.function = function
		self.args = args
		self.kwargs = kwargs
		self.signals = TTSSignals()
	@pyqtSlot()
	def run(self):
		self.signals.start.emit("BOT Status:   GENERATING  ... ")
		try:
			result = self.function(*self.args, **self.kwargs)
		except:
			self.signals.finished.emit(0)
		else:
			self.signals.finished.emit(1)
		

def main():
	app = QtWidgets.QApplication(sys.argv)

	# set stylesheet
	file = QFile(":/dark/stylesheet.qss")
	file.open(QFile.ReadOnly | QFile.Text)
	stream = QTextStream(file)
	app.setStyleSheet(stream.readAll())
	# code goes here
	mainWindow = PyShine_LIVE_PLOT_APP()
	mainWindow.showMaximized() 

	app.exec_()



main()