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



class MplCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black')
		self.axes = fig.add_subplot(111)
		self.axes.axis("off")
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()

class PyShine_LIVE_PLOT_APP(QtWidgets.QMainWindow):
	def __init__(self, params):
		print("Initializing ... \n")
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('new_main.ui',self)
		self.resize(888, 600)
		self.tmpfile = 'temp.wav'
		self.wav_file = "Files/output_"
		self.iter = 0
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.setWindowIcon(icon)
		self.threadpool = QtCore.QThreadPool()	
		self.stt_pool = QtCore.QThreadPool()
		self.ttt_pool = QtCore.QThreadPool()
		self.tts_pool = QtCore.QThreadPool()
		self.params = params
		self.CHUNK = self.params["CHUNK"]
		#### Change this 
		self.plot_queue = queue.Queue(maxsize=self.CHUNK)
		self.txt_queue = queue.Queue()
		self.audio_queue = queue.Queue()
		
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
		self.timer.setInterval(10) #msec
		self.timer.timeout.connect(self.update_plot)
		self.timer.start()
		self.data=[0]

		self.record_allowed = True
		self.query_allowed = False
		self.generating_allowed = True
		self.plotting = True
		self.stopped = False
		self.speaking_allowed =True
		self.pushButton_2.setEnabled(False)
		self.pushButton.setEnabled(False)
		# print("Loading STT Model ... \n")
		self.stt_model = self.params["stt_model"]
		# print("Loading TTT Model ... \n")
		self.ttt_model = self.params["ttt_model"]
		# print("Loading TTS Model ... \n")
		self.tts_model = self.params["tts_model"]

		self.response=""
		self.query=""
		self.filepaths=[]


		# self.label_6.setText("BOT Status:   IDLE  ... ")		
		self.welcome_message = self.params["welcome_message"]
		self.ttt_model.conv.append_message(self.ttt_model.conv.roles[1], self.welcome_message)
		self.txt_queue.put_nowait(self.welcome_message)
		
		tts_worker = TTSWorker(tts_model = self.tts_model, txt_queue=self.txt_queue)
		tts_worker.signals.update_plot.connect(self.update_audio_queue)
		tts_worker.signals.start.connect(self.update_status)
		tts_worker.signals.init.connect(self.start)
		self.tts_pool.start(tts_worker)

		self.pushButton_2.clicked.connect(self.stop_worker)
		self.pushButton.clicked.connect(self.start_worker_4_speaker)
		# self.pushButton.clicked.connect(self.start_worker_1_mic)

	def start(self):
		print("Initialiazing is finished.\n")
		self.pushButton_2.setEnabled(True)
		self.pushButton.setEnabled(True)
		


	def display_status(self, text):
		self.label_6.setText(text)

	def update_status(self, text):
		status_worker = Worker(self.display_status, text)
		self.threadpool.start(status_worker)
		
	
	def record_speech(self):
		print("Recording is started ...\n")
		self.query = self.stt_model.run()
		print("self.stt_model.run() ...\n")
		print(self.query)
		self.record_allowed = False
		self.query_allowed = True
		print("\nSpeech_to_text is finished\n")
		return self.query

	def start_worker_1_mic(self):
		print("start_worker_1 is reached\n")
		if not self.stopped:
			print("Recording is allowed\n")
			mic_worker = STTWorker(self.record_speech)
			mic_worker.signals.start.connect(self.update_status)
			mic_worker.signals.finished.connect(self.start_worker_2_vicuna)
			self.stt_pool.start(mic_worker)	

		else:
			print("Recording isn't allowed\n")
		
	
	def query_vicuna(self, query:str):
		curr_time = time.time()
		print("Querying Vicuna ...\n")
		self.response =""
		for outputs in self.tts_model.run(query):
			self.response+= " " + outputs
			print("\n\n", outputs, "\n\n")
			self.txt_queue.put(outputs)
		self.query_allowed = False
		self.generating_allowed = True
		print("\nText_to_text took {:.2f}s to finish\n".format(time.time()-curr_time))
		self.response = self.response.strip()
		return self.response
	
	def start_worker_2_vicuna(self, emit_sig:int):
		
		if emit_sig==1 and self.query_allowed and not self.stopped:
			# vicuna_worker = VicunaWorker(self.query_vicuna, query=self.query)
			vicuna_worker = VicunaWorker(self.ttt_model, query=self.query)
			vicuna_worker.signals.start.connect(self.update_status)
			vicuna_worker.signals.progress.connect(self.update_txt_queue)
			# vicuna_worker.signals.progress.connect(self.start_worker_3_tts)
			vicuna_worker.signals.finished.connect(self.start_worker_4_speaker)
			self.generating_allowed = True
			# vicuna_worker.signals.finished.connect(self.start_worker_3_tts)
			self.ttt_pool.start(vicuna_worker)
		elif emit_sig!=1:
			print("The mic wasn't able to record!\n")
		
		elif not self.query_allowed:
			print("Vicuna isn't available!\n")
		else:
			print("Unknown problem while querying! \n")
	
	def update_txt_queue(self, txt):
		self.txt_queue.put_nowait(txt)

	
	
	def update_audio_queue(self, data:bytes):
		self.audio_queue.put_nowait(data)
	
	def generate_plotdata(self):
		if not self.plot_queue.empty():
			audio_as_np_int16 = np.frombuffer(self.plot_queue.get_nowait(), dtype=np.int16)
			audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
			max_int16 = 2**15
			audio_normalised = audio_as_np_float32 / max_int16
			return audio_normalised
		else:
			return np.zeros((self.CHUNK,len(self.channels)))


	def start_worker_4_speaker(self):
		if self.plotting and not self.stopped:
			speaker_worker = AudioOutputWorker(audio_queue=self.audio_queue, CHUNK=self.CHUNK)
			speaker_worker.signals.waiting.connect(self.start_worker_1_mic)
			speaker_worker.signals.status.connect(self.update_status)
			speaker_worker.signals.progress.connect(self.update_plot_queue)
			self.threadpool.start(speaker_worker)
		else:
			print("The model couldn't generate a speech from the text response\n")

	def update_plot_queue(self, plot_data):
		self.plot_queue.put_nowait(plot_data)

	def stop_worker(self):
		self.label_6.setText("BOT Status:   STOPPED  ... ")
		self.stopped = True
		self.record_allowed = False
		self.query_allowed = False
		self.generating_allowed = False
		self.speaking_allowed = False
		self.plotting = False
		print("You have stopped all threads!\n")
		with self.plot_queue.mutex:
			self.plot_queue.queue.clear()
		
			

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



class AudioOutputWorkerSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal()
	progress = QtCore.pyqtSignal(bytes)
	waiting = QtCore.pyqtSignal()
	status = QtCore.pyqtSignal(str)

# www.pyshine.com
class AudioOutputWorker(QtCore.QRunnable):

	def __init__(self, audio_queue, CHUNK, *args, **kwargs):
		super(AudioOutputWorker, self).__init__()
		self.audio_queue = audio_queue
		self.running = True
		self.waiting = True
		self.CHUNK = CHUNK
		self.args = args
		self.kwargs = kwargs
		self.signals = AudioOutputWorkerSignals()
	
	def getAudio(self):
		while self.running:
			if not self.audio_queue.empty():
				self.waiting = False	
				CHUNK = self.CHUNK
				arr = np.frombuffer(self.audio_queue.get_nowait(), dtype=np.float32)
				sf.write("filepath.wav", arr, samplerate=24000)
				x,_ = librosa.load("filepath.wav", sr=24000)
				sf.write('tmp.wav', x, 24000)
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
				self.signals.status.emit("BOT Status:   SPEAKING  ... ")
				while(data):
					try:
						self.signals.progress.emit(data)
						# time.sleep(1)
						stream.write(data)
						data = wf.readframes(CHUNK)
					except:
						break
			else:
				if not self.waiting:
					self.signals.waiting.emit()
					self.waiting = True

	@pyqtSlot()
	def run(self):
		self.getAudio()
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
			print("result : " + result)
		except:
			self.signals.finished.emit(0)
		else:
			# self.signals.result.emit(result)
			self.signals.finished.emit(1)	


class VicunaSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal(int)
	error = QtCore.pyqtSignal(tuple)
	result = QtCore.pyqtSignal(str)
	progress = QtCore.pyqtSignal(str)
	start = QtCore.pyqtSignal(str)

# www.pyshine.com
class VicunaWorker(QtCore.QRunnable):

	def __init__(self, ttt_model, query, *args, **kwargs):
		super(VicunaWorker, self).__init__()
		self.ttt_model = ttt_model
		self.query = query
		self.args = args
		self.kwargs = kwargs
		self.signals = VicunaSignals()

	@pyqtSlot()
	def run(self):
		self.signals.start.emit("BOT Status:   Querying  ... ")
		# Retrieve args/kwargs here; and fire processing using them
		try:
			for outputs in self.ttt_model.run(self.query):
				self.signals.progress.emit(outputs)
		except:
			traceback.print_exc()
			exctype, value = sys.exc_info()[:2]
			# self.signals.error.emit((exctype, value, traceback.format_exc()))
			self.signals.finished.emit(0)
		else:
			# self.signals.result.emit(1)  # Return the result of the processing
			self.signals.finished.emit(1)
		# finally:
		# 	self.signals.finished.emit()  # Done



class TTSSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal()
	update_plot = QtCore.pyqtSignal(bytes)
	start = QtCore.pyqtSignal(str)
	waiting = QtCore.pyqtSignal()
	init = QtCore.pyqtSignal()

# www.pyshine.com
class TTSWorker(QtCore.QRunnable):

	def __init__(self, tts_model, txt_queue, running=True, *args, **kwargs):
		super(TTSWorker, self).__init__()
		self.tts_model = tts_model
		self.txt_queue = txt_queue
		self.running = running
		self.args = args
		self.kwargs = kwargs
		self.signals = TTSSignals()
		self.init = True
	
	def generate_audio(self, progress):
		return self.tts_model.run_ex(progress)
	
	@pyqtSlot()
	def run(self):
		self.signals.start.emit("BOT Status:   GENERATING  ... ")
		while self.running:
			try:
				if not self.txt_queue.empty():
					result = self.generate_audio(progress = self.txt_queue.get_nowait())
					print("done\n")
					self.signals.update_plot.emit(result.tobytes())
				else:
					# print("txt_queue is empty \n")
					# self.signals.waiting.emit()
					pass
			except:
				pass
			
			if self.init:
				self.signals.start.emit("BOT Status:   IDLE  ... ")
				self.signals.init.emit()
				self.init = False
		self.signals.finished.emit()
		

def main():
	params = {
	"CHUNK": 1024,
	"stt_model" : WhisperLargeV2(),
	"ttt_model" : FastChatModel(),
	"tts_model" : Bark(voice_preset = "v2/de_speaker_5"),
	"welcome_message" : "Hallo, Ich heiße Vicuna, wie kann ich dir helfen?",
	}
	app = QtWidgets.QApplication(sys.argv)

	# set stylesheet
	file = QFile(":/dark/stylesheet.qss")
	file.open(QFile.ReadOnly | QFile.Text)
	stream = QTextStream(file)
	app.setStyleSheet(stream.readAll())
	# code goes here
	mainWindow = PyShine_LIVE_PLOT_APP(params=params)
	mainWindow.showMaximized() 

	app.exec_()



main()