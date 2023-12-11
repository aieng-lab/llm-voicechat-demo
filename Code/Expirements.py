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
import debugpy


class MplCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black')
		self.axes = fig.add_subplot(111)
		self.axes.axis("off")
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()

class VicunaSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal()
	error = QtCore.pyqtSignal(str)
	progress = QtCore.pyqtSignal(str)
	start = QtCore.pyqtSignal(str)
	end_of_generation = QtCore.pyqtSignal()

class VicunaWorker(QtCore.QRunnable):

	def __init__(self, query, welcome_message, *args, **kwargs):
		super(VicunaWorker, self).__init__()
		self.ttt_model = FastChatModel()
		self.welcome_message = welcome_message
		self.query = query
		self.waiting = True
		self.running = True
		self.args = args
		self.kwargs = kwargs
		self.signals = VicunaSignals()
		self.init = True

	def run(self):
		# debugpy.debug_this_thread()
		print("\nVicunaWorker\n")
		while self.running:
			
			if self.init:
				self.ttt_model.conv.append_message(self.ttt_model.conv.roles[1], self.welcome_message)
				self.init = False

			if not self.waiting:
				self.signals.start.emit("BOT Status:   Querying  ... ")
				# self.function(*self.args, **self.kwargs)
				for outputs in self.ttt_model.run(self.query):
					if outputs == "END":
						self.signals.end_of_generation.emit()
					else:
						print(outputs)
						self.signals.progress.emit(outputs)
				self.waiting = True


		self.signals.finished.emit()
	
	def wake(self, query):
		self.query = query
		self.waiting = False


class TTSSignals(QtCore.QObject):
	finished = QtCore.pyqtSignal()
	update_plot = QtCore.pyqtSignal(object)
	start = QtCore.pyqtSignal(str)
	init = QtCore.pyqtSignal()
	end_of_generation = QtCore.pyqtSignal()


class TTSWorker(QtCore.QRunnable):

	def __init__(self, tts_model=None, txt_queue=None, running=True, *args, **kwargs):
		super(TTSWorker, self).__init__()
		# self.tts_model = tts_model
		self.tts_model = Bark(voice_preset = "v2/de_speaker_5")
		self.txt_queue = txt_queue
		self.running = running
		self.args = args
		self.kwargs = kwargs
		self.signals = TTSSignals()
		self.init = True
	
	def generate_audio(self, progress):
		return self.tts_model.run_ex(progress)
	
	def run(self):
		# debugpy.debug_this_thread()
		print("TTSWorker\n")
		while self.running:
			if not self.txt_queue.empty():
				item = self.txt_queue.get()
				if item == "END":
					self.signals.end_of_generation.emit()
					time.sleep(10)
					# break
				else:
					print("########\n")
					print(item, "\n")
					print("########\n")
					self.signals.start.emit("BOT Status:   GENERATING  ... ")
					result = self.generate_audio(item)
					# chunck = 1024 * 4
					# result = np.ones((chunck, 1), dtype=np.float32)
					print("Audio generated for item\n")
					self.signals.update_plot.emit(result)
					self.signals.finished.emit()
			
				if self.init:
					self.signals.start.emit("BOT Status:   IDLE  ... ")
					self.signals.init.emit()
					self.init = False
			else:
				continue
	
	def stop_running(self):
		self.running=False


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
		# debugpy.debug_this_thread()
		# print("Worker\n")
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
		self.waiting = False
		self.sleeping = True
		self.CHUNK = CHUNK
		self.args = args
		self.kwargs = kwargs
		self.signals = AudioOutputWorkerSignals()

	def run(self):
		# debugpy.debug_this_thread()
		print("AudioOutputWorker\n")
		while self.active:
			if not self.sleeping:
				if not self.audio_queue.empty():
					self.signals.status.emit("BOT Status:   SPEAKING  ... ")	
					self.function(*self.args, **self.kwargs)
				else:
					if  self.waiting:
						print("######### I'm waiting #########\n")
						self.signals.waiting.emit()
						self.waiting = False
						time.sleep(10)
		self.signals.finished.emit()

	def stop_running(self):
		self.active=False

	def wait(self):
		self.waiting = True
	
	def wake(self):
		self.sleeping = False

class STTSignals(QtCore.QObject):
	start = QtCore.pyqtSignal(str)
	finished = QtCore.pyqtSignal()
	result = QtCore.pyqtSignal(str)

# www.pyshine.com
class STTWorker(QtCore.QRunnable):

	def __init__(self, *args, **kwargs):
		super(STTWorker, self).__init__()
		self.stt_model = WhisperLargeV2()
		self.args = args
		self.kwargs = kwargs
		self.signals = STTSignals()


	def run(self):
		# debugpy.debug_this_thread()
		print("STTWorker\n")
		with sr.Microphone() as source:
			print("Microphone is started ... \n")
			self.stt_model.r.adjust_for_ambient_noise(source)
			print("Microphone is listening ... \n")
			self.signals.start.emit("BOT Status:   lISTENING  ... ")
			audio = self.stt_model.r.listen(source)
			print("Microphone is recognizing ... \n")
			self.signals.start.emit("BOT Status:   RECOGNIZING  ... ")
			audio_data = audio.get_flac_data()
			text=self.stt_model.p(audio_data, max_new_tokens=500, generate_kwargs={"language": "german"})["text"]
		print("result : " + text)
		self.signals.result.emit(text)
		self.signals.finished.emit()	
		


class PyShine_LIVE_PLOT_APP(QtWidgets.QMainWindow):
	def __init__(self, params):
		print("Initializing ... \n")
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('main.ui',self)
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
		self.timer.setInterval(30) #msec
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
		

		self.response=""
		self.query=""
		self.filepaths=[]

		self.welcome_message = self.params["welcome_message"]
		self.txt_queue.put_nowait(self.welcome_message)
		self.txt_queue.put_nowait("END")



		self.status_worker = None
		
		
		self.tts_worker = TTSWorker(txt_queue=self.txt_queue)
		self.tts_worker.signals.update_plot.connect(self.update_audio_queue)
		self.tts_worker.signals.start.connect(self.update_status)
		self.tts_worker.signals.end_of_generation.connect(self.freeze_speaker)
		self.tts_worker.signals.init.connect(self.start)
		self.tts_pool.start(self.tts_worker)
		
		
		self.mic_worker = STTWorker()
		
		self.vicuna_worker = VicunaWorker(query=self.query, welcome_message = self.welcome_message)
		self.vicuna_worker.signals.start.connect(self.update_status)
		self.vicuna_worker.signals.progress.connect(self.update_txt_queue)
		self.ttt_pool.start(self.vicuna_worker)

	
		self.speaker_worker = AudioOutputWorker(function=self.getAudio, audio_queue=self.audio_queue, CHUNK=self.CHUNK)
		self.speaker_worker.signals.waiting.connect(self.start_mic_worker)
		self.speaker_worker.signals.status.connect(self.update_status)
		self.threadpool.start(self.speaker_worker)


		self.pushButton_2.clicked.connect(self.stop_worker)
		self.pushButton.clicked.connect(self.start_worker_4_speaker)
	
	def freeze_speaker(self):
		self.speaker_worker.wait()

	def wake_vicuna(self, query):
		self.vicuna_worker.wake(query)

	def start(self):
		print("Initialiazing is finished.\n")
		self.pushButton_2.setEnabled(True)
		self.pushButton.setEnabled(True)
		


	def display_status(self, text):
		self.label_6.setText(text)

	def update_status(self, text):
		self.status_worker = Worker(self.display_status, text)
		self.threadpool.start(self.status_worker)
		

	def start_mic_worker(self):
		print("start_mic_worker is reached\n")
		if not self.stopped:
			print("Recording is allowed\n")
			self.mic_worker.signals.start.connect(self.update_status)
			self.mic_worker.signals.result.connect(self.start_worker_2_vicuna)
			self.stt_pool.start(self.mic_worker)	
		else:
			print("Recording isn't allowed\n")
		
	
	def query_vicuna(self):
		curr_time = time.time()
		print("Querying Vicuna ...\n")
		# self.response =["Hallo, da ist Salim. Ich komme aus Syrien und bin 28 Jahre alt.",
		# 		  "Ich mache einen Master in Informatik an der Universität Passau.",
		# 		  "Ich habe im Moment einen Minijob an der Uni. Dabei muss ich ein VoiceBot entwickeln.",
		# 		  "END"]
		for item in self.response:
			# print(item, "\n")
			self.txt_queue.put(item)
		print("Response is added to txt_queue\n")
		# for outputs in self.tts_model.run(self.query):
		# 	self.response+= " " + outputs
		# 	print("\n\n", outputs, "\n\n")
		# 	self.txt_queue.put(outputs)
		self.query_allowed = False
		self.generating_allowed = True
		print("\nText_to_text took {:.2f}s to finish\n".format(time.time()-curr_time))
		return self.response
	
	def start_worker_2_vicuna(self, query):
		if self.query_allowed and not self.stopped:
			print("\nstart_worker_2_vicuna is reached\n")
			self.query = query
			self.wake_vicuna(query)

		elif not self.query_allowed:
			print("Vicuna isn't available!\n")
		elif self.stopped:
			print("You have stopped the app!\n")
		else:
			print("Unknown problem while querying! \n")
	
	def update_txt_queue(self, txt):
		self.txt_queue.put(txt)

	def getAudio(self):
		if not self.audio_queue.empty():
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
			while(data):
				try:
					audio_as_np_int16 = np.frombuffer(data, dtype=np.int16)
					audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
					max_int16 = 2**15
					audio_normalised = audio_as_np_float32 / max_int16
					self.plot_queue.put_nowait(audio_normalised)
					stream.write(data)
					data = wf.readframes(CHUNK)
				except:
					break
	
	def update_audio_queue(self, data):
		self.audio_queue.put_nowait(data)
	
	def generate_plotdata(self):
		if not self.plot_queue.empty():
			audio_normalised = self.plot_queue.get_nowait()
			return audio_normalised
		else:
			return np.zeros((self.CHUNK,len(self.channels)))


	def start_worker_4_speaker(self):
		if self.plotting and not self.stopped:
			self.speaker_worker.wake()

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
		self.speaker_worker.stop_running()
		self.tts_worker.stop_running()
		print("You have stopped all threads!\n")
		with self.plot_queue.mutex:
			self.plot_queue.queue.clear()
		
		with self.audio_queue.mutex:
			self.audio_queue.queue.clear()
		
		with self.txt_queue.mutex:
			self.txt_queue.queue.clear()
		
			

	def update_plot(self):
		try:
			if self.timer.isActive():
				threads_num = self.threadpool.activeThreadCount() + self.stt_pool.activeThreadCount() + self.ttt_pool.activeThreadCount() + self.tts_pool.activeThreadCount() + 1
			else:
				threads_num = self.threadpool.activeThreadCount() + self.stt_pool.activeThreadCount() + self.ttt_pool.activeThreadCount() + self.tts_pool.activeThreadCount()
			# print('ACTIVE THREADS:',threads_num,end=" \r")
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





		

def main():
	params = {
	"CHUNK": 2048,
	# "stt_model" : WhisperLargeV2(),
	# "ttt_model" : FastChatModel(),
	# "tts_model" : Bark(voice_preset = "v2/de_speaker_5"),
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
