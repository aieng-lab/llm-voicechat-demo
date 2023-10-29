# Welcome to PyShine
# This is part 16 of the PyQt5 learning series
# Based on parameters, the GUI will plot Video using OpenCV and Audio using Matplotlib in PyQt5
# We will use Qthreads to run the audio/Video streams

import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
import sounddevice as sd
from PyQt5.QtGui import QImage
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtMultimedia import QAudioDeviceInfo,QAudio,QCameraInfo
import time
import queue
import os
import wave, pyaudio, pdb
# import cv2,imutils
from PyQt5.QtWidgets import QFileDialog
import traceback
import librosa
import soundfile as sf
from Models import *
# For details visit pyshine.com




input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)

class MplCanvas(FigureCanvas):
	def __init__(self, parent=None, width=5, height=4, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = fig.add_subplot(111)
		self.axes.axis("off")
		super(MplCanvas, self).__init__(fig)
		fig.tight_layout()

class PyShine_LIVE_PLOT_APP(QtWidgets.QMainWindow):
	def __init__(self):
		print("Initializing ... \n")
		QtWidgets.QMainWindow.__init__(self)
		self.ui = uic.loadUi('main.ui',self)
		self.resize(888, 600)
		self.tmpfile = 'temp.wav'

		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("PyShine.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		self.setWindowIcon(icon)
		self.threadpool = QtCore.QThreadPool()	
		self.threadpool.setMaxThreadCount(2)
		self.stt_pool = QtCore.QThreadPool()
		self.ttt_pool = QtCore.QThreadPool()
		self.tts_pool = QtCore.QThreadPool()
		self.CHUNK = 1024
		self.q = queue.Queue(maxsize=self.CHUNK)
		self.devices_list= []
		for device in input_audio_deviceInfos:
			self.devices_list.append(device.deviceName())
		
		self.comboBox.addItems(self.devices_list)
		self.comboBox.currentIndexChanged['QString'].connect(self.update_now)
		self.comboBox.setCurrentIndex(0)
		
		self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
		self.ui.gridLayout_4.addWidget(self.canvas, 2, 1, 1, 1)
		self.reference_plot = None
		

		self.device = self.devices_list[0]
		self.window_length = 1000
		self.downsample = 1
		self.channels = [1]
		self.interval = 1
		
		
		device_info =  sd.query_devices(self.device, 'input')
		
		self.samplerate = device_info['default_samplerate']
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		sd.default.samplerate = self.samplerate
		
		self.plotdata =  np.zeros((length,len(self.channels)))
		self.timer = QtCore.QTimer()
		self.timer.setInterval(self.interval) #msec
		self.timer.timeout.connect(self.update_plot)
		self.timer.start()
		self.data=[0]
		self.lineEdit.textChanged['QString'].connect(self.update_window_length)
		self.lineEdit_2.textChanged['QString'].connect(self.update_sample_rate)
		self.lineEdit_3.textChanged['QString'].connect(self.update_down_sample)
		self.lineEdit_4.textChanged['QString'].connect(self.update_interval)
		self.speaking_allowed = False
		self.record_allowed = True
		self.query_allowed = False
		self.generating_allowed = False
		self.stopped = False
		# self.pushButton.clicked.connect(self.start_worker_1_mic)
		self.pushButton.clicked.connect(self.start_app)
		self.pushButton_2.clicked.connect(self.stop_worker)
		self.stt_model = WhisperTiny()
		self.ttt_model = FastChatModel()
		self.tts_model = SpeechT5()
		self.response=""
		self.query=""
		self.filepath=""
		print("Initialiazing is finished.\n")
		
	
	def start_app(self):
		self.stopped = False
		self.record_allowed = True
		self.start_worker_1_mic()
	
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
			self.lineEdit.setEnabled(False)
			self.lineEdit_2.setEnabled(False)
			self.lineEdit_3.setEnabled(False)
			self.lineEdit_4.setEnabled(False)
			self.comboBox.setEnabled(False)
			self.pushButton.setEnabled(False)
			# self.canvas.axes.clear()
			self.mic_worker = STTWorker(self.record_speech)
			self.mic_worker.signals.finished.connect(self.start_worker_2_vicuna)
			# self.mic_worker.signals.result.connect(self.start_worker_2_vicuna)
			# self.threadpool.start(self.mic_worker)
			self.stt_pool.start(self.mic_worker)	

			# self.reference_plot = None
			self.timer.setInterval(self.interval) #msec
		else:
			print("Recording isn't allowed\n")
		print("start_worker_1 is finished\n")
		
	
	def query_vicuna(self, query:str):
		curr_time = time.time()
		print("Querying Vicuna ...\n")
		# self.response = self.ttt_model.generate_start(query)
		self.response = self.ttt_model.run(query)
		print(self.response)
		self.query_allowed = False
		self.generating_allowed = True
		print("\nText_to_text took {:.2f}s to finish\n".format(time.time()-curr_time))
		return self.response
	
	def start_worker_2_vicuna(self, emit_sig:int):
		
		if emit_sig==1 and self.query_allowed and self.stopped is False:
			vicuna_worker = VicunaWorker(self.query_vicuna, query=self.query)
			vicuna_worker.signals.finished.connect(self.start_worker_3_tts)
			# self.threadpool.start(vicuna_worker)
			self.ttt_pool.start(vicuna_worker)
		else:
			print("The mic wasn't able to record or Vicuna isn't available!\n")
		
		
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
			tts_worker.signals.finished.connect(self.start_worker_4_speaker)

			# self.threadpool.start(tts_worker)
			self.tts_pool.start(tts_worker)
		else:
			print("Vicuna wasn't able to process your query!\n")
	
		

	def getAudio(self, filepath):
		curr_time=time.time()
		if self.speaking_allowed is True:
			# QtWidgets.QApplication.processEvents()	
			CHUNK = self.CHUNK
			x,_ = librosa.load(filepath, sr=16000)
			sf.write('tmp.wav', x, 16000)
			wf = wave.open('tmp.wav', 'rb')
			# wf = wave.open(self.tmpfile, 'rb')
			p = pyaudio.PyAudio()
			stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
							channels=wf.getnchannels(),
							rate=wf.getframerate(),
							output=True,
							frames_per_buffer=CHUNK)
			self.samplerate = wf.getframerate()
			sd.default.samplerate = self.samplerate
			while(self.speaking_allowed is True):
				
				# QtWidgets.QApplication.processEvents()    
				data = wf.readframes(CHUNK)
				audio_as_np_int16 = np.frombuffer(data, dtype=np.int16)
				audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
				# Normalise float32 array                                                   
				max_int16 = 2**15
				audio_normalised = audio_as_np_float32 / max_int16
				try:
					self.q.put_nowait(audio_normalised)
					stream.write(data)
				
					# if self.speaking_allowed is False:
					# 	break
				except:
					self.speaking_allowed = False
					self.q.queue.clear()

			
		self.pushButton.setEnabled(True)
		self.lineEdit.setEnabled(True)
		self.lineEdit_2.setEnabled(True)
		self.lineEdit_3.setEnabled(True)
		self.lineEdit_4.setEnabled(True)
		self.comboBox.setEnabled(True)	
		self.speaking_allowed=False
		self.record_allowed = True
		print("Playing and Plotting took {:.2f}s to finish\n".format(time.time()-curr_time))


	def start_worker_4_speaker(self, emit_sig):
		if emit_sig==1 and self.speaking_allowed and self.stopped is False:
			speaker_worker = Worker(self.getAudio, filepath=self.filepath)
			speaker_worker.signals.finished.connect(self.start_worker_1_mic)
			self.threadpool.start(speaker_worker)
		else:
			print("The model couldn't generate a speech from the text response\n")

	def stop_worker(self):
		self.stopped = True
		self.record_allowed = False
		self.query_allowed = False
		self.generating_allowed = False
		self.speaking_allowed = False
		print("You have stopped all threads!\n")
		self.pushButton.setEnabled(True)
		self.lineEdit.setEnabled(True)
		self.lineEdit_2.setEnabled(True)
		self.lineEdit_3.setEnabled(True)
		self.lineEdit_4.setEnabled(True)
		self.comboBox.setEnabled(True)	
		with self.q.mutex:
			self.q.queue.clear()
		# self.threadpool.close()
		
	def update_now(self,value):
		self.device = self.devices_list.index(value)
		

	def update_window_length(self,value):
		self.window_length = int(value)
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		self.plotdata =  np.zeros((length,len(self.channels)))
		

	def update_sample_rate(self,value):
		self.samplerate = int(value)
		sd.default.samplerate = self.samplerate
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		self.plotdata =  np.zeros((length,len(self.channels)))
		
	
	def update_down_sample(self,value):
		self.downsample = int(value)
		length  = int(self.window_length*self.samplerate/(1000*self.downsample))
		self.plotdata =  np.zeros((length,len(self.channels)))
	

	def update_interval(self,value):
		self.interval = int(value)
		
		

	def update_plot(self):
		try:
			
			threads_num = self.threadpool.activeThreadCount() + self.stt_pool.activeThreadCount() + self.ttt_pool.activeThreadCount() + self.tts_pool.activeThreadCount()
			print('ACTIVE THREADS:',threads_num,end=" \r")
			while  self.speaking_allowed is True:
				# QtWidgets.QApplication.processEvents()	
				try: 
					self.data = self.q.get_nowait()
					
					
				except queue.Empty:
					break
				
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
		try:
			result = self.function(*self.args, **self.kwargs)
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
	start = QtCore.pyqtSignal()

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
		self.signals.start.emit()
		# Retrieve args/kwargs here; and fire processing using them
		try:
			result = self.function(*self.args, **self.kwargs)
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
	finished = QtCore.pyqtSignal(int)
	# result = QtCore.pyqtSignal(str)

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
		try:
			result = self.function(*self.args, **self.kwargs)
			# self.signals.result.emit(result)
		except:
			self.signals.finished.emit(0)
		else:
			self.signals.finished.emit(1)
		

app = QtWidgets.QApplication(sys.argv)
mainWindow = PyShine_LIVE_PLOT_APP()
mainWindow.show()
sys.exit(app.exec_())

