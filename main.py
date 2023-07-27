from Models import *

stt_model = PythonDummySTTModel()
ttt_model = FastChatModel()
tts_model = GTTSAPI()

app = App(stt_model, ttt_model, tts_model)

app.run()