import gradio as gr
import random
import time
from gtts import gTTS
from transformers import pipeline
from Code.Models import  *
import threading

class ChatBotGUI():
    def __init__(self):
        self.stt_model = None
        self.ttt_model = None
        self.tts_model = None
        self.lock = threading.Lock()

    def set_stt_model(self, name:str):
        self.lock.acquire()
        if self.stt_model is not None:
            del self.stt_model
        self.stt_model = self.get_instance(name)
        self.lock.release()
    
    def set_ttt_model(self, name:str):
        self.lock.acquire()
        if self.ttt_model is not None:
            del self.ttt_model
            torch.cuda.empty_cache()
        self.ttt_model = self.get_instance(name)
        self.lock.release()
        
    def set_tts_model(self, name):
        self.lock.acquire()
        if self.tts_model is not None:
            del self.tts_model
        self.tts_model = self.get_instance(name)
        self.lock.release()
        
    def get_instance(self, name, args=None):
        Klass = globals()[name]
        if args is not None:
            instance = Klass(args)
        else:
            instance =Klass()
        return instance
    
    def add_text_msg(self, history, text):
        self.lock.acquire()
        history = history + [(text, None)]
        self.lock.release()
        return history, gr.update(value="", interactive=False)
        
        
    def bot(self, history):
        self.lock.acquire()
        if self.ttt_model is None:
            history[-1][1] = "You haven't selected a Text-to-Text model yet."
        else:
            history[-1][1] = self.ttt_model.run(history[-1][0])
        self.lock.release()
        return history
    
    def add_voice_msg(self, history):
        self.lock.acquire()
        if self.stt_model is None:
            history = history + [("You haven't selected a Speech-to-Text model yet.", None)]
        else:
            text = self.stt_model.run()
            history = history + [(text, None)]
        self.lock.release()
        return history
    
    def start_speaker(self, history, lang='en', slow=False):
        self.lock.acquire()
        if self.tts_model is None:
            return "select_text_to_speech_model.wav"
        self.lock.release()
        return self.tts_model.run(text=history[-1][1], lang=lang, slow=slow) 
    
    def start_waveform(self, history, lang='en', slow=False):
        self.lock.acquire()
        if self.tts_model is None:
            return "select_text_to_speech_model.wav"
        self.lock.release()
        return gr.make_waveform(self.tts_model.run(text=history[-1][1], lang=lang, slow=slow), animate=True, bars_color="#0000ff", bar_width=0.4)
    
    def clear_history(self):
        self.lock.acquire()
        del self.stt_model
        del self.ttt_model
        del self.tts_model
        
        self.stt_model = None
        self.ttt_model = None
        self.tts_model = None
        torch.cuda.empty_cache()
        self.lock.release()

    def build_GUI(self):
        
        with gr.Blocks() as demo:
            state = gr.State(value="")
            with gr.Row():
                stt_list = gr.Dropdown(["PythonDummySTTModel", "WhisperModel", "WhisperHuggingFace"], 
                                       label="Speech-to-Text")
                
                ttt_list = gr.Dropdown(["FastChatModel", "DialoGPT", "SomeOtherModel"], 
                                       label="Text-to-Text")
                
                tts_list = gr.Dropdown(["GTTSAPI", "Amazon", "Apple"], 
                                       label="Text-to-Speech")
                
                list_choice = gr.Textbox(visible=False)
                
            chatbot = gr.Chatbot([], elem_id="chatbot", height=700)
            speaker = gr.Audio(autoplay=False, label="Bot Speaker", elem_id="chatbox_voice", visible=False)
            speaker_video = gr.Video(autoplay=True, label="BOT Voice", height=300, width=2000)
        
            with gr.Row():
                with gr.Column():
                    txt = gr.Textbox( label="Text Message", 
                                     placeholder="Enter text and press enter",
                                     interactive=True, 
                                     container=False)
                
                with gr.Column():
                    # audio_message = gr.Audio(source="microphone", type="filepath", label="Voice Message", streaming=True)
                    audio_message = gr.Button(value="Record")
        
            clear = gr.ClearButton()
            
            stt_list.change(self.set_stt_model, stt_list, list_choice, queue=False)
            ttt_list.change(self.set_ttt_model, ttt_list, list_choice, queue=False)
            tts_list.change(self.set_tts_model, tts_list, list_choice, queue=False)

            txt_msg = txt.submit(self.add_text_msg, [chatbot, txt], [chatbot, txt], queue=False).then(
                self.bot, chatbot, chatbot
            )
            
            # txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False).then(start_speaker, [chatbot], [speaker], queue=False)

            txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False).then(self.start_waveform, [chatbot], [speaker_video], queue=False)
            
            # voice_msg = audio_message.stream(fn=add_voice_msg, inputs=[audio_message, chatbot], outputs=[chatbot], show_progress=False, queue=False)
            
            # voice_msg = audio_message.click(
            #     fn=add_voice_msg, inputs=[audio_message, chatbot], outputs=[chatbot], show_progress='full', queue=False).then(
            #     bot, chatbot, chatbot).then(
            #     start_speaker, [chatbot], [speaker], queue=False)
            
            voice_msg = audio_message.click(
                fn=self.add_voice_msg, inputs=[chatbot], outputs=[chatbot], show_progress='full', queue=False).then(
                self.bot, chatbot, chatbot, queue=False).then(
                self.start_waveform, [chatbot], [speaker_video], queue=False)
            
            outputs = [state, stt_list, ttt_list, tts_list, list_choice, chatbot, speaker, speaker_video, txt, audio_message]
            # clear.click(lambda: [self.clear_history()],outputs=[chatbot, state, speaker, txt, audio_message, speaker_video])
            clear.click(lambda: [None] * len(outputs), outputs=outputs, queue=False).then(
                self.clear_history, None, None, queue=False).then(
                lambda: [gr.update()] * len(outputs), None, outputs=outputs, queue=False)
            
        return demo


    def start_demo(self, share=False):
        demo = self.build_GUI()
        demo.queue()
        demo.launch(share=share)