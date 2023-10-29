import torch
import speech_recognition as sr
# from langchain.llms import OpenAI
import gtts
from abc import ABC, abstractmethod
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, Conversation, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import argparse
from fastchat.conversation import conv_templates, SeparatorStyle
from datasets import load_dataset
import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech

# OPENAI_API_KEY="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"





    
class STTStrategy(ABC):
    """Interface for Speech-to-Text Models.
    ----------
    Functions:
        run:  Convert speech input to text.
                parameters: None
                Return: str
    ----------
    """
    @abstractmethod
    def run(self) -> str:
        pass

class TTTStrategy(ABC):
    """Interface for Text-to-Text Models.
    ----------
    Functions:
        run:  Reply to the input text with text.
                parameters: text:str, ex: Statement, question, etc.
                Return: str
    ----------
    """
    @abstractmethod
    def run(self, text:str) -> str:
        pass


class TTSStrategy(ABC):
    """Interface for Text-to-Speech Models.
    ----------
    Functions:
        run:  Convert text input to audio object, save it into a .mp3 file then play it.
                parameters: text:str
                Return: None.
    ----------
    """
    @abstractmethod
    def run(self, text:str):
        pass

class App():
    """Our Speech-to-Speech application class with no GUI.
    ----------
    Attributes:
        stt_model:STTStrategy 'Speech-to-Text Model'.

        ttt_model:TTTStrategy 'Text-to-Text Model'.

        tts_model:TTSStrategy 'Text-to-Speech Model'.

    ----------
    Functions:
        run_STT: call the run() function of the stt_model and print the output.

        run_TTT: call the run(str) function of the ttt_model and return the output.
    ----------
    """
    def __init__(self, 
                 stt_model:STTStrategy,
                 ttt_model:TTTStrategy,
                 tts_model:TTSStrategy):
        """
        ----------
        Parameters:
            stt_model:STTStrategy 'Speech-to-Text Model'.

            ttt_model:TTTStrategy 'Text-to-Text Model'.

            tts_model:TTSStrategy 'Text-to-Speech Model'.

        ----------
        Return: None
        ----------
        """
        self.stt_model=stt_model
        self.ttt_model=ttt_model
        self.tts_model=tts_model
    
    def run_STT(self):
        """ Call the run() function of the stt_model and print the output.
        ----------
        Parameters: None
        ----------
        Return:
            text:str, the converted text from a speech.  
        ----------
        """
        text = self.stt_model.run()
        return text
    def run_TTT(self, text:str):
        """ Call the run(str) function of the ttt_model and return the output.
        ----------
        Parameters:
            text:str.

        ----------
        Return: 
            reply:str.
        ----------
        """
        reply = self.ttt_model.run(text)
        return reply
    
    def run_TTS(self, text:str):
        """ Call the run(str) function of the tts_model.
        ----------
        Parameters: 
            text:str.
        ----------
        Return: None
        ----------
        """
        self.tts_model.run(text)

    def run(self):
        """Run all of the applications conponents.
        ----------
        Parameters: None
        ----------
        Return: None
        ----------
        """
        live = True
        while(live):
            inp = self.run_STT()
            if inp.lower() in ["bye", "good bye"]:
                live = False
            else:
                self.run_TTS(self.run_TTT(inp))



class PythonDummySTTModel(STTStrategy):
    """A speech to text strategy using google api for speech recognition. Subclass of STTStrategy.
    ----------
    Attributes: 
        r: Recognizer object from speech_recognition module.

    ----------
    Functions:
        run_file: Read an audio file and convert its content to text.

        run: Record a speech using microphone then convert and return it as text.
    ----------
    """
    def __init__(self):
        """
        ----------
        Parameters: None

        ----------
        Return: None
        ----------
        """
        super().__init__()
        self.r = sr.Recognizer()
	
    def run_file(self, path):
        """ Read an audio file and convert its content to text.
        ----------
        Parameters: path:str the directory of the audio file.
      
        ----------
        Return: 
            text:str, the converted text from the recorded speech
        ----------
        """
        with sr.AudioFile(path) as source:
            audio_listened = self.r.listen(source)
            text = self.r.recognize_google(audio_listened)
        return text
		
    def run(self):
        """ Record a speech using the microphone then convert and return it as text.
        ----------
        Parameters: None
      
        ----------
        Return: 
            text:str, the converted text from the recorded speech
        ----------
        """
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source) 
            audio_data = self.r.listen(source)
            print("Recognizing...")
            text = self.r.recognize_google(audio_data, language="en-US", show_all=True)
        return text["alternative"][0]["transcript"]

        
class WhisperModel(STTStrategy):
    """A speech to text abstract strategy using Whisper model from HuggingFace hub. Subclass of STTStrategy.
    ----------
    Attributes: 
        r: Recognizer object from speech_recognition module.
        
        p: refers to pipeline from transformers module.

    ----------
    Functions:
        run: record a speech using the microphone then convert and return it as text.
    ----------
    """    
    def __init__(self):
        """
        ----------
        Parameters: None

        ----------
        Return: None
        ----------
        """
        super().__init__()
        self.r = sr.Recognizer()
        self.p = None


    def run(self)-> str:
        """ Record a speech using the microphone then convert and return it as text.
        ----------
        Parameters: None
      
        ----------
        Return: 
            text:str, the converted text from the recorded speech
        ----------
        """
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source)
            audio = self.r.listen(source)
            text=self.p(audio.get_flac_data(), max_new_tokens=500)["text"]
        return text
    
    def run_gradio(self, audio):
        """ Convert audio data recorded with a gradio microphone to text.
        ----------
        Parameters:
            audio: byte string representing the contents of a FLAC file containing the audio represented by the AudioData instance.
        ----------
        Return: 
           The converted text from the recorded speech
        ----------

        """
        return self.p(audio, max_new_tokens=500)["text"]

class WhisperTiny(WhisperModel):
    """A speech to text strategy using Whisper model from HuggingFace hub. Subclass of WhisperModel.
    ----------
    Attributes: 
        p: pipeline from transformers module with automatic-speech-recognition task and whisper-tiny model.

    ----------
    Functions:
        run: record a speech using the microphone then convert and return it as text.
    ----------
    """    
    def __init__(self):
        """
        ----------
        Parameters: None

        ----------
        Return: None
        ----------
        """
        super().__init__()
        self.p = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    def run(self)-> str:
        """ Record a speech using the microphone then convert and return it as text.
        ----------
        Parameters: None
      
        ----------
        Return: 
            text:str, the converted text from the recorded speech
        ----------
        """
        return super().run()
    
    def run_gradio(self, audio):
        """ Convert audio data recorded with a gradio microphone to text.
        ----------
        Parameters:
            audio: byte string representing the contents of a FLAC file containing the audio represented by the AudioData instance.
        ----------
        Return: 
           The converted text from the recorded speech
        ----------

        """
        return super().run_gradio(audio)


class WhisperMedium(WhisperModel):
    """A speech to text strategy using Whisper model from HuggingFace hub. Subclass of WhisperModel.
    ----------
    Attributes: 
        p: pipeline from transformers module with automatic-speech-recognition task and whisper-medium model.

    ----------
    Functions:
        run: record a speech using the microphone then convert and return it as text.
    ----------
    """    
    def __init__(self):
        """
        ----------
        Parameters: None

        ----------
        Return: None
        ----------
        """
        super().__init__()
        self.p = pipeline("automatic-speech-recognition", model="openai/whisper-medium")

    def run(self)-> str:
        """ Record a speech using the microphone then convert and return it as text.
        ----------
        Parameters: None
      
        ----------
        Return: 
            text:str, the converted text from the recorded speech
        ----------
        """
        return super().run()
    
    def run_gradio(self, audio):
        """ Convert audio data recorded with a gradio microphone to text.
        ----------
        Parameters:
            audio: byte string representing the contents of a FLAC file containing the audio represented by the AudioData instance.
        ----------
        Return: 
           The converted text from the recorded speech
        ----------

        """
        return super().run_gradio(audio)
    
class WhisperLarge(WhisperModel):
    """A speech to text strategy using Whisper model from HuggingFace hub. Subclass of WhisperModel.
    ----------
    Attributes: 
        p: pipeline from transformers module with automatic-speech-recognition task and whisper-large model.

    ----------
    Functions:
        run: record a speech using the microphone then convert and return it as text.
    ----------
    """    
    def __init__(self):
        """
        ----------
        Parameters: None

        ----------
        Return: None
        ----------
        """
        super().__init__()    
        self.p = pipeline("automatic-speech-recognition", model="openai/whisper-large")

    def run(self)-> str:
        """ Record a speech using the microphone then convert and return it as text.
        ----------
        Parameters: None
      
        ----------
        Return: 
            text:str, the converted text from the recorded speech
        ----------
        """
        return super().run()

    def run_gradio(self, audio):
        """ Convert audio data recorded with a gradio microphone to text.
        ----------
        Parameters:
            audio: byte string representing the contents of a FLAC file containing the audio represented by the AudioData instance.
        ----------
        Return: 
           The converted text from the recorded speech
        ----------

        """
        return super().run_gradio(audio)
    
class WhisperLargeV2(WhisperModel):
    """A speech to text strategy using Whisper model from HuggingFace hub. Subclass of WhisperModel.
    ----------
    Attributes: 
        p: pipeline from transformers module with automatic-speech-recognition task and whisper-large-v2 model.

    ----------
    Functions:
        run: record a speech using the microphone then convert and return it as text.
    ----------
    """    
    def __init__(self):
        """
        ----------
        Parameters: None

        ----------
        Return: None
        ----------
        """
        super().__init__()
        self.p = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

    def run(self)-> str:
        """ Record a speech using the microphone then convert and return it as text.
        ----------
        Parameters: None
      
        ----------
        Return: 
            text:str, the converted text from the recorded speech
        ----------
        """
        return super().run()          

    def run_gradio(self, audio):
        """ Convert audio data recorded with a gradio microphone to text.
        ----------
        Parameters:
            audio: byte string representing the contents of a FLAC file containing the audio represented by the AudioData instance.
        ----------
        Return: 
           The converted text from the recorded speech
        ----------

        """
        return super().run_gradio(audio)



# class OpenAIModel(TTTStrategy):
#     """ A text to text strategy using OpenAI api. Subclass of TTTStrategy
#     ----------
#     Attributes:
#         llm: OpenAI model.

#     ----------
#     Functions:
#         run: generate text using OpenAI based on the input text.
#     ----------
#     """
#     def __init__(self):
#         """
#         ----------
#         Parameters: None

#         ----------
#         Return: None
#         ----------
#         """
#         super().__init__()
#         self.llm = OpenAI(openai_api_key = OPENAI_API_KEY)
	
#     def run(self, text:str)->str:
#         """Generate text using OpenAI based on the input text.
#         ----------
#         Parameters:
#                 text:str, input text.
#         ----------
#         Return:
#             reply:str, text generated by OpenAI.
#         ----------
#         """
#         reply = self.llm.predict(text).strip()
#         print("AI : " + reply)
#         return reply

    
class FastChatModel(TTTStrategy):
    """ A text to text strategy based on FastChat project using Vicuna model. Subclass of TTTStrategy
    ----------
    Attributes:
        args:dict:
            model_name: the name of the used vicuna model.
            device:str, 'cpu', 'cuda', 'mps'
            num_gpus:str, number of GPUs to be used by the model.
            load_8bit:bool,  whether to use the 8bit compression for low memory.
            conv_template:str, specify the template of the conversation.
            temperature:float.
            max_new_token:int, max amount of generated characters.
            debug:bool.
        
        model: the model used to generate text.

        tokenizer: the tokenizer used to encode inputs and decode outputs of the model.

        conv: conversation template.

    ----------
    Functions:
        load_model: load both the model and the tokenizer from transformers module corresponding to model_name.

        generate_stream:

        generate-start:

        run:

    ----------
    """
    def __init__(self):
        """
        ----------
        Parameters: None
        ----------
        Return: None
        ----------
        """
        super().__init__()            
        self.args = dict(model_name='lmsys/vicuna-7b-v1.3',
                        device='cuda',
                        num_gpus='1',
                        load_8bit=True,
                        conv_template='vicuna_v1.1',
                        temperature=0.7,
                        max_new_tokens=512,
                        debug=False)
        self.args = argparse.Namespace(**self.args)
        self.model, self.tokenizer = self.load_model(self.args.model_name, self.args.device, self.args.num_gpus, self.args.load_8bit)
        # Chat
        self.conv = conv_templates[self.args.conv_template].copy()

    
    def load_model(self, model_name, device, num_gpus=1, load_8bit=True):
        if device == "cpu":
            kwargs = {}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if load_8bit:
                if num_gpus != "auto" and int(num_gpus) != 1:
                    print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
                kwargs.update({"load_in_8bit": True, "device_map": "auto"})
            else:
                if num_gpus == "auto":
                    kwargs["device_map"] = "auto"
                else:
                    num_gpus = int(num_gpus)
                    if num_gpus != 1:
                        kwargs.update({
                            "device_map": "auto",
                            "max_memory": {i: "13GiB" for i in range(num_gpus)},
                        })
        else:
            raise ValueError(f"Invalid device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

        # calling model.cuda() mess up weights if loading 8-bit weights
        if device == "cuda" and num_gpus == 1 and not load_8bit:
            model.to("cuda")
        elif device == "mps":
            model.to("mps")

        return model, tokenizer
    
    @torch.inference_mode()
    def generate_stream(self, tokenizer, model, params, device,
                        context_len=2048, stream_interval=2):
        """Adapted from fastchat/serve/model_worker.py::generate_stream"""

        prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        outputs = []
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=device)
                out = model(input_ids=torch.as_tensor([[token]], device=device),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]

            if device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break

            # if i == max_new_tokens - 1 or stopped:
            #     output = tokenizer.decode(output_ids, skip_special_tokens=True)
            #     pos = output.rfind(stop_str, l_prompt)
            #     if pos != -1:
            #         output = output[:pos]
            #         stopped = True
            #     outputs.append(output)
            # if stopped:
            #     break

        del past_key_values
        # return outputs

    def generate_start(self, inp):
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        params = {
        "model": self.args.model_name,
        "prompt": prompt,
        "temperature": self.args.temperature,
        "max_new_tokens": self.args.max_new_tokens,
        "stop": self.conv.sep if self.conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE else self.conv.sep2,
        }

        print(f"{self.conv.roles[1]}: ", end="", flush=True)
        pre = 0
        #new
        outputs = self.generate_stream(self.tokenizer, self.model, params, self.args.device)
        for output in outputs:
            output = output[len(prompt) + 1:].strip()
            output = output.split(" ")
            now = len(output)
            if now - 1 > pre:
                # print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        # print(" ".join(outputs[pre:]), flush=True)
        print("Streaming finished\n")

        self.conv.messages[-1][-1] = " ".join(output)
        # print(self.conv.messages[-1][1], "\n")
        return self.conv.messages[-1][1]
        # return self.generate_stream(self.tokenizer, self.model, params, self.args.device)


    def run(self, inp):
        print("Vicuna has been reached\n")
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        print("Input is appended\n")
        prompt = self.conv.get_prompt()
        print("Prompt was fetched\n")
        params = {
        "model": self.args.model_name,
        "prompt": prompt,
        "temperature": self.args.temperature,
        "max_new_tokens": self.args.max_new_tokens,
        "stop": self.conv.sep if self.conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE else self.conv.sep2,
        }

        # print(f"{self.conv.roles[1]}: ", end="", flush=True)
        pre = 0
        print("Streaming started\n")
        for outputs in self.generate_stream(self.tokenizer, self.model, params, self.args.device):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                # print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        # print(" ".join(outputs[pre:]), flush=True)
        print("Streaming finished\n")

        self.conv.messages[-1][-1] = " ".join(outputs)
        # print(self.conv.messages[-1][1], "\n")
        return self.conv.messages[-1][1]
    
    def clear_history(self):
        self.conv.messages.clear()
    
    def clear_cache(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()


class PersonaGPT(TTTStrategy):
    """ A text to text strategy, it builds on the DialoGPT-medium pretrained model based on the GPT-2 architecture. Subclass of TTTStrategy
    ----------
    Attributes:


    ----------
    Functions:

    ----------
    """
    def __init__(self):
        """
        ----------
        Parameters: None
        ----------
        Return: None
        ----------
        """
        super().__init__()            
        self.tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT", padding_side='left')
        self.model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.dialog_hx = []

    def flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def to_data(self, x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()
    
    def to_var(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
        return x
    
    def display_dialog_history(self):
        for j, line in enumerate(self.dialog_hx):
            msg = self.tokenizer.decode(line)
            if j %2 == 0:
                print(">> User: "+ msg)
            else:
                print("Bot: "+msg)
                print()
    
    def generate_next(self, bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_new_tokens=1000):
        full_msg = self.model.generate(bot_input_ids, do_sample=True,
                                                top_k=top_k, top_p=top_p, 
                                                max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        msg = self.to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        return msg

    def run(self, inp):
        # encode the user input
        user_inp = self.tokenizer.encode(inp + self.tokenizer.eos_token)
        # append to the chat history
        self.dialog_hx.append(user_inp)
            
        # generated a response while limiting the total chat history to 1000 tokens, 
        bot_input_ids = self.to_var([self.flatten(self.dialog_hx)]).long()
        msg = self.generate_next(bot_input_ids)
        self.dialog_hx.append(msg)
        return self.tokenizer.decode(msg, skip_special_tokens=True)
    
    def clear_history(self):
        self.dialog_hx.clear()
    
    def clear_cache(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()


class BotPipline(TTTStrategy):
    """ A text to text strategy. Subclass of TTTStrategy
    ----------
    Attributes:


    ----------
    Functions:

    ----------
    """
    def __init__(self):
        """
        ----------
        Parameters: None
        ----------
        Return: None
        ----------
        """
        super().__init__()
        self.conv = pipeline("conversational", model=self.model_name, device=0)
        self.conversation = Conversation()

    def run(self, inp):
        
        self.conversation.add_user_input(inp)    
        
        return self.conv(self.conversation).generated_responses[-1]
    
    def clear_history(self):
        del self.conversation
        self.conversation = Conversation()
    
    def clear_cache(self):
        del self.conv
        self.conv = None
        torch.cuda.empty_cache()


class BlenderBot(BotPipline):

    def __init__(self):
        self.model_name = "facebook/blenderbot-400M-distill"
        super().__init__()
    
    def run(self, inp):
        return super().run(inp)
    
    def clear_history(self):
        return super().clear_history()
    
    def clear_cache(self):
        return super().clear_cache()
    

class Guanaco(BotPipline):

    def __init__(self):
        self.model_name = "Fredithefish/Guanaco-3B-Uncensored-v2"
        super().__init__()
    
    def run(self, inp):
        return super().run(inp)
    
    def clear_history(self):
        return super().clear_history()
    
    def clear_cache(self):
        return super().clear_cache()


    


class GTTSAPI(TTSStrategy):
    """
    ----------
    Attributes: None

    ----------
    Functions:
        run(): 
    ----------
    """
    def __init__(self):
        super().__init__()
        self.lang = "en"
        self.slow=False

    def run(self, text:str, filepath:str = "text_to_speech.wav"):
        """
        ----------
        Parameters:

        ----------
        Return:
        ----------
        """
        tts = gtts.gTTS(text, lang=self.lang, slow=self.slow)
        tts.save(filepath)
        return filepath

class SpeechT5(TTSStrategy):
    """
    ----------
    Attributes: None

    ----------
    Functions:
        run(): 
    ----------
    """

    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        
        self.speakers_list = {
            'awb': 0,     # Scottish male
            'bdl': 1138,  # US male
            'clb': 2271,  # US female
            'jmk': 3403,  # Canadian male
            'ksp': 4535,  # Indian male
            'rms': 5667,  # US male
            'slt': 6799   # US female
        }
        self.speaker = self.speakers_list["slt"]

    def run(self, text:str, filepath:str = "text_to_speech.wav"):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        if self.speaker is not None:
            # load xvector containing speaker's voice characteristics from a dataset
            speaker_embeddings = torch.tensor(self.embeddings_dataset[self.speaker]["xvector"]).unsqueeze(0).to(self.device)
        else:
            # random vector, meaning a random voice
            speaker_embeddings = torch.randn((1, 512)).to(self.device)
        # generate speech with the models
        speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
        
        sf.write(filepath, speech.cpu().numpy(), samplerate=16000)
        return filepath
		

class Espnet(TTSStrategy):
    """
    ----------
    Attributes: None

    ----------
    Functions:
        run(): 
    ----------
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

    def run(self, text:str, filepath:str = "/text_to_speech.wav"):
        speech = self.model(text)
        
        sf.write(filepath, speech['wav'].numpy(), samplerate=16000)
        return filepath
		
