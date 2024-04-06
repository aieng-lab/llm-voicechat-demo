import torch
import speech_recognition as sr
from abc import ABC, abstractmethod

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, BarkModel
from transformers import Conversation, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import argparse
from fastchat.conversation import conv_templates, SeparatorStyle
from datasets import load_dataset
import soundfile as sf
# from espnet2.bin.tts_inference import Text2Speech
from transformers import WhisperProcessor
# from optimum.bettertransformer import BetterTransformer
from datasets import load_dataset
import numpy as np
from TTS.api import TTS
import sys




    
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
            print("Microphone is started ... \n")
            self.r.adjust_for_ambient_noise(source)
            print("Microphone is listening ... \n")
            audio = self.r.listen(source)
            print("Microphone is recognizing ... \n")
            text=self.p(audio.get_flac_data(), max_new_tokens=500, generate_kwargs={"language": "german"})["text"]
            print(text, "\n")
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
        
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="german", task="transcribe")
        # self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        self.p = pipeline(task="automatic-speech-recognition", model="openai/whisper-tiny", generate_kwargs={"forced_decoder_ids": self.forced_decoder_ids})
        



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
            text:str, the converted text from the recorded speechstt_model
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
        self.p = pipeline("automatic-speech-recognition", model="openai/whisper-large", device=0)

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
    def __init__(self, device=None):
        """
        ----------
        Parameters: None

        ----------
        Return: None
        ----------
        """
        super().__init__()
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.torch_dtype = torch.float16 if device=="cuda:0" else torch.float32
        

        self.model_id = "openai/whisper-large-v2"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        # self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

        self.r = sr.Recognizer()
        
        self.p = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    # def run(self)-> str:
    #     """ Record a speech using the microphone then convert and return it as text.
    #     ----------
    #     Parameters: None
      
    #     ----------
    #     Return: 
    #         text:str, the converted text from the recorded speech
    #     ----------
    #     """
    #     print("stt_model hase been reached ...\n")
    #     with sr.Microphone() as source:
    #         print("Microphone is started ... \n")
    #         self.r.adjust_for_ambient_noise(source)
    #         print("Microphone is listening ... \n")
    #         audio = self.r.listen(source)
    #         print("Microphone is recognizing ... \n")
    #         audio_data = audio.get_flac_data()
    #         text=self.p(audio_data, max_new_tokens=500, generate_kwargs={"language": "german"})["text"]
    #         # text = "Hallo Vicuna, kannst du mir eine kurze Geschichte erzählen?"
    #     return text
    #     # return super().run()

    def run(self, audio):
        return self.p(audio, max_new_tokens=500, generate_kwargs={"language": "german"})["text"]

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
        self.args = dict(model_name='lmsys/vicuna-7b-v1.5-16k',
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
        self.welcome_message = "Hallo, Ich heiße Vicuna, wie kann ich dir helfen?"
        self.conv.append_message(self.conv.roles[1], self.welcome_message)

    
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


        del past_key_values

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

        pre = 0
        #new
        outputs = self.generate_stream(self.tokenizer, self.model, params, self.args.device)
        for output in outputs:
            output = output[len(prompt)-3:].strip()
            output = output.split(" ")
            now = len(output)
            if now - 1 > pre:
                pre = now - 1
        self.conv.messages[-1][-1] = " ".join(output)
        return self.conv.messages[-1][1]


    def run(self, inp):
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
        # print(prompt)
        yielded_output = ""
        for outputs in self.generate_stream(self.tokenizer, self.model, params, self.args.device):
            if outputs.endswith(('.', '?', '!', ':')):
                
                if len(yielded_output) == 0:
                    outputs = outputs[len(prompt)-3:].strip()
                else:
                    outputs = outputs[len(prompt) + len(yielded_output) -3:].strip()
                
                yielded_output+= outputs + " "
                yield outputs
        yield "END"
        self.conv.messages[-1][-1] = yielded_output.strip()
    
    def clear_history(self):
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[1], self.welcome_message)
    
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
        self.speaker = self.speakers_list["rms"]

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
		

# class Espnet(TTSStrategy):
#     """
#     ----------
#     Attributes: None

#     ----------
#     Functions:
#         run(): 
#     ----------
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

#     def run(self, text:str, filepath:str = "/text_to_speech.wav"):
#         speech = self.model(text)
        
#         sf.write(filepath, speech['wav'].numpy(), samplerate=16000)
#         return filepath
    
#     def run(self, text:str, filepath:str = "/text_to_speech.wav"):
#         speech = self.model(text)
#         return speech['wav'].numpy().astype(np.float32)
		


class BarkSmall(TTSStrategy):
    """
    ----------
    Attributes: None

    ----------
    Functions:
        run(): 
    ----------
    """

    def __init__(self, voice_preset = "v2/de_speaker_5") -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to(self.device)
        # self.model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16, use_flash_attention_2=True).to(self.device)

        self.processor = AutoProcessor.from_pretrained("suno/bark-small")

        self.voice_preset = voice_preset


    def run(self, text:str, filepath:str = "text_to_speech.wav"):
        
        inputs = self.processor(text, voice_preset=self.voice_preset).to(self.device)
        print("Generating started \n")
        audio_array = self.model.generate(**inputs)
        print("Generating finished\n")
        sampling_rate = self.model.generation_config.sample_rate
        data = audio_array.cpu().numpy().squeeze()
        data = data.astype(np.float32)
        # sf.write(filepath, data, samplerate=sampling_rate)
        return data

class Bark(TTSStrategy):
    """
    ----------
    Attributes: None

    ----------
    Functions:
        run(): 
    ----------
    """

    def __init__(self, voice_preset = "v2/de_speaker_5") -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(self.device)

        self.processor = AutoProcessor.from_pretrained("suno/bark")

        self.voice_preset = voice_preset


    def run(self, text:str, filepath:str = "text_to_speech.wav"):
        
        inputs = self.processor(text, voice_preset=self.voice_preset).to(self.device)
        print("Generating started \n")
        audio_array = self.model.generate(**inputs)
        print("Generating finished\n")
        sampling_rate = self.model.generation_config.sample_rate
        data = audio_array.cpu().numpy().squeeze()
        data = data.astype(np.float32)
        # sf.write(filepath, data, samplerate=sampling_rate)
        # return filepath
        return data

#"Hallo, da is Salim. Ich komme aus Syrien und bin 28 Jahre alt. Ich mache einen Master in Informatik in Passau. Ich habe im Moment einen Minijob an der Uni. Dabei muss ich ein VoiceBot entwickeln."
    


class XTTS_V2(TTSStrategy):
    """
    ----------
    Attributes: None

    ----------
    Functions:
        run(): 
    ----------
    """

    def __init__(self, speaker:str = "welcome_message_2.wav") -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.model = TTS("tts_models/de/thorsten/vits").to('cuda:0')

        self.voice_preset = speaker



    def run(self, text:str, filepath:str = "text_to_speech.wav"):
        
        print("Generating started \n")
        # audio_list = self.model.tts(text=text, speaker_wav=self.voice_preset, language="de")
        audio_list = self.model.tts(text=text, speaker_wav=self.voice_preset)
        data = np.asarray(audio_list, dtype=np.float32)
        print("Generating finished\n")
        # sf.write(filepath, data, samplerate=sampling_rate)
        return data
