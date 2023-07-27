import torch
import numpy as np
import speech_recognition as sr
import os
from langchain.llms import OpenAI
import gtts
from playsound import playsound
from abc import ABC, abstractmethod
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import argparse
import time
from fastchat.conversation import conv_templates, SeparatorStyle


OPENAI_API_KEY="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"





    
class STTStrategy(ABC):
    """Interface for Speech-to-Text Models.
    ----------
    Functions:
        run():  Convert speech input to text.
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
        run():  Reply to the input text with text.
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
        run():  Convert text input to audio object, save it into a .mp3 file then play it.
                parameters: text:str
                Return: None.
    ----------
    """
    @abstractmethod
    def run(self, text:str):
        pass

class App():
    """Our Speech-to-Speech application class.
    ----------
    Parameters:
        stt_model:STTStrategy 'Speech-to-Text Model'.

        ttt_model:TTTStrategy 'Text-to-Text Model'.

        tts_model:TTSStrategy 'Text-to-Speech Model'.

    ----------
    Functions:
        run_STT(None)-> str

        run_TTT(str) -> str

        run_TTS(str) -> None

        run(None) -> None
    ----------
    """
    def __init__(self, 
                 stt_model:STTStrategy,
                 ttt_model:TTTStrategy,
                 tts_model:TTSStrategy):
        """
        ----------
        Parameters:

        ----------
        Return:
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
        print("You: " + text)
        return text

    def run_TTT(self, text:str):
        """ Call the run(str) function of the ttt_model and print the output.
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
    """
    ----------
    Parameters:

    ----------
    Functions:
    ----------
    """
    def __init__(self):
        """
        ----------
        Parameters:

        ----------
        Return:
        ----------
        """
        super().__init__()
        self.r = sr.Recognizer()
	
    def run_file(self, path):
        """ Record a speech using the microphone and convert it to text.
        ----------
        Parameters: None
      
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
        """
        ----------
        Parameters:

        ----------
        Return:
            text
        ----------
        """
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source) 
            audio_data = self.r.listen(source)
            print("Recognizing...")
            text = self.r.recognize_google(audio_data, language="en-US", show_all=True)
        return text["alternative"][0]["transcript"]
        
class WhisperModel(STTStrategy):
    """
    ----------
    Parameters:

    ----------
    Functions:
    ----------
    """    
    def __init__(self):
        """
        ----------
        Parameters:

        ----------
        Return:
        ----------
        """
        super().__init__()
        self.r = sr.Recognizer()
#         self.p = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        self.p = pipeline("automatic-speech-recognition", model="openai/whisper-base")

    def run(self)-> str:
        """
        ----------
        Parameters:

        ----------
        Return:
        ----------
        """
        with sr.Microphone() as source:
            self.r.adjust_for_ambient_noise(source)
            audio = self.r.listen(source)
            text=self.p(audio.get_flac_data())["text"]
        return text
              





class OpenAIModel(TTTStrategy):
    """
    ----------
    Parameters:

    ----------
    Functions:
    ----------
    """
    def __init__(self):
        """
        ----------
        Parameters:

        ----------
        Return:
        ----------
        """
        super().__init__()
        self.llm = OpenAI(openai_api_key = OPENAI_API_KEY)
	
    def run(self, text:str)->str:
        """
        ----------
        Parameters:

        ----------
        Return:
        ----------
        """
        reply = self.llm.predict(text).strip()
        print("AI : " + reply)
        return reply

    
class FastChatModel(TTTStrategy):
    """
    ----------
    Parameters:

    ----------
    Functions:
    ----------
    """
    def __init__(self):
        """
        ----------
        Parameters:

        ----------
        Return:
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

        print(f"{self.conv.roles[1]}: ", end="", flush=True)
        pre = 0
        for outputs in self.generate_stream(self.tokenizer, self.model, params, self.args.device):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        print(" ".join(outputs[pre:]), flush=True)

        self.conv.messages[-1][-1] = " ".join(outputs)
        return self.conv.messages[-1][1]
    
    def clear_history(self):
        self.conv.messages.clear()


class GTTSAPI(TTSStrategy):
    """
    ----------
    Parameters:

    ----------
    Functions:
    ----------
    """
    def run(self, text:str, filepath:str = "../Files/text_to_speech.wav"):
        """
        ----------
        Parameters:

        ----------
        Return:
        ----------
        """
        tts = gtts.gTTS(text)
        tts.save(filepath)
        playsound(filepath)

		

