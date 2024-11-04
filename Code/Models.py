import torch
import speech_recognition as sr
from abc import ABC, abstractmethod
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor
import argparse
from fastchat.conversation import conv_templates, SeparatorStyle, register_conv_template, Conversation

# from fastchat.serve.serve_chatglm import ChatGLMServeAPI

import numpy as np
from TTS.api import TTS
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from openai import OpenAI



    
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
    
class TTIStrategy(ABC):
    """Interface for Text-to-Image deffusion Models.
    ----------
    Functions:
        run:  Generate an image out of the input text.
                parameters: text:str
                Return: PIL.Image.Image.
    ----------
    """
    @abstractmethod
    def run(self, text:str):
        pass


        
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
            # text=self.p(audio.get_flac_data(), max_new_tokens=500, generate_kwargs={"language": "german"})["text"]
            text=self.p(audio.get_flac_data(), max_new_tokens=500)["text"]
            print(text, "\n")
        return text

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
        # self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="german", task="transcribe")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(task="transcribe")
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

    "Phrases or text snippets that are commonly returned by the model when it is applied for silent audio."
    SILENCE_PHRASES = [
        "Untertitel von Stephanie Geiges",
        "Untertitel im Auftrag des ZDF für funk, 2017",
        "Untertitelung des ZDF für funk, 2017",
        "Untertitelung im Auftrag des ZDF, 2021",
        " Danke für's Zuschauen!",
        "Bis zum nächsten Mal.",
    ]

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
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

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

    # def run(self, audio):
    #     # return self.p(audio, max_new_tokens=500, generate_kwargs={"language": None})["text"]
    #     return self.p(audio, max_new_tokens=500, generate_kwargs={"task": "transcribe"})["text"]
    
    def run(self, audio, language):
        # return self.p(audio, max_new_tokens=500, generate_kwargs={"language":"german", "task":"transcribe"})["text"]
        # return self.p(audio, generate_kwargs={"language":"german"})["text"]
        try:
            if language=="multi":
                text = self.p(audio, max_new_tokens=500, generate_kwargs={"task": "transcribe"})["text"]
            elif language == "de":
                text = self.p(audio, max_new_tokens=500, generate_kwargs={"language": "german"})["text"]
            else:
                text = self.p(audio, max_new_tokens=500, generate_kwargs={"language": "english"})["text"]

            if not text.strip() in self.SILENCE_PHRASES:
                return text, True
        except Exception:
            pass
        if language == "de":
            return "Es tut mir Leid, Es gab ein Problem mit dem aufgenommenen Audio. Kannst du deine Anfrage nochmal stellen?", False
        else:
            return "I'm sorry, there was a problem with the recorded audio. Can you make your request again?", False


    
class FastChatModel(TTTStrategy):
    """ A text to text strategy based on FastChat project using Vicuna model. Subclass of TTTStrategy
    ----------
    Attributes:
        args:dict:
            model_name (str): the name of the used vicuna model.
            device (str): 'cpu', 'cuda', 'mps'
            num_gpus (int, optional): number of GPUs to be used by the model.
            quantization (bool, optional):  whether to use the compression for low memory.
            conv_template (str): specify the template of the conversation.
            temperature (float) : .
            max_new_token:int, max amount of generated characters.
            debug:bool.
        
        model: the model used to generate text.

        tokenizer: the tokenizer used to encode inputs and decode outputs of the model.

        conv: conversation template.
    """
    def __init__(self, params, model_name='lmsys/vicuna-7b-v1.5-16k'):
        super().__init__()     
        self.params = params       
        self.args = dict(model_name=model_name,
                        #model_name='lmsys/vicuna-7b-v1.5-16k',
                        #  model_name="hugging-quants/Meta-Llama-3.1-8B-BNB-NF4",
                        device='cuda',
                        num_gpus=1,
                        quantization=True,
                        # conv_template="vicuna_de_v1.1",
                        # conv_template='vicuna_v1.1',
                        conv_template='custom_vicuna',
                        temperature=0.7,
                        max_new_tokens=512,
                        debug=False,)
        self.args = argparse.Namespace(**self.args)
        self.model, self.tokenizer = self.load_model(self.args.model_name, self.args.device, self.args.num_gpus, self.args.quantization)
        
        register_conv_template(
            Conversation(
                name=self.args.conv_template,
                system_message= self.params["text_generation_system_prompt"],
                roles=("USER", "ASSISTANT"),
                sep_style=SeparatorStyle.ADD_COLON_TWO,
                sep=" ",
                sep2="</s>",
            )
        )
        
        self.conv = conv_templates[self.args.conv_template].copy()
        self.welcome_message = self.params["welcome_message"]
        self.conv.append_message(self.conv.roles[1], self.welcome_message)
        self.shift = 0

    
    def load_model(self, model_name, device, num_gpus=1, quantization=True):
        """Loads both the model and tokenizer.
        Args:
            model_name (str): model's path on local machine or model's id on HuggingFace.co
            device (str): On which the model will be loaded. 'cpu', 'cuda', 'mps'
            num_gpus (int, optional): number of GPUs to be used by the model. Defaults to 1.
            quantization (bool, optional): to use the compression for low memory. Defaults to True.
        Raises:
            ValueError: If the selected device is invalid.
        Returns:
            model (AutoModelForCausalLM): the loaded model.
            tokenizer (AutoTokenizer): the loaded tokenizer.
        """

        if device == "cpu":
            kwargs = {}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if quantization:
                if num_gpus != "auto" and int(num_gpus) != 1:
                    print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
                kwargs.update({"load_in_8bit": True, "device_map": "auto"})
                # kwargs.update({"load_in_4bit": True, "device_map": "auto"})
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
        if device == "cuda" and num_gpus == 1 and not quantization:
            model.to("cuda")
        elif device == "mps":
            model.to("mps")

        return model, tokenizer
    
    @torch.inference_mode()
    def generate_stream(self, tokenizer, model, params, device,
                        context_len=2048, stream_interval=2):
        """Adapted from fastchat/serve/model_worker.py::generate_stream.
        Generates and stream text.
        Args:
            tokenizer (AutoTokenizer): Pre-trained tokenizer.
            model (AutoModelForCausalLM): Pre-trained large language model
            params (dict): other arguments.
            device (str): On which the model will be loaded. 'cpu', 'cuda', 'mps'
            context_len (int, optional): Length of history to be considered plus the new generated text. Defaults to 2048.
            stream_interval (int, optional): Number of tokens to be yielded each time. Defaults to 2.
        Yields:
            (str): the generated text.
        """

        prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = int(params.get("max_new_tokens", 256))
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        # The input is then preprocessed to ensure it is within a specific length
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
        """Generates text based on the input query.
        Args:
            inp (str): input query
        Returns:
            str: Model's response
        """
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
        self.shift+=3
        outputs = self.generate_stream(self.tokenizer, self.model, params, self.args.device)
        for output in outputs:
            response = output[len(prompt)-self.shift:]
        self.conv.messages[-1][-1] = response
        return response


    def run(self, inp):
        """Generates and streams text based on the input query.
        Args:
            inp (str): input query
        Yields:
            str: Model's response
        """
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
        yielded_output = ""
        self.shift+=4
        for outputs in self.generate_stream(self.tokenizer, self.model, params, self.args.device):
            if outputs.endswith(('.', '?', '!', ':')):
                
                outputs = outputs[len(prompt)+len(yielded_output)-self.shift:].strip()
                yielded_output+= outputs + " "
                yield outputs
        yield "END"
        self.conv.messages[-1][-1] = yielded_output.strip()
    
    def clear_history(self):
        """Clear chat's history.
        """
        self.conv.messages.clear()
        self.conv = conv_templates[self.args.conv_template].copy()
        self.conv.append_message(self.conv.roles[1], self.welcome_message)
        self.shift = 0
        
    def clear_cache(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        

class FastChatLlama(FastChatModel):
    def __init__(self, params, *args, **kwargs):
        super(FastChatLlama, self).__init__(params,
                                            model_name="meta-llama/Meta-Llama-3.1-8B",
                                            #model_name="hugging-quants/Meta-Llama-3.1-8B-BNB-NF4",
                                            *args, **kwargs)
    
    def load_model(self, model_name, device, num_gpus=1, quantization=True):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,  # Enable 4-bit precision
                                        bnb_4bit_use_double_quant=True,  # Optional: Enable double quantization
                                        bnb_4bit_quant_type="nf4",  # Use NF4 quantization
                                    )

        # Define a simplified rope_scaling configuration
        rope_scaling = {
            "type": "linear",  # Assuming 'linear' scaling for RoPE
            "factor": 8.0  # Adjust the factor as per your model's requirements
        }

        # Use the Hugging Face transformers library to load the model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for better memory efficiency
            device_map="auto",  # Automatically assigns the model to available GPUs/CPUs
            quantization_config=bnb_config,  # Updated: Use BitsAndBytesConfig
            rope_scaling=rope_scaling  # Include the simplified rope_scaling configuration
        )  
        return model, tokenizer


# class Llama8BNF4(TTTStrategy):
#     def __init__(self, params, model_name: str = "hugging-quants/Meta-Llama-3.1-8B-BNB-NF4", max_history_length: int = 2048):
#         """
#         Initializes the Llama8BNF4 class by loading the specified quantized Llama model
#         and tokenizer using Hugging Face's transformers library. It also maintains a conversation history.
        
#         Args:
#         - model_name (str): The name of the model to load from Hugging Face.
#         - max_history_length (int): Maximum token length for conversation history.
#         """
#         super().__init__()
#         self.params = params
#         print("Loading the tokenizer and model...")

#         # Load the tokenizer for the model
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad_token to eos_token

#         # Load the model with quantization configuration (using BNB-NF4)
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16  # Set to float16 for performance
#         )

#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             quantization_config=quantization_config,
#             device_map="auto"  # Automatically maps model to available GPUs
#         )

#         # Initialize the system role and conversation history
#         # self.system_role = self.params["text_generation_system_prompt"]
#         self.system_role = (
#             "You are a helpful Assistant. Answer the user's query without repeating the input. "
#             "Respond to queries in the language they are provided. Do not attempt to complete the user's conversation." 
#             "Always write the role Assistant like this Assistant"
#             "--- "
#             "Du bist ein hilfreicher Assistant. Beantworte die Anfrage des Users, ohne die Eingabe zu wiederholen. "
#             "Antworte in der Sprache, in der die Anfrage gestellt wird. Versuche nicht, das Gespräch des Users zu beenden."
#             "Schreib immer Assistant statt Assistenz und Assistent"
#         )
#         self.conv = [self.system_role]  # Conversation history starts with the system role
#         self.conv.append(f"Assistant: {self.params['welcome_message']}") # Add welcome message
#         self.max_history_length = max_history_length  # Set max history token length
#         print("Model and tokenizer loaded successfully.")

#     def update_conv(self, user_input: str, model_response: str):
#         """
#         Updates the conversation history with the user query and model response.
        
#         Args:
#         - user_input (str): The user's input query.
#         - model_response (str): The model's generated response.
#         """
#         self.conv.append(f"User: {user_input}")
#         self.conv.append(f"Assistant: {model_response}")

#         # Truncate the conversation if it exceeds the maximum token length
#         conv_text = " ".join(self.conv)
#         conv_tokens = self.tokenizer(conv_text, return_tensors="pt")["input_ids"]
        
#         if conv_tokens.shape[1] > self.max_history_length:
#             excess_tokens = conv_tokens.shape[1] - self.max_history_length
#             # Trim the excess tokens from the conversation history
#             conv_tokens = conv_tokens[:, excess_tokens:]
#             self.conv = self.tokenizer.decode(conv_tokens[0], skip_special_tokens=True).split("Assistant:")

#     def run(self, query: str) -> str:
#         """
#         Generates a conversational response based on the input query using the Llama model.
#         The response is influenced by the conversation history and the system role.
        
#         Args:
#         - query (str): The input string query to generate a response for.
        
#         Returns:
#         - response (str): The generated conversational response.
#         """
#         print(f"Processing query: {query}")
        
#         # Combine the conversation history (including system role) with the current query
#         full_conv = " ".join(self.conv + [f"User: {query}"])
        
#         # Tokenize the conversation and the user input
#         inputs = self.tokenizer(full_conv, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

#         print(f"Input token length: {inputs['input_ids'].shape[1]}")  # Debug token length

#         # Generate a response from the model
#         try:
#             output_tokens = self.model.generate(
#                 inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_length=self.max_history_length,
#                 num_beams=5,
#                 no_repeat_ngram_size=2,
#                 early_stopping=True,
#                 pad_token_id=self.tokenizer.pad_token_id,  # Handle padding
#                 eos_token_id=self.tokenizer.eos_token_id   # Handle end of sequence
#             )
            
#             # Decode the generated tokens into a human-readable string
#             response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            
            
#             # Post-process to replace "Assistent" with "Assistant"
#             response = response.replace("Assistent", "Assistant")
            
#             # Post-process to replace "Assistent" with "Assistant"
#             response = response.replace("Assistenz", "Assistant")
            
#             # Ensure response starts with "Assistant:" and avoid continuing the conversation
#             response = response.split("Assistant:")[-1].strip()
            

#             if response:
#                 print(f"Generated response: {response}")
#                 self.update_conv(query, response)
#                 return response
#             else:
#                 print("Warning: The model generated an empty response.")
#                 return "I'm sorry, I didn't understand that. Could you rephrase?"
#         except Exception as e:
#             print(f"Error during generation: {e}")
#             return "I'm sorry, something went wrong while generating the response."



class OpenAIAPI(TTTStrategy):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.client = OpenAI(api_key=self.params["openai_api_key"], base_url=self.params["openai_base_url"])
        self.model_name = self.params["openai_model"]
        self.conv = [
            {"role": "system", "content": self.params["text_generation_system_prompt"]},
            {"role": "assistant", "content": self.params["welcome_message"]}
        ]

    def run(self, query):
                
        # Add a new entry with 'user' role and put the query in the 'content' at the end of the conversation.
        self.conv.append({"role": "user", "content": query})
        stream = self.client.chat.completions.create(model=self.model_name,
                                                    messages=self.conv,
                                                    stream=True,)
        
        #Holds the temporal unprocessed chunks of the response.
        leftover = ""
        
        # Holds the processed chunks of the response to be yielded.
        output = ""
        
        #Holds the entire response to be added to conv.
        assistant_response = ""
        #Get the message out of the model's response and process the response chunks to get smoother streaming.
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                output += chunk.choices[0].delta.content
                assistant_response+= chunk.choices[0].delta.content
                
                if "." in output:
                    output = output.split(".")
                    leftover = output[-1]
                    output = ".".join(output[:-1])+ "."
                    
                    yield output                
                    output = leftover
        if len(output)>1:
            yield output
        yield "END"                                
        
        # Add the answer to a new entry at the end of the conversation with role 'assistant'.
        self.conv.append({"role": "assistant", "content": assistant_response})
    
    def clear_history(self):
        """Clear chat's history.
        """
        self.conv = [
            {"role": "system", "content": self.params["text_generation_system_prompt"]},
            {"role": "assistant", "content": self.params["welcome_message"]}
        ]
        
        
        
class MixtralAPI(TTTStrategy):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.client = OpenAI(api_key=self.params["openai_api_key"],
                             base_url=self.params["openai_base_url"])
        
        self.model = self.params["openai_model"]
        self.conv = [
            {"role": "system", "content": self.params["text_generation_system_prompt"]},
            {"role": "assistant", "content": self.params["welcome_message"]}
        ]

    def run(self, query):
                
        # Add a new entry with 'user' role and put the query in the 'content' at the end of the conversation.
        self.conv.append({"role": "user", "content": query})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.conv
        )
        #Get the message out of the model's response.
        response = completion.choices[0].message.content
        
        # Add the answer to a new entry at the end of the conversation with role 'assistant'.
        self.conv.append({"role": "assistant", "content": response})                    

        return response
    
    def clear_history(self):
        """Clear chat's history.
        """
        self.conv = [
            {"role": "system", "content": self.params["text_generation_system_prompt"]},
            {"role": "assistant", "content": self.params["welcome_message"]}
        ]


class ThorstenVits(TTSStrategy):
    """A text to speech strategy using /de/thorsten/vits model from Coqui project. Subclass of TTTStrategy
        
        Args:
            speaker (str): path to voice file.
    """

    def __init__(self, speaker:str = "welcome_message.wav") -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.model = TTS("tts_models/de/thorsten/vits").to('cuda:0')

        self.voice_preset = speaker



    def run(self, text:str, filepath:str = "text_to_speech.wav"):
        """Generates speech of the provided text.
        Args:
            text (str): input of which the speech is generated.
            filepath (str, optional): the file path to save the speech in. Defaults to "text_to_speech.wav".
        Returns:
            (numpy.ndarray): Numpy array of audio bytes.
        """        
        # print("Generating started \n")        
        # speed=2.0, emotion="Sad"
        try:
            audio_list = self.model.tts(text=text, speaker_wav=self.voice_preset)
            data = np.asarray(audio_list, dtype=np.float32)
            # print("Generating finished\n")
            return data
        except Exception as e:
            print(e)
            return np.array([], dtype=np.float32)
    
    def run_to_file(self, text:str, filepath:str = "text_to_speech.wav"):
        """Generates speech of the provided text.
        Args:
            text (str): input of which the speech is generated.
            filepath (str, optional): the file path to save the speech in. Defaults to "text_to_speech.wav".
        Returns:
            (numpy.ndarray): Numpy array of audio bytes.
        """        
        # print("Generating started \n")        
        # speed=2.0, emotion="Sad"

        if text:
            try:
                returned_text = self.model.tts_to_file(text=text, speaker_wav=self.voice_preset, file_path=filepath)
                return returned_text
            except Exception as e:
                print(e)
        return ""
        
        
class XTTS_V2(TTSStrategy):
    """A text to speech strategy using /de/thorsten/vits model from Coqui project. Subclass of TTTStrategy
        
        Args:
            speaker (str): path to voice file.
    """

    def __init__(self, speaker:str = "welcome_message.wav") -> None:
        super().__init__()
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

        self.voice_preset = speaker



    def run(self, text:str, filepath:str = "text_to_speech.wav", language:str="en"):
        """Generates speech of the provided text.
        Args:
            text (str): input of which the speech is generated.
            filepath (str, optional): the file path to save the speech in. Defaults to "text_to_speech.wav".
        Returns:
            (numpy.ndarray): Numpy array of audio bytes.
        """        
        # print("Generating started \n")        
        # speed=2.0, emotion="Sad"
        try:
            audio_list = self.model.tts(text=text, speaker_wav=self.voice_preset,language=language)
            data = np.asarray(audio_list, dtype=np.float32)
            # print("Generating finished\n")
            return data
        except Exception as e:
            print(e)
            return np.array([], dtype=np.float32)
    
    
    def run_to_file(self, text:str, language:str, filepath:str = "text_to_speech.wav"):
        """Generates speech of the provided text.
        Args:
            text (str): input of which the speech is generated.
            filepath (str, optional): the file path to save the speech in. Defaults to "text_to_speech.wav".
        Returns:
            (numpy.ndarray): Numpy array of audio bytes.
        """        
        # print("Generating started \n")        
        # speed=2.0, emotion="Sad"
        try:
            returned_text = self.model.tts_to_file(text=text, speaker_wav=self.voice_preset, language=language, file_path=filepath)
            return returned_text
        except Exception as e:
            print(e)
            return ""
        
class StableDiffusion(TTIStrategy):
    
    def __init__(self) -> None:
        super().__init__()
        repo_id = "stabilityai/stable-diffusion-2-base"
        device = "cuda"
        self.pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")
        
    def run(self, text: str):
        return self.pipe(text).images[0]
