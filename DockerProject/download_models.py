from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
from TTS.api import TTS
import os
import torch

main_path = os.path.dirname(os.path.realpath(__file__))
model_path = main_path + "/models/"

# Check if the directory already exists
if not os.path.exists(model_path):
    # Create the directory
    os.makedirs(model_path)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device=="cuda" else torch.float32

    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    whisper_model.save_pretrained(model_path+"WhisperLargeV2/")

    whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
    whisper_processor.save_pretrained(model_path+"WhisperLargeV2/")

    kwargs = {"torch_dtype": torch.float16}
    kwargs.update({"load_in_8bit": True, "device_map": "auto"})

    fastchat_model = model = AutoModelForCausalLM.from_pretrained('lmsys/vicuna-7b-v1.5-16k', low_cpu_mem_usage=True, **kwargs)
    fastchat_model.save_pretrained(model_path+"FastChat/")

    fastchat_tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5-16k', use_fast=False)
    fastchat_tokenizer.save_pretrained(model_path+"FastChat/")

    # tts_model = TTS("tts_models/de/thorsten/vits")
    # tts_model.s
    
TTS_HOME = os.environ.get("TTS_HOME")
XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
print(TTS_HOME, XDG_DATA_HOME)
