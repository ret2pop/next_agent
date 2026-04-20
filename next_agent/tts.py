from abc import ABC, abstractmethod
import torch
import soundfile as sf
from py_qwen3_tts_cpp.model import Qwen3TTSModel

from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import sounddevice as sd
import numpy as np

import os

class TTSProvider(ABC):
    @abstractmethod
    def generate_audio(self, text: str, output_path: str = "output.wav") -> None:
        """Generates audio from text and saves it to the given path."""
        pass
class KokoroProvider(TTSProvider):
    def __init__(self):
        self.pipeline = KPipeline(lang_code='a')

    def generate_audio(self, text: str, output_path: str="output.wav") -> None:
        print("🔊 Streaming voice...", flush=True)
        generator = self.pipeline(text, voice='af_sky')
        
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as stream:
            for i, (gs, ps, audio) in enumerate(generator):
                audio_data = audio.reshape(-1, 1)
                stream.write(audio_data)
                
        print("✅ Finished speaking!", flush=True)

class QwenTTSProvider(TTSProvider):
    def __init__(
        self, 
        model_name: str = "qwen3-tts-0.6b-q4-k-m",
    ):
        print(f"🔊 Loading Qwen-TTS Model ({model_name})...", flush=True)
        self.model = Qwen3TTSModel(
            tts_model=model_name,
            n_threads=4
        )
        print("✅ Qwen-TTS Ready.", flush=True)

    def generate_audio(self, text: str, output_path: str = "agent_response.wav") -> None:
        if not text.strip():
            return

        try:
            print("generating voice...", flush=True)
            result = self.model.synthesize(text, language="en")
            print("generated!", flush=True)
            self.model.save_audio(result, "audio-output.wav")
        except Exception as e:
            print(f"[!] TTS Generation failed: {e}", flush=True)
