from abc import ABC, abstractmethod
from faster_whisper import WhisperModel
import os

class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_file_path: str) -> str:
        """Takes an audio file path and returns the transcribed text."""
        pass

class FasterWhisperProvider(STTProvider):
    def __init__(self, model_size="small.en", device="cpu", compute_type="int8"):
        # device="cpu" keeps your VRAM completely safe!
        # compute_type="int8" makes it run blazingly fast on standard RAM.
        print(f"🦻 Loading Faster-Whisper ({model_size}) on {device.upper()}...", flush=True)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("✅ Faster-Whisper Ready.", flush=True)

    def transcribe(self, audio_file_path: str) -> str:
        if not os.path.exists(audio_file_path):
            return ""
            
        print("✍️ Transcribing...", flush=True)
        # beam_size=5 is the sweet spot for accuracy vs speed
        segments, info = self.model.transcribe(audio_file_path, beam_size=5)
        
        # Combine all spoken segments into one string
        text = "".join([segment.text for segment in segments])
        return text.strip()
