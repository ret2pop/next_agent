from abc import ABC, abstractmethod
from langchain_community.chat_message_histories import ChatMessageHistory
from .tts import TTSProvider 
from .stt import STTProvider 

class Agent(ABC):
    def __init__(self, llm, tools, commands, tts_provider: TTSProvider = None, stt_provider: STTProvider = None):
        self.llm = llm
        self.tools = tools
        self.commands = commands
        self.tts_provider = tts_provider
        self.tts_provider = stt_provider
        self.history = ChatMessageHistory()

    @abstractmethod
    def start(self):
        """Starts the main interaction loop for the agent."""
        pass
