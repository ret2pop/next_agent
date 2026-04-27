import re
import os
import hashlib
import subprocess
import threading
import queue
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.application import get_app
from prompt_toolkit.key_binding import KeyBindings

from .vars import (
    DEFAULT_MODEL,
    REPO_PATH,
    DB_PATH,
    MAX_TOOL_CALLS,
    MEMORY_FILE,
    MD_DIR,
    SRC_ROOT,
    AGENT_ROOT,
    EMAIL_ROOT,
    BASE_URL,
    SUMMARY_MODEL,
    SUMMARY_BASE_URL,
)
from .memory import CodebaseRAG
from .tool import tool_registry
from .command import command_registry

# --- NEW IMPORTS ---
from .agent_base import Agent
from .tts import KokoroProvider
from .stt import FasterWhisperProvider


class LocalAgent(Agent):
    def __init__(self):
        tts = KokoroProvider()
        stt = FasterWhisperProvider()
        llm = ChatOllama(model=DEFAULT_MODEL, temperature=0, base_url=BASE_URL)
        self.stt_queue = queue.Queue()

        super().__init__(
            llm=llm,
            tools=tool_registry,
            commands=command_registry,
            tts_provider=tts,
            stt_provider=stt,
        )

        self.monorepo_rag = CodebaseRAG(
            repo_path=REPO_PATH, db_path=DB_PATH, index_name="monorepo_index"
        )
        self.agent_rag = CodebaseRAG(
            repo_path=AGENT_ROOT, db_path=DB_PATH, index_name="agent_index"
        )
        self.email_rag = CodebaseRAG(
            repo_path=EMAIL_ROOT,
            db_path=DB_PATH,
            index_name="email_index",
            valid_exts=None,
            email_store=True,
        )
        self.voice = False
        self.tts_playing = False
        self.interrupt_event = threading.Event()

        if self.tools.schemas:
            self.llm = self.llm.bind_tools(self.tools.schemas)

        self.msg_count = 0
        self.session_file = None

        self.kb = KeyBindings()

        @self.kb.add("enter")
        def _(event):
            """Submit the prompt on Enter."""
            event.current_buffer.validate_and_handle()

        @self.kb.add("c-o")
        def _(event):
            """Insert a newline on Shift+Enter."""
            event.current_buffer.insert_text("\n")

        @self.kb.add("escape")
        def _(event):
            """Exit voice mode and return to text mode."""
            if getattr(self, "voice", False):
                self.voice = False
                event.app.exit(result="<SWITCH_TO_TEXT>")

        @self.kb.add("space")
        def _(event):
            """Interrupt the agent's speech, or insert a space normally."""
            if getattr(self, "tts_playing", False):
                self.interrupt_event.set()
                event.app.exit(result="<INTERRUPTED>")
            else:
                event.current_buffer.insert_text(" ")

        os.makedirs(MD_DIR, exist_ok=True)
        self._inject_system_prompt()
        self.cli_history = InMemoryHistory()

    def _inject_system_prompt(self):
        tool_instructions = self.tools.get_prompt_instructions()
        current_time = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")

        ltm_content = ""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                ltm_content = f.read().strip()

        prompt_content = (
            "You are the personal Digital Famulus to Preston Pan. You are a warm, supportive, "
            "and highly capable collaborator.\n\n"
            "=== SPATIAL AWARENESS (YOUR ENVIRONMENT) ===\n"
            f"You are operating within Preston's local filesystem at these key locations:\n"
            f"- MONOREPO PATH ON DISK: `{REPO_PATH}`.\n"
            f"- PROJECT SOURCE ROOT: `{SRC_ROOT}`.\n"
            f"- YOUR CORE (AGENT_ROOT): `{AGENT_ROOT}`.\n\n"
            f"=== TEMPORAL ANCHORING ===\nCurrent Time: {current_time}.\n\n"
            f"=== USER CONTEXT ===\n{ltm_content}\n\n"
            "=== THE TOOLS ===\n"
            f"{tool_instructions}\n\n"
            "=== FORMATTING RULES: MARKDOWN, LATEX, & CITATIONS ===\n"
            "1. STRUCTURE: Use standard Markdown.\n"
            "2. LATEX: For math/science, use $$ (e.g., $$E=mc^2$$).\n"
            "3. INLINE CITATIONS: MUST cite inline.\n"
            "4. CODE: Use fenced blocks.\n\n"
            "=== OPERATIONAL PROTOCOL ===\n"
            "- Start with a Markdown heading.\n"
            "- Maintain your warm, Chief-of-Staff persona."
        )

        if self.history.messages and isinstance(
            self.history.messages[0], SystemMessage
        ):
            self.history.messages[0] = SystemMessage(content=prompt_content)
        elif not self.history.messages:
            self.history.add_message(SystemMessage(content=prompt_content))
        else:
            self.history.messages.insert(0, SystemMessage(content=prompt_content))

    def strip_thought(self, text):
        if not text:
            return ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _open_in_emacs(self, filepath):
        abs_path = os.path.abspath(filepath)
        print(f"\n[DEBUG] 🚀 Attempting to open: {abs_path}", flush=True)
        try:
            cmd = ["emacsclient", "-n", abs_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ File opened in Emacs buffer.", flush=True)
            else:
                print(
                    f"⚠️ emacsclient returned error: {result.stderr.strip()}", flush=True
                )
        except Exception as e:
            print(f"❌ Failed to execute emacsclient: {e}", flush=True)

    def _save_to_md(self, user_input, agent_response):
        first_save = False
        if not self.session_file:
            first_save = True
            hash_input = (user_input + agent_response).encode("utf-8")
            guid = hashlib.sha256(hash_input).hexdigest()[:12]
            self.session_file = os.path.join(MD_DIR, f"session_{guid}.md")
            with open(self.session_file, "w", encoding="utf-8") as f:
                f.write(
                    f"# Session {guid}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\n"
                )

        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(f"## User Prompt\n{user_input}\n\n")
            f.write(f"## Agent Response\n{agent_response}\n\n")
            f.write("---\n\n")

        if first_save:
            self._open_in_emacs(self.session_file)

    def _voice_worker(self):
        """Background thread: records mic audio, detects interruptions, and transcribes."""

        # [Insert your PyAudio/VAD recording loop here]
        # Conceptually, this loop should continuously read microphone chunks.

        # 1. Voice Interruption Logic:
        # If self.tts_playing is True AND the mic volume spikes (user started speaking):
        #     self.interrupt_event.set()

        # 2. Recording Logic:
        # Record audio until silence is detected, then save to a temporary file.
        temp_audio_path = "temp_user_input.wav"

        # 3. Transcribe:
        if not self.interrupt_event.is_set():
            transcribed_text = self.stt_provider.transcribe(temp_audio_path)

            if transcribed_text:
                # Safely push the result to the active prompt_toolkit event loop
                app = get_app()
                if app and app.is_running:
                    app.exit(result=transcribed_text)

    def _record_audio(self, filename="temp_user_input.wav", duration=5, fs=16000):
        """Blocks the thread while recording from the default microphone."""
        # fs=16000 is the sample rate Whisper expects natively
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # <--- THIS is what actually blocks the thread while you speak!
        write(filename, fs, recording)
        return filename

    def _speak_response(self, text, llm):
        """Generates and plays TTS audio, allowing for real-time interruption."""
        if not getattr(self, "voice", False):
            return

        self.tts_playing = True
        self.interrupt_event.clear()

        try:
            print("🧹 Cleaning text for speech...", end="\r", flush=True)
            filter_prompt = (
                "You are a text-cleaning assistant. Rewrite the following text so it can be "
                "read smoothly by a Text-To-Speech engine. Remove all markdown formatting (*, #, _, etc.), "
                "code blocks, URLs, and unpronounceable special characters. "
                "Return ONLY the plain spoken text, nothing else.\n\n"
                f"Text:\n{text}"
            )
            
            clean_response = llm.invoke([HumanMessage(content=filter_prompt)])
            spoken_text = getattr(clean_response, 'content', str(clean_response))
            
            print(" " * 40, end="\r", flush=True)

            sentences = re.split(r"(?<=[.!?]) +", spoken_text.strip())

            for sentence in sentences:
                if self.interrupt_event.is_set():
                    print(" 🛑 [Agent Muted]", flush=True)
                    break

                if not sentence.strip():
                    continue

                self.tts_provider.generate_audio(sentence)

        except Exception as e:
            print(f"\n[!] TTS Error: {e}", flush=True)
            
        finally:
            # 5. Release the state
            self.tts_playing = False


    def _voice_worker(self):
        import os
        print("\n🎤 [Thread] Listening... (You have 5 seconds to speak)", flush=True)
        
        temp_audio_path = self._record_audio(duration=5)
        
        if os.path.exists(temp_audio_path):
            file_size = os.path.getsize(temp_audio_path)
            print(f"📁 [Debug] Audio file created. Size: {file_size} bytes", flush=True)
            if file_size < 1000:
                print("⚠️ [Debug] File is way too small. Mic is likely capturing silence or failing.", flush=True)
        else:
            print("❌ [Debug] ERROR: No audio file was created!", flush=True)
        transcribed_text = self.stt_provider.transcribe(temp_audio_path)
        print(f"✅ [Thread] STT Finished: '{transcribed_text}'", flush=True)
        self.stt_queue.put(transcribed_text)
    def distill_and_exit(self):
        print("\n🧠 Distilling context...", flush=True)
        print("Goodbye!", flush=True)
        exit(0)

    def start(self):
        print(f"\n🤖 NextAgent Ready.", flush=True)
        print("-" * 60, flush=True)

        clean_llm = ChatOllama(model=SUMMARY_MODEL, temperature=0, base_url=SUMMARY_BASE_URL)

        while True:
            self.tts_playing = False
            self.interrupt_event.clear()

            try:
                if getattr(self, "voice", False):
                    listener_thread = threading.Thread(
                        target=self._voice_worker, daemon=True
                    )
                    listener_thread.start()

                    user_input = self.stt_queue.get() 
                    
                    print("captured!")
                else:
                    user_input = pt_prompt(
                        "\nYou: ",
                        history=self.cli_history,
                        multiline=True,
                        key_bindings=self.kb,
                        prompt_continuation="... ",
                    ).strip()

            except (KeyboardInterrupt, EOFError):
                self.distill_and_exit()

            if user_input == "<SWITCH_TO_TEXT>":
                continue
            if user_input == "<INTERRUPTED>":
                continue

            if not user_input:
                continue
            if user_input.startswith("/"):
                self.commands.execute(user_input, self)
                continue

            self.history.add_message(HumanMessage(content=user_input))

            iteration = 0
            executed_tool_calls = set()

            while True:
                if iteration >= MAX_TOOL_CALLS:
                    self.history.add_message(
                        SystemMessage(
                            content=(
                                "You have reached the maximum search limit. Based on the tool results "
                                "provided in the history above, write your final response now in STRICT Org mode. "
                            )
                        )
                    )

                    final_response = clean_llm.invoke(self.history.messages)
                    clean_content = self.strip_thought(final_response.content)

                    self.history.add_message(AIMessage(content=clean_content))
                    print(f"Agent: {clean_content}", flush=True)
                    self._save_to_md(user_input, clean_content)

                    # Call the updated _speak_response
                    self._speak_response(clean_content, clean_llm)
                    break

                print(f"🤔 Thinking (Step {iteration + 1})...", end="\r", flush=True)
                calls_left = MAX_TOOL_CALLS - iteration
                reminder_msg = SystemMessage(
                    content=f"SYSTEM REMINDER: You have {calls_left} tool call(s) remaining for this request. "
                    f"If you have enough information, respond with your final answer to the user."
                )
                messages_to_send = self.history.messages + [reminder_msg]
                response_msg = self.llm.invoke(messages_to_send)

                # CASE: Final Answer
                if not response_msg.tool_calls:
                    clean_content = self.strip_thought(response_msg.content)
                    if not clean_content:
                        iteration = MAX_TOOL_CALLS
                        continue

                    self.history.add_message(AIMessage(content=clean_content))
                    print(" " * 40, end="\r", flush=True)
                    print(f"Agent: {clean_content}", flush=True)
                    self._save_to_md(user_input, clean_content)

                    # Call the updated _speak_response
                    self._speak_response(clean_content, clean_llm)
                    break

                # CASE: Tool Calls
                self.history.add_message(response_msg)
                for tc in response_msg.tool_calls:
                    call_sig = f"{tc['name']}({str(tc['args'])})"
                    if call_sig in executed_tool_calls:
                        result = "Error: Duplicate search. You already have this info."
                    elif not self.tools.has_tool(tc["name"]):
                        available = ", ".join(self.tools.get_tool_names())
                        result = f"Error: Tool '{tc['name']}' unknown. Available: [{available}]."
                    else:
                        result = self.tools.execute(tc, self)
                        executed_tool_calls.add(call_sig)

                    self.history.add_message(
                        ToolMessage(
                            content=str(result), tool_call_id=tc["id"], name=tc["name"]
                        )
                    )

                iteration += 1
