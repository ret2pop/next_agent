import re
import os
import hashlib
import subprocess
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from .vars import DEFAULT_MODEL, REPO_PATH, DB_PATH, MAX_TOOL_CALLS, MEMORY_FILE, MD_DIR, SRC_ROOT, AGENT_ROOT, EMAIL_ROOT
from .memory import CodebaseRAG
from .tool import tool_registry
from .command import command_registry

# --- NEW IMPORTS ---
from .agent_base import Agent
from .tts import KokoroProvider

class LocalAgent(Agent):
    def __init__(self):
        tts = KokoroProvider() 
        llm = ChatOllama(model=DEFAULT_MODEL, temperature=0)

        super().__init__(
            llm=llm,
            tools=tool_registry, 
            commands=command_registry, 
            tts_provider=tts,
        )
        
        self.monorepo_rag = CodebaseRAG(repo_path=REPO_PATH, db_path=DB_PATH, index_name="monorepo_index")
        self.agent_rag = CodebaseRAG(repo_path=AGENT_ROOT, db_path=DB_PATH, index_name="agent_index")
        self.email_rag = CodebaseRAG(repo_path=EMAIL_ROOT, db_path=DB_PATH, index_name="email_index", valid_exts=None, email_store = True)
        self.voice = False
        
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
        
        if self.history.messages and isinstance(self.history.messages[0], SystemMessage):
            self.history.messages[0] = SystemMessage(content=prompt_content)
        elif not self.history.messages:
            self.history.add_message(SystemMessage(content=prompt_content))
        else:
            self.history.messages.insert(0, SystemMessage(content=prompt_content))

    def strip_thought(self, text):
        if not text: return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _open_in_emacs(self, filepath):
        abs_path = os.path.abspath(filepath)
        print(f"\n[DEBUG] 🚀 Attempting to open: {abs_path}", flush=True)
        try:
            cmd = f"emacsclient -n '{abs_path}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ File opened in Emacs buffer.", flush=True)
            else:
                print(f"⚠️ emacsclient returned error: {result.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"❌ Failed to execute emacsclient: {e}", flush=True)

    def _save_to_md(self, user_input, agent_response):
        first_save = False
        if not self.session_file:
            first_save = True
            hash_input = (user_input + agent_response).encode('utf-8')
            guid = hashlib.sha256(hash_input).hexdigest()[:12]
            self.session_file = os.path.join(MD_DIR, f"session_{guid}.md")
            with open(self.session_file, "w", encoding="utf-8") as f:
                f.write(f"# Session {guid}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(f"## User Prompt\n{user_input}\n\n")
            f.write(f"## Agent Response\n{agent_response}\n\n")
            f.write("---\n\n")

        if first_save:
            self._open_in_emacs(self.session_file)

    def _speak_response(self, text: str, clean_llm: ChatOllama):
        """Passes the raw Org-mode text through an LLM to normalize it for TTS."""
        if not self.tts_provider or not self.voice:
            return

        print("🎙️ Normalizing text for speech...", flush=True)
        
        system_msg = SystemMessage(content=(
            "You are a warm, highly human, and engaging text-to-speech scriptwriter. "
            "Your job is to translate written assistant responses into natural, conversational spoken English. "
            "Rules:\n"
            "1. Strip entirely all Org-mode/Markdown formatting, URLs, and code blocks (do not try to read code aloud).\n"
            "2. Spell out numbers, symbols, and currencies smoothly and naturally (e.g., '$10' becomes 'ten dollars').\n"
            "3. Ensure the phrasing sounds warm and human when spoken, rather than robotic or stiff.\n"
            "4. Output ONLY the final spoken script. Do not include any meta-introductions or filler like 'Here is the converted text'."
        ))
        
        user_msg = HumanMessage(content=f"Convert the following:\n\n{text}")
        
        tts_response = clean_llm.invoke([system_msg, user_msg])
        spoken_text = self.strip_thought(tts_response.content)
        
        self.tts_provider.generate_audio(spoken_text)

    def distill_and_exit(self):
        print("\n🧠 Distilling context...", flush=True)
        print("Goodbye!", flush=True)
        exit(0)

    def start(self):
        print(f"\n🤖 NextAgent Ready.", flush=True)
        print("-" * 60, flush=True)
        
        clean_llm = ChatOllama(model=DEFAULT_MODEL, temperature=0)

        while True:
            try:
                user_input = pt_prompt("\nYou: ", history=self.cli_history, multiline=True, 
                                       key_bindings=self.kb, prompt_continuation="... ").strip()
            except (KeyboardInterrupt, EOFError):
                self.distill_and_exit()
                
            if not user_input: continue
            if user_input.startswith("/"):
                self.commands.execute(user_input, self)
                continue

            self.history.add_message(HumanMessage(content=user_input))
            
            iteration = 0
            executed_tool_calls = set() 

            while True:
                if iteration >= MAX_TOOL_CALLS:
                    self.history.add_message(SystemMessage(content=(
                        "You have reached the maximum search limit. Based on the tool results "
                        "provided in the history above, write your final response now in STRICT Org mode. "
                    )))
                    
                    final_response = clean_llm.invoke(self.history.messages)
                    clean_content = self.strip_thought(final_response.content)
                    
                    self.history.add_message(AIMessage(content=clean_content))
                    print(f"Agent: {clean_content}", flush=True)
                    self._save_to_md(user_input, clean_content)
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
                    self._speak_response(clean_content, clean_llm)
                    break

                # CASE: Tool Calls
                self.history.add_message(response_msg)
                for tc in response_msg.tool_calls:
                    call_sig = f"{tc['name']}({str(tc['args'])})"
                    if call_sig in executed_tool_calls:
                        result = "Error: Duplicate search. You already have this info."
                    elif not self.tools.has_tool(tc['name']):
                        available = ", ".join(self.tools.get_tool_names())
                        result = f"Error: Tool '{tc['name']}' unknown. Available: [{available}]."
                    else:
                        result = self.tools.execute(tc, self)
                        executed_tool_calls.add(call_sig)

                    self.history.add_message(ToolMessage(content=str(result), tool_call_id=tc['id'], name=tc['name']))
                
                iteration += 1
