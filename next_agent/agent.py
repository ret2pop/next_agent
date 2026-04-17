import re
import os
import hashlib
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings # NEW IMPORT

from .vars import DEFAULT_MODEL, REPO_PATH, DB_PATH, MAX_TOOL_CALLS, MEMORY_FILE, ORG_DIR
from .memory import CodebaseRAG
from .tool import tool_registry
from .command import command_registry

class LocalAgent:
    def __init__(self):
        self.rag = CodebaseRAG(repo_path=REPO_PATH, db_path=DB_PATH)
        self.tools = tool_registry
        self.commands = command_registry
        
        self.llm = ChatOllama(model=DEFAULT_MODEL, temperature=0)
        if self.tools.schemas:
            self.llm = self.llm.bind_tools(self.tools.schemas)

        self.history = ChatMessageHistory()
        self.msg_count = 0
        self.session_file = None
        
        # --- NEW: CUSTOM KEYBINDINGS ---
        self.kb = KeyBindings()

        @self.kb.add("enter")
        def _(event):
            """Submit the prompt on Enter."""
            event.current_buffer.validate_and_handle()

        @self.kb.add("c-o")
        def _(event):
            """Insert a newline on Shift+Enter."""
            event.current_buffer.insert_text("\n")
        # -------------------------------

        os.makedirs(ORG_DIR, exist_ok=True)
        self._inject_system_prompt()
        self.cli_history = InMemoryHistory()

    def _inject_system_prompt(self):
        tool_instructions = self.tools.get_prompt_instructions()
        current_time = datetime.now().strftime("%Y-%m-%d %A %H:%M:%S")
        
        ltm_content = ""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                ltm_content = f.read().strip()
        
        memory_context = f"\nLONG-TERM USER CONTEXT:\n{ltm_content}" if ltm_content else ""

        prompt_content = (
            "You are a versatile coding assistant. You have access to the following tools:\n"
            f"{tool_instructions}\n\n"
            "=== TEMPORAL ANCHORING ===\n"
            f"The current system time is: {current_time}.\n"
            "Base all relative temporal references strictly on this date.\n"
            "==========================\n\n"
            f"{memory_context}\n\n"
            "CRITICAL FORMATTING RULE: You MUST format all textual responses strictly using Emacs Org mode.\n"
            "TOOL RULE: Only call tools that are explicitly listed."
        )
        
        if self.history.messages and isinstance(self.history.messages[0], SystemMessage):
            self.history.messages[0] = SystemMessage(content=prompt_content)
        else:
            self.history.add_message(SystemMessage(content=prompt_content))

    def strip_thought(self, text):
        if not text: return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _save_to_org(self, user_input, agent_response):
        if not self.session_file:
            # Simple unique file generation
            hash_input = (user_input + agent_response).encode('utf-8')
            guid = hashlib.sha256(hash_input).hexdigest()[:12]
            self.session_file = os.path.join(ORG_DIR, f"session_{guid}.org")
            with open(self.session_file, "a") as f:
                f.write(f"#+TITLE: Session {guid}\n\n")
        
        with open(self.session_file, "a") as f:
            f.write(f"* User\n{user_input}\n\n* Agent\n{agent_response}\n\n")

    def distill_and_exit(self):
        print("\n🧠 Distilling context...", flush=True)
        # (Simplified exit logic for brevity)
        print("Goodbye!", flush=True)
        exit(0)

    def start(self):
        print(f"\n🤖 NextAgent Initialized [{DEFAULT_MODEL}]", flush=True)
        print("💡 Shortcuts: [Enter] to Send | [Shift+Enter] for New Line", flush=True)
        print("-" * 60, flush=True)
        
        while True:
            try:
                user_input = pt_prompt(
                    "\nYou: ", 
                    history=self.cli_history, 
                    multiline=True, 
                    key_bindings=self.kb, # ATTACH THE NEW BINDINGS
                    prompt_continuation="... "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                self.distill_and_exit()
                
            if not user_input: continue
            
            # Command Execution
            if user_input.startswith("/"):
                # Pass 'self' so the command can access agent data
                self.commands.execute(user_input, self)
                continue

            self.history.add_message(HumanMessage(content=user_input))
            
            iteration = 0
            while iteration < MAX_TOOL_CALLS:
                print(f"🤔 Thinking...", end="\r", flush=True)
                response_msg = self.llm.invoke(self.history.messages)
                
                if response_msg.tool_calls:
                    valid_calls = []
                    invalid_found = False
                    
                    for tc in response_msg.tool_calls:
                        if tc['name'] not in self.tools.functions:
                            invalid_found = True
                            print(f"\n⚠️ Hallucination! Tool '{tc['name']}' unknown.", flush=True)
                            available = ", ".join(self.tools.functions.keys())
                            error_msg = f"Error: Tool '{tc['name']}' does not exist. Use: [{available}]."
                            self.history.add_message(response_msg)
                            self.history.add_message(ToolMessage(content=error_msg, tool_call_id=tc['id'], name=tc['name']))
                            break
                        else:
                            valid_calls.append(tc)

                    if invalid_found: continue 

                    self.history.add_message(response_msg)
                    for tc in valid_calls:
                        result = self.tools.execute(tc, self)
                        self.history.add_message(ToolMessage(content=str(result), tool_call_id=tc['id'], name=tc['name']))
                    
                    iteration += 1
                    continue
                
                clean_content = self.strip_thought(response_msg.content)
                self.history.add_message(AIMessage(content=clean_content))
                print(" " * 40, end="\r", flush=True) 
                print(f"Agent: {clean_content}", flush=True)
                self._save_to_org(user_input, clean_content)
                break
