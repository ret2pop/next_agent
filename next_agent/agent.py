import re
import os
import hashlib
import subprocess
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings # NEW IMPORT

from .vars import DEFAULT_MODEL, REPO_PATH, DB_PATH, MAX_TOOL_CALLS, MEMORY_FILE, MD_DIR, SRC_ROOT, AGENT_ROOT, EMAIL_ROOT
from .memory import CodebaseRAG
from .tool import tool_registry
from .command import command_registry

class LocalAgent:
    def __init__(self):
        self.monorepo_rag = CodebaseRAG(repo_path=REPO_PATH, db_path=DB_PATH, index_name="monorepo_index")
        self.agent_rag = CodebaseRAG(repo_path=AGENT_ROOT, db_path=DB_PATH, index_name="agent_index")
        self.email_rag = CodebaseRAG(repo_path=EMAIL_ROOT, db_path=DB_PATH, index_name="email_index", valid_exts=None)
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
            f"- PROJECT SOURCE ROOT: `{SRC_ROOT}` (Where other individual repositories live).\n"
            f"- YOUR CORE (AGENT_ROOT): `{AGENT_ROOT}` (Where your own tools, cache, and logic reside).\n"

            f"=== TEMPORAL ANCHORING ===\nCurrent Time: {current_time}.\n\n"
            f"=== USER CONTEXT ===\n{ltm_content}\n\n"
            "=== THE TOOLS ===\n"
            f"{tool_instructions}\n\n"

            "=== FORMATTING RULES: MARKDOWN, LATEX, & CITATIONS ===\n"
            "1. STRUCTURE: Use standard Markdown (# for H1, ## for H2, etc.).\n"
            "2. LATEX: For math/science, use $$ (e.g., $$E=mc^2$$).\n"
            "3. INLINE CITATIONS: This is critical. Whenever you use information from a tool "
            "result (RAG or Web Search), you MUST cite it inline.\n"
            "   - For local files: Use the filename as a link to its path (e.g., `[main.py](~/monorepo/main.py)`).\n"
            "   - For web results: Use the title as a link to the URL (e.g., `[NixOS Wiki](https://nixos.wiki/...)`).\n"
            "4. CODE: Use fenced blocks (```python).\n\n"

            "=== OPERATIONAL PROTOCOL ===\n"
            "- Start with a Markdown heading.\n"
            "- If multiple sources contradict, mention both and cite them.\n"
            "- Maintain your warm, Chief-of-Staff persona."
        )
        
        if self.history.messages and isinstance(self.history.messages[0], SystemMessage):
            # Update existing system prompt
            self.history.messages[0] = SystemMessage(content=prompt_content)
        elif not self.history.messages:
            # If history is completely empty, standard add_message works
            self.history.add_message(SystemMessage(content=prompt_content))
        else:
            # If there are messages but no system prompt, insert into the underlying list
            self.history.messages.insert(0, SystemMessage(content=prompt_content))

    def strip_thought(self, text):
        if not text: return ""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _open_in_emacs(self, filepath):
        """Attempts to open the newly created file in your existing Emacs session."""
        # Ensure path is absolute and shell-escaped
        abs_path = os.path.abspath(filepath)
        
        print(f"\n[DEBUG] 🚀 Attempting to open: {abs_path}", flush=True)
        
        try:
            # We use shell=True so it picks up your shell's PATH and environment
            # We also capture output to see the actual error if it fails
            cmd = f"emacsclient -n '{abs_path}'"
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ File opened in Emacs buffer.", flush=True)
            else:
                print(f"⚠️ emacsclient returned error: {result.stderr.strip()}", flush=True)
                print(f"💡 Hint: Is the Emacs server running? (M-x server-start)", flush=True)
                
        except Exception as e:
            print(f"❌ Failed to execute emacsclient: {e}", flush=True)

    def _save_to_md(self, user_input, agent_response):
        """Saves interaction to a .md file."""
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

    def distill_and_exit(self):
        print("\n🧠 Distilling context...", flush=True)
        # (Simplified exit logic for brevity)
        print("Goodbye!", flush=True)
        exit(0)

    def start(self):
        print(f"\n🤖 NextAgent Ready.", flush=True)
        print("-" * 60, flush=True)
        
        # Create a 'Clean' LLM instance that HAS NO TOOLS BOUND
        # This is used for the final summary to prevent the model from asking for more tools.
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
                # 1. Check if we have exceeded our 'Search turns'
                if iteration >= MAX_TOOL_CALLS:
                    print(f"\n[!] Search limit reached. Synthesizing final Org-mode response...", flush=True)
                    
                    # We inject a final prompt and use the 'clean_llm' which has no tools.
                    # This forces the model to use the data it already has.
                    self.history.add_message(SystemMessage(content=(
                        "You have reached the maximum search limit. Based on the tool results "
                        "provided in the history above, write your final response now in STRICT Org mode. "
                        "Do not mention the tool limit to the user; just provide the summary."
                    )))
                    
                    final_response = clean_llm.invoke(self.history.messages)
                    clean_content = self.strip_thought(final_response.content)
                    
                    self.history.add_message(AIMessage(content=clean_content))
                    print(f"Agent: {clean_content}", flush=True)
                    self._save_to_md(user_input, clean_content)
                    break

                # 2. Standard reasoning turn
                print(f"🤔 Thinking (Step {iteration + 1})...", end="\r", flush=True)
                calls_left = MAX_TOOL_CALLS - iteration
                reminder_msg = SystemMessage(
                    content=f"SYSTEM REMINDER: You have {calls_left} tool call(s) remaining for this request. "
                            f"If you have enough information, respond with your final answer to the user."
                )
                messages_to_send = self.history.messages + [reminder_msg]
                response_msg = self.llm.invoke(messages_to_send)
                
                # CASE: The model gives a final text answer
                if not response_msg.tool_calls:
                    clean_content = self.strip_thought(response_msg.content)
                    
                    # If the model is empty for some reason, try to force a response
                    if not clean_content:
                        iteration = MAX_TOOL_CALLS
                        continue

                    self.history.add_message(AIMessage(content=clean_content))
                    print(" " * 40, end="\r", flush=True) 
                    print(f"Agent: {clean_content}", flush=True)
                    self._save_to_md(user_input, clean_content)
                    break

                # CASE: The model requests tools
                self.history.add_message(response_msg)
                
                for tc in response_msg.tool_calls:
                    call_sig = f"{tc['name']}({str(tc['args'])})"
                    
                    if call_sig in executed_tool_calls:
                        result = "Error: Duplicate search. You already have this info. Please summarize or try a different query."
                    
                    # --- FIXED HALLUCINATION CHECK ---
                    elif not self.tools.has_tool(tc['name']):
                        available = ", ".join(self.tools.get_tool_names())
                        result = f"Error: Tool '{tc['name']}' unknown. Available: [{available}]."
                    # ---------------------------------
                    
                    else:
                        result = self.tools.execute(tc, self)
                        executed_tool_calls.add(call_sig)

                    self.history.add_message(ToolMessage(content=str(result), tool_call_id=tc['id'], name=tc['name']))
                
                iteration += 1
