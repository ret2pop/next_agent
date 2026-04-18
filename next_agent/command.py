from langchain_ollama import ChatOllama

class CommandRegistry:
    """A modular way to add, store, and execute slash commands."""
    def __init__(self):
        self.commands = {}

    def register(self, trigger: str, description: str):
        def decorator(func):
            self.commands[trigger] = {"func": func, "desc": description}
            return func
        return decorator

    def execute(self, command_string: str, agent_instance) -> bool:
        parts = command_string.strip().split()
        if not parts:
            return False
            
        trigger = parts[0].lower()
        if trigger in self.commands:
            self.commands[trigger]["func"](agent_instance, parts[1:])
            return True
        elif trigger.startswith("/"):
            print(f"⚠️ Unknown command: {trigger}. Type /help to see commands.")
            return True
            
        return False

# --- Global Registry Instance ---
command_registry = CommandRegistry()

# --- Command Implementations ---
@command_registry.register("/help", "List all available commands")
def cmd_help(agent, args):
    print("\n🛠️  Available Commands:")
    for trigger, data in agent.commands.commands.items():
        print(f"  {trigger:<15} - {data['desc']}")

@command_registry.register("/rebuild", "Rebuild the monorepo vector database")
def cmd_rebuild(agent, args):
    agent.monorepo_rag.rebuild_index()
    agent.agent_rag.rebuild_index()
    agent.email_rag.rebuild_index()

@command_registry.register("/memory", "Show the current conversation history")
def cmd_memory(agent, args):
    print("\n🧠 Current Memory State:")
    if not agent.history.messages:
        print("  (Empty)")
    for msg in agent.history.messages:
        # Skip printing the system message to keep it clean
        if msg.type == "system": continue
        print(f"  [{msg.type.upper()}]: {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")

@command_registry.register("/clear", "Clear the current conversation history")
def cmd_clear(agent, args):
    agent.history.clear()
    # Re-inject the system prompt after clearing
    agent._inject_system_prompt()
    print("\n🧹 Memory cleared.")

@command_registry.register("/quit", "Save memory and exit the agent")
def cmd_quit(agent, args):
    agent.distill_and_exit()

from .vars import DEFAULT_MODEL
from langchain_ollama import ChatOllama

@command_registry.register("/reload", "Reload the agent's tools and system prompt")
def cmd_reload(agent, args):
    print("🔄 Reloading tool registry and system instructions...", flush=True)
    
    # 1. Re-scan the tools directory
    agent.tools.load_tools() 
    
    # 2. Re-bind the updated schemas to the LLM
    agent.llm = ChatOllama(model=DEFAULT_MODEL, temperature=0)
    if agent.tools.schemas:
        agent.llm = agent.llm.bind_tools(agent.tools.schemas)
        
    # 3. Refresh system prompt so it learns the new instructions
    agent._inject_system_prompt()
    
    print("✅ Reload complete. New tools are now live.", flush=True)
