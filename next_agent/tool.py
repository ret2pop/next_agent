import os
import json
import glob
from .vars import TOOLS_DIR, AGENDA_FILE
from .search import TavilyProvider

class ToolRegistry:
    def __init__(self, tools_dir: str):
        self.tools_dir = tools_dir
        self.schemas = []
        self.instructions = [] # Store the extracted prompt instructions
        self.functions = {}
        self._load_schemas()

    def _load_schemas(self):
        if not os.path.exists(self.tools_dir):
            os.makedirs(self.tools_dir, exist_ok=True)
            return

        for filepath in glob.glob(os.path.join(self.tools_dir, "*.json")):
            with open(filepath, 'r') as f:
                try:
                    schema = json.load(f)
                    
                    # 1. Extract the custom instruction and name
                    name = schema.get("function", {}).get("name", "unknown")
                    instruction = schema.pop("agent_instruction", f"Use this tool when you need to execute {name}.")
                    
                    # 2. Store the formatted instruction for the System Prompt
                    self.instructions.append(f"`{name}`: {instruction}")
                    
                    # 3. Append the sanitized schema for LangChain
                    self.schemas.append(schema)
                    print(f"🔧 Loaded tool schema: {name}")
                except json.JSONDecodeError:
                    print(f"⚠️ Error parsing {filepath}")

    def get_prompt_instructions(self) -> str:
        """Returns a numbered, formatted string of all tool instructions."""
        if not self.instructions:
            return "You have no additional tools available."
        
        return "\n".join([f"{i+1}. {inst}" for i, inst in enumerate(self.instructions)])

    def register(self, tool_name: str):
        def decorator(func):
            self.functions[tool_name] = func
            return func
        return decorator

    def execute(self, tool_call, agent_instance) -> str:
        name = tool_call["name"]
        args = tool_call["args"]
        
        if name in self.functions:
            print(f"\n🛠️  Executing tool: {name}")
            return self.functions[name](agent_instance, **args)
        return f"Error: Tool '{name}' not found."


# --- Global Registry Instance ---
tool_registry = ToolRegistry(tools_dir=TOOLS_DIR)

# --- Tool Implementations ---
@tool_registry.register("monorepo_query")
def execute_monorepo_query(agent, query: str) -> str:
    return agent.rag.search(query)

@tool_registry.register("web_search")
def execute_web_search(agent, query: str) -> str:
    try:
        provider = TavilyProvider()
        return provider.search(query, num_results=10)
    except Exception as e:
        return str(e)

@tool_registry.register("append_agenda")
def execute_append_agenda(agent, task_name: str, scheduled_date: str, description: str) -> str:
    """Appends a correctly formatted Org-mode scheduled item to the agenda file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(AGENDA_FILE), exist_ok=True)
        
        with open(AGENDA_FILE, "a", encoding="utf-8") as f:
            f.write(f"* {task_name}\n")
            f.write(f"SCHEDULED: <{scheduled_date}>\n")
            f.write(f"{description}\n\n")
            
        return f"Success: Task '{task_name}' added to agenda for {scheduled_date}."
    except Exception as e:
        return f"Error: Failed to write to agenda. Details: {str(e)}"

@tool_registry.register("create_bash_tool")
def execute_create_bash_tool(agent, tool_name, description, bash_template, properties):
    """Preston's tool-creation tool: Just writes a JSON manifest."""
    manifest = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys())
            }
        },
        "bash_template": bash_template
    }
    
    path = os.path.join(TOOLS_DIR, f"{tool_name}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    return f"Success: Manifest created for {tool_name}. Run /reload to initialize."
