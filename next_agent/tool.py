import os
import json
from .vars import TOOLS_DIR, AGENDA_FILE
from .search import TavilyProvider
import subprocess
import shlex

class ToolRegistry:
    def __init__(self):
        self.explicit_functions = {}
        self.schemas = []
        self.load_tools()

    def register(self, name):
        """Decorator to register a pure Python tool."""
        def decorator(func):
            self.explicit_functions[name] = func
            return func
        return decorator

    def load_tools(self):
        """Scans the tools directory to build the schema list."""
        self.schemas = []
        if not os.path.exists(TOOLS_DIR):
            os.makedirs(TOOLS_DIR)
            
        for filename in os.listdir(TOOLS_DIR):
            if filename.endswith(".json"):
                with open(os.path.join(TOOLS_DIR, filename), "r") as f:
                    try:
                        config = json.load(f)
                        # We keep the schema so the LLM knows the tool exists
                        self.schemas.append(config)
                    except json.JSONDecodeError:
                        print(f"⚠️ Failed to parse {filename}")

    def execute(self, tool_call, agent_instance):
        name = tool_call["name"]
        args = tool_call["args"]

        # 1. Check if there is an explicit Python function registered
        if name in self.explicit_functions:
            return self.explicit_functions[name](agent_instance, **args)

        # 2. Otherwise, look for a declarative bash template in the schemas
        for schema in self.schemas:
            if schema["function"]["name"] == name and "bash_template" in schema:
                return self._run_bash_template(schema["bash_template"], args)

        return f"Error: No implementation found for tool '{name}'."

    def _run_bash_template(self, template, args):
        """Executes a declarative bash command with strict shell escaping."""
        try:
            cmd = template
            for key, value in args.items():
                target = "{{{" + key + "}}}"
                
                safe_value = shlex.quote(str(value)) 
                
                cmd = cmd.replace(target, safe_value)
            
            if "{{{" in cmd and "}}}" in cmd:
                print(f"⚠️ Warning: Unreplaced variables detected in bash command:\n{cmd}", flush=True)

            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return res.stdout if res.returncode == 0 else f"Error: {res.stderr}"
        except Exception as e:
            return f"Fault: {str(e)}"

    def get_prompt_instructions(self) -> str:
        """
        Formats the currently loaded tool schemas into a clear instruction 
        block for the LLM's system prompt.
        """
        if not self.schemas:
            return "No tools are currently available."

        instructions = []
        for schema in self.schemas:
            func = schema.get("function", {})
            name = func.get("name")
            desc = func.get("description")
            params = func.get("parameters", {}).get("properties", {})
            
            instr = f"- Tool: {name}\n  Description: {desc}\n  Parameters: {list(params.keys())}"
            instructions.append(instr)

        return "\n".join(instructions)

    def get_tool_names(self) -> list:
        """Returns a list of all tool names currently known to the LLM."""
        return [schema["function"]["name"] for schema in self.schemas if "function" in schema]

    def has_tool(self, name: str) -> bool:
        """Checks if a tool exists in the loaded schemas."""
        return name in self.get_tool_names()

# Instantiate the singleton
tool_registry = ToolRegistry()

# --- Tool Implementations ---
@tool_registry.register("monorepo_query")
def execute_monorepo_query(agent, query: str) -> str:
    return agent.monorepo_rag.search(query)

@tool_registry.register("agent_code_query")
def execute_agent_code_query(agent, query: str):
    """Search the agent's own source code to understand its architecture."""
    return agent.agent_rag.search(query)

@tool_registry.register("email_query")
def execute_email_query(agent, query: str):
    """Search the agent's own source code to understand its architecture."""
    return agent.email_rag.search(query)

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
def execute_create_bash_tool(agent, **kwargs) -> str:
    """
    Constructs a declarative JSON tool manifest in the TOOLS_DIR.
    Uses kwargs to gracefully handle schema naming updates.
    """
    try:
        tool_name = kwargs.get("tool_name")
        description = kwargs.get("description")
        
        # Safely catch either the new name or the old name
        bash_template = kwargs.get("bash_template") or kwargs.get("bash_command_template")
        
        # Safely handle either a dictionary (new style) or a JSON string (old style)
        properties = kwargs.get("properties")
        if properties is None and "parameters_json" in kwargs:
            import json
            properties = json.loads(kwargs["parameters_json"])

        if not all([tool_name, description, bash_template, properties]):
            return "❌ Error: Missing required arguments. I need tool_name, description, bash_template, and properties."

        # Ensure the tools directory exists relative to AGENT_ROOT
        os.makedirs(TOOLS_DIR, exist_ok=True)
        
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
        
        # Absolute path based on AGENT_ROOT
        manifest_path = os.path.join(TOOLS_DIR, f"{tool_name}.json")
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            import json
            json.dump(manifest, f, indent=2)
            
        return f"✅ Manifest for '{tool_name}' written to {manifest_path}. Please run /reload to initialize."
    except Exception as e:
        return f"❌ Failed to create tool manifest: {str(e)}"
