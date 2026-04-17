import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
REPO_PATH = os.path.expanduser(os.getenv("REPO_PATH", "~/monorepo"))
DB_PATH = os.path.expanduser(os.getenv("DB_PATH", "~/.cache/next_agent"))
TOOLS_DIR = os.getenv("TOOLS_DIR", "./tools")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "3"))
MEMORY_FILE = os.path.expanduser("~/.cache/long_term_memory.txt")
ORG_DIR = os.path.expanduser("~/org/next_agent")
AGENDA_FILE = os.path.expanduser("~/org/agenda.org")
