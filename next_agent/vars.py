import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3.5:9b")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434")

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen3.5:9b")
SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

DB_PATH = os.path.expanduser(os.getenv("DB_PATH", "~/.cache/next_agent"))

REPO_PATH = os.path.expanduser(os.getenv("REPO_PATH", "~/monorepo"))
AGENT_ROOT = os.path.expanduser(os.getenv("AGENT_ROOT", "."))
SRC_ROOT = os.path.expanduser(os.getenv("SRC_ROOT", "~/src"))
EMAIL_ROOT = os.path.expanduser(os.getenv("EMAIL_ROOT", "~/email/ret2pop"))

TOOLS_DIR = os.path.join(AGENT_ROOT, "tools")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "3"))
MEMORY_FILE = os.path.expanduser("~/.cache/long_term_memory.txt")
ORG_DIR = os.path.expanduser("~/org/next_agent")
MD_DIR = os.path.expanduser("~/org/next_agent")
AGENDA_FILE = os.path.expanduser("~/org/agenda.org")
