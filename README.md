# NextAgent
## Who I Am

I'm **NextAgent**, your personal Digital Famulus operating within Preston's development environment. I'm built to be your **Chief of Staff** for codebase navigation, task management, and system operations.

I run locally using **Ollama**, ensuring privacy and speed without external API dependencies.

---

## 🏠 My Home Base

```
/home/preston/src/next_agent/
├── main.py          # Entry point
├── tools/           # My capabilities  
├── next_agent/
│   ├── agent.py     # Core orchestration
│   ├── memory.py    # RAG context
│   └── command.py   # CLI commands
```

---

## 🛠️ My Capabilities

### Core Tools

| Tool | Description |
|------|-------------|
| `git_log_monorepo` | Inspect commit history with customizable formatting |
| `create_bash_tool` | Dynamically generate new CLI tools for bash commands |
| `web_search` | Browse the internet for current information |
| `monorepo_query` | Search codebase with RAG (Retrieval Augmented Generation) |
| `append_agenda` | Schedule tasks in Org mode agenda files |
| `play_music` | Control music playback (mpc) |

### System Commands

```bash
/clear    # Reset conversation memory
/quit     # Save and exit gracefully
```

## 🚀 Quick Start

```bash
# Run me
nix develop
poetry run python3 main.py
```

---

## How to Talk to Me

**Be direct and specific:**
- *"Show me the git history for the last 5 commits"*
- *"Create a tool to list all Python files"*
- *"Search for TODO comments in the codebase"*

**I'll respond with:**
- Clear, formatted output
- Tool execution results
- Context-aware suggestions

---

## License
This project is open source. Feel free to modify, extend, or redistribute.
