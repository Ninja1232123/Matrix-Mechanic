# Codes-Masterpiece

A complete developer debugging and code intelligence ecosystem. Nine integrated tools that fix bugs, understand code, and make you better.

## The Arsenal

| Tool | What It Does |
|------|--------------|
| **Universal Debugger** | Auto-fix 50+ Python error types in seconds |
| **AI Debug Companion** | TUI that watches your terminal and suggests fixes in real-time |
| **Type-Guardian** | Auto-fix type errors, add type hints |
| **CodeSeek** | Semantic code search - ask questions in natural language |
| **DevKnowledge** | Local-first knowledge graph with auto-linking |
| **DevMaster** | Unified CLI that orchestrates all tools + learns your coding style |
| **CodeArchaeology** | Dig through git history, find hotspots, understand evolution |
| **Deploy-Shield** | Validate deployments before they fail |
| **DevNarrative** | Generate changelogs and release notes from commits |

## Quick Start

```bash
git clone https://github.com/Ninja1232123/Codes-Masterpiece
cd Codes-Masterpiece

# Try the auto-fixer
python demo_wow.py

# Install everything
./install_all.sh  # or install_all.bat on Windows
```

## Universal Debugger

The core. Fixes 50+ Python error types automatically.

```bash
# Watch it work
python demo_wow.py

# Three modes
DEBUG_MODE=development python mode_aware_debugger.py script.py  # Learn
DEBUG_MODE=review python mode_aware_debugger.py script.py       # Approve each fix
DEBUG_MODE=production python mode_aware_debugger.py script.py   # Auto-fix all
```

## DevMaster - The Orchestrator

One CLI to access everything:

```bash
pip install -e devmaster

devmaster status          # See what's available
devmaster analyze         # Run analysis
devmaster learn start .   # Start learning your coding style
devmaster learn profile   # See your coding profile
devmaster learn tip       # Get personalized improvement tips
```

## CodeSeek - Semantic Search

Search code with natural language:

```bash
pip install -e codeseek

codeseek index .                           # Index your codebase
codeseek search "functions that validate"  # Natural language search
codeseek similar path/to/file.py           # Find similar code
```

## Type-Guardian

Fix type errors automatically:

```bash
pip install -e type-guardian

type-guardian fix src/           # Auto-fix type errors
type-guardian annotate module.py # Add type hints
type-guardian check .            # Run type checking
```

## Deploy-Shield

Catch deployment issues before production:

```bash
pip install -e deploy-shield

deploy-shield validate           # Full validation
deploy-shield check-env          # Environment variables
deploy-shield test-connections   # Database/API connections
```

## Documentation

Each tool has its own README:
- [AI Debug Companion](ai-debug-companion/README.md)
- [CodeSeek](codeseek/README.md)
- [DevKnowledge](devknowledge/README.md)
- [Type-Guardian](type-guardian/README.md)
- [Deploy-Shield](deploy-shield/README.md)
- [CodeArchaeology](codearchaeology/README.md)
- [DevMaster](devmaster/README.md)

## The Nervous System

The tools aren't just separate utilities - they're connected through a central event bus.

```bash
devmaster nerve status        # See system activity
devmaster nerve integrations  # View how tools connect
devmaster nerve flow          # Visualize event flow
devmaster nerve watch         # Files flagged for attention
devmaster nerve suggestions   # Error patterns to add to debugger
```

**How it works:**
- CodeArchaeology finds a hotspot → Universal Debugger adds it to watchlist
- DevMaster learner detects recurring error → Suggests new debugger pattern
- CodeSeek indexes code → DevKnowledge links documentation
- Deploy-Shield catches failure → AI Debug Companion analyzes it

One nervous system. Nine tools. All connected.

## Architecture

```
Codes-Masterpiece/
├── universal_debugger.py    # Core auto-fixer (60K lines)
├── ai-debug-companion/      # Real-time TUI debugger
├── type-guardian/           # Type error fixer
├── codeseek/               # Semantic code search
├── devknowledge/           # Knowledge graph
├── devmaster/              # Unified CLI + coding coach + nervous system
├── codearchaeology/        # Git history analysis
├── deploy-shield/          # Deployment validation
└── devnarrative/           # Changelog generation
```

## The Philosophy

No AI guessing. Hard-coded solutions that work 100% of the time for known patterns. AI assistance only where pattern matching can't reach.

---

**Never debug the same error twice.**
