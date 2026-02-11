# AI Debugging Companion

An intelligent TUI tool that watches your development session, analyzes errors in real-time, and suggests fixes based on your codebase history and common patterns.

## Features

### Core Capabilities
- **Real-time Error Monitoring**: Watches terminal output, log files, and test results for errors
- **Smart Error Parsing**: Extracts structured information from error messages across multiple languages
- **Historical Context**: Searches git history for similar errors and successful fixes
- **Intelligent Suggestions**: Provides fix suggestions based on patterns, history, and context
- **Beautiful TUI**: Clean terminal interface showing errors, context, and suggestions side-by-side

### Advanced Features
- **Pattern Learning**: Remembers successful fixes and applies them to similar future errors
- **Multi-language Support**: Python, JavaScript/TypeScript, Go, Rust, Java, and more
- **Stack Trace Analysis**: Understands and navigates stack traces
- **Quick Actions**: Jump to error location, apply suggested fixes, search similar issues

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TUI Interface Layer                     │
│  (Real-time display, user interaction, keyboard shortcuts)  │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────┴─────────────────────────────────────┐
│                   Controller/Orchestrator                    │
│         (Coordinates components, manages state)              │
└─┬───────────┬────────────┬────────────────┬─────────────────┘
  │           │            │                │
┌─▼────────┐ ┌▼─────────┐ ┌▼──────────────┐ ┌▼───────────────┐
│  Error   │ │  Error   │ │   History     │ │   Suggestion   │
│ Monitor  │ │  Parser  │ │   Analyzer    │ │     Engine     │
│          │ │          │ │               │ │                │
│ Watches: │ │ Extracts:│ │ Searches:     │ │ Generates:     │
│ - Logs   │ │ - Type   │ │ - Git commits │ │ - Fix patterns │
│ - Output │ │ - Message│ │ - Past errors │ │ - Code changes │
│ - Files  │ │ - Stack  │ │ - Solutions   │ │ - Suggestions  │
│ - Tests  │ │ - Context│ │ - Patterns    │ │ - Quick fixes  │
└──────────┘ └──────────┘ └───────────────┘ └────────────────┘
```

## Technology Stack

- **Language**: Python 3.10+
- **TUI Framework**: Textual (modern, reactive TUI framework)
- **Git Integration**: GitPython
- **Pattern Matching**: Regex + AST parsing
- **Error Detection**: Custom parsers for various languages/frameworks
- **Storage**: SQLite for pattern database

## Installation

### From Source

```bash
# Clone the repository
cd ai-debug-companion

# Install in development mode
pip install -e .

# Or install with dependencies
pip install -e ".[dev]"
```

### From PyPI (Coming Soon)

```bash
pip install ai-debug-companion
```

## Usage

### Interactive Watch Mode

The most powerful way to use the AI Debug Companion is in watch mode:

```bash
# Watch current directory
debug-companion watch

# Watch specific directory
debug-companion watch --path /path/to/project
```

This opens an interactive TUI that displays:
- **Error Panel**: Current error with full details and stack trace
- **Suggestions Tab**: AI-generated fix suggestions ranked by confidence
- **History Tab**: Similar errors from your git history with their fixes

**Keyboard shortcuts:**
- `q` - Quit
- `n` - Next error
- `p` - Previous error
- `r` - Refresh display

### Execute and Monitor

Run a command and monitor it for errors:

```bash
# Run tests with monitoring
debug-companion exec -- pytest tests/

# Run build with monitoring
debug-companion exec -- npm run build

# Run any command
debug-companion exec -- python my_script.py
```

The tool will:
1. Execute your command
2. Parse output for errors in real-time
3. Show suggestions immediately when errors are detected
4. Display a summary when complete

### Analyze Error Logs

Analyze existing error logs or files:

```bash
# Analyze a log file
debug-companion analyze error.log

# Verbose output with suggestions
debug-companion analyze error.log --verbose

# Analyze from stdin
cat error.log | debug-companion analyze -
```

### View Error Statistics

See common error patterns from your git history:

```bash
# Show errors from last 30 days
debug-companion stats

# Custom time range
debug-companion stats --days 90

# Specific repository
debug-companion stats --path /path/to/repo
```

### Demo Mode

See the tool in action with example errors:

```bash
debug-companion demo
```

## Examples

### Example 1: Python Import Error

When you encounter an import error:

```
ModuleNotFoundError: No module named 'requests'
```

The AI Debug Companion will:
1. Detect the error and parse it
2. Show suggestions like:
   - "Install missing module: requests" (90% confidence)
   - Command: `pip install requests`
3. Show similar past fixes from your git history
4. Display when/where similar errors were fixed before

### Example 2: JavaScript Undefined Error

For JavaScript errors:

```
TypeError: Cannot read property 'length' of undefined
```

You'll get:
- Check for undefined values before accessing properties
- Add null/undefined checks
- Similar patterns from your codebase
- Links to commits where you fixed similar issues

### Example 3: Development Workflow

```bash
# Terminal 1: Start the companion in watch mode
debug-companion watch

# Terminal 2: Run your tests
pytest tests/test_api.py

# The companion automatically detects errors and shows:
# - What went wrong
# - Where it happened
# - How you fixed similar issues before
# - Suggested fixes ranked by confidence
```

## Supported Languages

- ✅ **Python**: ImportError, AttributeError, TypeError, SyntaxError, etc.
- ✅ **JavaScript/TypeScript**: ReferenceError, TypeError, Module errors
- ✅ **Rust**: Compiler errors with error codes
- ✅ **Go**: Panics and runtime errors
- 🚧 **Java**: Coming soon
- 🚧 **C/C++**: Coming soon

## How It Works

### 1. Error Detection
The companion monitors multiple sources:
- Terminal output from running commands
- Log files in your project
- Test runner output
- Build system errors

### 2. Error Parsing
Advanced language-specific parsers extract:
- Error type and message
- File location and line number
- Stack traces
- Code context

### 3. Historical Analysis
Searches your git history for:
- Commits that fixed similar errors
- Common error patterns in your codebase
- Successful resolution strategies
- Code changes that resolved issues

### 4. Intelligent Suggestions
Combines multiple sources:
- **Historical fixes**: What worked before in your codebase
- **Common patterns**: Known solutions for common errors
- **Language-specific**: Smart fixes for each language
- **Contextual**: Based on your specific error details

## Configuration

Create a `.debug-companion.json` in your project root:

```json
{
  "watch_paths": ["."],
  "ignore_patterns": ["node_modules/", "venv/", "*.pyc"],
  "max_history_commits": 1000,
  "suggestion_threshold": 0.3
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_debug_companion

# Run specific test file
pytest tests/test_parsers.py
```

### Code Quality

```bash
# Format code
black ai_debug_companion/

# Lint
ruff check ai_debug_companion/
```

## Contributing

Contributions are welcome! Areas where you can help:

- 🌍 Add support for more programming languages
- 🎨 Improve the TUI interface
- 🧠 Enhance suggestion algorithms
- 📚 Add more common error patterns
- 🐛 Fix bugs and improve error parsing

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with:
- [Textual](https://textual.textualize.io/) - Amazing TUI framework
- [GitPython](https://gitpython.readthedocs.io/) - Git integration
- [Click](https://click.palletsprojects.com/) - CLI framework
- [Watchdog](https://github.com/gorakhargosh/watchdog) - File monitoring

---

**Built with AI, for developers** 🤖✨
