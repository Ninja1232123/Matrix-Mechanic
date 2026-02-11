"""
Command-line interface for AI Debug Companion.
"""

import asyncio
import sys
from pathlib import Path

import click

from . import __version__
from .monitor import ErrorMonitor
from .parsers import UniversalErrorParser
from .suggestions import SuggestionEngine
from .tui import run_tui


@click.group()
@click.version_option(version=__version__)
def main():
    """
    AI Debug Companion - Your intelligent debugging assistant.

    Watches your development session, detects errors, and suggests fixes
    based on git history and common patterns.
    """
    pass


@main.command()
@click.option(
    '--path',
    default='.',
    type=click.Path(exists=True),
    help='Directory to watch for errors'
)
def watch(path: str):
    """
    Start the interactive TUI to monitor errors in real-time.

    Opens a beautiful terminal interface that displays errors as they occur,
    along with intelligent suggestions and historical context.
    """
    click.echo(f"🔍 Starting AI Debug Companion in {path}")
    click.echo("Press 'q' to quit, 'n/p' for next/previous error")
    click.echo("")

    try:
        run_tui(path)
    except KeyboardInterrupt:
        click.echo("\n👋 Goodbye!")
    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('command', nargs=-1, required=True)
@click.option(
    '--path',
    default='.',
    type=click.Path(exists=True),
    help='Working directory for the command'
)
def exec(command: tuple[str, ...], path: str):
    """
    Execute a command and monitor it for errors.

    Example: debug-companion exec -- npm test
    """
    click.echo(f"[RUN]: {' '.join(command)}")
    click.echo("")

    monitor = ErrorMonitor(path)
    suggestion_engine = SuggestionEngine(path)

    # Track errors
    detected_errors = []

    def on_error(error):
        detected_errors.append(error)
        click.echo(f"\n[ERROR] {error}", err=True)

        # Generate suggestions immediately
        suggestions = suggestion_engine.generate_suggestions(error)
        if suggestions:
            click.echo("\n[INFO] Suggestions:", err=True)
            for i, sug in enumerate(suggestions[:3], 1):
                confidence = int(sug.confidence * 100)
                click.echo(f"  {i}. {sug.title} ({confidence}%)", err=True)
                click.echo(f"     {sug.description}", err=True)

    monitor.add_error_callback(on_error)

    # Run the command
    try:
        return_code = asyncio.run(monitor.monitor_command(list(command)))

        click.echo(f"\n[OK] Command completed with exit code: {return_code}")

        if detected_errors:
            click.echo(f"\n📊 Summary: {len(detected_errors)} error(s) detected")

        sys.exit(return_code)

    except KeyboardInterrupt:
        click.echo("\n⚠️  Command interrupted")
        sys.exit(130)
    except Exception as e:
        click.echo(f"\n[ERROR] Error executing command: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('file', type=click.File('r'))
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Show detailed analysis'
)
def analyze(file, verbose: bool):
    """
    Analyze a file or error log for errors and suggestions.

    Example: debug-companion analyze error.log
    """
    content = file.read()

    parser = UniversalErrorParser()
    suggestion_engine = SuggestionEngine()

    errors = parser.parse_multiple(content)

    if not errors:
        click.echo("[OK] No errors detected in the file")
        return

    click.echo(f"📋 Found {len(errors)} error(s):\n")

    for i, error in enumerate(errors, 1):
        click.echo(f"{i}. {error}")

        if verbose:
            if error.stack_trace:
                click.echo("   Stack trace:")
                for frame in error.stack_trace[:3]:
                    click.echo(f"     - {frame}")

            suggestions = suggestion_engine.generate_suggestions(error)
            if suggestions:
                click.echo("   Suggestions:")
                for sug in suggestions[:2]:
                    confidence = int(sug.confidence * 100)
                    click.echo(f"     • {sug.title} ({confidence}%)")

        click.echo("")


@main.command()
@click.option(
    '--path',
    default='.',
    type=click.Path(exists=True),
    help='Repository path'
)
@click.option(
    '--days',
    default=30,
    type=int,
    help='Number of days to analyze'
)
def stats(path: str, days: int):
    """
    Show error patterns from git history.

    Analyzes recent commits to identify common error patterns
    and frequently fixed issues.
    """
    from .history import GitHistoryAnalyzer

    analyzer = GitHistoryAnalyzer(path)
    patterns = analyzer.get_recent_error_patterns(days)

    if not patterns:
        click.echo(f"📊 No error patterns found in the last {days} days")
        return

    click.echo(f"📊 Error patterns from the last {days} days:\n")

    # Sort by frequency
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)

    for error_type, count in sorted_patterns[:10]:
        click.echo(f"  {error_type}: {count} fix(es)")


@main.command()
def demo():
    """
    Run a demo to see the AI Debug Companion in action.

    Creates sample errors to demonstrate the tool's capabilities.
    """
    click.echo("🎬 Running AI Debug Companion demo...\n")

    from .models import ErrorSeverity, Language, ParsedError, StackFrame

    # Create sample errors
    errors = [
        ParsedError(
            error_type="ImportError",
            message="No module named 'requests'",
            severity=ErrorSeverity.ERROR,
            language=Language.PYTHON,
            stack_trace=[
                StackFrame(
                    file_path="main.py",
                    line_number=5,
                    function_name="<module>"
                )
            ]
        ),
        ParsedError(
            error_type="TypeError",
            message="Cannot read property 'length' of undefined",
            severity=ErrorSeverity.ERROR,
            language=Language.JAVASCRIPT,
            stack_trace=[
                StackFrame(
                    file_path="app.js",
                    line_number=42,
                    function_name="processData"
                )
            ]
        )
    ]

    suggestion_engine = SuggestionEngine()

    for i, error in enumerate(errors, 1):
        click.echo(f"Example {i}: {error}\n")

        suggestions = suggestion_engine.generate_suggestions(error)

        if suggestions:
            click.echo("[INFO] Suggestions:")
            for j, sug in enumerate(suggestions[:2], 1):
                confidence = int(sug.confidence * 100)
                click.echo(f"  {j}. {sug.title} ({confidence}%)")
                click.echo(f"     {sug.description}")

        click.echo("\n" + "="*60 + "\n")

    click.echo("✨ Demo complete! Run 'debug-companion watch' to start monitoring.")


@main.command()
@click.argument('command', nargs=-1, required=True)
@click.option(
    '--apply',
    is_flag=True,
    help='Actually apply the fixes (default is dry-run)'
)
@click.option(
    '--auto',
    is_flag=True,
    help='Automatically apply high-confidence fixes without confirmation'
)
@click.option(
    '--path',
    default='.',
    type=click.Path(exists=True),
    help='Working directory for the command'
)
def fix(command: tuple[str, ...], apply: bool, auto: bool, path: str):
    """
    Execute a command, detect errors, and suggest/apply fixes.

    This is like 'exec' but with automatic fix generation.
    Use --apply to actually apply fixes (otherwise shows preview).
    Use --auto to skip confirmation for high-confidence fixes.

    Example: debug-companion fix -- python script.py
    Example: debug-companion fix --apply -- python script.py
    """
    from .fixer import AutoFixer

    click.echo(f"[RUN] with auto-fix: {' '.join(command)}")
    click.echo(f"Mode: {'APPLY' if apply else 'DRY-RUN'}")
    click.echo("")

    monitor = ErrorMonitor(path)
    suggestion_engine = SuggestionEngine(path)
    fixer = AutoFixer(dry_run=not apply)

    detected_errors = []
    generated_fixes = []

    def on_error(error):
        detected_errors.append(error)
        click.echo(f"\n[ERROR] {error}", err=True)

        # Generate suggestions
        suggestions = suggestion_engine.generate_suggestions(error)

        if suggestions:
            click.echo("\n[INFO] Suggestions:", err=True)
            for i, sug in enumerate(suggestions[:3], 1):
                confidence = int(sug.confidence * 100)
                click.echo(f"  {i}. {sug.title} ({confidence}%)", err=True)

            # Generate fixes
            fixes = fixer.generate_all_fixes(error, suggestions[:3])

            if fixes:
                click.echo("\n[FIX] Available fixes:", err=True)
                for i, fix_obj in enumerate(fixes, 1):
                    click.echo(f"\n  Fix {i}:", err=True)
                    click.echo(f"    Confidence: {int(fix_obj.confidence * 100)}%", err=True)
                    click.echo(f"    {fix_obj.explanation}", err=True)

                    if fix_obj.file_path != "<command>":
                        click.echo(f"    {fix_obj.file_path}:{fix_obj.line_number}", err=True)
                        click.echo(f"    - {fix_obj.original_code}", err=True)
                        click.echo(f"    + {fix_obj.fixed_code[:100]}...", err=True)
                    else:
                        click.echo(f"    Command: {fix_obj.fixed_code}", err=True)

                    generated_fixes.append(fix_obj)

    monitor.add_error_callback(on_error)

    # Run the command
    try:
        return_code = asyncio.run(monitor.monitor_command(list(command)))

        click.echo(f"\n[OK] Command completed with exit code: {return_code}")

        if generated_fixes:
            click.echo(f"\n[FIX] Generated {len(generated_fixes)} fix(es)")

            if apply:
                click.echo("\n[APPLY] Applying fixes...")
                for i, fix_obj in enumerate(generated_fixes, 1):
                    # Auto-apply high confidence fixes if --auto is set
                    should_apply = auto and fix_obj.confidence >= 0.85

                    if not auto and fix_obj.confidence >= 0.75:
                        # Ask for confirmation
                        click.echo(f"\n  Apply fix {i}? ({fix_obj.explanation})")
                        click.echo(f"  Confidence: {int(fix_obj.confidence * 100)}%")
                        should_apply = click.confirm("  Apply this fix?", default=False)

                    if should_apply:
                        if fixer.apply_fix(fix_obj):
                            click.echo(f"  [OK] Applied fix {i}")
                        else:
                            click.echo(f"  [ERROR] Failed to apply fix {i}")
            else:
                click.echo("\n[INFO] Run with --apply to apply these fixes")
                click.echo("[INFO] Run with --apply --auto to auto-apply high-confidence fixes")

        sys.exit(return_code)

    except KeyboardInterrupt:
        click.echo("\n⚠️  Command interrupted")
        sys.exit(130)
    except Exception as e:
        click.echo(f"\n[ERROR] Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.group()
def ai():
    """Manage AI provider configuration."""
    pass


@ai.command('setup')
@click.option(
    '--provider',
    type=click.Choice(['claude', 'openai', 'ollama', 'groq', 'openrouter']),
    prompt=True,
    help='AI provider to use'
)
@click.option(
    '--api-key',
    prompt=True,
    hide_input=True,
    default='',
    help='API key (leave empty for Ollama)'
)
@click.option(
    '--model',
    help='Model name (optional, uses provider default if not specified)'
)
def ai_setup(provider: str, api_key: str, model: str):
    """
    Configure AI provider for intelligent code fixing.

    Supports: Claude, OpenAI, Ollama (local), Groq, OpenRouter
    """
    from .ai_provider import create_config_template
    import json

    config_path = Path.home() / ".devtools-ai.json"

    config = {
        "provider": provider,
        "api_key": api_key if api_key else None,
    }

    if model:
        config["model"] = model

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    click.echo(f"[OK] AI provider configured: {provider}")
    click.echo(f"📁 Config saved to: {config_path}")

    if provider == "ollama":
        click.echo("\n[INFO] Make sure Ollama is running: ollama serve")
        click.echo("[INFO] Pull a model: ollama pull llama3.1")


@ai.command('status')
def ai_status():
    """Show AI provider status and available providers."""
    from .ai_provider import AIProviderFactory

    click.echo("🔍 Checking AI providers...\n")

    available = AIProviderFactory.list_available()

    if available:
        click.echo("[OK] Available providers:")
        for provider in available:
            click.echo(f"  • {provider}")
    else:
        click.echo("[ERROR] No AI providers configured")
        click.echo("\n[INFO] Configure one with: debug-companion ai setup")
        click.echo("[INFO] Or run Ollama locally (free!): ollama serve")
        return

    # Try to get the current provider
    try:
        provider = AIProviderFactory.from_config()
        click.echo(f"\n🎯 Active provider: {provider.__class__.__name__.replace('Provider', '').lower()}")
        click.echo(f"📦 Model: {provider.model}")
    except Exception as e:
        click.echo(f"\n⚠️  Error loading provider: {e}")


@ai.command('test')
@click.option(
    '--prompt',
    default="Write a Python function to add two numbers",
    help='Test prompt'
)
def ai_test(prompt: str):
    """Test AI provider with a simple prompt."""
    from .ai_provider import AIProviderFactory

    try:
        provider = AIProviderFactory.from_config()
        click.echo(f"🧪 Testing {provider.__class__.__name__}...")
        click.echo(f"[APPLY] Prompt: {prompt}\n")

        response = provider.generate(prompt, max_tokens=200)

        click.echo("[OK] Response:")
        click.echo(response.content)

        if response.tokens_used:
            click.echo(f"\n📊 Tokens used: {response.tokens_used}")

    except Exception as e:
        click.echo(f"[ERROR] Test failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
