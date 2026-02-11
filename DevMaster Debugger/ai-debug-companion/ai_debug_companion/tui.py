"""
Beautiful TUI interface for the AI Debug Companion.
"""

from datetime import datetime
from typing import Optional

from rich.syntax import Syntax
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Label, Static, TabbedContent, TabPane
from textual.reactive import reactive

from .history import GitHistoryAnalyzer
from .models import ParsedError, Suggestion
from .monitor import ErrorMonitor
from .suggestions import SuggestionEngine


class ErrorDisplay(Static):
    """Widget to display a parsed error."""

    def __init__(self, error: ParsedError, **kwargs):
        super().__init__(**kwargs)
        self.error = error

    def compose(self) -> ComposeResult:
        error = self.error

        # Error header with type and severity
        severity_color = {
            'info': 'blue',
            'warning': 'yellow',
            'error': 'red',
            'critical': 'bold red'
        }.get(error.severity.value, 'white')

        header_text = Text()
        header_text.append(f"[{error.severity.value.upper()}] ", style=severity_color)
        header_text.append(error.error_type, style="bold")
        header_text.append(f" ({error.language.value})", style="dim")

        yield Label(header_text)
        yield Label(f"  {error.message}", classes="error-message")

        # Location
        if error.primary_location:
            loc = error.primary_location
            location_text = f"  📍 {loc.file_path}:{loc.line_number}"
            if loc.function_name:
                location_text += f" in {loc.function_name}"
            yield Label(location_text, classes="error-location")

        # Timestamp
        time_str = error.timestamp.strftime("%H:%M:%S")
        yield Label(f"  🕐 {time_str}", classes="error-time")


class SuggestionDisplay(Static):
    """Widget to display a suggestion."""

    def __init__(self, suggestion: Suggestion, index: int, **kwargs):
        super().__init__(**kwargs)
        self.suggestion = suggestion
        self.index = index

    def compose(self) -> ComposeResult:
        sug = self.suggestion

        # Confidence bar
        confidence_pct = int(sug.confidence * 100)
        confidence_color = "green" if confidence_pct >= 70 else "yellow" if confidence_pct >= 50 else "red"

        title_text = Text()
        title_text.append(f"{self.index}. ", style="bold cyan")
        title_text.append(sug.title, style="bold")
        title_text.append(f" ({confidence_pct}%)", style=confidence_color)

        yield Label(title_text)
        yield Label(f"   {sug.description}", classes="suggestion-desc")

        # Show action if available
        if sug.action:
            yield Label(f"   💡 Action: {sug.action}", classes="suggestion-action")


class ErrorPanel(VerticalScroll):
    """Panel showing the current error and its details."""

    current_error: reactive[Optional[ParsedError]] = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def watch_current_error(self, error: Optional[ParsedError]):
        """React to error changes."""
        self.refresh_display()

    def refresh_display(self):
        """Refresh the error display."""
        self.remove_children()

        if not self.current_error:
            self.mount(Label("No errors detected yet. Monitoring...", classes="placeholder"))
            return

        self.mount(ErrorDisplay(self.current_error))

        # Show stack trace if available
        if self.current_error.stack_trace:
            self.mount(Label("\n📚 Stack Trace:", classes="section-header"))
            for i, frame in enumerate(self.current_error.stack_trace[:5]):  # Show top 5
                frame_text = f"  {i+1}. {frame.file_path}:{frame.line_number}"
                if frame.function_name:
                    frame_text += f" in {frame.function_name}"
                self.mount(Label(frame_text, classes="stack-frame"))


class SuggestionsPanel(VerticalScroll):
    """Panel showing suggestions for fixing the error."""

    current_suggestions: reactive[list[Suggestion]] = reactive(list)

    def watch_current_suggestions(self, suggestions: list[Suggestion]):
        """React to suggestions changes."""
        self.refresh_display()

    def refresh_display(self):
        """Refresh the suggestions display."""
        self.remove_children()

        if not self.current_suggestions:
            self.mount(Label("No suggestions yet.", classes="placeholder"))
            return

        self.mount(Label("🔧 Suggested Fixes:", classes="section-header"))

        for i, suggestion in enumerate(self.current_suggestions[:5], 1):
            self.mount(SuggestionDisplay(suggestion, i))
            if i < len(self.current_suggestions):
                self.mount(Label(""))  # Spacing


class HistoryPanel(VerticalScroll):
    """Panel showing related fixes from git history."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history_items = []

    def show_history(self, error: ParsedError, analyzer: GitHistoryAnalyzer):
        """Show historical fixes for an error."""
        self.remove_children()
        self.mount(Label("🕰️  Similar Fixes from History", classes="section-header"))

        fixes = analyzer.find_similar_fixes(error, max_results=5)

        if not fixes:
            self.mount(Label("  No similar fixes found in history.", classes="placeholder"))
            return

        for fix in fixes:
            self.mount(Label(f"\n  {fix.commit_hash[:8]} - {fix.commit_message[:60]}", classes="commit-msg"))
            self.mount(Label(f"  👤 {fix.author}", classes="commit-author"))
            self.mount(Label(f"  📅 {fix.date.strftime('%Y-%m-%d %H:%M')}", classes="commit-date"))
            self.mount(Label(f"  🎯 Similarity: {int(fix.similarity_score * 100)}%", classes="similarity"))


class DebugCompanionApp(App):
    """Main TUI application for the AI Debug Companion."""

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        background: $primary;
        color: $text;
    }

    Footer {
        background: $panel;
    }

    .error-message {
        color: $warning;
        margin: 0 0 0 2;
    }

    .error-location {
        color: $accent;
        margin: 0 0 0 2;
    }

    .error-time {
        color: $text-muted;
        margin: 0 0 0 2;
    }

    .section-header {
        color: $primary;
        text-style: bold;
        margin: 1 0;
    }

    .suggestion-desc {
        color: $text;
        margin: 0 0 0 3;
    }

    .suggestion-action {
        color: $success;
        margin: 0 0 1 3;
    }

    .stack-frame {
        color: $text-muted;
        margin: 0 0 0 2;
    }

    .placeholder {
        color: $text-muted;
        text-style: italic;
        margin: 2;
    }

    .commit-msg {
        color: $warning;
    }

    .commit-author, .commit-date, .similarity {
        color: $text-muted;
        margin: 0 0 0 2;
    }

    ErrorPanel, SuggestionsPanel, HistoryPanel {
        border: solid $primary;
        height: 1fr;
        padding: 1;
    }

    #error-container {
        width: 1fr;
    }

    #suggestions-container {
        width: 1fr;
    }

    Horizontal {
        height: 1fr;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("n", "next_error", "Next Error"),
        ("p", "prev_error", "Previous Error"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, watch_path: str = "."):
        super().__init__()
        self.watch_path = watch_path
        self.monitor = ErrorMonitor(watch_path)
        self.suggestion_engine = SuggestionEngine(watch_path)
        self.history_analyzer = GitHistoryAnalyzer(watch_path)
        self.errors: list[ParsedError] = []
        self.current_error_index = -1

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="error-container"):
                yield ErrorPanel(id="error-panel")

            with TabbedContent():
                with TabPane("Suggestions", id="suggestions-tab"):
                    yield SuggestionsPanel(id="suggestions-panel")

                with TabPane("History", id="history-tab"):
                    yield HistoryPanel(id="history-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Set up the app when mounted."""
        self.title = "AI Debug Companion"
        self.sub_title = f"Watching: {self.watch_path}"

        # Register error callback
        self.monitor.add_error_callback(self.on_error_detected)

        # Start monitoring
        self.monitor.start_watching_files()

    def on_error_detected(self, error: ParsedError):
        """Handle detected errors."""
        self.errors.append(error)
        self.current_error_index = len(self.errors) - 1
        self.call_later(self.update_display)

    def update_display(self):
        """Update the display with current error."""
        if self.current_error_index < 0 or self.current_error_index >= len(self.errors):
            return

        error = self.errors[self.current_error_index]

        # Update error panel
        error_panel = self.query_one("#error-panel", ErrorPanel)
        error_panel.current_error = error

        # Generate and show suggestions
        suggestions = self.suggestion_engine.generate_suggestions(error)
        suggestions_panel = self.query_one("#suggestions-panel", SuggestionsPanel)
        suggestions_panel.current_suggestions = suggestions

        # Update history panel
        history_panel = self.query_one("#history-panel", HistoryPanel)
        history_panel.show_history(error, self.history_analyzer)

    def action_next_error(self):
        """Show next error."""
        if self.errors and self.current_error_index < len(self.errors) - 1:
            self.current_error_index += 1
            self.update_display()

    def action_prev_error(self):
        """Show previous error."""
        if self.errors and self.current_error_index > 0:
            self.current_error_index -= 1
            self.update_display()

    def action_refresh(self):
        """Refresh the current display."""
        self.update_display()

    def on_unmount(self):
        """Clean up when app closes."""
        self.monitor.stop_watching_files()


def run_tui(watch_path: str = "."):
    """Run the TUI application."""
    app = DebugCompanionApp(watch_path)
    app.run()
