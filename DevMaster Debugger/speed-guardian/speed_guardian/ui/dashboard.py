"""Interactive TUI dashboard for Speed-Guardian"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    DataTable,
    Button,
    Input,
    Label,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.binding import Binding
from textual.reactive import reactive
from pathlib import Path
from datetime import datetime

from ..core import SpeedGuardian
from ..models import Severity, PerformanceReport


class PerformanceSummary(Static):
    """Widget displaying performance summary"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report = None

    def update_report(self, report: PerformanceReport):
        """Update with new report"""
        self.report = report
        self.refresh()

    def render(self) -> str:
        """Render the summary"""
        if not self.report:
            return "[dim]No profile data loaded[/dim]"

        summary = self.report.summary
        lines = []
        lines.append(f"[bold cyan]Target:[/bold cyan] {self.report.target}")
        lines.append(f"[bold cyan]Time:[/bold cyan] {summary['total_time_ms']:.2f}ms")
        lines.append(f"[bold cyan]Calls:[/bold cyan] {summary['total_calls']:,}")
        lines.append(f"[bold cyan]Bottlenecks:[/bold cyan] {summary['bottlenecks']}")

        critical = summary.get('critical_bottlenecks', 0)
        if critical > 0:
            lines.append(f"[bold red]Critical Issues:[/bold red] {critical}")

        if 'optimizations_found' in summary:
            lines.append(f"[bold yellow]Optimizations:[/bold yellow] {summary['optimizations_found']}")

        return "\n".join(lines)


class BottlenecksTable(Static):
    """Widget displaying bottlenecks table"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottlenecks = []

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield DataTable()

    def on_mount(self) -> None:
        """Setup table on mount"""
        table = self.query_one(DataTable)
        table.add_columns("Function", "Time (ms)", "% Total", "Calls", "Severity")

    def update_bottlenecks(self, bottlenecks):
        """Update bottlenecks data"""
        self.bottlenecks = bottlenecks
        table = self.query_one(DataTable)
        table.clear()

        for bottleneck in bottlenecks[:50]:
            severity_map = {
                Severity.CRITICAL: "🔴 CRITICAL",
                Severity.HIGH: "🟡 HIGH",
                Severity.MEDIUM: "🔵 MEDIUM",
                Severity.LOW: "🟢 LOW",
            }

            table.add_row(
                bottleneck.function_name[:40],
                f"{bottleneck.time_ms:.2f}",
                f"{bottleneck.percentage:.1f}%",
                f"{bottleneck.calls:,}",
                severity_map[bottleneck.severity]
            )


class OptimizationsTable(Static):
    """Widget displaying optimization suggestions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizations = []

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield DataTable()

    def on_mount(self) -> None:
        """Setup table on mount"""
        table = self.query_one(DataTable)
        table.add_columns("Type", "Description", "Speedup", "Confidence", "Status")

    def update_optimizations(self, optimizations):
        """Update optimizations data"""
        self.optimizations = optimizations
        table = self.query_one(DataTable)
        table.clear()

        for opt in optimizations[:50]:
            status = "✓ Applied" if opt.applied else "○ Suggested"

            table.add_row(
                opt.type.value,
                opt.description[:50],
                f"{opt.estimated_speedup:.1f}x",
                f"{opt.confidence:.0%}",
                status
            )


class SpeedGuardianDashboard(App):
    """Interactive Speed-Guardian Dashboard"""

    CSS = """
    Screen {
        background: $surface;
    }

    Header {
        background: $primary;
    }

    PerformanceSummary {
        height: auto;
        border: solid $primary;
        padding: 1 2;
        margin: 1 0;
    }

    DataTable {
        height: 100%;
    }

    Button {
        margin: 1 2;
    }

    Input {
        margin: 1 2;
    }

    TabPane {
        padding: 1 2;
    }

    #summary-container {
        height: auto;
        margin: 1 0;
    }

    #action-bar {
        height: auto;
        padding: 1 2;
        background: $panel;
    }

    #status {
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("p", "profile", "Profile"),
        Binding("o", "optimize", "Optimize"),
    ]

    TITLE = "Speed-Guardian Dashboard"
    SUB_TITLE = "Performance Profiler & Optimizer"

    current_file = reactive("")
    status_message = reactive("Ready")

    def __init__(self):
        super().__init__()
        self.sg = SpeedGuardian()
        self.current_report = None

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        with Container(id="summary-container"):
            yield PerformanceSummary(id="summary")

        with Horizontal(id="action-bar"):
            yield Input(placeholder="Enter Python file path...", id="file-input")
            yield Button("Profile", variant="primary", id="profile-btn")
            yield Button("Optimize", variant="success", id="optimize-btn")
            yield Button("Clear", variant="default", id="clear-btn")

        with TabbedContent():
            with TabPane("Bottlenecks"):
                yield BottlenecksTable(id="bottlenecks")

            with TabPane("Optimizations"):
                yield OptimizationsTable(id="optimizations")

            with TabPane("Details"):
                yield ScrollableContainer(
                    Static(id="details-text"),
                    id="details-container"
                )

        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event"""
        self.update_status("Ready - Enter a Python file path and click Profile")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "profile-btn":
            self.action_profile()
        elif event.button.id == "optimize-btn":
            self.action_optimize()
        elif event.button.id == "clear-btn":
            self.action_clear()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id == "file-input":
            self.action_profile()

    def action_profile(self) -> None:
        """Profile the current file"""
        file_input = self.query_one("#file-input", Input)
        file_path = file_input.value.strip()

        if not file_path:
            self.update_status("Error: No file specified")
            return

        path = Path(file_path)
        if not path.exists():
            self.update_status(f"Error: File not found: {file_path}")
            return

        self.update_status(f"Profiling {file_path}...")

        try:
            # Profile the file
            report = self.sg.profile_script(path, analyze_patterns=True)
            self.current_report = report
            self.current_file = file_path

            # Update UI
            self._update_display(report)

            self.update_status(f"Profile complete - {len(report.bottlenecks)} bottlenecks found")

        except Exception as e:
            self.update_status(f"Error: {str(e)}")

    def action_optimize(self) -> None:
        """Optimize the current file"""
        if not self.current_report or not self.current_report.optimizations:
            self.update_status("No optimizations available - profile a file first")
            return

        self.update_status("Applying optimizations...")

        try:
            result = self.sg.auto_fixer.apply_optimizations(
                self.current_report.optimizations,
                dry_run=False
            )

            applied = len(result['applied'])
            failed = len(result['failed'])

            self.update_status(f"Applied {applied} optimizations ({failed} failed)")

            # Refresh display
            self._update_display(self.current_report)

        except Exception as e:
            self.update_status(f"Error: {str(e)}")

    def action_clear(self) -> None:
        """Clear the current data"""
        self.current_report = None
        self.current_file = ""

        # Clear displays
        summary = self.query_one("#summary", PerformanceSummary)
        summary.report = None
        summary.refresh()

        bottlenecks = self.query_one("#bottlenecks", BottlenecksTable)
        bottlenecks.query_one(DataTable).clear()

        optimizations = self.query_one("#optimizations", OptimizationsTable)
        optimizations.query_one(DataTable).clear()

        details = self.query_one("#details-text", Static)
        details.update("")

        self.update_status("Cleared")

    def action_refresh(self) -> None:
        """Refresh the current profile"""
        if self.current_file:
            file_input = self.query_one("#file-input", Input)
            file_input.value = self.current_file
            self.action_profile()

    def _update_display(self, report: PerformanceReport) -> None:
        """Update all display widgets with report data"""
        # Update summary
        summary = self.query_one("#summary", PerformanceSummary)
        summary.update_report(report)

        # Update bottlenecks
        bottlenecks = self.query_one("#bottlenecks", BottlenecksTable)
        bottlenecks.update_bottlenecks(report.bottlenecks)

        # Update optimizations
        optimizations = self.query_one("#optimizations", OptimizationsTable)
        optimizations.update_optimizations(report.optimizations)

        # Update details
        details_text = self.sg.generate_report_text(report)
        details = self.query_one("#details-text", Static)
        details.update(details_text)

    def update_status(self, message: str) -> None:
        """Update status message"""
        self.status_message = message
        status = self.query_one("#status", Static)
        timestamp = datetime.now().strftime("%H:%M:%S")
        status.update(f"[dim]{timestamp}[/dim] {message}")


def run_dashboard():
    """Run the dashboard application"""
    app = SpeedGuardianDashboard()
    app.run()
