"""Command-line interface for Chaos-Guardian"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core import ChaosGuardian
from .models import ChaosExperiment, ChaosConfig, ChaosType

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Chaos-Guardian: Chaos Engineering for Python 🌪️"""
    pass


@main.command()
@click.option('--exception-rate', default=0.1, help='Exception injection rate (0.0-1.0)')
@click.option('--latency-ms', default=1000, help='Latency to inject (milliseconds)')
@click.option('--iterations', default=100, help='Number of test iterations')
def test(exception_rate, latency_ms, iterations):
    """Run a quick chaos test"""
    console.print("\n[bold red]🌪️  Chaos-Guardian Test[/bold red]")
    console.print(f"Exception rate: {exception_rate}")
    console.print(f"Latency: {latency_ms}ms")
    console.print(f"Iterations: {iterations}\n")

    cg = ChaosGuardian()

    # Create test function
    @cg.chaos(exception_rate=exception_rate, latency_ms=latency_ms)
    def test_function():
        return "success"

    # Run test
    successes = 0
    failures = 0

    for i in range(iterations):
        try:
            result = test_function()
            successes += 1
        except Exception as e:
            failures += 1

    # Display results
    _display_test_results(successes, failures, iterations)


@main.command()
def stats():
    """Show chaos statistics"""
    cg = ChaosGuardian()
    stats = cg.get_stats()

    console.print("\n[bold]Chaos Statistics[/bold]")
    console.print(f"Total Events: {stats['total_chaos_events']}")
    console.print(f"Enabled: {stats['enabled']}")
    console.print(f"Kill Switch: {stats['kill_switch_active']}")

    if stats['by_type']:
        console.print("\nEvents by Type:")
        for chaos_type, count in stats['by_type'].items():
            console.print(f"  {chaos_type}: {count}")


def _display_test_results(successes: int, failures: int, total: int):
    """Display test results"""
    success_rate = (successes / total * 100) if total > 0 else 0
    resilience_score = success_rate

    # Determine resilience level
    if resilience_score >= 90:
        level = "[green]EXCELLENT[/green]"
        emoji = "💪"
    elif resilience_score >= 75:
        level = "[yellow]GOOD[/yellow]"
        emoji = "👍"
    elif resilience_score >= 50:
        level = "[orange]FAIR[/orange]"
        emoji = "⚠️"
    else:
        level = "[red]POOR[/red]"
        emoji = "🔥"

    summary = f"""
[bold]Test Results[/bold]

Total Iterations: {total}
✅ Successes: {successes}
❌ Failures: {failures}

Success Rate: {success_rate:.1f}%
Resilience Score: {resilience_score:.1f}/100

Resilience Level: {level} {emoji}
    """

    console.print(Panel(summary, title="[bold red]Chaos Test Complete[/bold red]", expand=False))

    # Recommendations
    if resilience_score < 75:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        console.print("• Add more error handling (try/except blocks)")
        console.print("• Implement retry logic with exponential backoff")
        console.print("• Add input validation")
        console.print("• Consider circuit breaker patterns")


if __name__ == '__main__':
    main()
