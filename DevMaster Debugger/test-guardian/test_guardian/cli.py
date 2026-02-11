"""Command-line interface for Test-Guardian"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core import TestGuardian
from .models import TestFramework, TestType

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Test-Guardian: Automatic Test Generation for Python"""
    pass


@main.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output test file path')
@click.option('--framework', '-f', type=click.Choice(['pytest', 'unittest']), default='pytest', help='Test framework')
@click.option('--function', help='Generate tests for specific function only')
def generate(file, output, framework, function):
    """Generate tests for a Python file"""
    console.print(f"\n[bold cyan]Test-Guardian[/bold cyan]")
    console.print(f"Generating tests for: [yellow]{file}[/yellow]\n")

    file_path = Path(file)
    output_path = Path(output) if output else None

    test_framework = TestFramework.PYTEST if framework == 'pytest' else TestFramework.UNITTEST

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing code...", total=None)

        # Initialize Test-Guardian
        tg = TestGuardian(framework=test_framework)

        # Generate tests
        if function:
            progress.update(task, description=f"Generating tests for {function}...")
            report = tg.generate_tests_for_function(file_path, function, output_path)
        else:
            progress.update(task, description="Generating tests...")
            report = tg.generate_tests_for_file(file_path, output_path)

        progress.update(task, description="Generation complete!")

    # Display results
    _display_report(report)


@main.command()
@click.argument('file', type=click.Path(exists=True))
def preview(file):
    """Preview tests that would be generated"""
    console.print(f"\n[bold cyan]Test-Guardian - Preview Mode[/bold cyan]")
    console.print(f"Previewing tests for: [yellow]{file}[/yellow]\n")

    file_path = Path(file)
    tg = TestGuardian()

    preview_text = tg.preview_tests(file_path)
    console.print(preview_text)


@main.command()
@click.argument('file', type=click.Path(exists=True))
def analyze(file):
    """Analyze a file's testability"""
    console.print(f"\n[bold cyan]Test-Guardian - Code Analysis[/bold cyan]")
    console.print(f"Analyzing: [yellow]{file}[/yellow]\n")

    file_path = Path(file)
    tg = TestGuardian()

    # Analyze functions
    functions = tg.code_analyzer.find_testable_functions(file_path)

    # Create table
    table = Table(title="Testable Functions")
    table.add_column("Function", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Complexity", justify="right")
    table.add_column("Pure", justify="center")
    table.add_column("Raises", style="yellow")

    for func in functions:
        is_pure = "✓" if func.is_pure else "✗"
        raises = ", ".join(func.raises) if func.raises else "-"

        table.add_row(
            func.name,
            str(len(func.parameters)),
            str(func.complexity),
            is_pure,
            raises
        )

    console.print(table)

    # Summary
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Testable functions: {len(functions)}")
    console.print(f"Pure functions: {sum(1 for f in functions if f.is_pure)}")
    console.print(f"Functions with exceptions: {sum(1 for f in functions if f.raises)}")


def _display_report(report):
    """Display test generation report"""
    # Summary panel
    summary_text = (
        f"[bold]Target:[/bold] {report.target}\n"
        f"[bold]Tests Generated:[/bold] {report.summary['tests_generated']}\n"
        f"[bold]Fixtures:[/bold] {report.summary.get('fixtures', 0)}\n"
        f"[bold]Mocks:[/bold] {report.summary.get('mocks', 0)}\n"
        f"[bold]Output:[/bold] {report.generated_file}"
    )

    console.print(Panel(summary_text, title="[bold cyan]Test Generation Complete[/bold cyan]", expand=False))

    # Test breakdown
    if report.test_suite.test_cases:
        console.print("\n[bold]Generated Tests[/bold]")

        # Count by type
        test_counts = {}
        for tc in report.test_suite.test_cases:
            test_counts[tc.test_type.value] = test_counts.get(tc.test_type.value, 0) + 1

        table = Table()
        table.add_column("Test Type", style="cyan")
        table.add_column("Count", justify="right")

        for test_type, count in sorted(test_counts.items()):
            table.add_row(test_type.title(), str(count))

        console.print(table)

    # Next steps
    console.print("\n[bold green]Success![/bold green]")
    console.print(f"Tests written to: [yellow]{report.generated_file}[/yellow]")
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"  1. Review generated tests in {report.generated_file}")
    console.print("  2. Fill in TODO placeholders with expected values")
    console.print("  3. Run tests: pytest " + report.generated_file)


if __name__ == '__main__':
    main()
