"""Command-line interface for Speed-Guardian"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core import SpeedGuardian
from .config import SpeedGuardianConfig
from .models import Severity

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Speed-Guardian: Automatic Performance Profiler and Optimizer"""
    pass


@main.command()
@click.argument('script', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output report file')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--analyze-patterns/--no-patterns', default=True, help='Analyze code patterns')
def profile(script, output, format, analyze_patterns):
    """Profile a Python script for performance"""
    console.print(f"\n[bold cyan]Speed-Guardian Profiler[/bold cyan]")
    console.print(f"Profiling: [yellow]{script}[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Profiling...", total=None)

        # Initialize Speed-Guardian
        sg = SpeedGuardian()

        # Profile the script
        report = sg.profile_script(Path(script), analyze_patterns=analyze_patterns)

        progress.update(task, description="Analysis complete!")

    # Display results
    _display_report(report)

    # Save report if requested
    if output:
        sg.save_report(report, Path(output))
        console.print(f"\n[green]Report saved to {output}[/green]")


@main.command()
@click.argument('script', type=click.Path(exists=True))
@click.option('--auto-fix/--no-auto-fix', default=False, help='Automatically apply fixes')
@click.option('--dry-run', is_flag=True, help='Show what would be done without applying')
@click.option('--output', '-o', type=click.Path(), help='Output report file')
def optimize(script, auto_fix, dry_run, output):
    """Profile and optimize a Python script"""
    console.print(f"\n[bold cyan]Speed-Guardian Optimizer[/bold cyan]")
    console.print(f"Analyzing: [yellow]{script}[/yellow]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)

        # Initialize Speed-Guardian
        sg = SpeedGuardian()

        # Profile and optimize
        result = sg.profile_and_optimize(
            Path(script),
            auto_fix=auto_fix,
            dry_run=dry_run
        )

        progress.update(task, description="Analysis complete!")

    report = result['report']
    fix_results = result.get('fix_results')

    # Display performance report
    _display_report(report)

    # Display fix results
    if fix_results:
        console.print("\n[bold]Optimization Results[/bold]")
        console.print(f"Applied: [green]{len(fix_results['applied'])}[/green]")
        console.print(f"Failed: [red]{len(fix_results['failed'])}[/red]")
        console.print(f"Skipped: [yellow]{len(fix_results['skipped'])}[/yellow]")

    # Save report if requested
    if output:
        sg.save_report(report, Path(output))
        console.print(f"\n[green]Report saved to {output}[/green]")


@main.command()
@click.argument('file', type=click.Path(exists=True))
def complexity(file):
    """Analyze code complexity"""
    console.print(f"\n[bold cyan]Speed-Guardian Complexity Analyzer[/bold cyan]")
    console.print(f"Analyzing: [yellow]{file}[/yellow]\n")

    sg = SpeedGuardian()
    analysis = sg.analyze_complexity(Path(file))

    # Create table
    table = Table(title="Function Complexity")
    table.add_column("Function", style="cyan")
    table.add_column("Cyclomatic", justify="right")
    table.add_column("Nesting", justify="right")
    table.add_column("Loops", justify="right")
    table.add_column("Complexity", style="yellow")
    table.add_column("LOC", justify="right")

    for func_name, metrics in analysis.items():
        complexity_style = "green"
        if metrics['cyclomatic_complexity'] > 15:
            complexity_style = "red"
        elif metrics['cyclomatic_complexity'] > 10:
            complexity_style = "yellow"

        table.add_row(
            func_name,
            str(metrics['cyclomatic_complexity']),
            str(metrics['nesting_depth']),
            str(metrics['loop_count']),
            f"[{complexity_style}]{metrics['estimated_time_complexity']}[/{complexity_style}]",
            str(metrics['lines_of_code'])
        )

    console.print(table)

    # Summary
    summary = sg.complexity_analyzer.get_summary(analysis)
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Total Functions: {summary['total_functions']}")
    console.print(f"Average Complexity: {summary['avg_complexity']:.1f}")
    console.print(f"Max Complexity: {summary['max_complexity']}")


@main.command()
def dashboard():
    """Launch interactive performance dashboard"""
    from .ui.dashboard import run_dashboard
    run_dashboard()


@main.command()
@click.option('--config-file', type=click.Path(), help='Configuration file path')
def init(config_file):
    """Initialize Speed-Guardian configuration"""
    config_path = Path(config_file) if config_file else Path('.speed_guardian.yaml')

    if config_path.exists():
        console.print(f"[yellow]Configuration file already exists: {config_path}[/yellow]")
        if not click.confirm("Overwrite?"):
            return

    # Create default config
    config = SpeedGuardianConfig()
    config.to_file(config_path)

    console.print(f"[green]Configuration file created: {config_path}[/green]")


def _display_report(report):
    """Display performance report using Rich"""
    # Summary panel
    summary_text = (
        f"[bold]Target:[/bold] {report.target}\n"
        f"[bold]Total Time:[/bold] {report.summary['total_time_ms']:.2f}ms\n"
        f"[bold]Total Calls:[/bold] {report.summary['total_calls']:,}\n"
        f"[bold]Bottlenecks:[/bold] {report.summary['bottlenecks']}"
    )

    if 'optimizations_found' in report.summary:
        summary_text += f"\n[bold]Optimizations:[/bold] {report.summary['optimizations_found']}"

    console.print(Panel(summary_text, title="[bold cyan]Performance Summary[/bold cyan]", expand=False))

    # Bottlenecks table
    if report.bottlenecks:
        console.print("\n[bold]Top Bottlenecks[/bold]")

        table = Table(show_header=True)
        table.add_column("Function", style="cyan")
        table.add_column("Time (ms)", justify="right")
        table.add_column("% Total", justify="right")
        table.add_column("Calls", justify="right")
        table.add_column("Severity", justify="center")

        for bottleneck in report.bottlenecks[:10]:
            severity_style = {
                Severity.CRITICAL: "[red]CRITICAL[/red]",
                Severity.HIGH: "[yellow]HIGH[/yellow]",
                Severity.MEDIUM: "[blue]MEDIUM[/blue]",
                Severity.LOW: "[green]LOW[/green]",
            }

            table.add_row(
                bottleneck.function_name[:40],
                f"{bottleneck.time_ms:.2f}",
                f"{bottleneck.percentage:.1f}%",
                f"{bottleneck.calls:,}",
                severity_style[bottleneck.severity]
            )

        console.print(table)

    # Optimizations
    if report.optimizations:
        console.print("\n[bold]Suggested Optimizations[/bold]")

        opt_table = Table(show_header=True)
        opt_table.add_column("Type", style="cyan")
        opt_table.add_column("Description", style="yellow")
        opt_table.add_column("Speedup", justify="right")
        opt_table.add_column("Confidence", justify="right")

        for opt in report.optimizations[:10]:
            opt_table.add_row(
                opt.type.value,
                opt.description[:60],
                f"{opt.estimated_speedup:.1f}x",
                f"{opt.confidence:.0%}"
            )

        console.print(opt_table)

        # Total speedup
        total_speedup = report.get_total_speedup()
        if total_speedup > 1.0:
            console.print(f"\n[bold green]Estimated Total Speedup: {total_speedup:.2f}x faster[/bold green]")


if __name__ == '__main__':
    main()
