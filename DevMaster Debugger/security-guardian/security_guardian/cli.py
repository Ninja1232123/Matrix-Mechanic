"""Command-line interface for Security-Guardian"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core import SecurityGuardian
from .models import Severity

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def main():
    """Security-Guardian: Automatic Security Vulnerability Scanner"""
    pass


@main.command()
@click.argument('target', type=click.Path(exists=True))
@click.option('--recursive/--no-recursive', default=True, help='Scan subdirectories')
@click.option('--output', '-o', type=click.Path(), help='Output report file')
def scan(target, recursive, output):
    """Scan Python files for security vulnerabilities"""
    console.print(f"\n[bold red]Security-Guardian Scanner[/bold red]")
    console.print(f"Scanning: [yellow]{target}[/yellow]\n")

    sg = SecurityGuardian()
    target_path = Path(target)

    # Scan target
    if target_path.is_file():
        report = sg.scan_file(target_path)
    else:
        report = sg.scan_directory(target_path, recursive=recursive)

    # Display results
    _display_report(report)

    # Save report if requested
    if output:
        report_text = sg.generate_report_text(report)
        with open(output, 'w') as f:
            f.write(report_text)
        console.print(f"\n[green]Report saved to {output}[/green]")


def _display_report(report):
    """Display security report using Rich"""
    # Summary panel
    summary_text = (
        f"[bold]Target:[/bold] {report.target}\n"
        f"[bold]Total Issues:[/bold] {report.get_total_issues()}\n"
        f"[bold red]Critical:[/bold red] {report.get_critical_count()}\n"
        f"[bold yellow]High:[/bold yellow] {report.get_high_count()}\n"
        f"[bold]Secrets:[/bold] {len(report.secrets)}\n"
        f"[bold]Vulnerable Deps:[/bold] {len(report.dependency_issues)}"
    )

    console.print(Panel(summary_text, title="[bold red]Security Scan Results[/bold red]", expand=False))

    # Vulnerabilities table
    if report.vulnerabilities:
        console.print("\n[bold]Vulnerabilities[/bold]")

        table = Table()
        table.add_column("Severity", style="red")
        table.add_column("Type")
        table.add_column("Location")
        table.add_column("Description", style="yellow")

        for vuln in report.vulnerabilities[:15]:
            severity_style = {
                Severity.CRITICAL: "[red bold]CRITICAL[/red bold]",
                Severity.HIGH: "[yellow]HIGH[/yellow]",
                Severity.MEDIUM: "[blue]MEDIUM[/blue]",
                Severity.LOW: "[green]LOW[/green]",
            }

            table.add_row(
                severity_style[vuln.severity],
                vuln.type.value,
                f"{Path(vuln.file_path).name}:{vuln.line_number}",
                vuln.description[:60]
            )

        console.print(table)

    # Secrets found
    if report.secrets:
        console.print(f"\n[bold red]⚠️  {len(report.secrets)} Hardcoded Secrets Detected![/bold red]")
        console.print("[dim]Review and move to environment variables or secret management[/dim]")

    # Dependencies
    if report.dependency_issues:
        console.print(f"\n[bold yellow]⚠️  {len(report.dependency_issues)} Vulnerable Dependencies![/bold yellow]")
        for dep in report.dependency_issues:
            console.print(f"  - {dep.package_name} v{dep.current_version}: {', '.join(dep.vulnerabilities)}")


if __name__ == '__main__':
    main()
