"""
Command-line interface for CodeArchaeology.
"""

from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .analyzer import CodeArchaeologist
from .models import RiskLevel

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """
    CodeArchaeology - Dig through your code's history.

    Understand evolution, discover patterns, find hotspots, and learn from the past.
    """
    pass


@main.command()
@click.option('--repo', default='.', help='Repository path')
def init(repo: str):
    """Initialize CodeArchaeology in a repository."""
    try:
        archaeologist = CodeArchaeologist(repo)
        console.print(f"✅ CodeArchaeology initialized", style="green bold")
        console.print(f"   Repository: {archaeologist.repo.working_dir}", style="dim")

        # Quick stats
        stats = archaeologist.get_summary_stats()
        console.print(f"   Files tracked: {stats['total_files']}", style="dim")
        console.print(f"   Total changes: {stats['total_changes']}", style="dim")

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
@click.option('--threshold', default=10, help='Minimum changes to be a hotspot')
@click.option('--limit', default=20, help='Number of hotspots to show')
@click.option('--since', help='Since date (YYYY-MM-DD)')
def hotspots(repo: str, threshold: int, limit: int, since: str):
    """Find code hotspots (frequently changed files)."""
    try:
        archaeologist = CodeArchaeologist(repo)

        # Parse since date
        since_date = None
        if since:
            since_date = datetime.strptime(since, '%Y-%m-%d')

        hotspots_list = archaeologist.find_hotspots(threshold=threshold, since=since_date)

        if not hotspots_list:
            console.print("✨ No hotspots found - your code is stable!", style="green")
            return

        console.print(f"\n🔥 [bold]Code Hotspots[/bold] (Top {min(limit, len(hotspots_list))})\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", justify="right")
        table.add_column("File")
        table.add_column("Changes", justify="right")
        table.add_column("Churn Rate", justify="right")
        table.add_column("Risk")
        table.add_column("Last Changed")

        for i, hotspot in enumerate(hotspots_list[:limit], 1):
            # Risk indicator
            risk_styles = {
                RiskLevel.LOW: "green",
                RiskLevel.MODERATE: "yellow",
                RiskLevel.HIGH: "orange",
                RiskLevel.CRITICAL: "red bold"
            }
            risk_style = risk_styles.get(hotspot.risk_level, "white")
            risk_text = Text(hotspot.risk_level.value.upper(), style=risk_style)

            # Format churn rate
            churn = f"{hotspot.churn_rate:.2f}/day"

            # Format last changed
            days_ago = (datetime.now() - hotspot.last_changed).days
            if days_ago == 0:
                last_changed = "today"
            elif days_ago == 1:
                last_changed = "yesterday"
            else:
                last_changed = f"{days_ago}d ago"

            table.add_row(
                str(i),
                hotspot.file_path,
                str(hotspot.change_count),
                churn,
                risk_text,
                last_changed
            )

        console.print(table)

        # Analysis
        critical = len([h for h in hotspots_list if h.risk_level == RiskLevel.CRITICAL])
        high = len([h for h in hotspots_list if h.risk_level == RiskLevel.HIGH])

        console.print(f"\n📊 Analysis:")
        if critical > 0:
            console.print(f"   ⚠️  {critical} CRITICAL risk file(s) - immediate attention needed", style="red bold")
        if high > 0:
            console.print(f"   🔶 {high} HIGH risk file(s) - consider refactoring", style="yellow")
        if critical == 0 and high == 0:
            console.print(f"   ✅ All hotspots are manageable", style="green")

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
@click.option('--age', default=180, help='Minimum days since last change')
@click.option('--limit', default=20, help='Number of results to show')
def abandoned(repo: str, age: int, limit: int):
    """Find potentially abandoned code."""
    try:
        archaeologist = CodeArchaeologist(repo)
        abandoned_list = archaeologist.find_abandoned_code(age_threshold_days=age)

        if not abandoned_list:
            console.print("✨ No abandoned code found!", style="green")
            return

        console.print(f"\n🏚️  [bold]Potentially Abandoned Code[/bold]\n")

        for i, abandoned in enumerate(abandoned_list[:limit], 1):
            # Calculate age
            months_abandoned = abandoned.days_abandoned // 30

            # Determine recommendation
            if abandoned.is_truly_abandoned:
                recommendation = "🗑️  Recommend: Archive or remove"
                style = "red"
            else:
                recommendation = "⚠️  Investigate: May still be needed"
                style = "yellow"

            panel_content = f"""[bold]{abandoned.file_path}[/bold]

Last modified: {abandoned.last_modified.strftime('%Y-%m-%d')} ({months_abandoned} months ago)
Initial activity: {abandoned.initial_commits} commits
Created: {abandoned.creation_date.strftime('%Y-%m-%d')}

{recommendation}
"""

            console.print(Panel(panel_content, border_style=style, title=f"#{i}"))

        # Summary
        truly_abandoned = len([a for a in abandoned_list if a.is_truly_abandoned])
        console.print(f"\n📊 Found {len(abandoned_list)} old files, {truly_abandoned} appear truly abandoned")

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
@click.option('--threshold', default=0.5, help='Minimum coupling score (0-1)')
@click.option('--limit', default=15, help='Number of pairs to show')
def coupling(repo: str, threshold: float, limit: int):
    """Find temporal coupling (files that change together)."""
    try:
        archaeologist = CodeArchaeologist(repo)
        coupling_list = archaeologist.analyze_coupling(coupling_threshold=threshold)

        if not coupling_list:
            console.print("✨ No significant coupling found", style="green")
            return

        console.print(f"\n🔗 [bold]Temporal Coupling[/bold] (files that change together)\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", justify="right")
        table.add_column("File 1")
        table.add_column("File 2")
        table.add_column("Coupling", justify="right")
        table.add_column("Changes", justify="right")

        for i, coupling in enumerate(coupling_list[:limit], 1):
            coupling_pct = f"{coupling.coupling_score * 100:.1f}%"

            table.add_row(
                str(i),
                coupling.file1,
                coupling.file2,
                coupling_pct,
                str(coupling.change_count)
            )

        console.print(table)

        console.print(f"\n💡 Tip: High coupling may indicate:")
        console.print(f"   - Related functionality that could be refactored together")
        console.print(f"   - Hidden dependencies between modules")
        console.print(f"   - Opportunities for better code organization")

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
@click.option('--limit', default=20, help='Number of files to show')
def knowledge(repo: str, limit: int):
    """Analyze knowledge distribution (who knows what)."""
    try:
        archaeologist = CodeArchaeologist(repo)
        knowledge_maps = archaeologist.analyze_knowledge_distribution()

        if not knowledge_maps:
            console.print("No knowledge data available", style="yellow")
            return

        console.print(f"\n🧠 [bold]Knowledge Distribution[/bold]\n")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("File")
        table.add_column("Authors", justify="right")
        table.add_column("Bus Factor", justify="right")
        table.add_column("Primary Author")
        table.add_column("Concentration")

        for knowledge in knowledge_maps[:limit]:
            concentration_pct = f"{knowledge.knowledge_concentration * 100:.0f}%"

            # Color code bus factor
            if knowledge.bus_factor == 1:
                bus_style = "red"
            elif knowledge.bus_factor == 2:
                bus_style = "yellow"
            else:
                bus_style = "green"

            bus_text = Text(str(knowledge.bus_factor), style=bus_style)

            table.add_row(
                knowledge.file_path,
                str(len(knowledge.authors)),
                bus_text,
                knowledge.primary_author or "N/A",
                concentration_pct
            )

        console.print(table)

        # Analysis
        silos = len([k for k in knowledge_maps if k.bus_factor == 1])
        console.print(f"\n📊 Analysis:")
        if silos > 10:
            console.print(f"   ⚠️  {silos} files have bus factor of 1 (knowledge silos)", style="red")
            console.print(f"   💡 Consider pair programming or code reviews", style="dim")
        elif silos > 0:
            console.print(f"   🔶 {silos} files have bus factor of 1", style="yellow")
        else:
            console.print(f"   ✅ Good knowledge distribution!", style="green")

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
def patterns(repo: str):
    """Detect patterns in code evolution."""
    try:
        archaeologist = CodeArchaeologist(repo)
        patterns_list = archaeologist.detect_patterns()

        if not patterns_list:
            console.print("No significant patterns detected", style="yellow")
            return

        console.print(f"\n🔍 [bold]Detected Patterns[/bold]\n")

        for i, pattern in enumerate(patterns_list, 1):
            confidence_pct = int(pattern.confidence * 100)

            # Pattern type emoji
            type_emoji = {
                "bug_cluster": "🐛",
                "refactor_cycle": "♻️",
                "knowledge_silo": "🏝️"
            }.get(pattern.pattern_type, "📌")

            panel_content = f"""[bold]{pattern.description}[/bold]

Confidence: {confidence_pct}%

Affected files ({len(pattern.files)} total):
"""

            for file in pattern.files[:5]:
                panel_content += f"  • {file}\n"

            if len(pattern.files) > 5:
                panel_content += f"  ... and {len(pattern.files) - 5} more\n"

            panel_content += f"\nEvidence:\n"
            for evidence in pattern.evidence:
                panel_content += f"  • {evidence}\n"

            console.print(Panel(
                panel_content,
                title=f"{type_emoji} {pattern.pattern_type}",
                border_style="cyan"
            ))

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
def stats(repo: str):
    """Show repository statistics and summary."""
    try:
        archaeologist = CodeArchaeologist(repo)
        stats = archaeologist.get_summary_stats()

        console.print(f"\n📊 [bold]Repository Statistics[/bold]\n")

        # Main stats
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Files", f"{stats['total_files']:,}")
        table.add_row("Total Changes", f"{stats['total_changes']:,}")
        table.add_row("Total Authors", str(stats['total_authors']))
        table.add_row("Hotspots", str(stats['hotspot_count']))
        table.add_row("High Risk Files", str(stats['high_risk_files']))

        console.print(table)

        # Age distribution
        console.print(f"\n📅 [bold]File Age Distribution[/bold]\n")

        age_table = Table(show_header=True, header_style="bold cyan")
        age_table.add_column("Age Range")
        age_table.add_column("Files", justify="right")

        for age_range, count in stats['age_distribution'].items():
            age_table.add_row(age_range, str(count))

        console.print(age_table)

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


if __name__ == '__main__':
    main()
