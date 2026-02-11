"""
Command-line interface for DevNarrative.
"""

from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .analyzer import GitAnalyzer
from .models import FeatureStory, StoryConfig
from .sessions import SessionDetector
from .story import StoryGenerator

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """
    DevNarrative - Transform git history into human stories.

    Generate readable narratives from your development work.
    """
    pass


@main.command()
@click.option('--repo', default='.', help='Repository path')
def init(repo: str):
    """Initialize DevNarrative in a repository."""
    try:
        analyzer = GitAnalyzer(repo)
        console.print(f"✅ DevNarrative initialized in {analyzer.repo_path}", style="green")
        console.print(f"   Repository: {analyzer.repo.working_dir}", style="dim")

        # Show quick stats
        recent_commits = analyzer.get_commits_this_week()
        console.print(f"   This week: {len(recent_commits)} commits", style="dim")

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
@click.option('--author', help='Filter by author email')
@click.option('--export', help='Export to file')
@click.option('--format', type=click.Choice(['terminal', 'markdown', 'html']), default='terminal')
def today(repo: str, author: str, export: str, format: str):
    """Generate story for today's work."""
    try:
        analyzer = GitAnalyzer(repo)
        commits = analyzer.get_commits_today(author=author)

        if not commits:
            console.print("📭 No commits found for today", style="yellow")
            return

        # Detect sessions and create story
        detector = SessionDetector()
        day_story = detector.create_day_story(datetime.now(), commits)

        # Generate and display story
        generator = StoryGenerator()

        if format == 'terminal':
            generator.print_day_story(day_story)
        else:
            content = generator.generate_day_story(day_story)
            if export:
                generator.export_to_file(content, export, format)
            else:
                console.print(content)

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
@click.option('--author', help='Filter by author email')
@click.option('--export', help='Export to file')
@click.option('--format', type=click.Choice(['terminal', 'markdown', 'html']), default='terminal')
def week(repo: str, author: str, export: str, format: str):
    """Generate story for this week's work."""
    try:
        analyzer = GitAnalyzer(repo)
        commits = analyzer.get_commits_this_week(author=author)

        if not commits:
            console.print("📭 No commits found for this week", style="yellow")
            return

        # Get Monday of this week
        now = datetime.now()
        monday = now - timedelta(days=now.weekday())
        monday_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        # Create week story
        detector = SessionDetector()
        week_story = detector.create_week_story(monday_start, commits)

        # Generate and display story
        generator = StoryGenerator()

        if format == 'terminal':
            generator.print_week_story(week_story)
        else:
            content = generator.generate_week_story(week_story)
            if export:
                generator.export_to_file(content, export, format)
            else:
                console.print(content)

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.argument('feature_name')
@click.option('--repo', default='.', help='Repository path')
@click.option('--author', help='Filter by author email')
@click.option('--export', help='Export to file')
def feature(feature_name: str, repo: str, author: str, export: str):
    """Generate story for a specific feature."""
    try:
        analyzer = GitAnalyzer(repo)
        commits = analyzer.get_feature_commits(feature_name, author=author)

        if not commits:
            console.print(f"📭 No commits found for feature: {feature_name}", style="yellow")
            return

        # Create feature story
        feature_story = FeatureStory(
            feature_name=feature_name,
            start_date=min(c.date for c in commits),
            end_date=max(c.date for c in commits),
            commits=commits
        )

        # Generate overview
        feature_story.overview = _generate_feature_overview(commits)

        # Generate and display story
        generator = StoryGenerator()
        generator.print_feature_story(feature_story)

        if export:
            content = generator.generate_feature_story(feature_story)
            generator.export_to_file(content, export, 'markdown')

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--from', 'from_date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--to', 'to_date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--repo', default='.', help='Repository path')
@click.option('--author', help='Filter by author email')
@click.option('--export', help='Export to file')
def range(from_date: str, to_date: str, repo: str, author: str, export: str):
    """Generate story for a date range."""
    try:
        # Parse dates
        start_date = datetime.strptime(from_date, '%Y-%m-%d')
        end_date = datetime.strptime(to_date, '%Y-%m-%d')

        analyzer = GitAnalyzer(repo)
        commits = analyzer.get_commits_in_range(start_date, end_date, author=author)

        if not commits:
            console.print(f"📭 No commits found in range {from_date} to {to_date}", style="yellow")
            return

        # Create week story (reusing for range)
        detector = SessionDetector()
        week_story = detector.create_week_story(start_date, commits)

        # Override dates
        week_story.start_date = start_date
        week_story.end_date = end_date

        # Generate and display story
        generator = StoryGenerator()
        generator.print_week_story(week_story)

        if export:
            content = generator.generate_week_story(week_story)
            generator.export_to_file(content, export, 'markdown')

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


@main.command()
@click.option('--repo', default='.', help='Repository path')
@click.option('--since', help='Since date (YYYY-MM-DD)')
def stats(repo: str, since: str):
    """Show repository statistics."""
    try:
        analyzer = GitAnalyzer(repo)

        # Parse since date
        since_date = None
        if since:
            since_date = datetime.strptime(since, '%Y-%m-%d')

        # Get stats
        author_stats = analyzer.get_author_stats(since=since_date)

        # Display
        console.print("\n📊 [bold]Repository Statistics[/bold]\n")

        if since_date:
            console.print(f"Since: {since_date.strftime('%B %d, %Y')}\n")

        from rich.table import Table

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Author")
        table.add_column("Commits", justify="right")
        table.add_column("Insertions", justify="right")
        table.add_column("Deletions", justify="right")
        table.add_column("Files", justify="right")

        for author, stats_data in author_stats.items():
            table.add_row(
                author,
                str(stats_data['commits']),
                f"+{stats_data['insertions']}",
                f"-{stats_data['deletions']}",
                str(stats_data['files'])
            )

        console.print(table)

    except ValueError as e:
        console.print(f"❌ {e}", style="red")
        raise click.Abort()


def _generate_feature_overview(commits: list) -> str:
    """Generate feature overview from commits."""
    if not commits:
        return "No commits"

    first_commit = commits[0].summary
    last_commit = commits[-1].summary

    return (
        f"This feature was developed through {len(commits)} commits, "
        f"starting with '{first_commit}' and culminating in '{last_commit}'."
    )


if __name__ == '__main__':
    main()
