"""
Story generator that creates narratives from git history.
"""

from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .models import DayStory, FeatureStory, StoryConfig, WeekStory, WorkSession


class StoryGenerator:
    """Generates narrative stories from development data."""

    def __init__(self, config: Optional[StoryConfig] = None):
        self.config = config or StoryConfig()
        self.console = Console()

    def generate_day_story(self, day_story: DayStory) -> str:
        """Generate narrative for a day."""
        if not day_story.sessions:
            return self._format_no_activity(day_story.date)

        lines = []

        # Title
        date_str = day_story.date.strftime("%A, %B %d, %Y")
        lines.append(f"# 📖 Your Development Story - {date_str}\n")

        # Sessions
        for session in day_story.sessions:
            lines.append(self._format_session(session))

        # Daily Summary
        lines.append("## Daily Summary\n")
        lines.append(f"**{day_story.summary}**\n")

        insertions, deletions = day_story.total_changes
        lines.append(f"📊 Total Activity: {day_story.total_commits} commits | +{insertions} -{deletions} lines")
        lines.append(f"⏱️  Active Time: {day_story.active_hours:.1f} hours\n")

        if day_story.highlights:
            lines.append("**Key Achievements:**")
            for highlight in day_story.highlights:
                lines.append(f"- {highlight}")
            lines.append("")

        if day_story.challenges:
            lines.append("**Challenges Overcome:**")
            for challenge in day_story.challenges:
                lines.append(f"- {challenge}")
            lines.append("")

        return "\n".join(lines)

    def generate_week_story(self, week_story: WeekStory) -> str:
        """Generate narrative for a week."""
        lines = []

        # Title
        start_str = week_story.start_date.strftime("%B %d")
        end_str = week_story.end_date.strftime("%B %d, %Y")
        lines.append(f"# 📅 Your Week in Development - {start_str} to {end_str}\n")

        # Weekly Overview
        lines.append("## Overview\n")
        lines.append(f"**{week_story.summary}**\n")

        insertions, deletions = week_story.total_changes
        lines.append(f"📊 Total Activity: {week_story.total_commits} commits | +{insertions} -{deletions} lines")
        lines.append(f"📅 Active Days: {week_story.active_days}/7\n")

        # Day by day
        lines.append("## Day by Day\n")

        for day_story in week_story.day_stories:
            if day_story.total_commits > 0:
                day_name = day_story.date.strftime("%A, %B %d")
                lines.append(f"### {day_name}\n")

                if day_story.sessions:
                    for session in day_story.sessions:
                        lines.append(f"**{session.time_of_day} Session:** {session.main_focus}")
                        lines.append(f"  - {len(session.commits)} commits")
                        ins, dels = session.total_changes
                        lines.append(f"  - +{ins} -{dels} lines\n")

        # Weekly Achievements
        if week_story.major_achievements:
            lines.append("## Major Achievements\n")
            for achievement in week_story.major_achievements:
                lines.append(f"- {achievement}")
            lines.append("")

        # Patterns
        if week_story.patterns:
            lines.append("## Patterns & Insights\n")
            for pattern in week_story.patterns:
                lines.append(f"- {pattern}")
            lines.append("")

        return "\n".join(lines)

    def generate_feature_story(self, feature_story: FeatureStory) -> str:
        """Generate narrative for a feature."""
        lines = []

        # Title
        lines.append(f"# 🚀 Feature Story: {feature_story.feature_name}\n")

        # Timeline
        start_str = feature_story.start_date.strftime("%B %d, %Y")
        end_str = feature_story.end_date.strftime("%B %d, %Y")
        lines.append(f"**Development Period:** {start_str} to {end_str} ({feature_story.duration_days} days)")
        lines.append(f"**Total Commits:** {len(feature_story.commits)}\n")

        # Overview
        if feature_story.overview:
            lines.append("## Overview\n")
            lines.append(f"{feature_story.overview}\n")

        # Development Phases
        if feature_story.phases:
            lines.append("## Development Phases\n")
            for i, phase in enumerate(feature_story.phases, 1):
                lines.append(f"{i}. {phase}")
            lines.append("")

        # Key Decisions
        if feature_story.key_decisions:
            lines.append("## Key Technical Decisions\n")
            for decision in feature_story.key_decisions:
                lines.append(f"- {decision}")
            lines.append("")

        # Challenges
        if feature_story.challenges:
            lines.append("## Challenges & Solutions\n")
            for challenge in feature_story.challenges:
                lines.append(f"- {challenge}")
            lines.append("")

        # Final State
        if feature_story.final_state:
            lines.append("## Final State\n")
            lines.append(f"{feature_story.final_state}\n")

        return "\n".join(lines)

    def _format_session(self, session: WorkSession) -> str:
        """Format a work session."""
        lines = []

        # Session header
        start_time = session.start_time.strftime("%I:%M %p")
        lines.append(f"## {session.time_of_day} Session - {start_time}\n")

        # Main focus
        lines.append(f"### 🎯 {session.main_focus}\n")

        # Narrative
        narrative = self._generate_session_narrative(session)
        lines.append(f"{narrative}\n")

        # Statistics
        insertions, deletions = session.total_changes
        lines.append(f"**Statistics:**")
        lines.append(f"- {len(session.commits)} commits")
        lines.append(f"- +{insertions} -{deletions} lines")
        lines.append(f"- {len(session.files_modified)} files modified")
        lines.append(f"- Duration: {session.duration_hours:.1f} hours\n")

        # Achievements
        if session.achievements:
            lines.append("**Achievements:**")
            for achievement in session.achievements:
                lines.append(f"- ✅ {achievement}")
            lines.append("")

        # Challenges
        if session.challenges:
            lines.append("**Challenges:**")
            for challenge in session.challenges:
                lines.append(f"- ⚠️  {challenge}")
            lines.append("")

        return "\n".join(lines)

    def _generate_session_narrative(self, session: WorkSession) -> str:
        """Generate narrative text for a session."""
        commits = session.commits

        if len(commits) == 1:
            return f"You made a single commit focusing on {session.main_focus.lower()}."

        # Analyze commit progression
        first_commit = commits[0].summary
        last_commit = commits[-1].summary

        narrative = f"You started by {first_commit.lower()}. "

        if len(commits) > 3:
            narrative += f"After several iterations, "
        else:
            narrative += ""

        narrative += f"you completed the work by {last_commit.lower()}."

        # Add challenge context if exists
        if session.challenges:
            narrative += f" The process involved overcoming {session.challenges[0].lower()}."

        return narrative

    def _format_no_activity(self, date: datetime) -> str:
        """Format message for days with no activity."""
        date_str = date.strftime("%A, %B %d, %Y")
        return f"# 📖 Development Story - {date_str}\n\nNo development activity recorded for this day.\n"

    def print_day_story(self, day_story: DayStory):
        """Print day story to terminal with rich formatting."""
        markdown = self.generate_day_story(day_story)
        md = Markdown(markdown)
        self.console.print(md)

    def print_week_story(self, week_story: WeekStory):
        """Print week story to terminal with rich formatting."""
        markdown = self.generate_week_story(week_story)
        md = Markdown(markdown)
        self.console.print(md)

    def print_feature_story(self, feature_story: FeatureStory):
        """Print feature story to terminal with rich formatting."""
        markdown = self.generate_feature_story(feature_story)
        md = Markdown(markdown)
        self.console.print(md)

    def export_to_file(self, content: str, filename: str, format: str = "markdown"):
        """Export story to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            if format == "markdown":
                f.write(content)
            elif format == "html":
                f.write(self._convert_to_html(content))
            else:
                f.write(content)

        self.console.print(f"✅ Story exported to {filename}", style="green")

    def _convert_to_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML (basic implementation)."""
        # Basic HTML wrapper
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Development Story</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
<pre>{markdown_content}</pre>
</body>
</html>"""
        return html
