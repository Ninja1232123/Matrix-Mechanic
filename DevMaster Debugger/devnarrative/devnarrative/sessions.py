"""
Work session detection from commits.
"""

from datetime import datetime, timedelta
from typing import Optional

from .models import Commit, CommitType, DayStory, WeekStory, WorkSession


class SessionDetector:
    """Detects work sessions from commit history."""

    def __init__(self, session_gap_minutes: int = 60):
        """
        Initialize session detector.

        Args:
            session_gap_minutes: Gap in minutes to consider separate sessions
        """
        self.session_gap = timedelta(minutes=session_gap_minutes)

    def detect_sessions(self, commits: list[Commit]) -> list[WorkSession]:
        """
        Group commits into work sessions.

        Args:
            commits: List of commits (should be sorted by date)

        Returns:
            List of work sessions
        """
        if not commits:
            return []

        # Sort by date
        sorted_commits = sorted(commits, key=lambda c: c.date)

        sessions = []
        current_session_commits = [sorted_commits[0]]
        session_start = sorted_commits[0].date

        for i in range(1, len(sorted_commits)):
            commit = sorted_commits[i]
            prev_commit = sorted_commits[i - 1]

            # Check if this commit belongs to current session
            time_gap = commit.date - prev_commit.date

            if time_gap <= self.session_gap:
                # Same session
                current_session_commits.append(commit)
            else:
                # New session - save current one
                session = self._create_session(session_start, current_session_commits)
                sessions.append(session)

                # Start new session
                current_session_commits = [commit]
                session_start = commit.date

        # Don't forget last session
        if current_session_commits:
            session = self._create_session(session_start, current_session_commits)
            sessions.append(session)

        return sessions

    def _create_session(self, start_time: datetime, commits: list[Commit]) -> WorkSession:
        """Create a WorkSession from commits."""
        end_time = commits[-1].date

        # Analyze session
        main_focus = self._determine_main_focus(commits)
        achievements = self._extract_achievements(commits)
        challenges = self._extract_challenges(commits)

        # Collect files
        files_modified = set()
        for commit in commits:
            files_modified.update(commit.files_changed)

        return WorkSession(
            start_time=start_time,
            end_time=end_time,
            commits=commits,
            main_focus=main_focus,
            achievements=achievements,
            challenges=challenges,
            files_modified=files_modified
        )

    def _determine_main_focus(self, commits: list[Commit]) -> str:
        """Determine the main focus of the session."""
        # Count commit types
        type_counts = {}
        for commit in commits:
            type_counts[commit.commit_type] = type_counts.get(commit.commit_type, 0) + 1

        # Find most common type
        if type_counts:
            main_type = max(type_counts, key=type_counts.get)

            # Find most frequently modified file/module
            file_counts = {}
            for commit in commits:
                for file in commit.files_changed:
                    # Extract module/directory
                    parts = file.split('/')
                    if len(parts) > 1:
                        module = parts[0]
                    else:
                        module = file

                    file_counts[module] = file_counts.get(module, 0) + 1

            main_module = max(file_counts, key=file_counts.get) if file_counts else "code"

            # Generate focus description
            focus_map = {
                CommitType.FEATURE: f"Building {main_module}",
                CommitType.BUGFIX: f"Fixing issues in {main_module}",
                CommitType.REFACTOR: f"Refactoring {main_module}",
                CommitType.TEST: f"Writing tests for {main_module}",
                CommitType.DOCS: f"Documenting {main_module}",
            }

            return focus_map.get(main_type, f"Working on {main_module}")

        return "Development work"

    def _extract_achievements(self, commits: list[Commit]) -> list[str]:
        """Extract key achievements from commits."""
        achievements = []

        # Feature implementations
        feature_commits = [c for c in commits if c.commit_type == CommitType.FEATURE]
        if feature_commits:
            achievements.append(f"Implemented {len(feature_commits)} new feature(s)")

        # Bug fixes
        bugfix_commits = [c for c in commits if c.commit_type == CommitType.BUGFIX]
        if bugfix_commits:
            achievements.append(f"Fixed {len(bugfix_commits)} bug(s)")

        # Tests added
        test_commits = [c for c in commits if c.commit_type == CommitType.TEST]
        if test_commits:
            achievements.append(f"Added {len(test_commits)} test(s)")

        # Large contributions
        total_insertions = sum(c.insertions for c in commits)
        if total_insertions > 500:
            achievements.append(f"Significant code additions (+{total_insertions} lines)")

        return achievements

    def _extract_challenges(self, commits: list[Commit]) -> list[str]:
        """Extract challenges from commit patterns."""
        challenges = []

        # Reverts indicate problems
        reverts = [c for c in commits if c.commit_type == CommitType.REVERT]
        if reverts:
            challenges.append("Had to revert some changes")

        # Many commits on same file indicates difficulty
        file_commits = {}
        for commit in commits:
            for file in commit.files_changed:
                file_commits[file] = file_commits.get(file, 0) + 1

        difficult_files = [f for f, count in file_commits.items() if count >= 4]
        if difficult_files:
            challenges.append(f"Complex work on {len(difficult_files)} file(s)")

        return challenges

    def create_day_story(self, date: datetime, commits: list[Commit]) -> DayStory:
        """Create a story for a single day."""
        # Filter commits for this day
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        day_commits = [
            c for c in commits
            if day_start <= c.date < day_end
        ]

        # Detect sessions
        sessions = self.detect_sessions(day_commits)

        # Create day story
        story = DayStory(
            date=date,
            sessions=sessions
        )

        # Generate summary
        if day_commits:
            story.summary = self._generate_day_summary(sessions)
            story.highlights = self._extract_day_highlights(sessions)
            story.challenges = self._extract_day_challenges(sessions)

        return story

    def create_week_story(
        self,
        start_date: datetime,
        commits: list[Commit]
    ) -> WeekStory:
        """Create a story for a week."""
        # Create day stories for each day
        day_stories = []
        current_date = start_date

        for _ in range(7):
            day_story = self.create_day_story(current_date, commits)
            day_stories.append(day_story)
            current_date += timedelta(days=1)

        # Create week story
        story = WeekStory(
            start_date=start_date,
            end_date=start_date + timedelta(days=6),
            day_stories=day_stories
        )

        # Generate summary
        story.summary = self._generate_week_summary(day_stories)
        story.major_achievements = self._extract_week_achievements(day_stories)
        story.patterns = self._extract_week_patterns(day_stories)

        return story

    def _generate_day_summary(self, sessions: list[WorkSession]) -> str:
        """Generate summary for a day."""
        if not sessions:
            return "No development activity"

        total_commits = sum(len(s.commits) for s in sessions)
        main_focuses = [s.main_focus for s in sessions if s.main_focus]

        if len(main_focuses) == 1:
            return f"Focused on {main_focuses[0].lower()} with {total_commits} commits"
        else:
            return f"Worked across multiple areas with {total_commits} commits"

    def _extract_day_highlights(self, sessions: list[WorkSession]) -> list[str]:
        """Extract highlights from a day's sessions."""
        highlights = []
        for session in sessions:
            highlights.extend(session.achievements)
        return list(set(highlights))[:3]  # Top 3 unique highlights

    def _extract_day_challenges(self, sessions: list[WorkSession]) -> list[str]:
        """Extract challenges from a day's sessions."""
        challenges = []
        for session in sessions:
            challenges.extend(session.challenges)
        return list(set(challenges))

    def _generate_week_summary(self, day_stories: list[DayStory]) -> str:
        """Generate summary for a week."""
        active_days = [d for d in day_stories if d.total_commits > 0]

        if not active_days:
            return "No development activity this week"

        total_commits = sum(d.total_commits for d in active_days)
        return f"Active {len(active_days)} days with {total_commits} total commits"

    def _extract_week_achievements(self, day_stories: list[DayStory]) -> list[str]:
        """Extract major achievements from the week."""
        all_highlights = []
        for day in day_stories:
            all_highlights.extend(day.highlights)

        # Count occurrences
        highlight_counts = {}
        for highlight in all_highlights:
            highlight_counts[highlight] = highlight_counts.get(highlight, 0) + 1

        # Return top 5 most common
        sorted_highlights = sorted(
            highlight_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [h for h, _ in sorted_highlights[:5]]

    def _extract_week_patterns(self, day_stories: list[DayStory]) -> list[str]:
        """Extract patterns from the week."""
        patterns = []

        # Check for consistent work times
        morning_days = sum(1 for d in day_stories
                          if any(s.time_of_day == "Morning" for s in d.sessions))
        if morning_days >= 4:
            patterns.append("Consistent morning work schedule")

        # Check for focus areas
        all_focuses = []
        for day in day_stories:
            for session in day.sessions:
                if session.main_focus:
                    all_focuses.append(session.main_focus)

        if len(set(all_focuses)) <= 2 and len(all_focuses) > 5:
            patterns.append("Maintained focus on specific areas")

        return patterns
