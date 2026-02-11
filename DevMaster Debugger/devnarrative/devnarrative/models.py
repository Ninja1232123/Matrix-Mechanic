"""
Data models for DevNarrative.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class CommitType(Enum):
    """Types of commits based on their nature."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    CHORE = "chore"
    MERGE = "merge"
    REVERT = "revert"
    UNKNOWN = "unknown"


@dataclass
class Commit:
    """Represents a git commit."""
    hash: str
    author: str
    email: str
    date: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0

    # Analysis
    commit_type: CommitType = CommitType.UNKNOWN
    branch: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @property
    def short_hash(self) -> str:
        """Get short commit hash."""
        return self.hash[:8]

    @property
    def summary(self) -> str:
        """Get first line of commit message."""
        return self.message.split('\n')[0]

    @property
    def body(self) -> str:
        """Get commit message body."""
        lines = self.message.split('\n')
        return '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

    def __str__(self) -> str:
        return f"{self.short_hash}: {self.summary}"


@dataclass
class WorkSession:
    """Represents a work session (group of related commits)."""
    start_time: datetime
    end_time: datetime
    commits: list[Commit] = field(default_factory=list)

    # Analysis
    main_focus: str = ""
    achievements: list[str] = field(default_factory=list)
    challenges: list[str] = field(default_factory=list)
    files_modified: set[str] = field(default_factory=set)

    @property
    def duration_hours(self) -> float:
        """Get session duration in hours."""
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 3600

    @property
    def total_changes(self) -> tuple[int, int]:
        """Get total insertions and deletions."""
        insertions = sum(c.insertions for c in self.commits)
        deletions = sum(c.deletions for c in self.commits)
        return (insertions, deletions)

    @property
    def time_of_day(self) -> str:
        """Get time of day classification."""
        hour = self.start_time.hour
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    def __str__(self) -> str:
        return f"{self.time_of_day} Session: {len(self.commits)} commits, {self.duration_hours:.1f}h"


@dataclass
class DayStory:
    """Story of a day's work."""
    date: datetime
    sessions: list[WorkSession] = field(default_factory=list)

    # Summary
    summary: str = ""
    highlights: list[str] = field(default_factory=list)
    challenges: list[str] = field(default_factory=list)

    @property
    def total_commits(self) -> int:
        """Total commits in the day."""
        return sum(len(s.commits) for s in self.sessions)

    @property
    def total_changes(self) -> tuple[int, int]:
        """Total changes in the day."""
        insertions = sum(ins for s in self.sessions for ins, _ in [s.total_changes])
        deletions = sum(dels for s in self.sessions for _, dels in [s.total_changes])
        return (insertions, deletions)

    @property
    def active_hours(self) -> float:
        """Total active hours."""
        return sum(s.duration_hours for s in self.sessions)

    def __str__(self) -> str:
        date_str = self.date.strftime("%A, %B %d, %Y")
        return f"Story of {date_str}: {self.total_commits} commits"


@dataclass
class WeekStory:
    """Story of a week's work."""
    start_date: datetime
    end_date: datetime
    day_stories: list[DayStory] = field(default_factory=list)

    # Summary
    summary: str = ""
    major_achievements: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)

    @property
    def total_commits(self) -> int:
        """Total commits in the week."""
        return sum(day.total_commits for day in self.day_stories)

    @property
    def total_changes(self) -> tuple[int, int]:
        """Total changes in the week."""
        insertions = sum(ins for day in self.day_stories for ins, _ in [day.total_changes])
        deletions = sum(dels for day in self.day_stories for _, dels in [day.total_changes])
        return (insertions, deletions)

    @property
    def active_days(self) -> int:
        """Number of days with commits."""
        return len([d for d in self.day_stories if d.total_commits > 0])

    def __str__(self) -> str:
        return f"Week of {self.start_date.strftime('%b %d')}: {self.total_commits} commits, {self.active_days} days"


@dataclass
class FeatureStory:
    """Story of a feature's development."""
    feature_name: str
    start_date: datetime
    end_date: datetime
    commits: list[Commit] = field(default_factory=list)

    # Narrative
    overview: str = ""
    phases: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    challenges: list[str] = field(default_factory=list)
    final_state: str = ""

    @property
    def duration_days(self) -> int:
        """Duration in days."""
        delta = self.end_date - self.start_date
        return delta.days + 1

    def __str__(self) -> str:
        return f"Feature '{self.feature_name}': {len(self.commits)} commits over {self.duration_days} days"


@dataclass
class StoryConfig:
    """Configuration for story generation."""
    style: str = "detailed"  # detailed, concise, technical
    include_stats: bool = True
    include_code_snippets: bool = False
    max_commit_detail: int = 5
    highlight_challenges: bool = True
    exclude_patterns: list[str] = field(default_factory=list)
