"""
Data models for CodeArchaeology.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    """Risk level for code."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FileHistory:
    """History of changes to a file."""
    file_path: str
    total_changes: int = 0
    authors: set[str] = field(default_factory=set)
    first_commit: Optional[datetime] = None
    last_commit: Optional[datetime] = None
    lines_added: int = 0
    lines_deleted: int = 0

    @property
    def age_days(self) -> int:
        """Age of file in days."""
        if self.first_commit and self.last_commit:
            return (self.last_commit - self.first_commit).days
        return 0

    @property
    def days_since_change(self) -> int:
        """Days since last change."""
        if self.last_commit:
            return (datetime.now() - self.last_commit).days
        return 0

    @property
    def churn_rate(self) -> float:
        """Changes per day."""
        if self.age_days > 0:
            return self.total_changes / self.age_days
        return 0.0

    @property
    def risk_level(self) -> RiskLevel:
        """Assess risk level based on churn."""
        if self.churn_rate > 1.0:  # More than 1 change per day
            return RiskLevel.CRITICAL
        elif self.churn_rate > 0.5:
            return RiskLevel.HIGH
        elif self.churn_rate > 0.2:
            return RiskLevel.MODERATE
        return RiskLevel.LOW

    def __str__(self) -> str:
        return f"{self.file_path}: {self.total_changes} changes by {len(self.authors)} author(s)"


@dataclass
class Hotspot:
    """Code hotspot (frequently changed code)."""
    file_path: str
    change_count: int
    risk_level: RiskLevel
    churn_rate: float
    last_changed: datetime
    primary_authors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.file_path} ({change_count} changes, {self.risk_level.value} risk)"


@dataclass
class CodeLineage:
    """Evolution of a specific code element (function, class)."""
    file_path: str
    element_name: str
    commits: list = field(default_factory=list)  # List of commit info

    @property
    def first_appearance(self) -> Optional[datetime]:
        """When this code first appeared."""
        if self.commits:
            return self.commits[0].get('date')
        return None

    @property
    def last_modified(self) -> Optional[datetime]:
        """When this code was last modified."""
        if self.commits:
            return self.commits[-1].get('date')
        return None

    @property
    def modification_count(self) -> int:
        """Number of times modified."""
        return len(self.commits)

    @property
    def stability_score(self) -> float:
        """Stability score (0-1, higher is more stable)."""
        if not self.commits:
            return 1.0

        # Consider both frequency and recent changes
        days_since_change = (datetime.now() - self.last_modified).days if self.last_modified else 0

        if days_since_change > 90:  # No changes in 3 months
            return 0.9
        elif days_since_change > 30:
            return 0.7
        elif self.modification_count < 5:
            return 0.6
        else:
            return 0.3  # Frequently changed and recent

    def __str__(self) -> str:
        return f"{self.element_name} in {self.file_path} ({self.modification_count} changes)"


@dataclass
class AbandonedCode:
    """Potentially abandoned code."""
    file_path: str
    last_modified: datetime
    creation_date: datetime
    initial_commits: int  # How many commits when active
    days_abandoned: int
    referenced_by: list[str] = field(default_factory=list)

    @property
    def is_truly_abandoned(self) -> bool:
        """Check if code appears to be truly abandoned."""
        # Abandoned if: old, no references, had initial activity
        return (
            self.days_abandoned > 180 and  # 6 months old
            len(self.referenced_by) == 0 and
            self.initial_commits >= 3  # Had real development
        )

    def __str__(self) -> str:
        return f"{self.file_path} (abandoned {self.days_abandoned} days)"


@dataclass
class CouplingInfo:
    """Information about temporal coupling between files."""
    file1: str
    file2: str
    coupling_score: float  # 0-1, how often they change together
    change_count: int

    def __str__(self) -> str:
        score_pct = int(self.coupling_score * 100)
        return f"{self.file1} <-> {self.file2} ({score_pct}% coupled)"


@dataclass
class KnowledgeMap:
    """Map of who knows what code."""
    file_path: str
    authors: dict[str, int] = field(default_factory=dict)  # author -> commit count

    @property
    def primary_author(self) -> Optional[str]:
        """Author with most commits."""
        if self.authors:
            return max(self.authors, key=self.authors.get)
        return None

    @property
    def knowledge_concentration(self) -> float:
        """How concentrated knowledge is (0-1, higher = more concentrated)."""
        if not self.authors:
            return 0.0

        total = sum(self.authors.values())
        if total == 0:
            return 0.0

        # Calculate concentration using Gini coefficient approach
        sorted_counts = sorted(self.authors.values(), reverse=True)
        if len(sorted_counts) == 1:
            return 1.0

        primary_ratio = sorted_counts[0] / total
        return primary_ratio

    @property
    def bus_factor(self) -> int:
        """Minimum number of people who must leave to lose knowledge."""
        if not self.authors:
            return 0

        total = sum(self.authors.values())
        sorted_counts = sorted(self.authors.values(), reverse=True)

        cumsum = 0
        for i, count in enumerate(sorted_counts):
            cumsum += count
            if cumsum > total * 0.5:  # More than 50% of knowledge
                return i + 1

        return len(self.authors)

    def __str__(self) -> str:
        return f"{self.file_path}: {len(self.authors)} authors, bus factor: {self.bus_factor}"


@dataclass
class Pattern:
    """Detected pattern in code evolution."""
    pattern_type: str  # "bug_cluster", "refactor_cycle", etc.
    description: str
    files: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.pattern_type}: {self.description} ({int(self.confidence * 100)}%)"


@dataclass
class ArchaeologyReport:
    """Complete archaeology report."""
    generated_at: datetime = field(default_factory=datetime.now)
    hotspots: list[Hotspot] = field(default_factory=list)
    abandoned_code: list[AbandonedCode] = field(default_factory=list)
    coupling: list[CouplingInfo] = field(default_factory=list)
    knowledge_maps: list[KnowledgeMap] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
