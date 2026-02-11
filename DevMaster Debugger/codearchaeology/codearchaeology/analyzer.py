"""
Git repository analyzer for code archaeology.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import git

from .models import (
    AbandonedCode,
    CouplingInfo,
    FileHistory,
    Hotspot,
    KnowledgeMap,
    Pattern,
    RiskLevel,
)


class CodeArchaeologist:
    """Main analyzer for code archaeology."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        try:
            self.repo = git.Repo(self.repo_path, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"Not a git repository: {repo_path}")

    def analyze_file_history(
        self,
        file_path: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> dict[str, FileHistory]:
        """
        Analyze history of files.

        Args:
            file_path: Specific file to analyze (or None for all)
            since: Only analyze commits since this date

        Returns:
            Dict mapping file paths to their history
        """
        histories = {}

        # Get commits
        kwargs = {}
        if since:
            kwargs['since'] = since

        try:
            if file_path:
                commits = list(self.repo.iter_commits(paths=file_path, **kwargs))
            else:
                commits = list(self.repo.iter_commits(**kwargs))
        except git.GitCommandError:
            return histories

        # Build history for each file
        file_data = defaultdict(lambda: {
            'changes': 0,
            'authors': set(),
            'first': None,
            'last': None,
            'added': 0,
            'deleted': 0
        })

        for commit in commits:
            commit_date = datetime.fromtimestamp(commit.committed_date)

            try:
                stats = commit.stats.files

                for changed_file, changes in stats.items():
                    data = file_data[changed_file]
                    data['changes'] += 1
                    data['authors'].add(commit.author.email)

                    if data['first'] is None or commit_date < data['first']:
                        data['first'] = commit_date
                    if data['last'] is None or commit_date > data['last']:
                        data['last'] = commit_date

                    data['added'] += changes.get('insertions', 0)
                    data['deleted'] += changes.get('deletions', 0)

            except Exception:
                continue

        # Convert to FileHistory objects
        for file_path, data in file_data.items():
            histories[file_path] = FileHistory(
                file_path=file_path,
                total_changes=data['changes'],
                authors=data['authors'],
                first_commit=data['first'],
                last_commit=data['last'],
                lines_added=data['added'],
                lines_deleted=data['deleted']
            )

        return histories

    def find_hotspots(
        self,
        threshold: int = 10,
        since: Optional[datetime] = None
    ) -> list[Hotspot]:
        """
        Find code hotspots (frequently changed files).

        Args:
            threshold: Minimum number of changes to be considered a hotspot
            since: Only consider commits since this date

        Returns:
            List of hotspots sorted by change frequency
        """
        histories = self.analyze_file_history(since=since)

        hotspots = []
        for file_path, history in histories.items():
            if history.total_changes >= threshold:
                hotspot = Hotspot(
                    file_path=file_path,
                    change_count=history.total_changes,
                    risk_level=history.risk_level,
                    churn_rate=history.churn_rate,
                    last_changed=history.last_commit,
                    primary_authors=list(history.authors)[:3]  # Top 3
                )
                hotspots.append(hotspot)

        # Sort by change count
        hotspots.sort(key=lambda h: h.change_count, reverse=True)
        return hotspots

    def find_abandoned_code(
        self,
        age_threshold_days: int = 180,
        min_initial_commits: int = 3
    ) -> list[AbandonedCode]:
        """
        Find potentially abandoned code.

        Args:
            age_threshold_days: Minimum days since last change
            min_initial_commits: Minimum commits when it was active

        Returns:
            List of potentially abandoned code
        """
        histories = self.analyze_file_history()
        now = datetime.now()
        abandoned = []

        for file_path, history in histories.items():
            if not history.last_commit:
                continue

            days_since_change = (now - history.last_commit).days

            # Check if it appears abandoned
            if (days_since_change >= age_threshold_days and
                history.total_changes >= min_initial_commits):

                abandoned_code = AbandonedCode(
                    file_path=file_path,
                    last_modified=history.last_commit,
                    creation_date=history.first_commit,
                    initial_commits=history.total_changes,
                    days_abandoned=days_since_change,
                    referenced_by=[]  # TODO: detect references
                )
                abandoned.append(abandoned_code)

        # Sort by days abandoned
        abandoned.sort(key=lambda a: a.days_abandoned, reverse=True)
        return abandoned

    def analyze_coupling(
        self,
        min_changes: int = 5,
        coupling_threshold: float = 0.5
    ) -> list[CouplingInfo]:
        """
        Find temporal coupling (files that change together).

        Args:
            min_changes: Minimum changes to consider
            coupling_threshold: Minimum coupling score (0-1)

        Returns:
            List of coupled file pairs
        """
        # Track which files change together
        file_changes = defaultdict(set)  # file -> set of commit hashes

        try:
            for commit in self.repo.iter_commits():
                changed_files = list(commit.stats.files.keys())

                for file in changed_files:
                    file_changes[file].add(commit.hexsha)

        except Exception:
            pass

        # Calculate coupling between file pairs
        coupling_info = []
        files = list(file_changes.keys())

        for i, file1 in enumerate(files):
            for file2 in files[i + 1:]:
                # Count how often they change together
                commits1 = file_changes[file1]
                commits2 = file_changes[file2]

                common = commits1 & commits2
                if len(common) < min_changes:
                    continue

                # Calculate coupling score (Jaccard similarity)
                union = commits1 | commits2
                coupling_score = len(common) / len(union) if union else 0

                if coupling_score >= coupling_threshold:
                    coupling_info.append(CouplingInfo(
                        file1=file1,
                        file2=file2,
                        coupling_score=coupling_score,
                        change_count=len(common)
                    ))

        # Sort by coupling score
        coupling_info.sort(key=lambda c: c.coupling_score, reverse=True)
        return coupling_info

    def analyze_knowledge_distribution(self) -> list[KnowledgeMap]:
        """
        Analyze who knows what code (knowledge distribution).

        Returns:
            List of knowledge maps for files
        """
        histories = self.analyze_file_history()
        knowledge_maps = []

        # Get detailed author info for each file
        for file_path in histories.keys():
            try:
                # Count commits per author for this file
                author_commits = defaultdict(int)

                for commit in self.repo.iter_commits(paths=file_path):
                    author_commits[commit.author.email] += 1

                knowledge_map = KnowledgeMap(
                    file_path=file_path,
                    authors=dict(author_commits)
                )
                knowledge_maps.append(knowledge_map)

            except Exception:
                continue

        # Sort by knowledge concentration (highest first)
        knowledge_maps.sort(key=lambda k: k.knowledge_concentration, reverse=True)
        return knowledge_maps

    def detect_patterns(self) -> list[Pattern]:
        """
        Detect patterns in code evolution.

        Returns:
            List of detected patterns
        """
        patterns = []
        histories = self.analyze_file_history()

        # Pattern 1: Bug Fix Clusters
        bug_files = []
        for file_path, history in histories.items():
            # Heuristic: files with many small commits might indicate bug fixes
            if history.total_changes > 20 and history.churn_rate > 0.5:
                bug_files.append(file_path)

        if bug_files:
            patterns.append(Pattern(
                pattern_type="bug_cluster",
                description=f"Found {len(bug_files)} files with frequent small changes (possible bug clusters)",
                files=bug_files[:5],  # Top 5
                confidence=0.7,
                evidence=[f"{len(bug_files)} files with high churn rate"]
            ))

        # Pattern 2: Refactoring Cycles
        stable_then_active = []
        for file_path, history in histories.items():
            # Heuristic: was stable, then suddenly active
            if history.days_since_change < 30 and history.age_days > 180:
                stable_then_active.append(file_path)

        if stable_then_active:
            patterns.append(Pattern(
                pattern_type="refactor_cycle",
                description="Files recently modified after long stability (possible refactoring)",
                files=stable_then_active[:5],
                confidence=0.6,
                evidence=[f"{len(stable_then_active)} files modified after stability"]
            ))

        # Pattern 3: Knowledge Silos
        knowledge_maps = self.analyze_knowledge_distribution()
        silos = [k for k in knowledge_maps if k.bus_factor == 1]

        if len(silos) > 5:
            patterns.append(Pattern(
                pattern_type="knowledge_silo",
                description=f"{len(silos)} files have single-person knowledge",
                files=[s.file_path for s in silos[:5]],
                confidence=0.8,
                evidence=[f"Bus factor of 1 for {len(silos)} files"]
            ))

        return patterns

    def get_file_age_distribution(self) -> dict[str, int]:
        """Get distribution of file ages."""
        histories = self.analyze_file_history()
        distribution = {
            "0-30 days": 0,
            "30-90 days": 0,
            "90-180 days": 0,
            "180-365 days": 0,
            "1+ years": 0
        }

        for history in histories.values():
            days = history.days_since_change

            if days <= 30:
                distribution["0-30 days"] += 1
            elif days <= 90:
                distribution["30-90 days"] += 1
            elif days <= 180:
                distribution["90-180 days"] += 1
            elif days <= 365:
                distribution["180-365 days"] += 1
            else:
                distribution["1+ years"] += 1

        return distribution

    def get_summary_stats(self) -> dict:
        """Get summary statistics about the repository."""
        histories = self.analyze_file_history()

        total_files = len(histories)
        total_changes = sum(h.total_changes for h in histories.values())
        all_authors = set()
        for h in histories.values():
            all_authors.update(h.authors)

        hotspots = self.find_hotspots(threshold=10)
        high_risk = len([h for h in hotspots if h.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)])

        return {
            "total_files": total_files,
            "total_changes": total_changes,
            "total_authors": len(all_authors),
            "hotspot_count": len(hotspots),
            "high_risk_files": high_risk,
            "age_distribution": self.get_file_age_distribution()
        }
