"""
Git history analyzer to find similar errors and their fixes.
"""

import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import git

from .models import HistoricalFix, ParsedError


class GitHistoryAnalyzer:
    """Analyzes git history to find similar errors and fixes."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        try:
            self.repo = git.Repo(self.repo_path, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            self.repo = None

    def find_similar_fixes(
        self,
        error: ParsedError,
        max_results: int = 5,
        max_commits: int = 500
    ) -> list[HistoricalFix]:
        """Find commits that might have fixed similar errors."""
        if not self.repo:
            return []

        fixes = []
        error_signature = self._create_error_signature(error)

        # Analyze recent commits
        commits = list(self.repo.iter_commits(max_count=max_commits))

        for commit in commits:
            # Look for error-related keywords in commit message
            if not self._is_fix_commit(commit.message):
                continue

            # Calculate similarity
            similarity = self._calculate_similarity(
                error_signature,
                commit.message,
                self._get_commit_diff(commit)
            )

            if similarity > 0.3:  # Threshold for relevance
                fix = self._create_historical_fix(commit, error_signature, similarity)
                fixes.append(fix)

        # Sort by similarity score
        fixes.sort(key=lambda x: x.similarity_score, reverse=True)
        return fixes[:max_results]

    def find_fixes_for_file(
        self,
        file_path: str,
        error_type: Optional[str] = None,
        max_results: int = 5
    ) -> list[HistoricalFix]:
        """Find commits that modified a specific file and might be fixes."""
        if not self.repo:
            return []

        fixes = []
        rel_path = self._make_relative_path(file_path)

        # Check if file is within repo - if not, return empty list
        if not self._is_file_in_repo(file_path):
            return []

        try:
            commits = list(self.repo.iter_commits(paths=rel_path, max_count=100))
        except git.GitCommandError:
            # File might not be tracked or other git error
            return []

        for commit in commits:
            if self._is_fix_commit(commit.message):
                # Check if error type matches commit message
                similarity = 0.5  # Base similarity for file-based matches
                if error_type and error_type.lower() in commit.message.lower():
                    similarity = 0.8

                fix = self._create_historical_fix(commit, error_type or "", similarity)
                fixes.append(fix)

        fixes.sort(key=lambda x: x.similarity_score, reverse=True)
        return fixes[:max_results]

    def get_recent_error_patterns(self, days: int = 30) -> dict[str, int]:
        """Get patterns of errors from recent commits."""
        if not self.repo:
            return {}

        error_patterns: dict[str, int] = {}
        cutoff_date = datetime.now().timestamp() - (days * 86400)

        for commit in self.repo.iter_commits():
            if commit.committed_date < cutoff_date:
                break

            if self._is_fix_commit(commit.message):
                # Extract error type from message
                error_types = self._extract_error_types(commit.message)
                for error_type in error_types:
                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

        return error_patterns

    def _create_error_signature(self, error: ParsedError) -> str:
        """Create a searchable signature for an error."""
        parts = [error.error_type, error.message]

        if error.primary_location:
            parts.append(error.primary_location.file_path)

        return " ".join(parts).lower()

    def _is_fix_commit(self, message: str) -> bool:
        """Check if commit message indicates a fix."""
        fix_keywords = ['fix', 'bug', 'error', 'issue', 'resolve', 'patch', 'correct']
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in fix_keywords)

    def _calculate_similarity(
        self,
        error_signature: str,
        commit_message: str,
        diff: str
    ) -> float:
        """Calculate similarity between error and commit."""
        # Combine commit message and diff for comparison
        commit_text = f"{commit_message} {diff}".lower()

        # Use sequence matcher for similarity
        matcher = SequenceMatcher(None, error_signature, commit_text)
        base_similarity = matcher.ratio()

        # Boost if error type appears in commit
        words = error_signature.split()
        if words:
            error_type = words[0]
            if error_type in commit_text:
                base_similarity += 0.2

        return min(base_similarity, 1.0)

    def _get_commit_diff(self, commit: git.Commit) -> str:
        """Get the diff for a commit."""
        try:
            if commit.parents:
                diff = self.repo.git.diff(commit.parents[0].hexsha, commit.hexsha)
            else:
                diff = self.repo.git.show(commit.hexsha)
            return diff
        except Exception:
            return ""

    def _create_historical_fix(
        self,
        commit: git.Commit,
        error_pattern: str,
        similarity: float
    ) -> HistoricalFix:
        """Create a HistoricalFix from a commit."""
        files_changed = list(commit.stats.files.keys())
        diff = self._get_commit_diff(commit)

        return HistoricalFix(
            commit_hash=commit.hexsha,
            commit_message=commit.message.strip(),
            author=str(commit.author),
            date=datetime.fromtimestamp(commit.committed_date),
            files_changed=files_changed,
            error_pattern=error_pattern,
            similarity_score=similarity,
            diff=diff
        )

    def _extract_error_types(self, message: str) -> list[str]:
        """Extract error types mentioned in commit message."""
        # Look for common error patterns
        error_pattern = re.compile(r'\b(\w+(?:Error|Exception))\b', re.IGNORECASE)
        return error_pattern.findall(message)

    def _make_relative_path(self, file_path: str) -> str:
        """Make file path relative to repo root."""
        path = Path(file_path).resolve()
        try:
            return str(path.relative_to(self.repo.working_dir))
        except ValueError:
            return str(path)

    def _is_file_in_repo(self, file_path: str) -> bool:
        """Check if file is within the repository."""
        if not self.repo:
            return False

        path = Path(file_path).resolve()
        repo_path = Path(self.repo.working_dir).resolve()

        try:
            path.relative_to(repo_path)
            return True
        except ValueError:
            return False
