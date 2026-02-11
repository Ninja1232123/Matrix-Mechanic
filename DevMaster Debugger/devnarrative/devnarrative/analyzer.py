"""
Git repository analyzer.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import git

from .models import Commit, CommitType


class GitAnalyzer:
    """Analyzes git repository history."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        try:
            self.repo = git.Repo(self.repo_path, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"Not a git repository: {repo_path}")

    def get_commits(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        author: Optional[str] = None,
        branch: Optional[str] = None
    ) -> list[Commit]:
        """
        Get commits from repository.

        Args:
            since: Start date
            until: End date
            author: Filter by author email
            branch: Filter by branch

        Returns:
            List of commits
        """
        # Build kwargs for iter_commits
        kwargs = {}

        if since:
            kwargs['since'] = since
        if until:
            kwargs['until'] = until
        if author:
            kwargs['author'] = author

        # Get branch
        if branch:
            rev = branch
        else:
            rev = 'HEAD'

        commits = []

        try:
            for git_commit in self.repo.iter_commits(rev, **kwargs):
                commit = self._convert_commit(git_commit)
                commits.append(commit)
        except git.GitCommandError as e:
            print(f"Warning: {e}")
            return []

        return commits

    def get_commits_today(self, author: Optional[str] = None) -> list[Commit]:
        """Get commits from today."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_commits(since=today_start, author=author)

    def get_commits_this_week(self, author: Optional[str] = None) -> list[Commit]:
        """Get commits from this week (Monday to now)."""
        now = datetime.now()
        # Get Monday of current week
        monday = now - timedelta(days=now.weekday())
        monday_start = monday.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_commits(since=monday_start, author=author)

    def get_commits_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        author: Optional[str] = None
    ) -> list[Commit]:
        """Get commits in date range."""
        return self.get_commits(since=start_date, until=end_date, author=author)

    def get_feature_commits(
        self,
        feature_name: str,
        author: Optional[str] = None
    ) -> list[Commit]:
        """Get commits related to a feature (by searching commit messages)."""
        all_commits = self.get_commits(author=author)

        feature_commits = []
        for commit in all_commits:
            if feature_name.lower() in commit.message.lower():
                feature_commits.append(commit)

        return feature_commits

    def _convert_commit(self, git_commit: git.Commit) -> Commit:
        """Convert GitPython commit to our Commit model."""
        # Get commit stats
        stats = git_commit.stats.total
        insertions = stats.get('insertions', 0)
        deletions = stats.get('deletions', 0)

        # Get changed files
        files_changed = list(git_commit.stats.files.keys())

        # Classify commit type
        commit_type = self._classify_commit(git_commit.message)

        # Get current branch (approximation)
        try:
            branch = self.repo.active_branch.name
        except:
            branch = None

        return Commit(
            hash=git_commit.hexsha,
            author=git_commit.author.name,
            email=git_commit.author.email,
            date=datetime.fromtimestamp(git_commit.committed_date),
            message=git_commit.message.strip(),
            files_changed=files_changed,
            insertions=insertions,
            deletions=deletions,
            commit_type=commit_type,
            branch=branch
        )

    def _classify_commit(self, message: str) -> CommitType:
        """Classify commit based on message."""
        message_lower = message.lower()

        # Merge commits
        if message.startswith('Merge'):
            return CommitType.MERGE

        # Revert commits
        if 'revert' in message_lower:
            return CommitType.REVERT

        # Feature keywords
        feature_keywords = ['feat', 'feature', 'add', 'implement', 'create', 'new']
        if any(kw in message_lower for kw in feature_keywords):
            return CommitType.FEATURE

        # Bug fix keywords
        bugfix_keywords = ['fix', 'bug', 'issue', 'resolve', 'patch']
        if any(kw in message_lower for kw in bugfix_keywords):
            return CommitType.BUGFIX

        # Refactor keywords
        refactor_keywords = ['refactor', 'cleanup', 'optimize', 'improve', 'restructure']
        if any(kw in message_lower for kw in refactor_keywords):
            return CommitType.REFACTOR

        # Test keywords
        test_keywords = ['test', 'spec', 'testing']
        if any(kw in message_lower for kw in test_keywords):
            return CommitType.TEST

        # Docs keywords
        docs_keywords = ['doc', 'docs', 'documentation', 'readme', 'comment']
        if any(kw in message_lower for kw in docs_keywords):
            return CommitType.DOCS

        # Chore keywords
        chore_keywords = ['chore', 'update', 'bump', 'upgrade']
        if any(kw in message_lower for kw in chore_keywords):
            return CommitType.CHORE

        return CommitType.UNKNOWN

    def get_author_stats(self, since: Optional[datetime] = None) -> dict:
        """Get statistics for repository authors."""
        commits = self.get_commits(since=since)

        stats = {}
        for commit in commits:
            if commit.author not in stats:
                stats[commit.author] = {
                    'commits': 0,
                    'insertions': 0,
                    'deletions': 0,
                    'files': set()
                }

            stats[commit.author]['commits'] += 1
            stats[commit.author]['insertions'] += commit.insertions
            stats[commit.author]['deletions'] += commit.deletions
            stats[commit.author]['files'].update(commit.files_changed)

        # Convert sets to counts
        for author in stats:
            stats[author]['files'] = len(stats[author]['files'])

        return stats

    def detect_challenges(self, commits: list[Commit]) -> list[str]:
        """Detect challenges based on commit patterns."""
        challenges = []

        # Check for reverts
        reverts = [c for c in commits if c.commit_type == CommitType.REVERT]
        if reverts:
            challenges.append(f"Had to revert {len(reverts)} commit(s)")

        # Check for many small commits on same files (debugging)
        file_commit_count = {}
        for commit in commits:
            for file in commit.files_changed:
                file_commit_count[file] = file_commit_count.get(file, 0) + 1

        heavily_modified = [f for f, count in file_commit_count.items() if count >= 5]
        if heavily_modified:
            challenges.append(f"Complex work on {heavily_modified[0]}")

        # Check for merge conflicts (merge commits with many changes)
        complex_merges = [
            c for c in commits
            if c.commit_type == CommitType.MERGE and len(c.files_changed) > 5
        ]
        if complex_merges:
            challenges.append("Resolved merge conflicts")

        return challenges
