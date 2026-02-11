"""Regression detector - tracks performance over git commits"""

import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

try:
    import git
    HAS_GIT = True
except ImportError:
    HAS_GIT = False

from ..models import BenchmarkResult, PerformanceRegression, ProfileResult
from ..config import RegressionConfig


class RegressionDetector:
    """Detects performance regressions across git commits"""

    def __init__(self, config: Optional[RegressionConfig] = None, repo_path: Optional[Path] = None):
        self.config = config or RegressionConfig()
        self.repo_path = repo_path or Path.cwd()
        self.db_path = Path(self.config.benchmark_db_path)

        if not HAS_GIT:
            raise ImportError("GitPython not installed. Install with: pip install gitpython")

        try:
            self.repo = git.Repo(self.repo_path, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            raise ValueError(f"Not a git repository: {self.repo_path}")

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing benchmarks"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                time_ms REAL NOT NULL,
                memory_mb REAL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(function_name, file_path, commit_hash)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_commit_hash
            ON benchmarks(commit_hash)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_function_file
            ON benchmarks(function_name, file_path)
        ''')

        conn.commit()
        conn.close()

    def store_benchmark(self, benchmark: BenchmarkResult):
        """Store a benchmark result in the database"""
        if not self.config.store_benchmarks:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO benchmarks
                (function_name, file_path, commit_hash, time_ms, memory_mb, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                benchmark.function_name,
                benchmark.file_path,
                benchmark.commit_hash,
                benchmark.time_ms,
                benchmark.memory_mb,
                benchmark.timestamp.isoformat(),
                json.dumps(benchmark.metadata)
            ))

            conn.commit()
        finally:
            conn.close()

    def store_profile(self, profile: ProfileResult, file_path: str):
        """Store profile results as benchmarks"""
        commit_hash = self._get_current_commit()
        timestamp = datetime.now()

        for func_key, stats in profile.function_stats.items():
            benchmark = BenchmarkResult(
                function_name=stats['function'],
                file_path=file_path,
                commit_hash=commit_hash,
                time_ms=stats['cumtime'] * 1000,
                memory_mb=0.0,  # Not available from cProfile
                timestamp=timestamp,
                metadata={
                    'ncalls': stats['ncalls'],
                    'tottime': stats['tottime'],
                }
            )
            self.store_benchmark(benchmark)

    def detect_regressions(
        self,
        current_profile: ProfileResult,
        file_path: str
    ) -> List[PerformanceRegression]:
        """Detect performance regressions by comparing with previous commits"""
        if not self.config.enable:
            return []

        regressions = []
        current_commit = self._get_current_commit()

        # Get previous commits to compare
        previous_commits = self._get_previous_commits(self.config.commits_to_check)

        for func_key, current_stats in current_profile.function_stats.items():
            func_name = current_stats['function']
            current_time_ms = current_stats['cumtime'] * 1000

            # Find historical data for this function
            historical = self._get_historical_benchmarks(
                func_name,
                file_path,
                previous_commits
            )

            if not historical:
                continue

            # Compare with most recent benchmark
            latest = historical[0]
            old_time_ms = latest['time_ms']

            if old_time_ms > 0:
                slowdown_factor = current_time_ms / old_time_ms

                if slowdown_factor >= self.config.slowdown_threshold:
                    # Regression detected
                    commit_msg = self._get_commit_message(latest['commit_hash'])

                    regression = PerformanceRegression(
                        function_name=func_name,
                        file_path=file_path,
                        commit_hash=current_commit,
                        commit_message=self._get_commit_message(current_commit),
                        old_time_ms=old_time_ms,
                        new_time_ms=current_time_ms,
                        slowdown_factor=slowdown_factor,
                        timestamp=datetime.now()
                    )
                    regressions.append(regression)

        return regressions

    def get_performance_history(
        self,
        function_name: str,
        file_path: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get performance history for a function"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT commit_hash, time_ms, memory_mb, timestamp, metadata
            FROM benchmarks
            WHERE function_name = ? AND file_path = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (function_name, file_path, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'commit_hash': row[0],
                'time_ms': row[1],
                'memory_mb': row[2],
                'timestamp': datetime.fromisoformat(row[3]),
                'metadata': json.loads(row[4]) if row[4] else {},
            })

        conn.close()
        return results

    def _get_current_commit(self) -> str:
        """Get current git commit hash"""
        return self.repo.head.commit.hexsha

    def _get_previous_commits(self, count: int) -> List[str]:
        """Get list of previous commit hashes"""
        commits = []
        for commit in self.repo.iter_commits(max_count=count + 1):
            commits.append(commit.hexsha)
        # Exclude current commit
        return commits[1:] if len(commits) > 1 else []

    def _get_commit_message(self, commit_hash: str) -> str:
        """Get commit message for a commit hash"""
        try:
            commit = self.repo.commit(commit_hash)
            return commit.message.strip()
        except:
            return "Unknown"

    def _get_historical_benchmarks(
        self,
        function_name: str,
        file_path: str,
        commit_hashes: List[str]
    ) -> List[Dict[str, Any]]:
        """Get historical benchmarks for a function from specific commits"""
        if not commit_hashes:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        placeholders = ','.join(['?'] * len(commit_hashes))
        query = f'''
            SELECT commit_hash, time_ms, memory_mb, timestamp
            FROM benchmarks
            WHERE function_name = ? AND file_path = ? AND commit_hash IN ({placeholders})
            ORDER BY timestamp DESC
        '''

        cursor.execute(query, [function_name, file_path] + commit_hashes)

        results = []
        for row in cursor.fetchall():
            results.append({
                'commit_hash': row[0],
                'time_ms': row[1],
                'memory_mb': row[2],
                'timestamp': datetime.fromisoformat(row[3]),
            })

        conn.close()
        return results

    def compare_commits(
        self,
        commit1: str,
        commit2: str,
        file_path: str
    ) -> Dict[str, Any]:
        """Compare performance between two commits"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get benchmarks for commit1
        cursor.execute('''
            SELECT function_name, time_ms
            FROM benchmarks
            WHERE commit_hash = ? AND file_path = ?
        ''', (commit1, file_path))

        commit1_data = {row[0]: row[1] for row in cursor.fetchall()}

        # Get benchmarks for commit2
        cursor.execute('''
            SELECT function_name, time_ms
            FROM benchmarks
            WHERE commit_hash = ? AND file_path = ?
        ''', (commit2, file_path))

        commit2_data = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        # Compare
        comparison = {}
        for func_name in commit1_data.keys() & commit2_data.keys():
            time1 = commit1_data[func_name]
            time2 = commit2_data[func_name]

            if time1 > 0:
                change_factor = time2 / time1
                comparison[func_name] = {
                    'time1_ms': time1,
                    'time2_ms': time2,
                    'change_factor': change_factor,
                    'slower': change_factor > 1.0,
                }

        return comparison

    def get_slowest_functions(
        self,
        commit_hash: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get slowest functions for a commit"""
        commit = commit_hash or self._get_current_commit()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT function_name, file_path, time_ms, memory_mb
            FROM benchmarks
            WHERE commit_hash = ?
            ORDER BY time_ms DESC
            LIMIT ?
        ''', (commit, limit))

        results = []
        for row in cursor.fetchall():
            results.append({
                'function': row[0],
                'file': row[1],
                'time_ms': row[2],
                'memory_mb': row[3],
            })

        conn.close()
        return results

    def clear_old_benchmarks(self, days: int = 30):
        """Clear benchmarks older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM benchmarks
            WHERE timestamp < ?
        ''', (datetime.fromtimestamp(cutoff_date).isoformat(),))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted
