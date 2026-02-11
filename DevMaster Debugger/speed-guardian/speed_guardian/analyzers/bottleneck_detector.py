"""Bottleneck detector - identifies performance bottlenecks"""

from typing import List, Dict, Any
from pathlib import Path

from ..models import Bottleneck, Severity, ProfileResult


class BottleneckDetector:
    """Detects performance bottlenecks in profiled code"""

    def __init__(
        self,
        critical_threshold_ms: float = 1000.0,
        high_threshold_ms: float = 100.0,
        medium_threshold_ms: float = 10.0,
        critical_percentage: float = 50.0,
        high_percentage: float = 20.0,
        medium_percentage: float = 5.0,
    ):
        self.critical_threshold_ms = critical_threshold_ms
        self.high_threshold_ms = high_threshold_ms
        self.medium_threshold_ms = medium_threshold_ms
        self.critical_percentage = critical_percentage
        self.high_percentage = high_percentage
        self.medium_percentage = medium_percentage

    def detect_bottlenecks(self, profile: ProfileResult) -> List[Bottleneck]:
        """Detect bottlenecks from profile results"""
        bottlenecks = []

        total_time_ms = profile.total_time_ms

        for func_key, stats in profile.function_stats.items():
            cumtime_ms = stats['cumtime'] * 1000
            ncalls = stats['ncalls']

            if cumtime_ms < 1.0:  # Skip functions < 1ms
                continue

            percentage = (cumtime_ms / total_time_ms * 100) if total_time_ms > 0 else 0
            time_per_call_ms = cumtime_ms / ncalls if ncalls > 0 else 0

            # Determine severity
            severity = self._determine_severity(cumtime_ms, percentage)

            if severity:
                bottleneck = Bottleneck(
                    function_name=stats['function'],
                    file_path=stats['filename'],
                    line_number=stats['line'],
                    time_ms=cumtime_ms,
                    calls=ncalls,
                    time_per_call_ms=time_per_call_ms,
                    percentage=percentage,
                    severity=severity,
                    description=self._generate_description(
                        stats['function'],
                        cumtime_ms,
                        percentage,
                        ncalls
                    ),
                )
                bottlenecks.append(bottleneck)

        # Sort by time (descending)
        bottlenecks.sort(key=lambda b: b.time_ms, reverse=True)

        return bottlenecks

    def _determine_severity(self, time_ms: float, percentage: float) -> Severity:
        """Determine severity of a bottleneck"""
        # Critical: >1s OR >50% of total time
        if time_ms >= self.critical_threshold_ms or percentage >= self.critical_percentage:
            return Severity.CRITICAL

        # High: >100ms OR >20% of total time
        if time_ms >= self.high_threshold_ms or percentage >= self.high_percentage:
            return Severity.HIGH

        # Medium: >10ms OR >5% of total time
        if time_ms >= self.medium_threshold_ms or percentage >= self.medium_percentage:
            return Severity.MEDIUM

        # Low: everything else
        return Severity.LOW

    def _generate_description(
        self,
        function_name: str,
        time_ms: float,
        percentage: float,
        calls: int
    ) -> str:
        """Generate a human-readable description of the bottleneck"""
        desc_parts = []

        if time_ms >= 1000:
            desc_parts.append(f"takes {time_ms/1000:.2f}s")
        else:
            desc_parts.append(f"takes {time_ms:.2f}ms")

        desc_parts.append(f"({percentage:.1f}% of total time)")

        if calls > 1:
            desc_parts.append(f"called {calls:,} times")

        return f"{function_name} " + ", ".join(desc_parts)

    def find_frequent_callers(
        self,
        profile: ProfileResult,
        min_calls: int = 100
    ) -> List[Dict[str, Any]]:
        """Find functions that are called very frequently"""
        frequent = []

        for func_key, stats in profile.function_stats.items():
            ncalls = stats['ncalls']
            if ncalls >= min_calls:
                frequent.append({
                    'function': stats['function'],
                    'file': stats['filename'],
                    'line': stats['line'],
                    'calls': ncalls,
                    'total_time_ms': stats['cumtime'] * 1000,
                    'time_per_call_ms': (stats['cumtime'] / ncalls) * 1000 if ncalls > 0 else 0,
                })

        # Sort by number of calls (descending)
        frequent.sort(key=lambda f: f['calls'], reverse=True)

        return frequent

    def find_slow_functions(
        self,
        profile: ProfileResult,
        min_time_per_call_ms: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Find functions with slow individual calls"""
        slow = []

        for func_key, stats in profile.function_stats.items():
            ncalls = stats['ncalls']
            cumtime = stats['cumtime']

            if ncalls == 0:
                continue

            time_per_call_ms = (cumtime / ncalls) * 1000

            if time_per_call_ms >= min_time_per_call_ms:
                slow.append({
                    'function': stats['function'],
                    'file': stats['filename'],
                    'line': stats['line'],
                    'time_per_call_ms': time_per_call_ms,
                    'calls': ncalls,
                    'total_time_ms': cumtime * 1000,
                })

        # Sort by time per call (descending)
        slow.sort(key=lambda s: s['time_per_call_ms'], reverse=True)

        return slow

    def find_io_bottlenecks(self, profile: ProfileResult) -> List[Dict[str, Any]]:
        """Find I/O related bottlenecks"""
        io_patterns = [
            'read', 'write', 'open', 'close',
            'socket', 'connect', 'send', 'recv',
            'select', 'poll', 'epoll',
            'query', 'execute', 'fetch',
            'get', 'post', 'request',
        ]

        io_bottlenecks = []

        for func_key, stats in profile.function_stats.items():
            func_name_lower = stats['function'].lower()

            # Check if function name contains I/O patterns
            is_io = any(pattern in func_name_lower for pattern in io_patterns)

            if is_io:
                cumtime_ms = stats['cumtime'] * 1000
                if cumtime_ms > 1.0:  # Only significant ones
                    io_bottlenecks.append({
                        'function': stats['function'],
                        'file': stats['filename'],
                        'line': stats['line'],
                        'time_ms': cumtime_ms,
                        'calls': stats['ncalls'],
                        'pattern': self._identify_io_pattern(func_name_lower),
                    })

        # Sort by time (descending)
        io_bottlenecks.sort(key=lambda b: b['time_ms'], reverse=True)

        return io_bottlenecks

    def _identify_io_pattern(self, func_name: str) -> str:
        """Identify the type of I/O operation"""
        if any(p in func_name for p in ['read', 'write', 'open', 'close']):
            return 'file_io'
        elif any(p in func_name for p in ['socket', 'connect', 'send', 'recv']):
            return 'network_io'
        elif any(p in func_name for p in ['query', 'execute', 'fetch']):
            return 'database_io'
        elif any(p in func_name for p in ['get', 'post', 'request']):
            return 'http_io'
        else:
            return 'io'

    def analyze_call_patterns(
        self,
        profile: ProfileResult
    ) -> Dict[str, Any]:
        """Analyze overall call patterns"""
        total_calls = profile.total_calls
        total_time_ms = profile.total_time_ms

        # Calculate average time per call
        avg_time_per_call_ms = total_time_ms / total_calls if total_calls > 0 else 0

        # Count functions by call frequency
        call_distribution = {
            'single_call': 0,      # Called once
            'few_calls': 0,        # 2-10 calls
            'moderate_calls': 0,   # 11-100 calls
            'many_calls': 0,       # 101-1000 calls
            'very_many_calls': 0,  # >1000 calls
        }

        for stats in profile.function_stats.values():
            ncalls = stats['ncalls']
            if ncalls == 1:
                call_distribution['single_call'] += 1
            elif ncalls <= 10:
                call_distribution['few_calls'] += 1
            elif ncalls <= 100:
                call_distribution['moderate_calls'] += 1
            elif ncalls <= 1000:
                call_distribution['many_calls'] += 1
            else:
                call_distribution['very_many_calls'] += 1

        return {
            'total_calls': total_calls,
            'total_time_ms': total_time_ms,
            'avg_time_per_call_ms': avg_time_per_call_ms,
            'call_distribution': call_distribution,
            'unique_functions': len(profile.function_stats),
        }

    def get_summary(self, bottlenecks: List[Bottleneck]) -> Dict[str, Any]:
        """Get summary of detected bottlenecks"""
        if not bottlenecks:
            return {
                'total': 0,
                'by_severity': {},
                'top_bottleneck': None,
            }

        by_severity = {
            'critical': len([b for b in bottlenecks if b.severity == Severity.CRITICAL]),
            'high': len([b for b in bottlenecks if b.severity == Severity.HIGH]),
            'medium': len([b for b in bottlenecks if b.severity == Severity.MEDIUM]),
            'low': len([b for b in bottlenecks if b.severity == Severity.LOW]),
        }

        return {
            'total': len(bottlenecks),
            'by_severity': by_severity,
            'top_bottleneck': bottlenecks[0] if bottlenecks else None,
            'total_time_in_bottlenecks_ms': sum(b.time_ms for b in bottlenecks),
        }
