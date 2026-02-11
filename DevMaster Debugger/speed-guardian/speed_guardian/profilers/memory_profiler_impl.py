"""Memory profiler implementation"""

import sys
import tracemalloc
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
import psutil
import os


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time"""
    current_mb: float
    peak_mb: float
    allocations: int
    timestamp: float


class MemoryProfilerImpl:
    """Memory profiler for tracking memory usage"""

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.baseline: Optional[MemorySnapshot] = None

    def start_tracking(self):
        """Start tracking memory allocations"""
        tracemalloc.start()
        self.baseline = self._take_snapshot()

    def stop_tracking(self):
        """Stop tracking memory allocations"""
        tracemalloc.stop()

    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile memory usage of a function"""
        # Start tracking
        tracemalloc.start()

        # Get process for system-wide memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Take baseline snapshot
        snapshot_before = tracemalloc.take_snapshot()

        # Run function
        result = func(*args, **kwargs)

        # Take after snapshot
        snapshot_after = tracemalloc.take_snapshot()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Compare snapshots
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        # Parse allocations
        allocations = []
        for stat in top_stats[:20]:  # Top 20 allocations
            allocations.append({
                'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'line': stat.traceback[0].lineno if stat.traceback else 0,
                'size_mb': stat.size / 1024 / 1024,
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count': stat.count,
                'count_diff': stat.count_diff,
            })

        return {
            'function': func.__name__,
            'return_value': result,
            'memory_used_mb': mem_after - mem_before,
            'peak_memory_mb': peak / 1024 / 1024,
            'current_memory_mb': current / 1024 / 1024,
            'allocations': allocations,
            'top_allocation': allocations[0] if allocations else None,
        }

    def profile_script(self, script_path: str) -> Dict[str, Any]:
        """Profile memory usage of a script"""
        tracemalloc.start()

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024

        snapshot_before = tracemalloc.take_snapshot()

        # Execute script
        with open(script_path, 'r') as f:
            code = compile(f.read(), script_path, 'exec')
            globals_dict = {'__name__': '__main__', '__file__': script_path}
            exec(code, globals_dict)

        snapshot_after = tracemalloc.take_snapshot()
        mem_after = process.memory_info().rss / 1024 / 1024

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Compare snapshots
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        allocations = []
        for stat in top_stats[:20]:
            allocations.append({
                'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'line': stat.traceback[0].lineno if stat.traceback else 0,
                'size_mb': stat.size / 1024 / 1024,
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count': stat.count,
                'count_diff': stat.count_diff,
            })

        return {
            'script': script_path,
            'memory_used_mb': mem_after - mem_before,
            'peak_memory_mb': peak / 1024 / 1024,
            'current_memory_mb': current / 1024 / 1024,
            'allocations': allocations,
        }

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        current, peak = tracemalloc.get_traced_memory()
        process = psutil.Process(os.getpid())

        return MemorySnapshot(
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
            allocations=len(tracemalloc.get_traced_memory()),
            timestamp=process.create_time(),
        )

    def get_top_allocations(
        self,
        snapshot_before: Any,
        snapshot_after: Any,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top N memory allocations"""
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        allocations = []
        for stat in top_stats[:n]:
            allocations.append({
                'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'line': stat.traceback[0].lineno if stat.traceback else 0,
                'size_mb': stat.size / 1024 / 1024,
                'size_diff_mb': stat.size_diff / 1024 / 1024,
                'count': stat.count,
                'count_diff': stat.count_diff,
            })

        return allocations

    def find_memory_leaks(
        self,
        threshold_mb: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Find potential memory leaks"""
        if not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        leaks = []
        for stat in top_stats:
            size_mb = stat.size / 1024 / 1024
            if size_mb > threshold_mb:
                leaks.append({
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'line': stat.traceback[0].lineno if stat.traceback else 0,
                    'size_mb': size_mb,
                    'count': stat.count,
                })

        return leaks

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get overall memory summary"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        summary = {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
        }

        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            summary.update({
                'traced_current_mb': current / 1024 / 1024,
                'traced_peak_mb': peak / 1024 / 1024,
            })

        return summary

    def format_memory_stats(
        self,
        allocations: List[Dict[str, Any]]
    ) -> str:
        """Format memory stats for display"""
        if not allocations:
            return "No memory allocations tracked"

        lines = []
        lines.append(f"{'File:Line':<50} {'Size (MB)':<12} {'Diff (MB)':<12} {'Count':<10}")
        lines.append("-" * 90)

        for alloc in allocations:
            file_line = f"{alloc['filename']}:{alloc['line']}"
            lines.append(
                f"{file_line:<50} "
                f"{alloc['size_mb']:<12.2f} "
                f"{alloc['size_diff_mb']:<12.2f} "
                f"{alloc['count']:<10}"
            )

        return "\n".join(lines)

    def compare_memory(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare memory usage between two points"""
        return {
            'memory_increase_mb': after['memory_used_mb'] - before['memory_used_mb'],
            'peak_increase_mb': after['peak_memory_mb'] - before['peak_memory_mb'],
            'allocation_increase': (
                len(after.get('allocations', [])) - len(before.get('allocations', []))
            ),
        }
