"""Data models for Speed-Guardian"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime


class OptimizationType(Enum):
    """Types of optimizations"""
    CACHING = "caching"
    LOOP_COMPREHENSION = "loop_comprehension"
    ALGORITHM = "algorithm"
    ASYNC_AWAIT = "async_await"
    DATABASE_QUERY = "database_query"
    IO_BUFFERING = "io_buffering"
    MEMORY_REUSE = "memory_reuse"
    LAZY_EVALUATION = "lazy_evaluation"
    VECTORIZATION = "vectorization"
    PARALLEL_PROCESSING = "parallel_processing"


class Severity(Enum):
    """Severity levels for bottlenecks"""
    CRITICAL = "critical"  # >1s or >50% of total time
    HIGH = "high"  # >100ms or >20% of total time
    MEDIUM = "medium"  # >10ms or >5% of total time
    LOW = "low"  # <10ms or <5% of total time


@dataclass
class Bottleneck:
    """Represents a performance bottleneck"""
    function_name: str
    file_path: str
    line_number: int
    time_ms: float
    calls: int
    time_per_call_ms: float
    percentage: float
    severity: Severity
    description: str
    context: Optional[str] = None

    def __str__(self) -> str:
        return (
            f"{self.severity.value.upper()}: {self.function_name} "
            f"({self.file_path}:{self.line_number}) - "
            f"{self.time_ms:.2f}ms ({self.percentage:.1f}%)"
        )


@dataclass
class Optimization:
    """Represents a suggested or applied optimization"""
    type: OptimizationType
    file_path: str
    line_number: int
    original_code: str
    optimized_code: str
    description: str
    estimated_speedup: float  # Factor (2.0 = 2x faster)
    confidence: float  # 0.0-1.0
    applied: bool = False
    actual_speedup: Optional[float] = None

    def __str__(self) -> str:
        status = "✓ Applied" if self.applied else "○ Suggested"
        return (
            f"{status} [{self.type.value}] {self.file_path}:{self.line_number}\n"
            f"  {self.description}\n"
            f"  Estimated speedup: {self.estimated_speedup:.1f}x"
        )


@dataclass
class ProfileResult:
    """Results from profiling a function or module"""
    target: str
    total_time_ms: float
    total_calls: int
    timestamp: datetime
    function_stats: Dict[str, Any]
    line_stats: Optional[Dict[int, Any]] = None
    memory_stats: Optional[Dict[str, Any]] = None

    def get_top_functions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N slowest functions"""
        sorted_funcs = sorted(
            self.function_stats.items(),
            key=lambda x: x[1].get('cumtime', 0),
            reverse=True
        )
        return [
            {
                'name': name,
                'time_ms': stats.get('cumtime', 0) * 1000,
                'calls': stats.get('ncalls', 0),
                'time_per_call_ms': (stats.get('cumtime', 0) / stats.get('ncalls', 1)) * 1000,
            }
            for name, stats in sorted_funcs[:n]
        ]


@dataclass
class PerformanceReport:
    """Complete performance analysis report"""
    target: str
    profile_result: ProfileResult
    bottlenecks: List[Bottleneck]
    optimizations: List[Optimization]
    summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def get_critical_issues(self) -> List[Bottleneck]:
        """Get only critical and high severity bottlenecks"""
        return [
            b for b in self.bottlenecks
            if b.severity in [Severity.CRITICAL, Severity.HIGH]
        ]

    def get_applied_optimizations(self) -> List[Optimization]:
        """Get optimizations that have been applied"""
        return [o for o in self.optimizations if o.applied]

    def get_total_speedup(self) -> float:
        """Calculate total estimated speedup from all optimizations"""
        applied = self.get_applied_optimizations()
        if not applied:
            return 1.0
        # Multiply speedups (conservative estimate)
        total = 1.0
        for opt in applied:
            if opt.actual_speedup:
                total *= opt.actual_speedup
            else:
                total *= opt.estimated_speedup
        return total


@dataclass
class PerformanceRegression:
    """Detected performance regression"""
    function_name: str
    file_path: str
    commit_hash: str
    commit_message: str
    old_time_ms: float
    new_time_ms: float
    slowdown_factor: float
    timestamp: datetime

    def __str__(self) -> str:
        return (
            f"REGRESSION: {self.function_name} slowed down by {self.slowdown_factor:.1f}x\n"
            f"  {self.file_path}\n"
            f"  {self.old_time_ms:.2f}ms → {self.new_time_ms:.2f}ms\n"
            f"  Commit: {self.commit_hash[:8]} - {self.commit_message}"
        )


@dataclass
class BenchmarkResult:
    """Benchmark result for a specific function"""
    function_name: str
    file_path: str
    commit_hash: str
    time_ms: float
    memory_mb: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
