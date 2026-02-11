"""
Performance-Surgeon - Automatic Performance Optimizer

Detects and fixes performance bottlenecks in Python code.
"""

__version__ = "1.0.0"

from .core import PerformanceSurgeon, QuickProfiler, PerformanceFinding
from .patterns.performance_patterns import PerformancePattern, Severity, Category, ALL_PATTERNS
from .optimizers.auto_optimizer import AutoOptimizer, OptimizationResult

__all__ = [
    'PerformanceSurgeon',
    'QuickProfiler',
    'PerformanceFinding',
    'PerformancePattern',
    'Severity',
    'Category',
    'ALL_PATTERNS',
    'AutoOptimizer',
    'OptimizationResult',
]
