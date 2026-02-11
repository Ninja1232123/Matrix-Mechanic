"""
Speed-Guardian: Automatic Performance Profiler and Optimizer

Speed-Guardian analyzes your Python code, identifies performance bottlenecks,
and automatically applies optimizations to make your code faster.
"""

__version__ = "1.0.0"
__author__ = "Codes-Masterpiece"

from .core import SpeedGuardian
from .models import (
    ProfileResult,
    Bottleneck,
    Optimization,
    PerformanceReport,
)

__all__ = [
    "SpeedGuardian",
    "ProfileResult",
    "Bottleneck",
    "Optimization",
    "PerformanceReport",
]
