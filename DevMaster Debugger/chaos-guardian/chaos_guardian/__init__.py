"""
Chaos-Guardian: Chaos Engineering for Python

Break things intentionally to test resilience!
"""

__version__ = "1.0.0"
__author__ = "Codes-Masterpiece"

from .core import ChaosGuardian, chaos
from .models import ChaosExperiment, ChaosResult

__all__ = ["ChaosGuardian", "chaos", "ChaosExperiment", "ChaosResult"]
