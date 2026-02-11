"""
Test-Guardian: Automatic Test Generation for Python

Test-Guardian analyzes your Python code and automatically generates comprehensive
test suites including unit tests, fixtures, and mocks.
"""

__version__ = "1.0.0"
__author__ = "Codes-Masterpiece"

from .core import TestGuardian
from .models import TestCase, TestSuite, Fixture, MockSpec

__all__ = [
    "TestGuardian",
    "TestCase",
    "TestSuite",
    "Fixture",
    "MockSpec",
]
