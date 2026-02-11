"""
Security-Guardian: Automatic Security Vulnerability Scanner

Scans Python code for security vulnerabilities including SQL injection,
XSS, secrets exposure, and vulnerable dependencies.
"""

__version__ = "1.0.0"
__author__ = "Codes-Masterpiece"

from .core import SecurityGuardian
from .models import Vulnerability, Severity

__all__ = ["SecurityGuardian", "Vulnerability", "Severity"]
