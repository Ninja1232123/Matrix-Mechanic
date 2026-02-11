"""Data models for Security-Guardian"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class Severity(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"  # Immediate fix required
    HIGH = "high"          # Fix ASAP
    MEDIUM = "medium"      # Fix soon
    LOW = "low"            # Fix when possible
    INFO = "info"          # Informational


class VulnerabilityType(Enum):
    """Types of vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    SECRETS_EXPOSURE = "secrets_exposure"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    WEAK_CRYPTO = "weak_crypto"
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    INSECURE_DEPENDENCIES = "insecure_dependencies"
    UNSAFE_EVAL = "unsafe_eval"


@dataclass
class Vulnerability:
    """Represents a security vulnerability"""
    type: VulnerabilityType
    severity: Severity
    file_path: str
    line_number: int
    code_snippet: str
    description: str
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    owasp_category: Optional[str] = None
    fix_suggestion: str = ""
    confidence: float = 1.0  # 0.0-1.0

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.type.value} "
            f"at {self.file_path}:{self.line_number}"
        )


@dataclass
class SecretMatch:
    """Detected secret/credential"""
    secret_type: str  # api_key, password, token, etc.
    file_path: str
    line_number: int
    matched_pattern: str
    context: str

@dataclass
class DependencyIssue:
    """Vulnerable or outdated dependency"""
    package_name: str
    current_version: str
    latest_version: Optional[str]
    vulnerabilities: List[str] = field(default_factory=list)
    severity: Severity = Severity.INFO


@dataclass
class SecurityReport:
    """Complete security scan report"""
    target: str
    vulnerabilities: List[Vulnerability]
    secrets: List[SecretMatch]
    dependency_issues: List[DependencyIssue]
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_critical_count(self) -> int:
        return len([v for v in self.vulnerabilities if v.severity == Severity.CRITICAL])

    def get_high_count(self) -> int:
        return len([v for v in self.vulnerabilities if v.severity == Severity.HIGH])

    def get_total_issues(self) -> int:
        return len(self.vulnerabilities) + len(self.secrets) + len(self.dependency_issues)
