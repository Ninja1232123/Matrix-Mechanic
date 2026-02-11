"""
Data models for error tracking and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    UNKNOWN = "unknown"


@dataclass
class StackFrame:
    """Represents a single frame in a stack trace."""
    file_path: str
    line_number: int
    function_name: Optional[str] = None
    code_context: Optional[str] = None

    def __str__(self) -> str:
        if self.function_name:
            return f"{self.file_path}:{self.line_number} in {self.function_name}"
        return f"{self.file_path}:{self.line_number}"


@dataclass
class ParsedError:
    """Structured representation of an error."""
    error_type: str
    message: str
    severity: ErrorSeverity
    language: Language
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: list[StackFrame] = field(default_factory=list)
    source_file: Optional[str] = None
    line_number: Optional[int] = None
    full_output: str = ""

    @property
    def primary_location(self) -> Optional[StackFrame]:
        """Get the primary error location (usually first frame)."""
        return self.stack_trace[0] if self.stack_trace else None

    def __str__(self) -> str:
        location = f" at {self.primary_location}" if self.primary_location else ""
        return f"[{self.severity.value}] {self.error_type}: {self.message}{location}"


@dataclass
class HistoricalFix:
    """Represents a fix from git history."""
    commit_hash: str
    commit_message: str
    author: str
    date: datetime
    files_changed: list[str]
    error_pattern: str
    similarity_score: float
    diff: str = ""

    def __str__(self) -> str:
        return f"{self.commit_hash[:8]}: {self.commit_message} (similarity: {self.similarity_score:.2f})"


@dataclass
class Suggestion:
    """A suggested fix for an error."""
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    category: str  # e.g., "pattern", "history", "common_fix"
    action: Optional[str] = None  # e.g., "apply_patch", "open_file"
    action_data: dict = field(default_factory=dict)
    related_fixes: list[HistoricalFix] = field(default_factory=list)

    def __str__(self) -> str:
        confidence_pct = int(self.confidence * 100)
        return f"{self.title} ({confidence_pct}% confidence)"
