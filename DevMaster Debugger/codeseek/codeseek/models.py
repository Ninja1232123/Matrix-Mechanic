"""
Data models for CodeSeek.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    UNKNOWN = "unknown"


class SymbolType(Enum):
    """Types of code symbols."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, etc.)."""
    id: Optional[int] = None
    name: str = ""
    symbol_type: SymbolType = SymbolType.FUNCTION
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    code: str = ""
    docstring: Optional[str] = None
    language: Language = Language.UNKNOWN

    # Metadata
    params: list[str] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)  # async, static, public, etc.
    parent: Optional[str] = None  # Parent class/module

    # Search metadata
    calls: list[str] = field(default_factory=list)  # Functions/methods called
    imports: list[str] = field(default_factory=list)  # Imports used

    created_at: datetime = field(default_factory=datetime.now)

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    @property
    def short_code(self) -> str:
        """Get abbreviated code for display."""
        lines = self.code.split('\n')
        if len(lines) <= 10:
            return self.code
        return '\n'.join(lines[:10]) + '\n...'

    def __str__(self) -> str:
        return f"{self.symbol_type.value} {self.qualified_name} @ {self.file_path}:{self.start_line}"


@dataclass
class FileIndex:
    """Metadata about an indexed file."""
    id: Optional[int] = None
    file_path: str = ""
    language: Language = Language.UNKNOWN
    size_bytes: int = 0
    line_count: int = 0
    symbol_count: int = 0
    indexed_at: datetime = field(default_factory=datetime.now)
    content_hash: str = ""  # For detecting changes

    def __str__(self) -> str:
        return f"{self.file_path} ({self.language.value}, {self.symbol_count} symbols)"


@dataclass
class SearchResult:
    """Result from a code search query."""
    symbol: CodeSymbol
    score: float
    match_type: str = "semantic"  # semantic, exact, structural, pattern
    highlights: list[tuple[int, int]] = field(default_factory=list)  # Line ranges
    context_before: str = ""
    context_after: str = ""

    @property
    def file_path(self) -> str:
        return self.symbol.file_path

    @property
    def line_number(self) -> int:
        return self.symbol.start_line

    def __str__(self) -> str:
        score_pct = int(self.score * 100)
        return f"{self.symbol.qualified_name} ({score_pct}%) - {self.file_path}:{self.line_number}"


@dataclass
class Embedding:
    """Vector embedding for a code symbol."""
    id: Optional[int] = None
    symbol_id: int = 0
    vector: list[float] = field(default_factory=list)
    model_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass
class Query:
    """Parsed search query."""
    raw_query: str
    intent: str  # What the user wants: "find functions", "similar to", etc.
    filters: dict = field(default_factory=dict)  # symbol_type, language, etc.
    semantic_query: str = ""  # Processed for semantic search

    def __str__(self) -> str:
        return f"Query('{self.raw_query}', intent={self.intent})"


@dataclass
class IndexStats:
    """Statistics about the code index."""
    total_files: int = 0
    total_symbols: int = 0
    total_lines: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    symbol_types: dict[str, int] = field(default_factory=dict)
    index_size_mb: float = 0.0
    last_indexed: Optional[datetime] = None

    def __str__(self) -> str:
        return f"Index: {self.total_files} files, {self.total_symbols} symbols"
