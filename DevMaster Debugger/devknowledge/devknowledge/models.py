"""
Data models for DevKnowledge.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DocumentType(Enum):
    """Types of documents in the knowledge base."""
    NOTE = "note"
    CODE = "code"
    DOCUMENTATION = "documentation"
    SNIPPET = "snippet"


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    MARKDOWN = "markdown"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    id: Optional[int] = None
    title: str = ""
    content: str = ""
    doc_type: DocumentType = DocumentType.NOTE
    language: Language = Language.TEXT
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    # Extracted information
    symbols: list[str] = field(default_factory=list)  # Functions, classes, etc.
    imports: list[str] = field(default_factory=list)  # Import statements

    def __str__(self) -> str:
        return f"{self.title} ({self.doc_type.value})"


@dataclass
class Link:
    """Represents a link between two documents."""
    id: Optional[int] = None
    source_id: int = 0
    target_id: int = 0
    link_type: str = "related"  # related, references, implements, etc.
    strength: float = 0.0  # Similarity score 0-1
    created_at: datetime = field(default_factory=datetime.now)
    auto_generated: bool = True

    def __str__(self) -> str:
        return f"Link({self.source_id} -> {self.target_id}, {self.strength:.2f})"


@dataclass
class Embedding:
    """Vector embedding for a document."""
    id: Optional[int] = None
    document_id: int = 0
    vector: list[float] = field(default_factory=list)
    model_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass
class SearchResult:
    """Result from a search query."""
    document: Document
    score: float
    matched_field: str = "content"  # title, content, tags, etc.
    highlights: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.document.title} (score: {self.score:.3f})"


@dataclass
class GraphNode:
    """Node in the knowledge graph for visualization."""
    document: Document
    links: list[Link] = field(default_factory=list)
    depth: int = 0

    def __str__(self) -> str:
        return f"{self.document.title} ({len(self.links)} links)"


@dataclass
class Tag:
    """Tag with usage statistics."""
    name: str
    count: int = 0
    doc_type: Optional[DocumentType] = None

    def __str__(self) -> str:
        return f"{self.name} ({self.count})"
