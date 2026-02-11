"""
Code parsing and understanding for multiple languages.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional

from .models import Document, Language


class CodeParser(ABC):
    """Base class for language-specific code parsers."""

    @abstractmethod
    def can_parse(self, language: Language) -> bool:
        """Check if this parser can handle the language."""
        pass

    @abstractmethod
    def extract_symbols(self, code: str) -> list[str]:
        """Extract function/class names from code."""
        pass

    @abstractmethod
    def extract_imports(self, code: str) -> list[str]:
        """Extract import statements."""
        pass


class PythonParser(CodeParser):
    """Parser for Python code."""

    def can_parse(self, language: Language) -> bool:
        return language == Language.PYTHON

    def extract_symbols(self, code: str) -> list[str]:
        """Extract function and class definitions."""
        symbols = []

        # Class definitions
        class_pattern = re.compile(r'^class\s+(\w+)', re.MULTILINE)
        symbols.extend(class_pattern.findall(code))

        # Function definitions
        func_pattern = re.compile(r'^(?:async\s+)?def\s+(\w+)', re.MULTILINE)
        symbols.extend(func_pattern.findall(code))

        return symbols

    def extract_imports(self, code: str) -> list[str]:
        """Extract import statements."""
        imports = []

        # import foo
        import_pattern = re.compile(r'^import\s+([\w.]+)', re.MULTILINE)
        imports.extend(import_pattern.findall(code))

        # from foo import bar
        from_pattern = re.compile(r'^from\s+([\w.]+)\s+import', re.MULTILINE)
        imports.extend(from_pattern.findall(code))

        return list(set(imports))  # Deduplicate


class JavaScriptParser(CodeParser):
    """Parser for JavaScript/TypeScript code."""

    def can_parse(self, language: Language) -> bool:
        return language in (Language.JAVASCRIPT, Language.TYPESCRIPT)

    def extract_symbols(self, code: str) -> list[str]:
        """Extract function and class definitions."""
        symbols = []

        # Class definitions
        class_pattern = re.compile(r'class\s+(\w+)', re.MULTILINE)
        symbols.extend(class_pattern.findall(code))

        # Function declarations
        func_pattern = re.compile(r'function\s+(\w+)', re.MULTILINE)
        symbols.extend(func_pattern.findall(code))

        # Arrow functions assigned to const/let/var
        arrow_pattern = re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(', re.MULTILINE)
        symbols.extend(arrow_pattern.findall(code))

        # Method definitions in objects
        method_pattern = re.compile(r'(\w+)\s*\([^)]*\)\s*{', re.MULTILINE)
        symbols.extend(method_pattern.findall(code))

        return list(set(symbols))  # Deduplicate

    def extract_imports(self, code: str) -> list[str]:
        """Extract import statements."""
        imports = []

        # import foo from 'bar'
        import_from_pattern = re.compile(r'import\s+.+\s+from\s+[\'"](.+?)[\'"]', re.MULTILINE)
        imports.extend(import_from_pattern.findall(code))

        # import 'foo'
        import_pattern = re.compile(r'import\s+[\'"](.+?)[\'"]', re.MULTILINE)
        imports.extend(import_pattern.findall(code))

        # const foo = require('bar')
        require_pattern = re.compile(r'require\s*\(\s*[\'"](.+?)[\'"]\s*\)', re.MULTILINE)
        imports.extend(require_pattern.findall(code))

        return list(set(imports))


class GoParser(CodeParser):
    """Parser for Go code."""

    def can_parse(self, language: Language) -> bool:
        return language == Language.GO

    def extract_symbols(self, code: str) -> list[str]:
        """Extract function and type definitions."""
        symbols = []

        # Function definitions
        func_pattern = re.compile(r'^func\s+(?:\([^)]+\)\s+)?(\w+)', re.MULTILINE)
        symbols.extend(func_pattern.findall(code))

        # Type definitions
        type_pattern = re.compile(r'^type\s+(\w+)', re.MULTILINE)
        symbols.extend(type_pattern.findall(code))

        # Interface definitions
        interface_pattern = re.compile(r'^type\s+(\w+)\s+interface', re.MULTILINE)
        symbols.extend(interface_pattern.findall(code))

        return list(set(symbols))

    def extract_imports(self, code: str) -> list[str]:
        """Extract import statements."""
        imports = []

        # Single import
        single_import = re.compile(r'import\s+"(.+?)"', re.MULTILINE)
        imports.extend(single_import.findall(code))

        # Multi-line imports
        multi_import = re.compile(r'import\s*\(\s*((?:"[^"]+"\s*)+)\)', re.MULTILINE | re.DOTALL)
        for match in multi_import.findall(code):
            pkg_pattern = re.compile(r'"(.+?)"')
            imports.extend(pkg_pattern.findall(match))

        return list(set(imports))


class RustParser(CodeParser):
    """Parser for Rust code."""

    def can_parse(self, language: Language) -> bool:
        return language == Language.RUST

    def extract_symbols(self, code: str) -> list[str]:
        """Extract function, struct, and trait definitions."""
        symbols = []

        # Function definitions
        func_pattern = re.compile(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)', re.MULTILINE)
        symbols.extend(func_pattern.findall(code))

        # Struct definitions
        struct_pattern = re.compile(r'^(?:pub\s+)?struct\s+(\w+)', re.MULTILINE)
        symbols.extend(struct_pattern.findall(code))

        # Trait definitions
        trait_pattern = re.compile(r'^(?:pub\s+)?trait\s+(\w+)', re.MULTILINE)
        symbols.extend(trait_pattern.findall(code))

        # Enum definitions
        enum_pattern = re.compile(r'^(?:pub\s+)?enum\s+(\w+)', re.MULTILINE)
        symbols.extend(enum_pattern.findall(code))

        return list(set(symbols))

    def extract_imports(self, code: str) -> list[str]:
        """Extract use statements."""
        imports = []

        # use foo::bar
        use_pattern = re.compile(r'^use\s+([\w:]+)', re.MULTILINE)
        imports.extend(use_pattern.findall(code))

        return list(set(imports))


class UniversalCodeParser:
    """Combines all language-specific parsers."""

    def __init__(self):
        self.parsers: list[CodeParser] = [
            PythonParser(),
            JavaScriptParser(),
            GoParser(),
            RustParser(),
        ]

    def parse_document(self, doc: Document) -> Document:
        """Parse a document and extract code information."""
        if doc.doc_type.value not in ('code', 'snippet'):
            return doc

        # Find appropriate parser
        parser = None
        for p in self.parsers:
            if p.can_parse(doc.language):
                parser = p
                break

        if not parser:
            return doc

        # Extract symbols and imports
        doc.symbols = parser.extract_symbols(doc.content)
        doc.imports = parser.extract_imports(doc.content)

        return doc

    def detect_language(self, code: str, filename: Optional[str] = None) -> Language:
        """Detect programming language from code or filename."""
        if filename:
            ext = filename.split('.')[-1].lower() if '.' in filename else ''
            extension_map = {
                'py': Language.PYTHON,
                'js': Language.JAVASCRIPT,
                'jsx': Language.JAVASCRIPT,
                'ts': Language.TYPESCRIPT,
                'tsx': Language.TYPESCRIPT,
                'go': Language.GO,
                'rs': Language.RUST,
                'java': Language.JAVA,
                'cpp': Language.CPP,
                'cc': Language.CPP,
                'c': Language.CPP,
                'md': Language.MARKDOWN,
            }
            if ext in extension_map:
                return extension_map[ext]

        # Try to detect from code patterns
        if re.search(r'^def\s+\w+|^class\s+\w+|^import\s+\w+', code, re.MULTILINE):
            return Language.PYTHON
        elif re.search(r'function\s+\w+|class\s+\w+|const\s+\w+\s*=', code, re.MULTILINE):
            return Language.JAVASCRIPT
        elif re.search(r'^func\s+\w+|^type\s+\w+|^package\s+\w+', code, re.MULTILINE):
            return Language.GO
        elif re.search(r'^fn\s+\w+|^struct\s+\w+|^use\s+[\w:]+', code, re.MULTILINE):
            return Language.RUST

        return Language.TEXT


def extract_tags_from_content(doc: Document) -> list[str]:
    """Auto-generate tags from document content."""
    tags = set(doc.tags)  # Start with existing tags

    # Extract from symbols (languages, frameworks, etc.)
    if doc.imports:
        for imp in doc.imports:
            # Extract top-level package name
            parts = imp.replace('/', '.').replace('::', '.').split('.')
            if parts:
                tags.add(parts[0].lower())

    # Language tag
    if doc.language != Language.TEXT:
        tags.add(doc.language.value)

    # Common keywords from content
    content_lower = doc.content.lower()
    keywords = [
        'async', 'await', 'promise', 'callback',
        'api', 'http', 'rest', 'graphql',
        'database', 'sql', 'nosql',
        'test', 'testing', 'unit', 'integration',
        'error', 'exception', 'handling',
        'performance', 'optimization',
        'security', 'auth', 'authentication',
        'docker', 'kubernetes', 'deployment',
    ]

    for keyword in keywords:
        if keyword in content_lower:
            tags.add(keyword)

    return sorted(list(tags))
