"""
Multi-language AST parser for extracting code symbols.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .models import CodeSymbol, Language, SymbolType


class LanguageParser(ABC):
    """Base class for language-specific parsers."""

    @abstractmethod
    def can_parse(self, language: Language) -> bool:
        """Check if this parser handles the language."""
        pass

    @abstractmethod
    def parse_file(self, file_path: str, content: str) -> list[CodeSymbol]:
        """Parse file and extract symbols."""
        pass


class PythonParser(LanguageParser):
    """Parser for Python code."""

    def can_parse(self, language: Language) -> bool:
        return language == Language.PYTHON

    def parse_file(self, file_path: str, content: str) -> list[CodeSymbol]:
        """Parse Python file using regex (simple implementation)."""
        symbols = []
        lines = content.split('\n')

        # Find classes
        class_pattern = re.compile(r'^class\s+(\w+)(\([^)]*\))?:', re.MULTILINE)
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1)

            code = '\n'.join(lines[start_line - 1:end_line])
            docstring = self._extract_docstring(code)

            symbols.append(CodeSymbol(
                name=class_name,
                symbol_type=SymbolType.CLASS,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                docstring=docstring,
                language=Language.PYTHON
            ))

        # Find functions
        func_pattern = re.compile(r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)
        current_class = None

        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            params_str = match.group(2)
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_block_end(lines, start_line - 1)

            code = '\n'.join(lines[start_line - 1:end_line])
            docstring = self._extract_docstring(code)

            # Parse parameters
            params = [p.strip().split(':')[0].split('=')[0].strip()
                     for p in params_str.split(',') if p.strip()]

            # Check if it's a method (indented)
            line_content = lines[start_line - 1]
            is_method = line_content.startswith('    ') or line_content.startswith('\t')

            # Detect async
            is_async = 'async' in lines[start_line - 1]
            modifiers = ['async'] if is_async else []

            symbols.append(CodeSymbol(
                name=func_name,
                symbol_type=SymbolType.METHOD if is_method else SymbolType.FUNCTION,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                docstring=docstring,
                language=Language.PYTHON,
                params=params,
                modifiers=modifiers
            ))

        return symbols

    def _find_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end of a Python block (class/function)."""
        if start_idx >= len(lines):
            return start_idx + 1

        # Find initial indentation
        start_line = lines[start_idx]
        base_indent = len(start_line) - len(start_line.lstrip())

        # Scan forward to find end of block
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]

            # Empty or comment line
            if not line.strip() or line.strip().startswith('#'):
                continue

            # Calculate indentation
            current_indent = len(line) - len(line.lstrip())

            # If we're back to or less than base indentation, block ended
            if current_indent <= base_indent:
                return i

        return len(lines)

    def _extract_docstring(self, code: str) -> Optional[str]:
        """Extract docstring from code."""
        # Look for triple-quoted strings at the start
        docstring_pattern = re.compile(r'^\s*["\']{{3}}(.+?)["\']{{3}}', re.DOTALL)
        match = docstring_pattern.search(code)
        if match:
            return match.group(1).strip()
        return None


class JavaScriptParser(LanguageParser):
    """Parser for JavaScript/TypeScript code."""

    def can_parse(self, language: Language) -> bool:
        return language in (Language.JAVASCRIPT, Language.TYPESCRIPT)

    def parse_file(self, file_path: str, content: str) -> list[CodeSymbol]:
        """Parse JavaScript/TypeScript file."""
        symbols = []
        lines = content.split('\n')

        # Find classes
        class_pattern = re.compile(r'class\s+(\w+)', re.MULTILINE)
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_brace_block_end(lines, start_line - 1)

            code = '\n'.join(lines[start_line - 1:end_line])

            symbols.append(CodeSymbol(
                name=class_name,
                symbol_type=SymbolType.CLASS,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                language=Language.JAVASCRIPT
            ))

        # Find function declarations
        func_pattern = re.compile(r'(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)
        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            params_str = match.group(2)
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_brace_block_end(lines, start_line - 1)

            code = '\n'.join(lines[start_line - 1:end_line])
            params = [p.strip().split(':')[0].split('=')[0].strip()
                     for p in params_str.split(',') if p.strip()]

            is_async = 'async' in lines[start_line - 1]

            symbols.append(CodeSymbol(
                name=func_name,
                symbol_type=SymbolType.FUNCTION,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                language=Language.JAVASCRIPT,
                params=params,
                modifiers=['async'] if is_async else []
            ))

        # Find arrow functions assigned to const/let/var
        arrow_pattern = re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>', re.MULTILINE)
        for match in arrow_pattern.finditer(content):
            func_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            # Arrow functions might not have explicit braces
            end_line = min(start_line + 20, len(lines))  # Heuristic

            code = '\n'.join(lines[start_line - 1:end_line])

            symbols.append(CodeSymbol(
                name=func_name,
                symbol_type=SymbolType.FUNCTION,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                language=Language.JAVASCRIPT
            ))

        return symbols

    def _find_brace_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end of a brace-delimited block."""
        if start_idx >= len(lines):
            return start_idx + 1

        brace_count = 0
        found_opening = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        return i + 1

        return len(lines)


class GoParser(LanguageParser):
    """Parser for Go code."""

    def can_parse(self, language: Language) -> bool:
        return language == Language.GO

    def parse_file(self, file_path: str, content: str) -> list[CodeSymbol]:
        """Parse Go file."""
        symbols = []
        lines = content.split('\n')

        # Find function definitions
        func_pattern = re.compile(r'^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(([^)]*)\)', re.MULTILINE)
        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            params_str = match.group(2)
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_brace_block_end(lines, start_line - 1)

            code = '\n'.join(lines[start_line - 1:end_line])
            params = [p.strip().split()[0] for p in params_str.split(',') if p.strip()]

            symbols.append(CodeSymbol(
                name=func_name,
                symbol_type=SymbolType.FUNCTION,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                language=Language.GO,
                params=params
            ))

        # Find struct definitions
        struct_pattern = re.compile(r'^type\s+(\w+)\s+struct', re.MULTILINE)
        for match in struct_pattern.finditer(content):
            struct_name = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            end_line = self._find_brace_block_end(lines, start_line - 1)

            code = '\n'.join(lines[start_line - 1:end_line])

            symbols.append(CodeSymbol(
                name=struct_name,
                symbol_type=SymbolType.STRUCT,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                code=code,
                language=Language.GO
            ))

        return symbols

    def _find_brace_block_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end of a brace-delimited block."""
        if start_idx >= len(lines):
            return start_idx + 1

        brace_count = 0
        found_opening = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        return i + 1

        return len(lines)


class UniversalParser:
    """Combines all language parsers."""

    def __init__(self):
        self.parsers: list[LanguageParser] = [
            PythonParser(),
            JavaScriptParser(),
            GoParser(),
        ]

    def parse_file(self, file_path: str, language: Language) -> list[CodeSymbol]:
        """Parse a file with the appropriate parser."""
        # Find matching parser
        for parser in self.parsers:
            if parser.can_parse(language):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return parser.parse_file(file_path, content)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    return []

        return []

    def detect_language(self, file_path: str) -> Language:
        """Detect language from file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        extension_map = {
            '.py': Language.PYTHON,
            '.js': Language.JAVASCRIPT,
            '.jsx': Language.JAVASCRIPT,
            '.ts': Language.TYPESCRIPT,
            '.tsx': Language.TYPESCRIPT,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.java': Language.JAVA,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.c': Language.C,
        }

        return extension_map.get(ext, Language.UNKNOWN)
