"""
Error parsers for different programming languages and frameworks.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional

from .models import ErrorSeverity, Language, ParsedError, StackFrame


class ErrorParser(ABC):
    """Base class for language-specific error parsers."""

    @abstractmethod
    def can_parse(self, text: str) -> bool:
        """Check if this parser can handle the given text."""
        pass

    @abstractmethod
    def parse(self, text: str) -> Optional[ParsedError]:
        """Parse error from text, return None if parsing fails."""
        pass


class PythonErrorParser(ErrorParser):
    """Parser for Python errors and exceptions."""

    TRACEBACK_PATTERN = re.compile(r'Traceback \(most recent call last\):', re.MULTILINE)
    ERROR_PATTERN = re.compile(
        r'^(\w+(?:\.\w+)*(?:Error|Exception|Warning)): (.+)$', re.MULTILINE
    )
    STACK_FRAME_PATTERN = re.compile(
        r'  File "(.+?)", line (\d+), in (.+?)\n\s+(.+?)(?=\n|$)', re.MULTILINE
    )

    def can_parse(self, text: str) -> bool:
        return bool(self.TRACEBACK_PATTERN.search(text)) or bool(self.ERROR_PATTERN.search(text))

    def parse(self, text: str) -> Optional[ParsedError]:
        # Find error type and message
        error_match = self.ERROR_PATTERN.search(text)
        if not error_match:
            return None

        error_type = error_match.group(1)
        message = error_match.group(2)

        # Extract stack frames
        stack_trace = []
        for match in self.STACK_FRAME_PATTERN.finditer(text):
            frame = StackFrame(
                file_path=match.group(1),
                line_number=int(match.group(2)),
                function_name=match.group(3),
                code_context=match.group(4).strip()
            )
            stack_trace.append(frame)

        # Determine severity
        severity = ErrorSeverity.ERROR
        if 'Warning' in error_type:
            severity = ErrorSeverity.WARNING
        elif error_type in ('SystemExit', 'KeyboardInterrupt'):
            severity = ErrorSeverity.INFO

        return ParsedError(
            error_type=error_type,
            message=message,
            severity=severity,
            language=Language.PYTHON,
            stack_trace=stack_trace,
            full_output=text
        )


class JavaScriptErrorParser(ErrorParser):
    """Parser for JavaScript/Node.js errors."""

    ERROR_PATTERN = re.compile(
        r'^(\w+(?:Error)?): (.+)$', re.MULTILINE
    )
    STACK_FRAME_PATTERN = re.compile(
        r'at (?:(.+?) \()?(.+?):(\d+):(\d+)\)?', re.MULTILINE
    )

    def can_parse(self, text: str) -> bool:
        # JavaScript errors should have both error message and "at " in stack trace
        # or just "at " for stack-only outputs
        return 'at ' in text and (':' in text)

    def parse(self, text: str) -> Optional[ParsedError]:
        error_match = self.ERROR_PATTERN.search(text)
        if not error_match:
            return None

        error_type = error_match.group(1)
        message = error_match.group(2)

        # Extract stack frames
        stack_trace = []
        for match in self.STACK_FRAME_PATTERN.finditer(text):
            frame = StackFrame(
                file_path=match.group(2),
                line_number=int(match.group(3)),
                function_name=match.group(1) if match.group(1) else None
            )
            stack_trace.append(frame)

        return ParsedError(
            error_type=error_type,
            message=message,
            severity=ErrorSeverity.ERROR,
            language=Language.JAVASCRIPT,
            stack_trace=stack_trace,
            full_output=text
        )


class RustErrorParser(ErrorParser):
    """Parser for Rust compiler errors."""

    ERROR_PATTERN = re.compile(
        r'error(?:\[E\d+\])?: (.+)', re.MULTILINE
    )
    LOCATION_PATTERN = re.compile(
        r'--> (.+?):(\d+):(\d+)', re.MULTILINE
    )

    def can_parse(self, text: str) -> bool:
        return 'error:' in text.lower() and '-->' in text

    def parse(self, text: str) -> Optional[ParsedError]:
        error_match = self.ERROR_PATTERN.search(text)
        if not error_match:
            return None

        message = error_match.group(1)
        error_type = "CompileError"

        # Find location
        location_match = self.LOCATION_PATTERN.search(text)
        stack_trace = []
        if location_match:
            frame = StackFrame(
                file_path=location_match.group(1),
                line_number=int(location_match.group(2))
            )
            stack_trace.append(frame)

        return ParsedError(
            error_type=error_type,
            message=message,
            severity=ErrorSeverity.ERROR,
            language=Language.RUST,
            stack_trace=stack_trace,
            full_output=text
        )


class GoErrorParser(ErrorParser):
    """Parser for Go errors and panics."""

    PANIC_PATTERN = re.compile(r'panic: (.+)', re.MULTILINE)
    ERROR_PATTERN = re.compile(r'# (.+?)\n(.+?):(\d+):', re.MULTILINE)
    GOROUTINE_PATTERN = re.compile(
        r'(.+?):(\d+) \+0x[\da-f]+', re.MULTILINE
    )

    def can_parse(self, text: str) -> bool:
        return 'panic:' in text or 'goroutine' in text

    def parse(self, text: str) -> Optional[ParsedError]:
        panic_match = self.PANIC_PATTERN.search(text)
        if not panic_match:
            return None

        message = panic_match.group(1)
        error_type = "panic"

        # Extract goroutine stack
        stack_trace = []
        for match in self.GOROUTINE_PATTERN.finditer(text):
            frame = StackFrame(
                file_path=match.group(1),
                line_number=int(match.group(2))
            )
            stack_trace.append(frame)

        return ParsedError(
            error_type=error_type,
            message=message,
            severity=ErrorSeverity.CRITICAL,
            language=Language.GO,
            stack_trace=stack_trace,
            full_output=text
        )


class UniversalErrorParser:
    """Combines all language-specific parsers."""

    def __init__(self):
        self.parsers: list[ErrorParser] = [
            JavaScriptErrorParser(),  # Check JS first - more specific stack traces
            RustErrorParser(),
            GoErrorParser(),
            PythonErrorParser(),  # Python last - more general patterns
        ]

    def parse(self, text: str) -> Optional[ParsedError]:
        """Try to parse error with all available parsers."""
        for parser in self.parsers:
            if parser.can_parse(text):
                result = parser.parse(text)
                if result:
                    return result
        return None

    def parse_multiple(self, text: str) -> list[ParsedError]:
        """Parse multiple errors from text."""
        errors = []
        lines = text.split('\n')

        # Try to find error boundaries and parse each section
        current_section = []
        for line in lines:
            current_section.append(line)
            section_text = '\n'.join(current_section)

            error = self.parse(section_text)
            if error:
                errors.append(error)
                current_section = []

        return errors
