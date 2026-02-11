"""Tests for error parsers."""

import pytest

from ai_debug_companion.models import ErrorSeverity, Language
from ai_debug_companion.parsers import (
    GoErrorParser,
    JavaScriptErrorParser,
    PythonErrorParser,
    RustErrorParser,
    UniversalErrorParser,
)


class TestPythonErrorParser:
    """Tests for Python error parser."""

    def test_parse_import_error(self):
        """Test parsing ImportError."""
        error_text = """
Traceback (most recent call last):
  File "main.py", line 5, in <module>
    import requests
ModuleNotFoundError: No module named 'requests'
        """

        parser = PythonErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.error_type == "ModuleNotFoundError"
        assert "requests" in error.message
        assert error.language == Language.PYTHON
        assert error.severity == ErrorSeverity.ERROR
        assert len(error.stack_trace) == 1
        assert error.stack_trace[0].file_path == "main.py"
        assert error.stack_trace[0].line_number == 5

    def test_parse_attribute_error(self):
        """Test parsing AttributeError."""
        error_text = """
Traceback (most recent call last):
  File "test.py", line 10, in process_data
    result = obj.nonexistent_method()
AttributeError: 'NoneType' object has no attribute 'nonexistent_method'
        """

        parser = PythonErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.error_type == "AttributeError"
        assert "nonexistent_method" in error.message
        assert error.stack_trace[0].function_name == "process_data"

    def test_parse_type_error(self):
        """Test parsing TypeError."""
        error_text = """
Traceback (most recent call last):
  File "calc.py", line 15, in calculate
    total = numbers + "hello"
TypeError: can only concatenate list (not "str") to list
        """

        parser = PythonErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.error_type == "TypeError"
        assert "concatenate" in error.message


class TestJavaScriptErrorParser:
    """Tests for JavaScript error parser."""

    def test_parse_reference_error(self):
        """Test parsing ReferenceError."""
        error_text = """
ReferenceError: undefinedVariable is not defined
    at processData (app.js:42:5)
    at main (app.js:100:3)
        """

        parser = JavaScriptErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.error_type == "ReferenceError"
        assert "undefinedVariable" in error.message
        assert error.language == Language.JAVASCRIPT
        assert len(error.stack_trace) >= 1
        assert error.stack_trace[0].function_name == "processData"
        assert error.stack_trace[0].file_path == "app.js"
        assert error.stack_trace[0].line_number == 42

    def test_parse_type_error(self):
        """Test parsing TypeError."""
        error_text = """
TypeError: Cannot read property 'length' of undefined
    at checkLength (utils.js:20:15)
        """

        parser = JavaScriptErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.error_type == "TypeError"
        assert "undefined" in error.message


class TestRustErrorParser:
    """Tests for Rust error parser."""

    def test_parse_compile_error(self):
        """Test parsing Rust compile error."""
        error_text = """
error[E0308]: mismatched types
 --> src/main.rs:10:5
  |
10|     "hello"
  |     ^^^^^^^ expected `i32`, found `&str`
        """

        parser = RustErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.error_type == "CompileError"
        assert "mismatched types" in error.message
        assert error.language == Language.RUST
        assert len(error.stack_trace) == 1
        assert "src/main.rs" in error.stack_trace[0].file_path
        assert error.stack_trace[0].line_number == 10


class TestGoErrorParser:
    """Tests for Go error parser."""

    def test_parse_panic(self):
        """Test parsing Go panic."""
        error_text = """
panic: runtime error: invalid memory address or nil pointer dereference

goroutine 1 [running]:
main.processData(...)
    /home/user/app/main.go:42 +0x89
main.main()
    /home/user/app/main.go:50 +0x2a
        """

        parser = GoErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.error_type == "panic"
        assert "invalid memory address" in error.message
        assert error.language == Language.GO
        assert error.severity == ErrorSeverity.CRITICAL


class TestUniversalErrorParser:
    """Tests for universal error parser."""

    def test_parse_python_error(self):
        """Test that universal parser detects Python errors."""
        error_text = """
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    raise ValueError("test error")
ValueError: test error
        """

        parser = UniversalErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.language == Language.PYTHON
        assert error.error_type == "ValueError"

    def test_parse_javascript_error(self):
        """Test that universal parser detects JavaScript errors."""
        error_text = """
ReferenceError: x is not defined
    at main (index.js:1:1)
        """

        parser = UniversalErrorParser()
        error = parser.parse(error_text)

        assert error is not None
        assert error.language == Language.JAVASCRIPT
        assert error.error_type == "ReferenceError"

    def test_parse_no_error(self):
        """Test parsing text with no error."""
        text = "This is just normal output with no errors."

        parser = UniversalErrorParser()
        error = parser.parse(text)

        assert error is None

    def test_parse_multiple_errors(self):
        """Test parsing multiple errors from text."""
        text = """
Error 1:
Traceback (most recent call last):
  File "a.py", line 1
    x = 1
SyntaxError: invalid syntax

Error 2:
ReferenceError: undefined
    at test.js:1:1
        """

        parser = UniversalErrorParser()
        errors = parser.parse_multiple(text)

        assert len(errors) >= 1  # Should find at least one error
