"""Tests for suggestion engine."""

import pytest

from ai_debug_companion.models import ErrorSeverity, Language, ParsedError, StackFrame
from ai_debug_companion.suggestions import SuggestionEngine


class TestSuggestionEngine:
    """Tests for suggestion engine."""

    def test_python_import_error_suggestion(self):
        """Test suggestion for Python ImportError."""
        error = ParsedError(
            error_type="ModuleNotFoundError",
            message="No module named 'requests'",
            severity=ErrorSeverity.ERROR,
            language=Language.PYTHON
        )

        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(error)

        assert len(suggestions) > 0

        # Should suggest installing the module
        install_suggestions = [s for s in suggestions if 'install' in s.title.lower()]
        assert len(install_suggestions) > 0
        assert 'requests' in install_suggestions[0].description

    def test_python_attribute_error_suggestion(self):
        """Test suggestion for Python AttributeError."""
        error = ParsedError(
            error_type="AttributeError",
            message="'NoneType' object has no attribute 'method'",
            severity=ErrorSeverity.ERROR,
            language=Language.PYTHON
        )

        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(error)

        assert len(suggestions) > 0
        # Should mention checking for None
        assert any('None' in s.description or 'type' in s.description.lower()
                   for s in suggestions)

    def test_javascript_module_not_found_suggestion(self):
        """Test suggestion for JavaScript module not found."""
        error = ParsedError(
            error_type="MODULE_NOT_FOUND",
            message="Cannot find module 'express'",
            severity=ErrorSeverity.ERROR,
            language=Language.JAVASCRIPT
        )

        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(error)

        assert len(suggestions) > 0

        # Should suggest npm install
        install_suggestions = [s for s in suggestions if 'npm install' in s.description]
        assert len(install_suggestions) > 0

    def test_javascript_undefined_suggestion(self):
        """Test suggestion for JavaScript undefined error."""
        error = ParsedError(
            error_type="TypeError",
            message="Cannot read property 'x' of undefined",
            severity=ErrorSeverity.ERROR,
            language=Language.JAVASCRIPT
        )

        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(error)

        assert len(suggestions) > 0
        assert any('undefined' in s.title.lower() or 'undefined' in s.description.lower()
                   for s in suggestions)

    def test_suggestion_confidence(self):
        """Test that suggestions have reasonable confidence scores."""
        error = ParsedError(
            error_type="ModuleNotFoundError",
            message="No module named 'nonexistent'",
            severity=ErrorSeverity.ERROR,
            language=Language.PYTHON
        )

        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(error)

        # All suggestions should have confidence between 0 and 1
        for suggestion in suggestions:
            assert 0.0 <= suggestion.confidence <= 1.0

    def test_common_pattern_suggestions(self):
        """Test that common patterns generate suggestions."""
        error = ParsedError(
            error_type="SyntaxError",
            message="invalid syntax",
            severity=ErrorSeverity.ERROR,
            language=Language.PYTHON
        )

        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(error)

        # Should have at least one suggestion even without git history
        assert len(suggestions) > 0
