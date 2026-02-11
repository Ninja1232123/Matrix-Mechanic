"""
Suggestion engine that generates fix recommendations based on errors and history.
"""

import re
from typing import Optional

from .history import GitHistoryAnalyzer
from .models import HistoricalFix, ParsedError, Suggestion


class SuggestionEngine:
    """Generates suggestions for fixing errors."""

    def __init__(self, repo_path: str = "."):
        self.history_analyzer = GitHistoryAnalyzer(repo_path)
        self.common_patterns = self._load_common_patterns()

    def generate_suggestions(self, error: ParsedError) -> list[Suggestion]:
        """Generate all suggestions for an error."""
        suggestions = []

        # Get suggestions from git history
        suggestions.extend(self._suggestions_from_history(error))

        # Get suggestions from common patterns
        suggestions.extend(self._suggestions_from_patterns(error))

        # Get language-specific suggestions
        suggestions.extend(self._language_specific_suggestions(error))

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions

    def _suggestions_from_history(self, error: ParsedError) -> list[Suggestion]:
        """Generate suggestions based on git history."""
        suggestions = []

        # Find similar fixes in history
        historical_fixes = self.history_analyzer.find_similar_fixes(error, max_results=3)

        for fix in historical_fixes:
            suggestion = self._create_suggestion_from_fix(fix)
            suggestions.append(suggestion)

        # Check file-specific history if we have a location
        if error.primary_location:
            file_fixes = self.history_analyzer.find_fixes_for_file(
                error.primary_location.file_path,
                error.error_type,
                max_results=2
            )
            for fix in file_fixes:
                suggestion = self._create_suggestion_from_fix(fix)
                suggestions.append(suggestion)

        return suggestions

    def _suggestions_from_patterns(self, error: ParsedError) -> list[Suggestion]:
        """Generate suggestions based on common error patterns."""
        suggestions = []

        # Look up common patterns for this error type
        if error.error_type in self.common_patterns:
            patterns = self.common_patterns[error.error_type]
            for pattern in patterns:
                if pattern['matcher'](error):
                    suggestion = Suggestion(
                        title=pattern['title'],
                        description=pattern['description'],
                        confidence=pattern['confidence'],
                        category='common_fix',
                        action=pattern.get('action'),
                        action_data=pattern.get('action_data', {})
                    )
                    suggestions.append(suggestion)

        return suggestions

    def _language_specific_suggestions(self, error: ParsedError) -> list[Suggestion]:
        """Generate language-specific suggestions."""
        suggestions = []

        # Python-specific
        if error.language.value == 'python':
            suggestions.extend(self._python_suggestions(error))
        # JavaScript-specific
        elif error.language.value in ('javascript', 'typescript'):
            suggestions.extend(self._javascript_suggestions(error))

        return suggestions

    def _python_suggestions(self, error: ParsedError) -> list[Suggestion]:
        """Python-specific error suggestions."""
        suggestions = []

        # ImportError suggestions
        if 'ImportError' in error.error_type or 'ModuleNotFoundError' in error.error_type:
            module_match = re.search(r"No module named '(\w+)'", error.message)
            if module_match:
                module_name = module_match.group(1)
                suggestions.append(Suggestion(
                    title=f"Install missing module: {module_name}",
                    description=f"Run: pip install {module_name}",
                    confidence=0.9,
                    category='common_fix',
                    action='command',
                    action_data={'command': f'pip install {module_name}'}
                ))

        # AttributeError suggestions
        elif 'AttributeError' in error.error_type:
            suggestions.append(Suggestion(
                title="Check object type and available attributes",
                description="The object might be None or of a different type than expected",
                confidence=0.7,
                category='common_fix'
            ))

        # IndentationError
        elif 'IndentationError' in error.error_type:
            suggestions.append(Suggestion(
                title="Fix indentation",
                description="Check for mixed tabs and spaces, or incorrect indentation level",
                confidence=0.95,
                category='common_fix'
            ))

        # ZeroDivisionError
        elif 'ZeroDivisionError' in error.error_type:
            suggestions.append(Suggestion(
                title="Add division by zero check",
                description="Add a condition to check if the divisor is zero before division",
                confidence=0.9,
                category='common_fix'
            ))
            suggestions.append(Suggestion(
                title="Validate input data",
                description="Ensure input values are validated before mathematical operations",
                confidence=0.75,
                category='common_fix'
            ))

        # KeyError
        elif 'KeyError' in error.error_type:
            suggestions.append(Suggestion(
                title="Use dict.get() with default value",
                description="Replace dict[key] with dict.get(key, default) to avoid KeyError",
                confidence=0.85,
                category='common_fix'
            ))
            suggestions.append(Suggestion(
                title="Check if key exists",
                description="Use 'if key in dict:' before accessing the key",
                confidence=0.8,
                category='common_fix'
            ))

        # IndexError
        elif 'IndexError' in error.error_type:
            suggestions.append(Suggestion(
                title="Check list length before accessing",
                description="Verify the list has enough elements before accessing by index",
                confidence=0.85,
                category='common_fix'
            ))

        # ValueError
        elif 'ValueError' in error.error_type:
            suggestions.append(Suggestion(
                title="Validate input values",
                description="Check that input values match expected format/range before processing",
                confidence=0.75,
                category='common_fix'
            ))

        # NameError
        elif 'NameError' in error.error_type:
            suggestions.append(Suggestion(
                title="Check variable name spelling",
                description="The variable might be misspelled or not defined yet",
                confidence=0.8,
                category='common_fix'
            ))

        # FileNotFoundError
        elif 'FileNotFoundError' in error.error_type:
            suggestions.append(Suggestion(
                title="Add file existence check or create file",
                description="Wrap file operations in try/except or check file exists before opening",
                confidence=0.85,
                category='common_fix'
            ))
            suggestions.append(Suggestion(
                title="Validate input data",
                description="Check that file paths are valid before attempting to open",
                confidence=0.75,
                category='common_fix'
            ))

        # JSONDecodeError
        elif 'JSONDecodeError' in error.error_type:
            suggestions.append(Suggestion(
                title="Add JSON parsing error handling",
                description="Wrap json.loads() in try/except to handle invalid JSON",
                confidence=0.8,
                category='common_fix'
            ))
            suggestions.append(Suggestion(
                title="Validate input data",
                description="Verify JSON string format before parsing",
                confidence=0.7,
                category='common_fix'
            ))

        return suggestions

    def _javascript_suggestions(self, error: ParsedError) -> list[Suggestion]:
        """JavaScript/TypeScript-specific suggestions."""
        suggestions = []

        # Cannot find module
        if 'Cannot find module' in error.message or 'MODULE_NOT_FOUND' in error.error_type:
            module_match = re.search(r"Cannot find module '(.+?)'", error.message)
            if module_match:
                module_name = module_match.group(1)
                suggestions.append(Suggestion(
                    title=f"Install missing module: {module_name}",
                    description=f"Run: npm install {module_name}",
                    confidence=0.9,
                    category='common_fix',
                    action='command',
                    action_data={'command': f'npm install {module_name}'}
                ))

        # undefined is not a function/object
        elif 'undefined' in error.message.lower():
            suggestions.append(Suggestion(
                title="Check for undefined values",
                description="Add null/undefined checks before accessing properties or methods",
                confidence=0.75,
                category='common_fix'
            ))

        return suggestions

    def _create_suggestion_from_fix(self, fix: HistoricalFix) -> Suggestion:
        """Create a suggestion from a historical fix."""
        # Extract key changes from diff
        description = self._summarize_diff(fix.diff)

        return Suggestion(
            title=f"Similar fix: {fix.commit_message[:60]}",
            description=description,
            confidence=fix.similarity_score * 0.8,  # Slightly lower than similarity
            category='history',
            action='view_commit',
            action_data={'commit_hash': fix.commit_hash},
            related_fixes=[fix]
        )

    def _summarize_diff(self, diff: str) -> str:
        """Create a brief summary of changes from a diff."""
        if not diff:
            return "No diff available"

        lines = diff.split('\n')
        additions = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))

        # Find modified files
        files = re.findall(r'diff --git a/(.+?) b/', diff)
        files_str = ', '.join(files[:3])
        if len(files) > 3:
            files_str += f" and {len(files) - 3} more"

        return f"Modified {files_str}: +{additions} -{deletions} lines"

    def _load_common_patterns(self) -> dict:
        """Load common error patterns and their fixes."""
        return {
            'SyntaxError': [
                {
                    'title': 'Check for missing brackets or quotes',
                    'description': 'Syntax errors often result from unclosed brackets, parentheses, or quotes',
                    'confidence': 0.7,
                    'matcher': lambda e: True
                }
            ],
            'TypeError': [
                {
                    'title': 'Verify variable types',
                    'description': 'Check that variables are of the expected type before operations',
                    'confidence': 0.65,
                    'matcher': lambda e: True
                }
            ],
            'ReferenceError': [
                {
                    'title': 'Variable not declared',
                    'description': 'Make sure the variable is declared before use',
                    'confidence': 0.8,
                    'matcher': lambda e: True
                }
            ],
        }
