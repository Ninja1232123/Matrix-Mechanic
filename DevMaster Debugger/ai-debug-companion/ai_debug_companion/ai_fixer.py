"""
AI-powered code fixer using flexible AI provider system.

Uses AI to generate intelligent fixes for complex errors that pattern matching can't handle.
"""

from typing import Optional
from pathlib import Path

from .models import ParsedError, Suggestion
from .fixer import CodeFix
from .ai_provider import AIProviderFactory, AIProvider


class AICodeFixer:
    """AI-powered code fixer for complex errors."""

    def __init__(self, provider: Optional[AIProvider] = None):
        """
        Initialize with an AI provider.

        If provider is None, will auto-detect from config/environment.
        """
        self.provider = provider or self._get_provider()

    def _get_provider(self) -> Optional[AIProvider]:
        """Get AI provider, return None if not available."""
        try:
            return AIProviderFactory.from_config()
        except:
            # AI provider not configured - that's okay, we'll fall back to pattern-based fixes
            return None

    def is_available(self) -> bool:
        """Check if AI-powered fixing is available."""
        return self.provider is not None

    def generate_intelligent_fix(
        self,
        error: ParsedError,
        suggestion: Suggestion,
        context_lines: int = 10
    ) -> Optional[CodeFix]:
        """
        Generate an intelligent fix using AI.

        Args:
            error: The parsed error
            suggestion: The suggestion to implement
            context_lines: Number of lines of context to include

        Returns:
            CodeFix if successful, None otherwise
        """
        if not self.is_available():
            return None

        location = error.primary_location
        if not location:
            return None

        # Read the file to get context
        try:
            with open(location.file_path, 'r') as f:
                lines = f.readlines()

            if location.line_number > len(lines):
                return None

            # Get context around the error
            line_idx = location.line_number - 1
            start = max(0, line_idx - context_lines)
            end = min(len(lines), line_idx + context_lines + 1)

            context = ''.join(lines[start:end])
            error_line = lines[line_idx].rstrip()

        except (FileNotFoundError, IOError):
            return None

        # Generate fix using AI
        prompt = self._create_fix_prompt(
            error=error,
            suggestion=suggestion,
            error_line=error_line,
            context=context,
            line_number=location.line_number
        )

        try:
            response = self.provider.generate(prompt, max_tokens=500)
            fixed_code = self._extract_fix(response.content, error_line)

            if fixed_code and fixed_code != error_line:
                return CodeFix(
                    file_path=location.file_path,
                    line_number=location.line_number,
                    original_code=error_line,
                    fixed_code=fixed_code,
                    explanation=f"AI-generated fix using {self.provider.model}",
                    confidence=0.75  # Slightly lower confidence for AI fixes
                )

        except Exception as e:
            print(f"⚠️  AI fix generation failed: {e}")
            return None

        return None

    def _create_fix_prompt(
        self,
        error: ParsedError,
        suggestion: Suggestion,
        error_line: str,
        context: str,
        line_number: int
    ) -> str:
        """Create a prompt for the AI to generate a fix."""
        return f"""You are a code fixing assistant. Fix the following error.

Error Type: {error.error_type}
Error Message: {error.message}
Language: {error.language.value}
Line {line_number}: {error_line}

Suggestion: {suggestion.title}
{suggestion.description}

Code Context:
```
{context}
```

Instructions:
1. Provide ONLY the fixed version of line {line_number}
2. Keep the same indentation
3. Keep the same variable names
4. Make minimal changes
5. Do NOT include line numbers or explanations
6. Output ONLY the fixed line of code, nothing else

Fixed line {line_number}:"""

    def _extract_fix(self, ai_response: str, original_line: str) -> Optional[str]:
        """Extract the fixed code from AI response."""
        # Clean up the response
        response = ai_response.strip()

        # Remove common AI response artifacts
        if response.startswith("```"):
            # Extract code from markdown code block
            lines = response.split('\n')
            code_lines = []
            in_code = False

            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    code_lines.append(line)

            if code_lines:
                response = '\n'.join(code_lines).strip()

        # Take only the first line if multiple lines returned
        if '\n' in response:
            response = response.split('\n')[0].strip()

        # Preserve original indentation
        original_indent = len(original_line) - len(original_line.lstrip())
        response_indent = len(response) - len(response.lstrip())

        if response_indent != original_indent:
            # Fix indentation
            response = ' ' * original_indent + response.lstrip()

        # Sanity check: must be similar enough to original
        if len(response) == 0:
            return None

        # Don't return if it's just repeating the error
        if response == original_line:
            return None

        return response

    def explain_error(self, error: ParsedError, max_tokens: int = 300) -> Optional[str]:
        """
        Get an AI explanation of what caused the error and how to fix it.

        Returns a human-readable explanation.
        """
        if not self.is_available():
            return None

        prompt = f"""Explain this error briefly and suggest how to fix it.

Error: {error.error_type}
Message: {error.message}
Language: {error.language.value}

Provide a concise explanation (2-3 sentences) that a developer can understand quickly.
Focus on:
1. What caused the error
2. How to fix it
3. How to prevent it in the future

Keep it practical and actionable."""

        try:
            response = self.provider.generate(prompt, max_tokens=max_tokens)
            return response.content.strip()
        except Exception as e:
            print(f"⚠️  AI explanation failed: {e}")
            return None

    def suggest_improvements(
        self,
        file_path: str,
        line_number: int,
        code: str,
        max_tokens: int = 400
    ) -> Optional[str]:
        """
        Suggest code improvements for a specific line.

        Args:
            file_path: Path to the file
            line_number: Line number
            code: The code to improve
            max_tokens: Maximum tokens for response

        Returns:
            Improvement suggestions or None
        """
        if not self.is_available():
            return None

        # Detect language from file extension
        ext = Path(file_path).suffix
        language = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.go': 'Go',
            '.rs': 'Rust',
            '.java': 'Java'
        }.get(ext, 'code')

        prompt = f"""Review this {language} code and suggest improvements.

File: {file_path}
Line {line_number}: {code}

Suggest improvements for:
1. Error handling
2. Code clarity
3. Best practices
4. Performance (if relevant)

Be concise and specific. Focus on the most important improvements."""

        try:
            response = self.provider.generate(prompt, max_tokens=max_tokens)
            return response.content.strip()
        except Exception as e:
            print(f"⚠️  AI suggestions failed: {e}")
            return None
