"""
Automatic code fixing engine with hard-coded solutions.

Every Python error has a pattern-matched solution. No AI needed.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import ParsedError


@dataclass
class CodeFix:
    """Represents a concrete fix that can be applied to code."""
    file_path: str
    line_number: int
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float

    def preview(self) -> str:
        """Generate a preview of the fix."""
        return f"""
File: {self.file_path}:{self.line_number}

Original:
  {self.original_code}

Fixed:
  {self.fixed_code}

Explanation: {self.explanation}
Confidence: {int(self.confidence * 100)}%
"""


def get_indent(line):
    """Extract indentation from line."""
    return ' ' * (len(line) - len(line.lstrip()))


def wrap_in_try_except(line, exception_type, indent_level=0):
    """Wrap line in try/except block with proper indentation."""
    base_indent = ' ' * indent_level
    inner_indent = ' ' * (indent_level + 4)

    return f"{base_indent}try:\n{inner_indent}{line.strip()}\n{base_indent}except {exception_type}:\n{inner_indent}return {{}}\n"


def get_indented_block(lines, start_idx):
    """Get all lines in an indented block starting from start_idx."""
    if start_idx >= len(lines):
        return ([], 0)

    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    block_lines = [lines[start_idx].rstrip()]

    idx = start_idx + 1
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == '':
            block_lines.append('')
            idx += 1
            continue

        line_indent = len(line) - len(line.lstrip())
        if line_indent <= base_indent:
            break

        block_lines.append(line.rstrip())
        idx += 1

    return (block_lines, base_indent)


def wrap_block_in_try_except(block_lines, base_indent, exception_type):
    """Wrap a multi-line block in try/except."""
    spaces = ' ' * base_indent
    inner_spaces = ' ' * (base_indent + 4)

    fixed_lines = [f"{spaces}try:"]
    for block_line in block_lines:
        if block_line.strip():
            line_indent = len(block_line) - len(block_line.lstrip())
            extra_indent = line_indent - base_indent
            fixed_lines.append(f"{inner_spaces}{' ' * extra_indent}{block_line.lstrip()}")
        else:
            fixed_lines.append('')
    fixed_lines.append(f"{spaces}except {exception_type}:")
    fixed_lines.append(f"{inner_spaces}return {{}}")

    return '\n'.join(fixed_lines) + '\n'


def _fix_name_error(line, indent, error_msg):
    """Fix NameError by initializing undefined variable."""
    pattern = r"name '(\w+)' is not defined"
    match = re.search(pattern, error_msg)
    if match:
        var_name = match.group(1)
        return f"{indent}{var_name} = None  # Initialize variable\n{line}"
    return line


def _fix_import_error(line, indent, error_msg):
    """Fix ImportError by adding pip install comment."""
    pattern = r"No module named '(\w+)'"
    match = re.search(pattern, error_msg)
    if match:
        module_name = match.group(1)
        return f"{indent}# Run: pip install {module_name}\n{line}"
    return line


def _fix_unbound_local_error(line, indent, error_msg):
    """Fix UnboundLocalError by initializing variable."""
    pattern = r"local variable '(\w+)' referenced"
    match = re.search(pattern, error_msg)
    if match:
        var_name = match.group(1)
        return f"{indent}{var_name} = None  # Initialize\n{line}"
    return line


# EVERY PYTHON ERROR WITH SOLUTION HARD-CODED
ERROR_DATABASE = {
    'FileNotFoundError': {
        'description': 'File or directory does not exist',
        'patterns': [
            {
                'detect': r'open\s*\(',
                'fix': lambda line, indent, error_msg: wrap_in_try_except(line, 'FileNotFoundError', len(indent)),
                'multiline': True,
                'confidence': 0.85,
                'explanation': 'Added try/except for file operations (wraps entire block)'
            }
        ]
    },

    'KeyError': {
        'description': 'Dictionary key does not exist',
        'patterns': [
            {
                'detect': r"(\w+)\[(['\"])([^'\"]+)\2\]",
                'fix': lambda line, indent, error_msg: re.sub(
                    r"(\w+)\[(['\"])([^'\"]+)\2\]",
                    r"\1.get(\2\3\2, None)",
                    line,
                    count=1
                ),
                'multiline': False,
                'confidence': 0.9,
                'explanation': "Changed dict['key'] to dict.get('key', None)"
            }
        ]
    },

    'ZeroDivisionError': {
        'description': 'Division by zero',
        'patterns': [
            {
                'detect': r'(\S+)\s*/\s*(\S+)',
                'fix': lambda line, indent, error_msg: re.sub(
                    r'(\S+)\s*/\s*(\S+)',
                    r'(\1 / \2 if \2 != 0 else 0)',
                    line,
                    count=1
                ),
                'multiline': False,
                'confidence': 0.85,
                'explanation': 'Added check for division by zero'
            }
        ]
    },

    'IndexError': {
        'description': 'List index out of range',
        'patterns': [
            {
                'detect': r'\[(\d+)\]',
                'fix': lambda line, indent, error_msg: re.sub(
                    r"(\w+(?:\[['\"]?\w+['\"]?\])?)\[(\d+)\]",
                    r'(\1[\2] if len(\1) > \2 else None)',
                    line,
                    count=1
                ),
                'multiline': False,
                'confidence': 0.75,
                'explanation': 'Added length check before indexing'
            }
        ]
    },

    'JSONDecodeError': {
        'description': 'Invalid JSON format',
        'patterns': [
            {
                'detect': r'json\.loads?\s*\(',
                'fix': lambda line, indent, error_msg: wrap_in_try_except(line, 'json.JSONDecodeError', len(indent)),
                'multiline': True,
                'confidence': 0.8,
                'explanation': 'Added try/except for JSON parsing'
            }
        ]
    },

    'ValueError': {
        'description': 'Invalid value for operation',
        'patterns': [
            {
                'detect': r'\b(max|min)\s*\(([^)]+)\)',
                'fix': lambda line, indent, error_msg: wrap_in_try_except(line, 'ValueError', len(indent)),
                'multiline': True,
                'confidence': 0.75,
                'explanation': 'Added try/except for value operations'
            },
            {
                'detect': r'\bint\s*\(',
                'fix': lambda line, indent, error_msg: wrap_in_try_except(line, 'ValueError', len(indent)),
                'multiline': True,
                'confidence': 0.75,
                'explanation': 'Added try/except for int conversion'
            },
            {
                'detect': r'\bfloat\s*\(',
                'fix': lambda line, indent, error_msg: wrap_in_try_except(line, 'ValueError', len(indent)),
                'multiline': True,
                'confidence': 0.75,
                'explanation': 'Added try/except for float conversion'
            }
        ]
    },

    'AttributeError': {
        'description': 'Attribute does not exist on object',
        'patterns': [
            {
                'detect': r'(\w+)\.(\w+)',
                'fix': lambda line, indent, error_msg: re.sub(
                    r'(\w+)\.(\w+)',
                    r"getattr(\1, '\2', None)",
                    line,
                    count=1
                ),
                'multiline': False,
                'confidence': 0.7,
                'explanation': 'Changed obj.attr to getattr(obj, "attr", None)'
            }
        ]
    },

    'TypeError': {
        'description': 'Operation on incompatible types',
        'patterns': [
            {
                'detect': r'.*',
                'fix': lambda line, indent, error_msg: f"{indent}if value is not None:\n{indent}    {line.strip()}\n",
                'multiline': True,
                'confidence': 0.6,
                'explanation': 'Added None check'
            }
        ]
    },

    'NameError': {
        'description': 'Variable name not defined',
        'patterns': [
            {
                'detect': r'.*',
                'fix': lambda line, indent, error_msg: _fix_name_error(line, indent, error_msg),
                'multiline': False,
                'confidence': 0.8,
                'explanation': 'Initialize undefined variable'
            }
        ]
    },

    'ImportError': {
        'description': 'Module import failed',
        'patterns': [
            {
                'detect': r'import\s+',
                'fix': lambda line, indent, error_msg: _fix_import_error(line, indent, error_msg),
                'multiline': False,
                'confidence': 0.95,
                'explanation': 'Added pip install comment'
            }
        ]
    },

    'ModuleNotFoundError': {
        'description': 'Module not found (Python 3.6+)',
        'patterns': [
            {
                'detect': r'import\s+',
                'fix': lambda line, indent, error_msg: _fix_import_error(line, indent, error_msg),
                'multiline': False,
                'confidence': 0.95,
                'explanation': 'Added pip install comment'
            }
        ]
    },

    'UnboundLocalError': {
        'description': 'Local variable referenced before assignment',
        'patterns': [
            {
                'detect': r'.*',
                'fix': lambda line, indent, error_msg: _fix_unbound_local_error(line, indent, error_msg),
                'multiline': False,
                'confidence': 0.8,
                'explanation': 'Initialize variable before use'
            }
        ]
    }
}


class AutoFixer:
    """Simple database-driven auto-fixer. No AI, just pattern matching."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.fixes_applied = []

    def generate_fix(self, error: ParsedError, suggestion=None) -> Optional[CodeFix]:
        """Generate fix from error using ERROR_DATABASE."""
        if not error.primary_location:
            return None

        # Extract error type (handle module-qualified names)
        error_type = error.error_type
        if '.' in error_type:
            error_type = error_type.split('.')[-1]

        if error_type not in ERROR_DATABASE:
            return None

        # Try all stack locations
        locations_to_check = []
        if error.stack_trace:
            locations_to_check = error.stack_trace
        elif error.primary_location:
            locations_to_check = [error.primary_location]

        for location in locations_to_check:
            fix = self._try_fix_at_location(error, error_type, location)
            if fix:
                return fix

        return None

    def _try_fix_at_location(self, error: ParsedError, error_type: str, location) -> Optional[CodeFix]:
        """Try to generate fix at specific location."""
        file_path = location.file_path

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            if location.line_number > len(lines):
                return None

            line_idx = location.line_number - 1
            target_line = lines[line_idx]
            indent = get_indent(target_line)

            # Try each pattern for this error type
            for pattern in ERROR_DATABASE[error_type]['patterns']:
                if re.search(pattern['detect'], target_line):
                    fixed = pattern['fix'](target_line, indent, error.message)

                    if fixed and fixed != target_line:
                        return CodeFix(
                            file_path=file_path,
                            line_number=location.line_number,
                            original_code=target_line.rstrip(),
                            fixed_code=fixed.rstrip(),
                            explanation=pattern['explanation'],
                            confidence=pattern['confidence']
                        )

        except (FileNotFoundError, IOError):
            return None

        return None

    def apply_fix(self, fix: CodeFix) -> bool:
        """Apply a fix to the file."""
        if self.dry_run:
            return False

        try:
            file_path = Path(fix.file_path)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            line_idx = fix.line_number - 1

            # Check if this needs multi-line block wrapping
            target_line = lines[line_idx]
            needs_block_wrap = '\n' in fix.fixed_code and (
                re.search(r'\b(with|for|while)\b', target_line) or
                target_line.strip().endswith(':')
            )

            if needs_block_wrap:
                # Get the entire indented block
                block_lines, base_indent = get_indented_block(lines, line_idx)

                if block_lines:
                    # Determine error type from fixed_code
                    error_type = None
                    if 'FileNotFoundError' in fix.fixed_code:
                        error_type = 'FileNotFoundError'
                    elif 'JSONDecodeError' in fix.fixed_code:
                        error_type = 'json.JSONDecodeError'
                    elif 'ValueError' in fix.fixed_code:
                        error_type = 'ValueError'

                    if error_type:
                        fixed = wrap_block_in_try_except(block_lines, base_indent, error_type)
                        lines_to_replace = len(block_lines)
                        fixed_lines = fixed.split('\n')
                        new_lines = [line + '\n' for line in fixed_lines if line]
                        lines[line_idx:line_idx + lines_to_replace] = new_lines
                    else:
                        # Fallback to simple replacement
                        lines[line_idx] = fix.fixed_code + '\n'
                else:
                    lines[line_idx] = fix.fixed_code + '\n'
            else:
                # Single line replacement
                lines[line_idx] = fix.fixed_code + '\n'

            # Write back
            with open(file_path, 'w') as f:
                f.writelines(lines)

            self.fixes_applied.append(fix)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to apply fix: {e}")
            return False

    def generate_all_fixes(self, error: ParsedError, suggestions=None) -> list[CodeFix]:
        """Generate all possible fixes for an error."""
        fix = self.generate_fix(error, None)
        if fix:
            return [fix]
        return []
