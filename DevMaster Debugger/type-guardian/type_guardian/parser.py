"""
Parser - Parse mypy output into structured error objects
"""

import re
from typing import List, Dict, Optional
from pathlib import Path


class MypyParser:
    """Parse mypy error messages into structured format"""
    
    # Error pattern: file.py:line:col: error: message [code]
    ERROR_PATTERN = re.compile(
        r'^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): '
        r'(?P<severity>\w+): (?P<message>.+?)(?:\s+\[(?P<code>[\w-]+)\])?$'
    )
    
    def __init__(self):
        self.error_categories = {
            'missing return type': 'missing_type_hint',
            'missing type annotation': 'missing_type_hint',
            'need type annotation': 'missing_type_hint',
            'item "none" of "optional': 'optional_none',
            'optional[': 'optional_none',
            'argument .* has incompatible type': 'type_mismatch',
            'incompatible return value type': 'return_type_mismatch',
            'has type "any"': 'any_type',
            'need type parameter': 'generic_type',
            'list[': 'collection_type',
            'dict[': 'collection_type',
            'set[': 'collection_type',
        }
    
    def parse_errors(self, error_lines: List[str]) -> List[Dict]:
        """
        Parse mypy error output into structured format
        
        Args:
            error_lines: Raw error lines from mypy
            
        Returns:
            List of parsed error dictionaries
        """
        parsed_errors = []
        
        for line in error_lines:
            error = self._parse_error_line(line)
            if error:
                parsed_errors.append(error)
        
        return parsed_errors
    
    def _parse_error_line(self, line: str) -> Optional[Dict]:
        """Parse a single error line"""
        match = self.ERROR_PATTERN.match(line)
        
        if not match:
            return None
        
        error = {
            'file': match.group('file'),
            'line': int(match.group('line')),
            'col': int(match.group('col')),
            'severity': match.group('severity'),
            'message': match.group('message'),
            'code': match.group('code') or 'no-code',
        }
        
        # Categorize error
        error['category'] = self._categorize_error(error['message'])
        
        # Extract additional context
        error['context'] = self._extract_context(error)
        
        return error
    
    def _categorize_error(self, message: str) -> str:
        """Categorize error based on message"""
        message_lower = message.lower()
        
        for pattern, category in self.error_categories.items():
            if pattern in message_lower:
                return category
        
        return 'unknown'
    
    def _extract_context(self, error: Dict) -> Dict:
        """Extract additional context from error message"""
        context = {}
        
        message = error['message']
        
        # Extract variable/function names
        # Pattern: "variable/function 'name'"
        name_match = re.search(r"['\"](\w+)['\"]", message)
        if name_match:
            context['name'] = name_match.group(1)
        
        # Extract type information
        # Pattern: has type "Type"
        type_match = re.search(r'has type ["\'](.+?)["\']', message)
        if type_match:
            context['current_type'] = type_match.group(1)
        
        # Extract expected type
        # Pattern: expected "Type"
        expected_match = re.search(r'expected ["\'](.+?)["\']', message)
        if expected_match:
            context['expected_type'] = expected_match.group(1)
        
        # Get code context from file
        try:
            file_path = Path(error['file'])
            if file_path.exists():
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    line_idx = error['line'] - 1
                    
                    if 0 <= line_idx < len(lines):
                        context['code_line'] = lines[line_idx].rstrip()
                        
                        # Get surrounding context (±2 lines)
                        start = max(0, line_idx - 2)
                        end = min(len(lines), line_idx + 3)
                        context['code_context'] = ''.join(lines[start:end])
        except Exception:
            pass
        
        return context
    
    def group_by_file(self, errors: List[Dict]) -> Dict[str, List[Dict]]:
        """Group errors by file"""
        grouped = {}
        
        for error in errors:
            file = error['file']
            if file not in grouped:
                grouped[file] = []
            grouped[file].append(error)
        
        return grouped
    
    def group_by_category(self, errors: List[Dict]) -> Dict[str, List[Dict]]:
        """Group errors by category"""
        grouped = {}
        
        for error in errors:
            category = error['category']
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(error)
        
        return grouped
    
    def filter_fixable(self, errors: List[Dict]) -> List[Dict]:
        """Filter to only fixable errors"""
        fixable_categories = {
            'missing_type_hint',
            'optional_none',
            'collection_type',
            'generic_type',
        }
        
        return [e for e in errors if e['category'] in fixable_categories]
    
    def format_error(self, error: Dict, show_context: bool = True) -> str:
        """Format error for display"""
        lines = [
            f"{error['file']}:{error['line']}:{error['col']}",
            f"  {error['severity']}: {error['message']}",
        ]
        
        if error.get('code'):
            lines[-1] += f" [{error['code']}]"
        
        if show_context and error['context'].get('code_line'):
            lines.append(f"  > {error['context']['code_line']}")
        
        return '\n'.join(lines)
