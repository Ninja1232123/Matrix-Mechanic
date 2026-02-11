"""
Type Pattern Database - Store common type error patterns and fixes
"""

import json
from pathlib import Path
from typing import Dict, Optional, List


class TypePatternDatabase:
    """Database of type error patterns and their fixes"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.fix_history: List[Dict] = []
    
    def _load_patterns(self) -> Dict:
        """Load type patterns from database"""
        return {
            'missing_return_type': {
                'detect': r"error: Missing return type annotation",
                'category': 'missing_type_hint',
                'confidence': 0.95,
                'description': 'Function needs return type annotation',
                'example': 'def func() -> ReturnType:'
            },
            
            'missing_param_type': {
                'detect': r"error: Missing type annotation for parameter '(\w+)'",
                'category': 'missing_type_hint',
                'confidence': 0.92,
                'description': 'Parameter needs type annotation',
                'example': 'def func(param: ParamType):'
            },
            
            'optional_none_check': {
                'detect': r"error: Item \"None\" of \"Optional\[.+\]\" has no attribute",
                'category': 'optional_none',
                'confidence': 0.98,
                'description': 'Need None check before accessing Optional value',
                'example': 'if value is not None: value.attr'
            },
            
            'list_need_type': {
                'detect': r"error: Need type annotation for '(\w+)' \(hint: \"(\w+)\[<type>\]\"\)",
                'category': 'collection_type',
                'confidence': 0.94,
                'description': 'Collection needs type parameter',
                'example': 'items: List[ItemType] = []'
            },
            
            'dict_need_type': {
                'detect': r"error: Need type annotation for '(\w+)' \(hint: \"Dict\[<type>, <type>\]\"\)",
                'category': 'collection_type',
                'confidence': 0.94,
                'description': 'Dict needs key and value types',
                'example': 'data: Dict[str, Any] = {}'
            },
            
            'any_type': {
                'detect': r"has type \"Any\"",
                'category': 'any_type',
                'confidence': 0.88,
                'description': 'Value has Any type, should be more specific',
                'example': 'Use specific type instead of Any'
            },
            
            'incompatible_type': {
                'detect': r"error: Incompatible types in assignment",
                'category': 'type_mismatch',
                'confidence': 0.85,
                'description': 'Types don\'t match in assignment',
                'example': 'Ensure assigned value matches variable type'
            },
            
            'generic_missing_param': {
                'detect': r"error: Missing type parameters for generic type",
                'category': 'generic_type',
                'confidence': 0.93,
                'description': 'Generic type needs type parameters',
                'example': 'def func(items: List[T]) -> T:'
            },
        }
    
    def suggest_fix(self, error: Dict) -> Optional[Dict]:
        """
        Suggest a fix for an error
        
        Args:
            error: Parsed error dictionary
            
        Returns:
            Fix suggestion dictionary or None
        """
        category = error['category']
        
        # Find matching pattern
        for pattern_name, pattern in self.patterns.items():
            if pattern['category'] == category:
                return {
                    'pattern': pattern_name,
                    'description': pattern['description'],
                    'example': pattern['example'],
                    'confidence': pattern['confidence']
                }
        
        return None
    
    def get_fix(self, error: Dict) -> Optional[Dict]:
        """
        Get a concrete fix for an error
        
        Args:
            error: Parsed error dictionary
            
        Returns:
            Fix dictionary with old/new code or None
        """
        category = error['category']
        context = error.get('context', {})
        
        if category == 'missing_type_hint':
            # Suggest adding type hint
            code_line = context.get('code_line', '')
            
            if 'def ' in code_line:
                # Function missing return type
                return {
                    'file': error['file'],
                    'line': error['line'],
                    'old': code_line.rstrip(':'),
                    'new': code_line.rstrip(':') + ' -> ReturnType'
                }
        
        elif category == 'optional_none':
            # Suggest None check
            code_line = context.get('code_line', '')
            var_name = context.get('name')
            
            if var_name and '.' in code_line:
                return {
                    'file': error['file'],
                    'line': error['line'],
                    'old': code_line,
                    'new': f'if {var_name} is not None: {code_line}'
                }
        
        elif category == 'collection_type':
            # Suggest collection type annotation
            code_line = context.get('code_line', '')
            var_name = context.get('name')
            
            if var_name and '=' in code_line:
                if '[' in code_line:
                    # List
                    return {
                        'file': error['file'],
                        'line': error['line'],
                        'old': f'{var_name} = []',
                        'new': f'{var_name}: List[Any] = []'
                    }
                elif '{' in code_line:
                    # Dict
                    return {
                        'file': error['file'],
                        'line': error['line'],
                        'old': f'{var_name} = {{}}',
                        'new': f'{var_name}: Dict[str, Any] = {{}}'
                    }
        
        return None
    
    def record_fix(self, error: Dict, fix: Dict, success: bool):
        """Record a fix attempt for learning"""
        self.fix_history.append({
            'error': error,
            'fix': fix,
            'success': success
        })
    
    def save_history(self, path: Path):
        """Save fix history to file"""
        with open(path, 'w') as f:
            for entry in self.fix_history:
                f.write(json.dumps(entry) + '\n')
    
    def get_statistics(self) -> Dict:
        """Get fix statistics"""
        if not self.fix_history:
            return {}
        
        total = len(self.fix_history)
        successful = sum(1 for e in self.fix_history if e['success'])
        
        # Count by category
        by_category = {}
        for entry in self.fix_history:
            category = entry['error']['category']
            if category not in by_category:
                by_category[category] = {'total': 0, 'success': 0}
            
            by_category[category]['total'] += 1
            if entry['success']:
                by_category[category]['success'] += 1
        
        return {
            'total_fixes': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0,
            'by_category': by_category
        }


class TypePatternLearner:
    """Learn new type patterns from successful fixes"""
    
    def __init__(self, database: TypePatternDatabase):
        self.db = database
    
    def learn_from_history(self):
        """Analyze fix history to learn new patterns"""
        # Group successful fixes by error message pattern
        patterns = {}
        
        for entry in self.db.fix_history:
            if not entry['success']:
                continue
            
            error = entry['error']
            message = error['message']
            
            # Extract pattern from message
            pattern_key = self._extract_pattern(message)
            
            if pattern_key not in patterns:
                patterns[pattern_key] = []
            
            patterns[pattern_key].append(entry)
        
        # Find patterns that occur frequently
        for pattern_key, entries in patterns.items():
            if len(entries) >= 5:  # At least 5 occurrences
                # This is a learnable pattern
                confidence = sum(1 for e in entries if e['success']) / len(entries)
                
                if confidence >= 0.9:  # 90% success rate
                    # Add to database
                    print(f"Learned new pattern: {pattern_key} (confidence: {confidence:.2f})")
    
    def _extract_pattern(self, message: str) -> str:
        """Extract a generalized pattern from error message"""
        # Replace specific names with placeholders
        import re
        
        # Replace quoted names
        pattern = re.sub(r"'[^']*'", "'<name>'", message)
        
        # Replace numbers
        pattern = re.sub(r'\d+', '<num>', pattern)
        
        return pattern
