"""
Environment Variable Validator
"""

import os
import re
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse


class EnvironmentValidator:
    """Validate environment variables"""
    
    PLACEHOLDER_PATTERNS = [
        r'your[-_].*[-_]here',
        r'<.*>',
        r'PLEASE[-_]SET',
        r'CHANGE[-_]ME',
        r'example',
        r'localhost.*production',
    ]
    
    FORMAT_VALIDATORS = {
        'DATABASE_URL': lambda v: v.startswith(('postgresql://', 'mysql://', 'sqlite://')),
        'REDIS_URL': lambda v: v.startswith('redis://'),
        'PORT': lambda v: v.isdigit() and 0 < int(v) < 65536,
        'API_KEY': lambda v: len(v) > 10 and not any(p in v.lower() for p in ['example', 'test', 'demo']),
        'SECRET_KEY': lambda v: len(v) >= 32,
    }
    
    def __init__(self):
        self.env_example_path = Path('.env.example')
        self.env_path = Path('.env')
    
    def validate(self) -> List[Dict]:
        """Validate all environment variables"""
        issues = []
        
        # Parse required vars from .env.example
        required_vars = self._parse_env_example()
        
        if not required_vars:
            return issues
        
        for var_name, example_value in required_vars.items():
            # Check if variable is set
            if var_name not in os.environ:
                issues.append({
                    'type': 'missing_env_var',
                    'severity': 'critical',
                    'message': f'{var_name} not set',
                    'explanation': f'Required environment variable {var_name} is missing',
                    'recommendation': f'Set {var_name} in your environment or .env file',
                    'fix': lambda: self._prompt_for_var(var_name)
                })
                continue
            
            actual_value = os.environ[var_name]
            
            # Check for placeholder values
            if self._is_placeholder(actual_value):
                issues.append({
                    'type': 'placeholder_env_var',
                    'severity': 'critical',
                    'message': f'{var_name} uses placeholder value',
                    'explanation': 'Using example/placeholder values will cause runtime errors',
                    'recommendation': f'Replace with actual value for {var_name}',
                    'fix': lambda: self._prompt_for_var(var_name)
                })
                continue
            
            # Check if value matches example (probably not changed)
            if actual_value == example_value and self._looks_like_example(example_value):
                issues.append({
                    'type': 'unchanged_env_var',
                    'severity': 'high',
                    'message': f'{var_name} appears unchanged from example',
                    'explanation': 'Value matches .env.example, probably not customized',
                    'recommendation': f'Update {var_name} with actual production value'
                })
            
            # Validate format
            if not self._validate_format(var_name, actual_value):
                issues.append({
                    'type': 'invalid_env_format',
                    'severity': 'high',
                    'message': f'{var_name} has invalid format',
                    'explanation': f'The value for {var_name} does not match expected format',
                    'recommendation': 'Check format and update accordingly'
                })
            
            # Check for localhost in production-related vars
            if self._is_production_var(var_name) and 'localhost' in actual_value.lower():
                issues.append({
                    'type': 'localhost_in_production',
                    'severity': 'critical',
                    'message': f'{var_name} points to localhost',
                    'explanation': 'localhost is not accessible in production environments',
                    'recommendation': 'Use actual production host/service name'
                })
        
        return issues
    
    def _parse_env_example(self) -> Dict[str, str]:
        """Parse .env.example file"""
        if not self.env_example_path.exists():
            return {}
        
        vars_dict = {}
        with open(self.env_example_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    vars_dict[key.strip()] = value.strip()
        
        return vars_dict
    
    def _is_placeholder(self, value: str) -> bool:
        """Check if value is a placeholder"""
        value_lower = value.lower()
        for pattern in self.PLACEHOLDER_PATTERNS:
            if re.search(pattern, value_lower):
                return True
        return False
    
    def _looks_like_example(self, value: str) -> bool:
        """Check if value looks like an example"""
        example_indicators = ['example', 'test', 'demo', 'localhost', 'changeme']
        value_lower = value.lower()
        return any(indicator in value_lower for indicator in example_indicators)
    
    def _validate_format(self, var_name: str, value: str) -> bool:
        """Validate variable format"""
        # Check specific validators
        for pattern, validator in self.FORMAT_VALIDATORS.items():
            if pattern in var_name:
                return validator(value)
        
        return True  # No specific validation
    
    def _is_production_var(self, var_name: str) -> bool:
        """Check if variable is production-related"""
        prod_indicators = ['DATABASE', 'REDIS', 'API', 'HOST', 'URL']
        return any(indicator in var_name.upper() for indicator in prod_indicators)
    
    def _prompt_for_var(self, var_name: str):
        """Prompt user for variable value"""
        from rich.prompt import Prompt
        value = Prompt.ask(f"Enter value for {var_name}")
        
        # Update .env file
        if self.env_path.exists():
            with open(self.env_path, 'a') as f:
                f.write(f"\n{var_name}={value}\n")
        else:
            with open(self.env_path, 'w') as f:
                f.write(f"{var_name}={value}\n")
