"""Type analyzer - infers and validates types"""

from typing import Any, Optional, List


class TypeAnalyzer:
    """Analyzes and infers types for test generation"""

    def infer_type(self, value: Any) -> str:
        """Infer type from a value"""
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, list):
            return "List"
        elif isinstance(value, dict):
            return "Dict"
        elif isinstance(value, tuple):
            return "Tuple"
        elif isinstance(value, set):
            return "Set"
        else:
            return type(value).__name__

    def get_type_repr(self, value: Any) -> str:
        """Get string representation for type hint"""
        value_type = self.infer_type(value)

        if value_type == "List" and value:
            elem_type = self.infer_type(value[0])
            return f"List[{elem_type}]"
        elif value_type == "Dict" and value:
            key = list(value.keys())[0]
            val = list(value.values())[0]
            return f"Dict[{self.infer_type(key)}, {self.infer_type(val)}]"
        else:
            return value_type

    def is_compatible(self, value: Any, type_hint: str) -> bool:
        """Check if value is compatible with type hint"""
        if not type_hint:
            return True

        inferred = self.infer_type(value)
        type_lower = type_hint.lower()

        # Handle None/Optional
        if value is None:
            return 'optional' in type_lower or 'none' in type_lower

        # Simple type matching
        if inferred.lower() in type_lower:
            return True

        return False

    def generate_default_value(self, type_hint: Optional[str]) -> Any:
        """Generate a reasonable default value for a type"""
        if not type_hint:
            return None

        type_lower = type_hint.lower()

        if 'int' in type_lower:
            return 0
        elif 'float' in type_lower:
            return 0.0
        elif 'str' in type_lower:
            return ""
        elif 'bool' in type_lower:
            return False
        elif 'list' in type_lower:
            return []
        elif 'dict' in type_lower:
            return {}
        elif 'tuple' in type_lower:
            return ()
        elif 'set' in type_lower:
            return set()
        else:
            return None
