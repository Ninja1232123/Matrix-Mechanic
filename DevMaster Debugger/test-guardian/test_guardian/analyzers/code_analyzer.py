"""Code analyzer - analyzes Python code using AST"""

import ast
import inspect
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models import FunctionAnalysis


class CodeAnalyzer:
    """Analyzes Python code to understand structure for test generation"""

    def analyze_file(self, file_path: Path) -> List[FunctionAnalysis]:
        """Analyze all functions in a file"""
        with open(file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source, str(file_path))
        analyses = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis = self._analyze_function(node, file_path)
                analyses.append(analysis)

        return analyses

    def analyze_function(self, func: callable) -> FunctionAnalysis:
        """Analyze a Python function object"""
        source_file = inspect.getsourcefile(func)
        source_lines, line_number = inspect.getsourcelines(func)
        source = ''.join(source_lines)

        tree = ast.parse(source)
        func_node = tree.body[0]

        return self._analyze_function(func_node, Path(source_file), line_number)

    def _analyze_function(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        line_number: Optional[int] = None
    ) -> FunctionAnalysis:
        """Analyze a function AST node"""
        # Extract parameters
        parameters = self._extract_parameters(node)

        # Extract return type
        return_type = self._extract_return_type(node)

        # Find exceptions that can be raised
        raises = self._find_raised_exceptions(node)

        # Check if calls external dependencies
        calls_external = self._calls_external_deps(node)

        # Check if function is pure (no side effects)
        is_pure = self._is_pure_function(node)

        # Calculate complexity
        complexity = self._calculate_complexity(node)

        return FunctionAnalysis(
            name=node.name,
            file_path=str(file_path),
            line_number=line_number or node.lineno,
            parameters=parameters,
            return_type=return_type,
            raises=raises,
            calls_external=calls_external,
            is_pure=is_pure,
            complexity=complexity,
        )

    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters with type hints"""
        parameters = []

        for arg in node.args.args:
            param = {
                'name': arg.arg,
                'type_hint': None,
                'has_default': False,
                'default_value': None,
            }

            # Extract type annotation
            if arg.annotation:
                param['type_hint'] = ast.unparse(arg.annotation)

            parameters.append(param)

        # Add defaults
        defaults = node.args.defaults
        if defaults:
            # Defaults align with the last N parameters
            offset = len(parameters) - len(defaults)
            for i, default in enumerate(defaults):
                param_idx = offset + i
                if param_idx < len(parameters):
                    parameters[param_idx]['has_default'] = True
                    parameters[param_idx]['default_value'] = ast.unparse(default)

        return parameters

    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation"""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _find_raised_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """Find exceptions that can be raised"""
        exceptions = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if child.exc:
                    if isinstance(child.exc, ast.Call):
                        if isinstance(child.exc.func, ast.Name):
                            exceptions.add(child.exc.func.id)
                    elif isinstance(child.exc, ast.Name):
                        exceptions.add(child.exc.id)

        return list(exceptions)

    def _calls_external_deps(self, node: ast.FunctionDef) -> bool:
        """Check if function calls external dependencies"""
        external_indicators = [
            'request', 'get', 'post', 'put', 'delete',  # HTTP
            'query', 'execute', 'fetch', 'commit',  # Database
            'open', 'read', 'write',  # File I/O
            'socket', 'connect', 'send', 'recv',  # Network
        ]

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id.lower() in external_indicators:
                        return True
                elif isinstance(child.func, ast.Attribute):
                    if child.func.attr.lower() in external_indicators:
                        return True

        return False

    def _is_pure_function(self, node: ast.FunctionDef) -> bool:
        """Heuristic to determine if function is pure (no side effects)"""
        # Check for common side effects
        for child in ast.walk(node):
            # Global/nonlocal modifications
            if isinstance(child, (ast.Global, ast.Nonlocal)):
                return False

            # Print statements (side effect)
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == 'print':
                    return False

            # Attribute assignment (possible side effect)
            if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store):
                return False

            # File/network operations
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in ['open', 'write', 'print']:
                        return False

        # If has return and no obvious side effects, likely pure
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        return has_return

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def find_testable_functions(
        self,
        file_path: Path,
        min_complexity: int = 1
    ) -> List[FunctionAnalysis]:
        """Find functions worth testing"""
        all_functions = self.analyze_file(file_path)

        testable = []
        for func in all_functions:
            # Skip private functions (unless complex)
            if func.name.startswith('_') and func.complexity < 5:
                continue

            # Skip very simple functions
            if func.complexity < min_complexity:
                continue

            testable.append(func)

        return testable

    def extract_edge_case_candidates(
        self,
        func_analysis: FunctionAnalysis
    ) -> Dict[str, List[Any]]:
        """Identify edge case values for parameters"""
        edge_cases = {}

        for param in func_analysis.parameters:
            param_name = param['name']
            type_hint = param['type_hint']

            candidates = []

            if type_hint:
                # Generate edge cases based on type
                if 'int' in type_hint.lower():
                    candidates = [0, 1, -1, 999999, -999999]
                elif 'str' in type_hint.lower():
                    candidates = ['', 'a', 'test', ' ', '\n', 'x' * 1000]
                elif 'list' in type_hint.lower():
                    candidates = [[], [1], [1, 2, 3], list(range(100))]
                elif 'dict' in type_hint.lower():
                    candidates = [{}, {'key': 'value'}, {'a': 1, 'b': 2}]
                elif 'bool' in type_hint.lower():
                    candidates = [True, False]
                elif 'float' in type_hint.lower():
                    candidates = [0.0, 1.0, -1.0, 0.5, 999.999, -999.999]

            # Always include None as edge case
            candidates.append(None)

            edge_cases[param_name] = candidates

        return edge_cases
