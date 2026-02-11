"""Dependency analyzer - identifies external dependencies to mock"""

import ast
from typing import List, Set, Dict, Any
from pathlib import Path


class DependencyAnalyzer:
    """Analyzes external dependencies that need mocking"""

    def analyze_function_dependencies(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze what external calls a function makes"""
        return {
            'external_calls': self._find_external_calls(node),
            'imported_modules': self._find_imported_modules(node),
            'file_operations': self._find_file_operations(node),
            'network_calls': self._find_network_calls(node),
            'database_calls': self._find_database_calls(node),
        }

    def _find_external_calls(self, node: ast.FunctionDef) -> List[str]:
        """Find calls to external functions/methods"""
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # module.function or object.method
                    if isinstance(child.func.value, ast.Name):
                        calls.append(f"{child.func.value.id}.{child.func.attr}")

        return calls

    def _find_imported_modules(self, node: ast.AST) -> Set[str]:
        """Find imported modules"""
        modules = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                for alias in child.names:
                    modules.add(alias.name)
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    modules.add(child.module)

        return modules

    def _find_file_operations(self, node: ast.FunctionDef) -> List[str]:
        """Find file I/O operations"""
        file_ops = []
        file_functions = ['open', 'read', 'write', 'close']

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in file_functions:
                        file_ops.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if child.func.attr in file_functions:
                        file_ops.append(child.func.attr)

        return file_ops

    def _find_network_calls(self, node: ast.FunctionDef) -> List[str]:
        """Find network/HTTP operations"""
        network_ops = []
        network_keywords = ['request', 'get', 'post', 'put', 'delete', 'fetch', 'http']

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if any(kw in child.func.attr.lower() for kw in network_keywords):
                        network_ops.append(child.func.attr)

        return network_ops

    def _find_database_calls(self, node: ast.FunctionDef) -> List[str]:
        """Find database operations"""
        db_ops = []
        db_keywords = ['query', 'execute', 'fetch', 'commit', 'rollback']

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    if any(kw in child.func.attr.lower() for kw in db_keywords):
                        db_ops.append(child.func.attr)

        return db_ops

    def needs_mocking(self, func_analysis: Any) -> bool:
        """Determine if a function needs mocking"""
        return func_analysis.calls_external or len(func_analysis.raises) > 0
