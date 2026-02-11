"""Pattern detector - identifies inefficient code patterns"""

import ast
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models import Optimization, OptimizationType


class PatternDetector:
    """Detects inefficient code patterns that can be optimized"""

    def __init__(self):
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> List[Dict[str, Any]]:
        """Initialize detection patterns"""
        return [
            {
                'name': 'list_append_in_loop',
                'type': OptimizationType.LOOP_COMPREHENSION,
                'description': 'List append in loop can be replaced with list comprehension',
                'estimated_speedup': 1.5,
                'confidence': 0.9,
            },
            {
                'name': 'repeated_dict_lookup',
                'type': OptimizationType.CACHING,
                'description': 'Repeated dictionary lookup can be cached',
                'estimated_speedup': 2.0,
                'confidence': 0.8,
            },
            {
                'name': 'function_called_in_loop',
                'type': OptimizationType.CACHING,
                'description': 'Function with constant args called repeatedly, consider caching',
                'estimated_speedup': 3.0,
                'confidence': 0.7,
            },
            {
                'name': 'nested_loops',
                'type': OptimizationType.ALGORITHM,
                'description': 'Nested loops detected, O(n²) complexity, consider optimization',
                'estimated_speedup': 10.0,
                'confidence': 0.6,
            },
            {
                'name': 'string_concatenation_in_loop',
                'type': OptimizationType.MEMORY_REUSE,
                'description': 'String concatenation in loop, use join() instead',
                'estimated_speedup': 5.0,
                'confidence': 0.9,
            },
            {
                'name': 'inefficient_filtering',
                'type': OptimizationType.LOOP_COMPREHENSION,
                'description': 'Multiple loops for filtering, can be combined',
                'estimated_speedup': 2.0,
                'confidence': 0.8,
            },
            {
                'name': 'missing_lru_cache',
                'type': OptimizationType.CACHING,
                'description': 'Pure function without caching, consider @lru_cache',
                'estimated_speedup': 10.0,
                'confidence': 0.7,
            },
            {
                'name': 'sync_io_in_loop',
                'type': OptimizationType.ASYNC_AWAIT,
                'description': 'Synchronous I/O in loop, consider async/await',
                'estimated_speedup': 20.0,
                'confidence': 0.6,
            },
            {
                'name': 'global_variable_lookup',
                'type': OptimizationType.CACHING,
                'description': 'Global variable lookup in tight loop, cache locally',
                'estimated_speedup': 1.3,
                'confidence': 0.9,
            },
            {
                'name': 'unnecessary_list_copy',
                'type': OptimizationType.MEMORY_REUSE,
                'description': 'Unnecessary list copy, use iterator or view',
                'estimated_speedup': 2.0,
                'confidence': 0.8,
            },
        ]

    def analyze_file(self, file_path: Path) -> List[Optimization]:
        """Analyze a Python file for inefficient patterns"""
        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source, str(file_path))
            optimizations = []

            # Detect various patterns
            optimizations.extend(self._detect_list_append_in_loop(tree, file_path, source))
            optimizations.extend(self._detect_nested_loops(tree, file_path, source))
            optimizations.extend(self._detect_string_concat_in_loop(tree, file_path, source))
            optimizations.extend(self._detect_missing_lru_cache(tree, file_path, source))
            optimizations.extend(self._detect_repeated_dict_lookup(tree, file_path, source))
            optimizations.extend(self._detect_function_in_loop(tree, file_path, source))
            optimizations.extend(self._detect_sync_io_in_loop(tree, file_path, source))

            return optimizations

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []

    def _detect_list_append_in_loop(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[Optimization]:
        """Detect list.append() in loops that can become comprehensions"""
        optimizations = []
        source_lines = source.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check if loop body contains list.append()
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            if child.func.attr == 'append':
                                # Found list.append in loop
                                lineno = node.lineno

                                # Extract original code
                                original = self._get_node_source(node, source_lines)

                                # Generate optimized version (simplified)
                                optimized = self._generate_list_comprehension(node, source_lines)

                                if optimized:
                                    optimizations.append(Optimization(
                                        type=OptimizationType.LOOP_COMPREHENSION,
                                        file_path=str(file_path),
                                        line_number=lineno,
                                        original_code=original,
                                        optimized_code=optimized,
                                        description="Replace loop with list comprehension",
                                        estimated_speedup=1.5,
                                        confidence=0.8,
                                    ))
                                break

        return optimizations

    def _detect_nested_loops(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[Optimization]:
        """Detect nested loops (O(n²) complexity)"""
        optimizations = []
        source_lines = source.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check for nested loops
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While)):
                        lineno = node.lineno
                        original = self._get_node_source(node, source_lines)

                        optimizations.append(Optimization(
                            type=OptimizationType.ALGORITHM,
                            file_path=str(file_path),
                            line_number=lineno,
                            original_code=original,
                            optimized_code="# Consider using a hash map or set for O(n) lookup",
                            description="Nested loop detected (O(n²)), consider algorithmic optimization",
                            estimated_speedup=10.0,
                            confidence=0.6,
                        ))
                        break

        return optimizations

    def _detect_string_concat_in_loop(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[Optimization]:
        """Detect string concatenation in loops"""
        optimizations = []
        source_lines = source.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check for string concatenation
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign):
                        if isinstance(child.op, ast.Add):
                            # Found += in loop (could be string concat)
                            lineno = node.lineno
                            original = self._get_node_source(node, source_lines)

                            optimizations.append(Optimization(
                                type=OptimizationType.MEMORY_REUSE,
                                file_path=str(file_path),
                                line_number=lineno,
                                original_code=original,
                                optimized_code="# Use ''.join(list) instead of += in loop",
                                description="String concatenation in loop, use join() instead",
                                estimated_speedup=5.0,
                                confidence=0.7,
                            ))
                            break

        return optimizations

    def _detect_missing_lru_cache(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[Optimization]:
        """Detect pure functions that could benefit from @lru_cache"""
        optimizations = []
        source_lines = source.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has @lru_cache
                has_lru_cache = any(
                    isinstance(dec, ast.Name) and dec.id == 'lru_cache'
                    or isinstance(dec, ast.Attribute) and dec.attr == 'lru_cache'
                    for dec in node.decorator_list
                )

                if not has_lru_cache:
                    # Check if function is pure (heuristic)
                    is_pure = self._is_potentially_pure(node)

                    if is_pure:
                        lineno = node.lineno
                        original = self._get_node_source(node, source_lines)

                        optimized = f"@lru_cache(maxsize=128)\n{original}"

                        optimizations.append(Optimization(
                            type=OptimizationType.CACHING,
                            file_path=str(file_path),
                            line_number=lineno,
                            original_code=original,
                            optimized_code=optimized,
                            description=f"Add @lru_cache to pure function '{node.name}'",
                            estimated_speedup=10.0,
                            confidence=0.6,
                        ))

        return optimizations

    def _detect_repeated_dict_lookup(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[Optimization]:
        """Detect repeated dictionary lookups"""
        optimizations = []
        source_lines = source.split('\n')

        # This is a simplified detection
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.FunctionDef)):
                subscripts = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Subscript):
                        subscripts.append(child)

                # If same subscript appears multiple times, suggest caching
                if len(subscripts) > 2:
                    lineno = getattr(node, 'lineno', 0)
                    original = self._get_node_source(node, source_lines)

                    optimizations.append(Optimization(
                        type=OptimizationType.CACHING,
                        file_path=str(file_path),
                        line_number=lineno,
                        original_code=original,
                        optimized_code="# Cache dictionary lookups in local variable",
                        description="Repeated dictionary lookups detected, consider caching",
                        estimated_speedup=1.5,
                        confidence=0.5,
                    ))

        return optimizations

    def _detect_function_in_loop(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[Optimization]:
        """Detect expensive function calls in loops"""
        optimizations = []
        source_lines = source.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                calls_in_loop = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        calls_in_loop.append(child)

                if calls_in_loop:
                    lineno = node.lineno
                    original = self._get_node_source(node, source_lines)

                    optimizations.append(Optimization(
                        type=OptimizationType.CACHING,
                        file_path=str(file_path),
                        line_number=lineno,
                        original_code=original,
                        optimized_code="# Move invariant function calls outside loop",
                        description="Function calls in loop, consider hoisting if results don't change",
                        estimated_speedup=2.0,
                        confidence=0.5,
                    ))

        return optimizations

    def _detect_sync_io_in_loop(
        self,
        tree: ast.AST,
        file_path: Path,
        source: str
    ) -> List[Optimization]:
        """Detect synchronous I/O operations in loops"""
        optimizations = []
        source_lines = source.split('\n')

        io_functions = ['open', 'read', 'write', 'request', 'get', 'post', 'query']

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func_name = None
                        if isinstance(child.func, ast.Name):
                            func_name = child.func.id
                        elif isinstance(child.func, ast.Attribute):
                            func_name = child.func.attr

                        if func_name and func_name.lower() in io_functions:
                            lineno = node.lineno
                            original = self._get_node_source(node, source_lines)

                            optimizations.append(Optimization(
                                type=OptimizationType.ASYNC_AWAIT,
                                file_path=str(file_path),
                                line_number=lineno,
                                original_code=original,
                                optimized_code="# Consider async/await or parallel processing",
                                description=f"Synchronous I/O ({func_name}) in loop, consider async",
                                estimated_speedup=20.0,
                                confidence=0.6,
                            ))
                            break

        return optimizations

    def _is_potentially_pure(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is potentially pure (simple heuristic)"""
        # Check if function modifies global state or has side effects
        for node in ast.walk(func_node):
            # Has global/nonlocal statements
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return False
            # Calls print (side effect)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    return False
            # Has attribute assignment (possible side effect)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                return False

        # If function has return statement and no obvious side effects, consider it pure
        has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
        return has_return

    def _get_node_source(self, node: ast.AST, source_lines: List[str]) -> str:
        """Extract source code for an AST node"""
        if not hasattr(node, 'lineno'):
            return ""

        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1

        return '\n'.join(source_lines[start:end])

    def _generate_list_comprehension(
        self,
        loop_node: ast.AST,
        source_lines: List[str]
    ) -> Optional[str]:
        """Generate list comprehension from loop (simplified)"""
        # This is a simplified version - real implementation would be more complex
        return "# [item for item in iterable if condition]"

    def get_pattern_stats(self, optimizations: List[Optimization]) -> Dict[str, Any]:
        """Get statistics about detected patterns"""
        by_type = {}
        for opt in optimizations:
            type_name = opt.type.value
            if type_name not in by_type:
                by_type[type_name] = 0
            by_type[type_name] += 1

        total_estimated_speedup = sum(opt.estimated_speedup for opt in optimizations)
        avg_confidence = sum(opt.confidence for opt in optimizations) / len(optimizations) if optimizations else 0

        return {
            'total_patterns': len(optimizations),
            'by_type': by_type,
            'total_estimated_speedup': total_estimated_speedup,
            'avg_confidence': avg_confidence,
        }
