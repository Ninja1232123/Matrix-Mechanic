"""Complexity analyzer - analyzes algorithmic complexity"""

import ast
from typing import Dict, Any, List
from pathlib import Path


class ComplexityAnalyzer:
    """Analyzes algorithmic complexity of functions"""

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity of all functions in a file"""
        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source, str(file_path))
            results = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._analyze_function(node)
                    results[node.name] = complexity

            return results

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {}

    def _analyze_function(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze complexity of a single function"""
        return {
            'name': func_node.name,
            'cyclomatic_complexity': self._cyclomatic_complexity(func_node),
            'nesting_depth': self._max_nesting_depth(func_node),
            'loop_count': self._count_loops(func_node),
            'estimated_time_complexity': self._estimate_time_complexity(func_node),
            'lines_of_code': self._count_lines(func_node),
        }

    def _cyclomatic_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity

        for node in ast.walk(func_node):
            # Decision points add to complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # And/Or operators
                complexity += len(node.values) - 1

        return complexity

    def _max_nesting_depth(self, func_node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth"""
        def depth(node, current_depth=0):
            max_d = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    max_d = max(max_d, depth(child, current_depth + 1))
                else:
                    max_d = max(max_d, depth(child, current_depth))
            return max_d

        return depth(func_node)

    def _count_loops(self, func_node: ast.FunctionDef) -> int:
        """Count number of loops"""
        count = 0
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While)):
                count += 1
        return count

    def _estimate_time_complexity(self, func_node: ast.FunctionDef) -> str:
        """Estimate time complexity (rough heuristic)"""
        loop_count = self._count_loops(func_node)
        max_nesting = self._max_nesting_depth(func_node)

        # Detect nested loops
        nested_loop_depth = 0
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While)):
                depth = self._loop_nesting_depth(node)
                nested_loop_depth = max(nested_loop_depth, depth)

        if nested_loop_depth >= 3:
            return "O(n³) or worse"
        elif nested_loop_depth == 2:
            return "O(n²)"
        elif loop_count > 0:
            return "O(n)"
        else:
            return "O(1)"

    def _loop_nesting_depth(self, loop_node: ast.AST) -> int:
        """Calculate nesting depth for a specific loop"""
        depth = 1
        for child in ast.walk(loop_node):
            if child != loop_node and isinstance(child, (ast.For, ast.While)):
                depth = max(depth, self._loop_nesting_depth(child) + 1)
        return depth

    def _count_lines(self, func_node: ast.FunctionDef) -> int:
        """Count lines of code in function"""
        if hasattr(func_node, 'end_lineno'):
            return func_node.end_lineno - func_node.lineno + 1
        return 0

    def get_complex_functions(
        self,
        analysis: Dict[str, Any],
        complexity_threshold: int = 10
    ) -> List[Dict[str, Any]]:
        """Get functions that exceed complexity threshold"""
        complex_funcs = []

        for func_name, metrics in analysis.items():
            if metrics['cyclomatic_complexity'] >= complexity_threshold:
                complex_funcs.append(metrics)

        # Sort by complexity (descending)
        complex_funcs.sort(
            key=lambda f: f['cyclomatic_complexity'],
            reverse=True
        )

        return complex_funcs

    def get_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary statistics"""
        if not analysis:
            return {
                'total_functions': 0,
                'avg_complexity': 0,
                'max_complexity': 0,
                'total_loops': 0,
            }

        complexities = [m['cyclomatic_complexity'] for m in analysis.values()]
        loops = [m['loop_count'] for m in analysis.values()]

        return {
            'total_functions': len(analysis),
            'avg_complexity': sum(complexities) / len(complexities),
            'max_complexity': max(complexities),
            'total_loops': sum(loops),
            'functions_by_complexity': {
                'O(1)': len([m for m in analysis.values() if m['estimated_time_complexity'] == 'O(1)']),
                'O(n)': len([m for m in analysis.values() if m['estimated_time_complexity'] == 'O(n)']),
                'O(n²)': len([m for m in analysis.values() if m['estimated_time_complexity'] == 'O(n²)']),
                'O(n³)+': len([m for m in analysis.values() if 'worse' in m['estimated_time_complexity']]),
            }
        }
