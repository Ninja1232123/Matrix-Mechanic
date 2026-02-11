"""Line-by-line profiler implementation"""

import ast
import sys
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import linecache

try:
    from line_profiler import LineProfiler
    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False


class LineProfilerImpl:
    """Line-by-line profiler for detailed performance analysis"""

    def __init__(self):
        if not HAS_LINE_PROFILER:
            raise ImportError(
                "line_profiler not installed. "
                "Install with: pip install line_profiler"
            )
        self.profiler = None

    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile a function line-by-line"""
        profiler = LineProfiler()
        profiler.add_function(func)

        # Profile the function
        wrapped = profiler(func)
        result = wrapped(*args, **kwargs)

        # Get stats
        stats = self._parse_stats(profiler)

        return {
            'function': func.__name__,
            'return_value': result,
            'line_stats': stats,
            'hottest_line': self._find_hottest_line(stats),
        }

    def profile_functions(
        self,
        functions: List[Callable],
        entry_point: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile multiple functions"""
        profiler = LineProfiler()

        # Add all functions to profiler
        for func in functions:
            profiler.add_function(func)

        # Profile entry point
        wrapped = profiler(entry_point)
        result = wrapped(*args, **kwargs)

        # Get stats for all functions
        all_stats = {}
        for func in functions:
            stats = self._parse_stats(profiler, func)
            if stats:
                all_stats[func.__name__] = {
                    'line_stats': stats,
                    'hottest_line': self._find_hottest_line(stats),
                }

        return {
            'entry_point': entry_point.__name__,
            'return_value': result,
            'function_stats': all_stats,
        }

    def profile_script(self, script_path: Path) -> Dict[str, Any]:
        """Profile a script line-by-line"""
        profiler = LineProfiler()

        # Parse the script to find all functions
        with open(script_path, 'r') as f:
            tree = ast.parse(f.read(), str(script_path))

        functions = self._extract_functions(tree, script_path)

        # Read and compile script
        with open(script_path, 'r') as f:
            code = compile(f.read(), str(script_path), 'exec')

        # Add all functions to profiler
        globals_dict = {}
        exec(code, globals_dict)

        for func_name in functions:
            if func_name in globals_dict:
                func = globals_dict[func_name]
                if callable(func):
                    profiler.add_function(func)

        # Run the script with profiling
        profiler.enable()
        try:
            exec(code, globals_dict)
        except SystemExit:
            pass
        finally:
            profiler.disable()

        # Get stats for all functions
        all_stats = {}
        for func_name in functions:
            if func_name in globals_dict:
                func = globals_dict[func_name]
                stats = self._parse_stats(profiler, func)
                if stats:
                    all_stats[func_name] = {
                        'line_stats': stats,
                        'hottest_line': self._find_hottest_line(stats),
                    }

        return {
            'script': str(script_path),
            'function_stats': all_stats,
        }

    def _parse_stats(
        self,
        profiler: 'LineProfiler',
        func: Optional[Callable] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Parse line profiler stats"""
        if not profiler.code_map:
            return {}

        line_stats = {}

        for code_key, timing in profiler.code_map.items():
            # If func specified, only return stats for that function
            if func and code_key != func.__code__:
                continue

            filename = code_key.co_filename
            firstlineno = code_key.co_firstlineno

            # Get line timings
            for lineno, nhits, time in timing:
                if nhits == 0:
                    continue

                actual_lineno = firstlineno + lineno

                # Get source line
                source_line = linecache.getline(filename, actual_lineno).strip()

                line_stats[actual_lineno] = {
                    'line': actual_lineno,
                    'hits': nhits,
                    'time_us': time,  # Microseconds
                    'time_ms': time / 1000,  # Milliseconds
                    'time_per_hit_us': time / nhits if nhits > 0 else 0,
                    'source': source_line,
                    'filename': filename,
                }

        return line_stats

    def _find_hottest_line(
        self,
        line_stats: Dict[int, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find the line with the most time spent"""
        if not line_stats:
            return None

        hottest = max(
            line_stats.values(),
            key=lambda x: x['time_us']
        )
        return hottest

    def _extract_functions(self, tree: ast.AST, filename: Path) -> List[str]:
        """Extract function names from AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
        return functions

    def get_slow_lines(
        self,
        line_stats: Dict[int, Dict[str, Any]],
        threshold_ms: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Get lines that exceed time threshold"""
        return [
            stats for stats in line_stats.values()
            if stats['time_ms'] > threshold_ms
        ]

    def get_hot_lines(
        self,
        line_stats: Dict[int, Dict[str, Any]],
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top N lines by time"""
        sorted_lines = sorted(
            line_stats.values(),
            key=lambda x: x['time_us'],
            reverse=True
        )
        return sorted_lines[:top_n]

    def format_line_stats(
        self,
        line_stats: Dict[int, Dict[str, Any]]
    ) -> str:
        """Format line stats for display"""
        if not line_stats:
            return "No line stats available"

        lines = []
        lines.append(f"{'Line':<6} {'Hits':<10} {'Time (ms)':<12} {'Per Hit (μs)':<15} {'Source'}")
        lines.append("-" * 80)

        sorted_stats = sorted(
            line_stats.values(),
            key=lambda x: x['time_ms'],
            reverse=True
        )

        for stats in sorted_stats:
            lines.append(
                f"{stats['line']:<6} "
                f"{stats['hits']:<10} "
                f"{stats['time_ms']:<12.2f} "
                f"{stats['time_per_hit_us']:<15.2f} "
                f"{stats['source'][:50]}"
            )

        return "\n".join(lines)
