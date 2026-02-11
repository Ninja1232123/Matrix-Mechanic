"""cProfile-based function profiler"""

import cProfile
import pstats
import io
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import sys

from ..models import ProfileResult


class CProfileProfiler:
    """Function-level profiler using cProfile"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(".speed_guardian/profiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> ProfileResult:
        """Profile a single function call"""
        profiler = cProfile.Profile()

        # Profile the function
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Parse stats
        stats = self._parse_stats(profiler)

        # Calculate total time and calls
        total_time = sum(s['cumtime'] for s in stats.values())
        total_calls = sum(s['ncalls'] for s in stats.values())

        return ProfileResult(
            target=func.__name__,
            total_time_ms=total_time * 1000,
            total_calls=total_calls,
            timestamp=datetime.now(),
            function_stats=stats,
        )

    def profile_script(self, script_path: Path) -> ProfileResult:
        """Profile an entire Python script"""
        profiler = cProfile.Profile()

        # Read and compile script
        with open(script_path, 'r') as f:
            code = compile(f.read(), str(script_path), 'exec')

        # Profile the script
        globals_dict = {
            '__name__': '__main__',
            '__file__': str(script_path),
        }

        profiler.enable()
        try:
            exec(code, globals_dict)
        except SystemExit:
            pass  # Allow scripts that call sys.exit()
        finally:
            profiler.disable()

        # Parse stats
        stats = self._parse_stats(profiler)

        # Calculate total time and calls
        total_time = sum(s['cumtime'] for s in stats.values())
        total_calls = sum(s['ncalls'] for s in stats.values())

        return ProfileResult(
            target=str(script_path),
            total_time_ms=total_time * 1000,
            total_calls=total_calls,
            timestamp=datetime.now(),
            function_stats=stats,
        )

    def profile_module(self, module_name: str) -> ProfileResult:
        """Profile a Python module"""
        profiler = cProfile.Profile()

        # Import and run module
        profiler.enable()
        try:
            __import__(module_name)
        finally:
            profiler.disable()

        # Parse stats
        stats = self._parse_stats(profiler)

        # Calculate total time and calls
        total_time = sum(s['cumtime'] for s in stats.values())
        total_calls = sum(s['ncalls'] for s in stats.values())

        return ProfileResult(
            target=module_name,
            total_time_ms=total_time * 1000,
            total_calls=total_calls,
            timestamp=datetime.now(),
            function_stats=stats,
        )

    def _parse_stats(self, profiler: cProfile.Profile) -> Dict[str, Any]:
        """Parse cProfile stats into a dictionary"""
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')

        result = {}
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            # func is (filename, line, function_name)
            filename, line, func_name = func
            key = f"{filename}:{line}:{func_name}"

            result[key] = {
                'filename': filename,
                'line': line,
                'function': func_name,
                'ncalls': nc,
                'tottime': tt,  # Total time in this function (excluding subcalls)
                'cumtime': ct,  # Cumulative time (including subcalls)
                'percall_tot': tt / nc if nc > 0 else 0,
                'percall_cum': ct / nc if nc > 0 else 0,
                'callers': [
                    f"{cf[0]}:{cf[1]}:{cf[2]}"
                    for cf in callers.keys()
                ],
            }

        return result

    def save_profile(self, profiler: cProfile.Profile, name: str) -> Path:
        """Save profile data to file"""
        output_path = self.output_dir / f"{name}.prof"
        profiler.dump_stats(str(output_path))
        return output_path

    def load_profile(self, path: Path) -> pstats.Stats:
        """Load profile data from file"""
        return pstats.Stats(str(path))

    def compare_profiles(
        self,
        old_profile: ProfileResult,
        new_profile: ProfileResult
    ) -> Dict[str, Dict[str, float]]:
        """Compare two profile results"""
        comparison = {}

        # Find common functions
        old_funcs = set(old_profile.function_stats.keys())
        new_funcs = set(new_profile.function_stats.keys())
        common_funcs = old_funcs & new_funcs

        for func in common_funcs:
            old_stats = old_profile.function_stats[func]
            new_stats = new_profile.function_stats[func]

            old_time = old_stats['cumtime']
            new_time = new_stats['cumtime']

            if old_time > 0:
                change_factor = new_time / old_time
                change_percent = ((new_time - old_time) / old_time) * 100

                comparison[func] = {
                    'old_time_ms': old_time * 1000,
                    'new_time_ms': new_time * 1000,
                    'change_factor': change_factor,
                    'change_percent': change_percent,
                    'slower': change_factor > 1.0,
                }

        return comparison

    def get_top_functions(
        self,
        profile: ProfileResult,
        n: int = 10,
        sort_by: str = 'cumtime'
    ) -> list:
        """Get top N functions by specified metric"""
        sorted_funcs = sorted(
            profile.function_stats.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )

        return [
            {
                'name': stats['function'],
                'file': stats['filename'],
                'line': stats['line'],
                'calls': stats['ncalls'],
                'total_time_ms': stats['tottime'] * 1000,
                'cumulative_time_ms': stats['cumtime'] * 1000,
                'time_per_call_ms': stats['percall_cum'] * 1000,
            }
            for key, stats in sorted_funcs[:n]
        ]
