"""Core Speed-Guardian class"""

from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from datetime import datetime

from .config import SpeedGuardianConfig, DEFAULT_CONFIG
from .models import (
    ProfileResult,
    Bottleneck,
    Optimization,
    PerformanceReport,
    Severity,
)
from .profilers import CProfileProfiler, LineProfilerImpl, MemoryProfilerImpl
from .analyzers import BottleneckDetector, PatternDetector, ComplexityAnalyzer
from .optimizers import AutoFixer


class SpeedGuardian:
    """Main Speed-Guardian class - orchestrates profiling, analysis, and optimization"""

    def __init__(self, config: Optional[SpeedGuardianConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # Initialize components
        self.cprofile = CProfileProfiler()
        self.line_profiler = LineProfilerImpl()
        self.memory_profiler = MemoryProfilerImpl()
        self.bottleneck_detector = BottleneckDetector()
        self.pattern_detector = PatternDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.auto_fixer = AutoFixer()

    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> PerformanceReport:
        """Profile a single function and generate a complete report"""
        # Run cProfile
        profile_result = self.cprofile.profile_function(func, *args, **kwargs)

        # Detect bottlenecks
        bottlenecks = self.bottleneck_detector.detect_bottlenecks(profile_result)

        # No file to analyze for patterns in this case
        optimizations = []

        # Generate summary
        summary = {
            'total_time_ms': profile_result.total_time_ms,
            'total_calls': profile_result.total_calls,
            'bottlenecks': len(bottlenecks),
            'critical_bottlenecks': len([b for b in bottlenecks if b.severity == Severity.CRITICAL]),
        }

        return PerformanceReport(
            target=func.__name__,
            profile_result=profile_result,
            bottlenecks=bottlenecks,
            optimizations=optimizations,
            summary=summary,
        )

    def profile_script(
        self,
        script_path: Path,
        analyze_patterns: bool = True
    ) -> PerformanceReport:
        """Profile a Python script and generate a complete report"""
        # Run cProfile
        profile_result = self.cprofile.profile_script(script_path)

        # Detect bottlenecks
        bottlenecks = self.bottleneck_detector.detect_bottlenecks(profile_result)

        # Analyze patterns
        optimizations = []
        if analyze_patterns:
            optimizations = self.pattern_detector.analyze_file(script_path)

        # Filter optimizations by confidence
        optimizations = [
            opt for opt in optimizations
            if opt.confidence >= self.config.optimization.min_confidence
            and opt.estimated_speedup >= self.config.optimization.min_speedup
        ]

        # Generate summary
        summary = {
            'total_time_ms': profile_result.total_time_ms,
            'total_calls': profile_result.total_calls,
            'bottlenecks': len(bottlenecks),
            'critical_bottlenecks': len([b for b in bottlenecks if b.severity == Severity.CRITICAL]),
            'optimizations_found': len(optimizations),
        }

        return PerformanceReport(
            target=str(script_path),
            profile_result=profile_result,
            bottlenecks=bottlenecks,
            optimizations=optimizations,
            summary=summary,
        )

    def profile_and_optimize(
        self,
        script_path: Path,
        auto_fix: bool = False,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Profile a script and optionally apply optimizations"""
        # Profile
        report = self.profile_script(script_path, analyze_patterns=True)

        # Auto-fix if requested
        fix_results = None
        if auto_fix and report.optimizations:
            fix_results = self.auto_fixer.apply_optimizations(
                report.optimizations,
                dry_run=dry_run
            )

        return {
            'report': report,
            'fix_results': fix_results,
        }

    def analyze_complexity(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity of functions in a file"""
        return self.complexity_analyzer.analyze_file(file_path)

    def find_slow_functions(
        self,
        profile_result: ProfileResult,
        threshold_ms: float = 100.0
    ) -> List[Dict[str, Any]]:
        """Find slow functions from profile result"""
        return self.bottleneck_detector.find_slow_functions(
            profile_result,
            min_time_per_call_ms=threshold_ms
        )

    def find_frequent_callers(
        self,
        profile_result: ProfileResult,
        min_calls: int = 100
    ) -> List[Dict[str, Any]]:
        """Find frequently called functions"""
        return self.bottleneck_detector.find_frequent_callers(
            profile_result,
            min_calls=min_calls
        )

    def find_io_bottlenecks(
        self,
        profile_result: ProfileResult
    ) -> List[Dict[str, Any]]:
        """Find I/O related bottlenecks"""
        return self.bottleneck_detector.find_io_bottlenecks(profile_result)

    def compare_performance(
        self,
        old_profile: ProfileResult,
        new_profile: ProfileResult
    ) -> Dict[str, Any]:
        """Compare two profile results"""
        comparison = self.cprofile.compare_profiles(old_profile, new_profile)

        # Find regressions (functions that got slower)
        regressions = [
            {
                'function': func,
                **data
            }
            for func, data in comparison.items()
            if data['slower'] and data['change_factor'] > self.config.regression.slowdown_threshold
        ]

        # Find improvements (functions that got faster)
        improvements = [
            {
                'function': func,
                **data
            }
            for func, data in comparison.items()
            if not data['slower'] and data['change_factor'] < 0.9
        ]

        return {
            'regressions': regressions,
            'improvements': improvements,
            'comparison': comparison,
        }

    def generate_report_text(self, report: PerformanceReport) -> str:
        """Generate a text report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"Speed-Guardian Performance Report: {report.target}")
        lines.append(f"Generated: {report.timestamp}")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Time: {report.summary['total_time_ms']:.2f}ms")
        lines.append(f"Total Calls: {report.summary['total_calls']:,}")
        lines.append(f"Bottlenecks Found: {report.summary['bottlenecks']}")
        lines.append(f"  Critical: {report.summary.get('critical_bottlenecks', 0)}")
        if 'optimizations_found' in report.summary:
            lines.append(f"Optimizations Found: {report.summary['optimizations_found']}")
        lines.append("")

        # Top bottlenecks
        if report.bottlenecks:
            lines.append("TOP BOTTLENECKS")
            lines.append("-" * 80)
            for i, bottleneck in enumerate(report.bottlenecks[:10], 1):
                lines.append(f"{i}. {bottleneck}")
            lines.append("")

        # Optimizations
        if report.optimizations:
            lines.append("SUGGESTED OPTIMIZATIONS")
            lines.append("-" * 80)
            for i, opt in enumerate(report.optimizations[:10], 1):
                lines.append(f"{i}. {opt}")
            lines.append("")

        # Estimated speedup
        if report.optimizations:
            total_speedup = report.get_total_speedup()
            lines.append(f"Estimated Total Speedup: {total_speedup:.2f}x faster")
            lines.append("")

        lines.append("=" * 80)

        return '\n'.join(lines)

    def generate_report_dict(self, report: PerformanceReport) -> Dict[str, Any]:
        """Generate a dictionary report"""
        return {
            'target': report.target,
            'timestamp': report.timestamp.isoformat(),
            'summary': report.summary,
            'bottlenecks': [
                {
                    'function': b.function_name,
                    'file': b.file_path,
                    'line': b.line_number,
                    'time_ms': b.time_ms,
                    'percentage': b.percentage,
                    'severity': b.severity.value,
                    'description': b.description,
                }
                for b in report.bottlenecks
            ],
            'optimizations': [
                {
                    'type': o.type.value,
                    'file': o.file_path,
                    'line': o.line_number,
                    'description': o.description,
                    'estimated_speedup': o.estimated_speedup,
                    'confidence': o.confidence,
                    'applied': o.applied,
                }
                for o in report.optimizations
            ],
            'estimated_speedup': report.get_total_speedup(),
        }

    def save_report(self, report: PerformanceReport, output_path: Path) -> None:
        """Save report to file"""
        import json

        if output_path.suffix == '.json':
            # Save as JSON
            report_dict = self.generate_report_dict(report)
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
        else:
            # Save as text
            report_text = self.generate_report_text(report)
            with open(output_path, 'w') as f:
                f.write(report_text)

    def profile_with_memory(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Profile function with memory tracking"""
        # Profile execution time
        perf_report = self.profile_function(func, *args, **kwargs)

        # Profile memory
        memory_result = self.memory_profiler.profile_function(func, *args, **kwargs)

        return {
            'performance': perf_report,
            'memory': memory_result,
        }
