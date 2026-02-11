"""
Performance-Surgeon Core Scanner

Detects performance bottlenecks and anti-patterns in Python code.
"""

import ast
import re
import os
import cProfile
import pstats
import io
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .patterns.performance_patterns import (
        PerformancePattern,
        ALL_PATTERNS,
        Severity,
        get_patterns_by_severity,
        get_patterns_by_category
    )
except ImportError:
    from patterns.performance_patterns import (
        PerformancePattern,
        ALL_PATTERNS,
        Severity,
        get_patterns_by_severity,
        get_patterns_by_category
    )


@dataclass
class PerformanceFinding:
    """Represents a detected performance issue"""
    pattern: PerformancePattern
    file_path: str
    line_number: int
    line_content: str
    column: Optional[int] = None
    context_lines: List[str] = field(default_factory=list)
    estimated_impact: str = ""  # e.g., "10x slowdown"

    def __str__(self):
        severity_colors = {
            Severity.CRITICAL: "\033[91m",  # Red
            Severity.HIGH: "\033[93m",      # Yellow
            Severity.MEDIUM: "\033[94m",    # Blue
            Severity.LOW: "\033[92m",       # Green
            Severity.INFO: "\033[96m",      # Cyan
        }
        reset = "\033[0m"
        color = severity_colors.get(self.pattern.severity, "")

        return (
            f"{color}[{self.pattern.severity.value}]{reset} "
            f"{self.pattern.name}\n"
            f"  File: {self.file_path}:{self.line_number}\n"
            f"  Issue: {self.pattern.description}\n"
            f"  Current: {self.pattern.complexity_before} → Optimized: {self.pattern.complexity_after}\n"
            f"  Code: {self.line_content.strip()}\n"
            f"  Fix: {self.pattern.fix_strategy}\n"
            f"  Speedup: {self.pattern.estimated_speedup}"
        )


class PerformanceSurgeon:
    """Main performance scanner class"""

    def __init__(self,
                 patterns: List[PerformancePattern] = None,
                 severity_threshold: Severity = Severity.INFO):
        """
        Initialize the scanner

        Args:
            patterns: List of patterns to check (default: all patterns)
            severity_threshold: Minimum severity to report
        """
        self.patterns = patterns or ALL_PATTERNS
        self.severity_threshold = severity_threshold
        self.findings: List[PerformanceFinding] = []

    def scan_file(self, filepath: str) -> List[PerformanceFinding]:
        """
        Scan a single Python file for performance issues

        Args:
            filepath: Path to the Python file

        Returns:
            List of performance findings
        """
        findings = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Regex-based scanning
            findings.extend(self._scan_with_regex(filepath, lines, content))

            # AST-based scanning
            findings.extend(self._scan_with_ast(filepath, content))

        except Exception as e:
            print(f"Error scanning {filepath}: {e}")

        return findings

    def _scan_with_regex(self, filepath: str, lines: List[str], full_content: str) -> List[PerformanceFinding]:
        """Scan file using regex patterns"""
        findings = []

        for pattern in self.patterns:
            # Check severity threshold
            if self._severity_level(pattern.severity) < self._severity_level(self.severity_threshold):
                continue

            for regex in pattern.regex_patterns:
                try:
                    # Search in full content for multi-line patterns
                    for match in re.finditer(regex, full_content, re.MULTILINE):
                        # Find line number
                        line_num = full_content[:match.start()].count('\n') + 1
                        
                        if line_num <= len(lines):
                            line_content = lines[line_num - 1]
                            context = self._get_context_lines(lines, line_num, context_size=3)

                            finding = PerformanceFinding(
                                pattern=pattern,
                                file_path=filepath,
                                line_number=line_num,
                                line_content=line_content,
                                context_lines=context,
                                estimated_impact=pattern.estimated_speedup
                            )
                            findings.append(finding)
                            break  # Don't report same pattern multiple times per line
                
                except re.error:
                    continue

        return findings

    def _scan_with_ast(self, filepath: str, content: str) -> List[PerformanceFinding]:
        """Scan file using AST analysis for complex patterns"""
        findings = []

        try:
            tree = ast.parse(content)
            lines = content.split('\n')

            # Detect specific anti-patterns
            findings.extend(self._detect_nested_loops(tree, lines, filepath))
            findings.extend(self._detect_repeated_function_calls(tree, lines, filepath))
            
        except SyntaxError:
            pass

        return findings

    def _detect_nested_loops(self, tree: ast.AST, lines: List[str], filepath: str) -> List[PerformanceFinding]:
        """Detect nested loops that could be optimized"""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for nested loops
                for child in ast.walk(node):
                    if child != node and isinstance(child, ast.For):
                        # Found nested loop - check for 'in' operations
                        for subchild in ast.walk(child.body[0] if child.body else child):
                            if isinstance(subchild, ast.Compare):
                                for op in subchild.ops:
                                    if isinstance(op, ast.In):
                                        # Potential O(n²) pattern
                                        line_num = getattr(node, 'lineno', 0)
                                        if line_num > 0 and line_num <= len(lines):
                                            # Find matching pattern
                                            pattern = next((p for p in self.patterns 
                                                          if p.name == "nested_loops_in_instead_of_set"), None)
                                            if pattern:
                                                finding = PerformanceFinding(
                                                    pattern=pattern,
                                                    file_path=filepath,
                                                    line_number=line_num,
                                                    line_content=lines[line_num - 1],
                                                    context_lines=self._get_context_lines(lines, line_num, 3)
                                                )
                                                findings.append(finding)
        
        return findings

    def _detect_repeated_function_calls(self, tree: ast.AST, lines: List[str], filepath: str) -> List[PerformanceFinding]:
        """Detect expensive function calls that could be cached"""
        findings = []
        
        # Look for recursive functions without @lru_cache
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function is recursive
                is_recursive = False
                has_cache_decorator = False
                
                # Check decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and 'cache' in decorator.id.lower():
                        has_cache_decorator = True
                    elif isinstance(decorator, ast.Attribute) and 'cache' in decorator.attr.lower():
                        has_cache_decorator = True
                
                # Check if function calls itself
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == node.name:
                            is_recursive = True
                
                if is_recursive and not has_cache_decorator:
                    # Found uncached recursive function
                    line_num = node.lineno
                    if line_num <= len(lines):
                        pattern = next((p for p in self.patterns 
                                      if p.name == "uncached_expensive_function"), None)
                        if pattern:
                            finding = PerformanceFinding(
                                pattern=pattern,
                                file_path=filepath,
                                line_number=line_num,
                                line_content=lines[line_num - 1],
                                context_lines=self._get_context_lines(lines, line_num, 3)
                            )
                            findings.append(finding)
        
        return findings

    def _get_context_lines(self, lines: List[str], line_num: int, context_size: int = 2) -> List[str]:
        """Get surrounding lines for context"""
        start = max(0, line_num - context_size - 1)
        end = min(len(lines), line_num + context_size)
        return lines[start:end]

    def _severity_level(self, severity: Severity) -> int:
        """Convert severity to numeric level for comparison"""
        levels = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }
        return levels.get(severity, 0)

    def scan_directory(self, directory: str, recursive: bool = True) -> List[PerformanceFinding]:
        """
        Scan a directory for performance issues

        Args:
            directory: Path to directory
            recursive: Whether to scan subdirectories

        Returns:
            List of all findings
        """
        findings = []

        if recursive:
            for root, dirs, files in os.walk(directory):
                # Skip common non-code directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv']]

                for file in files:
                    if file.endswith('.py'):
                        filepath = os.path.join(root, file)
                        findings.extend(self.scan_file(filepath))
        else:
            for file in os.listdir(directory):
                if file.endswith('.py'):
                    filepath = os.path.join(directory, file)
                    if os.path.isfile(filepath):
                        findings.extend(self.scan_file(filepath))

        self.findings = findings
        return findings

    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics of findings"""
        summary = {
            "total": len(self.findings),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

        for finding in self.findings:
            severity = finding.pattern.severity.value.lower()
            if severity in summary:
                summary[severity] += 1

        return summary

    def profile_function(self, func, *args, **kwargs):
        """Profile a specific function"""
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        return result, s.getvalue()


class QuickProfiler:
    """Quick profiling utilities"""

    @staticmethod
    def profile_file(filepath: str) -> str:
        """Profile execution of a Python file"""
        profiler = cProfile.Profile()
        
        with open(filepath) as f:
            code = f.read()
        
        profiler.enable()
        exec(code, {'__name__': '__main__'})
        profiler.disable()
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)
        
        return s.getvalue()

    @staticmethod
    def quick_scan(path: str, severity_threshold: Severity = Severity.MEDIUM) -> Tuple[List[PerformanceFinding], Dict[str, int]]:
        """
        Quick scan with sensible defaults

        Args:
            path: File or directory to scan
            severity_threshold: Minimum severity to report

        Returns:
            Tuple of (findings, summary)
        """
        scanner = PerformanceSurgeon(severity_threshold=severity_threshold)

        if os.path.isfile(path):
            findings = scanner.scan_file(path)
        else:
            findings = scanner.scan_directory(path)

        return findings, scanner.get_summary()
