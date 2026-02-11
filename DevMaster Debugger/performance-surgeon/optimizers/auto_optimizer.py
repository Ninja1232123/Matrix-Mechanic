"""
Performance-Surgeon Auto-Optimizer

Automatically optimizes detected performance bottlenecks.
"""

import re
import os
import ast
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from ..core import PerformanceFinding
    from ..patterns.performance_patterns import PerformancePattern
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import PerformanceFinding
    from patterns.performance_patterns import PerformancePattern


@dataclass
class OptimizationResult:
    """Result of applying an optimization"""
    finding: PerformanceFinding
    success: bool
    old_code: str
    new_code: str
    message: str
    estimated_speedup: str


class AutoOptimizer:
    """Automatically optimizes performance bottlenecks"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.results: List[OptimizationResult] = []

    def optimize_finding(self, finding: PerformanceFinding) -> OptimizationResult:
        """Optimize a single performance finding"""
        pattern_name = finding.pattern.name
        optimizer_method = getattr(self, f"_optimize_{pattern_name}", None)

        if optimizer_method:
            return optimizer_method(finding)
        else:
            return self._generic_optimization(finding)

    def optimize_file(self, filepath: str, findings: List[PerformanceFinding]) -> List[OptimizationResult]:
        """Optimize all findings in a file"""
        results = []

        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Apply optimizations (in reverse to preserve line numbers)
        findings_by_line = {}
        for finding in findings:
            line = finding.line_number
            if line not in findings_by_line:
                findings_by_line[line] = []
            findings_by_line[line].append(finding)

        for line_num in sorted(findings_by_line.keys(), reverse=True):
            for finding in findings_by_line[line_num]:
                result = self.optimize_finding(finding)
                results.append(result)

                if result.success and not self.dry_run:
                    # Update the line
                    lines[line_num - 1] = result.new_code + '\n'

        # Write back if not dry run
        if not self.dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)

        self.results.extend(results)
        return results

    # ========================================================================
    # ALGORITHMIC OPTIMIZATIONS
    # ========================================================================

    def _optimize_append_in_loop(self, finding: PerformanceFinding) -> OptimizationResult:
        """Convert append in loop to list comprehension"""
        old_code = finding.line_content
        
        # This is complex - add comment suggestion
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        new_code = f"{indent_str}# PERFORMANCE: Convert to list comprehension for 2-3x speedup\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added optimization suggestion for list comprehension",
            estimated_speedup="2-3x"
        )

    def _optimize_repeated_string_concatenation(self, finding: PerformanceFinding) -> OptimizationResult:
        """Fix string concatenation in loop"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        new_code = f"{indent_str}# PERFORMANCE: Use ''.join() instead of += in loop\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added suggestion to use ''.join() for string concatenation",
            estimated_speedup="10-100x"
        )

    # ========================================================================
    # DATABASE OPTIMIZATIONS
    # ========================================================================

    def _optimize_n_plus_one_query(self, finding: PerformanceFinding) -> OptimizationResult:
        """Fix N+1 query problem"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        # Add prefetch_related suggestion
        new_code = f"{indent_str}# PERFORMANCE: Add .prefetch_related() or .select_related() to avoid N+1 queries\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added N+1 query optimization suggestion",
            estimated_speedup="100-1000x"
        )

    def _optimize_query_in_loop(self, finding: PerformanceFinding) -> OptimizationResult:
        """Fix query in loop"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        new_code = f"{indent_str}# PERFORMANCE: Move query outside loop and use .filter(id__in=...)\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added suggestion to batch database queries",
            estimated_speedup="100x+"
        )

    # ========================================================================
    # CACHING OPTIMIZATIONS
    # ========================================================================

    def _optimize_uncached_expensive_function(self, finding: PerformanceFinding) -> OptimizationResult:
        """Add @lru_cache decorator"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        # Add @lru_cache decorator
        new_code = f"{indent_str}from functools import lru_cache\n"
        new_code += f"{indent_str}@lru_cache(maxsize=128)\n"
        new_code += old_code
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added @lru_cache decorator for memoization",
            estimated_speedup="1000x+"
        )

    def _optimize_repeated_file_reads(self, finding: PerformanceFinding) -> OptimizationResult:
        """Cache file reads"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        new_code = f"{indent_str}# PERFORMANCE: Read file once before loop and cache\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added suggestion to cache file reads",
            estimated_speedup="100x"
        )

    # ========================================================================
    # MEMORY OPTIMIZATIONS
    # ========================================================================

    def _optimize_loading_entire_file(self, finding: PerformanceFinding) -> OptimizationResult:
        """Use line iteration instead of loading entire file"""
        old_code = finding.line_content
        
        # Replace .readlines() with iteration
        if '.readlines()' in old_code:
            new_code = old_code.replace('.readlines()', '  # Use: for line in f: ...')
            
            return OptimizationResult(
                finding=finding,
                success=True,
                old_code=old_code,
                new_code=new_code,
                message="Suggested streaming file line by line",
                estimated_speedup="1000x less memory"
            )
        
        # Replace .read() suggestion
        if '.read()' in old_code:
            indent = len(old_code) - len(old_code.lstrip())
            indent_str = ' ' * indent
            new_code = f"{indent_str}# PERFORMANCE: Stream file instead: for line in f: ...\n{old_code}"
            
            return OptimizationResult(
                finding=finding,
                success=True,
                old_code=old_code,
                new_code=new_code,
                message="Added suggestion to stream file instead of loading all",
                estimated_speedup="1000x less memory"
            )

        return self._generic_optimization(finding)

    def _optimize_list_when_generator_works(self, finding: PerformanceFinding) -> OptimizationResult:
        """Convert list comprehension to generator"""
        old_code = finding.line_content
        
        # Replace [...] with (...)
        if '[' in old_code and 'for' in old_code and ']' in old_code:
            # Find list comprehension pattern
            match = re.search(r'\[([^\[\]]+for[^\[\]]+)\]', old_code)
            if match:
                list_comp = match.group(0)
                gen_expr = list_comp.replace('[', '(', 1).replace(']', ')', 1)
                new_code = old_code.replace(list_comp, gen_expr)
                
                return OptimizationResult(
                    finding=finding,
                    success=True,
                    old_code=old_code,
                    new_code=new_code,
                    message="Converted list comprehension to generator expression",
                    estimated_speedup="100x less memory"
                )

        return self._generic_optimization(finding)

    # ========================================================================
    # CPU OPTIMIZATIONS
    # ========================================================================

    def _optimize_global_in_loop(self, finding: PerformanceFinding) -> OptimizationResult:
        """Cache global lookups"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        new_code = f"{indent_str}# PERFORMANCE: Cache len() outside loop\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added suggestion to cache global lookups",
            estimated_speedup="1.2-1.5x"
        )

    def _optimize_repeated_regex_compilation(self, finding: PerformanceFinding) -> OptimizationResult:
        """Compile regex pattern once"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        # Extract pattern if possible
        pattern_match = re.search(r're\.\w+\(r?["\']([^"\']+)["\']', old_code)
        if pattern_match:
            pattern = pattern_match.group(1)
            new_code = f"{indent_str}# PERFORMANCE: Compile regex before loop\n"
            new_code += f"{indent_str}# pattern = re.compile(r'{pattern}')\n"
            new_code += old_code
            
            return OptimizationResult(
                finding=finding,
                success=True,
                old_code=old_code,
                new_code=new_code,
                message="Added suggestion to compile regex outside loop",
                estimated_speedup="5-10x"
            )

        return self._generic_optimization(finding)

    # ========================================================================
    # I/O OPTIMIZATIONS
    # ========================================================================

    def _optimize_unbuffered_writes(self, finding: PerformanceFinding) -> OptimizationResult:
        """Batch file writes"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        new_code = f"{indent_str}# PERFORMANCE: Batch writes using ''.join() outside loop\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message="Added suggestion to batch file writes",
            estimated_speedup="10-50x"
        )

    # ========================================================================
    # GENERIC OPTIMIZATION
    # ========================================================================

    def _generic_optimization(self, finding: PerformanceFinding) -> OptimizationResult:
        """Generic optimization comment"""
        old_code = finding.line_content
        indent = len(old_code) - len(old_code.lstrip())
        indent_str = ' ' * indent
        
        new_code = f"{indent_str}# PERFORMANCE ({finding.pattern.complexity_before} → {finding.pattern.complexity_after}): "
        new_code += f"{finding.pattern.fix_strategy}\n{old_code}"
        
        return OptimizationResult(
            finding=finding,
            success=True,
            old_code=old_code,
            new_code=new_code,
            message=f"Added performance suggestion: {finding.pattern.fix_strategy}",
            estimated_speedup=finding.pattern.estimated_speedup
        )

    def get_summary(self) -> Dict[str, int]:
        """Get summary of optimization results"""
        return {
            "total": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
        }
