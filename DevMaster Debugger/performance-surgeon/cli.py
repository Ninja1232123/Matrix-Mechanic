#!/usr/bin/env python3
"""
Performance-Surgeon CLI

Usage:
    perf-surgeon scan <path>                # Scan for performance issues
    perf-surgeon optimize <path>            # Auto-optimize bottlenecks
    perf-surgeon optimize <path> --dry-run  # Preview optimizations
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from performance_surgeon.core import PerformanceSurgeon, QuickProfiler
    from performance_surgeon.optimizers.auto_optimizer import AutoOptimizer
    from performance_surgeon.patterns.performance_patterns import Severity
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from core import PerformanceSurgeon, QuickProfiler
    from optimizers.auto_optimizer import AutoOptimizer
    from patterns.performance_patterns import Severity


def print_summary(summary):
    """Print scan summary"""
    print(f"\n=== PERFORMANCE SCAN SUMMARY ===\n")
    print(f"Total Issues: {summary['total']}")
    print(f"  CRITICAL: {summary['critical']} (10x+ slowdown)")
    print(f"  HIGH: {summary['high']} (3-10x slowdown)")
    print(f"  MEDIUM: {summary['medium']} (2-3x slowdown)")
    print(f"  LOW: {summary['low']} (<2x slowdown)")
    print()


def cmd_scan(args):
    """Scan for performance issues"""
    path = args.path
    
    print(f"⚡ Performance-Surgeon - Scanning: {path}\n")
    
    findings, summary = QuickProfiler.quick_scan(path, Severity.MEDIUM)
    
    print_summary(summary)
    
    if findings:
        print("=== PERFORMANCE ISSUES ===\n")
        for i, finding in enumerate(findings[:20], 1):  # Limit output
            print(f"[{i}] {finding.pattern.severity.value} - {finding.pattern.name}")
            print(f"    File: {finding.file_path}:{finding.line_number}")
            print(f"    Complexity: {finding.pattern.complexity_before} → {finding.pattern.complexity_after}")
            print(f"    Speedup: {finding.pattern.estimated_speedup}")
            print(f"    Code: {finding.line_content.strip()}")
            print(f"    Fix: {finding.pattern.fix_strategy}\n")
    else:
        print("✓ No performance issues detected!")


def cmd_optimize(args):
    """Optimize performance issues"""
    path = args.path
    dry_run = args.dry_run
    
    mode = "DRY RUN" if dry_run else "LIVE MODE"
    print(f"⚡ Performance-Surgeon - Optimizing: {path} ({mode})\n")
    
    # Scan first
    scanner = PerformanceSurgeon()
    if os.path.isfile(path):
        findings = scanner.scan_file(path)
    else:
        findings = scanner.scan_directory(path)
    
    if not findings:
        print("✓ No performance issues found!")
        return
    
    print(f"Found {len(findings)} performance issue(s)\n")
    
    # Group by file
    by_file = {}
    for f in findings:
        if f.file_path not in by_file:
            by_file[f.file_path] = []
        by_file[f.file_path].append(f)
    
    # Optimize
    optimizer = AutoOptimizer(dry_run=dry_run)
    total_optimized = 0
    
    for filepath, file_findings in by_file.items():
        print(f"Optimizing: {filepath}")
        results = optimizer.optimize_file(filepath, file_findings)
        
        for result in results:
            if result.success:
                total_optimized += 1
                print(f"  ✓ {result.message} ({result.estimated_speedup})")
        print()
    
    print(f"=== SUMMARY ===")
    print(f"Optimized: {total_optimized}/{len(findings)} issues")
    
    if dry_run:
        print("\nThis was a DRY RUN. Run without --dry-run to apply changes.")


def main():
    parser = argparse.ArgumentParser(description='Performance-Surgeon - Performance Optimizer')
    subparsers = parser.add_subparsers(dest='command')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan for performance issues')
    scan_parser.add_argument('path', help='File or directory to scan')
    
    # Optimize command
    opt_parser = subparsers.add_parser('optimize', help='Optimize performance issues')
    opt_parser.add_argument('path', help='File or directory to optimize')
    opt_parser.add_argument('--dry-run', action='store_true', help='Preview changes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'scan':
        cmd_scan(args)
    elif args.command == 'optimize':
        cmd_optimize(args)


if __name__ == '__main__':
    main()
