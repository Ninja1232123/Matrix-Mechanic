#!/usr/bin/env python3
"""
The Complete Feedback Loop - Systematic Error Discovery and Auto-Fixing

Philosophy: "Every error you debug is a bug you'll never have to fix again.
            All you have to do is fix every error one more time."

This module connects three systems:
1. Adaptive error handling (dev vs prod behavior)
2. Error pattern collection (learning from runtime)
3. Universal debugger (auto-fixing based on patterns)

The feedback loop:
  Dev: Run → Crash → Collect Pattern → Fix Manually → Add to ERROR_DATABASE
  Prod: Run → Catch → Log Pattern → Notify Devs
  Maintenance: Analyze Patterns → Enhance ERROR_DATABASE → Deploy
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
from dataclasses import dataclass

from adaptive_error_handler import analyze_error_patterns, LOG_DIR
from universal_debugger import ERROR_DATABASE, run_and_capture_error, parse_error, fix_error


@dataclass
class ErrorInsight:
    """Insights from error pattern analysis."""
    error_type: str
    frequency: int
    in_database: bool
    example_traceback: str
    example_context: dict

    def __str__(self):
        status = "✅ AUTO-FIXABLE" if self.in_database else "❌ NEEDS PATTERN"
        return f"{self.error_type} ({self.frequency}x) {status}"


class FeedbackLoop:
    """
    Orchestrates the complete feedback loop between error discovery,
    pattern learning, and auto-fixing.
    """

    def __init__(self):
        self.error_patterns_file = LOG_DIR / "error_patterns.jsonl"
        self.insights: List[ErrorInsight] = []

    def analyze_runtime_errors(self) -> List[ErrorInsight]:
        """
        Analyze errors collected during runtime to identify which ones
        need to be added to ERROR_DATABASE.
        """
        if not self.error_patterns_file.exists():
            print("No runtime error patterns collected yet.")
            print("Run your application with adaptive_error_handler to collect data.")
            return []

        # Load all error patterns
        patterns = []
        with open(self.error_patterns_file, "r") as f:
            for line in f:
                try:
                    patterns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not patterns:
            return []

        # Group by error type
        error_groups: Dict[str, List[dict]] = {}
        for pattern in patterns:
            error_type = pattern.get("error_type", "Unknown")
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(pattern)

        # Generate insights
        insights = []
        for error_type, occurrences in error_groups.items():
            in_database = error_type in ERROR_DATABASE
            example = occurrences[0]

            insight = ErrorInsight(
                error_type=error_type,
                frequency=len(occurrences),
                in_database=in_database,
                example_traceback=example.get("traceback", ""),
                example_context=example.get("context", {})
            )
            insights.append(insight)

        self.insights = sorted(insights, key=lambda x: x.frequency, reverse=True)
        return self.insights

    def report_coverage(self):
        """
        Report on ERROR_DATABASE coverage vs. runtime errors.

        This shows how well our auto-fixer handles real-world errors.
        """
        if not self.insights:
            self.analyze_runtime_errors()

        if not self.insights:
            print("No insights available. Run analysis first.")
            return

        total_errors = sum(i.frequency for i in self.insights)
        covered_errors = sum(i.frequency for i in self.insights if i.in_database)
        coverage_percent = (covered_errors / total_errors * 100) if total_errors > 0 else 0

        print("\n" + "=" * 70)
        print("📊 FEEDBACK LOOP ANALYSIS")
        print("=" * 70)

        print(f"\n🎯 Error Coverage: {coverage_percent:.1f}%")
        print(f"   {covered_errors}/{total_errors} errors can be auto-fixed")
        print(f"   {len(ERROR_DATABASE)} error types in database")

        print("\n✅ Auto-fixable errors:")
        for insight in self.insights:
            if insight.in_database:
                print(f"   {insight}")

        uncovered = [i for i in self.insights if not i.in_database]
        if uncovered:
            print("\n❌ Errors needing patterns:")
            for insight in uncovered:
                print(f"   {insight}")
                print(f"      Context: {insight.example_context.get('function', 'unknown')}")

        print("\n💡 Recommendations:")
        if uncovered:
            print(f"   → Add patterns for {len(uncovered)} error types to ERROR_DATABASE")
            print(f"   → This would increase coverage to 100%")
        else:
            print(f"   → All runtime errors are covered! 🎉")

        print("\n" + "=" * 70)

    def discover_new_errors(self, script_path: str, test_inputs: List[str] = None):
        """
        Systematically discover errors by running script with various inputs.

        This is the "methodical and empirical" error discovery process.
        """
        if not Path(script_path).exists():
            print(f"Script not found: {script_path}")
            return

        print(f"\n🔍 Error Discovery Mode")
        print(f"   Target: {script_path}")
        print()

        discovered_errors = []
        test_cases = test_inputs or ["", "invalid", "null", "[]", "{}", "-1", "0"]

        for i, test_input in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Testing with input: {repr(test_input)}")

            # Run with test input
            result = subprocess.run(
                [sys.executable, script_path],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                error_type, error_file, error_line, error_msg = parse_error(
                    result.stderr, script_path
                )
                if error_type:
                    discovered_errors.append((error_type, error_line, test_input))
                    in_db = "✅" if error_type in ERROR_DATABASE else "❌"
                    print(f"   Found: {error_type} at line {error_line} {in_db}")

        print(f"\n📋 Discovery complete: {len(discovered_errors)} errors found")
        return discovered_errors

    def suggest_missing_patterns(self):
        """
        Suggest ERROR_DATABASE patterns for uncovered errors.

        This helps developers extend the auto-fixer.
        """
        if not self.insights:
            self.analyze_runtime_errors()

        uncovered = [i for i in self.insights if not i.in_database]

        if not uncovered:
            print("All error types are already in ERROR_DATABASE!")
            return

        print("\n" + "=" * 70)
        print("🛠️  SUGGESTED ERROR_DATABASE ADDITIONS")
        print("=" * 70)

        for insight in uncovered:
            print(f"\n'{insight.error_type}': {{")
            print(f"    'description': '{insight.error_type} (add description)',")
            print(f"    'patterns': [")
            print(f"        {{")
            print(f"            'detect': r'.*',  # TODO: Add specific pattern")
            print(f"            'fix': lambda line, indent, error_msg: line,  # TODO: Add fix")
            print(f"            'multiline': False,")
            print(f"            'confidence': 0.5")
            print(f"        }}")
            print(f"    ]")
            print(f"}},")
            print(f"# Frequency: {insight.frequency}x")
            print(f"# Example: {insight.example_context.get('function', 'unknown')}")

        print("\n" + "=" * 70)


class ErrorFuzzer:
    """
    Systematically generate inputs to trigger errors.

    This implements the "methodically create and identify errors" philosophy.
    """

    @staticmethod
    def generate_edge_cases() -> List[dict]:
        """Generate common edge case inputs that often cause errors."""
        return [
            # Empty/null cases
            {"type": "empty_string", "value": ""},
            {"type": "none", "value": None},
            {"type": "empty_list", "value": []},
            {"type": "empty_dict", "value": {}},

            # Boundary cases
            {"type": "zero", "value": 0},
            {"type": "negative", "value": -1},
            {"type": "max_int", "value": sys.maxsize},

            # Type mismatches
            {"type": "string_as_number", "value": "not_a_number"},
            {"type": "number_as_string", "value": 42},

            # Special characters
            {"type": "unicode", "value": "🔥💥🐛"},
            {"type": "escape_chars", "value": "\n\r\t\\\""},

            # Malformed data
            {"type": "invalid_json", "value": "{not json}"},
            {"type": "truncated", "value": "incomplete..."},
        ]

    @staticmethod
    def test_function(func, test_cases: List[dict]) -> List[Tuple[str, Exception]]:
        """
        Test a function with edge cases and collect errors.

        Returns list of (test_case_type, exception) tuples.
        """
        errors_found = []

        for test_case in test_cases:
            try:
                func(test_case["value"])
            except Exception as e:
                errors_found.append((test_case["type"], e))

        return errors_found


def main():
    """Demonstrate the complete feedback loop."""
    print("=" * 70)
    print("🔄 FEEDBACK LOOP - Error Discovery & Auto-Fixing System")
    print("=" * 70)

    loop = FeedbackLoop()

    # 1. Analyze runtime errors
    print("\n[1] Analyzing runtime error patterns...")
    insights = loop.analyze_runtime_errors()

    if insights:
        loop.report_coverage()
        loop.suggest_missing_patterns()
    else:
        print("   No runtime errors collected yet.")
        print("   Run your app with adaptive_error_handler to collect data.")

    # 2. Show error fuzzing capabilities
    print("\n[2] Error Fuzzing Capabilities")
    print("=" * 70)
    fuzzer = ErrorFuzzer()
    edge_cases = fuzzer.generate_edge_cases()
    print(f"   Generated {len(edge_cases)} edge case test inputs")
    print("   Use ErrorFuzzer.test_function() to discover errors systematically")

    # 3. Show current ERROR_DATABASE size
    print(f"\n[3] ERROR_DATABASE Status")
    print("=" * 70)
    print(f"   {len(ERROR_DATABASE)} error types with auto-fix patterns")
    print(f"   Coverage:")

    error_categories = {
        "File Operations": ["FileNotFoundError", "PermissionError", "IOError"],
        "Data Access": ["KeyError", "IndexError", "AttributeError"],
        "Type Errors": ["TypeError", "ValueError", "ZeroDivisionError"],
        "Network": ["ConnectionError", "TimeoutError"],
        "Imports": ["ImportError", "ModuleNotFoundError"],
    }

    for category, errors in error_categories.items():
        covered = sum(1 for e in errors if e in ERROR_DATABASE)
        total = len(errors)
        print(f"   {category}: {covered}/{total} covered")

    print("\n" + "=" * 70)
    print("💡 Feedback Loop Philosophy:")
    print("   Every error you debug is a bug you'll never have to fix again.")
    print("   All you have to do is fix every error one more time.")
    print("=" * 70)


if __name__ == "__main__":
    main()
