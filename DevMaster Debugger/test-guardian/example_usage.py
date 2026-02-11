"""
Example usage of Test-Guardian API
Demonstrates how to use Test-Guardian programmatically
"""

from test_guardian import TestGuardian, TestFramework
from pathlib import Path


def example_1_generate_for_file():
    """Example 1: Generate tests for entire file"""
    print("\n=== Example 1: Generate Tests for File ===")

    tg = TestGuardian()

    # Generate tests for demo_functions.py
    report = tg.generate_tests_for_file(
        Path("demo_functions.py"),
        output_path=Path("test_demo_generated.py")
    )

    print(f"Generated {report.summary['tests_generated']} tests")
    print(f"Fixtures: {report.summary['fixtures']}")
    print(f"Mocks: {report.summary['mocks']}")
    print(f"Output: {report.generated_file}")


def example_2_generate_for_function():
    """Example 2: Generate tests for specific function"""
    print("\n=== Example 2: Generate Tests for Specific Function ===")

    tg = TestGuardian()

    # Generate tests only for 'divide' function
    report = tg.generate_tests_for_function(
        Path("demo_functions.py"),
        "divide",
        output_path=Path("test_divide.py")
    )

    print(f"Function: divide")
    print(f"Tests generated: {report.summary['tests_generated']}")
    print(f"Output: {report.generated_file}")


def example_3_preview():
    """Example 3: Preview without generating"""
    print("\n=== Example 3: Preview Mode ===")

    tg = TestGuardian()

    preview = tg.preview_tests(Path("demo_functions.py"))
    print(preview)


def example_4_unittest_framework():
    """Example 4: Use unittest instead of pytest"""
    print("\n=== Example 4: Unittest Framework ===")

    tg = TestGuardian(framework=TestFramework.UNITTEST)

    report = tg.generate_tests_for_function(
        Path("demo_functions.py"),
        "add",
        output_path=Path("test_add_unittest.py")
    )

    print(f"Generated unittest tests: {report.generated_file}")


def example_5_analyze_testability():
    """Example 5: Analyze code testability"""
    print("\n=== Example 5: Analyze Testability ===")

    tg = TestGuardian()

    functions = tg.code_analyzer.find_testable_functions(
        Path("demo_functions.py"),
        min_complexity=1
    )

    print(f"Found {len(functions)} testable functions:")
    for func in functions:
        print(f"  - {func.name} (complexity: {func.complexity}, params: {len(func.parameters)})")


def main():
    """Run all examples"""
    print("=" * 60)
    print("Test-Guardian API Examples")
    print("=" * 60)

    example_1_generate_for_file()
    example_2_generate_for_function()
    example_3_preview()
    example_4_unittest_framework()
    example_5_analyze_testability()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nGenerated test files:")
    print("  - test_demo_generated.py")
    print("  - test_divide.py")
    print("  - test_add_unittest.py")
    print("\nTry running them:")
    print("  pytest test_demo_generated.py")
    print("  pytest test_divide.py")
    print("  python -m unittest test_add_unittest.py")


if __name__ == '__main__':
    main()
