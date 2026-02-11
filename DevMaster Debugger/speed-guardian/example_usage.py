"""
Example usage of Speed-Guardian API
Demonstrates how to use Speed-Guardian programmatically
"""

from speed_guardian import SpeedGuardian
from speed_guardian.config import SpeedGuardianConfig
from pathlib import Path


def example_1_profile_function():
    """Example 1: Profile a single function"""
    print("\n=== Example 1: Profile a Function ===")

    sg = SpeedGuardian()

    def slow_function(n):
        result = []
        for i in range(n):
            result.append(i * i)
        return result

    # Profile the function
    report = sg.profile_function(slow_function, 10000)

    # Display results
    print(f"Total time: {report.summary['total_time_ms']:.2f}ms")
    print(f"Bottlenecks found: {report.summary['bottlenecks']}")

    for bottleneck in report.bottlenecks[:3]:
        print(f"  - {bottleneck.description}")


def example_2_profile_script():
    """Example 2: Profile a script"""
    print("\n=== Example 2: Profile a Script ===")

    sg = SpeedGuardian()

    # Profile demo_slow.py
    script_path = Path("demo_slow.py")

    if script_path.exists():
        report = sg.profile_script(script_path, analyze_patterns=True)

        print(f"Script: {report.target}")
        print(f"Total time: {report.summary['total_time_ms']:.2f}ms")
        print(f"Optimizations found: {report.summary.get('optimizations_found', 0)}")

        # Show top optimizations
        print("\nTop Optimizations:")
        for i, opt in enumerate(report.optimizations[:5], 1):
            print(f"  {i}. [{opt.type.value}] {opt.description}")
            print(f"     Speedup: {opt.estimated_speedup:.1f}x")
    else:
        print("demo_slow.py not found - skipping")


def example_3_auto_optimize():
    """Example 3: Auto-optimize with dry run"""
    print("\n=== Example 3: Auto-Optimize (Dry Run) ===")

    sg = SpeedGuardian()

    script_path = Path("demo_slow.py")

    if script_path.exists():
        # Profile and optimize (dry run)
        result = sg.profile_and_optimize(
            script_path,
            auto_fix=True,
            dry_run=True  # Preview changes only
        )

        report = result['report']
        fix_results = result['fix_results']

        print(f"Would apply: {len(fix_results['applied'])} optimizations")
        print(f"Estimated speedup: {report.get_total_speedup():.2f}x")
    else:
        print("demo_slow.py not found - skipping")


def example_4_complexity_analysis():
    """Example 4: Analyze code complexity"""
    print("\n=== Example 4: Complexity Analysis ===")

    sg = SpeedGuardian()

    script_path = Path("demo_slow.py")

    if script_path.exists():
        analysis = sg.analyze_complexity(script_path)

        print(f"Functions analyzed: {len(analysis)}")

        # Show complex functions
        summary = sg.complexity_analyzer.get_summary(analysis)
        print(f"Average complexity: {summary['avg_complexity']:.1f}")
        print(f"Max complexity: {summary['max_complexity']}")

        # Show O(n²) functions
        quadratic = [
            name for name, metrics in analysis.items()
            if 'n²' in metrics['estimated_time_complexity']
        ]

        if quadratic:
            print(f"\nO(n²) functions detected:")
            for func in quadratic:
                print(f"  - {func}")
    else:
        print("demo_slow.py not found - skipping")


def example_5_find_bottlenecks():
    """Example 5: Find specific types of bottlenecks"""
    print("\n=== Example 5: Find Specific Bottlenecks ===")

    sg = SpeedGuardian()

    def test_function():
        import time
        data = {'key': 'value'}

        # Slow operation
        result = []
        for i in range(1000):
            result.append(data.get('key'))

        time.sleep(0.1)  # Simulate I/O
        return result

    # Profile
    report = sg.profile_function(test_function)
    profile = report.profile_result

    # Find I/O bottlenecks
    io_bottlenecks = sg.find_io_bottlenecks(profile)
    print(f"I/O bottlenecks: {len(io_bottlenecks)}")

    # Find slow functions
    slow_funcs = sg.find_slow_functions(profile, threshold_ms=10.0)
    print(f"Slow functions (>10ms): {len(slow_funcs)}")

    # Find frequent callers
    frequent = sg.find_frequent_callers(profile, min_calls=10)
    print(f"Frequently called functions (>10 calls): {len(frequent)}")


def example_6_with_config():
    """Example 6: Use custom configuration"""
    print("\n=== Example 6: Custom Configuration ===")

    # Create custom config
    config = SpeedGuardianConfig()
    config.optimization.enable_auto_fix = True
    config.optimization.min_confidence = 0.8
    config.optimization.min_speedup = 2.0

    sg = SpeedGuardian(config=config)

    print(f"Auto-fix enabled: {config.optimization.enable_auto_fix}")
    print(f"Min confidence: {config.optimization.min_confidence}")
    print(f"Min speedup: {config.optimization.min_speedup}x")


def example_7_save_report():
    """Example 7: Save report to file"""
    print("\n=== Example 7: Save Report ===")

    sg = SpeedGuardian()

    def demo_function(n):
        return sum(i * i for i in range(n))

    # Profile
    report = sg.profile_function(demo_function, 10000)

    # Save as JSON
    output_path = Path("performance_report.json")
    sg.save_report(report, output_path)
    print(f"Report saved to: {output_path}")

    # Save as text
    output_path_txt = Path("performance_report.txt")
    sg.save_report(report, output_path_txt)
    print(f"Report saved to: {output_path_txt}")


def example_8_memory_profiling():
    """Example 8: Profile with memory tracking"""
    print("\n=== Example 8: Memory Profiling ===")

    sg = SpeedGuardian()

    def memory_heavy(n):
        # Creates large list
        data = [i * i for i in range(n)]
        return sum(data)

    # Profile with memory
    result = sg.profile_with_memory(memory_heavy, 100000)

    perf = result['performance']
    memory = result['memory']

    print(f"Execution time: {perf.summary['total_time_ms']:.2f}ms")
    print(f"Memory used: {memory['memory_used_mb']:.2f}MB")
    print(f"Peak memory: {memory['peak_memory_mb']:.2f}MB")


def main():
    """Run all examples"""
    print("=" * 60)
    print("Speed-Guardian API Examples")
    print("=" * 60)

    example_1_profile_function()
    example_2_profile_script()
    example_3_auto_optimize()
    example_4_complexity_analysis()
    example_5_find_bottlenecks()
    example_6_with_config()
    example_7_save_report()
    example_8_memory_profiling()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
