# Speed-Guardian ⚡

> Automatic Performance Profiler and Optimizer for Python

Speed-Guardian analyzes your Python code, identifies performance bottlenecks, and automatically applies optimizations to make your code faster. It's the performance sibling to Type-Guardian and Deploy-Shield.

## Features

### 🔍 Multi-Level Profiling
- **Function-level profiling** with cProfile
- **Line-by-line profiling** for detailed analysis
- **Memory profiling** to track allocations
- **Async/await performance** monitoring

### 🎯 Intelligent Analysis
- **Bottleneck detection** with severity levels (Critical, High, Medium, Low)
- **Pattern recognition** for 40+ inefficient code patterns
- **Complexity analysis** (cyclomatic, time complexity, nesting depth)
- **I/O bottleneck identification**

### 🛠️ Auto-Fix Engine
- **Automatic caching** (@lru_cache) for pure functions
- **Loop to comprehension** transformations
- **Algorithm optimization** suggestions
- **Async/await** conversion hints
- **Memory optimization** recommendations

### 📊 Beautiful TUI Dashboard
- Interactive real-time performance metrics
- Tabbed interface (Bottlenecks, Optimizations, Details)
- Profile any Python file with one click
- Apply optimizations interactively

### 📈 Performance Regression Detection
- Track performance across git commits
- SQLite-based benchmark storage
- Automatic regression alerts
- Historical performance comparison

## Installation

```bash
cd speed-guardian
pip install -e .
```

### Dependencies

Speed-Guardian requires Python 3.8+ and the following packages:
- click (CLI interface)
- rich (beautiful terminal output)
- textual (TUI dashboard)
- psutil (system monitoring)
- memory_profiler (memory tracking)
- line_profiler (line-by-line profiling)
- gitpython (git integration)

## Quick Start

### 1. Profile a Script

```bash
speed-guardian profile my_script.py
```

This will:
- Profile the script execution
- Identify bottlenecks
- Suggest optimizations
- Display a beautiful report

### 2. Profile and Optimize

```bash
speed-guardian optimize my_script.py --auto-fix
```

This will automatically apply safe optimizations like:
- Adding @lru_cache decorators
- Suggesting algorithmic improvements
- Recommending async/await conversions

### 3. Launch Interactive Dashboard

```bash
speed-guardian dashboard
```

Interactive TUI with:
- Real-time profiling
- Bottleneck visualization
- One-click optimization
- Performance history

### 4. Analyze Code Complexity

```bash
speed-guardian complexity my_module.py
```

Shows:
- Cyclomatic complexity
- Nesting depth
- Time complexity estimates
- Lines of code per function

## Usage Examples

### Basic Profiling

```python
from speed_guardian import SpeedGuardian
from pathlib import Path

sg = SpeedGuardian()

# Profile a script
report = sg.profile_script(Path("my_script.py"))

# Display results
print(sg.generate_report_text(report))
```

### Profile a Function

```python
from speed_guardian import SpeedGuardian

sg = SpeedGuardian()

def my_function(n):
    result = []
    for i in range(n):
        result.append(i * 2)
    return result

# Profile the function
report = sg.profile_function(my_function, 10000)

# Get bottlenecks
for bottleneck in report.bottlenecks:
    print(bottleneck)
```

### Auto-Fix Mode

```python
from speed_guardian import SpeedGuardian
from pathlib import Path

sg = SpeedGuardian()

# Profile and optimize
result = sg.profile_and_optimize(
    Path("slow_script.py"),
    auto_fix=True,
    dry_run=False  # Set to True to preview changes
)

report = result['report']
fix_results = result['fix_results']

print(f"Applied: {len(fix_results['applied'])} optimizations")
print(f"Estimated speedup: {report.get_total_speedup():.2f}x")
```

### Regression Detection

```python
from speed_guardian.regression import RegressionDetector
from pathlib import Path

detector = RegressionDetector()

# Store current benchmarks
profile = sg.profile_script(Path("my_script.py"))
detector.store_profile(profile, "my_script.py")

# Detect regressions (after code changes)
regressions = detector.detect_regressions(profile, "my_script.py")

for regression in regressions:
    print(regression)
```

## Configuration

Create a `.speed_guardian.yaml` file:

```yaml
profile:
  enable_line_profiling: true
  enable_memory_profiling: true
  min_time_ms: 1.0

optimization:
  enable_auto_fix: false
  min_confidence: 0.7
  min_speedup: 1.2
  optimization_types:
    - caching
    - loop_comprehension
    - algorithm
    - database_query
    - io_buffering

regression:
  enable: true
  commits_to_check: 10
  slowdown_threshold: 1.5
  store_benchmarks: true

ui:
  theme: dark
  show_flamegraph: true
  show_memory: true
```

Initialize with:

```bash
speed-guardian init
```

## Architecture

```
Speed-Guardian
├── Profilers
│   ├── CProfileProfiler (function-level)
│   ├── LineProfilerImpl (line-by-line)
│   └── MemoryProfilerImpl (memory tracking)
├── Analyzers
│   ├── BottleneckDetector (find slow code)
│   ├── PatternDetector (inefficient patterns)
│   └── ComplexityAnalyzer (code complexity)
├── Optimizers
│   └── AutoFixer (apply optimizations)
├── Regression
│   └── RegressionDetector (track performance)
└── UI
    ├── CLI (command-line interface)
    └── Dashboard (interactive TUI)
```

## Detected Patterns

Speed-Guardian detects 40+ inefficient patterns including:

### Caching Opportunities
- Pure functions without @lru_cache
- Repeated dictionary lookups
- Function calls with constant args in loops

### Loop Optimizations
- List append in loop → list comprehension
- String concatenation in loop → join()
- Repeated function calls → hoist out of loop

### Algorithm Issues
- Nested loops (O(n²))
- Inefficient searching/filtering
- Unnecessary list copies

### I/O Bottlenecks
- Synchronous I/O in loops
- Unbuffered file operations
- Database N+1 queries

### Memory Issues
- Unnecessary memory allocations
- Large intermediate data structures
- Memory leaks

## CLI Commands

### profile
Profile a Python script and analyze performance.

```bash
speed-guardian profile script.py [OPTIONS]

Options:
  -o, --output PATH       Output report file
  -f, --format [text|json] Output format
  --analyze-patterns/--no-patterns  Analyze code patterns
```

### optimize
Profile and apply optimizations.

```bash
speed-guardian optimize script.py [OPTIONS]

Options:
  --auto-fix/--no-auto-fix  Automatically apply fixes
  --dry-run                 Preview changes without applying
  -o, --output PATH         Output report file
```

### complexity
Analyze code complexity.

```bash
speed-guardian complexity file.py
```

### dashboard
Launch interactive dashboard.

```bash
speed-guardian dashboard
```

### init
Initialize configuration.

```bash
speed-guardian init [--config-file PATH]
```

## Integration with DevMaster

Speed-Guardian integrates seamlessly with DevMaster:

```bash
devmaster analyze --performance
```

This combines:
- Speed-Guardian performance analysis
- CodeArchaeology hotspot detection
- DevNarrative for commit history
- Unified reporting

## Performance Metrics

Speed-Guardian tracks:
- **Execution time** (milliseconds)
- **Call counts** (how many times functions are called)
- **Time per call** (average execution time)
- **Memory usage** (MB allocated)
- **Complexity** (cyclomatic, time complexity)
- **Bottleneck severity** (Critical/High/Medium/Low)

## Best Practices

### 1. Profile Before Optimizing
Always profile first to find real bottlenecks. Don't guess!

```bash
speed-guardian profile my_app.py
```

### 2. Focus on Critical Issues
Start with critical and high-severity bottlenecks for maximum impact.

### 3. Use Regression Detection
Track performance across commits to catch regressions early.

```bash
# After each significant change
speed-guardian profile my_app.py --store-benchmarks
```

### 4. Review Auto-Fixes
Use `--dry-run` to preview optimizations before applying.

```bash
speed-guardian optimize my_app.py --auto-fix --dry-run
```

### 5. Combine Tools
Use Speed-Guardian with Type-Guardian and Deploy-Shield for complete code quality.

## Examples

### Example 1: Slow Loop

**Before:**
```python
def process_data(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result
```

**After Speed-Guardian:**
```python
def process_data(items):
    return [item * 2 for item in items]  # 1.5x faster
```

### Example 2: Missing Cache

**Before:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**After Speed-Guardian:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)  # 100x+ faster
```

### Example 3: Nested Loops

**Before:**
```python
def find_duplicates(list1, list2):
    duplicates = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                duplicates.append(item1)
    return duplicates
```

**After Speed-Guardian:**
```python
def find_duplicates(list1, list2):
    set2 = set(list2)
    return [item for item in list1 if item in set2]  # O(n) instead of O(n²)
```

## Roadmap

- [ ] Flamegraph visualization
- [ ] Multi-file project analysis
- [ ] CI/CD integration (GitHub Actions, GitLab CI)
- [ ] VS Code extension
- [ ] Team dashboard (multi-developer metrics)
- [ ] AI-powered optimization suggestions
- [ ] Async profiler for asyncio code
- [ ] GPU profiling support

## Contributing

Speed-Guardian is part of the Codes-Masterpiece ecosystem. Contributions welcome!

## License

MIT License - See LICENSE file for details

## Related Tools

- **Type-Guardian**: Automatic type error fixing
- **Deploy-Shield**: Deployment validation
- **Universal Debugger**: Runtime error fixing
- **AI Debug Companion**: Real-time error monitoring
- **DevMaster**: Unified CLI orchestrator

---

**Speed-Guardian** - Making Python blazingly fast, automatically! ⚡
