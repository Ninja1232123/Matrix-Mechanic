# Test-Guardian 🧪

> Automatic Test Generation for Python

Test-Guardian analyzes your Python code and automatically generates comprehensive test suites including unit tests, fixtures, mocks, and edge cases. Stop writing boilerplate tests manually!

## Features

### 🎯 Intelligent Test Generation
- **Normal cases** - Happy path tests with reasonable inputs
- **Edge cases** - Empty values, None, boundaries, extremes
- **Error cases** - Exception handling tests
- **Property-based tests** - Integration with Hypothesis

### 🔍 Smart Code Analysis
- AST-based function analysis
- Parameter extraction with type hints
- Return type inference
- Exception detection
- Complexity calculation
- Pure function identification

### 🛠️ Fixture & Mock Generation
- Automatic pytest fixture creation
- Mock generation for external dependencies
- File I/O mocking
- Network/HTTP mocking
- Database mocking

### 📝 Multiple Frameworks
- **pytest** (default) - Modern, feature-rich
- **unittest** - Standard library

## Installation

```bash
cd test-guardian
pip install -e .
```

## Quick Start

### Generate Tests for a File

```bash
test-guardian generate mymodule.py
```

This creates `test_mymodule.py` with tests for all functions.

### Preview Tests

```bash
test-guardian preview mymodule.py
```

See what tests would be generated without creating files.

### Analyze Testability

```bash
test-guardian analyze mymodule.py
```

View which functions are testable and their complexity.

## Usage Examples

### Example 1: Basic Function

**Input (`calculator.py`):**
```python
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
```

**Generate tests:**
```bash
test-guardian generate calculator.py
```

**Output (`test_calculator.py`):**
```python
import pytest
from calculator import add

def test_add_normal():
    """Test add with normal inputs"""
    # Arrange
    a = 42
    b = 42

    # Act
    result = add(a, b)

    # Assert
    assert result == 84

def test_add_edge_a_0():
    """Edge case: a = 0"""
    a = 0
    b = 42
    result = add(a, b)
    assert result is not None

def test_add_edge_b_0():
    """Edge case: b = 0"""
    a = 42
    b = 0
    result = add(a, b)
    assert result is not None
```

### Example 2: Function with Exceptions

**Input:**
```python
def divide(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

**Generated test includes:**
```python
def test_divide_raises_valueerror():
    """Test that divide raises ValueError"""
    a = None
    b = None

    with pytest.raises(ValueError):
        result = divide(a, b)
```

### Example 3: Function with External Deps

**Input:**
```python
def load_config(filename: str) -> dict:
    """Load configuration from file"""
    with open(filename, 'r') as f:
        return json.load(f)
```

**Generated test includes mocks:**
```python
from unittest.mock import Mock, patch

def test_load_config_normal():
    """Test load_config with normal inputs"""
    # Mock builtins.open
    mock_open = Mock(return_value=mock_file_handle)

    filename = "test"
    result = load_config(filename)
    assert result is not None
```

## CLI Commands

### generate
Generate tests for a Python file.

```bash
test-guardian generate FILE [OPTIONS]

Options:
  -o, --output PATH          Output test file path
  -f, --framework [pytest|unittest]  Test framework (default: pytest)
  --function NAME            Generate for specific function only
```

### preview
Preview tests without generating files.

```bash
test-guardian preview FILE
```

### analyze
Analyze code testability.

```bash
test-guardian analyze FILE
```

## Programmatic Usage

```python
from test_guardian import TestGuardian
from pathlib import Path

# Initialize
tg = TestGuardian()

# Generate tests for entire file
report = tg.generate_tests_for_file(Path("mymodule.py"))

print(f"Generated {report.summary['tests_generated']} tests")
print(f"Output: {report.generated_file}")

# Generate for specific function
report = tg.generate_tests_for_function(
    Path("mymodule.py"),
    "my_function"
)

# Preview without generating
preview = tg.preview_tests(Path("mymodule.py"))
print(preview)
```

## Test Types Generated

### Normal Cases
Tests with typical, valid inputs:
- `test_function_normal`

### Edge Cases
Tests with boundary values:
- Empty strings, lists, dicts
- Zero, negative numbers
- None values
- Very large values
- Special characters

### Error Cases
Tests that expect exceptions:
- `test_function_raises_exception`
- `test_function_none_parameter`

### Type-Based Generation

Test-Guardian generates context-appropriate values based on type hints:

| Type | Normal Value | Edge Values |
|------|--------------|-------------|
| `int` | 42 | 0, -1, 999999, -999999 |
| `str` | "test" | "", " ", "\n", "x"*1000 |
| `list` | [1, 2, 3] | [], [1], [1]*1000 |
| `dict` | {'key': 'value'} | {}, large dict |
| `bool` | True | True, False |

## Architecture

```
Test-Guardian
├── Analyzers
│   ├── CodeAnalyzer (AST-based)
│   ├── TypeAnalyzer
│   └── DependencyAnalyzer
├── Generators
│   ├── TestCaseGenerator
│   ├── FixtureGenerator
│   └── MockGenerator
├── Formatters
│   ├── PytestFormatter
│   └── UnittestFormatter
└── CLI
    └── Rich terminal output
```

## What Gets Generated

For each testable function, Test-Guardian creates:

1. **Normal test** - Happy path with typical values
2. **Edge tests** - Boundary values for each parameter
3. **Error tests** - Exception handling (if function raises)
4. **Fixtures** - For complex data types (dicts, lists)
5. **Mocks** - For external dependencies (files, network, DB)

## Configuration

Test-Guardian uses sensible defaults, but you can customize:

```python
from test_guardian import TestGuardian, TestFramework

# Use unittest instead of pytest
tg = TestGuardian(framework=TestFramework.UNITTEST)

# Analyze specific functions only
testable = tg.code_analyzer.find_testable_functions(
    file_path,
    min_complexity=2  # Skip very simple functions
)
```

## Integration with Other Tools

### With Universal Debugger
1. Fix bugs with Universal Debugger
2. Generate tests with Test-Guardian
3. Prevent regressions!

### With Type-Guardian
1. Add type hints with Type-Guardian
2. Test-Guardian uses hints for better test generation

### With Speed-Guardian
1. Optimize code with Speed-Guardian
2. Generate performance tests to ensure no regressions

## Best Practices

### 1. Review Generated Tests
Always review and customize generated tests:
- Replace `TODO` comments with actual expected values
- Adjust edge cases for your domain
- Add additional assertions

### 2. Use Type Hints
Functions with type hints get better tests:
```python
# Good - generates int-specific tests
def calculate(x: int, y: int) -> int:
    return x + y

# Less specific tests generated
def calculate(x, y):
    return x + y
```

### 3. Document Exceptions
Use docstrings or type hints to document exceptions:
```python
def parse_data(data: str) -> dict:
    """Parse JSON data

    Raises:
        ValueError: If data is invalid JSON
    """
    ...
```

### 4. Iterative Refinement
1. Generate initial tests
2. Run them: `pytest test_mymodule.py`
3. Review failures
4. Refine tests or code
5. Repeat

## Limitations

- **No execution**: Test-Guardian analyzes code statically, it doesn't execute it. You need to fill in expected values.
- **Simple heuristics**: Complexity and purity detection use heuristics, not formal analysis.
- **No integration tests**: Focuses on unit tests. Integration tests require manual creation.
- **Placeholder values**: Generated tests use placeholder assertions that need human review.

## Roadmap

- [ ] Coverage gap analysis
- [ ] Integration test generation
- [ ] Actual expected value inference (via execution)
- [ ] Property-based test generation with Hypothesis
- [ ] Parameterized test generation
- [ ] Test data generation from schemas
- [ ] Mutation testing integration
- [ ] CI/CD integration

## Examples

Check out the example files:
- `demo_functions.py` - Sample functions to test
- `example_usage.py` - API usage examples

## Contributing

Test-Guardian is part of the Codes-Masterpiece ecosystem. Contributions welcome!

## License

MIT License - See LICENSE file for details

## Related Tools

- **Universal Debugger**: Fix runtime errors automatically
- **Type-Guardian**: Add type hints automatically
- **Speed-Guardian**: Optimize performance automatically
- **Deploy-Shield**: Validate deployments

---

**Test-Guardian** - Stop writing boilerplate tests, start shipping features! 🧪
