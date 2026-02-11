# Contributing to AI Debug Companion

Thank you for your interest in contributing! This project was built collaboratively between human and AI, and we welcome contributions from everyone.

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/ai-debug-companion.git
   cd ai-debug-companion
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

Always run tests before submitting:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_debug_companion --cov-report=html

# Run specific tests
pytest tests/test_parsers.py -v
```

### Code Style

We use Black for formatting and Ruff for linting:

```bash
# Format code
black ai_debug_companion/ tests/

# Check linting
ruff check ai_debug_companion/ tests/

# Fix auto-fixable issues
ruff check --fix ai_debug_companion/
```

### Type Checking

We encourage type hints:

```bash
mypy ai_debug_companion/
```

## Areas for Contribution

### 🌍 Language Support

Add parsers for new programming languages:

1. Create a new parser class in `ai_debug_companion/parsers.py`
2. Inherit from `ErrorParser` base class
3. Implement `can_parse()` and `parse()` methods
4. Add tests in `tests/test_parsers.py`
5. Update language-specific suggestions in `suggestions.py`

Example structure:
```python
class MyLanguageErrorParser(ErrorParser):
    def can_parse(self, text: str) -> bool:
        # Detection logic
        pass

    def parse(self, text: str) -> Optional[ParsedError]:
        # Parsing logic
        pass
```

### 🎨 TUI Improvements

Enhance the user interface in `ai_debug_companion/tui.py`:

- Add new panels or views
- Improve keyboard navigation
- Add color themes
- Enhance error visualization

### 🧠 Suggestion Engine

Improve suggestion quality in `ai_debug_companion/suggestions.py`:

- Add more common error patterns
- Enhance similarity algorithms
- Add confidence score tuning
- Implement machine learning-based suggestions

### 📚 Documentation

- Improve README examples
- Add tutorials or blog posts
- Create video demonstrations
- Document edge cases

## Submitting Changes

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**
   - Describe what your changes do
   - Link any related issues
   - Include screenshots for UI changes
   - Ensure tests pass

## Commit Message Guidelines

Use clear, descriptive commit messages:

- ✅ `Add Rust error parser with test coverage`
- ✅ `Fix stack trace parsing for nested errors`
- ✅ `Improve suggestion confidence scoring`
- ❌ `fix bug`
- ❌ `update code`

## Code Review Process

1. Maintainers will review your PR
2. Address any feedback
3. Once approved, your PR will be merged
4. You'll be added to the contributors list!

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build something useful together.

---

Thank you for contributing! 🎉
