"""Core Test-Guardian class"""

from typing import List, Optional
from pathlib import Path

from .models import (
    TestSuite,
    TestFramework,
    TestGenerationReport,
    FunctionAnalysis,
)
from .analyzers import CodeAnalyzer, TypeAnalyzer, DependencyAnalyzer
from .generators import TestCaseGenerator, FixtureGenerator, MockGenerator
from .formatters import PytestFormatter, UnittestFormatter


class TestGuardian:
    """Main Test-Guardian class - orchestrates test generation"""

    def __init__(self, framework: TestFramework = TestFramework.PYTEST):
        self.framework = framework
        self.code_analyzer = CodeAnalyzer()
        self.type_analyzer = TypeAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.test_generator = TestCaseGenerator()
        self.fixture_generator = FixtureGenerator()
        self.mock_generator = MockGenerator()

        if framework == TestFramework.PYTEST:
            self.formatter = PytestFormatter()
        else:
            self.formatter = UnittestFormatter()

    def generate_tests_for_file(
        self,
        file_path: Path,
        output_path: Optional[Path] = None
    ) -> TestGenerationReport:
        """Generate tests for all functions in a file"""
        # Analyze all functions
        functions = self.code_analyzer.analyze_file(file_path)

        if not functions:
            raise ValueError(f"No testable functions found in {file_path}")

        # Generate tests for all functions
        all_test_cases = []
        all_fixtures = []
        all_mocks = []

        for func in functions:
            # Skip private/internal functions
            if func.name.startswith('_') and func.complexity < 3:
                continue

            # Generate test cases
            test_cases = self.test_generator.generate_tests(func)
            all_test_cases.extend(test_cases)

            # Generate fixtures
            fixtures = self.fixture_generator.generate_fixtures(func, test_cases)
            all_fixtures.extend(fixtures)

            # Check dependencies and generate mocks
            deps = self.dependency_analyzer.analyze_function_dependencies(
                self._get_ast_node(file_path, func.name)
            )
            if deps['external_calls'] or deps['file_operations']:
                mocks = self.mock_generator.generate_mocks(func, deps)
                all_mocks.extend(mocks)

        # Create test suite
        test_suite = TestSuite(
            target=file_path.stem,
            file_path=str(file_path),
            test_cases=all_test_cases,
            fixtures=all_fixtures,
            mocks=all_mocks,
            framework=self.framework,
        )

        # Determine output path
        if not output_path:
            output_path = Path(f"test_{file_path.stem}.py")

        # Generate test file
        self.formatter.save_test_file(test_suite, output_path)

        # Create report
        report = TestGenerationReport(
            target=str(file_path),
            test_suite=test_suite,
            coverage_gaps=[],  # TODO: Implement coverage analysis
            generated_file=str(output_path),
            summary={
                'tests_generated': len(all_test_cases),
                'fixtures': len(all_fixtures),
                'mocks': len(all_mocks),
                'functions_tested': len(functions),
            }
        )

        return report

    def generate_tests_for_function(
        self,
        file_path: Path,
        function_name: str,
        output_path: Optional[Path] = None
    ) -> TestGenerationReport:
        """Generate tests for a specific function"""
        # Analyze the specific function
        functions = self.code_analyzer.analyze_file(file_path)
        func = next((f for f in functions if f.name == function_name), None)

        if not func:
            raise ValueError(f"Function '{function_name}' not found in {file_path}")

        # Generate test cases
        test_cases = self.test_generator.generate_tests(func)

        # Generate fixtures
        fixtures = self.fixture_generator.generate_fixtures(func, test_cases)

        # Check dependencies and generate mocks
        deps = self.dependency_analyzer.analyze_function_dependencies(
            self._get_ast_node(file_path, func.name)
        )
        mocks = []
        if deps['external_calls'] or deps['file_operations']:
            mocks = self.mock_generator.generate_mocks(func, deps)

        # Create test suite
        test_suite = TestSuite(
            target=function_name,
            file_path=str(file_path),
            test_cases=test_cases,
            fixtures=fixtures,
            mocks=mocks,
            framework=self.framework,
        )

        # Determine output path
        if not output_path:
            output_path = Path(f"test_{function_name}.py")

        # Generate test file
        self.formatter.save_test_file(test_suite, output_path)

        # Create report
        report = TestGenerationReport(
            target=function_name,
            test_suite=test_suite,
            coverage_gaps=[],
            generated_file=str(output_path),
            summary={
                'tests_generated': len(test_cases),
                'fixtures': len(fixtures),
                'mocks': len(mocks),
            }
        )

        return report

    def _get_ast_node(self, file_path: Path, function_name: str):
        """Get AST node for a function"""
        import ast

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node

        return None

    def preview_tests(self, file_path: Path) -> str:
        """Preview what tests would be generated without creating files"""
        # Analyze functions
        functions = self.code_analyzer.analyze_file(file_path)

        lines = []
        lines.append(f"Test Preview for {file_path}")
        lines.append("=" * 60)
        lines.append("")

        for func in functions:
            if func.name.startswith('_') and func.complexity < 3:
                continue

            lines.append(f"Function: {func.name}")
            lines.append(f"  Parameters: {len(func.parameters)}")
            lines.append(f"  Complexity: {func.complexity}")
            lines.append(f"  Raises: {', '.join(func.raises) if func.raises else 'None'}")

            # Generate test cases
            test_cases = self.test_generator.generate_tests(func)

            lines.append(f"  Tests to generate: {len(test_cases)}")
            for tc in test_cases:
                lines.append(f"    - {tc.name} ({tc.test_type.value})")

            lines.append("")

        return "\n".join(lines)
