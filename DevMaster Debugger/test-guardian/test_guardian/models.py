"""Data models for Test-Guardian"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class TestType(Enum):
    """Types of test cases"""
    NORMAL = "normal"  # Happy path
    EDGE = "edge"  # Edge cases (empty, None, boundaries)
    ERROR = "error"  # Exception cases
    INTEGRATION = "integration"  # Integration tests
    PROPERTY = "property"  # Property-based tests


class TestFramework(Enum):
    """Supported test frameworks"""
    PYTEST = "pytest"
    UNITTEST = "unittest"


@dataclass
class TestInput:
    """Represents a test input"""
    name: str
    value: Any
    type_hint: Optional[str] = None
    description: str = ""


@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    function_name: str
    test_type: TestType
    inputs: List[TestInput]
    expected_output: Any
    expected_exception: Optional[str] = None
    description: str = ""
    requires_mock: bool = False
    requires_fixture: bool = False
    confidence: float = 1.0  # How confident we are this test is correct


@dataclass
class Fixture:
    """Represents a pytest fixture or unittest setUp"""
    name: str
    return_type: str
    code: str
    scope: str = "function"  # function, class, module, session
    dependencies: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class MockSpec:
    """Specification for a mock object"""
    target: str  # What to mock (module.Class or module.function)
    mock_type: str  # "function", "class", "method", "attribute"
    return_value: Any = None
    side_effect: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class TestSuite:
    """A complete test suite for a module/class/function"""
    target: str  # Module, class, or function name
    file_path: str
    test_cases: List[TestCase]
    fixtures: List[Fixture]
    mocks: List[MockSpec]
    imports: Set[str] = field(default_factory=set)
    framework: TestFramework = TestFramework.PYTEST

    def get_test_count(self) -> int:
        """Get total number of test cases"""
        return len(self.test_cases)

    def get_coverage_estimate(self) -> float:
        """Estimate test coverage (rough heuristic)"""
        # More test types = better coverage
        test_types = set(tc.test_type for tc in self.test_cases)
        type_coverage = len(test_types) / len(TestType) * 100

        # More test cases generally means better coverage
        case_count_score = min(len(self.test_cases) / 10, 1.0) * 100

        return (type_coverage + case_count_score) / 2


@dataclass
class FunctionAnalysis:
    """Analysis result for a function"""
    name: str
    file_path: str
    line_number: int
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    raises: List[str]  # Exceptions that can be raised
    calls_external: bool  # Calls external APIs/databases
    is_pure: bool  # No side effects
    complexity: int  # Cyclomatic complexity
    has_tests: bool = False
    existing_test_count: int = 0


@dataclass
class CoverageGap:
    """Represents a gap in test coverage"""
    function_name: str
    file_path: str
    line_numbers: List[int]
    gap_type: str  # "uncovered_lines", "missing_edge_cases", "missing_error_cases"
    severity: str  # "critical", "high", "medium", "low"
    description: str
    suggested_tests: List[str] = field(default_factory=list)


@dataclass
class TestGenerationReport:
    """Report of test generation results"""
    target: str
    test_suite: TestSuite
    coverage_gaps: List[CoverageGap]
    generated_file: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def get_summary_text(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Target: {self.target}")
        lines.append(f"Tests Generated: {self.test_suite.get_test_count()}")
        lines.append(f"Fixtures: {len(self.test_suite.fixtures)}")
        lines.append(f"Mocks: {len(self.test_suite.mocks)}")
        lines.append(f"Estimated Coverage: {self.test_suite.get_coverage_estimate():.1f}%")
        if self.coverage_gaps:
            lines.append(f"Coverage Gaps: {len(self.coverage_gaps)}")
        return "\n".join(lines)
