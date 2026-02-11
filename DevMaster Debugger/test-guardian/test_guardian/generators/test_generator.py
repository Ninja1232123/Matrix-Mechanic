"""Test case generator - generates test cases from function analysis"""

from typing import List, Dict, Any, Optional
import itertools

from ..models import TestCase, TestType, TestInput, FunctionAnalysis
from ..analyzers import TypeAnalyzer


class TestCaseGenerator:
    """Generates test cases for functions"""

    def __init__(self):
        self.type_analyzer = TypeAnalyzer()

    def generate_tests(
        self,
        func_analysis: FunctionAnalysis,
        include_edge_cases: bool = True,
        include_error_cases: bool = True
    ) -> List[TestCase]:
        """Generate all test cases for a function"""
        test_cases = []

        # 1. Normal/happy path tests
        normal_tests = self._generate_normal_tests(func_analysis)
        test_cases.extend(normal_tests)

        # 2. Edge case tests
        if include_edge_cases:
            edge_tests = self._generate_edge_case_tests(func_analysis)
            test_cases.extend(edge_tests)

        # 3. Error/exception tests
        if include_error_cases:
            error_tests = self._generate_error_tests(func_analysis)
            test_cases.extend(error_tests)

        return test_cases

    def _generate_normal_tests(self, func_analysis: FunctionAnalysis) -> List[TestCase]:
        """Generate normal/happy path test cases"""
        test_cases = []

        # Generate reasonable default values
        normal_inputs = []
        for param in func_analysis.parameters:
            test_input = TestInput(
                name=param['name'],
                value=self._generate_normal_value(param),
                type_hint=param.get('type_hint'),
                description=f"Normal value for {param['name']}"
            )
            normal_inputs.append(test_input)

        # Create test case
        test_case = TestCase(
            name=f"test_{func_analysis.name}_normal",
            function_name=func_analysis.name,
            test_type=TestType.NORMAL,
            inputs=normal_inputs,
            expected_output="PLACEHOLDER",  # Needs actual execution or user input
            description=f"Test {func_analysis.name} with normal inputs",
            confidence=0.7  # Medium confidence without execution
        )
        test_cases.append(test_case)

        return test_cases

    def _generate_edge_case_tests(self, func_analysis: FunctionAnalysis) -> List[TestCase]:
        """Generate edge case test cases"""
        test_cases = []

        # For each parameter, generate edge case variants
        for i, param in enumerate(func_analysis.parameters):
            edge_values = self._get_edge_case_values(param)

            for j, edge_value in enumerate(edge_values[:3]):  # Limit to 3 per param
                # Use normal values for other params
                inputs = []
                for k, p in enumerate(func_analysis.parameters):
                    if k == i:
                        # Use edge value for this param
                        test_input = TestInput(
                            name=p['name'],
                            value=edge_value,
                            type_hint=p.get('type_hint'),
                            description=f"Edge case value for {p['name']}"
                        )
                    else:
                        # Use normal value for others
                        test_input = TestInput(
                            name=p['name'],
                            value=self._generate_normal_value(p),
                            type_hint=p.get('type_hint'),
                            description=f"Normal value for {p['name']}"
                        )
                    inputs.append(test_input)

                test_case = TestCase(
                    name=f"test_{func_analysis.name}_edge_{param['name']}_{j}",
                    function_name=func_analysis.name,
                    test_type=TestType.EDGE,
                    inputs=inputs,
                    expected_output="PLACEHOLDER",
                    description=f"Edge case: {param['name']} = {edge_value}",
                    confidence=0.6
                )
                test_cases.append(test_case)

        return test_cases

    def _generate_error_tests(self, func_analysis: FunctionAnalysis) -> List[TestCase]:
        """Generate error/exception test cases"""
        test_cases = []

        # If function raises exceptions, test for them
        for exception in func_analysis.raises:
            # Generate inputs that should trigger this exception
            inputs = []
            for param in func_analysis.parameters:
                # Use potentially problematic values
                test_input = TestInput(
                    name=param['name'],
                    value=None,  # None often triggers exceptions
                    type_hint=param.get('type_hint'),
                    description=f"Value that may raise {exception}"
                )
                inputs.append(test_input)

            test_case = TestCase(
                name=f"test_{func_analysis.name}_raises_{exception.lower()}",
                function_name=func_analysis.name,
                test_type=TestType.ERROR,
                inputs=inputs,
                expected_output=None,
                expected_exception=exception,
                description=f"Test that {func_analysis.name} raises {exception}",
                confidence=0.5  # Lower confidence for exception tests
            )
            test_cases.append(test_case)

        # Test with None values if no type hints suggest it's okay
        for i, param in enumerate(func_analysis.parameters):
            type_hint = param.get('type_hint', '')
            if 'Optional' not in type_hint and type_hint:
                inputs = []
                for k, p in enumerate(func_analysis.parameters):
                    if k == i:
                        test_input = TestInput(
                            name=p['name'],
                            value=None,
                            type_hint=p.get('type_hint'),
                            description="None to test error handling"
                        )
                    else:
                        test_input = TestInput(
                            name=p['name'],
                            value=self._generate_normal_value(p),
                            type_hint=p.get('type_hint'),
                            description="Normal value"
                        )
                    inputs.append(test_input)

                test_case = TestCase(
                    name=f"test_{func_analysis.name}_none_{param['name']}",
                    function_name=func_analysis.name,
                    test_type=TestType.ERROR,
                    inputs=inputs,
                    expected_output=None,
                    expected_exception="TypeError",  # Common for None where not expected
                    description=f"Test with None for {param['name']}",
                    confidence=0.5
                )
                test_cases.append(test_case)

        return test_cases

    def _generate_normal_value(self, param: Dict[str, Any]) -> Any:
        """Generate a reasonable normal value for a parameter"""
        # If has default, use it
        if param.get('has_default'):
            return param.get('default_value', 'DEFAULT')

        # Otherwise, use type hint
        type_hint = param.get('type_hint')

        if not type_hint:
            return "test_value"

        type_lower = type_hint.lower()

        if 'int' in type_lower:
            return 42
        elif 'float' in type_lower:
            return 3.14
        elif 'str' in type_lower:
            return "test"
        elif 'bool' in type_lower:
            return True
        elif 'list' in type_lower:
            return [1, 2, 3]
        elif 'dict' in type_lower:
            return {'key': 'value'}
        elif 'tuple' in type_lower:
            return (1, 2)
        elif 'set' in type_lower:
            return {1, 2, 3}
        else:
            return "test_value"

    def _get_edge_case_values(self, param: Dict[str, Any]) -> List[Any]:
        """Get edge case values for a parameter"""
        type_hint = param.get('type_hint')

        if not type_hint:
            return [None, "", 0]

        type_lower = type_hint.lower()

        if 'int' in type_lower:
            return [0, -1, 999999, -999999]
        elif 'float' in type_lower:
            return [0.0, -1.0, 999.999, -999.999]
        elif 'str' in type_lower:
            return ["", " ", "x" * 1000, "\n\t"]
        elif 'bool' in type_lower:
            return [False, True]
        elif 'list' in type_lower:
            return [[], [1], list(range(1000))]
        elif 'dict' in type_lower:
            return [{}, {'key': 'value'}]
        else:
            return [None]

    def generate_property_tests(
        self,
        func_analysis: FunctionAnalysis
    ) -> List[TestCase]:
        """Generate property-based tests (for Hypothesis)"""
        test_cases = []

        if not func_analysis.is_pure:
            return test_cases  # Property tests work best for pure functions

        # Generate a property test
        test_case = TestCase(
            name=f"test_{func_analysis.name}_property",
            function_name=func_analysis.name,
            test_type=TestType.PROPERTY,
            inputs=[],  # Property tests use strategies, not specific inputs
            expected_output="PROPERTY",  # Special marker
            description=f"Property-based test for {func_analysis.name}",
            confidence=0.8
        )
        test_cases.append(test_case)

        return test_cases
