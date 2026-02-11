"""Fixture generator - generates pytest fixtures"""

from typing import List, Dict, Any
from ..models import Fixture, FunctionAnalysis


class FixtureGenerator:
    """Generates pytest fixtures for test data"""

    def generate_fixtures(
        self,
        func_analysis: FunctionAnalysis,
        test_cases: List[Any]
    ) -> List[Fixture]:
        """Generate fixtures for a function's tests"""
        fixtures = []

        # Generate fixtures for complex parameters
        for param in func_analysis.parameters:
            type_hint = param.get('type_hint', '')

            if self._needs_fixture(type_hint):
                fixture = self._create_fixture(param)
                if fixture:
                    fixtures.append(fixture)

        return fixtures

    def _needs_fixture(self, type_hint: str) -> bool:
        """Determine if a type needs a fixture"""
        if not type_hint:
            return False

        complex_types = ['dict', 'list', 'dataclass', 'object']
        return any(t in type_hint.lower() for t in complex_types)

    def _create_fixture(self, param: Dict[str, Any]) -> Fixture:
        """Create a fixture for a parameter"""
        name = f"{param['name']}_fixture"
        type_hint = param.get('type_hint', 'Any')

        # Generate fixture code
        code = self._generate_fixture_code(param)

        return Fixture(
            name=name,
            return_type=type_hint,
            code=code,
            scope="function",
            description=f"Fixture for {param['name']}"
        )

    def _generate_fixture_code(self, param: Dict[str, Any]) -> str:
        """Generate the fixture function code"""
        name = param['name']
        type_hint = param.get('type_hint', 'Any')

        if 'dict' in type_hint.lower():
            value = "{'key': 'value', 'number': 42}"
        elif 'list' in type_hint.lower():
            value = "[1, 2, 3, 4, 5]"
        else:
            value = "test_data"

        code = f'''@pytest.fixture
def {name}_fixture():
    """Fixture for {name}"""
    return {value}
'''
        return code
