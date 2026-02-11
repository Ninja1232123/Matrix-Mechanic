"""Mock generator - generates mocks for external dependencies"""

from typing import List, Dict, Any
from ..models import MockSpec, FunctionAnalysis


class MockGenerator:
    """Generates mock specifications for external dependencies"""

    def generate_mocks(
        self,
        func_analysis: FunctionAnalysis,
        dependencies: Dict[str, Any]
    ) -> List[MockSpec]:
        """Generate mocks for a function's dependencies"""
        mocks = []

        # Mock file operations
        if dependencies.get('file_operations'):
            for op in dependencies['file_operations']:
                mock = MockSpec(
                    target=f"builtins.{op}",
                    mock_type="function",
                    return_value="mock_file_handle" if op == 'open' else "mock_data",
                    description=f"Mock for {op} operation"
                )
                mocks.append(mock)

        # Mock network calls
        if dependencies.get('network_calls'):
            for call in dependencies['network_calls']:
                mock = MockSpec(
                    target=call,
                    mock_type="method",
                    return_value="{'status': 'success', 'data': 'mock'}",
                    description=f"Mock for network call {call}"
                )
                mocks.append(mock)

        # Mock database calls
        if dependencies.get('database_calls'):
            for call in dependencies['database_calls']:
                mock = MockSpec(
                    target=call,
                    mock_type="method",
                    return_value="[{'id': 1, 'name': 'test'}]",
                    description=f"Mock for database call {call}"
                )
                mocks.append(mock)

        return mocks
