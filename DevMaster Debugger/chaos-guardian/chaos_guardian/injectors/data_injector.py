"""Data chaos injector - corrupts data randomly"""

import random
from typing import Optional, Any

from ..models import ChaosConfig, ChaosEvent, ChaosType
from datetime import datetime


class DataInjector:
    """Injects data corruption to test validation"""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self.injection_count = 0

    def should_inject(self) -> bool:
        """Determine if chaos should be injected"""
        if not self.config.enabled:
            return False

        if self.config.max_failures and self.injection_count >= self.config.max_failures:
            return False

        return random.random() < self.config.corruption_rate

    def corrupt(self, data: Any, function_name: str) -> tuple[Any, Optional[ChaosEvent]]:
        """Corrupt data randomly"""
        if not self.should_inject():
            return data, None

        self.injection_count += 1

        # Choose corruption strategy based on type
        corrupted_data = self._corrupt_by_type(data)

        event = ChaosEvent(
            chaos_type=ChaosType.DATA_CORRUPTION,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'original_type': type(data).__name__,
                'corruption_type': 'random',
                'injection_count': self.injection_count,
            }
        )

        return corrupted_data, event

    def _corrupt_by_type(self, data: Any) -> Any:
        """Corrupt data based on its type"""
        if data is None:
            return None

        if isinstance(data, str):
            return self._corrupt_string(data)
        elif isinstance(data, int):
            return self._corrupt_int(data)
        elif isinstance(data, float):
            return self._corrupt_float(data)
        elif isinstance(data, list):
            return self._corrupt_list(data)
        elif isinstance(data, dict):
            return self._corrupt_dict(data)
        elif isinstance(data, bool):
            return not data  # Flip boolean
        else:
            return None  # Return None for unknown types

    def _corrupt_string(self, s: str) -> str:
        """Corrupt string data"""
        strategies = [
            lambda: "",  # Empty string
            lambda: None,  # None
            lambda: s[:len(s)//2],  # Truncate
            lambda: s + "CORRUPTED",  # Append garbage
            lambda: "�" * len(s),  # Replace with garbage
        ]
        return random.choice(strategies)()

    def _corrupt_int(self, n: int) -> int:
        """Corrupt integer data"""
        strategies = [
            lambda: 0,
            lambda: -n,
            lambda: n * 1000000,  # Huge number
            lambda: -999999,
            lambda: None,
        ]
        return random.choice(strategies)()

    def _corrupt_float(self, f: float) -> float:
        """Corrupt float data"""
        strategies = [
            lambda: 0.0,
            lambda: float('inf'),
            lambda: float('-inf'),
            lambda: float('nan'),
            lambda: None,
        ]
        return random.choice(strategies)()

    def _corrupt_list(self, lst: list) -> list:
        """Corrupt list data"""
        strategies = [
            lambda: [],  # Empty list
            lambda: None,
            lambda: lst[:1],  # Single element
            lambda: [None] * len(lst),  # All None
        ]
        return random.choice(strategies)()

    def _corrupt_dict(self, d: dict) -> dict:
        """Corrupt dictionary data"""
        strategies = [
            lambda: {},  # Empty dict
            lambda: None,
            lambda: {k: None for k in d},  # All values None
        ]
        return random.choice(strategies)()
