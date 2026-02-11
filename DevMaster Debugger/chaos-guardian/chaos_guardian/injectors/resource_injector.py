"""Resource chaos injector - simulates resource exhaustion"""

import random
import gc
from typing import Optional

from ..models import ChaosConfig, ChaosEvent, ChaosType
from datetime import datetime


class ResourceInjector:
    """Injects resource exhaustion scenarios"""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self.injection_count = 0
        self.memory_hogs = []  # Track allocated memory

    def should_inject(self) -> bool:
        """Determine if chaos should be injected"""
        if not self.config.enabled:
            return False

        if self.config.max_failures and self.injection_count >= self.config.max_failures:
            return False

        return random.random() < self.config.probability

    def inject(self, function_name: str) -> Optional[ChaosEvent]:
        """Inject resource chaos"""
        if not self.should_inject():
            return None

        chaos_scenarios = [
            self._memory_pressure,
            self._cpu_spike,
            self._resource_error,
        ]

        scenario = random.choice(chaos_scenarios)
        self.injection_count += 1

        return scenario(function_name)

    def _memory_pressure(self, function_name: str) -> ChaosEvent:
        """Simulate memory pressure"""
        # Allocate large chunk of memory
        size_mb = random.randint(10, 100)
        memory_hog = bytearray(size_mb * 1024 * 1024)
        self.memory_hogs.append(memory_hog)

        return ChaosEvent(
            chaos_type=ChaosType.MEMORY_LEAK,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'allocated_mb': size_mb,
                'injection_count': self.injection_count,
            }
        )

    def _cpu_spike(self, function_name: str) -> ChaosEvent:
        """Simulate CPU spike"""
        # Busy loop for short duration
        iterations = random.randint(1000000, 5000000)
        _ = sum(i * i for i in range(iterations))

        return ChaosEvent(
            chaos_type=ChaosType.RESOURCE,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'type': 'cpu_spike',
                'iterations': iterations,
            }
        )

    def _resource_error(self, function_name: str) -> ChaosEvent:
        """Raise resource error"""
        raise MemoryError("[CHAOS] Simulated memory exhaustion")

    def cleanup(self):
        """Clean up allocated resources"""
        self.memory_hogs.clear()
        gc.collect()
