"""Latency chaos injector - adds random delays"""

import time
import random
from typing import Optional

from ..models import ChaosConfig, ChaosEvent, ChaosType
from datetime import datetime


class LatencyInjector:
    """Injects random latency to test timeout handling"""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self.injection_count = 0

    def should_inject(self) -> bool:
        """Determine if chaos should be injected"""
        if not self.config.enabled:
            return False

        if self.config.max_failures and self.injection_count >= self.config.max_failures:
            return False

        return random.random() < self.config.probability

    def inject(self, function_name: str) -> Optional[ChaosEvent]:
        """Inject latency chaos"""
        if not self.should_inject():
            return None

        # Random latency between 50% and 150% of configured latency
        base_latency = self.config.latency_ms
        actual_latency = random.uniform(base_latency * 0.5, base_latency * 1.5)

        self.injection_count += 1

        # Sleep!
        time.sleep(actual_latency / 1000)

        # Create chaos event
        event = ChaosEvent(
            chaos_type=ChaosType.LATENCY,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            duration_ms=actual_latency,
            metadata={
                'latency_ms': actual_latency,
                'configured_latency_ms': base_latency,
                'injection_count': self.injection_count,
            }
        )

        return event
