"""Network chaos injector - simulates network failures"""

import random
from typing import Optional

from ..models import ChaosConfig, ChaosEvent, ChaosType
from datetime import datetime


class NetworkInjector:
    """Injects network failures to test resilience"""

    NETWORK_ERRORS = [
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        TimeoutError,
        OSError,
    ]

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
        """Inject network chaos"""
        if not self.should_inject():
            return None

        self.injection_count += 1

        # Choose random network error
        error_type = random.choice(self.NETWORK_ERRORS)
        error_msg = f"[CHAOS] Simulated network failure in {function_name}"

        event = ChaosEvent(
            chaos_type=ChaosType.NETWORK_FAILURE,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            exception=error_type(error_msg),
            metadata={
                'error_type': error_type.__name__,
                'injection_count': self.injection_count,
            }
        )

        # Raise the network error!
        raise error_type(error_msg)
