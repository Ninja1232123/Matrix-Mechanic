"""Exception chaos injector - randomly raises exceptions"""

import random
from typing import List, Type, Optional

from ..models import ChaosConfig, ChaosEvent, ChaosType
from datetime import datetime


class ExceptionInjector:
    """Injects random exceptions to test error handling"""

    DEFAULT_EXCEPTIONS = [
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        RuntimeError,
        IOError,
        ConnectionError,
        TimeoutError,
    ]

    def __init__(self, config: ChaosConfig):
        self.config = config
        self.exception_types = config.exception_types or self.DEFAULT_EXCEPTIONS
        self.injection_count = 0

    def should_inject(self) -> bool:
        """Determine if chaos should be injected"""
        if not self.config.enabled:
            return False

        # Check max failures
        if self.config.max_failures and self.injection_count >= self.config.max_failures:
            return False

        return random.random() < self.config.probability

    def inject(self, function_name: str) -> Optional[ChaosEvent]:
        """Inject exception chaos"""
        if not self.should_inject():
            return None

        # Choose random exception
        exception_type = random.choice(self.exception_types)
        exception_msg = f"[CHAOS] Injected {exception_type.__name__} in {function_name}"

        self.injection_count += 1

        # Create chaos event
        event = ChaosEvent(
            chaos_type=ChaosType.EXCEPTION,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            exception=exception_type(exception_msg),
            metadata={
                'exception_type': exception_type.__name__,
                'injection_count': self.injection_count,
            }
        )

        # Raise the exception!
        raise exception_type(exception_msg)

    def get_random_exception(self) -> Exception:
        """Get a random exception"""
        exception_type = random.choice(self.exception_types)
        return exception_type(f"[CHAOS] Random {exception_type.__name__}")
