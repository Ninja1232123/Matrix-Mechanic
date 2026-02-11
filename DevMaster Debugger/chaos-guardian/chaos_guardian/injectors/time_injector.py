"""Time chaos injector - manipulates time perception"""

import random
import time
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import patch

from ..models import ChaosConfig, ChaosEvent, ChaosType


class TimeInjector:
    """Injects time-related chaos - clock skew, time travel, date corruption"""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self.injection_count = 0
        self.time_offset = 0  # Seconds offset from real time

    def should_inject(self) -> bool:
        """Determine if chaos should be injected"""
        if not self.config.enabled:
            return False

        if self.config.max_failures and self.injection_count >= self.config.max_failures:
            return False

        return random.random() < self.config.probability

    def inject(self, function_name: str) -> Optional[ChaosEvent]:
        """Inject time chaos"""
        if not self.should_inject():
            return None

        self.injection_count += 1

        # Choose random time chaos scenario
        scenarios = [
            self._clock_skew,
            self._time_travel_past,
            self._time_travel_future,
            self._freeze_time,
            self._date_corruption,
        ]

        scenario = random.choice(scenarios)
        return scenario(function_name)

    def _clock_skew(self, function_name: str) -> ChaosEvent:
        """Simulate clock skew (time drift)"""
        # Random skew between -3600s and +3600s (±1 hour)
        skew_seconds = random.randint(-3600, 3600)
        self.time_offset = skew_seconds

        return ChaosEvent(
            chaos_type=ChaosType.TIMEOUT,  # Using TIMEOUT for time-related
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'type': 'clock_skew',
                'skew_seconds': skew_seconds,
                'skew_minutes': skew_seconds / 60,
            }
        )

    def _time_travel_past(self, function_name: str) -> ChaosEvent:
        """Travel back in time"""
        # Go back 1-365 days
        days_back = random.randint(1, 365)
        self.time_offset = -days_back * 86400

        return ChaosEvent(
            chaos_type=ChaosType.TIMEOUT,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'type': 'time_travel_past',
                'days_back': days_back,
                'target_date': (datetime.now() - timedelta(days=days_back)).isoformat(),
            }
        )

    def _time_travel_future(self, function_name: str) -> ChaosEvent:
        """Travel forward in time"""
        # Go forward 1-365 days
        days_forward = random.randint(1, 365)
        self.time_offset = days_forward * 86400

        return ChaosEvent(
            chaos_type=ChaosType.TIMEOUT,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'type': 'time_travel_future',
                'days_forward': days_forward,
                'target_date': (datetime.now() + timedelta(days=days_forward)).isoformat(),
            }
        )

    def _freeze_time(self, function_name: str) -> ChaosEvent:
        """Freeze time (time.time() returns same value)"""
        frozen_time = time.time()

        # This would require patching time.time() which is complex
        # For now, just record the event

        return ChaosEvent(
            chaos_type=ChaosType.TIMEOUT,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'type': 'freeze_time',
                'frozen_at': frozen_time,
            }
        )

    def _date_corruption(self, function_name: str) -> ChaosEvent:
        """Corrupt date values"""
        # Return chaos dates: epoch, far future, invalid
        chaos_dates = [
            datetime(1970, 1, 1),  # Unix epoch
            datetime(2099, 12, 31),  # Far future
            datetime(1900, 1, 1),  # Far past
        ]

        corrupted_date = random.choice(chaos_dates)

        return ChaosEvent(
            chaos_type=ChaosType.DATA_CORRUPTION,
            timestamp=datetime.now(),
            function_name=function_name,
            success=True,
            metadata={
                'type': 'date_corruption',
                'corrupted_date': corrupted_date.isoformat(),
            }
        )

    def get_current_time(self) -> datetime:
        """Get current time with chaos offset applied"""
        return datetime.now() + timedelta(seconds=self.time_offset)

    def reset_time(self):
        """Reset time offset"""
        self.time_offset = 0
