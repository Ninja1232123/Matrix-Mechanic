"""Disk chaos injector - simulates disk I/O failures"""

import random
import os
import tempfile
from typing import Optional

from ..models import ChaosConfig, ChaosEvent, ChaosType


class DiskInjector:
    """Injects disk I/O chaos - errors, full disk, corruption"""

    def __init__(self, config: ChaosConfig):
        self.config = config
        self.injection_count = 0
        self.temp_files = []  # Track temp files for cleanup

    def should_inject(self) -> bool:
        """Determine if chaos should be injected"""
        if not self.config.enabled:
            return False

        if self.config.max_failures and self.injection_count >= self.config.max_failures:
            return False

        return random.random() < self.config.probability

    def inject(self, function_name: str) -> Optional[ChaosEvent]:
        """Inject disk chaos"""
        if not self.should_inject():
            return None

        self.injection_count += 1

        # Choose random disk chaos scenario
        scenarios = [
            self._disk_full_error,
            self._io_error,
            self._permission_error,
            self._disk_slow,
        ]

        scenario = random.choice(scenarios)
        return scenario(function_name)

    def _disk_full_error(self, function_name: str) -> ChaosEvent:
        """Simulate disk full error"""
        event = ChaosEvent(
            chaos_type=ChaosType.RESOURCE,
            timestamp=None,
            function_name=function_name,
            success=True,
            exception=OSError("[CHAOS] Disk full - No space left on device"),
            metadata={
                'type': 'disk_full',
                'error_code': 'ENOSPC',
            }
        )

        # Raise the error
        raise OSError(28, "[CHAOS] Disk full - No space left on device")

    def _io_error(self, function_name: str) -> ChaosEvent:
        """Simulate I/O error"""
        event = ChaosEvent(
            chaos_type=ChaosType.RESOURCE,
            timestamp=None,
            function_name=function_name,
            success=True,
            exception=IOError("[CHAOS] Input/output error"),
            metadata={
                'type': 'io_error',
                'error_code': 'EIO',
            }
        )

        # Raise the error
        raise IOError(5, "[CHAOS] Input/output error")

    def _permission_error(self, function_name: str) -> ChaosEvent:
        """Simulate permission denied error"""
        event = ChaosEvent(
            chaos_type=ChaosType.RESOURCE,
            timestamp=None,
            function_name=function_name,
            success=True,
            exception=PermissionError("[CHAOS] Permission denied"),
            metadata={
                'type': 'permission_error',
                'error_code': 'EACCES',
            }
        )

        # Raise the error
        raise PermissionError(13, "[CHAOS] Permission denied")

    def _disk_slow(self, function_name: str) -> ChaosEvent:
        """Simulate slow disk I/O"""
        import time

        # Random delay 1-5 seconds
        delay = random.uniform(1, 5)
        time.sleep(delay)

        return ChaosEvent(
            chaos_type=ChaosType.LATENCY,
            timestamp=None,
            function_name=function_name,
            success=True,
            duration_ms=delay * 1000,
            metadata={
                'type': 'disk_slow',
                'delay_seconds': delay,
            }
        )

    def fill_disk(self, size_mb: int = 100):
        """Fill disk with garbage data (for testing)"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(b'0' * (size_mb * 1024 * 1024))
            temp_file.close()
            self.temp_files.append(temp_file.name)
        except:
            pass  # Disk might actually be full

    def cleanup(self):
        """Clean up temp files"""
        for filepath in self.temp_files:
            try:
                os.unlink(filepath)
            except:
                pass
        self.temp_files.clear()
