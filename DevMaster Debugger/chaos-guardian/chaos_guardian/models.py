"""Data models for Chaos-Guardian"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum


class ChaosType(Enum):
    """Types of chaos to inject"""
    EXCEPTION = "exception"
    LATENCY = "latency"
    RESOURCE = "resource"
    DATA_CORRUPTION = "data_corruption"
    NETWORK_FAILURE = "network_failure"
    TIMEOUT = "timeout"
    MEMORY_LEAK = "memory_leak"


class InjectionMode(Enum):
    """How chaos is injected"""
    RANDOM = "random"  # Random injection based on probability
    ALWAYS = "always"  # Always inject
    PERIODIC = "periodic"  # Inject periodically
    CONDITIONAL = "conditional"  # Based on condition


@dataclass
class ChaosConfig:
    """Configuration for chaos injection"""
    chaos_type: ChaosType
    probability: float = 0.1  # 10% chance
    enabled: bool = True
    mode: InjectionMode = InjectionMode.RANDOM

    # Type-specific configs
    exception_types: List[type] = field(default_factory=lambda: [Exception])
    latency_ms: int = 1000
    corruption_rate: float = 0.1

    # Safety
    max_failures: Optional[int] = None  # Max failures before stopping
    timeout_seconds: Optional[float] = None


@dataclass
class ChaosEvent:
    """A chaos event that occurred"""
    chaos_type: ChaosType
    timestamp: datetime
    function_name: str
    success: bool  # Did the chaos succeed in breaking something?
    exception: Optional[Exception] = None
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChaosExperiment:
    """A chaos engineering experiment"""
    name: str
    description: str
    target_function: Optional[Callable] = None
    chaos_configs: List[ChaosConfig] = field(default_factory=list)
    duration_seconds: Optional[int] = None
    hypothesis: str = ""  # What do we expect to happen?

    def __post_init__(self):
        if not self.chaos_configs:
            # Default: inject random exceptions
            self.chaos_configs = [
                ChaosConfig(
                    chaos_type=ChaosType.EXCEPTION,
                    probability=0.1
                )
            ]


@dataclass
class ChaosResult:
    """Results from a chaos experiment"""
    experiment: ChaosExperiment
    events: List[ChaosEvent]
    start_time: datetime
    end_time: datetime
    total_invocations: int = 0
    chaos_injected: int = 0
    failures_caught: int = 0
    failures_uncaught: int = 0
    resilience_score: float = 0.0  # 0-100

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary"""
        duration = (self.end_time - self.start_time).total_seconds()

        return {
            'experiment': self.experiment.name,
            'duration_seconds': duration,
            'total_calls': self.total_invocations,
            'chaos_events': self.chaos_injected,
            'failures_caught': self.failures_caught,
            'failures_uncaught': self.failures_uncaught,
            'resilience_score': self.resilience_score,
            'chaos_rate': (self.chaos_injected / self.total_invocations * 100) if self.total_invocations > 0 else 0,
        }


@dataclass
class ResilienceReport:
    """Overall resilience report"""
    target: str
    experiments: List[ChaosResult]
    overall_resilience: float  # 0-100
    weak_points: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
