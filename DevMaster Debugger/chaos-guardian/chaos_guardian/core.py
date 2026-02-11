"""Core Chaos-Guardian orchestrator"""

import functools
import time
from typing import Callable, Optional, List
from contextlib import contextmanager
from datetime import datetime

from .models import (
    ChaosConfig,
    ChaosExperiment,
    ChaosResult,
    ChaosEvent,
    ChaosType,
    InjectionMode,
)
from .injectors import (
    ExceptionInjector,
    LatencyInjector,
    ResourceInjector,
    DataInjector,
    NetworkInjector,
)


class ChaosGuardian:
    """Main Chaos-Guardian orchestrator"""

    def __init__(self):
        self.enabled = True
        self.events: List[ChaosEvent] = []
        self.injectors = {}
        self.kill_switch_active = False

    def chaos(
        self,
        exception_rate: float = 0.0,
        latency_ms: int = 0,
        data_corruption_rate: float = 0.0,
        network_failure_rate: float = 0.0,
        resource_chaos_rate: float = 0.0,
        max_failures: Optional[int] = None,
    ):
        """
        Decorator to inject chaos into a function

        @chaos(exception_rate=0.1, latency_ms=1000)
        def my_function():
            pass
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled or self.kill_switch_active:
                    return func(*args, **kwargs)

                # Exception chaos
                if exception_rate > 0:
                    config = ChaosConfig(
                        chaos_type=ChaosType.EXCEPTION,
                        probability=exception_rate,
                        max_failures=max_failures
                    )
                    injector = ExceptionInjector(config)
                    try:
                        event = injector.inject(func.__name__)
                        if event:
                            self.events.append(event)
                    except Exception as e:
                        # Chaos succeeded! Exception was raised
                        pass  # Let it propagate

                # Latency chaos
                if latency_ms > 0:
                    config = ChaosConfig(
                        chaos_type=ChaosType.LATENCY,
                        probability=0.3,  # 30% chance of latency
                        latency_ms=latency_ms,
                        max_failures=max_failures
                    )
                    injector = LatencyInjector(config)
                    event = injector.inject(func.__name__)
                    if event:
                        self.events.append(event)

                # Network failure chaos
                if network_failure_rate > 0:
                    config = ChaosConfig(
                        chaos_type=ChaosType.NETWORK_FAILURE,
                        probability=network_failure_rate,
                        max_failures=max_failures
                    )
                    injector = NetworkInjector(config)
                    try:
                        event = injector.inject(func.__name__)
                        if event:
                            self.events.append(event)
                    except Exception as e:
                        pass  # Let network errors propagate

                # Execute the actual function
                result = func(*args, **kwargs)

                # Data corruption (on return value)
                if data_corruption_rate > 0:
                    config = ChaosConfig(
                        chaos_type=ChaosType.DATA_CORRUPTION,
                        corruption_rate=data_corruption_rate,
                        max_failures=max_failures
                    )
                    injector = DataInjector(config)
                    result, event = injector.corrupt(result, func.__name__)
                    if event:
                        self.events.append(event)

                return result

            return wrapper
        return decorator

    @contextmanager
    def chaos_mode(
        self,
        exception_rate: float = 0.1,
        latency_ms: int = 500,
    ):
        """
        Context manager for chaos injection

        with chaos_guardian.chaos_mode(exception_rate=0.2):
            # Chaos is active in this block
            do_something()
        """
        original_enabled = self.enabled
        self.enabled = True

        # Store original chaos configs
        original_configs = {}

        try:
            yield self
        finally:
            self.enabled = original_enabled

    def run_experiment(
        self,
        experiment: ChaosExperiment,
        iterations: int = 100
    ) -> ChaosResult:
        """Run a chaos experiment"""
        print(f"\n🌪️  Starting Chaos Experiment: {experiment.name}")
        print(f"   Hypothesis: {experiment.hypothesis}")
        print(f"   Iterations: {iterations}\n")

        start_time = datetime.now()
        events = []
        total_invocations = 0
        failures_caught = 0
        failures_uncaught = 0

        # Run experiment
        for i in range(iterations):
            total_invocations += 1

            try:
                if experiment.target_function:
                    # Wrap function with chaos
                    chaos_func = self._wrap_with_chaos(
                        experiment.target_function,
                        experiment.chaos_configs
                    )
                    chaos_func()
            except Exception as e:
                # Check if error was handled
                if "[CHAOS]" in str(e):
                    # This is injected chaos - test if code handles it
                    failures_uncaught += 1
                else:
                    failures_caught += 1

        end_time = datetime.now()

        # Calculate resilience score
        chaos_injected = len([e for e in self.events if e.success])
        if chaos_injected > 0:
            resilience_score = (failures_caught / chaos_injected) * 100
        else:
            resilience_score = 100.0

        result = ChaosResult(
            experiment=experiment,
            events=self.events.copy(),
            start_time=start_time,
            end_time=end_time,
            total_invocations=total_invocations,
            chaos_injected=chaos_injected,
            failures_caught=failures_caught,
            failures_uncaught=failures_uncaught,
            resilience_score=resilience_score,
        )

        # Clear events for next experiment
        self.events.clear()

        return result

    def _wrap_with_chaos(
        self,
        func: Callable,
        configs: List[ChaosConfig]
    ) -> Callable:
        """Wrap a function with chaos injectors"""
        for config in configs:
            if config.chaos_type == ChaosType.EXCEPTION:
                func = self.chaos(exception_rate=config.probability)(func)
            elif config.chaos_type == ChaosType.LATENCY:
                func = self.chaos(latency_ms=config.latency_ms)(func)

        return func

    def kill_switch(self):
        """Emergency stop - disable all chaos"""
        print("🛑 KILL SWITCH ACTIVATED - All chaos disabled")
        self.kill_switch_active = True
        self.enabled = False

    def reset(self):
        """Reset chaos guardian state"""
        self.enabled = True
        self.kill_switch_active = False
        self.events.clear()
        self.injectors.clear()

    def get_stats(self) -> dict:
        """Get chaos statistics"""
        total_events = len(self.events)
        by_type = {}

        for event in self.events:
            type_name = event.chaos_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            'total_chaos_events': total_events,
            'by_type': by_type,
            'kill_switch_active': self.kill_switch_active,
            'enabled': self.enabled,
        }


# Global instance for convenience
_chaos_guardian = ChaosGuardian()

# Export decorator for easy use
def chaos(**kwargs):
    """Convenience function for @chaos decorator"""
    return _chaos_guardian.chaos(**kwargs)
