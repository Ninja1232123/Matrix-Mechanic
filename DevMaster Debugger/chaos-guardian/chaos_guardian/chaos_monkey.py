"""Chaos Monkey - Autonomous chaos agent"""

import random
import threading
import time
from typing import List, Callable, Optional
from datetime import datetime

from .models import ChaosConfig, ChaosType, ChaosEvent
from .injectors import (
    ExceptionInjector,
    LatencyInjector,
    ResourceInjector,
    DataInjector,
    NetworkInjector,
)


class ChaosMonkey:
    """
    Autonomous chaos agent that randomly wreaks havoc

    Inspired by Netflix's Chaos Monkey - but for Python!
    """

    def __init__(
        self,
        name: str = "ChaosMonkey",
        aggression: float = 0.1,  # 0.0 (peaceful) to 1.0 (destructive)
        interval_seconds: int = 5,
    ):
        self.name = name
        self.aggression = aggression
        self.interval_seconds = interval_seconds
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.events: List[ChaosEvent] = []
        self.targets: List[Callable] = []

    def add_target(self, func: Callable):
        """Add a function to target for chaos"""
        self.targets.append(func)

    def unleash(self):
        """Unleash the chaos monkey! 🐵💥"""
        if self.running:
            print(f"⚠️  {self.name} is already running!")
            return

        print(f"🐵 {self.name} UNLEASHED! Aggression: {self.aggression*100:.0f}%")
        print(f"   Chaos every {self.interval_seconds}s")
        print(f"   Targets: {len(self.targets)} functions")
        print("   Press Ctrl+C to stop\n")

        self.running = True
        self.thread = threading.Thread(target=self._chaos_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the chaos monkey"""
        print(f"\n🛑 Stopping {self.name}...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=self.interval_seconds + 1)

        print(f"✅ {self.name} stopped")
        print(f"   Total chaos events: {len(self.events)}")

    def _chaos_loop(self):
        """Main chaos loop"""
        while self.running:
            try:
                # Decide if chaos should happen this iteration
                if random.random() < self.aggression:
                    self._inject_chaos()

                # Sleep until next iteration
                time.sleep(self.interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Chaos Monkey error: {e}")

    def _inject_chaos(self):
        """Inject random chaos"""
        if not self.targets:
            return

        # Choose random chaos type
        chaos_types = [
            ChaosType.EXCEPTION,
            ChaosType.LATENCY,
            ChaosType.RESOURCE,
            ChaosType.NETWORK_FAILURE,
        ]

        chaos_type = random.choice(chaos_types)

        # Create event
        event = ChaosEvent(
            chaos_type=chaos_type,
            timestamp=datetime.now(),
            function_name=self.name,
            success=True,
            metadata={
                'monkey': self.name,
                'aggression': self.aggression,
            }
        )

        self.events.append(event)

        # Log chaos
        emoji = {
            ChaosType.EXCEPTION: "💥",
            ChaosType.LATENCY: "🐌",
            ChaosType.RESOURCE: "💾",
            ChaosType.NETWORK_FAILURE: "🌐❌",
        }

        print(
            f"{emoji.get(chaos_type, '🐵')} [{datetime.now().strftime('%H:%M:%S')}] "
            f"{self.name} injected {chaos_type.value}"
        )

    def get_stats(self) -> dict:
        """Get chaos monkey statistics"""
        by_type = {}
        for event in self.events:
            type_name = event.chaos_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            'name': self.name,
            'running': self.running,
            'aggression': self.aggression,
            'total_events': len(self.events),
            'by_type': by_type,
        }


class ChaosArmy:
    """Command multiple chaos monkeys"""

    def __init__(self):
        self.monkeys: List[ChaosMonkey] = []

    def recruit(self, monkey: ChaosMonkey):
        """Add a chaos monkey to the army"""
        self.monkeys.append(monkey)
        print(f"🐵 Recruited {monkey.name} to the chaos army!")

    def unleash_all(self):
        """Unleash all chaos monkeys!"""
        print(f"\n🐵🐵🐵 UNLEASHING {len(self.monkeys)} CHAOS MONKEYS! 🐵🐵🐵\n")

        for monkey in self.monkeys:
            monkey.unleash()
            time.sleep(0.5)  # Stagger starts

    def stop_all(self):
        """Stop all chaos monkeys"""
        for monkey in self.monkeys:
            monkey.stop()

    def get_total_stats(self) -> dict:
        """Get combined statistics"""
        total_events = sum(len(m.events) for m in self.monkeys)
        combined_by_type = {}

        for monkey in self.monkeys:
            stats = monkey.get_stats()
            for chaos_type, count in stats['by_type'].items():
                combined_by_type[chaos_type] = combined_by_type.get(chaos_type, 0) + count

        return {
            'total_monkeys': len(self.monkeys),
            'total_events': total_events,
            'by_type': combined_by_type,
            'monkeys': [m.get_stats() for m in self.monkeys],
        }
