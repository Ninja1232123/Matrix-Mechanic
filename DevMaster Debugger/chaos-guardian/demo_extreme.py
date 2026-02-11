"""
🌪️💥🔥 EXTREME CHAOS DEMO 🔥💥🌪️

WARNING: This is NOT production code!
This demo shows EXTREME chaos features.

- Time chaos (clock skew, time travel)
- Disk chaos (I/O failures)
- Chaos Monkeys (autonomous agents)
- Multiple chaos types simultaneously

FOR EDUCATIONAL PURPOSES ONLY!
"""

import time
from datetime import datetime
from chaos_guardian import chaos, ChaosGuardian
from chaos_guardian.chaos_monkey import ChaosMonkey, ChaosArmy
from chaos_guardian.injectors.time_injector import TimeInjector
from chaos_guardian.injectors.disk_injector import DiskInjector
from chaos_guardian.models import ChaosConfig, ChaosType


def demo_time_chaos():
    """Demo 1: Time Chaos - Time travel, clock skew"""
    print("\n" + "=" * 70)
    print("DEMO 1: TIME CHAOS ⏰")
    print("=" * 70)
    print("Testing time-based operations with chaos...")

    config = ChaosConfig(chaos_type=ChaosType.TIMEOUT, probability=0.5)
    time_injector = TimeInjector(config)

    def check_expiry(token_created_at):
        """Check if token is expired"""
        now = datetime.now()
        age_seconds = (now - token_created_at).total_seconds()

        if age_seconds > 3600:  # 1 hour expiry
            return "EXPIRED"
        return "VALID"

    # Normal test
    token_time = datetime.now()
    print(f"\n1. Normal check:")
    print(f"   Token created: {token_time}")
    print(f"   Status: {check_expiry(token_time)}")

    # Time travel to future
    print(f"\n2. Time travel test (simulated):")
    print(f"   [CHAOS] Traveling 2 hours into the future...")
    future_time = datetime.now()
    time_injector._time_travel_future("check_expiry")
    # Simulate checking with future time
    print(f"   Status would be: EXPIRED (if time actually changed)")

    # Clock skew
    print(f"\n3. Clock skew test:")
    event = time_injector._clock_skew("check_expiry")
    print(f"   [CHAOS] Clock skewed by {event.metadata['skew_minutes']:.1f} minutes")
    print(f"   This could cause:")
    print(f"   - Timestamp validation failures")
    print(f"   - JWT token issues")
    print(f"   - Distributed system sync problems")

    print(f"\n💡 Time chaos reveals timestamp handling bugs!")


def demo_disk_chaos():
    """Demo 2: Disk Chaos - I/O failures"""
    print("\n" + "=" * 70)
    print("DEMO 2: DISK CHAOS 💿")
    print("=" * 70)
    print("Testing file operations with disk failures...")

    config = ChaosConfig(chaos_type=ChaosType.RESOURCE, probability=0.4)
    disk_injector = DiskInjector(config)

    def save_data(filename, data):
        """Save data to file (fragile)"""
        with open(filename, 'w') as f:
            f.write(data)
        return "SAVED"

    def save_data_resilient(filename, data):
        """Save data with error handling"""
        try:
            with open(filename, 'w') as f:
                f.write(data)
            return "SAVED"
        except (OSError, IOError, PermissionError) as e:
            print(f"   [HANDLED] Disk error: {type(e).__name__}")
            # Fallback: queue for retry
            return "QUEUED_FOR_RETRY"

    print("\n1. Testing fragile save:")
    for i in range(5):
        try:
            result = save_data(f"/tmp/test_{i}.txt", "data")
            print(f"   ✅ Save {i+1}: {result}")
        except Exception as e:
            print(f"   ❌ Save {i+1} FAILED: {type(e).__name__}")

    print("\n2. Testing resilient save:")
    for i in range(5):
        result = save_data_resilient(f"/tmp/test_{i}.txt", "data")
        if result == "SAVED":
            print(f"   ✅ Save {i+1}: {result}")
        else:
            print(f"   🔄 Save {i+1}: {result}")

    print(f"\n💡 Disk chaos reveals I/O error handling!")


def demo_chaos_monkey():
    """Demo 3: Chaos Monkey - Autonomous chaos agent"""
    print("\n" + "=" * 70)
    print("DEMO 3: CHAOS MONKEY 🐵")
    print("=" * 70)
    print("Unleashing autonomous chaos agent...")

    # Create chaos monkey
    monkey = ChaosMonkey(
        name="Kong",
        aggression=0.3,  # 30% chaos rate
        interval_seconds=2,
    )

    # Target function
    @chaos(exception_rate=0.2, latency_ms=1000)
    def critical_service():
        """Critical service that monkey will attack"""
        return "SUCCESS"

    monkey.add_target(critical_service)

    print("\nUnleashing Kong for 10 seconds...")
    monkey.unleash()

    # Let it run
    try:
        for i in range(5):
            try:
                result = critical_service()
                print(f"   ✅ Service call {i+1}: {result}")
            except Exception as e:
                print(f"   ❌ Service call {i+1} FAILED: {type(e).__name__}")

            time.sleep(2)
    finally:
        monkey.stop()

    # Show stats
    stats = monkey.get_stats()
    print(f"\nKong's Rampage Statistics:")
    print(f"   Total chaos events: {stats['total_events']}")
    print(f"   Chaos by type: {stats['by_type']}")


def demo_chaos_army():
    """Demo 4: Chaos Army - Multiple monkeys!"""
    print("\n" + "=" * 70)
    print("DEMO 4: CHAOS ARMY 🐵🐵🐵")
    print("=" * 70)
    print("Deploying an army of chaos monkeys...")

    army = ChaosArmy()

    # Recruit monkeys
    army.recruit(ChaosMonkey("Alpha", aggression=0.2, interval_seconds=3))
    army.recruit(ChaosMonkey("Bravo", aggression=0.3, interval_seconds=2))
    army.recruit(ChaosMonkey("Charlie", aggression=0.4, interval_seconds=2))

    print(f"\nArmy size: {len(army.monkeys)} monkeys")
    print("Unleashing for 8 seconds...")

    army.unleash_all()

    # Let them wreak havoc
    try:
        time.sleep(8)
    finally:
        army.stop_all()

    # Combined stats
    stats = army.get_total_stats()
    print(f"\nArmy Statistics:")
    print(f"   Total chaos events: {stats['total_events']}")
    print(f"   Combined chaos by type: {stats['by_type']}")


def demo_extreme_combined():
    """Demo 5: ALL CHAOS AT ONCE 💥"""
    print("\n" + "=" * 70)
    print("DEMO 5: EXTREME COMBINED CHAOS 💥🌪️🔥")
    print("=" * 70)
    print("Combining ALL chaos types simultaneously...")

    cg = ChaosGuardian()

    @cg.chaos(
        exception_rate=0.15,
        latency_ms=1500,
        data_corruption_rate=0.1,
        network_failure_rate=0.1,
    )
    def ultra_critical_function():
        """Function under EXTREME chaos"""
        # This function experiences:
        # - Random exceptions
        # - Random latency
        # - Data corruption
        # - Network failures
        return {"status": "success", "data": [1, 2, 3]}

    print("\nRunning under EXTREME chaos (20 iterations)...")
    print("Every type of chaos is active!\n")

    successes = 0
    exceptions = 0
    slow_calls = 0
    corrupted = 0

    for i in range(20):
        start = time.time()

        try:
            result = ultra_critical_function()
            duration = time.time() - start

            if duration > 1.0:
                slow_calls += 1
                print(f"   🐌 Call {i+1}: Slow ({duration:.1f}s)")
            elif result is None or not isinstance(result, dict):
                corrupted += 1
                print(f"   🎲 Call {i+1}: Data corrupted")
            else:
                successes += 1
                print(f"   ✅ Call {i+1}: Success")

        except Exception as e:
            exceptions += 1
            print(f"   💥 Call {i+1}: {type(e).__name__}")

    # Results
    print(f"\nExtreme Chaos Results:")
    print(f"   ✅ Successes: {successes}")
    print(f"   💥 Exceptions: {exceptions}")
    print(f"   🐌 Slow calls: {slow_calls}")
    print(f"   🎲 Corrupted: {corrupted}")

    resilience = (successes / 20) * 100
    print(f"\nResilience Score: {resilience:.1f}/100")

    if resilience >= 50:
        print("💪 Your code survived EXTREME chaos!")
    else:
        print("🔥 EXTREME chaos overwhelmed your code!")


def main():
    print("\n" + "=" * 70)
    print("🌪️💥🔥 CHAOS-GUARDIAN: EXTREME MODE 🔥💥🌪️")
    print("=" * 70)
    print("\nWARNING: This demonstrates EXTREME chaos engineering!")
    print("These features test resilience in EXTREME scenarios.")
    print("\nRunning ALL extreme demos...\n")

    input("Press ENTER to start...")

    # Run all demos
    demo_time_chaos()
    time.sleep(2)

    demo_disk_chaos()
    time.sleep(2)

    demo_chaos_monkey()
    time.sleep(2)

    demo_chaos_army()
    time.sleep(2)

    demo_extreme_combined()

    # Final message
    print("\n" + "=" * 70)
    print("EXTREME CHAOS COMPLETE")
    print("=" * 70)
    print("\n🎯 Key Takeaways:")
    print("1. Time chaos reveals timestamp/expiry bugs")
    print("2. Disk chaos tests I/O error handling")
    print("3. Chaos Monkeys provide continuous chaos")
    print("4. Combined chaos shows real-world resilience")
    print("\n💪 If your code survives EXTREME mode, it's ready for ANYTHING!")
    print("\n⚠️  Remember: Use chaos responsibly!")
    print("    - Never in production without safeguards")
    print("    - Start with low aggression")
    print("    - Always have a kill switch")


if __name__ == '__main__':
    main()
