"""
Basic Chaos-Guardian demo
Shows how chaos reveals resilience (or lack thereof)
"""

from chaos_guardian import chaos, ChaosGuardian


# Example 1: Fragile function (no error handling)
@chaos(exception_rate=0.3)
def fragile_function():
    """This will break often!"""
    data = fetch_data()
    return data.upper()  # What if data is None? 💥

def fetch_data():
    return "hello world"


# Example 2: Resilient function (with error handling)
@chaos(exception_rate=0.3)
def resilient_function():
    """This handles chaos gracefully!"""
    try:
        data = fetch_data()
        if data is None:
            return "DEFAULT"
        return data.upper()
    except Exception as e:
        print(f"Handled error: {e}")
        return "ERROR_HANDLED"


# Example 3: Latency chaos
@chaos(latency_ms=2000)
def slow_operation():
    """Tests timeout handling"""
    return "completed"


# Example 4: Data corruption
@chaos(data_corruption_rate=0.2)
def get_user_age():
    """Returns age, but chaos might corrupt it"""
    return 25


def validate_age(age):
    """Validation catches corrupted data"""
    if age is None or age < 0 or age > 150:
        raise ValueError(f"Invalid age: {age}")
    return age


def main():
    print("=" * 60)
    print("Chaos-Guardian Basic Demo")
    print("=" * 60)

    cg = ChaosGuardian()

    # Test 1: Fragile vs Resilient
    print("\n1. FRAGILE FUNCTION TEST")
    print("-" * 40)

    fragile_successes = 0
    fragile_failures = 0

    for i in range(20):
        try:
            result = fragile_function()
            fragile_successes += 1
        except Exception as e:
            fragile_failures += 1
            print(f"  ❌ Iteration {i+1} failed: {type(e).__name__}")

    print(f"\nResults: {fragile_successes} successes, {fragile_failures} failures")
    print(f"Success rate: {(fragile_successes/20)*100:.0f}%")

    # Test 2: Resilient function
    print("\n2. RESILIENT FUNCTION TEST")
    print("-" * 40)

    resilient_successes = 0
    resilient_failures = 0

    for i in range(20):
        try:
            result = resilient_function()
            resilient_successes += 1
        except Exception as e:
            resilient_failures += 1

    print(f"Results: {resilient_successes} successes, {resilient_failures} failures")
    print(f"Success rate: {(resilient_successes/20)*100:.0f}%")

    # Test 3: Latency test
    print("\n3. LATENCY TEST")
    print("-" * 40)
    print("Running slow operation with 2s latency injection...")

    import time
    start = time.time()
    result = slow_operation()
    duration = time.time() - start

    print(f"Completed in {duration:.2f}s")
    if duration > 1.5:
        print("⚠️  Latency detected! Does your code timeout properly?")

    # Test 4: Data validation
    print("\n4. DATA CORRUPTION TEST")
    print("-" * 40)

    valid_count = 0
    invalid_count = 0

    for i in range(20):
        age = get_user_age()
        try:
            validated = validate_age(age)
            valid_count += 1
        except ValueError:
            invalid_count += 1
            print(f"  ⚠️  Caught corrupted age: {age}")

    print(f"\nResults: {valid_count} valid, {invalid_count} corrupted")
    print(f"Validation working: {'✅ YES' if invalid_count > 0 else '❌ NO'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    stats = cg.get_stats()
    print(f"Total chaos events: {stats['total_chaos_events']}")
    print("\nKey Learnings:")
    print("✅ Error handling prevents crashes")
    print("✅ Validation catches corrupted data")
    print("✅ Timeouts protect against slow operations")
    print("\n💡 Chaos reveals weaknesses before production does!")


if __name__ == '__main__':
    main()
