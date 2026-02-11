"""
Advanced Chaos-Guardian demo with experiments
Shows structured resilience testing
"""

from chaos_guardian import ChaosGuardian, ChaosExperiment, ChaosConfig, ChaosType
import time


# Simulate a microservice
class PaymentService:
    """Simulated payment service"""

    def __init__(self):
        self.retries = 3
        self.cache = {}

    def charge_card(self, user_id, amount):
        """Charge a credit card with retry logic"""
        for attempt in range(self.retries):
            try:
                # Simulate API call
                result = self._call_payment_api(user_id, amount)
                return result
            except ConnectionError:
                if attempt == self.retries - 1:
                    # Final attempt - use cache or fail gracefully
                    return self._fallback_payment(user_id, amount)
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

    def _call_payment_api(self, user_id, amount):
        """Simulated API call"""
        # This will be wrapped with chaos
        return {"status": "success", "amount": amount}

    def _fallback_payment(self, user_id, amount):
        """Fallback when API fails"""
        # Queue for later processing
        return {"status": "queued", "amount": amount}


def run_payment_resilience_experiment():
    """Test payment service resilience"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Payment Service Resilience")
    print("=" * 70)

    cg = ChaosGuardian()
    service = PaymentService()

    # Wrap payment API with chaos
    @cg.chaos(network_failure_rate=0.3, latency_ms=1000)
    def chaotic_payment_api(user_id, amount):
        return service._call_payment_api(user_id, amount)

    # Replace method with chaotic version
    service._call_payment_api = chaotic_payment_api

    # Run experiment
    successes = 0
    failures = 0
    fallbacks = 0

    for i in range(50):
        result = service.charge_card(user_id=i, amount=100.0)

        if result["status"] == "success":
            successes += 1
        elif result["status"] == "queued":
            fallbacks += 1
        else:
            failures += 1

    # Results
    print(f"\nResults:")
    print(f"  ✅ Successes: {successes}")
    print(f"  🔄 Fallbacks: {fallbacks}")
    print(f"  ❌ Failures: {failures}")

    resilience_score = ((successes + fallbacks) / 50) * 100
    print(f"\nResilience Score: {resilience_score:.1f}/100")

    if resilience_score >= 90:
        print("💪 EXCELLENT - Service is highly resilient!")
    elif resilience_score >= 75:
        print("👍 GOOD - Service handles most failures")
    else:
        print("⚠️  NEEDS IMPROVEMENT - Add more error handling")


def run_data_validation_experiment():
    """Test data validation resilience"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Data Validation Resilience")
    print("=" * 70)

    cg = ChaosGuardian()

    @cg.chaos(data_corruption_rate=0.25)
    def get_user_data(user_id):
        """Returns user data (might be corrupted)"""
        return {
            "id": user_id,
            "age": 30,
            "email": "user@example.com",
            "balance": 1000.0
        }

    def validate_user_data(data):
        """Validate user data"""
        if data is None:
            raise ValueError("Data is None")

        if not isinstance(data, dict):
            raise TypeError("Data must be dict")

        # Validate fields
        if "id" not in data or data["id"] is None:
            raise ValueError("Missing or invalid ID")

        if "age" in data and (data["age"] is None or data["age"] < 0):
            raise ValueError("Invalid age")

        if "balance" in data and (data["balance"] is None or data["balance"] < 0):
            raise ValueError("Invalid balance")

        return True

    # Run experiment
    valid_data = 0
    caught_corruption = 0

    for i in range(40):
        data = get_user_data(i)

        try:
            validate_user_data(data)
            valid_data += 1
        except (ValueError, TypeError) as e:
            caught_corruption += 1
            print(f"  🛡️  Caught corrupted data: {e}")

    # Results
    print(f"\nResults:")
    print(f"  ✅ Valid data: {valid_data}")
    print(f"  🛡️  Corruptions caught: {caught_corruption}")

    if caught_corruption > 0:
        print("\n💪 Validation is working! Corrupted data was caught.")
    else:
        print("\n⚠️  No corruptions caught - either lucky or chaos rate too low")


def run_timeout_experiment():
    """Test timeout handling"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Timeout Handling")
    print("=" * 70)

    cg = ChaosGuardian()

    @cg.chaos(latency_ms=3000)  # 3s delays
    def slow_database_query():
        """Simulated slow database query"""
        return {"results": [1, 2, 3]}

    def query_with_timeout(timeout_seconds=2.0):
        """Query with timeout"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Query timeout")

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            result = slow_database_query()
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            signal.alarm(0)
            raise

    # Run experiment
    timeouts = 0
    successes = 0

    for i in range(20):
        try:
            result = query_with_timeout(timeout_seconds=2.0)
            successes += 1
        except TimeoutError:
            timeouts += 1
            print(f"  ⏱️  Query {i+1} timed out")

    # Results
    print(f"\nResults:")
    print(f"  ✅ Completed: {successes}")
    print(f"  ⏱️  Timeouts: {timeouts}")

    if timeouts > 0:
        print("\n✅ Timeout handling is working!")
    else:
        print("\n💡 No timeouts triggered - increase latency or decrease timeout")


def main():
    print("=" * 70)
    print("🌪️  CHAOS-GUARDIAN RESILIENCE EXPERIMENTS")
    print("=" * 70)
    print("\nThese experiments test your code's ability to handle:")
    print("• Network failures")
    print("• Data corruption")
    print("• Latency and timeouts")
    print("\nRunning experiments...\n")

    # Run all experiments
    run_payment_resilience_experiment()
    run_data_validation_experiment()
    run_timeout_experiment()

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("1. Retry logic with exponential backoff prevents cascading failures")
    print("2. Data validation catches corrupted input before it causes damage")
    print("3. Timeouts prevent indefinite waits")
    print("4. Fallback mechanisms provide graceful degradation")
    print("\n🎯 Your code is now battle-tested against chaos!")


if __name__ == '__main__':
    main()
