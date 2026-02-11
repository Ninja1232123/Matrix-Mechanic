# Chaos-Guardian 🌪️

> **Break things intentionally. Build resilience unintentionally.**

Chaos-Guardian brings chaos engineering to Python. Test your code's resilience by intentionally injecting failures, latency, data corruption, and resource exhaustion.

## Why Chaos Engineering?

**Production WILL break.** The question is: will your code handle it gracefully?

Chaos-Guardian helps you find out *before* your users do.

## Features

### 🔥 Chaos Types

1. **Exception Chaos** - Random exceptions to test error handling
2. **Latency Chaos** - Delays to test timeout handling
3. **Resource Chaos** - Memory/CPU pressure to test limits
4. **Data Chaos** - Corruption to test validation
5. **Network Chaos** - Connection failures to test retry logic

### 💥 EXTREME MODE (NEW!)

6. **Time Chaos** ⏰ - Clock skew, time travel, date corruption
7. **Disk Chaos** 💿 - I/O errors, disk full, permission denied
8. **Chaos Monkeys** 🐵 - Autonomous chaos agents that run continuously
9. **Chaos Army** 🐵🐵🐵 - Deploy multiple chaos monkeys simultaneously

### 🎯 Usage Modes

- **@chaos decorator** - Inject chaos into functions
- **Context manager** - Chaos for code blocks
- **Experiments** - Structured resilience testing
- **CLI** - Quick chaos tests

### 🛡️ Safety Features

- **Kill switch** - Emergency stop
- **Max failures** - Limit chaos injection
- **Production guards** - Disable in prod
- **Configurable rates** - Control chaos probability

## Installation

```bash
cd chaos-guardian
pip install -e .
```

## Quick Start

### 1. Decorator Mode

```python
from chaos_guardian import chaos

@chaos(exception_rate=0.1, latency_ms=1000)
def fetch_user_data(user_id):
    # Your code here
    return {"id": user_id, "name": "Alice"}

# 10% chance of exception, occasional 1s latency
result = fetch_user_data(123)
```

### 2. Context Manager

```python
from chaos_guardian import ChaosGuardian

cg = ChaosGuardian()

with cg.chaos_mode(exception_rate=0.2, latency_ms=500):
    # Chaos is active in this block
    process_orders()
    send_notifications()
```

### 3. Experiments

```python
from chaos_guardian import ChaosGuardian, ChaosExperiment, ChaosConfig, ChaosType

cg = ChaosGuardian()

# Define experiment
experiment = ChaosExperiment(
    name="Database Resilience Test",
    description="Test if app handles DB failures gracefully",
    hypothesis="App should retry and use cache on DB failure",
    target_function=fetch_from_database,
    chaos_configs=[
        ChaosConfig(chaos_type=ChaosType.EXCEPTION, probability=0.3),
        ChaosConfig(chaos_type=ChaosType.LATENCY, latency_ms=2000),
    ]
)

# Run experiment
result = cg.run_experiment(experiment, iterations=100)

print(f"Resilience Score: {result.resilience_score}/100")
print(f"Failures Caught: {result.failures_caught}")
print(f"Failures Uncaught: {result.failures_uncaught}")
```

### 4. CLI

```bash
# Quick test
chaos-guardian test --exception-rate 0.15 --iterations 200

# View stats
chaos-guardian stats
```

## Examples

### Example 1: Testing Error Handling

**Before Chaos-Guardian:**
```python
def fetch_user(user_id):
    response = requests.get(f"/api/users/{user_id}")
    return response.json()
```

**What if the network fails?** 💥 Your app crashes!

**With Chaos-Guardian:**
```python
@chaos(network_failure_rate=0.2)
def fetch_user(user_id):
    try:
        response = requests.get(f"/api/users/{user_id}")
        return response.json()
    except ConnectionError:
        # Graceful fallback
        return get_cached_user(user_id)
```

**Result:** 20% of calls fail, but app handles it gracefully! ✅

### Example 2: Testing Timeout Handling

```python
@chaos(latency_ms=3000)  # Inject 3s delays
def slow_database_query():
    # Does your code timeout properly?
    conn.execute(query, timeout=2.0)
```

If your timeout logic works, chaos won't break you! If not, you'll find out! 🔥

### Example 3: Testing Data Validation

```python
@chaos(data_corruption_rate=0.1)
def process_payment(amount):
    # What if amount becomes None? Negative? Huge?
    if amount is None or amount <= 0:
        raise ValueError("Invalid amount")

    return charge_card(amount)
```

Chaos-Guardian corrupts the data 10% of the time. Does your validation catch it? 🎯

### Example 4: Complete Resilience Test

```python
from chaos_guardian import ChaosGuardian, chaos

cg = ChaosGuardian()

@chaos(
    exception_rate=0.1,
    latency_ms=1000,
    data_corruption_rate=0.05,
    network_failure_rate=0.1
)
def critical_operation():
    """This function MUST be resilient"""
    try:
        data = fetch_from_api()
        validate(data)
        result = process(data)
        save_to_database(result)
        return result
    except Exception as e:
        log_error(e)
        return fallback_value()

# Test resilience
for i in range(100):
    try:
        critical_operation()
        print(f"✅ Run {i+1} succeeded")
    except Exception as e:
        print(f"❌ Run {i+1} failed: {e}")

# Check stats
stats = cg.get_stats()
print(f"\nChaos events injected: {stats['total_chaos_events']}")
```

## Chaos Types Explained

### Exception Chaos

**What:** Randomly raises exceptions

**Tests:** Error handling, try/except blocks, logging

**Example:**
```python
@chaos(exception_rate=0.2)  # 20% exception rate
def risky_operation():
    # Might raise ValueError, TypeError, KeyError, etc.
    pass
```

### Latency Chaos

**What:** Adds random delays

**Tests:** Timeout handling, user experience, async patterns

**Example:**
```python
@chaos(latency_ms=2000)  # Up to 2s delay
def fetch_data():
    # Tests if your timeouts work
    pass
```

### Resource Chaos

**What:** Simulates memory/CPU pressure

**Tests:** Resource limits, memory leaks, performance degradation

**Example:**
```python
@chaos(resource_chaos_rate=0.1)
def memory_intensive():
    # Might trigger MemoryError
    pass
```

### Data Chaos

**What:** Corrupts return values

**Tests:** Input validation, data sanitization, type checking

**Example:**
```python
@chaos(data_corruption_rate=0.15)
def get_user_age():
    # Might return None, -999999, or corrupted data
    return user.age
```

### Network Chaos

**What:** Simulates connection failures

**Tests:** Retry logic, circuit breakers, fallbacks

**Example:**
```python
@chaos(network_failure_rate=0.25)
def api_call():
    # Might raise ConnectionError, TimeoutError
    pass
```

## Safety Features

### Kill Switch

```python
cg = ChaosGuardian()

# Enable chaos
@cg.chaos(exception_rate=0.5)
def dangerous_function():
    pass

# Emergency stop!
cg.kill_switch()

# All chaos disabled
```

### Max Failures

```python
@chaos(exception_rate=0.5, max_failures=10)
def test_function():
    # Only inject chaos 10 times, then stop
    pass
```

### Production Guard

```python
import os

cg = ChaosGuardian()

# Disable in production
if os.getenv('ENV') == 'production':
    cg.enabled = False
```

## Configuration

```python
from chaos_guardian import ChaosConfig, ChaosType

config = ChaosConfig(
    chaos_type=ChaosType.EXCEPTION,
    probability=0.2,  # 20% chance
    enabled=True,
    max_failures=50,  # Stop after 50 failures
    exception_types=[ValueError, TypeError],  # Specific exceptions
)
```

## Best Practices

### 1. Start Small

```python
# Start with low rates
@chaos(exception_rate=0.05)  # 5%
```

### 2. Test in Staging

**Never run chaos in production without explicit safeguards!**

### 3. Combine with Testing

```python
def test_user_service_resilience():
    """Pytest test with chaos"""
    @chaos(exception_rate=0.3)
    def test_function():
        result = user_service.get_user(123)
        assert result is not None

    # Run multiple times
    for _ in range(50):
        test_function()
```

### 4. Measure Resilience

Track your resilience score over time:
- < 50: 🔥 Critical - Fix immediately
- 50-75: ⚠️ Warning - Needs improvement
- 75-90: 👍 Good - Minor issues
- 90+: 💪 Excellent - Production ready

## CLI Commands

```bash
# Run chaos test
chaos-guardian test \\
    --exception-rate 0.15 \\
    --latency-ms 1000 \\
    --iterations 200

# View statistics
chaos-guardian stats
```

## Integration

### With pytest

```python
import pytest
from chaos_guardian import chaos

@chaos(exception_rate=0.2)
def test_user_api_resilience():
    """Test that API handles failures gracefully"""
    try:
        result = api.get_user(123)
        assert result is not None
    except Exception:
        # Expected - test error handling
        pass
```

### With CI/CD

```yaml
# .github/workflows/chaos-test.yml
- name: Chaos Resilience Test
  run: |
    pip install -e chaos-guardian
    chaos-guardian test --iterations 500
```

## Real-World Scenarios

### Scenario 1: Microservice Communication

```python
@chaos(network_failure_rate=0.2, latency_ms=5000)
def call_payment_service(order_id):
    """Payment service might be slow or down"""
    try:
        response = requests.post(
            f"{PAYMENT_SERVICE_URL}/charge",
            json={"order_id": order_id},
            timeout=3.0  # Chaos tests this!
        )
        return response.json()
    except (ConnectionError, TimeoutError):
        # Fallback: queue for retry
        payment_queue.add(order_id)
        return {"status": "queued"}
```

### Scenario 2: Database Operations

```python
@chaos(exception_rate=0.1, latency_ms=2000)
def save_user(user_data):
    """DB might fail or be slow"""
    retries = 3
    for attempt in range(retries):
        try:
            db.users.insert(user_data)
            return True
        except Exception as e:
            if attempt == retries - 1:
                # Final attempt failed
                log.error(f"Failed to save user: {e}")
                return False
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Scenario 3: External API Calls

```python
@chaos(network_failure_rate=0.15, data_corruption_rate=0.1)
def fetch_weather(city):
    """External API might fail or return bad data"""
    try:
        response = weather_api.get(city)

        # Validate response (chaos corrupts data)
        if not response or 'temperature' not in response:
            raise ValueError("Invalid weather data")

        return response
    except Exception as e:
        # Return cached weather or default
        return get_cached_weather(city) or {"temperature": None}
```

## EXTREME Chaos Features 💥

### Time Chaos ⏰

Test how your code handles time-related issues:

```python
from chaos_guardian.injectors.time_injector import TimeInjector
from chaos_guardian.models import ChaosConfig, ChaosType

config = ChaosConfig(chaos_type=ChaosType.TIMEOUT, probability=0.3)
time_injector = TimeInjector(config)

# Inject clock skew
time_injector._clock_skew("my_function")

# Time travel to the future
time_injector._time_travel_future("check_expiry")
```

**Tests:**
- Timestamp validation
- JWT token expiry
- Rate limiting with time windows
- Distributed system clock sync

### Disk Chaos 💿

Simulate disk failures:

```python
from chaos_guardian.injectors.disk_injector import DiskInjector

config = ChaosConfig(chaos_type=ChaosType.RESOURCE, probability=0.2)
disk_injector = DiskInjector(config)

# Causes random I/O errors, permission denied, disk full
disk_injector.inject("save_file")
```

**Tests:**
- File operation error handling
- Disk full scenarios
- Permission errors
- Slow disk I/O

### Chaos Monkey 🐵

Autonomous agent that continuously injects chaos:

```python
from chaos_guardian.chaos_monkey import ChaosMonkey

# Create chaos monkey
monkey = ChaosMonkey(
    name="Kong",
    aggression=0.3,  # 30% chaos rate
    interval_seconds=5,
)

# Add targets
monkey.add_target(my_critical_function)

# Unleash the monkey!
monkey.unleash()

# Let it run...
time.sleep(60)

# Stop when done
monkey.stop()

# View stats
stats = monkey.get_stats()
print(f"Chaos events: {stats['total_events']}")
```

**Use Cases:**
- Continuous chaos testing
- Long-running resilience tests
- Soak testing with chaos
- Production chaos (with extreme caution!)

### Chaos Army 🐵🐵🐵

Deploy multiple chaos monkeys:

```python
from chaos_guardian.chaos_monkey import ChaosMonkey, ChaosArmy

army = ChaosArmy()

# Recruit monkeys with different behaviors
army.recruit(ChaosMonkey("Alpha", aggression=0.2, interval_seconds=5))
army.recruit(ChaosMonkey("Bravo", aggression=0.4, interval_seconds=3))
army.recruit(ChaosMonkey("Charlie", aggression=0.6, interval_seconds=2))

# Unleash the army!
army.unleash_all()

# Stop all monkeys
army.stop_all()

# Combined stats
stats = army.get_total_stats()
```

### EXTREME Demo

See all extreme features in action:

```bash
python demo_extreme.py
```

This demonstrates:
- Time travel breaking timestamp logic
- Disk failures causing I/O errors
- Chaos monkeys wreaking continuous havoc
- ALL chaos types combined simultaneously!

## Roadmap

- [x] Time-travel chaos (clock skew) ✅
- [x] Disk I/O chaos ✅
- [x] Autonomous chaos monkeys ✅
- [ ] Distributed chaos (multiple processes)
- [ ] Kubernetes integration
- [ ] Grafana dashboards
- [ ] Chaos scheduling (automated tests)

## Philosophy

> **"Everything fails, all the time."** - Werner Vogels, Amazon CTO

Chaos-Guardian embraces this reality. By breaking things in controlled ways, you build systems that survive real-world chaos.

## Related Tools

- **Universal Debugger**: Fix bugs
- **Type-Guardian**: Add type safety
- **Test-Guardian**: Generate tests
- **Speed-Guardian**: Optimize performance
- **Security-Guardian**: Find vulnerabilities
- **Deploy-Shield**: Validate deployments

---

**Chaos-Guardian** - Break things to build better things! 🌪️💪
