"""
Demo script with intentional performance issues
This script demonstrates various performance problems that Speed-Guardian can detect
"""

import time
import random


def inefficient_fibonacci(n):
    """Fibonacci without caching - exponential time complexity"""
    if n <= 1:
        return n
    return inefficient_fibonacci(n - 1) + inefficient_fibonacci(n - 2)


def list_append_loop(n):
    """List append in loop instead of list comprehension"""
    result = []
    for i in range(n):
        result.append(i * 2)
    return result


def nested_loops_search(list1, list2):
    """Nested loops causing O(n²) complexity"""
    matches = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                matches.append(item1)
    return matches


def string_concat_loop(n):
    """String concatenation in loop - inefficient"""
    result = ""
    for i in range(n):
        result += str(i) + ","
    return result


def repeated_dict_lookup(data, keys):
    """Repeated dictionary lookups without caching"""
    results = []
    for key in keys:
        # Looking up 'config' multiple times
        if data.get('config'):
            if data['config'].get('enabled'):
                if key in data['config']:
                    results.append(data['config'][key])
    return results


def slow_filter(numbers):
    """Multiple loops for filtering - can be combined"""
    # First loop: filter evens
    evens = []
    for n in numbers:
        if n % 2 == 0:
            evens.append(n)

    # Second loop: filter > 10
    large = []
    for n in evens:
        if n > 10:
            large.append(n)

    # Third loop: square them
    squared = []
    for n in large:
        squared.append(n * n)

    return squared


def global_lookup_in_loop():
    """Global variable lookup in tight loop"""
    total = 0
    for i in range(10000):
        total += len(globals())  # Expensive global lookup every iteration
    return total


def unnecessary_copy(data):
    """Unnecessary list copying"""
    # Creates unnecessary copies
    temp1 = data[:]
    temp2 = temp1[:]
    temp3 = temp2[:]

    result = []
    for item in temp3:
        result.append(item * 2)

    return result


def sync_io_loop(filenames):
    """Synchronous I/O in loop - should be async"""
    results = []
    for filename in filenames:
        # Simulate file read
        time.sleep(0.01)  # Blocking I/O
        results.append(f"Content of {filename}")
    return results


def memory_heavy_operation(n):
    """Creates unnecessary large intermediate structures"""
    # Creates huge intermediate list
    intermediate = [i * i for i in range(n)]

    # Only needs the sum
    return sum(intermediate)


def main():
    """Run all slow functions"""
    print("Running slow demo...")

    # 1. Fibonacci without caching
    print("1. Fibonacci (exponential time)...")
    result = inefficient_fibonacci(20)
    print(f"   Result: {result}")

    # 2. List append in loop
    print("2. List append in loop...")
    result = list_append_loop(10000)
    print(f"   Result: {len(result)} items")

    # 3. Nested loops
    print("3. Nested loops search...")
    list1 = list(range(100))
    list2 = list(range(50, 150))
    result = nested_loops_search(list1, list2)
    print(f"   Result: {len(result)} matches")

    # 4. String concatenation
    print("4. String concatenation in loop...")
    result = string_concat_loop(1000)
    print(f"   Result: {len(result)} chars")

    # 5. Repeated dict lookup
    print("5. Repeated dictionary lookups...")
    data = {
        'config': {
            'enabled': True,
            'key1': 'value1',
            'key2': 'value2',
            'key3': 'value3',
        }
    }
    keys = ['key1', 'key2', 'key3'] * 100
    result = repeated_dict_lookup(data, keys)
    print(f"   Result: {len(result)} items")

    # 6. Multiple filter loops
    print("6. Multiple filter loops...")
    numbers = list(range(100))
    result = slow_filter(numbers)
    print(f"   Result: {len(result)} items")

    # 7. Global lookup in loop
    print("7. Global lookup in loop...")
    result = global_lookup_in_loop()
    print(f"   Result: {result}")

    # 8. Unnecessary copies
    print("8. Unnecessary list copies...")
    data = list(range(1000))
    result = unnecessary_copy(data)
    print(f"   Result: {len(result)} items")

    # 9. Sync I/O in loop
    print("9. Synchronous I/O in loop...")
    filenames = [f"file{i}.txt" for i in range(10)]
    result = sync_io_loop(filenames)
    print(f"   Result: {len(result)} files")

    # 10. Memory heavy operation
    print("10. Memory heavy operation...")
    result = memory_heavy_operation(100000)
    print(f"    Result: {result}")

    print("\nDemo complete!")
    print("\nTo profile this script, run:")
    print("  speed-guardian profile demo_slow.py")
    print("\nTo optimize this script, run:")
    print("  speed-guardian optimize demo_slow.py --auto-fix --dry-run")


if __name__ == '__main__':
    main()
