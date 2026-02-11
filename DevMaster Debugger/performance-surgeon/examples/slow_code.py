"""
Example Slow Code - Performance Anti-Patterns

This file contains INTENTIONALLY SLOW code to demonstrate
Performance-Surgeon's detection and optimization capabilities.
"""

import re


# CRITICAL: Nested loops with 'in' on list (O(n²))
def find_common_items_slow(list1, list2):
    """Find common items between two lists - SLOW VERSION"""
    common = []
    for item in list1:
        if item in list2:  # O(n) lookup on each iteration!
            common.append(item)
    return common


# HIGH: String concatenation in loop (O(n²))
def build_string_slow(items):
    """Build a string from items - SLOW VERSION"""
    result = ""
    for item in items:
        result += str(item) + ","  # Creates new string each time!
    return result


# MEDIUM: Append in loop instead of list comprehension
def process_items_slow(items):
    """Process items and collect results - SLOW VERSION"""
    results = []
    for item in items:
        results.append(item * 2)
    return results


# CRITICAL: Uncached recursive function (exponential time!)
def fibonacci_slow(n):
    """Calculate fibonacci - SLOW VERSION (no caching)"""
    if n < 2:
        return n
    return fibonacci_slow(n-1) + fibonacci_slow(n-2)  # Exponential!


# HIGH: Repeated regex compilation in loop
def validate_emails_slow(emails):
    """Validate email addresses - SLOW VERSION"""
    valid = []
    for email in emails:
        if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):  # Compiles each time!
            valid.append(email)
    return valid


# HIGH: Loading entire file into memory
def process_large_file_slow(filepath):
    """Process a large file - SLOW VERSION"""
    with open(filepath) as f:
        lines = f.readlines()  # Loads entire file into memory!
        count = 0
        for line in lines:
            if 'ERROR' in line:
                count += 1
    return count


# MEDIUM: List comprehension when generator would work
def sum_squares_slow(n):
    """Sum of squares - MEMORY INEFFICIENT"""
    return sum([x**2 for x in range(n)])  # Creates list in memory!


# LOW: Global lookup in tight loop
def process_with_global_lookup(items):
    """Process items with repeated global lookups"""
    results = []
    for i in range(len(items)):  # len() lookup each iteration
        results.append(items[i] * 2)
    return results


# HIGH: File writes in loop
def write_items_slow(items, filepath):
    """Write items to file - SLOW VERSION"""
    with open(filepath, 'w') as f:
        for item in items:
            f.write(str(item) + '\n')  # Writes to disk each iteration!


# MEDIUM: Unnecessary list copy
def iterate_with_copy(mylist):
    """Iterate with unnecessary copy"""
    temp = mylist[:]  # Unnecessary copy!
    results = []
    for item in temp:
        results.append(item * 2)
    return results


# CRITICAL: N+1 query pattern (simulated)
class FakeUser:
    """Simulated database model"""
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.posts = []  # Lazy loaded

    @staticmethod
    def get_all():
        return [FakeUser(i, f"User{i}") for i in range(100)]


def get_user_posts_slow():
    """Simulate N+1 query problem"""
    users = FakeUser.get_all()
    all_posts = []
    for user in users:
        # Each iteration triggers a database query!
        all_posts.extend(user.posts)  # N+1 queries!
    return all_posts


# ALL OPTIMIZED VERSIONS FOR COMPARISON:

def find_common_items_fast(list1, list2):
    """FAST VERSION - Use set for O(1) lookup"""
    set2 = set(list2)
    return [item for item in list1 if item in set2]


def build_string_fast(items):
    """FAST VERSION - Use join()"""
    return ','.join(str(item) for item in items)


def process_items_fast(items):
    """FAST VERSION - Use list comprehension"""
    return [item * 2 for item in items]


from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_fast(n):
    """FAST VERSION - Cached!"""
    if n < 2:
        return n
    return fibonacci_fast(n-1) + fibonacci_fast(n-2)


def validate_emails_fast(emails):
    """FAST VERSION - Compile regex once"""
    pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    return [email for email in emails if pattern.match(email)]


def process_large_file_fast(filepath):
    """FAST VERSION - Stream lines"""
    count = 0
    with open(filepath) as f:
        for line in f:  # Streams line by line!
            if 'ERROR' in line:
                count += 1
    return count


def sum_squares_fast(n):
    """FAST VERSION - Generator expression"""
    return sum(x**2 for x in range(n))  # No list created!


def process_with_local_cache(items):
    """FAST VERSION - Cache length"""
    n = len(items)  # Cache it!
    return [items[i] * 2 for i in range(n)]


def write_items_fast(items, filepath):
    """FAST VERSION - Batch writes"""
    with open(filepath, 'w') as f:
        f.write('\n'.join(str(item) for item in items))


if __name__ == '__main__':
    # Test slow vs fast
    import time
    
    # Test 1: Find common items
    list1 = list(range(1000))
    list2 = list(range(500, 1500))
    
    start = time.time()
    result_slow = find_common_items_slow(list1, list2)
    slow_time = time.time() - start
    
    start = time.time()
    result_fast = find_common_items_fast(list1, list2)
    fast_time = time.time() - start
    
    print(f"Find common items:")
    print(f"  Slow: {slow_time:.4f}s")
    print(f"  Fast: {fast_time:.4f}s")
    print(f"  Speedup: {slow_time / fast_time:.1f}x")
    
    # Test 2: Fibonacci
    print(f"\nFibonacci(30):")
    start = time.time()
    result_slow = fibonacci_slow(30)
    slow_time = time.time() - start
    print(f"  Slow: {slow_time:.4f}s")
    
    start = time.time()
    result_fast = fibonacci_fast(30)
    fast_time = time.time() - start
    print(f"  Fast: {fast_time:.6f}s")
    print(f"  Speedup: {slow_time / fast_time:.0f}x")
