"""
Performance-Surgeon Performance Patterns Database

Anti-patterns that cause performance bottlenecks and their optimized solutions.
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Callable


class Severity(Enum):
    """Performance impact severity"""
    CRITICAL = "CRITICAL"  # 10x+ slowdown
    HIGH = "HIGH"          # 3-10x slowdown
    MEDIUM = "MEDIUM"      # 2-3x slowdown
    LOW = "LOW"            # <2x slowdown
    INFO = "INFO"          # Best practice


class Category(Enum):
    """Performance issue categories"""
    ALGORITHMIC = "Algorithmic Complexity"
    DATABASE = "Database Performance"
    MEMORY = "Memory Management"
    CPU = "CPU Optimization"
    IO = "I/O Operations"
    CACHING = "Caching"
    CONCURRENCY = "Concurrency"


@dataclass
class PerformancePattern:
    """Represents a performance anti-pattern"""
    name: str
    category: Category
    severity: Severity
    description: str
    regex_patterns: List[str]
    complexity_before: str  # Big-O notation
    complexity_after: str   # Big-O notation
    fix_strategy: str
    example_slow: str
    example_fast: str
    estimated_speedup: str  # e.g., "10x", "100x"
    ast_check: Optional[Callable] = None


# ============================================================================
# ALGORITHMIC COMPLEXITY PATTERNS
# ============================================================================

ALGORITHMIC_PATTERNS = [
    PerformancePattern(
        name="nested_loops_in_instead_of_set",
        category=Category.ALGORITHMIC,
        severity=Severity.CRITICAL,
        description="Using nested loops with 'in' on list instead of set lookup",
        regex_patterns=[
            r'for\s+\w+\s+in\s+\w+:.*\n\s+if\s+\w+\s+in\s+\w+:',
        ],
        complexity_before="O(n × m)",
        complexity_after="O(n + m)",
        fix_strategy="Convert inner list to set for O(1) lookup",
        example_slow="""
for item in list1:
    if item in list2:  # O(m) lookup
        result.append(item)
""",
        example_fast="""
set2 = set(list2)  # O(m) conversion
for item in list1:
    if item in set2:  # O(1) lookup
        result.append(item)
""",
        estimated_speedup="100x for large lists"
    ),
    
    PerformancePattern(
        name="append_in_loop",
        category=Category.ALGORITHMIC,
        severity=Severity.MEDIUM,
        description="Using append() in loop instead of list comprehension",
        regex_patterns=[
            r'\w+\s*=\s*\[\]\s*\n\s*for\s+.*:\s*\n\s+\w+\.append\(',
        ],
        complexity_before="O(n)",
        complexity_after="O(n)",
        fix_strategy="Use list comprehension for better performance",
        example_slow="""
result = []
for item in items:
    result.append(process(item))
""",
        example_fast="""
result = [process(item) for item in items]
""",
        estimated_speedup="2-3x"
    ),
    
    PerformancePattern(
        name="repeated_string_concatenation",
        category=Category.ALGORITHMIC,
        severity=Severity.HIGH,
        description="String concatenation in loop (creates new string each time)",
        regex_patterns=[
            r'for\s+.*:\s*\n\s+\w+\s*\+=\s*["\']',
            r'for\s+.*:\s*\n\s+\w+\s*=\s*\w+\s*\+\s*["\']',
        ],
        complexity_before="O(n²)",
        complexity_after="O(n)",
        fix_strategy="Use ''.join() or list accumulation",
        example_slow="""
result = ""
for item in items:
    result += str(item)  # Creates new string each time!
""",
        example_fast="""
result = ''.join(str(item) for item in items)
""",
        estimated_speedup="10-100x for large strings"
    ),
    
    PerformancePattern(
        name="list_extend_in_loop",
        category=Category.ALGORITHMIC,
        severity=Severity.MEDIUM,
        description="Using extend() in loop instead of single extend",
        regex_patterns=[
            r'for\s+.*:\s*\n\s+\w+\.extend\(',
        ],
        complexity_before="O(n × m)",
        complexity_after="O(n + m)",
        fix_strategy="Collect and extend once, or use itertools.chain",
        example_slow="""
result = []
for sublist in lists:
    result.extend(sublist)  # Multiple memory reallocations
""",
        example_fast="""
from itertools import chain
result = list(chain.from_iterable(lists))
""",
        estimated_speedup="3-5x"
    ),
]

# ============================================================================
# DATABASE PATTERNS
# ============================================================================

DATABASE_PATTERNS = [
    PerformancePattern(
        name="n_plus_one_query",
        category=Category.DATABASE,
        severity=Severity.CRITICAL,
        description="N+1 query problem - loading related objects in loop",
        regex_patterns=[
            r'for\s+\w+\s+in\s+\w+\.(?:objects|query)\.',
            r'for\s+\w+\s+in\s+\w+:.*\.\w+\.all\(\)',
        ],
        complexity_before="O(n) queries",
        complexity_after="O(1) queries",
        fix_strategy="Use select_related() or prefetch_related()",
        example_slow="""
users = User.objects.all()
for user in users:
    posts = user.posts.all()  # N queries!
""",
        example_fast="""
users = User.objects.prefetch_related('posts')
for user in users:
    posts = user.posts.all()  # Already loaded!
""",
        estimated_speedup="100-1000x"
    ),
    
    PerformancePattern(
        name="query_in_loop",
        category=Category.DATABASE,
        severity=Severity.CRITICAL,
        description="Database query inside loop",
        regex_patterns=[
            r'for\s+.*:\s*\n\s+.*\.filter\(',
            r'for\s+.*:\s*\n\s+.*\.get\(',
        ],
        complexity_before="O(n) queries",
        complexity_after="O(1) queries",
        fix_strategy="Batch queries with filter(id__in=...)",
        example_slow="""
for user_id in user_ids:
    user = User.objects.get(id=user_id)  # N queries!
""",
        example_fast="""
users = User.objects.filter(id__in=user_ids)  # 1 query!
user_dict = {u.id: u for u in users}
""",
        estimated_speedup="100x+"
    ),
    
    PerformancePattern(
        name="select_all_columns",
        category=Category.DATABASE,
        severity=Severity.MEDIUM,
        description="Selecting all columns when only few needed",
        regex_patterns=[
            r'\.objects\.all\(\)',
            r'SELECT \*',
        ],
        complexity_before="Large data transfer",
        complexity_after="Minimal data transfer",
        fix_strategy="Use only() or values() to select specific fields",
        example_slow="""
users = User.objects.all()  # Gets all fields
names = [u.name for u in users]
""",
        example_fast="""
users = User.objects.only('name')  # Gets only name
names = [u.name for u in users]
""",
        estimated_speedup="2-5x"
    ),
]

# ============================================================================
# CACHING PATTERNS
# ============================================================================

CACHING_PATTERNS = [
    PerformancePattern(
        name="uncached_expensive_function",
        category=Category.CACHING,
        severity=Severity.HIGH,
        description="Expensive pure function called repeatedly without caching",
        regex_patterns=[
            r'def\s+\w+\([^)]*\):.*return',  # Needs AST analysis
        ],
        complexity_before="O(expensive) × calls",
        complexity_after="O(expensive) + O(1) × calls",
        fix_strategy="Add @lru_cache decorator",
        example_slow="""
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)  # Exponential!
""",
        example_fast="""
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)  # Cached!
""",
        estimated_speedup="1000x+"
    ),
    
    PerformancePattern(
        name="repeated_file_reads",
        category=Category.CACHING,
        severity=Severity.HIGH,
        description="Reading same file multiple times",
        regex_patterns=[
            r'for\s+.*:\s*\n\s+.*open\(',
            r'open\(["\'][^"\']+["\']\).*\n.*for\s+',
        ],
        complexity_before="O(n × file_size)",
        complexity_after="O(file_size)",
        fix_strategy="Read file once and cache",
        example_slow="""
for i in range(100):
    with open('data.txt') as f:
        data = f.read()  # Reads file 100 times!
""",
        example_fast="""
with open('data.txt') as f:
    data = f.read()  # Read once
for i in range(100):
    process(data)  # Use cached data
""",
        estimated_speedup="100x"
    ),
]

# ============================================================================
# MEMORY PATTERNS
# ============================================================================

MEMORY_PATTERNS = [
    PerformancePattern(
        name="loading_entire_file",
        category=Category.MEMORY,
        severity=Severity.HIGH,
        description="Loading entire large file into memory",
        regex_patterns=[
            r'\.read\(\)',
            r'\.readlines\(\)',
        ],
        complexity_before="O(file_size) memory",
        complexity_after="O(1) memory",
        fix_strategy="Use line iteration or chunked reading",
        example_slow="""
with open('large.txt') as f:
    lines = f.readlines()  # Loads entire file!
    for line in lines:
        process(line)
""",
        example_fast="""
with open('large.txt') as f:
    for line in f:  # Streams line by line
        process(line)
""",
        estimated_speedup="1000x less memory"
    ),
    
    PerformancePattern(
        name="list_when_generator_works",
        category=Category.MEMORY,
        severity=Severity.MEDIUM,
        description="Creating list when generator expression would work",
        regex_patterns=[
            r'\[.*for.*in.*\].*\)',  # List comp passed to function
        ],
        complexity_before="O(n) memory",
        complexity_after="O(1) memory",
        fix_strategy="Use generator expression with ()",
        example_slow="""
total = sum([x**2 for x in range(1000000)])  # Creates list!
""",
        example_fast="""
total = sum(x**2 for x in range(1000000))  # Generator!
""",
        estimated_speedup="100x less memory"
    ),
    
    PerformancePattern(
        name="unnecessary_list_copy",
        category=Category.MEMORY,
        severity=Severity.LOW,
        description="Creating unnecessary list copies",
        regex_patterns=[
            r'\w+\s*=\s*\w+\[:\]',
            r'\w+\s*=\s*list\(\w+\)',
        ],
        complexity_before="O(n) time & memory",
        complexity_after="O(1) time & memory",
        fix_strategy="Avoid copying unless mutation needed",
        example_slow="""
temp = mylist[:]  # Unnecessary copy
for item in temp:
    process(item)
""",
        example_fast="""
for item in mylist:  # No copy needed
    process(item)
""",
        estimated_speedup="2x"
    ),
]

# ============================================================================
# CPU PATTERNS
# ============================================================================

CPU_PATTERNS = [
    PerformancePattern(
        name="global_in_loop",
        category=Category.CPU,
        severity=Severity.LOW,
        description="Global variable lookup in tight loop",
        regex_patterns=[
            r'for\s+.*:\s*\n\s+.*(?:len|range|str|int)\(',
        ],
        complexity_before="O(n) lookups",
        complexity_after="O(1) lookup",
        fix_strategy="Cache global in local variable",
        example_slow="""
for i in range(len(items)):  # len() lookup each iteration
    process(items[i])
""",
        example_fast="""
n = len(items)  # Cache length
for i in range(n):
    process(items[i])
""",
        estimated_speedup="1.2-1.5x"
    ),
    
    PerformancePattern(
        name="repeated_regex_compilation",
        category=Category.CPU,
        severity=Severity.MEDIUM,
        description="Compiling regex pattern repeatedly",
        regex_patterns=[
            r'for\s+.*:\s*\n\s+.*re\.(?:match|search|findall)\(["\']',
        ],
        complexity_before="O(n × compile_time)",
        complexity_after="O(compile_time + n)",
        fix_strategy="Compile regex once outside loop",
        example_slow="""
for text in texts:
    if re.match(r'pattern', text):  # Compiles each time!
        process(text)
""",
        example_fast="""
pattern = re.compile(r'pattern')  # Compile once
for text in texts:
    if pattern.match(text):
        process(text)
""",
        estimated_speedup="5-10x"
    ),
]

# ============================================================================
# I/O PATTERNS
# ============================================================================

IO_PATTERNS = [
    PerformancePattern(
        name="unbuffered_writes",
        category=Category.IO,
        severity=Severity.HIGH,
        description="Writing to file in loop without buffering",
        regex_patterns=[
            r'for\s+.*:\s*\n\s+.*\.write\(',
        ],
        complexity_before="O(n) I/O operations",
        complexity_after="O(1) I/O operations",
        fix_strategy="Batch writes or use buffering",
        example_slow="""
with open('out.txt', 'w') as f:
    for item in items:
        f.write(str(item))  # Flushes each time!
""",
        example_fast="""
with open('out.txt', 'w') as f:
    f.write(''.join(str(item) for item in items))  # One write
""",
        estimated_speedup="10-50x"
    ),
]

# ============================================================================
# MASTER PATTERN DATABASE
# ============================================================================

PERFORMANCE_PATTERNS = {
    "algorithmic": ALGORITHMIC_PATTERNS,
    "database": DATABASE_PATTERNS,
    "caching": CACHING_PATTERNS,
    "memory": MEMORY_PATTERNS,
    "cpu": CPU_PATTERNS,
    "io": IO_PATTERNS,
}

# Flatten all patterns
ALL_PATTERNS = []
for pattern_list in PERFORMANCE_PATTERNS.values():
    ALL_PATTERNS.extend(pattern_list)


def get_patterns_by_severity(severity: Severity) -> List[PerformancePattern]:
    """Get all patterns matching a severity level"""
    return [p for p in ALL_PATTERNS if p.severity == severity]


def get_patterns_by_category(category: Category) -> List[PerformancePattern]:
    """Get all patterns matching a category"""
    return [p for p in ALL_PATTERNS if p.category == category]
