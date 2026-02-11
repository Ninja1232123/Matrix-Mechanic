"""
Demo functions for Test-Guardian
These functions demonstrate different scenarios that Test-Guardian handles
"""


def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def divide(numerator: float, denominator: float) -> float:
    """Divide two numbers

    Raises:
        ValueError: If denominator is zero
    """
    if denominator == 0:
        raise ValueError("Cannot divide by zero")
    return numerator / denominator


def process_list(items: list) -> list:
    """Process a list of items

    Returns doubled values for all items
    """
    if not items:
        return []

    result = []
    for item in items:
        result.append(item * 2)
    return result


def find_max(numbers: list) -> int:
    """Find maximum number in a list

    Raises:
        ValueError: If list is empty
    """
    if not numbers:
        raise ValueError("List cannot be empty")
    return max(numbers)


def format_name(first: str, last: str, title: str = None) -> str:
    """Format a person's name

    Args:
        first: First name
        last: Last name
        title: Optional title (Dr., Mr., Mrs., etc.)

    Returns:
        Formatted name string
    """
    if title:
        return f"{title} {first} {last}"
    return f"{first} {last}"


def validate_email(email: str) -> bool:
    """Validate an email address (simplified)"""
    if not email or '@' not in email:
        return False

    parts = email.split('@')
    if len(parts) != 2:
        return False

    return len(parts[0]) > 0 and len(parts[1]) > 0


def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price

    Args:
        price: Original price
        discount_percent: Discount percentage (0-100)

    Returns:
        Discounted price

    Raises:
        ValueError: If discount_percent is invalid
    """
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")

    discount_amount = price * (discount_percent / 100)
    return price - discount_amount


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries

    dict2 values override dict1 values for duplicate keys
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number (recursive)

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if n <= 1:
        return n

    return fibonacci(n - 1) + fibonacci(n - 2)


def parse_config(config_string: str) -> dict:
    """Parse configuration string (key=value format)

    Example: "host=localhost,port=8080"

    Returns:
        Dictionary of configuration values
    """
    if not config_string:
        return {}

    config = {}
    pairs = config_string.split(',')

    for pair in pairs:
        if '=' not in pair:
            continue

        key, value = pair.split('=', 1)
        config[key.strip()] = value.strip()

    return config
