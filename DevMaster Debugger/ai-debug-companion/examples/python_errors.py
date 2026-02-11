"""
Example Python script with various errors for testing the AI Debug Companion.

Run this script to see how the companion detects and suggests fixes.
"""


def import_error_example():
    """Example of an ImportError."""
    import requests  # This will fail if requests isn't installed
    response = requests.get("https://api.github.com")
    return response.json()


def attribute_error_example():
    """Example of an AttributeError."""
    my_var = None
    return my_var.some_method()  # AttributeError: 'NoneType' has no attribute 'some_method'


def type_error_example():
    """Example of a TypeError."""
    numbers = [1, 2, 3, 4, 5]
    result = numbers + "hello"  # TypeError: can only concatenate list to list
    return result


def index_error_example():
    """Example of an IndexError."""
    my_list = [1, 2, 3]
    return my_list[10]  # IndexError: list index out of range


def key_error_example():
    """Example of a KeyError."""
    my_dict = {"a": 1, "b": 2}
    return my_dict["c"]  # KeyError: 'c'


def division_by_zero_example():
    """Example of a ZeroDivisionError."""
    return 10 / 0


def name_error_example():
    """Example of a NameError."""
    return undefined_variable  # NameError: name 'undefined_variable' is not defined


if __name__ == "__main__":
    print("Running error examples...")
    print("This will generate various errors for the AI Debug Companion to analyze.")
    print()

    # Uncomment one at a time to test different errors
    # import_error_example()
    # attribute_error_example()
    # type_error_example()
    # index_error_example()
    # key_error_example()
    # division_by_zero_example()
    name_error_example()
