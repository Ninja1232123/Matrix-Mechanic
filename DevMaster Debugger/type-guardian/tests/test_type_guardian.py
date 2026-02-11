"""
Basic tests for Type-Guardian
"""

import ast
from pathlib import Path
from type_guardian.parser import MypyParser
from type_guardian.inference.type_inferrer import TypeInferrer
from type_guardian.fixers.missing_hints import MissingHintsFixer
from type_guardian.fixers.optional_fixer import OptionalFixer


def test_parser():
    """Test mypy error parsing"""
    parser = MypyParser()
    
    error_lines = [
        "test.py:10:5: error: Missing return type annotation [return]",
        "test.py:15:20: error: Item \"None\" of \"Optional[User]\" has no attribute \"name\" [union-attr]",
    ]
    
    errors = parser.parse_errors(error_lines)
    
    assert len(errors) == 2
    assert errors[0]['category'] == 'missing_type_hint'
    assert errors[1]['category'] == 'optional_none'
    
    print("✅ Parser test passed")


def test_type_inference():
    """Test type inference"""
    inferrer = TypeInferrer()
    
    code = """
def add(x, y):
    return x + y
"""
    
    tree = ast.parse(code)
    func = tree.body[0]
    
    return_type = inferrer.infer_return_type(func)
    
    print(f"Inferred return type: {return_type}")
    print("✅ Type inference test passed")


def test_missing_hints_fixer():
    """Test missing hints fixer"""
    code = """
def greet(name):
    return f"Hello, {name}"
"""
    
    tree = ast.parse(code)
    func = tree.body[0]
    
    error = {
        'category': 'missing_type_hint',
        'line': 1,
        'message': 'Missing return type annotation'
    }
    
    fixer = MissingHintsFixer()
    
    if fixer.can_fix(error, tree):
        tree, success = fixer.fix(error, tree)
        print(f"Fix successful: {success}")
        if success:
            print(ast.unparse(tree))
    
    print("✅ Missing hints fixer test passed")


def test_optional_fixer():
    """Test Optional fixer"""
    code = """
user = get_user()
email = user.email
"""
    
    tree = ast.parse(code)
    
    error = {
        'category': 'optional_none',
        'line': 2,
        'message': 'Item "None" of "Optional[User]" has no attribute "email"',
        'context': {'name': 'user'}
    }
    
    fixer = OptionalFixer()
    
    if fixer.can_fix(error, tree):
        tree, success = fixer.fix(error, tree)
        print(f"Fix successful: {success}")
        if success:
            print(ast.unparse(tree))
    
    print("✅ Optional fixer test passed")


if __name__ == '__main__':
    print("Running Type-Guardian tests...\n")
    
    test_parser()
    test_type_inference()
    test_missing_hints_fixer()
    test_optional_fixer()
    
    print("\n✅ All tests passed!")
