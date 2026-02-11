"""
Secure version of demo_vulnerable.py
Shows how to fix the security vulnerabilities
"""

import subprocess
import os
import secrets
import json


# 1. SQL Injection - FIXED with parameterized queries
def get_user_by_name_secure(username):
    """Use parameterized queries"""
    query = "SELECT * FROM users WHERE username = ?"
    # cursor.execute(query, (username,))  # ✅ Safe!
    return query


# 2. Command Injection - FIXED
def ping_server_secure(hostname):
    """Use list args and shell=False"""
    # Validate hostname first
    if not hostname.replace('.', '').replace('-', '').isalnum():
        raise ValueError("Invalid hostname")

    subprocess.run(['ping', '-c', '1', hostname], shell=False)  # ✅ Safe!


# 3. Hardcoded secrets - FIXED with environment variables
API_KEY = os.getenv('API_KEY')  # ✅ From environment
DB_PASSWORD = os.getenv('DB_PASSWORD')  # ✅ Secure!
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')  # ✅ Safe!


# 4. Unsafe eval - FIXED with ast.literal_eval
import ast

def calculate_expression_secure(user_input):
    """Use ast.literal_eval for safe evaluation"""
    try:
        result = ast.literal_eval(user_input)  # ✅ Safe for literals only
        return result
    except (ValueError, SyntaxError):
        raise ValueError("Invalid expression")


# 5. XSS - FIXED with proper escaping
import html

def render_user_comment_secure(comment):
    """Escape user input"""
    safe_comment = html.escape(comment)  # ✅ Escaped!
    return f"<div>{safe_comment}</div>"


# 6. Path Traversal - FIXED with validation
from pathlib import Path

def read_user_file_secure(filename):
    """Validate and sanitize filename"""
    # Remove any path traversal attempts
    safe_filename = Path(filename).name  # Only filename, no path
    base_dir = Path("/var/data")
    filepath = base_dir / safe_filename

    # Ensure file is within base directory
    if not filepath.resolve().is_relative_to(base_dir.resolve()):
        raise ValueError("Invalid file path")

    with open(filepath, 'r') as f:  # ✅ Safe!
        return f.read()


# 7. Insecure randomness - FIXED with secrets module
def generate_session_token_secure():
    """Use secrets module for cryptographic randomness"""
    return secrets.token_urlsafe(32)  # ✅ Cryptographically secure!


# 8. SQL Injection - FIXED
def find_products_secure(category):
    """Use parameterized query"""
    query = "SELECT * FROM products WHERE category = ?"
    # cursor.execute(query, (category,))  # ✅ Safe!
    return query


# 9. Connection string - FIXED
DATABASE_URL = os.getenv('DATABASE_URL')  # ✅ From environment, not hardcoded!


# 10. Unsafe pickle - FIXED with JSON
def load_user_data_secure(data):
    """Use JSON instead of pickle"""
    return json.loads(data)  # ✅ Safe! (for trusted data structures)


# Bonus: Use .env file for secrets
def load_environment():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Loads from .env file
    except ImportError:
        print("Install python-dotenv: pip install python-dotenv")


def main():
    print("This file shows secure coding practices")
    print("\nKey security improvements:")
    print("✅ Parameterized SQL queries")
    print("✅ No shell=True in subprocess")
    print("✅ Secrets from environment variables")
    print("✅ ast.literal_eval instead of eval")
    print("✅ HTML escaping for XSS prevention")
    print("✅ Path validation against traversal")
    print("✅ Cryptographic randomness with secrets module")
    print("✅ JSON instead of pickle")
    print("\nScan with: security-guardian scan demo_secure.py")
    print("Should find 0 critical vulnerabilities!")


if __name__ == '__main__':
    main()
