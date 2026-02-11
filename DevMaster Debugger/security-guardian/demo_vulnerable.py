"""
Demo file with intentional security vulnerabilities
DO NOT use this code in production!
For Security-Guardian demonstration only.
"""

import subprocess
import os


# 1. SQL Injection vulnerability
def get_user_by_name(username):
    """SQL Injection - string formatting in query"""
    query = f"SELECT * FROM users WHERE username = '{username}'"
    # cursor.execute(query)  # ❌ Vulnerable!
    return query


# 2. Command Injection vulnerability
def ping_server(hostname):
    """Command Injection - shell=True with user input"""
    cmd = f"ping -c 1 {hostname}"
    subprocess.run(cmd, shell=True)  # ❌ Vulnerable!


# 3. Hardcoded secrets
API_KEY = "sk_live_1234567890abcdefghijklmnop"  # ❌ Exposed secret!
DB_PASSWORD = "super_secret_password_123"  # ❌ Hardcoded password!
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"  # ❌ AWS key exposed!


# 4. Unsafe eval
def calculate_expression(user_input):
    """Code Injection - eval with user input"""
    result = eval(user_input)  # ❌ Extremely dangerous!
    return result


# 5. XSS vulnerability (in web context)
def render_user_comment(comment):
    """XSS - rendering user input without escaping"""
    # return render_template('comment.html', comment=mark_safe(comment))  # ❌ Unsafe!
    return f"<div>{comment}</div>"  # No escaping


# 6. Path Traversal vulnerability
def read_user_file(filename):
    """Path Traversal - no validation on filename"""
    filepath = f"/var/data/{filename}"
    with open(filepath, 'r') as f:  # ❌ User could use ../../../etc/passwd
        return f.read()


# 7. Insecure randomness
import random

def generate_session_token():
    """Weak crypto - using random instead of secrets"""
    return str(random.randint(1000000, 9999999))  # ❌ Predictable!


# 8. SQL Injection with format()
def find_products(category):
    """SQL Injection - .format() in query"""
    query = "SELECT * FROM products WHERE category = '{}'".format(category)
    # cursor.execute(query)  # ❌ Vulnerable!
    return query


# 9. Multiple secrets in connection string
DATABASE_URL = "postgresql://admin:password123@localhost:5432/mydb"  # ❌ Credentials exposed!


# 10. Unsafe pickle
import pickle

def load_user_data(data):
    """Insecure Deserialization - pickle with untrusted data"""
    return pickle.loads(data)  # ❌ Can execute arbitrary code!


def main():
    print("This file contains intentional security vulnerabilities")
    print("Run: security-guardian scan demo_vulnerable.py")
    print("\nExpected findings:")
    print("- SQL Injection vulnerabilities")
    print("- Command Injection")
    print("- Hardcoded secrets (API keys, passwords)")
    print("- Unsafe eval()")
    print("- XSS potential")
    print("- Path traversal")
    print("- Weak randomness")
    print("- Insecure deserialization")


if __name__ == '__main__':
    main()
