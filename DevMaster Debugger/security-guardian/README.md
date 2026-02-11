# Security-Guardian 🔒

> Automatic Security Vulnerability Scanner for Python

Security-Guardian scans your Python code for security vulnerabilities including SQL injection, XSS, hardcoded secrets, command injection, and vulnerable dependencies.

## Features

### 🛡️ Vulnerability Detection
- **SQL Injection** (CWE-89) - Detects unsafe database queries
- **XSS** (CWE-79) - Cross-site scripting vulnerabilities
- **Command Injection** (CWE-78) - Unsafe system commands
- **Path Traversal** - Directory traversal attacks
- **Unsafe Eval/Exec** (CWE-95) - Dynamic code execution

### 🔐 Secrets Detection
- API keys (AWS, GitHub, etc.)
- Hardcoded passwords
- Access tokens
- Private keys
- Database connection strings

### 📦 Dependency Scanning
- Vulnerable package versions
- Known CVEs
- Outdated dependencies

### 📊 OWASP Top 10 Coverage
- A03:2021 - Injection
- A07:2021 - Identification and Authentication Failures
- Aligned with industry standards

## Installation

```bash
cd security-guardian
pip install -e .
```

## Quick Start

### Scan a File

```bash
security-guardian scan myfile.py
```

### Scan a Directory

```bash
security-guardian scan /path/to/project
```

### Save Report

```bash
security-guardian scan myproject/ --output security_report.txt
```

## Usage Examples

### Example 1: SQL Injection Detection

**Vulnerable Code:**
```python
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)  # ❌ SQL Injection!
```

**Security-Guardian Output:**
```
[CRITICAL] sql_injection at myfile.py:2
Potential SQL injection: Use parameterized queries
Fix: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
```

**Fixed Code:**
```python
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))  # ✅ Safe!
```

### Example 2: Secrets Detection

**Vulnerable Code:**
```python
API_KEY = "sk_live_abc123xyz789"  # ❌ Hardcoded secret!
AWS_SECRET = "AKIA1234567890ABCDEF"  # ❌ Exposed!
```

**Security-Guardian Output:**
```
[CRITICAL] hardcoded_credentials at config.py:1
Hardcoded api_key detected
Fix: Use environment variables (os.getenv('API_KEY'))
```

**Fixed Code:**
```python
import os
API_KEY = os.getenv('API_KEY')  # ✅ Secure!
AWS_SECRET = os.getenv('AWS_SECRET')  # ✅ Safe!
```

### Example 3: Command Injection

**Vulnerable Code:**
```python
import subprocess

def ping_host(hostname):
    cmd = f"ping -c 1 {hostname}"
    subprocess.run(cmd, shell=True)  # ❌ Command injection!
```

**Security-Guardian Output:**
```
[CRITICAL] command_injection at network.py:4
Avoid shell=True and validate inputs
Fix: Use subprocess with list arguments and shell=False
```

**Fixed Code:**
```python
import subprocess

def ping_host(hostname):
    # Validate hostname first
    subprocess.run(['ping', '-c', '1', hostname], shell=False)  # ✅ Safe!
```

## CLI Commands

```bash
# Scan file or directory
security-guardian scan TARGET [OPTIONS]

Options:
  --recursive/--no-recursive  Scan subdirectories (default: True)
  -o, --output PATH          Save report to file
```

## Programmatic Usage

```python
from security_guardian import SecurityGuardian
from pathlib import Path

sg = SecurityGuardian()

# Scan a file
report = sg.scan_file(Path("myfile.py"))

print(f"Found {len(report.vulnerabilities)} vulnerabilities")
print(f"Critical: {report.get_critical_count()}")
print(f"Secrets: {len(report.secrets)}")

# Generate report
print(sg.generate_report_text(report))
```

## Detected Vulnerability Types

### Injection Attacks
- **SQL Injection** - String formatting in SQL queries
- **Command Injection** - shell=True, unsanitized input
- **XSS** - Unsafe HTML rendering
- **Code Injection** - eval(), exec() with user input

### Credentials & Secrets
- **API Keys** - AWS, GitHub, Generic
- **Passwords** - Hardcoded passwords
- **Tokens** - Access tokens, auth tokens
- **Private Keys** - RSA, DSA, EC keys
- **Connection Strings** - Database URLs with credentials

### Dependencies
- **Known CVEs** - Packages with security advisories
- **Outdated Versions** - Old package versions
- **Vulnerable Transitive Deps** - Indirect dependencies

## Severity Levels

- **CRITICAL** 🔴 - Immediate fix required (SQL injection, secrets, eval)
- **HIGH** 🟡 - Fix ASAP (XSS, command injection)
- **MEDIUM** 🔵 - Fix soon (weak crypto, insecure configs)
- **LOW** 🟢 - Fix when possible (code smells)
- **INFO** ⚪ - Informational only

## Best Practices

### 1. Scan Regularly
```bash
# Add to CI/CD pipeline
security-guardian scan . --output security_report.txt
```

### 2. Fix Critical First
Focus on CRITICAL and HIGH severity issues first.

### 3. Never Commit Secrets
Use `.env` files (gitignored) or secret management:
```python
# .env
API_KEY=your_key_here

# app.py
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('API_KEY')
```

### 4. Use Parameterized Queries
```python
# ❌ Bad
query = f"SELECT * FROM users WHERE name = '{name}'"

# ✅ Good
query = "SELECT * FROM users WHERE name = ?"
cursor.execute(query, (name,))
```

### 5. Validate All Input
```python
import re

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```

## Integration

### With CI/CD

**GitHub Actions:**
```yaml
- name: Security Scan
  run: |
    pip install -e security-guardian
    security-guardian scan . --output security_report.txt
    cat security_report.txt
```

### With Other Guardians

Security-Guardian works seamlessly with:
- **Universal Debugger** - Fix bugs, scan for security issues
- **Type-Guardian** - Add type hints, improve code safety
- **Test-Guardian** - Generate security tests
- **Deploy-Shield** - Validate secure deployments

## Limitations

- **Static Analysis Only** - Cannot detect runtime vulnerabilities
- **False Positives** - May flag safe code (review manually)
- **Python Only** - Currently supports Python code only
- **Simplified CVE DB** - Use dedicated tools like Safety for comprehensive CVE checking

## Roadmap

- [ ] More vulnerability patterns (SSRF, XXE, CSRF)
- [ ] Integration with Safety DB for CVEs
- [ ] Auto-fix mode (like Universal Debugger)
- [ ] Multi-language support (JavaScript, Go, etc.)
- [ ] SARIF format output for IDE integration
- [ ] Custom rule engine
- [ ] Compliance reporting (PCI-DSS, HIPAA)

## Related Tools

- **Universal Debugger**: Fix runtime errors
- **Type-Guardian**: Add type hints
- **Test-Guardian**: Generate tests
- **Speed-Guardian**: Optimize performance
- **Deploy-Shield**: Validate deployments

---

**Security-Guardian** - Secure your code automatically! 🔒
