"""Scanner for injection vulnerabilities (SQL, XSS, Command)"""

import ast
import re
from typing import List
from pathlib import Path

from ..models import Vulnerability, VulnerabilityType, Severity


class InjectionScanner:
    """Scans for SQL injection, XSS, and command injection vulnerabilities"""

    def __init__(self):
        # Dangerous functions for each injection type
        self.sql_functions = ['execute', 'executemany', 'raw', 'query']
        self.command_functions = ['system', 'popen', 'exec', 'eval', 'subprocess', 'shell']
        self.xss_functions = ['render_template', 'render', 'write', 'send']

    def scan_file(self, file_path: Path) -> List[Vulnerability]:
        """Scan a file for injection vulnerabilities"""
        vulns = []

        try:
            with open(file_path, 'r') as f:
                source = f.read()
                lines = source.split('\n')

            tree = ast.parse(source)

            # Scan for SQL injection
            vulns.extend(self._scan_sql_injection(tree, file_path, lines))

            # Scan for command injection
            vulns.extend(self._scan_command_injection(tree, file_path, lines))

            # Scan for XSS
            vulns.extend(self._scan_xss(tree, file_path, lines))

            # Scan for unsafe eval/exec
            vulns.extend(self._scan_unsafe_eval(tree, file_path, lines))

        except Exception as e:
            pass  # Skip files that can't be parsed

        return vulns

    def _scan_sql_injection(
        self,
        tree: ast.AST,
        file_path: Path,
        lines: List[str]
    ) -> List[Vulnerability]:
        """Detect potential SQL injection vulnerabilities"""
        vulns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if calling SQL execute methods
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.sql_functions:
                        # Check if using string formatting/concatenation
                        if node.args:
                            arg = node.args[0]

                            # Detect f-strings, % formatting, .format(), or +
                            is_vulnerable = False

                            if isinstance(arg, ast.JoinedStr):  # f-string
                                is_vulnerable = True
                            elif isinstance(arg, ast.BinOp):  # String concatenation or %
                                if isinstance(arg.op, (ast.Mod, ast.Add)):
                                    is_vulnerable = True
                            elif isinstance(arg, ast.Call):  # .format()
                                if isinstance(arg.func, ast.Attribute):
                                    if arg.func.attr == 'format':
                                        is_vulnerable = True

                            if is_vulnerable:
                                line_no = node.lineno
                                vulns.append(Vulnerability(
                                    type=VulnerabilityType.SQL_INJECTION,
                                    severity=Severity.CRITICAL,
                                    file_path=str(file_path),
                                    line_number=line_no,
                                    code_snippet=lines[line_no - 1].strip() if line_no <= len(lines) else "",
                                    description="Potential SQL injection: Use parameterized queries instead of string formatting",
                                    cwe_id="CWE-89",
                                    owasp_category="A03:2021 - Injection",
                                    fix_suggestion="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
                                    confidence=0.8
                                ))

        return vulns

    def _scan_command_injection(
        self,
        tree: ast.AST,
        file_path: Path,
        lines: List[str]
    ) -> List[Vulnerability]:
        """Detect command injection vulnerabilities"""
        vulns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name and any(cmd in func_name.lower() for cmd in self.command_functions):
                    # Check if using shell=True or string interpolation
                    is_vulnerable = False

                    # Check for shell=True
                    for keyword in node.keywords:
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                            if keyword.value.value is True:
                                is_vulnerable = True

                    # Check for string formatting in command
                    if node.args:
                        arg = node.args[0]
                        if isinstance(arg, (ast.JoinedStr, ast.BinOp, ast.Call)):
                            is_vulnerable = True

                    if is_vulnerable:
                        line_no = node.lineno
                        vulns.append(Vulnerability(
                            type=VulnerabilityType.COMMAND_INJECTION,
                            severity=Severity.CRITICAL,
                            file_path=str(file_path),
                            line_number=line_no,
                            code_snippet=lines[line_no - 1].strip() if line_no <= len(lines) else "",
                            description="Potential command injection: Avoid shell=True and validate inputs",
                            cwe_id="CWE-78",
                            owasp_category="A03:2021 - Injection",
                            fix_suggestion="Use subprocess with list arguments and shell=False",
                            confidence=0.7
                        ))

        return vulns

    def _scan_xss(
        self,
        tree: ast.AST,
        file_path: Path,
        lines: List[str]
    ) -> List[Vulnerability]:
        """Detect potential XSS vulnerabilities"""
        vulns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Check for render_template with |safe filter or direct HTML
                    if 'render' in node.func.id.lower():
                        line_no = node.lineno
                        code = lines[line_no - 1] if line_no <= len(lines) else ""

                        if '|safe' in code or 'mark_safe' in code:
                            vulns.append(Vulnerability(
                                type=VulnerabilityType.XSS,
                                severity=Severity.HIGH,
                                file_path=str(file_path),
                                line_number=line_no,
                                code_snippet=code.strip(),
                                description="Potential XSS: Unsafe HTML rendering without escaping",
                                cwe_id="CWE-79",
                                owasp_category="A03:2021 - Injection",
                                fix_suggestion="Escape user input or use auto-escaping templates",
                                confidence=0.6
                            ))

        return vulns

    def _scan_unsafe_eval(
        self,
        tree: ast.AST,
        file_path: Path,
        lines: List[str]
    ) -> List[Vulnerability]:
        """Detect unsafe eval/exec usage"""
        vulns = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        line_no = node.lineno
                        vulns.append(Vulnerability(
                            type=VulnerabilityType.UNSAFE_EVAL,
                            severity=Severity.CRITICAL,
                            file_path=str(file_path),
                            line_number=line_no,
                            code_snippet=lines[line_no - 1].strip() if line_no <= len(lines) else "",
                            description=f"Unsafe use of {node.func.id}(): Never use with untrusted input",
                            cwe_id="CWE-95",
                            owasp_category="A03:2021 - Injection",
                            fix_suggestion="Use ast.literal_eval() for safe evaluation or avoid dynamic code execution",
                            confidence=0.9
                        ))

        return vulns
