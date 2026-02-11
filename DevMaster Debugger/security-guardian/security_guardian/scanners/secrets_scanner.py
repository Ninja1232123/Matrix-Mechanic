"""Scanner for hardcoded secrets and credentials"""

import re
from typing import List
from pathlib import Path

from ..models import SecretMatch, Vulnerability, VulnerabilityType, Severity


class SecretsScanner:
    """Scans for hardcoded secrets, API keys, passwords, tokens"""

    def __init__(self):
        # Regex patterns for common secrets
        self.patterns = {
            'api_key': [
                r'(?i)(api[_-]?key|apikey|api[_-]?secret)["\']?\s*[:=]\s*["\']([a-zA-Z0-9_\-]{20,})["\']',
                r'(?i)(AKIA[0-9A-Z]{16})',  # AWS Access Key
            ],
            'password': [
                r'(?i)(password|passwd|pwd)["\']?\s*[:=]\s*["\']([^"\']{8,})["\']',
            ],
            'token': [
                r'(?i)(token|auth[_-]?token|access[_-]?token)["\']?\s*[:=]\s*["\']([a-zA-Z0-9_\-\.]{20,})["\']',
                r'(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}',  # GitHub tokens
            ],
            'private_key': [
                r'-----BEGIN (RSA |DSA |EC )?PRIVATE KEY-----',
            ],
            'connection_string': [
                r'(mongodb|mysql|postgres)://[^:]+:[^@]+@',
            ],
        }

    def scan_file(self, file_path: Path) -> List[SecretMatch]:
        """Scan a file for hardcoded secrets"""
        secrets = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, 1):
                for secret_type, patterns in self.patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            # Skip common false positives
                            if self._is_false_positive(line, match.group(0)):
                                continue

                            secrets.append(SecretMatch(
                                secret_type=secret_type,
                                file_path=str(file_path),
                                line_number=line_no,
                                matched_pattern=match.group(0)[:50],  # Truncate for safety
                                context=line.strip()[:100]
                            ))

        except Exception:
            pass

        return secrets

    def _is_false_positive(self, line: str, match: str) -> bool:
        """Check if match is likely a false positive"""
        false_positive_indicators = [
            'example', 'sample', 'test', 'fake', 'dummy',
            'placeholder', 'your_', 'xxx', '***', 'todo',
            'insert_here', 'replace_me'
        ]

        line_lower = line.lower()
        match_lower = match.lower()

        return any(indicator in line_lower or indicator in match_lower
                  for indicator in false_positive_indicators)

    def convert_to_vulnerabilities(self, secrets: List[SecretMatch]) -> List[Vulnerability]:
        """Convert secret matches to vulnerabilities"""
        vulns = []

        for secret in secrets:
            vulns.append(Vulnerability(
                type=VulnerabilityType.HARDCODED_CREDENTIALS,
                severity=Severity.CRITICAL,
                file_path=secret.file_path,
                line_number=secret.line_number,
                code_snippet=secret.context,
                description=f"Hardcoded {secret.secret_type} detected",
                cwe_id="CWE-798",
                owasp_category="A07:2021 - Identification and Authentication Failures",
                fix_suggestion="Use environment variables or secret management (e.g., .env, HashiCorp Vault)",
                confidence=0.7
            ))

        return vulns
