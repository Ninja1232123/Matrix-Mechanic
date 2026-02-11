"""Scanner for vulnerable and outdated dependencies"""

import re
from typing import List, Dict, Any
from pathlib import Path

from ..models import DependencyIssue, Severity


class DependencyScanner:
    """Scans for outdated and vulnerable dependencies"""

    def __init__(self):
        # Known vulnerable packages (simplified - real impl would use safety DB)
        self.known_vulnerabilities = {
            'django': {
                '3.0': ['CVE-2021-33203', 'CVE-2021-33571'],
                '2.2': ['CVE-2020-13254', 'CVE-2020-13596'],
            },
            'flask': {
                '1.0': ['CVE-2019-1010083'],
            },
            'requests': {
                '2.19': ['CVE-2018-18074'],
            },
            'pyyaml': {
                '5.3': ['CVE-2020-14343'],
            },
        }

    def scan_requirements(self, file_path: Path) -> List[DependencyIssue]:
        """Scan requirements.txt for vulnerable dependencies"""
        issues = []

        if not file_path.exists():
            return issues

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Parse package==version
                match = re.match(r'([a-zA-Z0-9_-]+)(==|>=|<=|>|<)([0-9\.]+)', line)
                if match:
                    package = match.group(1).lower()
                    version = match.group(3)

                    # Check for known vulnerabilities
                    if package in self.known_vulnerabilities:
                        vulns = self._check_vulnerabilities(package, version)
                        if vulns:
                            issues.append(DependencyIssue(
                                package_name=package,
                                current_version=version,
                                latest_version=None,  # Would query PyPI in real impl
                                vulnerabilities=vulns,
                                severity=Severity.HIGH
                            ))

        except Exception:
            pass

        return issues

    def _check_vulnerabilities(self, package: str, version: str) -> List[str]:
        """Check if a package version has known vulnerabilities"""
        vulns = []

        if package in self.known_vulnerabilities:
            version_vulns = self.known_vulnerabilities[package]

            # Check if current version is vulnerable
            for vuln_version, cves in version_vulns.items():
                if self._version_matches(version, vuln_version):
                    vulns.extend(cves)

        return vulns

    def _version_matches(self, current: str, vulnerable: str) -> bool:
        """Check if current version matches vulnerable version (simplified)"""
        # Simplified version comparison
        try:
            current_parts = [int(x) for x in current.split('.')]
            vuln_parts = [int(x) for x in vulnerable.split('.')]

            # Check major.minor match
            return current_parts[:2] == vuln_parts[:2]
        except:
            return False
