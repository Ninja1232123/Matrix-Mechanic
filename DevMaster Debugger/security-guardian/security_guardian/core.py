"""Core Security-Guardian class"""

from typing import List, Optional
from pathlib import Path

from .models import SecurityReport, Vulnerability, SecretMatch, DependencyIssue
from .scanners import InjectionScanner, SecretsScanner, DependencyScanner


class SecurityGuardian:
    """Main Security-Guardian class - orchestrates security scanning"""

    def __init__(self):
        self.injection_scanner = InjectionScanner()
        self.secrets_scanner = SecretsScanner()
        self.dependency_scanner = DependencyScanner()

    def scan_file(self, file_path: Path) -> SecurityReport:
        """Scan a single Python file for vulnerabilities"""
        vulnerabilities = []
        secrets = []

        # Scan for injection vulnerabilities
        vulns = self.injection_scanner.scan_file(file_path)
        vulnerabilities.extend(vulns)

        # Scan for hardcoded secrets
        secret_matches = self.secrets_scanner.scan_file(file_path)
        secrets.extend(secret_matches)

        # Convert secrets to vulnerabilities
        secret_vulns = self.secrets_scanner.convert_to_vulnerabilities(secret_matches)
        vulnerabilities.extend(secret_vulns)

        # Create report
        report = SecurityReport(
            target=str(file_path),
            vulnerabilities=vulnerabilities,
            secrets=secret_matches,
            dependency_issues=[],
            summary={
                'total_vulnerabilities': len(vulnerabilities),
                'critical': report.get_critical_count() if vulnerabilities else 0,
                'high': report.get_high_count() if vulnerabilities else 0,
                'secrets_found': len(secret_matches),
            }
        )

        return report

    def scan_directory(self, directory: Path, recursive: bool = True) -> SecurityReport:
        """Scan all Python files in a directory"""
        all_vulns = []
        all_secrets = []
        all_deps = []

        # Find all Python files
        if recursive:
            python_files = list(directory.rglob("*.py"))
        else:
            python_files = list(directory.glob("*.py"))

        # Scan each file
        for py_file in python_files:
            report = self.scan_file(py_file)
            all_vulns.extend(report.vulnerabilities)
            all_secrets.extend(report.secrets)

        # Scan dependencies
        req_file = directory / "requirements.txt"
        if req_file.exists():
            dep_issues = self.dependency_scanner.scan_requirements(req_file)
            all_deps.extend(dep_issues)

        # Create combined report
        report = SecurityReport(
            target=str(directory),
            vulnerabilities=all_vulns,
            secrets=all_secrets,
            dependency_issues=all_deps,
            summary={
                'files_scanned': len(python_files),
                'total_vulnerabilities': len(all_vulns),
                'critical': len([v for v in all_vulns if v.severity.value == 'critical']),
                'high': len([v for v in all_vulns if v.severity.value == 'high']),
                'secrets_found': len(all_secrets),
                'vulnerable_dependencies': len(all_deps),
            }
        )

        return report

    def generate_report_text(self, report: SecurityReport) -> str:
        """Generate human-readable text report"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"Security-Guardian Scan Report: {report.target}")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Issues: {report.get_total_issues()}")
        lines.append(f"Vulnerabilities: {len(report.vulnerabilities)}")
        lines.append(f"  Critical: {report.get_critical_count()}")
        lines.append(f"  High: {report.get_high_count()}")
        lines.append(f"Secrets Found: {len(report.secrets)}")
        lines.append(f"Dependency Issues: {len(report.dependency_issues)}")
        lines.append("")

        # Vulnerabilities
        if report.vulnerabilities:
            lines.append("VULNERABILITIES")
            lines.append("-" * 80)
            for i, vuln in enumerate(report.vulnerabilities[:20], 1):
                lines.append(f"{i}. {vuln}")
                lines.append(f"   {vuln.description}")
                lines.append(f"   Fix: {vuln.fix_suggestion}")
                lines.append("")

        # Secrets
        if report.secrets:
            lines.append("SECRETS DETECTED")
            lines.append("-" * 80)
            for i, secret in enumerate(report.secrets[:10], 1):
                lines.append(f"{i}. {secret.secret_type} at {secret.file_path}:{secret.line_number}")
                lines.append(f"   Context: {secret.context[:80]}...")
                lines.append("")

        # Dependencies
        if report.dependency_issues:
            lines.append("VULNERABLE DEPENDENCIES")
            lines.append("-" * 80)
            for i, dep in enumerate(report.dependency_issues, 1):
                lines.append(f"{i}. {dep.package_name} v{dep.current_version}")
                for cve in dep.vulnerabilities:
                    lines.append(f"   - {cve}")
                lines.append("")

        lines.append("=" * 80)

        return '\n'.join(lines)
