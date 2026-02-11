#!/usr/bin/env python3
"""
Deploy-Shield CLI - Validate deployments before they fail
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .validators.env_validator import EnvironmentValidator
from .validators.port_validator import PortValidator
from .validators.permission_validator import PermissionValidator
from .validators.database_validator import DatabaseValidator
from .validators.ssl_validator import SSLValidator
from .validators.resource_validator import ResourceValidator
from .validators.config_validator import ConfigValidator

from .fixers.env_fixer import EnvironmentFixer
from .fixers.dockerfile_fixer import DockerfileFixer
from .fixers.docker_compose_fixer import DockerComposeFixer

from .testers.connection_tester import ConnectionTester
from .testers.health_checker import HealthChecker

console = Console()


class DeployShieldCLI:
    """Main CLI for Deploy-Shield"""
    
    def __init__(self):
        self.validators = {
            'env': EnvironmentValidator(),
            'ports': PortValidator(),
            'permissions': PermissionValidator(),
            'database': DatabaseValidator(),
            'ssl': SSLValidator(),
            'resources': ResourceValidator(),
            'config': ConfigValidator()
        }
        
        self.fixers = {
            'env': EnvironmentFixer(),
            'dockerfile': DockerfileFixer(),
            'compose': DockerComposeFixer()
        }
        
        self.testers = {
            'connection': ConnectionTester(),
            'health': HealthChecker()
        }
    
    def validate(self,
                 aspect: Optional[str] = None,
                 mode: str = 'auto',
                 strict: bool = False) -> int:
        """Validate deployment configuration"""
        
        console.print("\n🔍 [bold cyan]Validating Deployment Configuration[/bold cyan]\n")
        
        # Determine which validators to run
        if aspect:
            validators_to_run = {aspect: self.validators[aspect]}
        else:
            validators_to_run = self.validators
        
        all_issues = []
        
        # Run validators
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for name, validator in validators_to_run.items():
                task = progress.add_task(f"Checking {name}...", total=None)
                
                try:
                    issues = validator.validate()
                    all_issues.extend(issues)
                    
                    # Show result
                    progress.remove_task(task)
                    status = self._get_status_icon(issues)
                    console.print(f"{name.capitalize():.<30} {status}")
                    
                except Exception as e:
                    progress.remove_task(task)
                    console.print(f"{name.capitalize():.<30} ❌ Error: {e}")
        
        console.print()
        
        # Show summary
        if not all_issues:
            console.print("🎉 [green]All checks passed! Ready to deploy.[/green]\n")
            self._show_next_steps()
            return 0
        
        # Categorize issues
        critical = [i for i in all_issues if i['severity'] == 'critical']
        high = [i for i in all_issues if i['severity'] == 'high']
        medium = [i for i in all_issues if i['severity'] == 'medium']
        
        console.print(f"❌ Found {len(critical)} critical, {len(high)} high, {len(medium)} medium priority issues\n")
        
        # Show issues
        if mode == 'learn':
            self._show_issues_learn(all_issues)
        else:
            self._show_issues_summary(all_issues)
        
        return 1
    
    def fix(self, mode: str = 'auto', dry_run: bool = False) -> int:
        """Fix deployment issues"""
        
        console.print("\n⚡ [bold green]Deploy-Shield Fix Mode[/bold green]\n")
        
        if dry_run:
            console.print("[yellow]Dry run - no changes will be made[/yellow]\n")
        
        # First validate to find issues
        console.print("Finding issues...\n")
        
        all_issues = []
        for validator in self.validators.values():
            try:
                issues = validator.validate()
                all_issues.extend(issues)
            except Exception as e:
                console.print(f"[red]Validation error: {e}[/red]")
        
        if not all_issues:
            console.print("✅ [green]No issues found![/green]")
            return 0
        
        console.print(f"Found {len(all_issues)} issues to fix\n")
        
        # Apply fixes based on mode
        if mode == 'auto':
            return self._auto_fix(all_issues, dry_run)
        elif mode == 'review':
            return self._review_fix(all_issues, dry_run)
        else:
            console.print(f"[red]Unknown mode: {mode}[/red]")
            return 1
    
    def test(self) -> int:
        """Test deployment readiness"""
        
        console.print("\n🧪 [bold cyan]Testing Deployment[/bold cyan]\n")
        
        results = []
        
        # Test database connection
        console.print("Testing database connection...")
        db_result = self.testers['connection'].test_database()
        results.append(('Database', db_result))
        
        # Test health endpoint
        console.print("Testing health endpoint...")
        health_result = self.testers['health'].check_health()
        results.append(('Health Check', health_result))
        
        # Show results
        console.print("\n📊 Test Results:\n")
        
        table = Table(show_header=True)
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        all_passed = True
        for test_name, result in results:
            if result['success']:
                status = "✅ Pass"
            else:
                status = "❌ Fail"
                all_passed = False
            
            table.add_row(test_name, status, result.get('message', ''))
        
        console.print(table)
        console.print()
        
        if all_passed:
            console.print("🎉 [green]All tests passed![/green]")
            return 0
        else:
            console.print("❌ [red]Some tests failed[/red]")
            return 1
    
    def checklist(self) -> int:
        """Show pre-deploy checklist"""
        
        console.print("\n📋 [bold cyan]Pre-Deploy Checklist[/bold cyan]\n")
        
        # Run all validators
        categories = {
            'Environment': ['env'],
            'Networking': ['ports', 'database'],
            'Resources': ['resources'],
            'Security': ['ssl', 'permissions'],
            'Configuration': ['config']
        }
        
        tree = Tree("Deployment Readiness")
        
        for category, validator_names in categories.items():
            category_branch = tree.add(f"[bold]{category}[/bold]")
            
            for name in validator_names:
                validator = self.validators[name]
                issues = validator.validate()
                
                if not issues:
                    category_branch.add(f"✅ {name.capitalize()}")
                else:
                    critical_count = len([i for i in issues if i['severity'] == 'critical'])
                    if critical_count > 0:
                        category_branch.add(f"❌ {name.capitalize()} ({critical_count} critical)")
                    else:
                        category_branch.add(f"⚠️  {name.capitalize()} ({len(issues)} warnings)")
        
        console.print(tree)
        console.print()
        
        return 0
    
    def generate(self, config_type: str) -> int:
        """Generate deployment configuration"""
        
        console.print(f"\n🏗️  [bold cyan]Generating {config_type}[/bold cyan]\n")
        
        if config_type == 'dockerfile':
            from .generators.docker_generator import DockerGenerator
            generator = DockerGenerator()
            path = generator.generate_dockerfile()
            console.print(f"✅ Generated: {path}")
        
        elif config_type == 'docker-compose':
            from .generators.docker_generator import DockerGenerator
            generator = DockerGenerator()
            path = generator.generate_docker_compose()
            console.print(f"✅ Generated: {path}")
        
        elif config_type == 'env':
            from .generators.env_generator import EnvGenerator
            generator = EnvGenerator()
            path = generator.generate_env()
            console.print(f"✅ Generated: {path}")
        
        elif config_type == 'k8s':
            from .generators.k8s_generator import K8sGenerator
            generator = K8sGenerator()
            paths = generator.generate_k8s_manifests()
            console.print(f"✅ Generated {len(paths)} Kubernetes manifests")
        
        else:
            console.print(f"[red]Unknown config type: {config_type}[/red]")
            return 1
        
        return 0
    
    def _get_status_icon(self, issues: List[dict]) -> str:
        """Get status icon based on issues"""
        if not issues:
            return "✅ All good"
        
        critical = len([i for i in issues if i['severity'] == 'critical'])
        high = len([i for i in issues if i['severity'] == 'high'])
        
        if critical > 0:
            return f"❌ {critical} critical"
        elif high > 0:
            return f"⚠️  {high} issues"
        else:
            return f"⚠️  {len(issues)} warnings"
    
    def _show_issues_learn(self, issues: List[dict]):
        """Show issues in learn mode with explanations"""
        for i, issue in enumerate(issues[:5], 1):
            console.print(f"\n[bold]Issue {i}/{len(issues)}:[/bold] {issue['type']}")
            console.print(f"  Severity: {issue['severity']}")
            console.print(f"  Message: {issue['message']}")
            
            if 'explanation' in issue:
                console.print(f"\n  💡 {issue['explanation']}")
            
            if 'recommendation' in issue:
                console.print(f"  🔧 {issue['recommendation']}")
        
        if len(issues) > 5:
            console.print(f"\n... and {len(issues) - 5} more issues")
    
    def _show_issues_summary(self, issues: List[dict]):
        """Show issues summary"""
        by_type = {}
        for issue in issues:
            issue_type = issue['type']
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append(issue)
        
        for issue_type, type_issues in by_type.items():
            console.print(f"\n[bold]{issue_type}:[/bold] {len(type_issues)} issues")
            for issue in type_issues[:3]:
                console.print(f"  • {issue['message']}")
            if len(type_issues) > 3:
                console.print(f"  ... and {len(type_issues) - 3} more")
    
    def _auto_fix(self, issues: List[dict], dry_run: bool) -> int:
        """Auto-fix all issues"""
        console.print("⚡ Auto-fixing issues...\n")
        
        fixed = 0
        for issue in issues:
            if 'fix' in issue and callable(issue['fix']):
                try:
                    if not dry_run:
                        issue['fix']()
                    console.print(f"✅ Fixed: {issue['message']}")
                    fixed += 1
                except Exception as e:
                    console.print(f"❌ Could not fix: {issue['message']} - {e}")
        
        console.print(f"\n📊 Results: {fixed}/{len(issues)} issues fixed\n")
        
        # Re-validate
        console.print("🎯 Re-validating...\n")
        return self.validate()
    
    def _review_fix(self, issues: List[dict], dry_run: bool) -> int:
        """Review and fix issues interactively"""
        fixed = 0
        
        for i, issue in enumerate(issues, 1):
            console.print(f"\n[bold]Fix {i}/{len(issues)}:[/bold] {issue['message']}")
            
            if 'fix_preview' in issue:
                console.print(f"\nProposed change:")
                console.print(issue['fix_preview'])
            
            choice = console.input("\nApply? [y/n/s/q]: ").lower()
            
            if choice == 'y':
                if 'fix' in issue and callable(issue['fix']):
                    try:
                        if not dry_run:
                            issue['fix']()
                        console.print("[green]✅ Applied[/green]")
                        fixed += 1
                    except Exception as e:
                        console.print(f"[red]❌ Error: {e}[/red]")
            elif choice == 's':
                console.print("[yellow]⏭️  Skipped[/yellow]")
            elif choice == 'q':
                break
        
        console.print(f"\n📊 Results: {fixed}/{len(issues)} issues fixed")
        return 0
    
    def _show_next_steps(self):
        """Show next steps after successful validation"""
        console.print("🚀 [bold]Next steps:[/bold]")
        console.print("   docker-compose up -d")
        console.print("   # or")
        console.print("   kubectl apply -f k8s/")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Deploy-Shield: Validate deployments before they fail",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate deployment')
    validate_parser.add_argument('--env', action='store_true', help='Check environment only')
    validate_parser.add_argument('--ports', action='store_true', help='Check ports only')
    validate_parser.add_argument('--permissions', action='store_true', help='Check permissions only')
    validate_parser.add_argument('--database', action='store_true', help='Check database only')
    validate_parser.add_argument('--ssl', action='store_true', help='Check SSL only')
    validate_parser.add_argument('--mode', choices=['auto', 'learn'], default='auto')
    validate_parser.add_argument('--strict', action='store_true', help='Strict validation')
    
    # Fix command
    fix_parser = subparsers.add_parser('fix', help='Fix deployment issues')
    fix_parser.add_argument('--mode', choices=['auto', 'review'], default='auto')
    fix_parser.add_argument('--dry-run', action='store_true', help='Preview changes')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test deployment readiness')
    
    # Checklist command
    checklist_parser = subparsers.add_parser('checklist', help='Show pre-deploy checklist')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate config files')
    generate_parser.add_argument('--dockerfile', action='store_true', help='Generate Dockerfile')
    generate_parser.add_argument('--docker-compose', action='store_true', help='Generate docker-compose.yml')
    generate_parser.add_argument('--env', action='store_true', help='Generate .env')
    generate_parser.add_argument('--k8s', action='store_true', help='Generate Kubernetes manifests')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = DeployShieldCLI()
    
    try:
        if args.command == 'validate':
            # Determine which aspect to validate
            aspect = None
            if args.env:
                aspect = 'env'
            elif args.ports:
                aspect = 'ports'
            elif args.permissions:
                aspect = 'permissions'
            elif args.database:
                aspect = 'database'
            elif args.ssl:
                aspect = 'ssl'
            
            return cli.validate(aspect, args.mode, args.strict)
        
        elif args.command == 'fix':
            return cli.fix(args.mode, args.dry_run)
        
        elif args.command == 'test':
            return cli.test()
        
        elif args.command == 'checklist':
            return cli.checklist()
        
        elif args.command == 'generate':
            if args.dockerfile:
                return cli.generate('dockerfile')
            elif args.docker_compose:
                return cli.generate('docker-compose')
            elif args.env:
                return cli.generate('env')
            elif args.k8s:
                return cli.generate('k8s')
            else:
                console.print("[red]Please specify what to generate[/red]")
                return 1
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if '--debug' in sys.argv:
            raise
        return 1


if __name__ == '__main__':
    sys.exit(main())
