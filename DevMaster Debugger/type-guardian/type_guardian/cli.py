#!/usr/bin/env python3
"""
Type-Guardian CLI - Main entry point
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from .runner import TypeGuardianRunner
from .parser import MypyParser
from .database.type_patterns import TypePatternDatabase

console = Console()


class TypeGuardianCLI:
    """Main CLI interface for Type-Guardian"""
    
    def __init__(self):
        self.runner = TypeGuardianRunner()
        self.parser = MypyParser()
        self.db = TypePatternDatabase()
        
    def fix(self, 
            path: Optional[str] = None,
            mode: str = 'auto',
            dry_run: bool = False,
            interactive: bool = False,
            strict: bool = False) -> int:
        """Fix type errors"""
        
        target = Path(path) if path else Path.cwd()
        
        console.print("\n⚡ [bold cyan]Type-Guardian Fix Mode[/bold cyan]\n")
        console.print(f"Target: {target}")
        console.print(f"Mode: {mode}")
        
        if dry_run:
            console.print("[yellow]Dry run - no changes will be made[/yellow]\n")
        
        # Run mypy to get errors
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running mypy...", total=None)
            errors = self.runner.run_mypy(target)
            progress.remove_task(task)
        
        if not errors:
            console.print("✅ [green]No type errors found![/green]")
            return 0
        
        # Parse and categorize errors
        parsed_errors = self.parser.parse_errors(errors)
        
        console.print(f"\n🔍 Found {len(parsed_errors)} type errors\n")
        
        # Show error breakdown
        self._show_error_breakdown(parsed_errors)
        
        if mode == 'learn':
            return self._learn_mode(parsed_errors)
        elif mode == 'review':
            return self._review_mode(parsed_errors, dry_run)
        elif mode == 'auto':
            return self._auto_mode(parsed_errors, dry_run, strict)
        else:
            console.print(f"[red]Unknown mode: {mode}[/red]")
            return 1
    
    def check(self, path: Optional[str] = None) -> int:
        """Check types without fixing"""
        target = Path(path) if path else Path.cwd()
        
        console.print("\n🔍 [bold cyan]Type-Guardian Check Mode[/bold cyan]\n")
        
        errors = self.runner.run_mypy(target)
        
        if not errors:
            console.print("✅ [green]No type errors found![/green]")
            return 0
        
        parsed_errors = self.parser.parse_errors(errors)
        console.print(f"\n❌ Found {len(parsed_errors)} type errors\n")
        
        self._show_error_breakdown(parsed_errors)
        
        return 1
    
    def annotate(self, path: str) -> int:
        """Add type hints to untyped code"""
        target = Path(path)
        
        console.print("\n📝 [bold cyan]Type-Guardian Annotate Mode[/bold cyan]\n")
        console.print(f"Target: {target}\n")
        
        # Find files without type hints
        files_to_annotate = self.runner.find_untyped_files(target)
        
        console.print(f"Found {len(files_to_annotate)} files needing annotations\n")
        
        # Add annotations
        results = self.runner.add_annotations(files_to_annotate)
        
        self._show_annotation_results(results)
        
        return 0
    
    def stub(self, path: str) -> int:
        """Generate stub files"""
        target = Path(path)
        
        console.print("\n📄 [bold cyan]Type-Guardian Stub Mode[/bold cyan]\n")
        console.print(f"Target: {target}\n")
        
        stubs = self.runner.generate_stubs(target)
        
        console.print(f"✅ Generated {len(stubs)} stub files\n")
        
        return 0
    
    def _show_error_breakdown(self, errors: List[dict]):
        """Show breakdown of error types"""
        from collections import Counter
        
        error_types = Counter(e['category'] for e in errors)
        
        table = Table(title="Error Breakdown", show_header=True)
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        
        for error_type, count in error_types.most_common():
            table.add_row(error_type, str(count))
        
        console.print(table)
        console.print()
    
    def _learn_mode(self, errors: List[dict]) -> int:
        """Learn mode - show errors with explanations"""
        console.print("\n🎓 [bold yellow]Learn Mode[/bold yellow]\n")
        
        for i, error in enumerate(errors[:5], 1):  # Show first 5
            console.print(f"[bold]Error {i}/{len(errors)}:[/bold]")
            console.print(f"  File: {error['file']}:{error['line']}")
            console.print(f"  Type: {error['category']}")
            console.print(f"  Message: {error['message']}\n")
            
            # Show code context
            if 'code_context' in error:
                console.print("  Code:")
                console.print(f"    {error['code_context']}\n")
            
            # Show suggested fix
            fix = self.db.suggest_fix(error)
            if fix:
                console.print("  💡 Suggested Fix:")
                console.print(f"    {fix['description']}\n")
                if 'example' in fix:
                    console.print("  Example:")
                    console.print(f"    {fix['example']}\n")
        
        if len(errors) > 5:
            console.print(f"... and {len(errors) - 5} more errors")
        
        return 0
    
    def _review_mode(self, errors: List[dict], dry_run: bool) -> int:
        """Review mode - interactive fix approval"""
        console.print("\n🔍 [bold yellow]Review Mode[/bold yellow]\n")
        
        fixes_applied = 0
        fixes_skipped = 0
        
        for i, error in enumerate(errors, 1):
            console.print(f"\n[bold]Fix {i}/{len(errors)}:[/bold] {error['category']}")
            console.print(f"File: {error['file']}:{error['line']}\n")
            
            fix = self.db.get_fix(error)
            if not fix:
                console.print("[yellow]No automatic fix available[/yellow]")
                fixes_skipped += 1
                continue
            
            # Show diff
            console.print("Proposed change:")
            console.print(f"[red]- {fix['old']}[/red]")
            console.print(f"[green]+ {fix['new']}[/green]\n")
            
            # Ask user
            choice = console.input("Apply? [y/n/s/q]: ").lower()
            
            if choice == 'y':
                if not dry_run:
                    self.runner.apply_fix(fix)
                fixes_applied += 1
                console.print("[green]✅ Applied[/green]")
            elif choice == 's':
                fixes_skipped += 1
                console.print("[yellow]⏭️  Skipped[/yellow]")
            elif choice == 'q':
                break
            else:
                fixes_skipped += 1
        
        console.print(f"\n📊 Results:")
        console.print(f"  Applied: {fixes_applied}")
        console.print(f"  Skipped: {fixes_skipped}")
        
        return 0
    
    def _auto_mode(self, errors: List[dict], dry_run: bool, strict: bool) -> int:
        """Auto mode - fix all errors automatically"""
        console.print("\n⚡ [bold green]Auto Mode[/bold green]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Applying fixes...", total=len(errors))
            
            results = self.runner.auto_fix_all(errors, dry_run=dry_run, strict=strict)
            
            progress.update(task, completed=len(errors))
        
        self._show_fix_results(results)
        
        # Re-run mypy
        console.print("\n🎯 Re-running mypy...")
        remaining_errors = self.runner.run_mypy(Path.cwd())
        
        if not remaining_errors:
            console.print("✅ [green]Success: no issues found![/green]")
        else:
            console.print(f"⚠️  [yellow]{len(remaining_errors)} errors remaining[/yellow]")
        
        return 0
    
    def _show_fix_results(self, results: dict):
        """Show results of auto-fixing"""
        console.print("\n📊 [bold]Results:[/bold]\n")
        
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("✅ Type hints added", str(results.get('hints_added', 0)))
        table.add_row("✅ Optional issues fixed", str(results.get('optional_fixed', 0)))
        table.add_row("✅ Typing imports added", str(results.get('imports_added', 0)))
        table.add_row("✅ Generic types fixed", str(results.get('generics_fixed', 0)))
        table.add_row("⏱️  Time", f"{results.get('time', 0):.1f}s")
        
        console.print(table)
        
        if results.get('manual_review'):
            console.print(f"\n⚠️  {len(results['manual_review'])} issues need manual review")
    
    def _show_annotation_results(self, results: dict):
        """Show results of annotation process"""
        table = Table(title="Annotation Results")
        table.add_column("File", style="cyan")
        table.add_column("Hints Added", style="green", justify="right")
        
        for file, count in results.items():
            table.add_row(str(file), str(count))
        
        console.print(table)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Type-Guardian: Auto-fix type errors and add type hints",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Fix command
    fix_parser = subparsers.add_parser('fix', help='Fix type errors')
    fix_parser.add_argument('path', nargs='?', help='Path to fix (default: current directory)')
    fix_parser.add_argument('--mode', choices=['auto', 'review', 'learn'], default='auto',
                           help='Fix mode (default: auto)')
    fix_parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    fix_parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    fix_parser.add_argument('--strict', action='store_true', help='Strict mode (no Any types)')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check types without fixing')
    check_parser.add_argument('path', nargs='?', help='Path to check')
    
    # Annotate command
    annotate_parser = subparsers.add_parser('annotate', help='Add type hints to untyped code')
    annotate_parser.add_argument('path', help='Path to annotate')
    
    # Stub command
    stub_parser = subparsers.add_parser('stub', help='Generate stub files')
    stub_parser.add_argument('path', help='Path to generate stubs for')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = TypeGuardianCLI()
    
    try:
        if args.command == 'fix':
            return cli.fix(args.path, args.mode, args.dry_run, args.interactive, args.strict)
        elif args.command == 'check':
            return cli.check(args.path)
        elif args.command == 'annotate':
            return cli.annotate(args.path)
        elif args.command == 'stub':
            return cli.stub(args.path)
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
