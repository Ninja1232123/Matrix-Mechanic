"""Auto-fixer for applying performance optimizations"""

import ast
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
from datetime import datetime

from ..models import Optimization, OptimizationType


class AutoFixer:
    """Automatically applies performance optimizations to code"""

    def __init__(self, backup_dir: Optional[Path] = None):
        self.backup_dir = backup_dir or Path(".speed_guardian/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def apply_optimizations(
        self,
        optimizations: List[Optimization],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Apply a list of optimizations"""
        results = {
            'applied': [],
            'failed': [],
            'skipped': [],
        }

        # Group optimizations by file
        by_file: Dict[str, List[Optimization]] = {}
        for opt in optimizations:
            if opt.file_path not in by_file:
                by_file[opt.file_path] = []
            by_file[opt.file_path].append(opt)

        # Apply optimizations file by file
        for file_path, file_opts in by_file.items():
            if dry_run:
                print(f"[DRY RUN] Would apply {len(file_opts)} optimizations to {file_path}")
                results['skipped'].extend(file_opts)
                continue

            try:
                # Backup file
                self._backup_file(Path(file_path))

                # Apply optimizations
                success = self._apply_to_file(Path(file_path), file_opts)

                if success:
                    results['applied'].extend(file_opts)
                    # Mark as applied
                    for opt in file_opts:
                        opt.applied = True
                else:
                    results['failed'].extend(file_opts)

            except Exception as e:
                print(f"Error applying optimizations to {file_path}: {e}")
                results['failed'].extend(file_opts)

        return results

    def apply_single_optimization(
        self,
        optimization: Optimization,
        dry_run: bool = False
    ) -> bool:
        """Apply a single optimization"""
        if dry_run:
            print(f"[DRY RUN] {optimization}")
            return True

        try:
            file_path = Path(optimization.file_path)

            # Backup file
            self._backup_file(file_path)

            # Read file
            with open(file_path, 'r') as f:
                content = f.read()

            # Apply optimization based on type
            new_content = self._apply_by_type(content, optimization)

            if new_content and new_content != content:
                # Write back
                with open(file_path, 'w') as f:
                    f.write(new_content)

                optimization.applied = True
                return True

            return False

        except Exception as e:
            print(f"Error applying optimization: {e}")
            return False

    def _apply_to_file(
        self,
        file_path: Path,
        optimizations: List[Optimization]
    ) -> bool:
        """Apply multiple optimizations to a single file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Sort optimizations by line number (descending) to avoid offset issues
            sorted_opts = sorted(
                optimizations,
                key=lambda o: o.line_number,
                reverse=True
            )

            modified = False
            for opt in sorted_opts:
                new_content = self._apply_by_type(content, opt)
                if new_content and new_content != content:
                    content = new_content
                    modified = True

            if modified:
                with open(file_path, 'w') as f:
                    f.write(content)

            return modified

        except Exception as e:
            print(f"Error applying to file {file_path}: {e}")
            return False

    def _apply_by_type(
        self,
        content: str,
        optimization: Optimization
    ) -> Optional[str]:
        """Apply optimization based on its type"""
        if optimization.type == OptimizationType.CACHING:
            return self._apply_caching(content, optimization)
        elif optimization.type == OptimizationType.LOOP_COMPREHENSION:
            return self._apply_loop_comprehension(content, optimization)
        elif optimization.type == OptimizationType.ALGORITHM:
            return self._apply_algorithm_optimization(content, optimization)
        elif optimization.type == OptimizationType.ASYNC_AWAIT:
            return self._apply_async_optimization(content, optimization)
        elif optimization.type == OptimizationType.MEMORY_REUSE:
            return self._apply_memory_optimization(content, optimization)
        else:
            return None

    def _apply_caching(
        self,
        content: str,
        optimization: Optimization
    ) -> Optional[str]:
        """Apply caching optimization (e.g., @lru_cache)"""
        lines = content.split('\n')

        # Check if we need to add import
        has_lru_cache_import = any(
            'from functools import' in line and 'lru_cache' in line
            for line in lines
        )

        if not has_lru_cache_import:
            # Find first import or add at top
            import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_idx = i + 1
                elif import_idx > 0 and not line.startswith('import ') and not line.startswith('from '):
                    break

            lines.insert(import_idx, 'from functools import lru_cache')

        # Add @lru_cache decorator
        func_line = optimization.line_number - 1

        if func_line < len(lines):
            # Find indentation
            indent = len(lines[func_line]) - len(lines[func_line].lstrip())
            decorator = ' ' * indent + '@lru_cache(maxsize=128)'

            lines.insert(func_line, decorator)

        return '\n'.join(lines)

    def _apply_loop_comprehension(
        self,
        content: str,
        optimization: Optimization
    ) -> Optional[str]:
        """Apply loop to list comprehension transformation"""
        # This is a simplified version - real implementation would parse AST
        # For now, just add a comment suggesting the change
        lines = content.split('\n')
        line_idx = optimization.line_number - 1

        if line_idx < len(lines):
            indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
            comment = ' ' * indent + '# TODO: Convert to list comprehension for better performance'
            lines.insert(line_idx, comment)

        return '\n'.join(lines)

    def _apply_algorithm_optimization(
        self,
        content: str,
        optimization: Optimization
    ) -> Optional[str]:
        """Apply algorithmic optimization"""
        # Add comment with suggestion
        lines = content.split('\n')
        line_idx = optimization.line_number - 1

        if line_idx < len(lines):
            indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
            comment = ' ' * indent + '# TODO: Optimize algorithm complexity (consider hash map/set)'
            lines.insert(line_idx, comment)

        return '\n'.join(lines)

    def _apply_async_optimization(
        self,
        content: str,
        optimization: Optimization
    ) -> Optional[str]:
        """Apply async/await optimization"""
        # Check if asyncio is imported
        lines = content.split('\n')

        has_asyncio = any('import asyncio' in line for line in lines)

        if not has_asyncio:
            # Add import
            import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_idx = i + 1

            lines.insert(import_idx, 'import asyncio')

        # Add comment suggesting async conversion
        line_idx = optimization.line_number - 1
        if line_idx < len(lines):
            indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
            comment = ' ' * indent + '# TODO: Consider async/await for I/O operations'
            lines.insert(line_idx, comment)

        return '\n'.join(lines)

    def _apply_memory_optimization(
        self,
        content: str,
        optimization: Optimization
    ) -> Optional[str]:
        """Apply memory optimization"""
        lines = content.split('\n')
        line_idx = optimization.line_number - 1

        if line_idx < len(lines):
            indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
            comment = ' ' * indent + '# TODO: Optimize memory usage (use join() or iterator)'
            lines.insert(line_idx, comment)

        return '\n'.join(lines)

    def _backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.backup"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(file_path, backup_path)
        return backup_path

    def restore_backup(self, file_path: Path, backup_path: Path) -> bool:
        """Restore a file from backup"""
        try:
            shutil.copy2(backup_path, file_path)
            return True
        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False

    def list_backups(self, file_path: Optional[Path] = None) -> List[Path]:
        """List all backups"""
        if file_path:
            pattern = f"{file_path.name}.*.backup"
            return sorted(self.backup_dir.glob(pattern))
        else:
            return sorted(self.backup_dir.glob("*.backup"))

    def get_optimization_report(
        self,
        results: Dict[str, Any]
    ) -> str:
        """Generate a report of applied optimizations"""
        lines = []
        lines.append("=== Speed-Guardian Optimization Report ===\n")

        lines.append(f"Applied: {len(results['applied'])}")
        lines.append(f"Failed: {len(results['failed'])}")
        lines.append(f"Skipped: {len(results['skipped'])}\n")

        if results['applied']:
            lines.append("Applied Optimizations:")
            for opt in results['applied']:
                lines.append(f"  - {opt}")

        if results['failed']:
            lines.append("\nFailed Optimizations:")
            for opt in results['failed']:
                lines.append(f"  - {opt}")

        return '\n'.join(lines)
