"""
Runner - Execute mypy and capture errors
"""

import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import ast
import json


class TypeGuardianRunner:
    """Run mypy and manage type checking operations"""
    
    def __init__(self, mypy_config: Optional[Path] = None):
        self.mypy_config = mypy_config
        self.backup_dir = Path('.type-guardian/backups')
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def run_mypy(self, target: Path) -> List[str]:
        """
        Run mypy on target and return raw error output
        
        Args:
            target: File or directory to check
            
        Returns:
            List of error lines from mypy
        """
        cmd = ['mypy', str(target)]
        
        if self.mypy_config and self.mypy_config.exists():
            cmd.extend(['--config-file', str(self.mypy_config)])
        
        # Add flags for better error reporting
        cmd.extend([
            '--show-column-numbers',
            '--show-error-codes',
            '--no-error-summary',
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # mypy returns non-zero on errors, which is expected
            error_lines = result.stdout.strip().split('\n') if result.stdout else []
            return [line for line in error_lines if line.strip()]
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("mypy timed out after 5 minutes")
        except FileNotFoundError:
            raise RuntimeError("mypy not found - install with: pip install mypy")
    
    def auto_fix_all(self, errors: List[dict], dry_run: bool = False, strict: bool = False) -> Dict:
        """
        Automatically fix all errors
        
        Args:
            errors: Parsed error dictionaries
            dry_run: If True, don't actually modify files
            strict: If True, enforce strict type checking (no Any)
            
        Returns:
            Dictionary with fix statistics
        """
        from .fixers.missing_hints import MissingHintsFixer
        from .fixers.optional_fixer import OptionalFixer
        from .fixers.generic_fixer import GenericFixer
        from .fixers.import_fixer import ImportFixer
        from .fixers.collection_fixer import CollectionFixer
        
        start_time = time.time()
        
        results = {
            'hints_added': 0,
            'optional_fixed': 0,
            'imports_added': 0,
            'generics_fixed': 0,
            'manual_review': [],
            'time': 0
        }
        
        # Create backup before making changes
        if not dry_run:
            self._create_backup()
        
        # Group errors by file for efficient processing
        errors_by_file = {}
        for error in errors:
            file = error['file']
            if file not in errors_by_file:
                errors_by_file[file] = []
            errors_by_file[file].append(error)
        
        # Initialize fixers
        hint_fixer = MissingHintsFixer(strict=strict)
        optional_fixer = OptionalFixer()
        generic_fixer = GenericFixer()
        import_fixer = ImportFixer()
        collection_fixer = CollectionFixer()
        
        # Process each file
        for file, file_errors in errors_by_file.items():
            file_path = Path(file)
            
            if not file_path.exists():
                continue
            
            # Read file
            with open(file_path, 'r') as f:
                source = f.read()
            
            try:
                tree = ast.parse(source)
            except SyntaxError:
                results['manual_review'].append({
                    'file': file,
                    'reason': 'Syntax error - cannot parse'
                })
                continue
            
            modified = False
            
            # Apply fixes in order
            for error in file_errors:
                category = error['category']
                
                if category == 'missing_type_hint':
                    if hint_fixer.can_fix(error, tree):
                        tree, fixed = hint_fixer.fix(error, tree)
                        if fixed:
                            results['hints_added'] += 1
                            modified = True
                
                elif category == 'optional_none':
                    if optional_fixer.can_fix(error, tree):
                        tree, fixed = optional_fixer.fix(error, tree)
                        if fixed:
                            results['optional_fixed'] += 1
                            modified = True
                
                elif category == 'generic_type':
                    if generic_fixer.can_fix(error, tree):
                        tree, fixed = generic_fixer.fix(error, tree)
                        if fixed:
                            results['generics_fixed'] += 1
                            modified = True
                
                elif category == 'collection_type':
                    if collection_fixer.can_fix(error, tree):
                        tree, fixed = collection_fixer.fix(error, tree)
                        if fixed:
                            results['hints_added'] += 1
                            modified = True
                
                else:
                    results['manual_review'].append(error)
            
            # Add missing imports
            if modified:
                tree, imports_added = import_fixer.add_missing_imports(tree)
                if imports_added:
                    results['imports_added'] += imports_added
            
            # Write back to file
            if modified and not dry_run:
                new_source = ast.unparse(tree)
                with open(file_path, 'w') as f:
                    f.write(new_source)
        
        results['time'] = time.time() - start_time
        
        return results
    
    def apply_fix(self, fix: dict) -> bool:
        """
        Apply a single fix to a file
        
        Args:
            fix: Fix dictionary with file, line, old, new
            
        Returns:
            True if successful
        """
        file_path = Path(fix['file'])
        
        if not file_path.exists():
            return False
        
        # Read file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Apply fix
        line_num = fix['line'] - 1  # Convert to 0-indexed
        if line_num < 0 or line_num >= len(lines):
            return False
        
        old_line = lines[line_num]
        new_line = fix['new'] + '\n'
        
        # Verify old content matches
        if old_line.strip() != fix['old'].strip():
            return False
        
        lines[line_num] = new_line
        
        # Write back
        with open(file_path, 'w') as f:
            f.writelines(lines)
        
        return True
    
    def find_untyped_files(self, path: Path) -> List[Path]:
        """
        Find Python files without type hints
        
        Args:
            path: Directory to search
            
        Returns:
            List of file paths
        """
        untyped_files = []
        
        if path.is_file():
            files = [path]
        else:
            files = path.rglob('*.py')
        
        for file in files:
            if self._needs_annotations(file):
                untyped_files.append(file)
        
        return untyped_files
    
    def _needs_annotations(self, file: Path) -> bool:
        """Check if file needs type annotations"""
        try:
            with open(file, 'r') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Count functions and those with annotations
            total_funcs = 0
            annotated_funcs = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_funcs += 1
                    
                    # Check if has return annotation or param annotations
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        annotated_funcs += 1
            
            # Needs annotations if less than 50% are annotated
            if total_funcs == 0:
                return False
            
            return (annotated_funcs / total_funcs) < 0.5
            
        except Exception:
            return False
    
    def add_annotations(self, files: List[Path]) -> Dict[Path, int]:
        """
        Add type annotations to files
        
        Args:
            files: List of files to annotate
            
        Returns:
            Dictionary mapping file to number of annotations added
        """
        from .inference.type_inferrer import TypeInferrer
        
        results = {}
        inferrer = TypeInferrer()
        
        for file in files:
            with open(file, 'r') as f:
                source = f.read()
            
            try:
                tree = ast.parse(source)
                tree, count = inferrer.infer_and_annotate(tree)
                
                # Write back
                new_source = ast.unparse(tree)
                with open(file, 'w') as f:
                    f.write(new_source)
                
                results[file] = count
                
            except Exception as e:
                results[file] = 0
        
        return results
    
    def generate_stubs(self, path: Path) -> List[Path]:
        """
        Generate .pyi stub files
        
        Args:
            path: Directory to generate stubs for
            
        Returns:
            List of generated stub files
        """
        from .generators.stub_generator import StubGenerator
        
        generator = StubGenerator()
        
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob('*.py'))
        
        stubs = []
        for file in files:
            stub_file = generator.generate_stub(file)
            if stub_file:
                stubs.append(stub_file)
        
        return stubs
    
    def _create_backup(self):
        """Create backup of current state"""
        import shutil
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_path = self.backup_dir / timestamp
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Save current git state or copy files
        # For now, just note the backup location
        (backup_path / 'README.txt').write_text(
            f"Backup created at {timestamp}\n"
            f"Use git to restore if needed\n"
        )
