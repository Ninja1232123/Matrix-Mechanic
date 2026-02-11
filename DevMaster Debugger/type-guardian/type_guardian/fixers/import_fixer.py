"""
Import Fixer - Add missing typing imports
"""

import ast
from typing import Set, Tuple


class ImportFixer:
    """Manage and add typing imports"""
    
    TYPING_IMPORTS = {
        'List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union', 'Any',
        'TypeVar', 'Generic', 'Protocol', 'Callable', 'Iterable',
        'Sequence', 'Mapping', 'Type', 'ClassVar', 'Final'
    }
    
    def add_missing_imports(self, tree: ast.AST) -> Tuple[ast.AST, int]:
        """
        Add missing typing imports to tree
        
        Args:
            tree: AST tree
            
        Returns:
            (modified_tree, count_added)
        """
        # Find all typing references in the tree
        needed_imports = self._find_needed_imports(tree)
        
        # Find existing imports
        existing_imports = self._find_existing_imports(tree)
        
        # Calculate what's missing
        missing = needed_imports - existing_imports
        
        if not missing:
            return tree, 0
        
        # Add import
        self._add_import_statement(tree, missing)
        
        return tree, len(missing)
    
    def _find_needed_imports(self, tree: ast.AST) -> Set[str]:
        """Find all typing names used in the tree"""
        needed = set()
        
        for node in ast.walk(tree):
            # Check annotations
            if isinstance(node, (ast.FunctionDef, ast.arg, ast.AnnAssign)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Name):
                        if child.id in self.TYPING_IMPORTS:
                            needed.add(child.id)
                    elif isinstance(child, ast.Subscript):
                        if isinstance(child.value, ast.Name):
                            if child.value.id in self.TYPING_IMPORTS:
                                needed.add(child.value.id)
            
            # Check TypeVar assignments
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id == 'TypeVar':
                            needed.add('TypeVar')
        
        return needed
    
    def _find_existing_imports(self, tree: ast.AST) -> Set[str]:
        """Find existing typing imports"""
        existing = set()
        
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.module == 'typing':
                    for alias in node.names:
                        if alias.name == '*':
                            # If importing *, assume all are imported
                            return self.TYPING_IMPORTS.copy()
                        existing.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'typing':
                        # If importing typing, assume all are available
                        return self.TYPING_IMPORTS.copy()
        
        return existing
    
    def _add_import_statement(self, tree: ast.AST, imports: Set[str]):
        """Add typing import statement to tree"""
        # Find where to insert
        insert_pos = self._find_import_position(tree)
        
        # Check if there's already a typing import we can extend
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.ImportFrom) and node.module == 'typing':
                # Add to existing import
                existing_names = {alias.name for alias in node.names}
                new_names = imports - existing_names
                
                if new_names:
                    node.names.extend([
                        ast.alias(name=name, asname=None)
                        for name in sorted(new_names)
                    ])
                return
        
        # Create new import statement
        import_node = ast.ImportFrom(
            module='typing',
            names=[ast.alias(name=name, asname=None) for name in sorted(imports)],
            level=0
        )
        
        tree.body.insert(insert_pos, import_node)
    
    def _find_import_position(self, tree: ast.AST) -> int:
        """Find appropriate position to insert import"""
        # Insert after docstring and other imports
        insert_pos = 0
        
        # Skip module docstring
        if (tree.body and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Constant)):
            insert_pos = 1
        
        # Find last import
        for i, node in enumerate(tree.body[insert_pos:], start=insert_pos):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                return i
        
        return len(tree.body)
    
    def merge_typing_imports(self, tree: ast.AST) -> ast.AST:
        """Merge multiple typing imports into single statement"""
        typing_imports = []
        other_nodes = []
        
        all_typing_names = set()
        
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'typing':
                for alias in node.names:
                    if alias.name != '*':
                        all_typing_names.add(alias.name)
            else:
                other_nodes.append(node)
        
        if all_typing_names:
            # Create single merged import
            merged_import = ast.ImportFrom(
                module='typing',
                names=[ast.alias(name=name, asname=None) for name in sorted(all_typing_names)],
                level=0
            )
            
            # Find where to insert
            insert_pos = 0
            if (other_nodes and 
                isinstance(other_nodes[0], ast.Expr) and 
                isinstance(other_nodes[0].value, ast.Constant)):
                insert_pos = 1
            
            other_nodes.insert(insert_pos, merged_import)
        
        tree.body = other_nodes
        
        return tree


class ImportOptimizer:
    """Optimize and clean up imports"""
    
    def remove_unused_imports(self, tree: ast.AST) -> ast.AST:
        """Remove unused typing imports"""
        # Find all names used in code
        used_names = self._find_used_names(tree)
        
        # Filter imports
        new_body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'typing':
                # Keep only used imports
                used_aliases = [
                    alias for alias in node.names
                    if alias.name in used_names or alias.name == '*'
                ]
                
                if used_aliases:
                    node.names = used_aliases
                    new_body.append(node)
            else:
                new_body.append(node)
        
        tree.body = new_body
        return tree
    
    def _find_used_names(self, tree: ast.AST) -> Set[str]:
        """Find all names used in the tree"""
        used = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
        
        return used
