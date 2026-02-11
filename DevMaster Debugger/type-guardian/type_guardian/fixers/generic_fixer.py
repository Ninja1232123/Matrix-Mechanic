"""
Generic Fixer - Fix generic type parameters (List[T], Dict[K, V], etc.)
"""

import ast
from typing import Tuple, Optional


class GenericFixer:
    """Fix generic type parameter errors"""
    
    def can_fix(self, error: dict, tree: ast.AST) -> bool:
        """Check if we can fix this error"""
        return error['category'] == 'generic_type'
    
    def fix(self, error: dict, tree: ast.AST) -> Tuple[ast.AST, bool]:
        """
        Fix generic type error
        
        Args:
            error: Error dictionary
            tree: AST tree
            
        Returns:
            (modified_tree, success)
        """
        line = error['line']
        
        # Find the node
        node = self._find_node_at_line(tree, line)
        
        if not node:
            return tree, False
        
        if isinstance(node, ast.FunctionDef):
            return self._fix_generic_function(node, tree, error)
        
        return tree, False
    
    def _find_node_at_line(self, tree: ast.AST, line: int) -> Optional[ast.AST]:
        """Find AST node at given line number"""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line:
                return node
        return None
    
    def _fix_generic_function(self, node: ast.FunctionDef, tree: ast.AST, error: dict) -> Tuple[ast.AST, bool]:
        """
        Fix generic function parameters
        
        Example:
            def first(items: List) -> item:  # Need List[T]
                return items[0] if items else None
        
        Becomes:
            T = TypeVar('T')
            def first(items: List[T]) -> Optional[T]:
                return items[0] if items else None
        """
        # Check if function uses generics
        type_var_name = self._find_or_create_typevar(tree, node)
        
        if not type_var_name:
            return tree, False
        
        # Update parameter annotations
        modified = False
        for arg in node.args.args:
            if arg.annotation:
                # Check if it's a generic type that needs parameters
                if self._needs_type_param(arg.annotation):
                    arg.annotation = self._add_type_param(arg.annotation, type_var_name)
                    modified = True
        
        # Update return annotation
        if node.returns and self._needs_type_param(node.returns):
            node.returns = self._add_type_param(node.returns, type_var_name)
            modified = True
        
        return tree, modified
    
    def _find_or_create_typevar(self, tree: ast.AST, func: ast.FunctionDef) -> Optional[str]:
        """Find existing TypeVar or create new one"""
        # Look for existing TypeVar definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id == 'TypeVar':
                            # Found existing TypeVar
                            if node.targets and isinstance(node.targets[0], ast.Name):
                                return node.targets[0].id
        
        # Create new TypeVar
        type_var_name = 'T'
        
        # Find where to insert (after imports, before first function)
        insert_pos = 0
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_pos = i + 1
            elif isinstance(node, ast.FunctionDef):
                break
        
        # Create TypeVar assignment
        typevar_assign = ast.Assign(
            targets=[ast.Name(id=type_var_name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='TypeVar', ctx=ast.Load()),
                args=[ast.Constant(value=type_var_name)],
                keywords=[]
            )
        )
        
        tree.body.insert(insert_pos, typevar_assign)
        
        return type_var_name
    
    def _needs_type_param(self, annotation: ast.AST) -> bool:
        """Check if annotation needs type parameters"""
        if isinstance(annotation, ast.Name):
            # Check if it's a generic type
            generic_types = {'List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union'}
            return annotation.id in generic_types
        
        if isinstance(annotation, ast.Subscript):
            # Already has type parameters
            return False
        
        return False
    
    def _add_type_param(self, annotation: ast.AST, type_var: str) -> ast.AST:
        """Add type parameter to annotation"""
        if isinstance(annotation, ast.Name):
            generic_type = annotation.id
            
            # Create subscript with type parameter
            if generic_type in {'List', 'Set'}:
                # List[T], Set[T]
                return ast.Subscript(
                    value=ast.Name(id=generic_type, ctx=ast.Load()),
                    slice=ast.Name(id=type_var, ctx=ast.Load()),
                    ctx=ast.Load()
                )
            elif generic_type == 'Dict':
                # Dict[str, T] - assume string keys
                return ast.Subscript(
                    value=ast.Name(id='Dict', ctx=ast.Load()),
                    slice=ast.Tuple(
                        elts=[
                            ast.Name(id='str', ctx=ast.Load()),
                            ast.Name(id=type_var, ctx=ast.Load())
                        ],
                        ctx=ast.Load()
                    ),
                    ctx=ast.Load()
                )
            elif generic_type == 'Optional':
                # Optional[T]
                return ast.Subscript(
                    value=ast.Name(id='Optional', ctx=ast.Load()),
                    slice=ast.Name(id=type_var, ctx=ast.Load()),
                    ctx=ast.Load()
                )
        
        return annotation
    
    def _replace_node(self, tree: ast.AST, old_node: ast.AST, new_node: ast.AST):
        """Replace old node with new node in tree"""
        for parent in ast.walk(tree):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if item is old_node:
                            value[i] = new_node
                            return
                elif value is old_node:
                    setattr(parent, field, new_node)
                    return
