"""
Optional Fixer - Fix Optional[T] and None handling issues
"""

import ast
from typing import Tuple, Optional


class OptionalFixer:
    """Fix Optional type and None check errors"""
    
    def can_fix(self, error: dict, tree: ast.AST) -> bool:
        """Check if we can fix this error"""
        return error['category'] == 'optional_none'
    
    def fix(self, error: dict, tree: ast.AST) -> Tuple[ast.AST, bool]:
        """
        Fix Optional/None error by adding None checks
        
        Args:
            error: Error dictionary
            tree: AST tree
            
        Returns:
            (modified_tree, success)
        """
        line = error['line']
        
        # Find the problematic node
        node = self._find_node_at_line(tree, line)
        
        if not node:
            return tree, False
        
        # Check if it's an attribute access on Optional type
        if isinstance(node, ast.Attribute):
            return self._fix_optional_attribute_access(node, tree, error)
        
        # Check if it's a method call on Optional type
        if isinstance(node, ast.Call):
            return self._fix_optional_call(node, tree, error)
        
        return tree, False
    
    def _find_node_at_line(self, tree: ast.AST, line: int) -> Optional[ast.AST]:
        """Find AST node at given line number"""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line:
                return node
        return None
    
    def _fix_optional_attribute_access(self, node: ast.Attribute, tree: ast.AST, error: dict) -> Tuple[ast.AST, bool]:
        """
        Fix Optional attribute access by adding None check
        
        Transform:
            user.email  # where user: Optional[User]
        To:
            user.email if user is not None else None
        """
        # Get the variable being accessed
        if not isinstance(node.value, ast.Name):
            return tree, False
        
        var_name = node.value.id
        
        # Check if already in a None check
        if self._is_already_guarded(tree, node, var_name):
            return tree, False
        
        # Create conditional expression
        none_check = ast.Compare(
            left=ast.Name(id=var_name, ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)]
        )
        
        conditional = ast.IfExp(
            test=none_check,
            body=node,
            orelse=ast.Constant(value=None)
        )
        
        # Replace node
        self._replace_node(tree, node, conditional)
        
        return tree, True
    
    def _fix_optional_call(self, node: ast.Call, tree: ast.AST, error: dict) -> Tuple[ast.AST, bool]:
        """Fix Optional method call"""
        # Similar to attribute access
        if isinstance(node.func, ast.Attribute):
            return self._fix_optional_attribute_access(node.func, tree, error)
        
        return tree, False
    
    def _is_already_guarded(self, tree: ast.AST, node: ast.AST, var_name: str) -> bool:
        """Check if node is already in a None check"""
        # Find parent If node
        for parent in ast.walk(tree):
            if isinstance(parent, ast.If):
                # Check if test compares var_name to None
                if self._checks_none(parent.test, var_name):
                    # Check if node is in the body
                    if self._node_in_body(node, parent.body):
                        return True
        
        return False
    
    def _checks_none(self, test: ast.AST, var_name: str) -> bool:
        """Check if test expression checks for None"""
        if isinstance(test, ast.Compare):
            # Check for "var is not None" or "var is None"
            if isinstance(test.left, ast.Name) and test.left.id == var_name:
                if any(isinstance(op, (ast.IsNot, ast.Is)) for op in test.ops):
                    if any(isinstance(c, ast.Constant) and c.value is None for c in test.comparators):
                        return True
        
        return False
    
    def _node_in_body(self, node: ast.AST, body: list) -> bool:
        """Check if node is in body"""
        for item in body:
            for child in ast.walk(item):
                if child is node:
                    return True
        return False
    
    def _replace_node(self, tree: ast.AST, old_node: ast.AST, new_node: ast.AST):
        """Replace old node with new node in tree"""
        for parent in ast.walk(tree):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if item is old_node:
                            value[i] = new_node
                            ast.copy_location(new_node, old_node)
                            return
                elif value is old_node:
                    setattr(parent, field, new_node)
                    ast.copy_location(new_node, old_node)
                    return


class NoneCheckAdder(ast.NodeTransformer):
    """Add None checks around Optional accesses"""
    
    def __init__(self, optional_vars: set):
        self.optional_vars = optional_vars  # Set of variable names that are Optional
    
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Add None check around attribute access"""
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            
            if var_name in self.optional_vars:
                # Wrap in None check
                none_check = ast.Compare(
                    left=ast.Name(id=var_name, ctx=ast.Load()),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)]
                )
                
                return ast.IfExp(
                    test=none_check,
                    body=node,
                    orelse=ast.Constant(value=None)
                )
        
        return self.generic_visit(node)
