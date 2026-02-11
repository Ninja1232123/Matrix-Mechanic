"""
Missing Hints Fixer - Add missing type hints to functions and variables
"""

import ast
from typing import Optional, Tuple, Any
from ..inference.type_inferrer import TypeInferrer


class MissingHintsFixer:
    """Fix missing type hint errors"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict  # If True, avoid Any types
        self.inferrer = TypeInferrer(strict=strict)
    
    def can_fix(self, error: dict, tree: ast.AST) -> bool:
        """Check if we can fix this error"""
        return error['category'] == 'missing_type_hint'
    
    def fix(self, error: dict, tree: ast.AST) -> Tuple[ast.AST, bool]:
        """
        Fix missing type hint error
        
        Args:
            error: Error dictionary
            tree: AST tree
            
        Returns:
            (modified_tree, success)
        """
        line = error['line']
        message = error['message']
        
        # Find the node at this line
        node = self._find_node_at_line(tree, line)
        
        if not node:
            return tree, False
        
        if isinstance(node, ast.FunctionDef):
            return self._fix_function_hints(node, tree, error)
        elif isinstance(node, ast.Assign):
            return self._fix_variable_hint(node, tree, error)
        
        return tree, False
    
    def _find_node_at_line(self, tree: ast.AST, line: int) -> Optional[ast.AST]:
        """Find AST node at given line number"""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line:
                return node
        return None
    
    def _fix_function_hints(self, node: ast.FunctionDef, tree: ast.AST, error: dict) -> Tuple[ast.AST, bool]:
        """Add missing type hints to function"""
        modified = False
        
        # Fix missing return type
        if 'return type' in error['message'].lower() and not node.returns:
            return_type = self.inferrer.infer_return_type(node)
            if return_type:
                node.returns = self._type_str_to_ast(return_type)
                modified = True
        
        # Fix missing parameter types
        for arg in node.args.args:
            if not arg.annotation:
                param_type = self.inferrer.infer_param_type(arg.arg, node)
                if param_type:
                    arg.annotation = self._type_str_to_ast(param_type)
                    modified = True
        
        return tree, modified
    
    def _fix_variable_hint(self, node: ast.Assign, tree: ast.AST, error: dict) -> Tuple[ast.AST, bool]:
        """Add type hint to variable"""
        # Get variable name
        if len(node.targets) != 1:
            return tree, False
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return tree, False
        
        var_name = target.id
        
        # Infer type
        var_type = self.inferrer.infer_variable_type(var_name, node, tree)
        
        if not var_type:
            return tree, False
        
        # Convert to AnnAssign
        ann_assign = ast.AnnAssign(
            target=target,
            annotation=self._type_str_to_ast(var_type),
            value=node.value,
            simple=1
        )
        
        # Replace node in tree
        self._replace_node(tree, node, ann_assign)
        
        return tree, True
    
    def _type_str_to_ast(self, type_str: str) -> ast.AST:
        """Convert type string to AST node"""
        try:
            # Parse as expression
            return ast.parse(type_str, mode='eval').body
        except SyntaxError:
            # Fallback to Any
            return ast.Name(id='Any', ctx=ast.Load())
    
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


class FunctionHintAdder(ast.NodeTransformer):
    """AST transformer to add function type hints"""
    
    def __init__(self, inferrer: TypeInferrer):
        self.inferrer = inferrer
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Add type hints to function"""
        # Add return type if missing
        if not node.returns:
            return_type = self.inferrer.infer_return_type(node)
            if return_type:
                node.returns = self._type_str_to_ast(return_type)
        
        # Add parameter types if missing
        for arg in node.args.args:
            if not arg.annotation:
                param_type = self.inferrer.infer_param_type(arg.arg, node)
                if param_type:
                    arg.annotation = self._type_str_to_ast(param_type)
        
        return self.generic_visit(node)
    
    def _type_str_to_ast(self, type_str: str) -> ast.AST:
        """Convert type string to AST node"""
        try:
            return ast.parse(type_str, mode='eval').body
        except SyntaxError:
            return ast.Name(id='Any', ctx=ast.Load())
