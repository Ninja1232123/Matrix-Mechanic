"""
Collection Fixer - Fix List, Dict, Set type annotations
"""

import ast
from typing import Tuple, Optional, Set


class CollectionFixer:
    """Fix collection type errors (List, Dict, Set)"""
    
    def can_fix(self, error: dict, tree: ast.AST) -> bool:
        """Check if we can fix this error"""
        return error['category'] == 'collection_type'
    
    def fix(self, error: dict, tree: ast.AST) -> Tuple[ast.AST, bool]:
        """
        Fix collection type error by inferring element types
        
        Args:
            error: Error dictionary
            tree: AST tree
            
        Returns:
            (modified_tree, success)
        """
        line = error['line']
        context = error.get('context', {})
        var_name = context.get('name')
        
        if not var_name:
            return tree, False
        
        # Find the assignment
        node = self._find_assignment(tree, var_name, line)
        
        if not node:
            return tree, False
        
        # Infer collection type from usage
        collection_type = self._infer_collection_type(node, tree, var_name)
        
        if not collection_type:
            return tree, False
        
        # Convert to annotated assignment
        ann_assign = ast.AnnAssign(
            target=node.targets[0],
            annotation=self._type_str_to_ast(collection_type),
            value=node.value,
            simple=1
        )
        
        # Replace node
        self._replace_node(tree, node, ann_assign)
        
        return tree, True
    
    def _find_assignment(self, tree: ast.AST, var_name: str, line: int) -> Optional[ast.Assign]:
        """Find assignment node for variable"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if hasattr(node, 'lineno') and node.lineno == line:
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            return node
        return None
    
    def _infer_collection_type(self, node: ast.Assign, tree: ast.AST, var_name: str) -> Optional[str]:
        """
        Infer collection element type from usage
        
        Strategies:
        1. Check initial value (e.g., [User(), User()] → List[User])
        2. Check append/insert calls
        3. Check assignments
        """
        # Check initial value
        if isinstance(node.value, (ast.List, ast.Set)):
            element_type = self._infer_from_literal(node.value)
            if element_type:
                container = 'List' if isinstance(node.value, ast.List) else 'Set'
                return f"{container}[{element_type}]"
        
        elif isinstance(node.value, ast.Dict):
            key_type, val_type = self._infer_dict_types(node.value)
            if key_type and val_type:
                return f"Dict[{key_type}, {val_type}]"
        
        # Check usage patterns
        usage_type = self._infer_from_usage(tree, var_name)
        if usage_type:
            return usage_type
        
        # Default to Any
        if isinstance(node.value, ast.List):
            return "List[Any]"
        elif isinstance(node.value, ast.Dict):
            return "Dict[str, Any]"
        elif isinstance(node.value, ast.Set):
            return "Set[Any]"
        
        return None
    
    def _infer_from_literal(self, node: ast.expr) -> Optional[str]:
        """Infer type from list/set literal"""
        if isinstance(node, (ast.List, ast.Set)):
            if not node.elts:
                return None
            
            # Get types of all elements
            types = set()
            for elt in node.elts:
                elt_type = self._get_expr_type(elt)
                if elt_type:
                    types.add(elt_type)
            
            if len(types) == 1:
                return types.pop()
            elif len(types) > 1:
                # Multiple types - use Union
                return f"Union[{', '.join(sorted(types))}]"
        
        return None
    
    def _infer_dict_types(self, node: ast.Dict) -> Tuple[Optional[str], Optional[str]]:
        """Infer key and value types from dict literal"""
        if not node.keys:
            return None, None
        
        key_types = set()
        val_types = set()
        
        for key, val in zip(node.keys, node.values):
            if key:
                key_type = self._get_expr_type(key)
                if key_type:
                    key_types.add(key_type)
            
            val_type = self._get_expr_type(val)
            if val_type:
                val_types.add(val_type)
        
        key_result = key_types.pop() if len(key_types) == 1 else 'str'
        val_result = val_types.pop() if len(val_types) == 1 else 'Any'
        
        return key_result, val_result
    
    def _get_expr_type(self, node: ast.expr) -> Optional[str]:
        """Get type of an expression"""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.Name):
            # Can't easily determine type of variable reference
            return None
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Assume constructor call → class name is the type
                return node.func.id
        
        return None
    
    def _infer_from_usage(self, tree: ast.AST, var_name: str) -> Optional[str]:
        """Infer type from how collection is used"""
        # Look for append/add calls
        appended_types = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Check for var_name.append(x)
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == var_name:
                            if node.func.attr in ('append', 'add'):
                                if node.args:
                                    arg_type = self._get_expr_type(node.args[0])
                                    if arg_type:
                                        appended_types.add(arg_type)
        
        if len(appended_types) == 1:
            return f"List[{appended_types.pop()}]"
        elif len(appended_types) > 1:
            return f"List[Union[{', '.join(sorted(appended_types))}]]"
        
        return None
    
    def _type_str_to_ast(self, type_str: str) -> ast.AST:
        """Convert type string to AST node"""
        try:
            return ast.parse(type_str, mode='eval').body
        except SyntaxError:
            return ast.Name(id='Any', ctx=ast.Load())
    
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
