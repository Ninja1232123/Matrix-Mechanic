"""
Type Inferrer - Infer types from code usage patterns
"""

import ast
from typing import Optional, Dict, List, Set, Any, Tuple


class TypeInferrer:
    """Infer types from code usage patterns"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict  # If True, avoid Any types
        self.type_cache: Dict[str, str] = {}
    
    def infer_return_type(self, func: ast.FunctionDef) -> Optional[str]:
        """
        Infer return type from function body
        
        Strategies:
        1. Find all return statements
        2. Infer type of returned expressions
        3. Unify types
        """
        return_types = set()
        has_none_return = False
        
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                if node.value is None:
                    has_none_return = True
                else:
                    ret_type = self._infer_expr_type(node.value, func)
                    if ret_type:
                        return_types.add(ret_type)
        
        if not return_types and not has_none_return:
            return 'None'
        
        if not return_types:
            return 'None'
        
        if len(return_types) == 1:
            ret_type = return_types.pop()
            if has_none_return:
                return f'Optional[{ret_type}]'
            return ret_type
        
        # Multiple return types
        unified = self._unify_types(return_types)
        if has_none_return:
            return f'Optional[{unified}]'
        return unified
    
    def infer_param_type(self, param_name: str, func: ast.FunctionDef) -> Optional[str]:
        """
        Infer parameter type from usage in function
        
        Strategies:
        1. Check operations performed on parameter
        2. Check method calls
        3. Check comparisons
        4. Check passed to other functions
        """
        # Collect usage patterns
        patterns = self._analyze_param_usage(param_name, func)
        
        if not patterns:
            return 'Any' if not self.strict else None
        
        # Infer from patterns
        return self._infer_from_patterns(patterns)
    
    def infer_variable_type(self, var_name: str, context: ast.AST, tree: ast.AST) -> Optional[str]:
        """
        Infer variable type from assignments and usage
        
        Args:
            var_name: Variable name
            context: Immediate context (assignment node)
            tree: Full tree for analysis
        """
        # First, check the assignment value
        if isinstance(context, ast.Assign):
            value_type = self._infer_expr_type(context.value, tree)
            if value_type and value_type != 'Any':
                return value_type
        
        # Check all assignments to this variable
        assignments = self._find_assignments(var_name, tree)
        types = set()
        
        for assign in assignments:
            value_type = self._infer_expr_type(assign.value, tree)
            if value_type:
                types.add(value_type)
        
        if not types:
            return 'Any' if not self.strict else None
        
        return self._unify_types(types)
    
    def infer_and_annotate(self, tree: ast.AST) -> Tuple[ast.AST, int]:
        """
        Infer types and add annotations to entire tree
        
        Returns:
            (modified_tree, count_of_annotations_added)
        """
        count = 0
        
        # Annotate functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Add return type
                if not node.returns:
                    ret_type = self.infer_return_type(node)
                    if ret_type:
                        node.returns = self._type_str_to_ast(ret_type)
                        count += 1
                
                # Add parameter types
                for arg in node.args.args:
                    if not arg.annotation:
                        param_type = self.infer_param_type(arg.arg, node)
                        if param_type:
                            arg.annotation = self._type_str_to_ast(param_type)
                            count += 1
        
        return tree, count
    
    def _infer_expr_type(self, expr: ast.expr, context: ast.AST) -> Optional[str]:
        """Infer type of an expression"""
        if isinstance(expr, ast.Constant):
            return self._python_type_to_typing(type(expr.value).__name__)
        
        elif isinstance(expr, ast.Name):
            # Look up variable type
            return self._lookup_variable_type(expr.id, context)
        
        elif isinstance(expr, ast.Call):
            # Function call - use return type if available
            if isinstance(expr.func, ast.Name):
                func_name = expr.func.id
                
                # Known constructors
                if func_name in {'str', 'int', 'float', 'bool', 'list', 'dict', 'set'}:
                    return self._python_type_to_typing(func_name)
                
                # Look up function definition
                func_def = self._find_function(func_name, context)
                if func_def and func_def.returns:
                    return ast.unparse(func_def.returns)
        
        elif isinstance(expr, ast.List):
            # Infer element type
            if expr.elts:
                elem_types = {self._infer_expr_type(e, context) for e in expr.elts}
                elem_types.discard(None)
                if len(elem_types) == 1:
                    return f"List[{elem_types.pop()}]"
            return "List[Any]"
        
        elif isinstance(expr, ast.Dict):
            # Infer key/value types
            key_types = {self._infer_expr_type(k, context) for k in expr.keys if k}
            val_types = {self._infer_expr_type(v, context) for v in expr.values}
            key_types.discard(None)
            val_types.discard(None)
            
            key_type = key_types.pop() if len(key_types) == 1 else 'str'
            val_type = val_types.pop() if len(val_types) == 1 else 'Any'
            return f"Dict[{key_type}, {val_type}]"
        
        elif isinstance(expr, ast.Set):
            if expr.elts:
                elem_types = {self._infer_expr_type(e, context) for e in expr.elts}
                elem_types.discard(None)
                if len(elem_types) == 1:
                    return f"Set[{elem_types.pop()}]"
            return "Set[Any]"
        
        elif isinstance(expr, ast.Tuple):
            # Fixed-size tuple
            elem_types = [self._infer_expr_type(e, context) for e in expr.elts]
            if all(elem_types):
                return f"Tuple[{', '.join(elem_types)}]"
            return "Tuple[Any, ...]"
        
        elif isinstance(expr, ast.BinOp):
            # Binary operation - infer from operands
            left_type = self._infer_expr_type(expr.left, context)
            right_type = self._infer_expr_type(expr.right, context)
            
            # Numeric operations
            if isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                if left_type in ('int', 'float') or right_type in ('int', 'float'):
                    return 'float' if 'float' in (left_type, right_type) else 'int'
        
        elif isinstance(expr, ast.Compare):
            return 'bool'
        
        elif isinstance(expr, ast.BoolOp):
            return 'bool'
        
        elif isinstance(expr, ast.IfExp):
            # Ternary - unify body and orelse types
            body_type = self._infer_expr_type(expr.body, context)
            else_type = self._infer_expr_type(expr.orelse, context)
            
            if body_type and else_type:
                if body_type == else_type:
                    return body_type
                return self._unify_types({body_type, else_type})
        
        return 'Any' if not self.strict else None
    
    def _analyze_param_usage(self, param_name: str, func: ast.FunctionDef) -> List[Dict]:
        """Analyze how parameter is used in function"""
        patterns = []
        
        for node in ast.walk(func):
            # Method calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id == param_name:
                            patterns.append({
                                'type': 'method_call',
                                'method': node.func.attr
                            })
            
            # Subscript access
            elif isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name):
                    if node.value.id == param_name:
                        patterns.append({
                            'type': 'subscript',
                            'key_type': self._infer_expr_type(node.slice, func)
                        })
            
            # Iteration
            elif isinstance(node, ast.For):
                if isinstance(node.iter, ast.Name):
                    if node.iter.id == param_name:
                        patterns.append({
                            'type': 'iteration'
                        })
            
            # Binary operations
            elif isinstance(node, ast.BinOp):
                if isinstance(node.left, ast.Name) and node.left.id == param_name:
                    patterns.append({
                        'type': 'binop',
                        'op': node.op.__class__.__name__
                    })
        
        return patterns
    
    def _infer_from_patterns(self, patterns: List[Dict]) -> Optional[str]:
        """Infer type from usage patterns"""
        for pattern in patterns:
            ptype = pattern['type']
            
            if ptype == 'method_call':
                method = pattern['method']
                
                # String methods
                if method in ('lower', 'upper', 'strip', 'split', 'replace'):
                    return 'str'
                
                # List methods
                if method in ('append', 'extend', 'pop', 'remove'):
                    return 'List[Any]'
                
                # Dict methods
                if method in ('get', 'keys', 'values', 'items'):
                    return 'Dict[str, Any]'
            
            elif ptype == 'subscript':
                key_type = pattern.get('key_type')
                if key_type == 'int':
                    return 'List[Any]'
                elif key_type == 'str':
                    return 'Dict[str, Any]'
            
            elif ptype == 'iteration':
                return 'Iterable[Any]'
            
            elif ptype == 'binop':
                op = pattern['op']
                if op in ('Add', 'Sub', 'Mult', 'Div'):
                    return 'int'
        
        return 'Any' if not self.strict else None
    
    def _unify_types(self, types: Set[str]) -> str:
        """Unify multiple types into single type"""
        if not types:
            return 'Any'
        
        types = {t for t in types if t != 'Any'}
        
        if not types:
            return 'Any'
        
        if len(types) == 1:
            return types.pop()
        
        # Check if all are numeric
        if types <= {'int', 'float'}:
            return 'float' if 'float' in types else 'int'
        
        # Use Union
        return f"Union[{', '.join(sorted(types))}]"
    
    def _python_type_to_typing(self, py_type: str) -> str:
        """Convert Python type to typing module type"""
        mapping = {
            'NoneType': 'None',
            'list': 'List[Any]',
            'dict': 'Dict[str, Any]',
            'set': 'Set[Any]',
            'tuple': 'Tuple[Any, ...]',
        }
        return mapping.get(py_type, py_type)
    
    def _find_assignments(self, var_name: str, tree: ast.AST) -> List[ast.Assign]:
        """Find all assignments to a variable"""
        assignments = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        assignments.append(node)
        
        return assignments
    
    def _lookup_variable_type(self, var_name: str, context: ast.AST) -> Optional[str]:
        """Look up the type of a variable"""
        # Check cache
        if var_name in self.type_cache:
            return self.type_cache[var_name]
        
        # Find assignment
        assignments = self._find_assignments(var_name, context)
        if assignments:
            # Use first assignment
            var_type = self._infer_expr_type(assignments[0].value, context)
            if var_type:
                self.type_cache[var_name] = var_type
                return var_type
        
        return None
    
    def _find_function(self, func_name: str, tree: ast.AST) -> Optional[ast.FunctionDef]:
        """Find function definition by name"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == func_name:
                    return node
        return None
    
    def _type_str_to_ast(self, type_str: str) -> ast.AST:
        """Convert type string to AST node"""
        try:
            return ast.parse(type_str, mode='eval').body
        except SyntaxError:
            return ast.Name(id='Any', ctx=ast.Load())
