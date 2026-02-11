"""
Stub Generator - Generate .pyi stub files for type hints
"""

import ast
from pathlib import Path
from typing import Optional


class StubGenerator:
    """Generate .pyi stub files"""
    
    def generate_stub(self, source_file: Path) -> Optional[Path]:
        """
        Generate stub file for Python source
        
        Args:
            source_file: Path to .py file
            
        Returns:
            Path to generated .pyi file or None
        """
        try:
            with open(source_file, 'r') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Generate stub content
            stub_content = self._generate_stub_content(tree)
            
            # Write stub file
            stub_file = source_file.with_suffix('.pyi')
            with open(stub_file, 'w') as f:
                f.write(stub_content)
            
            return stub_file
            
        except Exception as e:
            print(f"Error generating stub for {source_file}: {e}")
            return None
    
    def _generate_stub_content(self, tree: ast.AST) -> str:
        """Generate stub file content from AST"""
        lines = []
        
        # Add imports
        imports = self._extract_imports(tree)
        if imports:
            lines.extend(imports)
            lines.append('')
        
        # Add class and function definitions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                lines.extend(self._generate_class_stub(node))
                lines.append('')
            elif isinstance(node, ast.FunctionDef):
                lines.append(self._generate_function_stub(node))
                lines.append('')
            elif isinstance(node, ast.Assign):
                # Module-level constants
                stub = self._generate_variable_stub(node)
                if stub:
                    lines.append(stub)
        
        return '\n'.join(lines)
    
    def _extract_imports(self, tree: ast.AST) -> list:
        """Extract import statements"""
        imports = []
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
        
        return imports
    
    def _generate_class_stub(self, node: ast.ClassDef) -> list:
        """Generate stub for a class"""
        lines = []
        
        # Class definition
        bases = ', '.join(ast.unparse(base) for base in node.bases) if node.bases else ''
        if bases:
            lines.append(f'class {node.name}({bases}):')
        else:
            lines.append(f'class {node.name}:')
        
        # Docstring
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                docstring = node.body[0].value.value
                if isinstance(docstring, str):
                    lines.append(f'    """{docstring}"""')
        
        # Methods
        has_methods = False
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_stub = self._generate_function_stub(item, indent=1)
                lines.append(method_stub)
                has_methods = True
        
        if not has_methods:
            lines.append('    ...')
        
        return lines
    
    def _generate_function_stub(self, node: ast.FunctionDef, indent: int = 0) -> str:
        """Generate stub for a function/method"""
        ind = '    ' * indent
        
        # Build parameter list
        params = []
        
        # Handle self/cls
        if node.args.args and indent > 0:
            first_arg = node.args.args[0].arg
            if first_arg in ('self', 'cls'):
                params.append(first_arg)
                remaining_args = node.args.args[1:]
            else:
                remaining_args = node.args.args
        else:
            remaining_args = node.args.args
        
        # Add parameters with annotations
        for arg in remaining_args:
            if arg.annotation:
                params.append(f'{arg.arg}: {ast.unparse(arg.annotation)}')
            else:
                params.append(f'{arg.arg}: Any')
        
        # Add *args
        if node.args.vararg:
            vararg = node.args.vararg
            if vararg.annotation:
                params.append(f'*{vararg.arg}: {ast.unparse(vararg.annotation)}')
            else:
                params.append(f'*{vararg.arg}: Any')
        
        # Add **kwargs
        if node.args.kwarg:
            kwarg = node.args.kwarg
            if kwarg.annotation:
                params.append(f'**{kwarg.arg}: {ast.unparse(kwarg.annotation)}')
            else:
                params.append(f'**{kwarg.arg}: Any')
        
        param_str = ', '.join(params)
        
        # Return type
        if node.returns:
            return_str = f' -> {ast.unparse(node.returns)}'
        else:
            return_str = ' -> Any'
        
        return f'{ind}def {node.name}({param_str}){return_str}: ...'
    
    def _generate_variable_stub(self, node: ast.Assign) -> Optional[str]:
        """Generate stub for module-level variable"""
        if len(node.targets) != 1:
            return None
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None
        
        var_name = target.id
        
        # Try to infer type from value
        value_type = self._infer_value_type(node.value)
        
        if value_type:
            return f'{var_name}: {value_type}'
        else:
            return f'{var_name}: Any'
    
    def _infer_value_type(self, value: ast.expr) -> Optional[str]:
        """Infer type from value expression"""
        if isinstance(value, ast.Constant):
            py_type = type(value.value).__name__
            return py_type
        elif isinstance(value, ast.List):
            return 'List[Any]'
        elif isinstance(value, ast.Dict):
            return 'Dict[Any, Any]'
        elif isinstance(value, ast.Set):
            return 'Set[Any]'
        elif isinstance(value, ast.Tuple):
            return 'Tuple[Any, ...]'
        
        return None


class StubMerger:
    """Merge type stubs with source files"""
    
    def merge_stub_into_source(self, source_file: Path, stub_file: Path) -> bool:
        """
        Merge type annotations from stub into source file
        
        Args:
            source_file: Path to .py file
            stub_file: Path to .pyi file
            
        Returns:
            True if successful
        """
        try:
            # Parse both files
            with open(source_file, 'r') as f:
                source = f.read()
            with open(stub_file, 'r') as f:
                stub = f.read()
            
            source_tree = ast.parse(source)
            stub_tree = ast.parse(stub)
            
            # Merge annotations
            self._merge_annotations(source_tree, stub_tree)
            
            # Write back
            new_source = ast.unparse(source_tree)
            with open(source_file, 'w') as f:
                f.write(new_source)
            
            return True
            
        except Exception as e:
            print(f"Error merging stub: {e}")
            return False
    
    def _merge_annotations(self, source_tree: ast.AST, stub_tree: ast.AST):
        """Merge annotations from stub into source"""
        # Build map of stub definitions
        stub_map = {}
        
        for node in stub_tree.body:
            if isinstance(node, ast.FunctionDef):
                stub_map[node.name] = node
            elif isinstance(node, ast.ClassDef):
                stub_map[node.name] = node
        
        # Apply to source
        for node in source_tree.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in stub_map:
                    stub_func = stub_map[node.name]
                    self._merge_function_annotations(node, stub_func)
            elif isinstance(node, ast.ClassDef):
                if node.name in stub_map:
                    stub_class = stub_map[node.name]
                    self._merge_class_annotations(node, stub_class)
    
    def _merge_function_annotations(self, source_func: ast.FunctionDef, stub_func: ast.FunctionDef):
        """Merge function annotations"""
        # Merge return type
        if stub_func.returns and not source_func.returns:
            source_func.returns = stub_func.returns
        
        # Merge parameter types
        for src_arg, stub_arg in zip(source_func.args.args, stub_func.args.args):
            if stub_arg.annotation and not src_arg.annotation:
                src_arg.annotation = stub_arg.annotation
    
    def _merge_class_annotations(self, source_class: ast.ClassDef, stub_class: ast.ClassDef):
        """Merge class method annotations"""
        # Build method map
        stub_methods = {}
        for node in stub_class.body:
            if isinstance(node, ast.FunctionDef):
                stub_methods[node.name] = node
        
        # Merge methods
        for node in source_class.body:
            if isinstance(node, ast.FunctionDef):
                if node.name in stub_methods:
                    self._merge_function_annotations(node, stub_methods[node.name])
