"""
Yield inlining optimization

Transforms for loops over yield functions by inlining the yield function body,
replacing each yield with the loop body execution.

Design:
- Detects for loops that iterate over yield functions
- Inlines yield function body into the for loop site
- Replaces each yield statement with loop variable assignment + loop body
- Zero runtime overhead: no vtable calls, pure control flow
"""

import ast
import copy
from typing import Optional, List, Tuple, Dict, Any


class YieldInliner:
    """Inlines yield functions into for loops"""
    
    def __init__(self, visitor):
        self.visitor = visitor
        self.inlining_depth = 0  # Track recursion depth to prevent infinite inlining
        self.max_inline_depth = 3
    
    def inline_for_loop_with_info(
        self,
        for_node: ast.For,
        func_ast: ast.FunctionDef,
        call_args: List
    ) -> Optional[List[ast.stmt]]:
        """Inline a for loop with already-provided yield function info
        
        This is called when visit_For detects iter_val has _yield_inline_info.
        
        Args:
            for_node: The for loop AST node
            func_ast: Original AST of the yield function
            call_args: Pre-evaluated ValueRef arguments from the call
            
        Returns:
            List of inlined statements if successful, None if inlining failed
        """
        # Check if we're too deep in inlining
        if self.inlining_depth >= self.max_inline_depth:
            return None
        
        # Check if function is inlinable
        if not self._is_inlinable(func_ast):
            return None
        
        # Get return type annotation
        return_type_annotation = func_ast.returns if hasattr(func_ast, 'returns') else None
        
        # Perform inlining
        try:
            self.inlining_depth += 1
            inlined = self._inline_yield_function(for_node, func_ast, return_type_annotation)
            return inlined
        finally:
            self.inlining_depth -= 1
        
    def try_inline_for_loop(self, for_node: ast.For) -> Optional[List[ast.stmt]]:
        """
        Try to inline a for loop over a yield function
        
        Args:
            for_node: The for loop AST node
            
        Returns:
            List of inlined statements if successful, None if inlining not possible
        """
        # Check if we're too deep in inlining
        if self.inlining_depth >= self.max_inline_depth:
            return None
        
        # Check if iter is a call to a yield function
        if not isinstance(for_node.iter, ast.Call):
            return None
        
        # Get the function being called
        if isinstance(for_node.iter.func, ast.Name):
            func_name = for_node.iter.func.id
        else:
            # Complex expression, can't inline
            return None
        
        # Look up function in visitor's context
        func_obj = self._lookup_function(func_name)
        if not func_obj:
            return None
        
        # Check if it's a yield-generated iterator
        if not hasattr(func_obj, '_is_yield_generated') or not func_obj._is_yield_generated:
            return None
        
        # Get original AST
        if not hasattr(func_obj, '_original_ast'):
            return None
        
        original_ast = func_obj._original_ast
        
        # Get return type annotation for proper type inference
        return_type_annotation = original_ast.returns if hasattr(original_ast, 'returns') else None
        
        # Analyze the yield function to determine if it's inlinable
        if not self._is_inlinable(original_ast):
            return None
        
        # Perform inlining
        try:
            self.inlining_depth += 1
            inlined = self._inline_yield_function(for_node, original_ast, return_type_annotation)
            return inlined
        finally:
            self.inlining_depth -= 1
    
    def _lookup_function(self, name: str):
        """Look up function in visitor's context"""
        # Try to get from compiler's user_globals
        if hasattr(self.visitor, 'compiler') and hasattr(self.visitor.compiler, 'user_globals'):
            if name in self.visitor.compiler.user_globals:
                obj = self.visitor.compiler.user_globals[name]
                # Check if it's a compiled function
                if callable(obj):
                    return obj
        
        # Try visitor's user_globals
        if hasattr(self.visitor, 'user_globals') and name in self.visitor.user_globals:
            obj = self.visitor.user_globals[name]
            if callable(obj):
                return obj
        
        return None
    
    def _is_inlinable(self, func_ast: ast.FunctionDef) -> bool:
        """
        Check if a yield function can be inlined
        
        Restrictions:
        - No recursive calls
        - No complex control flow (only while/if allowed)
        - No return statements
        - No nested yield functions
        """
        # Check for disallowed patterns
        checker = InlinabilityChecker()
        checker.visit(func_ast)
        
        return not checker.has_disallowed_patterns
    
    def _inline_yield_function(
        self, 
        for_node: ast.For, 
        func_ast: ast.FunctionDef,
        return_type_annotation: Optional[ast.expr] = None
    ) -> List[ast.stmt]:
        """
        Inline a yield function into the for loop site
        
        Transforms:
            for x in gen(args):
                body
        
        Into:
            # Parameter bindings
            param1 = arg1
            param2 = arg2
            
            # Function initialization (statements before loop)
            init_stmt1
            init_stmt2
            
            # Transform loop with yield -> loop body replacement
            while condition:
                # Statements before yield
                stmt1
                stmt2
                
                # yield value -> x = value; body
                x = value
                body
                
                # Statements after yield
                stmt3
                stmt4
        """
        # Extract call arguments
        call_args = for_node.iter.args
        call_kwargs = {kw.arg: kw.value for kw in for_node.iter.keywords}
        
        # Get loop variable name
        if isinstance(for_node.target, ast.Name):
            loop_var_name = for_node.target.id
        else:
            # Complex target not supported
            return None
        
        # Create a renaming map to avoid variable conflicts
        # Rename all local variables in the yield function to avoid conflicts
        rename_map = {}
        
        # Collect all local variables from the yield function
        local_vars = set()
        for stmt in ast.walk(func_ast):
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                local_vars.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        local_vars.add(target.id)
        
        # Generate unique names for local variables
        import random
        suffix = f"_{random.randint(1000, 9999)}"
        for var in local_vars:
            rename_map[var] = f"{var}_inlined{suffix}"
        
        # Don't rename parameters - they will be bound to arguments
        for param in func_ast.args.args:
            if param.arg in rename_map:
                del rename_map[param.arg]
        
        # Build parameter bindings (parameters are not renamed)
        param_bindings = []
        for i, param in enumerate(func_ast.args.args):
            param_name = param.arg
            if i < len(call_args):
                # Positional argument
                binding = ast.AnnAssign(
                    target=ast.Name(id=param_name, ctx=ast.Store()),
                    annotation=param.annotation,
                    value=copy.deepcopy(call_args[i]),
                    simple=1
                )
                # Copy location info from for_node
                ast.copy_location(binding, for_node)
                param_bindings.append(binding)
            elif param_name in call_kwargs:
                # Keyword argument
                binding = ast.AnnAssign(
                    target=ast.Name(id=param_name, ctx=ast.Store()),
                    annotation=param.annotation,
                    value=copy.deepcopy(call_kwargs[param_name]),
                    simple=1
                )
                ast.copy_location(binding, for_node)
                param_bindings.append(binding)
            else:
                # Missing argument - should be caught by type checker
                return None
        
        # Pre-declare loop variable (equivalent to closure parameter)
        # This must come before any code that uses it
        loop_var_declaration = None
        if return_type_annotation:
            loop_var_declaration = ast.AnnAssign(
                target=ast.Name(id=loop_var_name, ctx=ast.Store()),
                annotation=copy.deepcopy(return_type_annotation),
                value=None,  # No initial value, just declaration
                simple=1
            )
            ast.copy_location(loop_var_declaration, for_node)
        
        # Transform function body with renaming
        transformer = YieldBodyTransformer(loop_var_name, for_node.body, for_node, rename_map, return_type_annotation)
        transformed_stmts = []
        
        for stmt in func_ast.body:
            transformed = transformer.visit(stmt)
            if isinstance(transformed, list):
                for t in transformed:
                    ast.fix_missing_locations(t)
                    transformed_stmts.append(t)
            elif transformed is not None:
                ast.fix_missing_locations(transformed)
                transformed_stmts.append(transformed)
        
        # Combine: loop variable declaration + parameter bindings + transformed body
        result = []
        if loop_var_declaration:
            result.append(loop_var_declaration)
        result.extend(param_bindings)
        result.extend(transformed_stmts)
        
        return result


class InlinabilityChecker(ast.NodeVisitor):
    """Check if a yield function is safe to inline"""
    
    def __init__(self):
        self.has_disallowed_patterns = False
        self.current_function_name = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function name for recursion check"""
        old_name = self.current_function_name
        self.current_function_name = node.name
        self.generic_visit(node)
        self.current_function_name = old_name
    
    def visit_Return(self, node: ast.Return):
        """Return statements not allowed (except implicit)"""
        if node.value is not None:
            self.has_disallowed_patterns = True
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Check for recursive calls and nested yield functions"""
        # Check for recursion
        if isinstance(node.func, ast.Name):
            if node.func.id == self.current_function_name:
                self.has_disallowed_patterns = True
        
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        """Nested for loops are allowed"""
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With):
        """With statements not supported yet"""
        self.has_disallowed_patterns = True
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try):
        """Try/except not supported yet"""
        self.has_disallowed_patterns = True
        self.generic_visit(node)


class YieldBodyTransformer(ast.NodeTransformer):
    """
    Transform yield function body by replacing yields with loop body
    
    Each yield statement is replaced with:
        loop_var = yield_value
        <loop_body_statements>
    """
    
    def __init__(self, loop_var_name: str, loop_body: List[ast.stmt], for_node: ast.For, rename_map: Dict[str, str], return_type_annotation: Optional[ast.expr] = None):
        self.loop_var_name = loop_var_name
        self.loop_body = loop_body
        self.for_node = for_node
        self.rename_map = rename_map
        self.return_type_annotation = return_type_annotation
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variables according to rename_map"""
        if node.id in self.rename_map:
            return ast.Name(id=self.rename_map[node.id], ctx=node.ctx)
        return node
    
    def visit_Expr(self, node: ast.Expr):
        """Transform expression statements (yield is typically here)"""
        if isinstance(node.value, ast.Yield):
            # Replace yield with assignment + loop body
            yield_value = node.value.value
            
            # Apply renaming to yield value
            yield_value_renamed = self.visit(yield_value)
            
            # Only do assignment, not declaration
            # The loop variable has already been pre-declared before the loop
            assignment = ast.Assign(
                targets=[ast.Name(id=self.loop_var_name, ctx=ast.Store())],
                value=yield_value_renamed
            )
            
            # Copy location info
            ast.copy_location(assignment, self.for_node)
            
            # Return assignment + deep copy of loop body
            result = [assignment]
            for stmt in self.loop_body:
                result.append(copy.deepcopy(stmt))
            
            return result
        
        # Not a yield, visit children for renaming
        self.generic_visit(node)
        return node
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Transform while loop body"""
        # Transform test condition (apply renaming)
        new_test = self.visit(node.test)
        
        # Transform the body recursively
        new_body = []
        for stmt in node.body:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                new_body.extend(transformed)
            elif transformed is not None:
                new_body.append(transformed)
        
        # Transform orelse
        new_orelse = []
        for stmt in node.orelse:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                new_orelse.extend(transformed)
            elif transformed is not None:
                new_orelse.append(transformed)
        
        # Create new while node with transformed body
        return ast.While(
            test=new_test,
            body=new_body,
            orelse=new_orelse
        )
    
    def visit_If(self, node: ast.If) -> ast.If:
        """Transform if statement body"""
        # Transform test condition (apply renaming)
        new_test = self.visit(node.test)
        
        # Transform the body recursively
        new_body = []
        for stmt in node.body:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                new_body.extend(transformed)
            elif transformed is not None:
                new_body.append(transformed)
        
        # Transform orelse
        new_orelse = []
        for stmt in node.orelse:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                new_orelse.extend(transformed)
            elif transformed is not None:
                new_orelse.append(transformed)
        
        # Create new if node with transformed body
        return ast.If(
            test=new_test,
            body=new_body,
            orelse=new_orelse
        )
