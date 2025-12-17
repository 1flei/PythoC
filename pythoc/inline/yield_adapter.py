"""
Yield Inlining Adapter

Thin adapter layer that connects the AST visitor's for-loop handling
to the universal inline kernel for yield functions.

This is a simple forwarding layer with minimal logic.
"""

import ast
import copy
from typing import List, Optional, Dict, Any

from .kernel import InlineKernel, InlineResult
from .scope_analyzer import ScopeContext
from .exit_rules import YieldExitRule


class YieldInlineAdapter:
    """
    Adapter for yield function inlining using the universal kernel
    
    This is a thin wrapper that:
    1. Detects yield function calls in for loops
    2. Extracts necessary information
    3. Forwards to InlineKernel with YieldExitRule
    4. Returns transformed statements
    """
    
    def __init__(self, visitor):
        """
        Args:
            visitor: The ASTVisitor instance (for context/scope info)
        """
        self.visitor = visitor
        self.kernel = InlineKernel()
    
    def try_inline_for_loop(
        self,
        for_node: ast.For,
        func_ast: ast.FunctionDef,
        call_node: ast.Call,
        func_obj=None
    ):
        """
        Try to inline a for loop over a yield function
        
        Args:
            for_node: The for loop AST node
            func_ast: The yield function's AST
            call_node: The original call AST node
            func_obj: Original function object (to access __globals__)
            
        Returns:
            (inlined_stmts, old_user_globals) tuple if successful, (None, None) if failed
            Caller MUST restore user_globals after visiting the statements!
        """
        # Validate basic requirements
        if not self._is_inlinable(func_ast):
            return (None, None)
        
        # Get loop variable name
        loop_var = self._extract_loop_var(for_node)
        if not loop_var:
            return (None, None)
        
        # Extract call arguments from call node
        call_args = call_node.args if isinstance(call_node, ast.Call) else []
        
        # Get caller context (available variables in current scope)
        caller_context = self._build_caller_context()
        
        # Extract return type annotation from function
        return_type_annotation = None
        if hasattr(func_ast, 'returns') and func_ast.returns:
            return_type_annotation = func_ast.returns
        
        # Create exit rule for yield transformation with type annotation
        exit_rule = YieldExitRule(
            loop_var=loop_var,
            loop_body=copy.deepcopy(for_node.body),
            return_type_annotation=return_type_annotation
        )
        
        # Get callee's globals for kernel
        callee_globals = None
        if func_obj and hasattr(func_obj, '__globals__'):
            callee_globals = func_obj.__globals__
        
        try:
            # Create inline operation with callee_globals
            try:
                op = self.kernel.create_inline_op(
                    callee_func=func_ast,
                    call_site=for_node.iter,  # The call expression
                    call_args=call_args,
                    caller_context=caller_context,
                    exit_rule=exit_rule,
                    callee_globals=callee_globals
                )
            except Exception as e:
                # If kernel rejects the operation, cannot inline
                from ..logger import logger
                logger.debug(f"Kernel rejected yield inline: {e}")
                return (None, None)
            
            # Execute inlining - kernel now returns InlineResult
            try:
                from ..logger import logger
                inline_result = self.kernel.execute_inline(op)
                inlined_stmts = inline_result.stmts
                
                # Merge required_globals into visitor's user_globals
                # Order: old_user_globals first, then required_globals
                # This ensures intrinsics like 'move' from kernel take precedence
                old_user_globals = self.visitor.ctx.user_globals
                merged_globals = {}
                if old_user_globals:
                    merged_globals.update(old_user_globals)
                merged_globals.update(inline_result.required_globals)
                self.visitor.ctx.user_globals = merged_globals
                
                # CRITICAL: Pre-declare loop variable(s) before the inlined statements
                # This is needed because yield points will only assign to them, not declare them
                if return_type_annotation and inlined_stmts:
                    decls = self._create_loop_var_declarations(loop_var, return_type_annotation, for_node)
                    for decl in reversed(decls):
                        inlined_stmts.insert(0, decl)
                
                # Return the statements AND old_user_globals for caller to restore
                # Caller MUST restore globals after visiting all statements!
                return (inlined_stmts, old_user_globals)
            except Exception as e:
                from ..logger import logger
                logger.warning(f"Yield inlining failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                return (None, None)
        except Exception as e:
            raise
    
    def _extract_loop_var(self, for_node: ast.For) -> Optional[ast.AST]:
        """Extract loop variable target from for node
        
        Returns the target AST node (Name or Tuple), or None if unsupported.
        Supports:
        - Simple Name: for x in ...
        - Tuple unpacking: for a, b in ...
        """
        target = for_node.target
        if isinstance(target, ast.Name):
            return target
        if isinstance(target, ast.Tuple):
            # Verify all elements are Names (no nested tuples for now)
            for elt in target.elts:
                if not isinstance(elt, ast.Name):
                    return None
            return target
        # Other complex targets not supported
        return None
    
    def _create_loop_var_declarations(
        self, 
        loop_var: ast.AST, 
        return_type_annotation: ast.expr,
        for_node: ast.For
    ) -> List[ast.stmt]:
        """Create variable declarations for loop variable(s)
        
        For simple Name: creates single AnnAssign
        For Tuple: creates declarations for each element using subscript types
        
        Args:
            loop_var: Loop variable target (Name or Tuple)
            return_type_annotation: Return type annotation (e.g., struct[i32, i32])
            for_node: Original for node for location info
            
        Returns:
            List of AnnAssign statements for variable declarations
        """
        decls = []
        
        if isinstance(loop_var, ast.Name):
            # Simple case: single variable
            decl = ast.AnnAssign(
                target=ast.Name(id=loop_var.id, ctx=ast.Store()),
                annotation=copy.deepcopy(return_type_annotation),
                value=None,
                simple=1
            )
            ast.copy_location(decl, for_node)
            decls.append(decl)
        elif isinstance(loop_var, ast.Tuple):
            # Tuple unpacking: declare each element
            # For struct[T1, T2, ...], extract element types
            for i, elt in enumerate(loop_var.elts):
                if isinstance(elt, ast.Name):
                    # Create type annotation for this element
                    # If return_type_annotation is struct[T1, T2], we need T_i
                    element_type = self._extract_tuple_element_type(return_type_annotation, i)
                    
                    decl = ast.AnnAssign(
                        target=ast.Name(id=elt.id, ctx=ast.Store()),
                        annotation=element_type,
                        value=None,
                        simple=1
                    )
                    ast.copy_location(decl, for_node)
                    decls.append(decl)
        
        return decls
    
    def _extract_tuple_element_type(self, type_annotation: ast.expr, index: int) -> ast.expr:
        """Extract the type of a tuple element from a struct type annotation
        
        For struct[T1, T2, ...], returns T_index
        If we can't determine the type, returns the full annotation
        
        Args:
            type_annotation: The full type annotation (e.g., struct[i32, i32])
            index: The element index
            
        Returns:
            Type annotation for the element
        """
        # Check if it's a Subscript like struct[T1, T2]
        if isinstance(type_annotation, ast.Subscript):
            slice_node = type_annotation.slice
            
            # Handle Tuple slice: struct[T1, T2]
            if isinstance(slice_node, ast.Tuple):
                if index < len(slice_node.elts):
                    return copy.deepcopy(slice_node.elts[index])
            
            # Handle single element: struct[T] (shouldn't happen for tuple unpacking)
            elif index == 0:
                return copy.deepcopy(slice_node)
        
        # Fallback: return the full annotation (might cause type errors, but better than nothing)
        return copy.deepcopy(type_annotation)
    
    def _build_caller_context(self) -> ScopeContext:
        """
        Build caller scope context from visitor state
        
        Returns all variables available in current scope
        """
        available_vars = set()
        
        # Get variables from visitor's variable registry
        if hasattr(self.visitor, 'ctx') and hasattr(self.visitor.ctx, 'var_registry'):
            registry = self.visitor.ctx.var_registry
            # Get all variables in current scope
            for var_info in registry.get_all_in_current_scope():
                available_vars.add(var_info.name)
        
        # Also check visitor's locals if available
        if hasattr(self.visitor, 'local_vars'):
            available_vars.update(self.visitor.local_vars.keys())
        
        return ScopeContext(available_vars=available_vars)
    
    def _is_inlinable(self, func_ast: ast.FunctionDef) -> bool:
        """
        Quick check if function is inlinable
        
        Current restrictions:
        - Must contain at least one yield
        - No return statements with values
        - No nested function definitions (for now)
        """
        checker = _YieldInlinabilityChecker()
        checker.visit(func_ast)
        
        return (
            checker.has_yield and
            not checker.has_return_value and
            not checker.has_nested_function
        )


class _YieldInlinabilityChecker(ast.NodeVisitor):
    """Simple checker for yield function inlinability"""
    
    def __init__(self):
        self.has_yield = False
        self.has_return_value = False
        self.has_nested_function = False
        self.depth = 0
    
    def visit_FunctionDef(self, node):
        """Track nested functions"""
        if self.depth > 0:
            self.has_nested_function = True
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
    
    def visit_AsyncFunctionDef(self, node):
        """Track nested async functions"""
        if self.depth > 0:
            self.has_nested_function = True
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1
    
    def visit_Lambda(self, node):
        """Lambdas are ok, don't count as nested functions"""
        self.generic_visit(node)
    
    def visit_Yield(self, node):
        """Record yield"""
        self.has_yield = True
        self.generic_visit(node)
    
    def visit_Return(self, node):
        """Check for return with value"""
        if node.value is not None:
            self.has_return_value = True
        self.generic_visit(node)
