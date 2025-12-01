"""
Yield Inlining Adapter

Thin adapter layer that connects the AST visitor's for-loop handling
to the universal inline kernel for yield functions.

This is a simple forwarding layer with minimal logic.
"""

import ast
import copy
from typing import List, Optional

from .kernel import InlineKernel
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
        
        # CRITICAL: Merge callee's globals into visitor's user_globals
        # This allows inlined code to access functions/builtins from the original module
        # NOTE: Caller MUST restore globals after visiting returned statements!
        if func_obj and hasattr(func_obj, '__globals__'):
            from ..logger import logger
            logger.debug(f"Merging globals from func_obj: {func_obj}")
            logger.debug(f"func_obj.__globals__ has {len(func_obj.__globals__)} keys")
            old_user_globals = self.visitor.ctx.user_globals
            # Create merged globals: callee's globals + current globals
            # Current globals take precedence (to avoid overriding)
            merged_globals = dict(func_obj.__globals__)
            if old_user_globals:
                merged_globals.update(old_user_globals)
            self.visitor.ctx.user_globals = merged_globals
            logger.debug(f"Merged user_globals now has {len(self.visitor.ctx.user_globals)} keys")
            if 'assume' in self.visitor.ctx.user_globals:
                logger.debug("'assume' IS in merged user_globals")
            else:
                logger.debug("'assume' NOT in merged user_globals")
                logger.debug(f"Keys in merged globals: {list(self.visitor.ctx.user_globals.keys())}")
        else:
            from ..logger import logger
            logger.debug(f"NOT merging globals: func_obj={func_obj}, has__globals__={hasattr(func_obj, '__globals__') if func_obj else 'N/A'}")
            old_user_globals = None
        
        try:
            # Create inline operation
            try:
                op = self.kernel.create_inline_op(
                    callee_func=func_ast,
                    call_site=for_node.iter,  # The call expression
                    call_args=call_args,
                    caller_context=caller_context,
                    exit_rule=exit_rule
                )
            except Exception as e:
                # If kernel rejects the operation, cannot inline
                from ..logger import logger
                logger.debug(f"Kernel rejected yield inline: {e}")
                return (None, None)
            
            # Execute inlining
            try:
                from ..logger import logger
                logger.debug(f"Before execute_inline: user_globals has {len(self.visitor.ctx.user_globals)} keys, 'assume' in it: {'assume' in self.visitor.ctx.user_globals}")
                inlined_stmts = self.kernel.execute_inline(op)
                logger.debug(f"After execute_inline: user_globals has {len(self.visitor.ctx.user_globals)} keys, 'assume' in it: {'assume' in self.visitor.ctx.user_globals}")
                
                # CRITICAL: Pre-declare loop variable before the inlined statements
                # This is needed because yield points will only assign to it, not declare it
                if return_type_annotation and inlined_stmts:
                    loop_var_decl = ast.AnnAssign(
                        target=ast.Name(id=loop_var, ctx=ast.Store()),
                        annotation=copy.deepcopy(return_type_annotation),
                        value=None,  # No initial value, just declaration
                        simple=1
                    )
                    ast.copy_location(loop_var_decl, for_node)
                    inlined_stmts.insert(0, loop_var_decl)
                
                # Return the statements AND old_user_globals for caller to restore
                # Caller MUST restore globals after visiting all statements!
                return (inlined_stmts, old_user_globals)
            except Exception as e:
                from ..logger import logger
                logger.warning(f"Yield inlining failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                # Restore globals before returning on error
                if old_user_globals is not None:
                    self.visitor.ctx.user_globals = old_user_globals
                return (None, None)
        except Exception as e:
            # Restore globals on any outer exception
            if old_user_globals is not None:
                self.visitor.ctx.user_globals = old_user_globals
            raise
    
    def _extract_loop_var(self, for_node: ast.For) -> Optional[str]:
        """Extract loop variable name from for node"""
        if isinstance(for_node.target, ast.Name):
            return for_node.target.id
        # Tuple unpacking and other complex targets not supported yet
        return None
    
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
