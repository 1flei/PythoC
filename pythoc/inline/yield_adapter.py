"""
Yield Inlining Adapter

Thin adapter layer that connects the AST visitor's for-loop handling
to the meta expansion pipeline for yield functions.

This is a simple forwarding layer with minimal logic.
"""

import ast
import copy
from typing import List, Optional, Dict, Any

from .scope_analyzer import ScopeContext, build_caller_context
from .exit_rules import YieldExitRule, _has_break_or_continue
from .kernel import ReturnValueChecker


class YieldInlineAdapter:
    """
    Adapter for yield function inlining using meta expansion pipeline

    This is a thin wrapper that:
    1. Detects yield function calls in for loops
    2. Extracts necessary information
    3. Forwards to meta.expand_inline() with YieldExitRule
    4. Returns transformed statements
    """

    def __init__(self, visitor):
        """
        Args:
            visitor: The ASTVisitor instance (for context/scope info)
        """
        self.visitor = visitor

    def try_inline_for_loop(
        self,
        for_node: ast.For,
        func_ast: ast.FunctionDef,
        call_node: ast.Call,
        func_obj=None,
        callee_globals_override=None,
    ):
        """
        Try to inline a for loop over a yield function

        Args:
            for_node: The for loop AST node
            func_ast: The yield function's AST
            call_node: The original call AST node
            func_obj: Original function object (to access __globals__)

        Returns:
            (InlineResult, after_else_label) tuple if successful,
            (None, None) if failed.
            Caller owns the full merge/restore lifecycle.
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

        # Check if loop body has break/continue - need special handling
        loop_body = copy.deepcopy(for_node.body)
        body_has_break_or_continue = _has_break_or_continue(loop_body)

        # Generate unique label for after-else (used by break to skip else)
        from ..utils import get_next_id
        after_else_label = f"_for_after_else_{get_next_id()}" if body_has_break_or_continue else None

        # Create exit rule for yield transformation with type annotation
        exit_rule = YieldExitRule(
            loop_var=loop_var,
            loop_body=loop_body,
            return_type_annotation=return_type_annotation,
            after_else_label=after_else_label
        )

        # Get callee's globals for kernel
        callee_globals = None
        if func_obj and hasattr(func_obj, '__globals__'):
            callee_globals = func_obj.__globals__
        if callee_globals_override is not None:
            if callee_globals is None:
                callee_globals = dict(callee_globals_override)
            else:
                merged = dict(callee_globals)
                merged.update(callee_globals_override)
                callee_globals = merged

        from .kernel import try_expand_inline
        inline_result = try_expand_inline(
            callee_ast=func_ast,
            callee_globals=callee_globals,
            call_args=call_args,
            call_site=for_node.iter,
            caller_context=caller_context,
            exit_rule=exit_rule,
        )
        if inline_result is None:
            return (None, None)

        # Return InlineResult and after_else_label for caller to own lifecycle
        return (inline_result, after_else_label)
    
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
    
    def _build_caller_context(self) -> ScopeContext:
        """
        Build caller scope context from visitor state
        
        Returns all variables available in current scope
        """
        scope_manager = getattr(self.visitor, 'scope_manager', None)
        local_vars = getattr(self.visitor, 'local_vars', None)
        return build_caller_context(
            scope_manager,
            local_vars=local_vars,
            visibility="current",
        )
    
    def _is_inlinable(self, func_ast: ast.FunctionDef) -> bool:
        """
        Quick check if function is inlinable
        
        Current restrictions:
        - Must contain at least one yield
        - No return statements with values
        
        Note: Nested functions are allowed - the InlineBodyTransformer.visit_FunctionDef
        handles variable renaming in nested function bodies correctly.
        """
        checker = _YieldInlinabilityChecker()
        checker.visit(func_ast)
        
        return (
            checker.has_yield and
            not checker.has_return_value
        )


class _YieldInlinabilityChecker(ReturnValueChecker):
    """Simple checker for yield function inlinability"""

    def __init__(self):
        super().__init__()
        self.has_yield = False

    def visit_Yield(self, node):
        """Record yield"""
        self.has_yield = True
        self.generic_visit(node)
