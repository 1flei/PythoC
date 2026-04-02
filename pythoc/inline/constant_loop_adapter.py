"""
Constant Loop Unroll Adapter

Transforms compile-time constant loops into repeated AST blocks.
This allows defer and other scope-based features to work correctly.

Example:
    for i in [1, 2, 3]:
        defer(cleanup)
        if i >= limit:
            break
        body

Transforms to:
    with label("_const_loop_0"):
        i = _iter_val_0  # Reference to pre-registered variable
        defer(cleanup)
        if i >= limit:
            goto_begin("_const_loop_exit_0")  # break
        body
    with label("_const_loop_1"):
        i = _iter_val_1
        ...

Key design:
- Iteration values are pre-registered as variables (like function parameters)
- AST references these variables by Name, not by embedding constants
- This is consistent with how ClosureAdapter handles parameters
"""

import ast
import copy
from typing import List, Optional, Any, Tuple, Dict, TYPE_CHECKING
from dataclasses import dataclass

from ..utils import get_next_id
from ..logger import logger
from ..valueref import ValueRef, wrap_value
from ..context import VariableInfo
from ..meta.template import quote_stmts, splice_stmts
from ._intrinsics import _PC_INTRINSICS
from .exit_rules import _empty_label_block, _goto_begin, _goto_end

if TYPE_CHECKING:
    from ..ast_visitor.visitor_impl import LLVMIRVisitor


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@quote_stmts
def _const_iter_block(label_name, target, value, body):
    """One unrolled iteration: labeled scope with binding + transformed body."""
    with __pc_intrinsics.label(label_name):  # noqa: F821
        target = value
        body


@dataclass
class ConstantLoopResult:
    """Result of constant loop transformation"""
    stmts: List[ast.stmt]  # Transformed statements
    exit_label: str        # Label for break target (after loop)
    after_else_label: Optional[str]  # Label after else clause (if has else)
    iter_var_names: List[str]  # Names of registered iteration value variables
    required_globals: Dict[str, Any]  # Globals needed for visiting stmts


class ConstantLoopAdapter:
    """
    Adapter for constant loop unrolling via AST transformation
    
    Transforms for loops over compile-time constant iterables into
    repeated scope blocks, allowing defer and other scope-based
    features to work correctly.
    
    Key design principle:
    - Iteration values are pre-registered as variables in var_registry
    - AST uses Name nodes to reference these variables
    - This is consistent with ClosureAdapter's parameter handling
    """
    
    def __init__(self, visitor: 'LLVMIRVisitor'):
        """
        Args:
            visitor: The ASTVisitor instance (for var_registry access)
        """
        self.visitor = visitor
    
    def transform_constant_loop(
        self,
        for_node: ast.For,
        iterable: Any
    ) -> ConstantLoopResult:
        """
        Transform a constant for loop into repeated scope blocks

        Args:
            for_node: The for loop AST node
            iterable: The compile-time constant iterable (list, tuple, range, etc.)

        Returns:
            ConstantLoopResult with transformed statements and registered var names
        """
        elements = list(iterable)
        required_globals = {'__pc_intrinsics': _PC_INTRINSICS}

        loop_id = get_next_id()
        exit_label = f"_const_loop_exit_{loop_id}"

        body_has_break = _has_break_in_body(for_node.body)
        body_has_continue = _has_continue_in_body(for_node.body)

        has_else = for_node.orelse and len(for_node.orelse) > 0
        after_else_label = (
            f"_const_loop_after_else_{loop_id}"
            if has_else and body_has_break else None
        )
        break_target = after_else_label if after_else_label else exit_label

        stmts = []
        iter_var_names = []

        # Empty iterable: optional else + exit label
        if len(elements) == 0:
            if has_else:
                stmts.extend(copy.deepcopy(for_node.orelse))
            stmts.append(
                _empty_label_block(ast.Constant(value=exit_label)).as_stmt)
            return ConstantLoopResult(
                stmts, exit_label, after_else_label,
                iter_var_names, required_globals)

        # Pre-register iteration value variables
        iter_value_vars = self._register_iteration_values(elements, loop_id)
        iter_var_names = list(iter_value_vars.keys())

        # Build per-iteration blocks via template
        for i, _element in enumerate(elements):
            iter_label = f"_const_loop_iter_{loop_id}_{i}"

            # Transform break/continue in a deep copy of the loop body
            transformed_body = self._transform_body(
                for_node.body, break_target, iter_label,
                body_has_break, body_has_continue)

            # Single template call produces the whole iteration block:
            #   with label(iter_label):
            #       target = _iter_val_N
            #       <transformed_body>
            iter_stmts = _const_iter_block(
                ast.Constant(value=iter_label),
                copy.deepcopy(for_node.target),
                ast.Name(id=iter_value_vars[i], ctx=ast.Load()),
                splice_stmts(transformed_body),
            ).as_stmts
            stmts.extend(iter_stmts)

        # Else clause (only reached when no break)
        if has_else:
            stmts.extend(copy.deepcopy(for_node.orelse))

        # Final label (exit or after-else)
        final_label = after_else_label if after_else_label else exit_label
        stmts.append(
            _empty_label_block(ast.Constant(value=final_label)).as_stmt)

        return ConstantLoopResult(
            stmts, exit_label, after_else_label,
            iter_var_names, required_globals)
    
    def _register_iteration_values(
        self, 
        elements: List[Any], 
        loop_id: int
    ) -> Dict[int, str]:
        """
        Pre-register all iteration values as variables
        
        Similar to how ClosureAdapter creates arg temps.
        
        Returns:
            Dict mapping iteration index to variable name
        """
        iter_vars = {}
        
        for i, element in enumerate(elements):
            var_name = f"_iter_val_{loop_id}_{i}"
            
            # Convert element to ValueRef if needed
            value_ref = self._element_to_valueref(element)
            
            # Register in var_registry (no alloca, pure value reference)
            var_info = VariableInfo(
                name=var_name,
                value_ref=value_ref,
                alloca=None,  # No alloca - pure value
                source="const_loop_iter_val",
                is_parameter=False
            )
            self.visitor.scope_manager.declare_variable(var_info, allow_shadow=True)
            
            iter_vars[i] = var_name
        
        return iter_vars
    
    def _element_to_valueref(self, element: Any) -> ValueRef:
        """Convert iteration element to ValueRef"""
        # If already ValueRef, use it (but create fresh copy without var_name)
        if isinstance(element, ValueRef):
            return element.clone(var_name=None, linear_path=None)
        
        # Wrap Python value
        from ..builtin_entities.python_type import PythonType
        return wrap_value(
            element,
            kind="python",
            type_hint=PythonType.wrap(element, is_constant=True)
        )
    
    def _transform_body(
        self,
        body: List[ast.stmt],
        break_target: str,
        continue_target: str,
        body_has_break: bool,
        body_has_continue: bool
    ) -> List[ast.stmt]:
        """Deep-copy + transform break/continue in loop body."""
        if not body_has_break and not body_has_continue:
            return [copy.deepcopy(s) for s in body]
        transformer = _BreakContinueTransformer(break_target, continue_target)
        return [transformer.visit(copy.deepcopy(s)) for s in body]


class _BreakContinueTransformer(ast.NodeTransformer):
    """Transform break/continue for constant loop unroll"""
    
    def __init__(self, break_target: str, continue_target: str):
        self.break_target = break_target
        self.continue_target = continue_target
        self.loop_depth = 0  # Track nested loop depth
    
    def visit_For(self, node: ast.For) -> ast.For:
        """Don't transform break/continue inside nested for loops"""
        self.loop_depth += 1
        result = self.generic_visit(node)
        self.loop_depth -= 1
        return result
    
    def visit_While(self, node: ast.While) -> ast.While:
        """Don't transform break/continue inside nested while loops"""
        self.loop_depth += 1
        result = self.generic_visit(node)
        self.loop_depth -= 1
        return result
    
    def visit_Break(self, node: ast.Break) -> ast.stmt:
        """Transform break to goto_begin(break_target)"""
        if self.loop_depth > 0:
            return node

        goto_call = _goto_begin(ast.Constant(value=self.break_target)).as_stmt
        ast.copy_location(goto_call, node)
        return goto_call
    
    def visit_Continue(self, node: ast.Continue) -> ast.stmt:
        """Transform continue to goto_end(continue_target)"""
        if self.loop_depth > 0:
            return node

        goto_call = _goto_end(ast.Constant(value=self.continue_target)).as_stmt
        ast.copy_location(goto_call, node)
        return goto_call


def _has_break_in_body(body: List[ast.stmt]) -> bool:
    """Check if body contains break (not in nested loops)"""
    for stmt in body:
        if isinstance(stmt, ast.Break):
            return True
        if isinstance(stmt, ast.If):
            if _has_break_in_body(stmt.body) or _has_break_in_body(stmt.orelse):
                return True
        elif isinstance(stmt, ast.With):
            if _has_break_in_body(stmt.body):
                return True
        elif isinstance(stmt, ast.Try):
            if (_has_break_in_body(stmt.body) or 
                _has_break_in_body(stmt.orelse) or
                _has_break_in_body(stmt.finalbody)):
                return True
            for handler in stmt.handlers:
                if _has_break_in_body(handler.body):
                    return True
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                if _has_break_in_body(case.body):
                    return True
        # Don't recurse into For/While
    return False


def _has_continue_in_body(body: List[ast.stmt]) -> bool:
    """Check if body contains continue (not in nested loops)"""
    for stmt in body:
        if isinstance(stmt, ast.Continue):
            return True
        if isinstance(stmt, ast.If):
            if _has_continue_in_body(stmt.body) or _has_continue_in_body(stmt.orelse):
                return True
        elif isinstance(stmt, ast.With):
            if _has_continue_in_body(stmt.body):
                return True
        elif isinstance(stmt, ast.Try):
            if (_has_continue_in_body(stmt.body) or 
                _has_continue_in_body(stmt.orelse) or
                _has_continue_in_body(stmt.finalbody)):
                return True
            for handler in stmt.handlers:
                if _has_continue_in_body(handler.body):
                    return True
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                if _has_continue_in_body(case.body):
                    return True
        # Don't recurse into For/While
    return False
