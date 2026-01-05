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

if TYPE_CHECKING:
    from ..ast_visitor.visitor_impl import LLVMIRVisitor


@dataclass
class ConstantLoopResult:
    """Result of constant loop transformation"""
    stmts: List[ast.stmt]  # Transformed statements
    exit_label: str        # Label for break target (after loop)
    after_else_label: Optional[str]  # Label after else clause (if has else)
    iter_var_names: List[str]  # Names of registered iteration value variables


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
        # Convert to list for iteration
        elements = list(iterable)
        
        # Generate unique IDs for this loop
        loop_id = get_next_id()
        exit_label = f"_const_loop_exit_{loop_id}"
        
        # Check if body has break/continue
        body_has_break = _has_break_in_body(for_node.body)
        body_has_continue = _has_continue_in_body(for_node.body)
        
        # If has else clause and break, need after_else label
        has_else = for_node.orelse and len(for_node.orelse) > 0
        after_else_label = f"_const_loop_after_else_{loop_id}" if has_else and body_has_break else None
        
        # The break target depends on whether there's an else clause
        # - If no else: break goes to exit_label
        # - If has else: break goes to after_else_label (skip else)
        break_target = after_else_label if after_else_label else exit_label
        
        stmts = []
        iter_var_names = []
        
        # Handle empty iterable
        if len(elements) == 0:
            # Empty loop: just execute else clause if present
            if has_else:
                stmts.extend(copy.deepcopy(for_node.orelse))
            # Add exit label
            stmts.append(self._create_label_scope(exit_label, [ast.Pass()], for_node))
            return ConstantLoopResult(stmts, exit_label, after_else_label, iter_var_names)
        
        # Parse loop variable pattern
        loop_var_pattern = self._parse_target_pattern(for_node.target, for_node)
        
        # Pre-register all iteration values as variables
        # This is like creating function parameters before inlining
        iter_value_vars = self._register_iteration_values(elements, loop_id)
        iter_var_names = list(iter_value_vars.keys())
        
        # Generate iteration blocks
        for i, element in enumerate(elements):
            iter_label = f"_const_loop_iter_{loop_id}_{i}"
            
            # Create iteration body
            iter_body = []
            
            # Bind loop variable(s) by referencing the pre-registered variable
            iter_value_var = iter_value_vars[i]
            bind_stmts = self._create_binding_stmts_from_var(
                loop_var_pattern, iter_value_var, element, for_node
            )
            iter_body.extend(bind_stmts)
            
            # Transform and add loop body
            for stmt in for_node.body:
                transformed = self._transform_break_continue(
                    copy.deepcopy(stmt),
                    break_target=break_target,
                    continue_target=iter_label,
                    body_has_break=body_has_break,
                    body_has_continue=body_has_continue
                )
                iter_body.append(transformed)
            
            # Always wrap in scoped label for defer support
            iter_stmt = self._create_label_scope(iter_label, iter_body, for_node)
            stmts.append(iter_stmt)
        
        # Add else clause if present (only reached if no break)
        if has_else:
            stmts.extend(copy.deepcopy(for_node.orelse))
        
        # Add exit/after_else label
        final_label = after_else_label if after_else_label else exit_label
        stmts.append(self._create_label_scope(final_label, [ast.Pass()], for_node))
        
        return ConstantLoopResult(stmts, exit_label, after_else_label, iter_var_names)
    
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
            self.visitor.ctx.var_registry.declare(var_info, allow_shadow=True)
            
            iter_vars[i] = var_name
        
        return iter_vars
    
    def _element_to_valueref(self, element: Any) -> ValueRef:
        """Convert iteration element to ValueRef"""
        # If already ValueRef, use it (but create fresh copy without var_name)
        if isinstance(element, ValueRef):
            return wrap_value(
                element.value,
                kind=element.kind,
                type_hint=element.type_hint,
                address=getattr(element, 'address', None)
            )
        
        # Wrap Python value
        from ..builtin_entities.python_type import PythonType
        return wrap_value(
            element,
            kind="python",
            type_hint=PythonType.wrap(element, is_constant=True)
        )
    
    def _parse_target_pattern(self, target: ast.expr, for_node: ast.For) -> Any:
        """Parse loop target pattern
        
        Returns:
            str for simple Name, list for Tuple (can be nested)
        """
        if isinstance(target, ast.Name):
            return target.id
        elif isinstance(target, ast.Tuple):
            return [self._parse_target_pattern(elt, for_node) for elt in target.elts]
        else:
            logger.error(
                f"Unsupported loop target type: {type(target).__name__}",
                node=for_node, exc_type=NotImplementedError
            )
    
    def _create_binding_stmts_from_var(
        self,
        pattern: Any,
        iter_var_name: str,
        element: Any,
        for_node: ast.For
    ) -> List[ast.stmt]:
        """Create assignment statements to bind loop variable(s) from iteration variable
        
        For simple pattern (single var):
            i = _iter_val_0
            
        For tuple pattern:
            a, b = _iter_val_0  (if element is tuple)
            OR
            a = _iter_val_0[0]
            b = _iter_val_0[1]
        
        Args:
            pattern: Variable pattern (str or list)
            iter_var_name: Name of the pre-registered iteration value variable
            element: The actual element value (for type checking tuple unpacking)
            for_node: Original for node for location info
            
        Returns:
            List of assignment statements
        """
        stmts = []
        
        if isinstance(pattern, str):
            # Simple assignment: i = _iter_val_N
            assign = ast.Assign(
                targets=[ast.Name(id=pattern, ctx=ast.Store())],
                value=ast.Name(id=iter_var_name, ctx=ast.Load())
            )
            ast.copy_location(assign, for_node)
            stmts.append(assign)
        elif isinstance(pattern, list):
            # Tuple unpacking: a, b = _iter_val_N
            # Create tuple target
            targets = ast.Tuple(
                elts=[self._pattern_to_store_target(p) for p in pattern],
                ctx=ast.Store()
            )
            assign = ast.Assign(
                targets=[targets],
                value=ast.Name(id=iter_var_name, ctx=ast.Load())
            )
            ast.copy_location(assign, for_node)
            stmts.append(assign)
        
        return stmts
    
    def _pattern_to_store_target(self, pattern: Any) -> ast.expr:
        """Convert pattern to AST store target"""
        if isinstance(pattern, str):
            return ast.Name(id=pattern, ctx=ast.Store())
        elif isinstance(pattern, list):
            return ast.Tuple(
                elts=[self._pattern_to_store_target(p) for p in pattern],
                ctx=ast.Store()
            )
        else:
            raise TypeError(f"Invalid pattern type: {type(pattern)}")
    
    def _create_label_scope(
        self, 
        label_name: str, 
        body: List[ast.stmt],
        for_node: ast.For
    ) -> ast.With:
        """Create a scoped label block: with label("name"): body"""
        label_call = ast.Call(
            func=ast.Name(id='label', ctx=ast.Load()),
            args=[ast.Constant(value=label_name)],
            keywords=[]
        )
        with_stmt = ast.With(
            items=[ast.withitem(context_expr=label_call, optional_vars=None)],
            body=body if body else [ast.Pass()]
        )
        ast.copy_location(with_stmt, for_node)
        ast.fix_missing_locations(with_stmt)
        return with_stmt
    
    def _transform_break_continue(
        self,
        stmt: ast.stmt,
        break_target: str,
        continue_target: str,
        body_has_break: bool,
        body_has_continue: bool
    ) -> ast.stmt:
        """Transform break/continue to goto statements"""
        if not body_has_break and not body_has_continue:
            return stmt
        
        transformer = _BreakContinueTransformer(break_target, continue_target)
        return transformer.visit(stmt)


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
        
        goto_call = ast.Expr(value=ast.Call(
            func=ast.Name(id='goto_begin', ctx=ast.Load()),
            args=[ast.Constant(value=self.break_target)],
            keywords=[]
        ))
        ast.copy_location(goto_call, node)
        return goto_call
    
    def visit_Continue(self, node: ast.Continue) -> ast.stmt:
        """Transform continue to goto_end(continue_target)"""
        if self.loop_depth > 0:
            return node
        
        goto_call = ast.Expr(value=ast.Call(
            func=ast.Name(id='goto_end', ctx=ast.Load()),
            args=[ast.Constant(value=self.continue_target)],
            keywords=[]
        ))
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
