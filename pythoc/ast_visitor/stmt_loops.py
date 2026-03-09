"""
Loop statement visitor mixin (for, while)
"""

import ast
import copy
from ..logger import logger
from ..scope_manager import ScopeType
from .control_flow_builder import ControlFlowBuilder


def _has_break_in_body(body: list) -> bool:
    """Check if body contains any break statement (at any nesting level)
    
    This is used to optimize for-else: if there's no break in the loop body,
    the else clause is guaranteed to execute, so we don't need break_flag check.
    
    Note: We only check for break at the top level of the for loop. Break inside
    nested loops doesn't affect the outer for-else.
    
    Args:
        body: List of AST statements (loop body)
        
    Returns:
        True if body contains a break that would exit the current loop
    """
    for stmt in body:
        if isinstance(stmt, ast.Break):
            return True
        # Recursively check control flow statements, but NOT nested loops
        # (break in nested loop doesn't affect outer loop)
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
        # Note: We do NOT recurse into For/While - break inside nested loop
        # doesn't break the outer loop
    return False


class LoopsMixin:
    """Mixin for loop statements: for, while"""
    
    def _get_cf_builder(self) -> ControlFlowBuilder:
        """Get or create the ControlFlowBuilder for this visitor"""
        if not hasattr(self, '_cf_builder') or self._cf_builder is None:
            self._cf_builder = ControlFlowBuilder(self)
        return self._cf_builder
    
    def visit_While(self, node: ast.While):
        """Handle while loops
        
        Linear type rule for loops:
            Loop body is a branch that may execute multiple times.
            Linear state at end of body must match state at start of body
            (loop invariant) - otherwise second iteration would see wrong state.
            
        Special cases:
            - while True: Infinite loop, exit only via break/return
              * With break: executes once (like a block), no loop invariant needed
              * Without break: infinite loop, code after while is unreachable
            - while False: Never executes, skip entirely
            
        Uses CFG-based linear state tracking via ControlFlowBuilder.
        """
        cf = self._get_cf_builder()
        
        # Note: We do NOT skip processing when terminated because the loop body
        # may contain label definitions that need to be registered for forward
        # goto resolution.
        
        # Check for compile-time constant condition
        condition_val = self.visit_expression(node.test)
        is_constant_condition = condition_val.is_python_value()
        constant_value = condition_val.get_python_value() if is_constant_condition else None
        
        # while False - never executes
        if is_constant_condition and not constant_value:
            logger.debug("while False - skipping loop body entirely")
            return
        
        # while True - special handling for infinite loop
        if is_constant_condition and constant_value:
            self._visit_while_true(node, cf)
            return
        
        # Normal while loop with runtime condition
        self._visit_while_normal(node, cf, condition_val)
    
    def _visit_while_true(self, node: ast.While, cf: ControlFlowBuilder):
        """Handle while True - infinite loop or single execution with break
        
        CFG structure for while True:
        - No loop header with condition check (condition is always true)
        - Body executes directly
        - break jumps to exit block
        - If body can fall through (no break/return), it loops back (infinite loop)
        - Code after while is only reachable via break
        
        Linear type handling:
        - Since it's either infinite loop or single execution (with break),
          NO loop invariant check is needed
        - Linear tokens created before while True can be consumed in the body
        - This is the key difference from normal while loops
        """
        # Create exit block for break targets
        loop_exit = cf.create_block("while_true_exit")
        
        # For while True, we don't need a separate header block
        # The body IS the loop - it either breaks out or loops back
        loop_body_start = cf.create_block("while_true_body")
        
        # Jump to body
        cf.branch(loop_body_start)
        cf.position_at_end(loop_body_start)
        
        # Use ScopeManager for unified scope/defer management
        # Enter LOOP scope with continue -> loop_body_start, break -> loop_exit
        with self.scope_manager.scope(ScopeType.LOOP, cf,
                                       continue_target=loop_body_start,
                                       break_target=loop_exit,
                                       node=node) as scope:
            # Execute loop body
            self._visit_stmt_list(node.body, add_to_cfg=True)

            # Defers are automatically emitted by scope_manager.exit_scope()
            # But for loop back, we need to emit manually before the branch
            if not cf.is_terminated():
                # Emit defers for this iteration before looping back
                self.scope_manager._emit_defers_for_scope(scope)

            # If body can fall through, it's an infinite loop - loop back
            if not cf.is_terminated():
                cf.branch(loop_body_start)
                cf.mark_loop_back(loop_body_start)

        # Position at exit block
        cf.position_at_end(loop_exit)
        
        # Check if exit is reachable (only via break)
        # If no edges lead to exit, mark as unreachable
        if not cf.cfg.get_predecessors(cf._get_cfg_block_id(loop_exit)):
            # No break in the loop - infinite loop, code after is unreachable
            cf.unreachable()
    
    def _visit_while_normal(self, node: ast.While, cf: ControlFlowBuilder, condition_val):
        """Handle normal while loop with runtime condition
        
        CFG structure:
        - Header block: evaluate condition
        - Body block: loop body
        - Else block (if present): executes when loop completes normally
        - Exit block: code after loop
        
        Linear type handling:
        - Loop invariant is checked by CFG linear checker at function end
        - This ensures multiple iterations see consistent linear state
        """
        # Create loop blocks
        loop_header = cf.create_block("while_header")
        loop_body = cf.create_block("while_body")
        loop_exit = cf.create_block("while_exit")
        
        # Create else block if needed
        has_else = node.orelse and len(node.orelse) > 0
        if has_else:
            else_block = cf.create_block("while_else")
        
        # Jump to loop header
        cf.branch(loop_header)
        
        # Loop header: check condition
        cf.position_at_end(loop_header)
        condition = self._to_boolean(self.visit_expression(node.test))
        
        # If has else: condition false -> else block, otherwise -> exit
        if has_else:
            cf.cbranch(condition, loop_body, else_block)
        else:
            cf.cbranch(condition, loop_body, loop_exit)
        
        # Loop body - use ScopeManager for unified scope/defer management
        cf.position_at_end(loop_body)
        
        # Enter LOOP scope with continue -> loop_header, break -> loop_exit
        with self.scope_manager.scope(ScopeType.LOOP, cf,
                                       continue_target=loop_header,
                                       break_target=loop_exit,
                                       node=node) as scope:
            # Execute loop body statements
            self._visit_stmt_list(node.body, add_to_cfg=True)

            # Note: Linear token checking is done by CFG linear checker
            # at function end, NOT here. The defer execution in scope exit
            # will consume linear tokens, and CFG checker will verify
            # all paths have consistent linear states.

        # Jump back to header (if not terminated by return/break)
        if not cf.is_terminated():
            cf.branch(loop_header)
            cf.mark_loop_back(loop_header)
        
        # Handle else block if present
        if has_else:
            cf.position_at_end(else_block)
            self._visit_stmt_list(node.orelse, add_to_cfg=True)
            if not cf.is_terminated():
                cf.branch(loop_exit)
        
        # Continue after loop
        cf.position_at_end(loop_exit)

    def visit_For(self, node: ast.For):
        """Handle for loops using iterator protocol
        
        Supports three protocols (in priority order):
        1. Yield inlining: inline yield function body for zero overhead
        2. Compile-time constant unrolling: for loops over constant sequences
        3. Generator expression replay: lower genexp carrier to yield-inline form
        """
        # First evaluate the iterator expression
        iter_val = self.visit_expression(node.iter)

        # Check for compile-time constant (Python value)
        if iter_val.is_python_value() and hasattr(iter_val.get_python_value(), "__iter__"):
            py_iterable = iter_val.get_python_value()
            self._visit_for_with_constant_unroll(node, py_iterable)
            return

        # Check for direct yield inlining info
        if hasattr(iter_val, '_yield_inline_info') and iter_val._yield_inline_info:
            self._visit_for_with_yield_inline(node, iter_val)
            return

        # Generator expression carrier: replay by building yield-inline metadata
        genexp_ast = self._extract_generator_expr_ast(iter_val)
        if genexp_ast is not None:
            self._attach_genexp_yield_inline_info(node, iter_val, genexp_ast)
            self._visit_for_with_yield_inline(node, iter_val)
            return

        logger.error(
            f"Unsupported iterator type: {iter_val}. "
            f"Only yield functions (via inlining), generator expressions (replay), "
            f"and compile-time constants are supported.",
            node=node, exc_type=TypeError
        )

    def _extract_generator_expr_ast(self, iter_val):
        """Extract GeneratorExp AST from a genexp carrier ValueRef."""
        info = getattr(iter_val, '_pc_generator_expr_info', None)
        if info:
            gen_ast = info.get('ast')
            if isinstance(gen_ast, ast.GeneratorExp):
                return gen_ast

        if iter_val.is_python_value():
            py_obj = iter_val.get_python_value()
            gen_ast = getattr(py_obj, '_pc_generator_expr', None)
            if isinstance(gen_ast, ast.GeneratorExp):
                return gen_ast

        return None

    def _attach_genexp_yield_inline_info(self, for_node: ast.For, iter_val, genexp_ast: ast.GeneratorExp):
        """Attach replay-by-expansion yield-inline metadata for generator expressions."""
        from ..utils import get_next_id

        func_name = f"__pc_genexp_inline_{get_next_id()}"
        func_ast = self._build_genexp_inline_function_ast(genexp_ast, func_name, for_node)

        call_node = ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[],
            keywords=[]
        )
        ast.copy_location(call_node, for_node.iter)
        ast.fix_missing_locations(call_node)

        iter_val._yield_inline_info = {
            'func_obj': None,
            'original_ast': func_ast,
            'call_node': call_node,
            'call_args': []
        }

    def _build_genexp_inline_function_ast(
        self,
        genexp_ast: ast.GeneratorExp,
        func_name: str,
        for_node: ast.For
    ) -> ast.FunctionDef:
        """Build a synthetic yield function AST from a GeneratorExp AST."""
        if not genexp_ast.generators:
            logger.error("Generator expression must contain at least one generator clause", node=for_node, exc_type=TypeError)

        for comp in genexp_ast.generators:
            if getattr(comp, 'is_async', 0):
                logger.error("Async generator expression is not supported", node=for_node, exc_type=TypeError)

        current_body = [
            ast.Expr(value=ast.Yield(value=copy.deepcopy(genexp_ast.elt)))
        ]

        for comp in reversed(genexp_ast.generators):
            body = current_body
            for cond in reversed(comp.ifs):
                body = [ast.If(test=copy.deepcopy(cond), body=body, orelse=[])]

            loop_stmt = ast.For(
                target=copy.deepcopy(comp.target),
                iter=copy.deepcopy(comp.iter),
                body=body,
                orelse=[],
                type_comment=None
            )
            current_body = [loop_stmt]

        func_ast = ast.FunctionDef(
            name=func_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
                vararg=None,
                kwarg=None,
            ),
            body=current_body,
            decorator_list=[],
            returns=None,
        )
        ast.copy_location(func_ast, for_node)
        ast.fix_missing_locations(func_ast)
        return func_ast
    
    def _visit_for_with_yield_inline(self, node: ast.For, iter_val):
        """Handle for loop with yield inline, including else clause
        
        For yield inline (e.g., refine), the expansion includes:
        - The yield function body with yield transformed to loop body
        - For-else follows Python semantics: executes if no break occurred
        
        Example:
            for x in refine(val, pred):
                body
            else:
                else_body
        
        Python for-else semantics:
        - else executes when loop completes normally (no break)
        - else does NOT execute when break is used
        
        Scoped label approach:
        - break in loop body -> goto_begin("_for_after_else_{id}")
        - continue in loop body -> goto_end("_yield_{id}")
        - After all yields and else, place with label("_for_after_else_{id}"): pass
        """
        from ..inline.yield_adapter import YieldInlineAdapter
        
        cf = self._get_cf_builder()
        adapter = YieldInlineAdapter(self)
        inline_info = iter_val._yield_inline_info
        
        # Extract callee context for resolving names during yield inlining
        func_obj = inline_info.get('func_obj', None)
        callee_globals = inline_info.get('callee_globals', None)

        # Get inlined statements
        # after_else_label is the label name for break to jump to (skip else)
        inlined_stmts, old_user_globals, after_else_label = adapter.try_inline_for_loop(
            node,
            inline_info['original_ast'],
            inline_info['call_node'],
            func_obj=func_obj,
            callee_globals_override=callee_globals,
        )
        
        if inlined_stmts is None:
            # Inlining failed - this is now an error
            logger.error(
                f"Yield function inlining failed for '{ast.unparse(node.iter)}'. "
                f"Yield functions must be inlinable (no complex control flow, recursion, etc.)",
                node=node, exc_type=TypeError
            )
        
        try:
            # Fix all missing locations in inlined statements
            for stmt in inlined_stmts:
                ast.fix_missing_locations(stmt)
            
            # Visit each inlined statement
            self._visit_stmt_list(inlined_stmts, add_to_cfg=True)
            
            # Execute else clause if present (only reached if no break occurred)
            if node.orelse:
                self._visit_stmt_list(node.orelse, add_to_cfg=True)
            
            # Place the after_else label as a scoped label (break jumps here to skip else)
            if after_else_label:
                # Create: with label("_for_after_else_{id}"): pass
                from ..inline._intrinsics import _intrinsic_name
                label_call = ast.Call(
                    func=_intrinsic_name('label'),
                    args=[ast.Constant(value=after_else_label)],
                    keywords=[]
                )
                label_stmt = ast.With(
                    items=[ast.withitem(context_expr=label_call, optional_vars=None)],
                    body=[ast.Pass()]
                )
                ast.copy_location(label_stmt, node)
                ast.fix_missing_locations(label_stmt)
                if not cf.is_terminated():
                    cf.add_stmt(label_stmt)
                    self.visit(label_stmt)
        finally:
            # CRITICAL: Restore globals after visiting all inlined statements
            if old_user_globals is not None:
                self.ctx.user_globals = old_user_globals

    def _visit_for_with_constant_unroll(self, node: ast.For, py_iterable):
        """Unroll for loop at compile time for constant iterables
        
        Uses AST transformation to convert the loop into repeated scope blocks.
        This allows defer and other scope-based features to work correctly.
        
        Transforms:
            for i in [1, 2, 3]:
                defer(cleanup)
                body
            else:
                else_body
        
        To:
            with label("_const_loop_iter_0"):
                i = 1
                defer(cleanup)
                body
            with label("_const_loop_iter_1"):
                i = 2
                defer(cleanup)
                body
            with label("_const_loop_iter_2"):
                i = 3
                defer(cleanup)
                body
            else_body
            with label("_const_loop_exit_N"):
                pass
        
        Each iteration is a separate scope (via label), so:
        - defer is registered and executed per-iteration automatically
        - break transforms to goto_begin(exit_label)
        - continue transforms to goto_end(iter_label)
        """
        from ..inline.constant_loop_adapter import ConstantLoopAdapter
        from ..inline._intrinsics import _PC_INTRINSICS

        cf = self._get_cf_builder()

        # Use AST transformation
        adapter = ConstantLoopAdapter(self)
        result = adapter.transform_constant_loop(node, py_iterable)

        # Debug: output transformed AST if enabled
        from ..utils.ast_debug import ast_debugger
        ast_debugger.capture(
            "constant_loop_unroll",
            result.stmts,
            original=ast.unparse(node) if hasattr(ast, 'unparse') else str(node),
            exit_label=result.exit_label
        )

        # Fix locations for all transformed statements
        for stmt in result.stmts:
            ast.copy_location(stmt, node)
            ast.fix_missing_locations(stmt)

        # Ensure intrinsic namespace is available in user_globals
        old_user_globals = self.ctx.user_globals
        merged_globals = {}
        if old_user_globals:
            merged_globals.update(old_user_globals)
        merged_globals['__pc_intrinsics'] = _PC_INTRINSICS
        self.ctx.user_globals = merged_globals
        
        try:
            # Visit the transformed statements
            self._visit_stmt_list(result.stmts, add_to_cfg=True)
        finally:
            # Restore user_globals
            self.ctx.user_globals = old_user_globals
