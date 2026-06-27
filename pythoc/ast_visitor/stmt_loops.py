"""
Loop statement visitor mixin (for, while)
"""

import ast
from ..logger import logger
from ..scope_manager import ScopeType


def _check_const_condition(test):
    """Detect compile-time constant conditions without generating IR.

    Uses AST-level analysis to detect ``while True``, ``while False``,
    ``while not True``, etc.  This avoids calling ``visit_expression``
    which would generate IR (and trigger side effects) for function-call
    conditions like ``while sealed.next(ptr(it)):``.
    """
    if isinstance(test, ast.Constant):
        return True, bool(test.value)
    if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Constant)):
        return True, not bool(test.operand.value)
    return False, None


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
        
        # Check for compile-time constant condition via AST analysis (not via
        # visit_expression, which would generate IR and trigger side effects
        # for function-call conditions like sealed.next(ptr(it))).
        is_constant_condition, constant_value = _check_const_condition(node.test)
        
        # while False - never executes
        if is_constant_condition and not constant_value:
            logger.debug("while False - skipping loop body entirely")
            return
        
        # while True - special handling for infinite loop
        if is_constant_condition and constant_value:
            self._visit_while_impl(node, cf, is_constant_true=True)
            return

        # Normal while loop with runtime condition
        self._visit_while_impl(node, cf, is_constant_true=False)
    
    def _visit_while_impl(self, node: ast.While, cf, is_constant_true: bool):
        """Unified while loop implementation.

        Handles both ``while True`` and ``while <cond>`` with the same
        control-flow skeleton:

            loop_top:           # header (cond) or body start (True)
                [cbranch]       # only for runtime condition
            loop_body:
                <body stmts>
                emit_defers     # back-edge defer emission
                branch loop_top
            [else_block:]       # only for runtime condition with else
            loop_exit:

        Args:
            node: The While AST node.
            cf: ControlFlowBuilder.
            is_constant_true: True for ``while True``, False for runtime cond.
        """
        has_else = not is_constant_true and node.orelse and len(node.orelse) > 0

        # -- Create blocks --
        loop_exit = cf.create_block("while_exit")

        if is_constant_true:
            # No separate header; body IS the loop top
            loop_top = cf.create_block("while_true_body")
            loop_body = loop_top
        else:
            loop_top = cf.create_block("while_header")
            loop_body = cf.create_block("while_body")
            if has_else:
                else_block = cf.create_block("while_else")

        # -- Branch to loop top --
        cf.branch(loop_top)

        # -- Condition check (runtime only) --
        if not is_constant_true:
            cf.position_at_end(loop_top)
            condition = self.value_dispatcher.to_boolean(self.visit_expression(node.test))
            false_target = else_block if has_else else loop_exit
            cf.cbranch(condition, loop_body, false_target)

        # -- Loop body --
        cf.position_at_end(loop_body)

        # continue target: loop_top (re-check cond) or loop_body (while True)
        with self.scope_manager.scope(ScopeType.LOOP, cf,
                                       continue_target=loop_top,
                                       break_target=loop_exit,
                                       node=node) as scope:
            self._visit_stmt_list(node.body, add_to_cfg=True)

            # Back-edge: emit defers for this iteration, then branch to loop top
            if not cf.is_terminated():
                self.scope_manager.emit_defers_to_depth(scope.depth - 1, cf)
            if not cf.is_terminated():
                cf.branch(loop_top)
                cf.mark_loop_back(loop_top)

        # -- Post-loop --
        if has_else:
            cf.position_at_end(else_block)
            self._visit_stmt_list(node.orelse, add_to_cfg=True)
            if not cf.is_terminated():
                cf.branch(loop_exit)

        cf.position_at_end(loop_exit)

        if is_constant_true and not cf.has_predecessors(loop_exit):
            # No break in while True -> infinite loop, code after is unreachable
            cf.unreachable()

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
        from ..inline.genexpr_builder import build_genexpr_yield_function_ast

        func_name = f"__pc_genexp_inline_{get_next_id()}"
        func_ast = build_genexpr_yield_function_ast(
            genexp_ast, func_name, for_node)

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
        from ..inline.kernel import inline_globals_scope

        cf = self._get_cf_builder()
        adapter = YieldInlineAdapter(self)
        inline_info = iter_val._yield_inline_info

        # Extract callee context for resolving names during yield inlining
        func_obj = inline_info.get('func_obj', None)
        callee_globals = inline_info.get('callee_globals', None)

        # The inlined generator body is spliced into this group's object code by
        # value, including the layout/size of any structs it allocates. If that
        # generator lives in another source file, this group must be recompiled
        # whenever that file changes; otherwise the cache keeps a stale frame
        # layout and corrupts memory. Record a source-embed dependency so the
        # incremental cache mtime-invalidates this group against the callee.
        self._record_inline_source_dependency(func_obj)

        # Get inlined result
        # after_else_label is the label name for break to jump to (skip else)
        inline_result, after_else_label = adapter.try_inline_for_loop(
            node,
            inline_info['original_ast'],
            inline_info['call_node'],
            func_obj=func_obj,
            callee_globals_override=callee_globals,
        )

        if inline_result is None:
            # Inlining failed - this is now an error
            logger.error(
                f"Yield function inlining failed for '{ast.unparse(node.iter)}'. "
                f"Yield functions must be inlinable (no complex control flow, recursion, etc.)",
                node=node, exc_type=TypeError
            )

        with inline_globals_scope(self, inline_result):
            # Visit each inlined statement
            self._visit_stmt_list(inline_result.stmts, add_to_cfg=True)

            # Execute else clause if present (only reached if no break occurred)
            if node.orelse:
                self._visit_stmt_list(node.orelse, add_to_cfg=True)

            # Place the after_else label as a scoped label (break jumps here to skip else)
            if after_else_label:
                from ..inline.exit_rules import _empty_label_block
                label_stmt = _empty_label_block(
                    ast.Constant(value=after_else_label)
                ).stmt
                ast.copy_location(label_stmt, node)
                ast.fix_missing_locations(label_stmt)
                if not cf.is_terminated():
                    cf.add_stmt(label_stmt)
                    self.visit(label_stmt)

    def _record_inline_source_dependency(self, func_obj):
        """Record a source-embed dependency on an inlined generator's file.

        Cross-module yield inlining copies the callee's body (and the layout of
        structs it allocates) into the current group's object file. The
        incremental cache only tracks each group's own source mtime, so without
        this edge a layout change in the callee leaves a stale frame size in
        this group's cached .o. Recording a "source_embed" dependency makes the
        cache rebuild this group when the callee's source file changes. Same-file
        inlining needs no edge: the group's own mtime already covers it.
        """
        if func_obj is None:
            return
        caller_group_key = self.current_group_key
        if not caller_group_key:
            return

        from ..utils.inspect_utils import get_function_file_with_inspect
        callee_file = get_function_file_with_inspect(func_obj)
        if not callee_file:
            return

        caller_file = caller_group_key[0] if len(caller_group_key) else None
        if caller_file == callee_file:
            return

        from ..build.deps import get_dependency_tracker
        get_dependency_tracker().record_group_dependency(
            tuple(caller_group_key),
            (callee_file, None, None, None),
            "source_embed",
        )

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
        from ..inline.kernel import InlineResult, inline_globals_scope

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

        # Use shared merge/restore lifecycle via inline_globals_scope
        inline_result = InlineResult(stmts=result.stmts, required_globals=result.required_globals)
        with inline_globals_scope(self, inline_result):
            # Visit the transformed statements
            self._visit_stmt_list(result.stmts, add_to_cfg=True)
