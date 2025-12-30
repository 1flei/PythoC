"""
Loop statement visitor mixin (for, while)
"""

import ast
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..registry import VariableInfo
from ..logger import logger
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
        
        # Don't process if current block is already terminated
        if cf.is_terminated():
            return
        
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
        
        # Push loop context: continue -> loop back to body start, break -> exit
        self.loop_stack.append((loop_body_start, loop_exit))
        
        # Enter scope for loop body
        self.ctx.var_registry.enter_scope()
        self.scope_depth += 1
        
        try:
            # Execute loop body
            for stmt in node.body:
                if not cf.is_terminated():
                    cf.add_stmt(stmt)
                    self.visit(stmt)
            
            # Check linear tokens in current scope
            for var_info in self.ctx.var_registry.get_all_in_current_scope():
                if var_info.linear_state is not None and var_info.linear_scope_depth == self.scope_depth:
                    if var_info.linear_state != 'consumed':
                        actual_line = self._get_actual_line_number(var_info.line_number)
                        logger.error(
                            f"Linear token '{var_info.name}' not consumed in while True body "
                            f"(declared at line {actual_line})", node
                        )
                    var_info.linear_state = None
            
            # If body can fall through, it's an infinite loop - loop back
            if not cf.is_terminated():
                cf.branch(loop_body_start)
                cf.mark_loop_back(loop_body_start)
        finally:
            self.scope_depth -= 1
            self.ctx.var_registry.exit_scope()
        
        # Pop loop context
        self.loop_stack.pop()
        
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
        
        # Push loop context for break/continue
        # break -> loop_exit (skips else)
        self.loop_stack.append((loop_header, loop_exit))
        
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
        
        # Loop body - increment scope depth for linear token restrictions
        cf.position_at_end(loop_body)
        
        # Enter new scope for loop body and increment scope depth
        self.ctx.var_registry.enter_scope()
        self.scope_depth += 1
        try:
            # Execute loop body statements
            for stmt in node.body:
                if not cf.is_terminated():
                    cf.add_stmt(stmt)
                    self.visit(stmt)
            
            # Check that all linear tokens created in loop are consumed
            for var_info in self.ctx.var_registry.get_all_in_current_scope():
                if var_info.linear_state is not None and var_info.linear_scope_depth == self.scope_depth:
                    if var_info.linear_state != 'consumed':
                        actual_line = self._get_actual_line_number(var_info.line_number)
                        logger.error(
                            f"Linear token '{var_info.name}' not consumed in loop "
                            f"(declared at line {actual_line})", node
                        )
                    # Clean up consumed token
                    var_info.linear_state = None
            
            # Loop invariant is checked by CFG linear checker at function end
        finally:
            # Decrement scope depth and exit scope
            self.scope_depth -= 1
            self.ctx.var_registry.exit_scope()
        
        # Jump back to header (if not terminated by return/break)
        if not cf.is_terminated():
            cf.branch(loop_header)
            cf.mark_loop_back(loop_header)
        
        # Pop loop context
        self.loop_stack.pop()
        
        # Handle else block if present
        if has_else:
            cf.position_at_end(else_block)
            for stmt in node.orelse:
                if not cf.is_terminated():
                    cf.add_stmt(stmt)
                    self.visit(stmt)
            if not cf.is_terminated():
                cf.branch(loop_exit)
        
        # Continue after loop
        cf.position_at_end(loop_exit)

    def visit_For(self, node: ast.For):
        """Handle for loops using iterator protocol
        
        Supports two protocols (in priority order):
        1. Yield inlining: inline yield function body for zero overhead (REQUIRED for yield)
        2. Compile-time constant unrolling: for loops over constant sequences
        
        Translates:
            for i in iterable:
                body
            else:
                else_body
        
        To (if inlined):
            # Inlined yield function body with yields replaced by loop body
            # else_body executes if no break occurred
        
        Note: Vtable iterator protocol has been removed. All yield functions
        must be inlined at compile time.
        """
        # First evaluate the iterator expression
        iter_val = self.visit_expression(node.iter)
        
        # Check for compile-time constant (Python value)
        if iter_val.is_python_value() and hasattr(iter_val.get_python_value(), "__iter__"):
            py_iterable = iter_val.get_python_value()
            self._visit_for_with_constant_unroll(node, py_iterable)
            return
        
        # Check for yield inlining (REQUIRED - no fallback to vtable)
        if hasattr(iter_val, '_yield_inline_info') and iter_val._yield_inline_info:
            self._visit_for_with_yield_inline(node, iter_val)
            return
        
        # No vtable support - error if not handled above
        logger.error(
            f"Unsupported iterator type: {iter_val}. "
            f"Only yield functions (via inlining) and compile-time constants are supported. "
            f"Vtable iterator protocol has been removed.",
            node=node, exc_type=TypeError
        )
    
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
        
        Pure goto approach:
        - break in loop body -> __goto("_for_after_else_{id}")
        - continue in loop body -> __goto("_yield_next_{id}")
        - After all yields and else, place __label("_for_after_else_{id}")
        """
        from ..inline.yield_adapter import YieldInlineAdapter
        
        cf = self._get_cf_builder()
        adapter = YieldInlineAdapter(self)
        inline_info = iter_val._yield_inline_info
        
        # Extract func_obj to get its __globals__
        func_obj = inline_info.get('func_obj', None)
        
        # Get inlined statements
        # after_else_label is the label name for break to jump to (skip else)
        inlined_stmts, old_user_globals, after_else_label = adapter.try_inline_for_loop(
            node,
            inline_info['original_ast'],
            inline_info['call_node'],
            func_obj=func_obj
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
            for stmt in inlined_stmts:
                if not cf.is_terminated():
                    cf.add_stmt(stmt)
                    self.visit(stmt)
            
            # Execute else clause if present (only reached if no break occurred)
            if node.orelse:
                for stmt in node.orelse:
                    if not cf.is_terminated():
                        cf.add_stmt(stmt)
                        self.visit(stmt)
            
            # Place the after_else label (break jumps here to skip else)
            if after_else_label:
                # Create __label("_for_after_else_{id}") statement
                label_stmt = ast.Expr(value=ast.Call(
                    func=ast.Name(id='__label', ctx=ast.Load()),
                    args=[ast.Constant(value=after_else_label)],
                    keywords=[]
                ))
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
        
        Translates:
            for i in [1, 2, 3]:
                body
            else:
                else_body
        
        To (unrolled with blocks for break/continue support):
            iter_0:
                i = 1
                body
                br iter_1
            iter_1:
                i = 2
                body
                br iter_2
            iter_2:
                i = 3
                body
                br check_break
            check_break:
                if broke: br after_else
                else: br for_else
            for_else:
                else_body
                br after_else
            after_else:
        """
        from ..builtin_entities.python_type import PythonType

        cf = self._get_cf_builder()
        py_iterable = list(py_iterable)
        
        # Handle empty iterator
        if len(py_iterable) == 0:
            # Empty iterator: execute else clause if present
            if node.orelse:
                for stmt in node.orelse:
                    if not cf.is_terminated():
                        cf.add_stmt(stmt)
                        self.visit(stmt)
            return
        
        # Get loop variable pattern (supports nested tuple unpacking)
        # Returns a structure that mirrors the target pattern:
        # - str for ast.Name
        # - list for ast.Tuple (can contain str or nested list)
        def parse_target_pattern(target):
            """Parse target pattern recursively, returns str or list"""
            if isinstance(target, ast.Name):
                return target.id
            elif isinstance(target, ast.Tuple):
                return [parse_target_pattern(elt) for elt in target.elts]
            else:
                logger.error("Unsupported loop target type in constant unroll",
                            node=node, exc_type=NotImplementedError)
        
        loop_var_pattern = parse_target_pattern(node.target)
        is_tuple_unpack = isinstance(loop_var_pattern, list)
        
        # Check if loop body contains any break statement
        body_has_break = _has_break_in_body(node.body)
        
        # Create loop exit block
        loop_exit = cf.create_block("const_loop_exit")
        
        # If there's an else clause and body has break, create after_else block
        # break will jump directly to after_else (skipping else)
        if node.orelse and body_has_break:
            after_else = cf.create_block("after_const_loop_else")
            # break target is after_else (skip else)
            break_target = after_else
        else:
            after_else = None
            # break target is loop_exit
            break_target = loop_exit
        
        try:
            # Unroll with basic blocks
            for i, element in enumerate(py_iterable):
                if cf.is_terminated():
                    break
                
                # Enter new scope for this iteration
                self.ctx.var_registry.enter_scope()
                # Increment scope depth for this iteration
                self.scope_depth += 1
                
                # Determine continue target
                is_last = (i == len(py_iterable) - 1)
                if is_last:
                    continue_target = loop_exit
                else:
                    continue_target = cf.create_block(f"const_loop_iter_{i+1}")
                
                # Push loop context for this iteration (for break/continue support)
                # continue -> jump to continue_target (next iteration or loop_exit if last)
                # break -> jump to break_target (after_else if has else+break, else loop_exit)
                self.loop_stack.append((continue_target, break_target))
                
                # Helper function to bind variables recursively
                def bind_vars_recursive(pattern, value):
                    """Recursively bind variables according to pattern.
                    pattern: str (variable name) or list (nested pattern)
                    value: Python value to bind
                    """
                    if isinstance(pattern, str):
                        # Simple variable binding
                        if isinstance(value, ValueRef):
                            elem_value_ref = value
                        else:
                            elem_value_ref = wrap_value(
                                value,
                                kind="python",
                                type_hint=PythonType.wrap(value, is_constant=True)
                            )
                        loop_var_info = VariableInfo(
                            name=pattern,
                            value_ref=elem_value_ref,
                            alloca=None,
                            source="for_loop_unrolled"
                        )
                        self.ctx.var_registry.declare(loop_var_info, allow_shadow=True)
                    elif isinstance(pattern, list):
                        # Nested tuple unpacking
                        if not isinstance(value, (tuple, list)) or len(value) != len(pattern):
                            logger.error(
                                f"Cannot unpack {value} into {len(pattern)} variables",
                                node=node, exc_type=TypeError
                            )
                        for sub_pattern, sub_value in zip(pattern, value):
                            bind_vars_recursive(sub_pattern, sub_value)
                    else:
                        logger.error(f"Invalid pattern type: {type(pattern)}", 
                                    node=node, exc_type=TypeError)
                
                # Bind loop variables using the pattern
                bind_vars_recursive(loop_var_pattern, element)
                
                try:
                    # Execute loop body (break/continue will use loop_stack)
                    for stmt in node.body:
                        if not cf.is_terminated():
                            cf.add_stmt(stmt)
                            self.visit(stmt)
                        else:
                            # Block is terminated, but we still need to process remaining statements
                            # to generate their basic blocks (they might be reachable from other paths)
                            # Create a new unreachable block to continue codegen
                            unreachable_block = cf.create_block("unreachable_cont")
                            cf.position_at_end(unreachable_block)
                            cf.add_stmt(stmt)
                            self.visit(stmt)
                    
                    # Check that all linear tokens created in this iteration are consumed
                    for var_info in self.ctx.var_registry.get_all_in_current_scope():
                        if var_info.linear_state is not None and var_info.linear_scope_depth == self.scope_depth:
                            if var_info.linear_state != 'consumed':
                                actual_line = self._get_actual_line_number(var_info.line_number)
                                logger.error(
                                    f"Linear token '{var_info.name}' not consumed in loop iteration "
                                    f"(declared at line {actual_line})", node
                                )
                finally:
                    # Pop loop context
                    self.loop_stack.pop()
                    # Linear tokens check is done by CFG checker at function end
                    # Decrement scope depth and exit scope
                    self.scope_depth -= 1
                    self.ctx.var_registry.exit_scope()
                
                # Branch to next iteration
                if not cf.is_terminated():
                    cf.branch(continue_target)
                
                # Position at next block for next iteration (if not last)
                # Note: Even if current block is terminated, we still need to process
                # remaining iterations, as the terminator might be in a conditional branch
                if not is_last:
                    cf.position_at_end(continue_target)
            
            # Position at exit (normal loop completion)
            cf.position_at_end(loop_exit)
            
            # Handle else clause if present
            if node.orelse:
                # Execute else clause (only reached if no break occurred)
                for stmt in node.orelse:
                    if not cf.is_terminated():
                        cf.add_stmt(stmt)
                        self.visit(stmt)
                
                # If has break, branch to after_else and position there
                if body_has_break and after_else:
                    if not cf.is_terminated():
                        cf.branch(after_else)
                    cf.position_at_end(after_else)
        finally:
            pass  # No break_flag cleanup needed anymore
