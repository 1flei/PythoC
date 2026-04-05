"""
Control flow statement visitor mixin (return, break, continue)
"""

import ast
from ..valueref import ensure_ir, ValueRef
from ..logger import logger
from ..scope_manager import ScopeType


class ControlFlowMixin:
    """Mixin for control flow statements: return, break, continue"""

    def _visit_stmt_list(self, stmts, add_to_cfg: bool = True):
        """Visit a list of statements, handling terminated blocks properly.
        
        When current block is terminated, we still need to visit nested
        with-label statements to register labels for forward goto resolution.
        Other statements are skipped when block is terminated.
        
        Args:
            stmts: List of AST statements to visit
            add_to_cfg: Whether to add statements to CFG (default True)
        """
        cf = self._get_cf_builder()
        for stmt in stmts:
            if cf.is_terminated():
                # Check if this is a with-label statement that needs visiting
                if isinstance(stmt, ast.With) and len(stmt.items) == 1:
                    ctx_expr = self.visit_expression(stmt.items[0].context_expr)
                    if ctx_expr.is_python_value():
                        py_val = ctx_expr.get_python_value()
                        if isinstance(py_val, tuple) and len(py_val) == 2 and py_val[0] == '__scoped_label__':
                            if add_to_cfg:
                                cf.add_stmt(stmt)
                            self.visit(stmt)
                            continue
                # Skip non-label statements when block is terminated
                continue
            if add_to_cfg:
                cf.add_stmt(stmt)
            self.visit(stmt)
    
    def visit_Return(self, node: ast.Return):
        """Handle return statements with termination check
        
        ABI coercion for struct returns is handled by LLVMBuilder.ret().
        
        Defer semantics follow Zig/Go: return value is evaluated BEFORE defers execute,
        so defer cannot modify the return value (unless using pointers to external state).
        """
        cf = self._get_cf_builder()
        
        # Only add return if block is not already terminated
        expected_pc_type = self.current_return_type_hint
        if not cf.is_terminated():
            # Evaluate return value first (before executing defers)
            value = None
            if node.value:
                # Evaluate the return value first to get ValueRef with tracking info
                value = self.visit_rvalue_expression(node.value)
                
                # Transfer linear ownership using ValueRef tracking info
                # This consumes all active linear paths in the returned value
                self._transfer_linear_ownership(value, reason="return", node=node)
                
                # convert to expected_pc_type is specified
                if expected_pc_type is not None:
                    value = self.implicit_coercer.coerce(value, expected_pc_type, node)
            
            # Execute all deferred calls after return value is evaluated (Zig/Go semantics)
            self.scope_manager.emit_defers_to_depth(0, cf)
            
            # Now generate the actual return
            if value is not None:
                # Check if return type is void
                from ..builtin_entities.types import void
                if expected_pc_type is not None and expected_pc_type == void:
                    self.builder.ret_void()
                else:
                    # LLVMBuilder.ret() handles ABI coercion automatically
                    value_ir = ensure_ir(value)
                    self.builder.ret(value_ir)
            else:
                self.builder.ret_void()

            # mark_return() is now called by self.builder.ret()/ret_void() automatically
        # else: block already terminated, this is unreachable code, silently ignore

    def visit_Break(self, node: ast.Break):
        """Handle break statements

        Deferred calls for all scopes from current down to loop scope are executed.
        """
        if not self.scope_manager.is_in_loop():
            logger.error("'break' outside loop", node=node, exc_type=SyntaxError)

        cf = self._get_cf_builder()
        if not cf.is_terminated():
            loop_scope_depth = self.scope_manager.get_loop_scope_depth()

            logger.debug(f"Break: current scope={self.scope_manager.current_depth}, loop scope={loop_scope_depth}")

            # Emit deferred calls from current scope down to loop scope (inclusive)
            self.scope_manager.emit_defers_to_depth(loop_scope_depth - 1, cf)

            # Get the break target from scope_manager
            _, break_block = self.scope_manager.get_loop_targets()

            # Add break edge to CFG and generate IR
            cf.branch(break_block, kind='break')

    def visit_Continue(self, node: ast.Continue):
        """Handle continue statements

        Deferred calls for all scopes from current down to loop scope are executed.
        """
        if not self.scope_manager.is_in_loop():
            logger.error("'continue' outside loop", node=node, exc_type=SyntaxError)

        cf = self._get_cf_builder()
        if not cf.is_terminated():
            loop_scope_depth = self.scope_manager.get_loop_scope_depth()

            # Emit deferred calls from current scope down to loop scope (inclusive)
            self.scope_manager.emit_defers_to_depth(loop_scope_depth - 1, cf)

            # Get the continue target from scope_manager
            continue_block, _ = self.scope_manager.get_loop_targets()

            # Add continue edge to CFG and generate IR
            cf.branch(continue_block, kind='continue')

    def visit_Expr(self, node: ast.Expr):
        """Handle expression statements (like function calls)"""
        result = self.visit_expression(node.value)
        
        # Check for dangling linear expressions
        # Linear values must be either assigned to a variable or passed to a function
        if isinstance(result, ValueRef) and self._is_linear_type(result.type_hint):
            logger.error(
                f"Linear expression at line {node.lineno} is not consumed. "
                f"Assign it to a variable or pass it to a function.",
                node=node, exc_type=TypeError
            )
        
        return result
    
    def visit_With(self, node: ast.With):
        """Handle with statements - currently only supports scoped labels
        
        Syntax:
            with label("name"):
                # body can use goto_begin("name") or goto_end("name")
                pass
        """
        # Currently only support single context manager
        if len(node.items) != 1:
            logger.error("with statement currently only supports single context manager",
                        node=node, exc_type=SyntaxError)
        
        item = node.items[0]
        
        # Evaluate the context expression
        ctx_expr = self.visit_expression(item.context_expr)
        
        # Check if this is a scoped label
        if ctx_expr.is_python_value():
            py_val = ctx_expr.get_python_value()
            if isinstance(py_val, tuple) and len(py_val) == 2 and py_val[0] == '__scoped_label__':
                label_name = py_val[1]
                self._visit_with_scoped_label(node, label_name)
                return
        
        # Other with statements not supported yet
        logger.error("with statement currently only supports 'label' context manager",
                    node=node, exc_type=NotImplementedError)
    
    def _visit_with_scoped_label(self, node: ast.With, label_name: str):
        """Handle with label("name"): statement
        
        Creates a scoped label with begin and end blocks.
        """
        from ..builtin_entities.scoped_label import label as LabelClass
        
        # Enter label scope (creates begin/end blocks, pushes context)
        ctx = LabelClass.enter_label_scope(self, label_name, node)
        
        # Use ScopeManager for the label body
        with self.scope_manager.scope(ScopeType.LABEL, self._get_cf_builder(),
                                       label_ctx=ctx, node=node) as scope:
            try:
                # Visit body statements (don't add to CFG, label body is special)
                self._visit_stmt_list(node.body, add_to_cfg=False)
            finally:
                # Exit label scope (branches to end block, pops label stack)
                # NOTE: Does NOT position_at_end - that's done after scope_manager exits
                LabelClass.exit_label_scope(self, ctx)

        # Position at end block AFTER scope_manager.exit_scope()
        # This ensures is_terminated() check in exit_scope works correctly
        cf = self._get_cf_builder()
        cf.position_at_end(ctx.end_block)
