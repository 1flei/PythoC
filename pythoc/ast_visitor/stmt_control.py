"""
Control flow statement visitor mixin (return, break, continue)
"""

import ast
from ..valueref import ensure_ir, ValueRef
from ..logger import logger
from ..scope_manager import ScopeType
from .control_flow_builder import ControlFlowBuilder


class ControlFlowMixin:
    """Mixin for control flow statements: return, break, continue"""
    
    def _get_cf_builder(self) -> ControlFlowBuilder:
        """Get or create the ControlFlowBuilder for this visitor"""
        if not hasattr(self, '_cf_builder') or self._cf_builder is None:
            func_name = ""
            if hasattr(self, 'current_function') and self.current_function:
                func_name = self.current_function.name
            self._cf_builder = ControlFlowBuilder(self, func_name)
        return self._cf_builder
    
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
    
    def _execute_deferred_calls_for_return(self):
        """Emit all deferred calls before return (all scopes)
        
        Note: We do NOT clear the defer stack here because other branches
        (e.g., else branch in if-then-return pattern) may also need to emit
        the same defers. The defer stack is managed at scope exit, not at return.
        
        Each branch that returns will emit its own copy of the deferred calls.
        
        Linear token checking is done by CFG merge point analysis at the
        virtual exit block, not here.
        """
        # Use ScopeManager to emit all defers
        cf = self._get_cf_builder()
        self.scope_manager.emit_all_defers(cf)
    
    def _emit_deferred_calls_for_scope(self, scope_depth: int):
        """Emit deferred calls for a specific scope (without unregistering)
        
        Used at normal block exit points. The unregister happens in finally block.
        
        Note: This is a legacy API. New code should use scope_manager directly.
        """
        # Find the scope with this depth and emit its defers
        for scope in reversed(self.scope_manager._scopes):
            if scope.depth == scope_depth:
                self.scope_manager._emit_defers_for_scope(scope)
                return
        # Fallback to old API if scope not found in scope_manager
        from ..builtin_entities.defer import emit_deferred_calls
        emit_deferred_calls(self, scope_depth=scope_depth)
    
    def _execute_deferred_calls_for_scope(self, scope_depth: int):
        """Emit and unregister deferred calls for a specific scope (legacy API)"""
        from ..builtin_entities.defer import execute_deferred_calls
        execute_deferred_calls(self, scope_depth=scope_depth)
    
    def _emit_deferred_calls_down_to_scope(self, target_scope_depth: int):
        """Emit deferred calls from current scope down to target scope (inclusive)
        
        Used by break/continue to emit defers for all nested scopes
        before jumping out to the loop scope. Does not unregister.
        """
        # Use ScopeManager to emit defers down to target depth
        cf = self._get_cf_builder()
        logger.debug(f"Emitting defers from scope {self.scope_manager.current_depth} down to {target_scope_depth}")
        self.scope_manager.exit_scopes_to(target_scope_depth - 1, cf, emit_defers=True)
    
    def _execute_deferred_calls_down_to_scope(self, target_scope_depth: int):
        """Execute deferred calls from current scope down to target scope (legacy API)"""
        self._emit_deferred_calls_down_to_scope(target_scope_depth)
    
    def visit_Return(self, node: ast.Return):
        """Handle return statements with termination check
        
        ABI coercion for struct returns is handled by LLVMBuilder.ret().
        
        Defer semantics follow Zig/Go: return value is evaluated BEFORE defers execute,
        so defer cannot modify the return value (unless using pointers to external state).
        """
        cf = self._get_cf_builder()
        
        # Only add return if block is not already terminated
        expected_pc_type = None
        for name, hint in self.func_type_hints.items():
            if name != '_sret_info':  # Skip internal sret info
                expected_pc_type = hint.get("return")
        if not cf.is_terminated():
            # Evaluate return value first (before executing defers)
            value = None
            if node.value:
                # Evaluate the return value first to get ValueRef with tracking info
                value = self.visit_expression(node.value)
                
                # Transfer linear ownership using ValueRef tracking info
                # This consumes all active linear paths in the returned value
                self._transfer_linear_ownership(value, reason="return", node=node)
                
                # convert to expected_pc_type is specified
                if expected_pc_type is not None:
                    value = self.implicit_coercer.coerce(value, expected_pc_type, node)
            
            # Execute all deferred calls after return value is evaluated (Zig/Go semantics)
            self._execute_deferred_calls_for_return()
            
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
            
            # Mark return in CFG
            cf.mark_return()
        # else: block already terminated, this is unreachable code, silently ignore

    def visit_Break(self, node: ast.Break):
        """Handle break statements
        
        Deferred calls for all scopes from current down to loop scope are executed.
        """
        if not self.loop_stack:
            logger.error("'break' outside loop", node=node, exc_type=SyntaxError)
        
        cf = self._get_cf_builder()
        if not cf.is_terminated():
            # Get loop scope depth from scope_manager or legacy loop_scope_stack
            loop_scope_depth = self.scope_manager.get_loop_scope_depth()
            if loop_scope_depth is None:
                # Fallback to legacy
                loop_scope_depth = self.loop_scope_stack[-1] if self.loop_scope_stack else self.scope_depth
            
            logger.debug(f"Break: current scope={self.scope_manager.current_depth}, loop scope={loop_scope_depth}")
            
            # Emit deferred calls from current scope down to loop scope (inclusive)
            self._emit_deferred_calls_down_to_scope(loop_scope_depth)
            
            # Get the break target from scope_manager or legacy loop_stack
            _, break_block = self.scope_manager.get_loop_targets()
            if break_block is None:
                # Fallback to legacy
                _, break_block = self.loop_stack[-1]
            
            # Add break edge to CFG and generate IR
            cf.branch(break_block)
            # Update the edge kind to 'break'
            for edge in reversed(cf.cfg.edges):
                if edge.target_id == cf._get_cfg_block_id(break_block):
                    edge.kind = 'break'
                    break

    def visit_Continue(self, node: ast.Continue):
        """Handle continue statements
        
        Deferred calls for all scopes from current down to loop scope are executed.
        """
        if not self.loop_stack:
            logger.error("'continue' outside loop", node=node, exc_type=SyntaxError)
        
        cf = self._get_cf_builder()
        if not cf.is_terminated():
            # Get loop scope depth from scope_manager or legacy loop_scope_stack
            loop_scope_depth = self.scope_manager.get_loop_scope_depth()
            if loop_scope_depth is None:
                # Fallback to legacy
                loop_scope_depth = self.loop_scope_stack[-1] if self.loop_scope_stack else self.scope_depth
            
            # Emit deferred calls from current scope down to loop scope (inclusive)
            self._emit_deferred_calls_down_to_scope(loop_scope_depth)
            
            # Get the continue target from scope_manager or legacy loop_stack
            continue_block, _ = self.scope_manager.get_loop_targets()
            if continue_block is None:
                # Fallback to legacy
                continue_block, _ = self.loop_stack[-1]
            
            # Add continue edge to CFG and generate IR
            cf.branch(continue_block)
            # Update the edge kind to 'continue'
            for edge in reversed(cf.cfg.edges):
                if edge.target_id == cf._get_cfg_block_id(continue_block):
                    edge.kind = 'continue'
                    break

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
                                       node=node) as scope:
            # Keep legacy scope_depth in sync
            self.scope_depth = self.scope_manager.current_depth
            
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
        
        # Restore scope_depth after exiting scope
        self.scope_depth = self.scope_manager.current_depth
