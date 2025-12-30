"""
Control flow statement visitor mixin (return, break, continue)
"""

import ast
from ..valueref import ensure_ir, ValueRef
from ..logger import logger
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
    
    def _execute_deferred_calls_for_return(self):
        """Emit all deferred calls before return (all scopes) and clear stack
        
        Return terminates the function, so we emit all defers and clear the stack.
        """
        from ..builtin_entities.defer import emit_deferred_calls, _init_defer_registry
        emit_deferred_calls(self, all_scopes=True)
        # Clear the defer stack since return terminates the function
        _init_defer_registry(self)
        self._defer_stack = []
    
    def _emit_deferred_calls_for_scope(self, scope_depth: int):
        """Emit deferred calls for a specific scope (without unregistering)
        
        Used at normal block exit points. The unregister happens in finally block.
        """
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
        from ..builtin_entities.defer import emit_deferred_calls
        # Emit defers from current scope down to target scope
        logger.debug(f"Emitting defers from scope {self.scope_depth} down to {target_scope_depth}")
        for depth in range(self.scope_depth, target_scope_depth - 1, -1):
            logger.debug(f"  Emitting defers for scope {depth}")
            emit_deferred_calls(self, scope_depth=depth)
    
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
            # This follows Zig/Go semantics: defer cannot modify the return value
            value = None
            if node.value:
                # Evaluate the return value first to get ValueRef with tracking info
                value = self.visit_expression(node.value)
                
                # Transfer linear ownership using ValueRef tracking info
                # This consumes all active linear paths in the returned value
                self._transfer_linear_ownership(value, reason="return", node=node)
                
                # convert to expected_pc_type is specified
                if expected_pc_type is not None:
                    value = self.type_converter.convert(value, expected_pc_type)
            
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
            # Get loop scope depth from loop_stack
            loop_scope_depth = self.loop_scope_stack[-1] if hasattr(self, 'loop_scope_stack') and self.loop_scope_stack else self.scope_depth
            
            logger.debug(f"Break: current scope={self.scope_depth}, loop scope={loop_scope_depth}")
            
            # Execute deferred calls from current scope down to loop scope (inclusive)
            self._execute_deferred_calls_down_to_scope(loop_scope_depth)
            
            # Get the break target (loop exit block or after_else block)
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
            # Get loop scope depth from loop_stack
            loop_scope_depth = self.loop_scope_stack[-1] if hasattr(self, 'loop_scope_stack') and self.loop_scope_stack else self.scope_depth
            
            # Execute deferred calls from current scope down to loop scope (inclusive)
            self._execute_deferred_calls_down_to_scope(loop_scope_depth)
            
            # Get the continue target (loop header block)
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
