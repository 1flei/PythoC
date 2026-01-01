"""
If statement visitor mixin
"""

import ast
from llvmlite import ir
from ..valueref import ensure_ir, ValueRef
from ..logger import logger
from .control_flow_builder import ControlFlowBuilder


class IfStatementMixin:
    """Mixin for if statement handling"""

    def _get_cf_builder(self) -> ControlFlowBuilder:
        """Get or create the ControlFlowBuilder for this visitor"""
        if not hasattr(self, '_cf_builder') or self._cf_builder is None:
            func_name = ""
            if hasattr(self, 'current_function') and self.current_function:
                func_name = self.current_function.name
            self._cf_builder = ControlFlowBuilder(self, func_name)
        return self._cf_builder
    
    def _reset_cf_builder(self):
        """Reset the ControlFlowBuilder (call at start of new function)"""
        self._cf_builder = None

    def process_condition(self, condition: ValueRef, then_fn, else_fn=None):
        """Handle condition with proper control flow
        
        Args:
            condition: ValueRef representing the condition to test
            then_branch: a callable that generates the then branch
            else_branch: None, or a callable that generates the else branch
        
        Callables receive no arguments and should generate code in current block.
        This allows match statements to bind variables before executing body.
        
        Returns:
            tuple: (then_terminated, else_terminated) - whether each branch terminates
        """
        # Handle Python constant condition (compile-time evaluation)
        if condition.is_python_value():
            py_cond = condition.get_python_value()
            if py_cond:
                then_fn()
                return (True, False)  # Only then branch executed
            else:
                if else_fn:
                    else_fn()
                return (False, True)  # Only else branch executed

        condition = self._to_boolean(condition)
        
        # Use ControlFlowBuilder for block operations
        cf = self._get_cf_builder()
        
        # Create basic blocks
        then_block = cf.create_block("then")
        
        # For if-else, we need an else block. For simple if, merge handles the "else" case
        if else_fn:
            else_block = cf.create_block("else")
            merge_block = cf.create_block("merge")
            cf.cbranch(condition, then_block, else_block)
        else:
            merge_block = cf.create_block("merge")
            cf.cbranch(condition, then_block, merge_block)
        
        # Generate then block
        cf.position_at_end(then_block)
        then_terminated = False
        then_fn()
        if not cf.is_terminated():
            cf.branch(merge_block)
        else:
            then_terminated = True
        
        # Generate else block if present
        else_terminated = False
        if else_fn:
            cf.position_at_end(else_block)
            else_fn()
            if not cf.is_terminated():
                cf.branch(merge_block)
            else:
                else_terminated = True
        
        # Handle merge block - continue execution here
        cf.position_at_end(merge_block)
        
        # Only add unreachable if ALL paths to merge are terminated
        # For simple if (no else), merge is reachable from the condition branch
        # For if-else, merge is unreachable only if both branches terminate
        if else_fn and then_terminated and else_terminated:
            cf.unreachable()
        
        return (then_terminated, else_terminated)
    
    def visit_If(self, node: ast.If):
        """Handle if statements with proper control flow
        
        Linear state tracking is done by the AST visitor during execution.
        CFG-based linear checking is done at function end via CFG linear checker.
        
        For compile-time constant conditions (if True/False), only one branch
        executes.
        
        Note: We do NOT skip processing when terminated because the if body
        may contain label definitions that need to be registered for forward
        goto resolution.
        """
        cf = self._get_cf_builder()

        # Capture linear state before if statement for branch restoration
        linear_states_before = cf.capture_linear_snapshot()
        
        # Normalize to callables
        def make_branch_fn(branch):
            # branch is a list of AST statements
            def execute_stmts():
                # Enter new scope for the if/else block
                # Increment scope_depth so defer can track block-level defers
                self.ctx.var_registry.enter_scope()
                self.scope_depth += 1
                current_scope = self.scope_depth
                try:
                    self._visit_stmt_list(branch, add_to_cfg=True)
                    # Emit deferred calls for this block before exiting (normal exit)
                    if not cf.is_terminated():
                        self._emit_deferred_calls_for_scope(current_scope)
                finally:
                    # Unregister defers for this scope (they've been emitted at all exit points)
                    from ..builtin_entities.defer import unregister_defers_for_scope
                    unregister_defers_for_scope(self, current_scope)
                    # Exit scope even if there's an error
                    self.scope_depth -= 1
                    self.ctx.var_registry.exit_scope()
            return execute_stmts
            
        condition = self.visit_expression(node.test)
        then_fn = make_branch_fn(node.body)
        else_fn = make_branch_fn(node.orelse) if node.orelse else None
        
        # Execute branches - linear states are tracked by AST visitor
        # CFG checker will validate at function end
        def then_fn_tracked():
            then_fn()
        
        def else_fn_tracked():
            if else_fn:
                # Reset to state before if for else branch
                # This ensures else branch starts with same state as then branch
                cf.restore_linear_snapshot(linear_states_before)
                else_fn()
        
        # Execute condition processing
        if else_fn:
            self.process_condition(condition, then_fn_tracked, else_fn_tracked)
            # Linear state validation is done by CFG checker at function end
        else:
            # Simple if without else
            then_terminated, _ = self.process_condition(condition, then_fn_tracked, None)
            
            # If then branch terminates (return/break/continue), the code after the if
            # only executes when the condition is false. In this case, linear tokens
            # should be restored to their state before the if.
            if then_terminated and not condition.is_python_value():
                cf.restore_linear_snapshot(linear_states_before)
            # Linear state validation is done by CFG checker at function end

