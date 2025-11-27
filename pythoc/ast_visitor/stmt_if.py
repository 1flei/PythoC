"""
If statement visitor mixin
"""

import ast
from llvmlite import ir
from ..valueref import ensure_ir, ValueRef
from ..logger import logger


class IfStatementMixin:
    """Mixin for if statement handling"""

    def process_condition(self, condition: ValueRef, then_fn, else_fn=None):
        """Handle condition with proper control flow
        
        Args:
            condition: ValueRef representing the condition to test
            then_branch: a callable that generates the then branch
            else_branch: None, or a callable that generates the else branch
        
        Callables receive no arguments and should generate code in current block.
        This allows match statements to bind variables before executing body.
        """
        logger.debug(f"If condition={condition}")
        
        # Handle Python constant condition (compile-time evaluation)
        if condition.is_python_value():
            py_cond = condition.get_python_value()
            if py_cond:
                then_fn()
            else:
                if else_fn:
                    else_fn()
            return

        condition = self._to_boolean(condition)
        
        # Create basic blocks
        then_block = self.current_function.append_basic_block(self.get_next_label("then"))
        
        # For if-else, we need an else block. For simple if, merge handles the "else" case
        if else_fn:
            else_block = self.current_function.append_basic_block(self.get_next_label("else"))
            merge_block = self.current_function.append_basic_block(self.get_next_label("merge"))
            self.builder.cbranch(condition, then_block, else_block)
        else:
            merge_block = self.current_function.append_basic_block(self.get_next_label("merge"))
            self.builder.cbranch(condition, then_block, merge_block)
        
        # Generate then block
        self.builder.position_at_end(then_block)
        then_terminated = False
        then_fn()
        if not self.builder.block.is_terminated:
            self.builder.branch(merge_block)
        else:
            then_terminated = True
        
        # Generate else block if present
        else_terminated = False
        if else_fn:
            self.builder.position_at_end(else_block)
            else_fn()
            if not self.builder.block.is_terminated:
                self.builder.branch(merge_block)
            else:
                else_terminated = True
        
        # Handle merge block - continue execution here
        self.builder.position_at_end(merge_block)
        
        # Only add unreachable if ALL paths to merge are terminated
        # For simple if (no else), merge is reachable from the condition branch
        # For if-else, merge is unreachable only if both branches terminate
        if else_fn and then_terminated and else_terminated:
            self.builder.unreachable()
    
    def visit_If(self, node: ast.If):
        """Handle if statements with proper control flow"""
        # Don't process if current block is already terminated
        if self.builder.block.is_terminated:
            return

        # Save linear token states (path-based) before if statement
        linear_states_before = {}
        for var_info in self.ctx.var_registry.get_all_in_current_scope():
            if var_info.linear_states:
                # Deep copy the states dict
                linear_states_before[var_info.name] = dict(var_info.linear_states)
        
        # Normalize to callables
        def make_branch_fn(branch):
            # branch is a list of AST statements
            def execute_stmts():
                # Enter new scope for the if/else block (for variable isolation)
                # but DON'T increment scope_depth (linear tokens can still be consumed)
                self.ctx.var_registry.enter_scope()
                try:
                    for stmt in branch:
                        if not self.builder.block.is_terminated:
                            self.visit(stmt)
                finally:
                    # Exit scope even if there's an error
                    self.ctx.var_registry.exit_scope()
            return execute_stmts
            
        condition = self.visit_expression(node.test)
        then_fn = make_branch_fn(node.body)
        else_fn = make_branch_fn(node.orelse) if node.orelse else None
        
        # Track linear token states (path-based) in both branches
        then_linear_states = {}
        else_linear_states = {}
        
        # Execute branches and capture linear token states
        def then_fn_tracked():
            then_fn()
            # Capture states after then branch
            nonlocal then_linear_states
            for var_info in self.ctx.var_registry.get_all_in_current_scope():
                if var_info.linear_states:
                    then_linear_states[var_info.name] = dict(var_info.linear_states)
        
        def else_fn_tracked():
            if else_fn:
                # Reset to state before if for else branch
                # First, clear any linear states added in then branch
                for var_info in self.ctx.var_registry.get_all_in_current_scope():
                    if var_info.linear_states:
                        var_info.linear_states.clear()
                
                # Then restore states that existed before if
                for name, states_dict in linear_states_before.items():
                    var_info = self.lookup_variable(name)
                    if var_info:
                        var_info.linear_states = dict(states_dict)
                
                else_fn()
                # Capture states after else branch
                nonlocal else_linear_states
                for var_info in self.ctx.var_registry.get_all_in_current_scope():
                    if var_info.linear_states:
                        else_linear_states[var_info.name] = dict(var_info.linear_states)
        
        # Execute condition processing with tracked branches
        if else_fn:
            self.process_condition(condition, then_fn_tracked, else_fn_tracked)
            
            # Check that both branches handled linear tokens consistently
            # All tokens (with all paths) must end in the same state in both branches
            all_vars = set(then_linear_states.keys()) | set(else_linear_states.keys())
            for var_name in all_vars:
                then_states = then_linear_states.get(var_name, {})
                else_states = else_linear_states.get(var_name, {})
                
                # Get all paths for this variable
                all_paths = set(then_states.keys()) | set(else_states.keys())
                for path in all_paths:
                    then_state = then_states.get(path, 'unknown')
                    else_state = else_states.get(path, 'unknown')
                    
                    # 'consumed' states are considered handled, compatible with 'unknown' or 'consumed'
                    # But 'active' must match exactly
                    states_compatible = (
                        then_state == else_state or
                        (then_state == 'consumed' and else_state == 'unknown') or
                        (then_state == 'unknown' and else_state == 'consumed')
                    )
                    
                    if not states_compatible:
                        path_str = f"{var_name}[{']['.join(map(str, path))}]" if path else var_name
                        raise TypeError(
                            f"Linear token '{path_str}' must be handled consistently in all branches: "
                            f"then={then_state}, else={else_state} (line {getattr(node, 'lineno', '?')})"
                        )
        else:
            # Simple if without else - linear tokens must remain active
            self.process_condition(condition, then_fn_tracked, None)
            
            # For simple if, tokens can only remain active
            # Check all paths haven't changed
            for var_name, states_before in linear_states_before.items():
                var_info = self.lookup_variable(var_name)
                states_after = var_info.linear_states if var_info else {}
                
                for path, state_before in states_before.items():
                    state_after = states_after.get(path, 'unknown')
                    if state_before == 'active' and state_after != 'active':
                        path_str = f"{var_name}[{']['.join(map(str, path))}]" if path else var_name
                        raise TypeError(
                            f"Linear token '{path_str}' modified in if without else branch. "
                            f"All branches must handle tokens consistently (line {getattr(node, 'lineno', '?')})"
                        )

