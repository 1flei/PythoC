"""
Control flow statement visitor mixin (return, break, continue)
"""

import ast
from ..valueref import ensure_ir, ValueRef
from ..logger import logger


class ControlFlowMixin:
    """Mixin for control flow statements: return, break, continue"""
    
    def visit_Return(self, node: ast.Return):
        """Handle return statements with termination check"""
        # Only add return if block is not already terminated
        expected_pc_type = None
        pc_func_name = None
        for name, hint in self.func_type_hints.items():
            pc_func_name = name
            expected_pc_type = hint.get("return")
        logger.debug("Return statement", func=pc_func_name, expected_type=expected_pc_type)
        if not self.builder.block.is_terminated:
            if node.value:
                # Evaluate the return value first to get ValueRef with tracking info
                value = self.visit_expression(node.value)
                
                # Transfer linear ownership using ValueRef tracking info
                # This consumes all active linear paths in the returned value
                self._transfer_linear_ownership(value, reason="return")
                
                # convert to expected_pc_type is specified
                if expected_pc_type is not None:
                    logger.debug("Converting return value", value=value, expected_type=expected_pc_type)
                    value = self.type_converter.convert(value, expected_pc_type)
                logger.debug("Return value", value=value)
                
                # Check if return type is void
                from ..builtin_entities.types import void
                if expected_pc_type is not None and expected_pc_type == void:
                    # Void return - don't return a value
                    self.builder.ret_void()
                else:
                    value_ir = ensure_ir(value)
                    self.builder.ret(value_ir)
            else:
                self.builder.ret_void()
        # else: block already terminated, this is unreachable code, silently ignore

    def visit_Break(self, node: ast.Break):
        """Handle break statements"""
        if not self.loop_stack:
            raise SyntaxError("'break' outside loop")
        
        if not self.builder.block.is_terminated:
            # Set break flag if for-else is active
            if hasattr(self, '_current_break_flag') and self._current_break_flag is not None:
                from llvmlite import ir
                self.builder.store(ir.Constant(ir.IntType(1), 1), self._current_break_flag)
            
            # Get the break target (loop exit block)
            _, break_block = self.loop_stack[-1]
            self.builder.branch(break_block)

    def visit_Continue(self, node: ast.Continue):
        """Handle continue statements"""
        if not self.loop_stack:
            raise SyntaxError("'continue' outside loop")
        
        if not self.builder.block.is_terminated:
            # Get the continue target (loop header block)
            continue_block, _ = self.loop_stack[-1]
            self.builder.branch(continue_block)

    def visit_Expr(self, node: ast.Expr):
        """Handle expression statements (like function calls)"""
        result = self.visit_expression(node.value)
        
        # Check for dangling linear expressions
        # Linear values must be either assigned to a variable or passed to a function
        if isinstance(result, ValueRef) and self._is_linear_type(result.type_hint):
            raise TypeError(
                f"Linear expression at line {node.lineno} is not consumed. "
                f"Assign it to a variable or pass it to a function."
            )
        
        return result
