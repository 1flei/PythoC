"""
Loop statement visitor mixin (for, while)
"""

import ast
from ..valueref import ValueRef, ensure_ir, wrap_value, get_type, get_type_hint
from ..registry import VariableInfo
from ..logger import logger


class LoopsMixin:
    """Mixin for loop statements: for, while"""
    
    def visit_While(self, node: ast.While):
        """Handle while loops"""
        # Don't process if current block is already terminated
        if self.builder.block.is_terminated:
            return
        
        # Create loop blocks
        loop_header = self.current_function.append_basic_block(self.get_next_label("while_header"))
        loop_body = self.current_function.append_basic_block(self.get_next_label("while_body"))
        loop_exit = self.current_function.append_basic_block(self.get_next_label("while_exit"))
        
        # Push loop context for break/continue
        self.loop_stack.append((loop_header, loop_exit))
        
        # Jump to loop header
        self.builder.branch(loop_header)
        
        # Loop header: check condition
        self.builder.position_at_end(loop_header)
        condition = self._to_boolean(self.visit_expression(node.test))
        self.builder.cbranch(condition, loop_body, loop_exit)
        
        # Loop body - increment scope depth for linear token restrictions
        self.builder.position_at_end(loop_body)
        
        # Enter new scope for loop body and increment scope depth
        self.ctx.var_registry.enter_scope()
        self.scope_depth += 1
        try:
            # Execute loop body statements
            for stmt in node.body:
                if not self.builder.block.is_terminated:
                    self.visit(stmt)
            
            # Check that all linear tokens created in loop are consumed
            for var_info in self.ctx.var_registry.get_all_in_current_scope():
                if var_info.linear_state is not None and var_info.linear_scope_depth == self.scope_depth:
                    if var_info.linear_state != 'consumed':
                        logger.error(
                            f"Linear token '{var_info.name}' not consumed in loop "
                            f"(declared at line {var_info.line_number})", node
                        )
                    # Clean up consumed token
                    var_info.linear_state = None
        finally:
            # Decrement scope depth and exit scope
            self.scope_depth -= 1
            self.ctx.var_registry.exit_scope()
        
        # Jump back to header (if not terminated by return/break)
        if not self.builder.block.is_terminated:
            self.builder.branch(loop_header)
        
        # Pop loop context
        self.loop_stack.pop()
        
        # Continue after loop
        self.builder.position_at_end(loop_exit)

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
        """Handle for loop with yield inline, including else clause"""
        from ..inline.yield_adapter import YieldInlineAdapter
        from llvmlite import ir
        
        adapter = YieldInlineAdapter(self)
        inline_info = iter_val._yield_inline_info
        
        # Extract func_obj to get its __globals__
        func_obj = inline_info.get('func_obj', None)
        
        inlined_stmts, old_user_globals = adapter.try_inline_for_loop(
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
            # If there's an else clause, we need to track if break occurred
            if node.orelse:
                # Create blocks for else handling
                else_block = self.current_function.append_basic_block(self.get_next_label("for_else"))
                after_else = self.current_function.append_basic_block(self.get_next_label("after_for_else"))
                
                # Allocate a flag to track if break occurred
                break_flag = self._create_alloca_in_entry(ir.IntType(1), "for_break_flag")
                self.builder.store(ir.Constant(ir.IntType(1), 0), break_flag)
                
                # Store break flag in loop context for break statement to set
                old_break_flag = getattr(self, '_current_break_flag', None)
                self._current_break_flag = break_flag
                
                try:
                    # Fix all missing locations in inlined statements
                    for stmt in inlined_stmts:
                        ast.fix_missing_locations(stmt)
                    
                    # Visit each inlined statement
                    for stmt in inlined_stmts:
                        if not self.builder.block.is_terminated:
                            self.visit(stmt)
                    
                    # After loop completes, check break flag
                    if not self.builder.block.is_terminated:
                        broke = self.builder.load(break_flag, "broke")
                        self.builder.cbranch(broke, after_else, else_block)
                    
                    # Else block: execute if no break
                    self.builder.position_at_end(else_block)
                    for stmt in node.orelse:
                        if not self.builder.block.is_terminated:
                            self.visit(stmt)
                    
                    if not self.builder.block.is_terminated:
                        self.builder.branch(after_else)
                    
                    # Continue after for-else
                    self.builder.position_at_end(after_else)
                finally:
                    self._current_break_flag = old_break_flag
            else:
                # No else clause, process normally
                # Fix all missing locations in inlined statements
                for stmt in inlined_stmts:
                    ast.fix_missing_locations(stmt)
                
                # Visit each inlined statement
                for stmt in inlined_stmts:
                    if not self.builder.block.is_terminated:
                        self.visit(stmt)
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
        from llvmlite import ir

        py_iterable = list(py_iterable)
        
        # Handle empty iterator
        if len(py_iterable) == 0:
            # Empty iterator: execute else clause if present
            if node.orelse:
                for stmt in node.orelse:
                    if not self.builder.block.is_terminated:
                        self.visit(stmt)
            return
        
        # Get loop variable name or names (for tuple unpacking)
        if isinstance(node.target, ast.Name):
            loop_var_names = [node.target.id]
            is_tuple_unpack = False
        elif isinstance(node.target, ast.Tuple):
            # Support tuple unpacking: for i, j in [(1,2), (3,4)]
            loop_var_names = []
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    loop_var_names.append(elt.id)
                else:
                    logger.error("Nested tuple unpacking not supported in constant unroll",
                                node=node, exc_type=NotImplementedError)
            is_tuple_unpack = True
        else:
            logger.error("Complex loop targets not supported in constant unroll",
                        node=node, exc_type=NotImplementedError)
        
        # Create loop exit block
        loop_exit = self.current_function.append_basic_block(
            self.get_next_label("const_loop_exit")
        )
        
        # If there's an else clause, need to track breaks
        if node.orelse:
            else_block = self.current_function.append_basic_block(
                self.get_next_label("const_loop_else")
            )
            after_else = self.current_function.append_basic_block(
                self.get_next_label("after_const_loop_else")
            )
            # Allocate break flag
            break_flag = self._create_alloca_in_entry(ir.IntType(1), "const_for_break_flag")
            self.builder.store(ir.Constant(ir.IntType(1), 0), break_flag)
            
            old_break_flag = getattr(self, '_current_break_flag', None)
            self._current_break_flag = break_flag
        else:
            break_flag = None
            old_break_flag = None
        
        try:
            # Unroll with basic blocks
            for i, element in enumerate(py_iterable):
                if self.builder.block.is_terminated:
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
                    continue_target = self.current_function.append_basic_block(
                        self.get_next_label(f"const_loop_iter_{i+1}")
                    )
                
                # Push loop context for this iteration (for break/continue support)
                # continue -> jump to continue_target (next iteration or loop_exit if last)
                # break -> jump to loop_exit (exit the entire loop)
                self.loop_stack.append((continue_target, loop_exit))
                
                # Set loop variable(s)
                if is_tuple_unpack:
                    # Unpack tuple element
                    if not isinstance(element, (tuple, list)) or len(element) != len(loop_var_names):
                        logger.error(
                            f"Cannot unpack {element} into {len(loop_var_names)} variables",
                            node=node, exc_type=TypeError
                        )
                    for var_name, elem_val in zip(loop_var_names, element):
                        elem_value_ref = wrap_value(
                            elem_val,
                            kind="python",
                            type_hint=PythonType.wrap(elem_val, is_constant=True)
                        )
                        loop_var_info = VariableInfo(
                            name=var_name,
                            value_ref=elem_value_ref,
                            alloca=None,
                            source="for_loop_unrolled"
                        )
                        self.ctx.var_registry.declare(loop_var_info, allow_shadow=True)
                else:
                    # Single variable
                    elem_value_ref = wrap_value(
                        element,
                        kind="python", 
                        type_hint=PythonType.wrap(element, is_constant=True)
                    )
                    loop_var_info = VariableInfo(
                        name=loop_var_names[0],
                        value_ref=elem_value_ref,
                        alloca=None,
                        source="for_loop_unrolled"
                    )
                    self.ctx.var_registry.declare(loop_var_info, allow_shadow=True)
                
                try:
                    # Execute loop body (break/continue will use loop_stack)
                    for stmt in node.body:
                        if not self.builder.block.is_terminated:
                            self.visit(stmt)
                        else:
                            # Block is terminated, but we still need to process remaining statements
                            # to generate their basic blocks (they might be reachable from other paths)
                            # Create a new unreachable block to continue codegen
                            unreachable_block = self.current_function.append_basic_block(
                                self.get_next_label("unreachable_cont")
                            )
                            self.builder.position_at_end(unreachable_block)
                            self.visit(stmt)
                    
                    # Check that all linear tokens created in this iteration are consumed
                    for var_info in self.ctx.var_registry.get_all_in_current_scope():
                        if var_info.linear_state is not None and var_info.linear_scope_depth == self.scope_depth:
                            if var_info.linear_state != 'consumed':
                                logger.error(
                                    f"Linear token '{var_info.name}' not consumed in loop iteration "
                                    f"(declared at line {var_info.line_number})", node
                                )
                finally:
                    # Pop loop context
                    self.loop_stack.pop()
                    # Check linear tokens before exiting scope
                    self._check_linear_tokens_consumed()
                    # Decrement scope depth and exit scope
                    self.scope_depth -= 1
                    self.ctx.var_registry.exit_scope()
                
                # Branch to next iteration
                if not self.builder.block.is_terminated:
                    self.builder.branch(continue_target)
                
                # Position at next block for next iteration (if not last)
                # Note: Even if current block is terminated, we still need to process
                # remaining iterations, as the terminator might be in a conditional branch
                if not is_last:
                    self.builder.position_at_end(continue_target)
            
            # Position at exit
            self.builder.position_at_end(loop_exit)
            
            # Handle else clause if present
            if node.orelse:
                # Check break flag
                broke = self.builder.load(break_flag, "const_broke")
                self.builder.cbranch(broke, after_else, else_block)
                
                # Else block
                self.builder.position_at_end(else_block)
                for stmt in node.orelse:
                    if not self.builder.block.is_terminated:
                        self.visit(stmt)
                
                if not self.builder.block.is_terminated:
                    self.builder.branch(after_else)
                
                # Continue after else
                self.builder.position_at_end(after_else)
        finally:
            if break_flag is not None:
                self._current_break_flag = old_break_flag
