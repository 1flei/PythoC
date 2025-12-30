"""
goto/label intrinsics for low-level control flow

__label("name")  - Define a label for the next statement
__goto("name")   - Jump to a label

These are designed for:
1. C code migration
2. AST transformation code generation (avoiding while True patterns)

Usage:
    __label("loop_start")
    x = x + 1
    if x < 10:
        __goto("loop_start")
    __label("done")

Note: Labels are function-scoped. Forward references are supported.
Compile-time checks:
- goto to undefined label -> error
- label with no corresponding goto -> error
"""
import ast
from .base import BuiltinFunction
from .types import void
from ..valueref import wrap_value
from ..logger import logger


def _init_goto_registry(visitor):
    """Initialize goto/label registry if not exists"""
    if not hasattr(visitor, '_label_registry'):
        visitor._label_registry = {}  # label_name -> (block, scope_depth, node)
        # pending_gotos stores: (block, label_name, goto_scope_depth, defer_snapshot, node)
        # defer_snapshot is a copy of defers that need to be executed when label is resolved
        visitor._pending_gotos = []
        visitor._used_labels = set()  # labels that have been referenced by goto


def _capture_defers_for_goto(visitor, goto_scope_depth: int, label_scope_depth: int):
    """Capture defers that need to be executed for a goto
    
    For hierarchical gotos, captures defers from goto_scope down to label_scope + 1.
    Returns a list of defer items that should be executed.
    
    Args:
        visitor: The AST visitor
        goto_scope_depth: Scope depth where goto is located
        label_scope_depth: Scope depth where label is located
        
    Returns:
        List of (callable_obj, func_ref, args, node) tuples
    """
    from .defer import _get_defers_for_scope, _init_defer_registry
    
    _init_defer_registry(visitor)
    defers_to_execute = []
    
    logger.debug(f"_capture_defers_for_goto: goto_scope={goto_scope_depth}, label_scope={label_scope_depth}")
    
    if goto_scope_depth > label_scope_depth:
        # Hierarchical goto: capture defers from goto_scope down to label_scope + 1
        for depth in range(goto_scope_depth, label_scope_depth, -1):
            scope_defers = _get_defers_for_scope(visitor, depth)
            logger.debug(f"  Capturing {len(scope_defers)} defers for scope {depth}")
            defers_to_execute.extend(scope_defers)
    # else: same scope goto exits no scopes, no defers to emit
    
    return defers_to_execute


def _emit_captured_defers(visitor, defers_to_execute):
    """Emit IR for captured defers
    
    Args:
        visitor: The AST visitor
        defers_to_execute: List of (callable_obj, func_ref, args, node) tuples
    """
    from .defer import _execute_single_defer
    
    for callable_obj, func_ref, args, node in defers_to_execute:
        _execute_single_defer(visitor, callable_obj, func_ref, args, node)


def _emit_defers_for_goto(visitor, goto_scope_depth: int, label_scope_depth: int):
    """Emit deferred calls for goto from goto_scope to label_scope (for backward goto)
    
    For hierarchical gotos (yield/inline/closure generated), executes defers
    from goto_scope down to label_scope + 1 (inclusive).
    
    IMPORTANT: goto only pops defers for its own scope (goto_scope_depth),
    not for outer scopes. This is because:
    - goto marks the end of the current block, so current block's defers are consumed
    - outer scope defers may still be needed on other control flow paths
    
    Example:
        defer1        # scope 0
        if cond:
          defer2      # scope 1  
          goto label  # emits defer2, defer1; but only pops defer2
        stmt          # defer1 still needed here if cond is false
    
    Args:
        visitor: The AST visitor
        goto_scope_depth: Scope depth where goto is located
        label_scope_depth: Scope depth where label is located
    """
    defers = _capture_defers_for_goto(visitor, goto_scope_depth, label_scope_depth)
    _emit_captured_defers(visitor, defers)
    
    # Only pop defers for the goto's own scope (current block ending)
    # Outer scope defers remain for other control flow paths
    from .defer import unregister_defers_for_scope
    unregister_defers_for_scope(visitor, goto_scope_depth)


class __label(BuiltinFunction):
    """__label("name") - Define a label for the next statement
    
    Creates a new basic block with the given name. The next statement
    will be placed in this block.
    
    Labels are function-scoped and can be referenced before definition
    (forward references).
    """
    
    @classmethod
    def get_name(cls) -> str:
        return '__label'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle __label("name") call
        
        This creates a new basic block and registers it in the label registry.
        The current block branches to the new block (fall-through).
        """
        if len(args) != 1:
            logger.error("__label() takes exactly 1 argument (label name)", 
                        node=node, exc_type=TypeError)
        
        label_arg = args[0]
        
        # Label name must be a compile-time string constant
        if not label_arg.is_python_value():
            logger.error("__label() argument must be a string literal",
                        node=node, exc_type=TypeError)
        
        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"__label() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)
        
        # Get the control flow builder
        cf = visitor._get_cf_builder()
        
        # Initialize label registry if not exists
        _init_goto_registry(visitor)
        
        # Check for duplicate label
        if label_name in visitor._label_registry:
            logger.error(f"Label '{label_name}' already defined in this function",
                        node=node, exc_type=SyntaxError)
        
        # Create the label block
        label_block = cf.create_block(f"label_{label_name}")
        
        # Register the label with its scope depth and definition node
        label_scope_depth = visitor.scope_depth
        visitor._label_registry[label_name] = (label_block, label_scope_depth, node)
        logger.debug(f"Registered label '{label_name}' at scope {label_scope_depth}")
        
        # Branch from current block to label block (fall-through)
        if not cf.is_terminated():
            cf.branch(label_block)
        
        # Position at the label block
        cf.position_at_end(label_block)
        
        # Resolve any pending gotos to this label
        label_scope_depth = visitor.scope_depth
        resolved = []
        for pending_block, pending_name, goto_scope_depth, defer_snapshot, pending_node in visitor._pending_gotos:
            if pending_name == label_name:
                # Save current position
                saved_block = visitor.builder.block
                visitor.builder.position_at_end(pending_block)
                
                # Temporarily clear terminated flag so defer can emit IR
                pending_block_id = cf._get_cfg_block_id(pending_block)
                was_terminated = cf._terminated.get(pending_block_id, False)
                cf._terminated[pending_block_id] = False
                
                # Filter defer_snapshot: execute defers for scopes > label_scope_depth
                # Same-scope goto exits no scopes, so no defers to emit
                # defer_snapshot contains (depth, callable_obj, func_ref, args, node)
                if goto_scope_depth > label_scope_depth:
                    # Hierarchical goto: execute defers for scopes > label_scope_depth
                    defers_to_execute = [
                        (callable_obj, func_ref, args, defer_node)
                        for depth, callable_obj, func_ref, args, defer_node in defer_snapshot
                        if depth > label_scope_depth
                    ]
                else:
                    # Same scope goto: no scopes exited, no defers
                    defers_to_execute = []
                logger.debug(f"Resolving forward goto '{pending_name}': executing {len(defers_to_execute)} defers "
                           f"(goto_scope={goto_scope_depth}, label_scope={label_scope_depth})")
                
                # Emit filtered defers
                _emit_captured_defers(visitor, defers_to_execute)
                
                # Generate the branch instruction
                visitor.builder.branch(label_block)
                
                # Restore terminated flag
                cf._terminated[pending_block_id] = was_terminated
                
                # Restore position
                visitor.builder.position_at_end(saved_block)
                
                # Add CFG edge for the goto
                label_block_id = cf._get_cfg_block_id(label_block)
                cf.cfg.add_edge(pending_block_id, label_block_id, kind='goto')
                
                resolved.append((pending_block, pending_name, goto_scope_depth, defer_snapshot, pending_node))
        
        # Remove resolved gotos
        for item in resolved:
            visitor._pending_gotos.remove(item)
        
        # Return void (this is a statement, not an expression)
        return wrap_value(None, kind='python', type_hint=void)


class __goto(BuiltinFunction):
    """__goto("name") - Jump to a label
    
    Generates an unconditional branch to the named label.
    Forward references are supported - if the label doesn't exist yet,
    the goto is recorded and resolved when the label is defined.
    
    Deferred calls for the current scope are executed before the jump.
    """
    
    @classmethod
    def get_name(cls) -> str:
        return '__goto'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle __goto("name") call
        
        This generates a branch to the label block. If the label doesn't
        exist yet (forward reference), the goto is recorded for later resolution.
        
        Deferred calls for the current scope are executed before the jump.
        """
        if len(args) != 1:
            logger.error("__goto() takes exactly 1 argument (label name)",
                        node=node, exc_type=TypeError)
        
        label_arg = args[0]
        
        # Label name must be a compile-time string constant
        if not label_arg.is_python_value():
            logger.error("__goto() argument must be a string literal",
                        node=node, exc_type=TypeError)
        
        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"__goto() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)
        
        # Get the control flow builder
        cf = visitor._get_cf_builder()
        
        # Initialize label registry if not exists
        _init_goto_registry(visitor)
        
        # Mark this label as used (referenced by goto)
        visitor._used_labels.add(label_name)
        
        # Check if block is already terminated
        if cf.is_terminated():
            # Unreachable code, silently ignore
            return wrap_value(None, kind='python', type_hint=void)
        
        # Check if label exists (backward goto)
        if label_name in visitor._label_registry:
            label_block, label_scope_depth, _ = visitor._label_registry[label_name]
            
            # Execute deferred calls for scopes between goto and label
            _emit_defers_for_goto(visitor, visitor.scope_depth, label_scope_depth)
            
            cf.branch(label_block)
            # Update edge kind to 'goto'
            for edge in reversed(cf.cfg.edges):
                if edge.target_id == cf._get_cfg_block_id(label_block):
                    edge.kind = 'goto'
                    break
        else:
            # Forward reference - capture defer snapshot for later execution
            # We don't know label's scope yet, so capture all defers from current scope down to 0
            # When label is resolved, we'll filter based on actual label scope
            current_block = visitor.builder.block
            goto_scope_depth = visitor.scope_depth
            
            # Capture all defers that might need to be executed (from goto_scope down to 0)
            # This is a superset; actual execution will be filtered at label resolution time
            from .defer import _init_defer_registry
            _init_defer_registry(visitor)
            
            # Copy the defer entries for scopes >= 0 (we'll filter at label time)
            defer_snapshot = []
            for item in visitor._defer_stack:
                depth, callable_obj, func_ref, args, defer_node = item
                if depth <= goto_scope_depth:
                    defer_snapshot.append((depth, callable_obj, func_ref, args, defer_node))
            
            logger.debug(f"Forward goto to '{label_name}': captured {len(defer_snapshot)} defers at scope {goto_scope_depth}")
            
            visitor._pending_gotos.append((current_block, label_name, goto_scope_depth, defer_snapshot, node))
            
            # Record exit snapshot for current block before terminating it
            current_block_id = cf._get_cfg_block_id(current_block)
            cf._exit_snapshots[current_block_id] = cf.capture_linear_snapshot()
            
            # Create a new unreachable block for subsequent code
            # This is needed because goto terminates the current block
            unreachable_block = cf.create_block(f"after_goto_{label_name}")
            cf.position_at_end(unreachable_block)
            
            # Mark current block as terminated in CFG
            # The actual branch will be added when label is defined
            cf._terminated[current_block_id] = True
        
        # Return void (this is a statement, not an expression)
        return wrap_value(None, kind='python', type_hint=void)


def check_goto_label_consistency(visitor, func_node):
    """Check that all gotos have matching labels and vice versa
    
    Called at the end of function compilation.
    
    Args:
        visitor: The visitor instance
        func_node: The function AST node (for error location)
        
    Raises compile errors for:
    - goto to undefined label (always an error)
    
    Raises warnings for:
    - label with no corresponding goto (may be intentional for fall-through)
    """
    if not hasattr(visitor, '_label_registry'):
        return  # No goto/label used in this function
    
    # Check for unresolved gotos (goto to undefined label) - this is always an error
    if visitor._pending_gotos:
        for pending_block, pending_name, goto_scope_depth, defer_snapshot, pending_node in visitor._pending_gotos:
            logger.error(f"Undefined label '{pending_name}' in goto statement",
                        node=pending_node, exc_type=SyntaxError)
    
    # Check for unused labels (label with no corresponding goto)
    # This is a warning, not an error, because labels can be used for fall-through
    for label_name, (label_block, label_scope_depth, label_node) in visitor._label_registry.items():
        if label_name not in visitor._used_labels:
            logger.warning(f"Label '{label_name}' defined but never used by any goto",
                          node=label_node)
