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
"""
import ast
from .base import BuiltinFunction
from .types import void
from ..valueref import wrap_value
from ..logger import logger


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
        if not hasattr(visitor, '_label_registry'):
            visitor._label_registry = {}
            visitor._pending_gotos = []
        
        # Check for duplicate label
        if label_name in visitor._label_registry:
            logger.error(f"Label '{label_name}' already defined in this function",
                        node=node, exc_type=SyntaxError)
        
        # Create the label block
        label_block = cf.create_block(f"label_{label_name}")
        
        # Register the label
        visitor._label_registry[label_name] = label_block
        logger.debug(f"Registered label '{label_name}'")
        
        # Branch from current block to label block (fall-through)
        if not cf.is_terminated():
            cf.branch(label_block)
        
        # Position at the label block
        cf.position_at_end(label_block)
        
        # Resolve any pending gotos to this label
        resolved = []
        for pending_block, pending_name, pending_node in visitor._pending_gotos:
            if pending_name == label_name:
                # Generate the branch instruction
                # Save current position
                saved_block = visitor.builder.block
                visitor.builder.position_at_end(pending_block)
                visitor.builder.branch(label_block)
                # Restore position
                visitor.builder.position_at_end(saved_block)
                
                # Add CFG edge for the goto
                pending_block_id = cf._get_cfg_block_id(pending_block)
                label_block_id = cf._get_cfg_block_id(label_block)
                cf.cfg.add_edge(pending_block_id, label_block_id, kind='goto')
                
                resolved.append((pending_block, pending_name, pending_node))
        
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
    """
    
    @classmethod
    def get_name(cls) -> str:
        return '__goto'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle __goto("name") call
        
        This generates a branch to the label block. If the label doesn't
        exist yet (forward reference), the goto is recorded for later resolution.
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
        if not hasattr(visitor, '_label_registry'):
            visitor._label_registry = {}
            visitor._pending_gotos = []
        
        # Check if block is already terminated
        if cf.is_terminated():
            # Unreachable code, silently ignore
            return wrap_value(None, kind='python', type_hint=void)
        
        # Check if label exists
        if label_name in visitor._label_registry:
            # Label exists, generate branch directly
            label_block = visitor._label_registry[label_name]
            cf.branch(label_block)
            # Update edge kind to 'goto'
            for edge in reversed(cf.cfg.edges):
                if edge.target_id == cf._get_cfg_block_id(label_block):
                    edge.kind = 'goto'
                    break
        else:
            # Forward reference - record for later resolution
            # Save current block for later branch generation
            current_block = visitor.builder.block
            visitor._pending_gotos.append((current_block, label_name, node))
            
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
