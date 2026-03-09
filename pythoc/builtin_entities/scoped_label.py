"""
Scoped goto/label intrinsics for structured control flow

with label("name"):    - Define a scoped label
    goto("name")       - Jump to beginning of label scope
    goto_end("name")   - Jump to end of label scope

Key properties:
1. Labels define scopes (via with statement)
2. Visibility rules:
   - goto: Can target self, ancestors, siblings, uncles
   - goto_end: Can ONLY target self and ancestors (must be inside target)
3. Defer execution follows parent_scope_depth model:
   - Both goto and goto_end exit to target's parent depth
   - Execute defers for all scopes being exited
"""
import ast
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from .base import BuiltinFunction
from .types import void
from ..valueref import wrap_value
from ..logger import logger


@dataclass
class LabelContext:
    """Context for a scoped label

    Key insight from design doc:
    - begin is at the 'with' statement level (visible to siblings)
    - end is inside the body (only visible from inside)

    Attributes:
        name: Label name (unique within function)
        parent_scope_depth: Scope depth at the 'with' statement level
        begin_block: IR block for goto target
        end_block: IR block for goto_end target
        node: Original AST node for error reporting
    """
    name: str
    parent_scope_depth: int    # Outside scope (at 'with' level)
    begin_block: object        # IR block
    end_block: object          # IR block
    node: ast.AST              # Original AST node for error reporting

    @property
    def scope_depth(self) -> int:
        """Inside scope depth (after entering) — always parent + 1"""
        return self.parent_scope_depth + 1


@dataclass
class PendingGoto:
    """A forward goto reference waiting to be resolved.
    
    Attributes:
        block: The IR block where the goto was issued
        label_name: Target label name
        goto_scope_depth: Scope depth when goto was issued
        defer_snapshot: List of (depth, callable_obj, func_ref, args, node) tuples
        node: AST node for error reporting
        is_goto_end: True if this is goto_end, False if goto
        emitted_to_depth: Defers with depth > this have been emitted.
                          Initialized to goto_scope_depth, decremented as scopes exit.
    """
    block: object
    label_name: str
    goto_scope_depth: int
    defer_snapshot: List  # List of (depth, callable_obj, func_ref, args, node)
    node: ast.AST
    is_goto_end: bool
    emitted_to_depth: int = field(default=None)
    
    def __post_init__(self):
        if self.emitted_to_depth is None:
            # Initially, no defers have been emitted
            # We'll emit defers for depth > emitted_to_depth
            # Start with goto_scope_depth + 1 so nothing is emitted yet
            self.emitted_to_depth = self.goto_scope_depth + 1


def _find_label_for_begin(visitor, label_name: str) -> Optional[LabelContext]:
    """Find label for goto. Delegates to scope_manager."""
    return visitor.scope_manager.find_label(label_name)


def _find_label_for_end(visitor, label_name: str) -> Optional[LabelContext]:
    """Find label for goto_end. Delegates to scope_manager."""
    return visitor.scope_manager.find_label_for_end(label_name)


def _is_ancestor_label(visitor, ctx: LabelContext) -> bool:
    """Check if a label is in the current ancestor chain. Delegates to scope_manager."""
    return visitor.scope_manager.is_ancestor_label(ctx)


def _check_sibling_crossing(visitor, target_ctx: LabelContext, node: ast.AST):
    """Check that sibling goto doesn't cross variable definitions or defer statements.
    
    For sibling/uncle goto (jumping to labels not in ancestor chain),
    we need to ensure we don't skip over:
    1. Variable definitions (would leave vars uninitialized)
    2. Defer statements (would skip defer registration)
    
    This is a compile-time check based on AST position.
    
    Note: This is a simplified check. A full implementation would need to
    track statement positions in the AST and compare them.
    """
    # For now, we allow sibling jumps without crossing check
    # A full implementation would:
    # 1. Record position of each label and goto in the function
    # 2. Check if there are var defs or defers between goto and target
    # 3. Error if crossing would skip initialization
    #
    # Since this requires significant AST analysis infrastructure,
    # we defer this to a future enhancement.
    pass


class label(BuiltinFunction):
    """label("name") - Scoped label context manager
    
    Used with 'with' statement to define a scoped label:
    
        with label("loop"):
            # code here can use goto("loop") or goto_end("loop")
            pass
    
    The label creates two IR blocks:
    - begin_block: target for goto (at 'with' level)
    - end_block: target for goto_end (inside body)
    
    Position model:
    - X.begin is at the 'with' statement level (visible to siblings)
    - X.end is inside X's body (only visible from inside X)
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'label'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle label("name") call - returns context manager info
        
        This is called when evaluating the context_expr in 'with label("name"):'
        The actual scope setup happens in visit_With.
        """
        if len(args) != 1:
            logger.error("label() takes exactly 1 argument (label name)",
                        node=node, exc_type=TypeError)
        
        label_arg = args[0]
        
        # Label name must be a compile-time string constant
        if not label_arg.is_python_value():
            logger.error("label() argument must be a string literal",
                        node=node, exc_type=TypeError)
        
        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"label() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)
        
        # Return label name wrapped - actual setup happens in visit_With
        return wrap_value(('__scoped_label__', label_name), kind='python', type_hint=void)
    
    @classmethod
    def enter_label_scope(cls, visitor, label_name: str, node: ast.With):
        """Called by visit_With to set up the label scope
        
        Creates begin/end blocks and registers label in tracking structures.
        Also resolves any pending forward goto references to this label.
        
        The parent_scope_depth is the current scope depth (at 'with' level).
        The scope_depth will be parent_scope_depth + 1 (inside the body).
        """
        cf = visitor._get_cf_builder()

        # Check for duplicate label name in function
        if label_name in visitor.scope_manager._all_labels:
            logger.error(f"Label '{label_name}' already defined in this function",
                        node=node, exc_type=SyntaxError)

        # Create begin and end blocks
        begin_block = cf.create_block(f"label_{label_name}_begin")
        end_block = cf.create_block(f"label_{label_name}_end")

        # Create label context
        # parent_scope_depth is current depth (at 'with' level)
        # scope_depth is derived as parent + 1 (inside body)
        parent_depth = visitor.scope_manager.current_depth
        ctx = LabelContext(
            name=label_name,
            parent_scope_depth=parent_depth,  # At 'with' level
            begin_block=begin_block,
            end_block=end_block,
            node=node
        )

        # Register in all tracking structures via scope_manager
        visitor.scope_manager.register_label(ctx)
        
        # Branch to begin block
        if not cf.is_terminated():
            cf.branch(begin_block)
        cf.position_at_end(begin_block)
        
        # Resolve any pending forward gotos to this label
        cls._resolve_pending_gotos(visitor, ctx)
        
        logger.debug(f"Entered label scope '{label_name}' at depth {ctx.scope_depth} "
                    f"(parent_depth={parent_depth})")
        
        return ctx
    
    @classmethod
    def _resolve_pending_gotos(cls, visitor, ctx: LabelContext):
        """Resolve pending forward goto references to this label.

        Delegates to scope_manager which owns the pending goto state.
        """
        visitor.scope_manager.resolve_pending_gotos_for_label(ctx)
        return ctx
    
    @classmethod
    def exit_label_scope(cls, visitor, ctx: LabelContext):
        """Called by visit_With to clean up the label scope

        Emits defers and branches to end block.

        IMPORTANT: Does NOT call position_at_end here - that must be done AFTER
        scope_manager.exit_scope() so that is_terminated() check works correctly.
        """
        cf = visitor._get_cf_builder()

        # If not terminated, emit defers and branch to end block
        # This must happen BEFORE scope_manager.exit_scope() which also checks is_terminated()
        if not cf.is_terminated():
            # Emit defers for this scope
            scope = visitor.scope_manager.current_scope
            if scope is not None:
                visitor.scope_manager._emit_defers_for_scope(scope)
            # Branch to end block (this terminates the block)
            cf.branch(ctx.end_block)

        # NOTE: position_at_end is done by caller AFTER scope_manager exits
        logger.debug(f"Exited label scope '{ctx.name}'")


class goto(BuiltinFunction):
    """goto("name") - Jump to beginning of label scope
    
    Can target:
    - Self (inside the label) - for loops
    - Ancestors (containing labels) - for nested loop break/continue
    - Siblings (labels at same level) - for state machines
    - Uncles (ancestor's siblings) - for multi-branch control flow
    
    Defer behavior:
    - Exit to target's parent_scope_depth
    - Execute defers for all scopes being exited
    - Then re-enter the target label
    
    Example - loop:
        with label("loop"):
            if done:
                goto_end("loop")
            # ... loop body ...
            goto("loop")  # Continue loop
    
    Example - state machine:
        with label("state_A"):
            process_A()
            goto("state_B")  # Jump to sibling
        with label("state_B"):
            process_B()
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'goto'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle goto("name") call
        
        Supports both backward references (label already defined) and
        forward references (label not yet defined, will be resolved later).
        """
        if len(args) != 1:
            logger.error("goto() takes exactly 1 argument (label name)",
                        node=node, exc_type=TypeError)
        
        label_arg = args[0]
        
        # Label name must be a compile-time string constant
        if not label_arg.is_python_value():
            logger.error("goto() argument must be a string literal",
                        node=node, exc_type=TypeError)
        
        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"goto() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)
        
        cf = visitor._get_cf_builder()
        
        # Check if block is already terminated
        if cf.is_terminated():
            return wrap_value(None, kind='python', type_hint=void)
        
        # Find label (can be ancestor, sibling, or uncle)
        ctx = _find_label_for_begin(visitor, label_name)
        
        if ctx is not None:
            # Backward reference: label already defined
            # Check if this is a sibling/uncle jump (not in ancestor chain)
            is_ancestor = _is_ancestor_label(visitor, ctx)
            if not is_ancestor:
                # Sibling/uncle: check crossing constraints
                _check_sibling_crossing(visitor, ctx, node)
            
            # Emit defers for scopes being exited (same as break/continue)
            visitor.scope_manager.exit_scopes_to(
                ctx.parent_scope_depth, visitor._get_cf_builder())
            
            # Branch to begin block
            cf.branch(ctx.begin_block)
            
            # Update CFG edge kind
            begin_block_id = cf._get_cfg_block_id(ctx.begin_block)
            for edge in reversed(cf.cfg.edges):
                if edge.target_id == begin_block_id:
                    edge.kind = 'goto'
                    break
        else:
            # Forward reference: label not yet defined
            # Capture current state for later resolution
            current_block = visitor.builder.block
            goto_scope_depth = visitor.scope_manager.current_depth
            
            # Capture defer snapshot for later execution via ScopeManager
            defer_snapshot = visitor.scope_manager.capture_defer_snapshot()
            
            logger.debug(f"Forward goto to '{label_name}': captured {len(defer_snapshot)} defers at scope {goto_scope_depth}")
            
            # Record pending goto for later resolution (to generate IR branch)
            pending = PendingGoto(
                block=current_block,
                label_name=label_name,
                goto_scope_depth=goto_scope_depth,
                defer_snapshot=defer_snapshot,
                node=node,
                is_goto_end=False
            )
            visitor.scope_manager.add_pending_goto(pending)
            
            # Mark current block as terminated in CFG (forward goto exits the block)
            current_block_id = cf._get_cfg_block_id(current_block)
            cf._terminated[current_block_id] = True
        
        return wrap_value(None, kind='python', type_hint=void)


# Backward compatibility alias
goto_begin = goto


class goto_end(BuiltinFunction):
    """goto_end("name") - Jump to end of label scope
    
    Can ONLY target:
    - Self (inside the label)
    - Ancestors (containing labels)
    
    Cannot target siblings or uncles because X.end is inside X's body,
    so only X's interior can see it.
    
    Defer behavior:
    - Exit to target's parent_scope_depth
    - Execute defers for all scopes being exited (including target)
    
    Example - early exit:
        with label("main"):
            defer(cleanup)
            if error:
                goto_end("main")  # cleanup() will be called
            # ... normal path ...
    
    Example - nested break:
        with label("outer"):
            with label("inner"):
                if done:
                    goto_end("outer")  # Exit both, run both defers
    """
    
    @classmethod
    def get_name(cls) -> str:
        return 'goto_end'
    
    @classmethod
    def handle_call(cls, visitor, func_ref, args, node: ast.Call):
        """Handle goto_end("name") call"""
        if len(args) != 1:
            logger.error("goto_end() takes exactly 1 argument (label name)",
                        node=node, exc_type=TypeError)
        
        label_arg = args[0]
        
        # Label name must be a compile-time string constant
        if not label_arg.is_python_value():
            logger.error("goto_end() argument must be a string literal",
                        node=node, exc_type=TypeError)
        
        label_name = label_arg.get_python_value()
        if not isinstance(label_name, str):
            logger.error(f"goto_end() argument must be a string, got {type(label_name).__name__}",
                        node=node, exc_type=TypeError)
        
        cf = visitor._get_cf_builder()
        
        # Check if block is already terminated
        if cf.is_terminated():
            return wrap_value(None, kind='python', type_hint=void)
        
        # Find label (must be in ancestor chain)
        ctx = _find_label_for_end(visitor, label_name)
        if ctx is None:
            logger.error(f"goto_end: label '{label_name}' not visible. "
                        f"goto_end can only target self or ancestors (must be inside the label).",
                        node=node, exc_type=SyntaxError)
        
        # Emit defers for scopes being exited (same as break/continue)
        visitor.scope_manager.exit_scopes_to(
            ctx.parent_scope_depth, visitor._get_cf_builder())
        
        # Branch to end block
        cf.branch(ctx.end_block)
        
        # Update CFG edge kind
        end_block_id = cf._get_cfg_block_id(ctx.end_block)
        for edge in reversed(cf.cfg.edges):
            if edge.target_id == end_block_id:
                edge.kind = 'goto_end'
                break
        
        return wrap_value(None, kind='python', type_hint=void)


def reset_label_tracking(visitor):
    """Reset label tracking at the start of each function.

    Delegates to scope_manager as the single source of truth.
    """
    visitor.scope_manager.reset_label_tracking()


def check_scoped_goto_consistency(visitor, func_node):
    """Check that all scoped gotos have been resolved.

    Delegates to scope_manager as the single source of truth.
    """
    visitor.scope_manager.check_goto_consistency(func_node)
