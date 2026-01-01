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
        scope_depth: Scope depth INSIDE the label body
        parent_scope_depth: Scope depth at the 'with' statement level
        begin_block: IR block for goto target
        end_block: IR block for goto_end target
        node: Original AST node for error reporting
    """
    name: str
    scope_depth: int           # Inside scope (after entering)
    parent_scope_depth: int    # Outside scope (at 'with' level)
    begin_block: object        # IR block
    end_block: object          # IR block
    node: ast.AST              # Original AST node for error reporting


def _init_label_tracking(visitor):
    """Initialize scoped label tracking structures
    
    Structures maintained:
    1. _scoped_label_stack: Current nesting chain (for ancestor lookup)
    2. _scope_labels: parent_depth -> list of labels (for sibling/uncle lookup)
    3. _all_labels: name -> LabelContext (for duplicate check and forward reference)
    4. _pending_scoped_gotos: List of forward goto references to resolve
    """
    if not hasattr(visitor, '_scoped_label_stack'):
        visitor._scoped_label_stack = []  # Stack of LabelContext (current nesting)
    if not hasattr(visitor, '_scope_labels'):
        visitor._scope_labels = {}  # parent_depth -> [LabelContext, ...]
    if not hasattr(visitor, '_all_labels'):
        visitor._all_labels = {}  # name -> LabelContext (for duplicate check)
    if not hasattr(visitor, '_pending_scoped_gotos'):
        # List of (current_block, label_name, goto_scope_depth, defer_snapshot, node, is_goto_end)
        visitor._pending_scoped_gotos = []


def _find_label_for_begin(visitor, label_name: str) -> Optional[LabelContext]:
    """Find label for goto. Checks ancestors and siblings/uncles.
    
    goto can target:
    - Self (inside the label)
    - Ancestors (containing labels)
    - Siblings (labels at same parent_depth)
    - Uncles (ancestor's siblings)
    
    Returns:
        LabelContext if found and visible, None otherwise
    """
    _init_label_tracking(visitor)
    
    # 1. Check ancestor chain (including self)
    for ctx in visitor._scoped_label_stack:
        if ctx.name == label_name:
            return ctx
    
    # 2. Check siblings and uncles
    # Siblings/uncles are labels whose parent_depth is in our ancestor chain
    # or at function level (depth 0)
    ancestor_depths = {0}  # Function level always included
    for ctx in visitor._scoped_label_stack:
        ancestor_depths.add(ctx.parent_scope_depth)
    
    for depth in ancestor_depths:
        if depth in visitor._scope_labels:
            for ctx in visitor._scope_labels[depth]:
                if ctx.name == label_name:
                    return ctx
    
    return None


def _find_label_for_end(visitor, label_name: str) -> Optional[LabelContext]:
    """Find label for goto_end. Only checks ancestors (must be inside target).
    
    goto_end can ONLY target:
    - Self (inside the label)
    - Ancestors (containing labels)
    
    This is because X.end is inside X's body, so only X's interior can see it.
    
    Returns:
        LabelContext if found in ancestor chain, None otherwise
    """
    _init_label_tracking(visitor)
    
    # Only check ancestor chain (including self)
    for ctx in visitor._scoped_label_stack:
        if ctx.name == label_name:
            return ctx
    
    return None


def _is_ancestor_label(visitor, ctx: LabelContext) -> bool:
    """Check if a label is in the current ancestor chain"""
    _init_label_tracking(visitor)
    return ctx in visitor._scoped_label_stack


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


def _emit_defers_for_scoped_goto(visitor, target_ctx: LabelContext, is_goto_end: bool, is_ancestor: bool):
    """Emit deferred calls for scoped goto.
    
    From design doc section 5.4 - Formal Rule:
    Both goto and goto_end exit to the target label's parent depth,
    executing all defers along the way.
    
    - goto("X"): exit to X's parent depth, jump to X.begin (re-enter X)
    - goto_end("X"): exit to X's parent depth, jump to X.end (skip rest of X)
    
    The key insight is that both operations conceptually exit to the same depth
    (the label's parent), they just jump to different positions afterward.
    
    IMPORTANT: This function only EMITS defer calls, it does NOT unregister them.
    Unregistering is handled by the normal scope exit path. This is because
    goto may be in a conditional branch, and the other branch still needs the
    defer registrations.
    
    Args:
        visitor: AST visitor
        target_ctx: The target label context
        is_goto_end: True if this is goto_end, False if goto
        is_ancestor: True if target is in ancestor chain (self or containing label)
    """
    from .defer import _get_defers_for_scope, _execute_single_defer
    
    current_scope = visitor.scope_depth
    target_parent = target_ctx.parent_scope_depth
    
    logger.debug(f"_emit_defers_for_scoped_goto: current={current_scope}, "
                f"target_parent={target_parent}, "
                f"is_goto_end={is_goto_end}, is_ancestor={is_ancestor}")
    
    # Simple rule from design doc: exit to target's parent depth
    # Execute defers for all scopes from current down to target_parent (exclusive)
    # i.e., scopes [current, current-1, ..., target_parent+1]
    for depth in range(current_scope, target_parent, -1):
        scope_defers = _get_defers_for_scope(visitor, depth)
        logger.debug(f"  Scope {depth}: {len(scope_defers)} defers")
        for callable_obj, func_ref, args, defer_node in scope_defers:
            _execute_single_defer(visitor, callable_obj, func_ref, args, defer_node)
        # NOTE: Do NOT unregister defers here!
        # The other branch (non-goto path) still needs them.


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
        _init_label_tracking(visitor)
        
        # Check for duplicate label name in function
        if label_name in visitor._all_labels:
            logger.error(f"Label '{label_name}' already defined in this function",
                        node=node, exc_type=SyntaxError)
        
        # Create begin and end blocks
        begin_block = cf.create_block(f"label_{label_name}_begin")
        end_block = cf.create_block(f"label_{label_name}_end")
        
        # Create label context
        # parent_scope_depth is current depth (at 'with' level)
        # scope_depth is parent + 1 (inside body, after visit_With increments)
        parent_depth = visitor.scope_depth
        ctx = LabelContext(
            name=label_name,
            scope_depth=parent_depth + 1,  # Inside scope (after increment)
            parent_scope_depth=parent_depth,  # At 'with' level
            begin_block=begin_block,
            end_block=end_block,
            node=node
        )
        
        # Register in all tracking structures
        visitor._scoped_label_stack.append(ctx)
        visitor._all_labels[label_name] = ctx
        
        # Register in scope_labels for sibling/uncle lookup
        if parent_depth not in visitor._scope_labels:
            visitor._scope_labels[parent_depth] = []
        visitor._scope_labels[parent_depth].append(ctx)
        
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
        """Resolve pending forward goto references to this label"""
        from .defer import _execute_single_defer
        
        cf = visitor._get_cf_builder()
        label_name = ctx.name
        target_parent = ctx.parent_scope_depth
        
        resolved = []
        for pending in visitor._pending_scoped_gotos:
            pending_block, pending_name, goto_scope_depth, defer_snapshot, pending_node, is_goto_end = pending
            
            if pending_name != label_name:
                continue
            
            if is_goto_end:
                # goto_end forward reference - should not happen for sibling jumps
                # (goto_end can only target ancestors, which are always backward refs)
                logger.error(f"Internal error: forward goto_end to '{label_name}'",
                            node=pending_node, exc_type=RuntimeError)
                continue
            
            # Save current position
            saved_block = visitor.builder.block
            visitor.builder.position_at_end(pending_block)
            
            # Temporarily clear terminated flag so defer can emit IR
            pending_block_id = cf._get_cfg_block_id(pending_block)
            was_terminated = cf._terminated.get(pending_block_id, False)
            cf._terminated[pending_block_id] = False
            
            # Execute defers for scopes from goto_scope down to target_parent (exclusive)
            # Filter defer_snapshot based on actual target
            defers_to_execute = [
                (callable_obj, func_ref, args, defer_node)
                for depth, callable_obj, func_ref, args, defer_node in defer_snapshot
                if depth > target_parent
            ]
            logger.debug(f"Resolving forward goto '{pending_name}': executing {len(defers_to_execute)} defers "
                        f"(goto_scope={goto_scope_depth}, target_parent={target_parent})")
            
            # Emit defers
            for callable_obj, func_ref, args, defer_node in defers_to_execute:
                _execute_single_defer(visitor, callable_obj, func_ref, args, defer_node)
            
            # Generate the branch instruction
            visitor.builder.branch(ctx.begin_block)
            
            # Restore terminated flag
            cf._terminated[pending_block_id] = was_terminated
            
            # Restore position
            visitor.builder.position_at_end(saved_block)
            
            # Add CFG edge for the goto
            begin_block_id = cf._get_cfg_block_id(ctx.begin_block)
            cf.cfg.add_edge(pending_block_id, begin_block_id, kind='goto')
            
            resolved.append(pending)
        
        # Remove resolved gotos
        for item in resolved:
            visitor._pending_scoped_gotos.remove(item)
        
        return ctx
    
    @classmethod
    def exit_label_scope(cls, visitor, ctx: LabelContext):
        """Called by visit_With to clean up the label scope
        
        Emits defers for this scope and branches to end block.
        """
        cf = visitor._get_cf_builder()
        from .defer import emit_deferred_calls, unregister_defers_for_scope
        
        # Execute defers for this scope if not already terminated
        if not cf.is_terminated():
            emit_deferred_calls(visitor, scope_depth=visitor.scope_depth)
            unregister_defers_for_scope(visitor, visitor.scope_depth)
            cf.branch(ctx.end_block)
        else:
            # Even if terminated (e.g., by goto), we need to unregister defers
            # to prevent them from being executed by sibling labels
            unregister_defers_for_scope(visitor, visitor.scope_depth)
        
        # Pop from label stack (but keep in _all_labels and _scope_labels for sibling access)
        _init_label_tracking(visitor)
        if visitor._scoped_label_stack and visitor._scoped_label_stack[-1].name == ctx.name:
            visitor._scoped_label_stack.pop()
        
        # Position at end block
        cf.position_at_end(ctx.end_block)
        
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
        _init_label_tracking(visitor)
        
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
            
            # Execute defers based on target relationship
            _emit_defers_for_scoped_goto(visitor, ctx, is_goto_end=False, is_ancestor=is_ancestor)
            
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
            goto_scope_depth = visitor.scope_depth
            
            # Capture defer snapshot for later execution
            from .defer import _init_defer_registry
            _init_defer_registry(visitor)
            defer_snapshot = []
            for item in visitor._defer_stack:
                depth, callable_obj, func_ref_defer, args_defer, defer_node = item
                if depth <= goto_scope_depth:
                    defer_snapshot.append((depth, callable_obj, func_ref_defer, args_defer, defer_node))
            
            logger.debug(f"Forward goto to '{label_name}': captured {len(defer_snapshot)} defers at scope {goto_scope_depth}")
            
            # Record pending goto for later resolution
            visitor._pending_scoped_gotos.append(
                (current_block, label_name, goto_scope_depth, defer_snapshot, node, False)  # False = goto (not goto_end)
            )
            
            # Record exit snapshot for current block
            current_block_id = cf._get_cfg_block_id(current_block)
            cf._exit_snapshots[current_block_id] = cf.capture_linear_snapshot()
            
            # Mark current block as terminated in CFG
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
        
        # Execute defers (goto_end always exits, so is_ancestor=True, is_goto_end=True)
        _emit_defers_for_scoped_goto(visitor, ctx, is_goto_end=True, is_ancestor=True)
        
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
    
    Called by the visitor when entering a new function.
    """
    visitor._scoped_label_stack = []
    visitor._scope_labels = {}
    visitor._all_labels = {}
    visitor._pending_scoped_gotos = []


def check_scoped_goto_consistency(visitor, func_node):
    """Check that all scoped gotos have been resolved.
    
    Called at the end of function compilation.
    
    Args:
        visitor: The visitor instance
        func_node: The function AST node (for error location)
    """
    _init_label_tracking(visitor)
    
    # Check for unresolved forward gotos
    if visitor._pending_scoped_gotos:
        for pending_block, pending_name, goto_scope_depth, defer_snapshot, pending_node, is_goto_end in visitor._pending_scoped_gotos:
            goto_type = "goto_end" if is_goto_end else "goto"
            logger.error(f"Undefined label '{pending_name}' in {goto_type} statement",
                        node=pending_node, exc_type=SyntaxError)
