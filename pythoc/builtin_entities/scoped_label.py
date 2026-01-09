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
    """Find label for goto. Checks ancestors and siblings/uncles.
    
    goto can target:
    - Self (inside the label)
    - Ancestors (containing labels)
    - Siblings (labels at same parent_depth)
    - Uncles (ancestor's siblings)
    
    Returns:
        LabelContext if found and visible, None otherwise
    """
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
    # Only check ancestor chain (including self)
    for ctx in visitor._scoped_label_stack:
        if ctx.name == label_name:
            return ctx
    
    return None


def _is_ancestor_label(visitor, ctx: LabelContext) -> bool:
    """Check if a label is in the current ancestor chain"""
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


# REMOVED: _simulate_defer_linear_consumption - old snapshot system
# Now we emit LinearTransition events directly to CFG


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
    current_scope = visitor.scope_depth
    target_parent = target_ctx.parent_scope_depth
    
    logger.debug(f"_emit_defers_for_scoped_goto: current={current_scope}, "
                f"target_parent={target_parent}, "
                f"is_goto_end={is_goto_end}, is_ancestor={is_ancestor}")
    
    # Use ScopeManager to emit defers for scopes being exited
    if hasattr(visitor, 'scope_manager'):
        visitor.scope_manager.exit_scopes_to(target_parent, visitor._get_cf_builder())
    else:
        logger.error("_emit_defers_for_scoped_goto: visitor has no scope_manager", node=None)


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
        """Resolve pending forward goto references to this label.
        
        Note: Some defers may have already been emitted by scope_manager.exit_scope()
        as scopes exited. We only emit defers that haven't been emitted yet
        (those with depth <= pending.emitted_to_depth and depth > target_parent).
        """
        from .defer import _execute_single_defer
        
        cf = visitor._get_cf_builder()
        label_name = ctx.name
        target_parent = ctx.parent_scope_depth
        
        resolved = []
        
        for pending in visitor._pending_scoped_gotos:
            if pending.label_name != label_name:
                continue
            
            if pending.is_goto_end:
                # goto_end forward reference - should not happen for sibling jumps
                # (goto_end can only target ancestors, which are always backward refs)
                logger.error(f"Internal error: forward goto_end to '{label_name}'",
                            node=pending.node, exc_type=RuntimeError)
                continue
            
            # Save current position
            saved_block = visitor.builder.block
            saved_cfg_block_id = cf._current_block_id
            
            # Get CFG block ID first (needed for logging)
            pending_block_id = cf._get_cfg_block_id(pending.block)
            
            # Position at pending block for both IR and CFG
            # This is CRITICAL: we need to update CFG's _current_block_id so that
            # linear events from defer execution are added to the correct CFG block
            cf.position_at_end(pending.block)
            
            # Temporarily clear terminated flag so defer can emit IR
            was_terminated = cf._terminated.get(pending_block_id, False)
            cf._terminated[pending_block_id] = False
            
            # Execute defers that haven't been emitted yet:
            # - depth <= emitted_to_depth (already emitted by scope exits)
            # - depth > target_parent (need to exit these scopes)
            # So we emit defers where: target_parent < depth <= emitted_to_depth
            # Wait, that's wrong. emitted_to_depth tracks what we've already emitted.
            # We need to emit defers where: target_parent < depth < emitted_to_depth
            # Because scope_manager emits defers for depth == current_scope.depth,
            # and updates emitted_to_depth to current_scope.depth.
            defers_to_execute = [
                (callable_obj, func_ref, args, defer_node)
                for depth, callable_obj, func_ref, args, defer_node in pending.defer_snapshot
                if target_parent < depth < pending.emitted_to_depth
            ]
            logger.debug(f"Resolving forward goto '{pending.label_name}': executing {len(defers_to_execute)} defers "
                        f"(goto_scope={pending.goto_scope_depth}, target_parent={target_parent}, "
                        f"emitted_to_depth={pending.emitted_to_depth})")
            
            # Emit defers
            for callable_obj, func_ref, args, defer_node in defers_to_execute:
                _execute_single_defer(visitor, callable_obj, func_ref, args, defer_node)
            
            # Generate the branch instruction
            visitor.builder.branch(ctx.begin_block)
            
            # Restore terminated flag
            cf._terminated[pending_block_id] = was_terminated
            
            # Restore position for both IR and CFG
            cf._current_block_id = saved_cfg_block_id
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
        
        Emits defers and branches to end block.
        
        IMPORTANT: Does NOT call position_at_end here - that must be done AFTER
        scope_manager.exit_scope() so that is_terminated() check works correctly.
        """
        cf = visitor._get_cf_builder()
        
        # If not terminated, emit defers and branch to end block
        # This must happen BEFORE scope_manager.exit_scope() which also checks is_terminated()
        if not cf.is_terminated():
            # Emit defers for this scope
            if hasattr(visitor, 'scope_manager'):
                for scope in visitor.scope_manager._scopes:
                    if scope.depth == visitor.scope_depth:
                        visitor.scope_manager._emit_defers_for_scope(scope)
                        break
            # Branch to end block (this terminates the block)
            cf.branch(ctx.end_block)
        
        # Pop from label stack (but keep in _all_labels and _scope_labels for sibling access)
        if visitor._scoped_label_stack and visitor._scoped_label_stack[-1].name == ctx.name:
            visitor._scoped_label_stack.pop()
        
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
            
            # Capture defer snapshot for later execution via ScopeManager
            defer_snapshot = []
            if not hasattr(visitor, 'scope_manager') or not visitor.scope_manager._scopes:
                logger.error("Forward goto: visitor has no scope_manager", node=node)
            else:
                # Capture from ScopeManager
                for scope in visitor.scope_manager._scopes:
                    if scope.depth <= goto_scope_depth:
                        for defer_info in scope.defers:
                            defer_snapshot.append((
                                scope.depth,
                                defer_info.callable_obj,
                                defer_info.func_ref,
                                defer_info.args,
                                defer_info.node
                            ))
            
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
            visitor._pending_scoped_gotos.append(pending)
            
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
    # Check for unresolved forward gotos
    if visitor._pending_scoped_gotos:
        for pending in visitor._pending_scoped_gotos:
            goto_type = "goto_end" if pending.is_goto_end else "goto"
            logger.error(f"Undefined label '{pending.label_name}' in {goto_type} statement",
                        node=pending.node, exc_type=SyntaxError)
