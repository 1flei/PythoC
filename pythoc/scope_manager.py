"""
Unified Scope Manager for pythoc

This module provides unified scope management that combines:
- Variable registry (var_registry)
- Defer stack management
- Linear type tracking
- Loop context (break/continue targets)

The key insight is that defer execution time == variable lifetime end.
When a scope exits:
1. Deferred calls for that scope are executed
2. Linear tokens must have been consumed
3. Variables go out of scope

This replaces the separate scope_depth, _defer_stack, and var_registry.enter_scope()/exit_scope()
with a single unified mechanism.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import ast

from .logger import logger
from .valueref import ValueRef


class ScopeType(Enum):
    """Type of scope for debugging and special handling"""
    FUNCTION = auto()   # Function body
    LOOP = auto()       # Loop body (while, for)
    IF = auto()         # If/else block
    LABEL = auto()      # Scoped label block
    MATCH = auto()      # Match case block


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
        """Inside scope depth (after entering) - always parent + 1"""
        return self.parent_scope_depth + 1


@dataclass
class DeferInfo:
    """Information about a deferred call"""
    callable_obj: Any           # Object with handle_call method
    func_ref: ValueRef          # Function reference
    args: List[ValueRef]        # Arguments (captured at registration)
    node: ast.AST               # AST node for error reporting


@dataclass
class Scope:
    """A single scope in the scope stack
    
    Contains all information needed for scope management:
    - Variables declared in this scope
    - Deferred calls registered in this scope
    - Linear tokens created in this scope
    - Loop targets (for break/continue)
    """
    depth: int
    scope_type: ScopeType
    
    # Variables declared in this scope (name -> VariableInfo)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Deferred calls registered in this scope
    defers: List[DeferInfo] = field(default_factory=list)
    
    # For loops: targets for break/continue
    continue_target: Optional[Any] = None  # Block to jump to on continue
    break_target: Optional[Any] = None     # Block to jump to on break

    # For labels: LabelContext (has begin_block, end_block, name)
    label_ctx: Optional[Any] = None
    
    def __repr__(self):
        return f"Scope(depth={self.depth}, type={self.scope_type.name}, vars={len(self.variables)}, defers={len(self.defers)})"


class ScopeManager:
    """Unified scope manager for variables, defers, and loop context

    This replaces the separate mechanisms:
    - var_registry.enter_scope() / exit_scope()
    - scope_depth tracking
    - _defer_stack management
    - loop_stack / loop_scope_stack

    Usage:
        with scope_manager.scope(ScopeType.IF) as scope:
            # Variables, defers automatically managed
            ...
        # Defers executed, scope cleaned up
    """
    
    def __init__(self):
        """Initialize scope manager with its own VariableRegistry"""
        from .registry import VariableRegistry
        self._var_registry = VariableRegistry()
        self._scopes: List[Scope] = []
        self._visitor: Optional[Any] = None  # Set by visitor

        # Label tracking state (single source of truth)
        # These persist after scope exit for sibling/uncle goto visibility
        self._scope_labels: Dict[int, List[Any]] = {}  # parent_depth -> [LabelContext]
        self._all_labels: Dict[str, Any] = {}        # name -> LabelContext

        # Scope hazards: track var declarations and defer registrations with
        # line numbers, keyed by scope depth. Used to detect forward sibling
        # gotos that skip over variable definitions or defer statements.
        self._scope_hazards: Dict[int, List[Tuple[int, str]]] = {}  # depth -> [(lineno, desc)]
    
    def set_visitor(self, visitor: Any):
        """Set the visitor for defer execution"""
        self._visitor = visitor
    
    @property
    def current_depth(self) -> int:
        """Get current scope depth"""
        return len(self._scopes)
    
    @property
    def current_scope(self) -> Optional[Scope]:
        """Get current scope"""
        return self._scopes[-1] if self._scopes else None
    
    def enter_scope(self, scope_type: ScopeType,
                    continue_target: Any = None,
                    break_target: Any = None,
                    label_ctx: Any = None) -> Scope:
        """Enter a new scope

        Args:
            scope_type: Type of scope (FUNCTION, LOOP, IF, etc.)
            continue_target: For loops, the block to jump to on continue
            break_target: For loops, the block to jump to on break
            label_ctx: For labels, the LabelContext

        Returns:
            The new Scope object
        """
        # Sync with var_registry
        self._var_registry.enter_scope()

        scope = Scope(
            depth=len(self._scopes) + 1,
            scope_type=scope_type,
            continue_target=continue_target,
            break_target=break_target,
            label_ctx=label_ctx
        )
        self._scopes.append(scope)
        
        logger.debug(f"Entered scope {scope}")
        return scope
    
    def exit_scope(self, cf: Any, node: Optional[ast.AST] = None) -> Scope:
        """Exit current scope

        This:
        1. Emits deferred calls for pending forward gotos from this scope
           (MUST happen before var_registry.exit_scope() so vars are still accessible)
        2. Emits deferred calls for normal exit (if block not terminated)
        3. Removes variables from registry

        Args:
            cf: ControlFlowBuilder for checking termination
            node: Optional AST node for error reporting

        Returns:
            The exited Scope
        """
        if not self._scopes:
            raise RuntimeError("Cannot exit scope: no scope to exit")

        scope = self._scopes.pop()

        try:
            # 1. Emit defers for pending forward gotos from this scope (via cfbuilder)
            if self._visitor:
                from pythoc.builtin_entities.defer import _execute_single_defer
                def defer_executor(callable_obj, func_ref, args, defer_node):
                    _execute_single_defer(self._visitor, callable_obj, func_ref, args, defer_node)
                cf.emit_defers_for_pending_gotos(scope.depth, defer_executor)

            # 2. Emit deferred calls for normal exit (if not terminated)
            # Note: scope is already popped from _scopes, so we call
            # _emit_defers_for_scope directly on the popped scope object.
            if not cf.is_terminated():
                self._emit_defers_for_scope(scope)
        finally:
            # 4. Sync with var_registry (must happen even if checks raise)
            self._var_registry.exit_scope()

        logger.debug(f"Exited scope {scope}")
        return scope
    
    def emit_defers_to_depth(self, target_depth: int, cf):
        """Emit defers from current scope down to target_depth (exclusive).

        This is the single entry point for all control-flow defer emission:
        - return:        emit_defers_to_depth(0, cf)           — all scopes
        - break/continue: emit_defers_to_depth(loop_depth-1, cf) — down to loop
        - backward goto: emit_defers_to_depth(parent_depth, cf) — down to target
        - scope exit:    emit_defers_to_depth(scope.depth-1, cf) — just this scope

        Does NOT pop scopes — the control flow jump or finally blocks handle that.

        Args:
            target_depth: Scopes with depth > target_depth have their defers emitted.
                          0 means emit all defers (return semantics).
            cf: ControlFlowBuilder for termination checks.
        """
        if cf.is_terminated():
            return

        for scope in reversed(self._scopes):
            if scope.depth <= target_depth:
                break
            self._emit_defers_for_scope(scope)
    
    def _emit_defers_for_scope(self, scope: Scope):
        """Emit deferred calls for a single scope.

        Defers execute in LIFO order within a scope.

        The defers list is NOT cleared - multiple code paths may need to emit
        the same defers (e.g., goto path and normal exit path are mutually
        exclusive at runtime but both need IR generated at compile time).
        """
        if not self._visitor:
            return

        from pythoc.builtin_entities.defer import _execute_single_defer

        cf = self._visitor._get_cf_builder()

        # Do NOT clear defers - they may be needed by other code paths.
        # Within a scope, defer unwinds in LIFO order.
        defers_to_emit = list(reversed(scope.defers))

        executed_count = 0
        for defer_info in defers_to_emit:
            if cf.is_terminated():
                logger.debug(f"  cf is terminated, stopping defer emit")
                break

            logger.debug(f"  emitting defer: {defer_info.callable_obj}")

            # Use the unified defer execution function
            _execute_single_defer(
                self._visitor,
                defer_info.callable_obj,
                defer_info.func_ref,
                defer_info.args,
                defer_info.node
            )
            executed_count += 1

        logger.debug(f"Emitted {executed_count}/{len(defers_to_emit)} defers for scope depth {scope.depth}")
    
    def register_defer(self, callable_obj: Any, func_ref: ValueRef, 
                       args: List[ValueRef], node: ast.AST):
        """Register a deferred call in the current scope
        
        Args:
            callable_obj: Object with handle_call method
            func_ref: Function reference
            args: Arguments (captured now, used at execution)
            node: AST node for error reporting
        """
        if not self._scopes:
            raise RuntimeError("Cannot register defer: no scope")
        
        defer_info = DeferInfo(
            callable_obj=callable_obj,
            func_ref=func_ref,
            args=args,
            node=node
        )
        self._scopes[-1].defers.append(defer_info)

        # Record hazard for sibling crossing check
        depth = self.current_depth
        lineno = getattr(node, 'lineno', None)
        if lineno is not None:
            self._scope_hazards.setdefault(depth, []).append(
                (lineno, "defer statement")
            )

        logger.debug(f"Registered defer at scope depth {self.current_depth}")
    
    def get_loop_targets(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Get the nearest loop's continue and break targets
        
        Returns:
            (continue_target, break_target) or (None, None) if not in a loop
        """
        for scope in reversed(self._scopes):
            if scope.scope_type == ScopeType.LOOP:
                return (scope.continue_target, scope.break_target)
        return (None, None)
    
    def get_loop_scope_depth(self) -> Optional[int]:
        """Get the depth of the nearest enclosing loop scope"""
        for scope in reversed(self._scopes):
            if scope.scope_type == ScopeType.LOOP:
                return scope.depth
        return None
    
    def is_in_loop(self) -> bool:
        """Check if currently inside a loop"""
        return self.get_loop_scope_depth() is not None
    
    def declare_variable(self, var_info: Any, allow_shadow: bool = False):
        """Declare a variable in the current scope

        This syncs with var_registry and tracks in scope.
        """
        self._var_registry.declare(var_info, allow_shadow=allow_shadow)
        if self._scopes:
            self._scopes[-1].variables[var_info.name] = var_info
        # Record hazard for sibling crossing check
        depth = self.current_depth
        lineno = getattr(var_info, 'line_number', None)
        if lineno is not None:
            self._scope_hazards.setdefault(depth, []).append(
                (lineno, f"variable '{var_info.name}'")
            )
    
    def lookup_variable(self, name: str) -> Optional[Any]:
        """Look up a variable in the scope chain"""
        return self._var_registry.lookup(name)

    def is_declared_in_current_scope(self, name: str) -> bool:
        """Check if variable is declared in the current scope"""
        return self._var_registry.is_declared_in_current_scope(name)

    def get_all_in_current_scope(self) -> List[Any]:
        """Get all variables in the current scope"""
        return self._var_registry.get_all_in_current_scope()

    @property
    def scopes(self):
        """Access underlying scope stack (for debug logging)"""
        return self._var_registry.scopes

    def get_all_visible(self) -> Dict[str, Any]:
        """Get all currently visible variables (from all scopes)"""
        return self._var_registry.get_all_visible()

    def scope(self, scope_type: ScopeType, cf: Any = None,
              continue_target: Any = None, break_target: Any = None,
              label_ctx: Any = None,
              node: Optional[ast.AST] = None):
        """Context manager for automatic scope management

        Usage:
            with scope_manager.scope(ScopeType.IF, cf, node=if_node) as scope:
                # ... code ...
            # Defers executed, scope cleaned up

        Args:
            scope_type: Type of scope (IF, LOOP, etc.)
            cf: ControlFlowBuilder for termination checks
            continue_target: Target block for continue statements
            break_target: Target block for break statements
            label_ctx: For labels, the LabelContext
            node: AST node for error reporting
        """
        return _ScopeContext(self, scope_type, cf, continue_target, break_target,
                             label_ctx, node)
    
    # ========================================================================
    # Label Tracking Methods
    # ========================================================================

    def reset_label_tracking(self):
        """Reset label tracking at the start of each function."""
        self._scope_labels = {}
        self._all_labels = {}
        self._scope_hazards = {}
        # Pending gotos are owned by ControlFlowBuilder (fresh per function)

    def register_label(self, ctx: Any):
        """Register a label in persistent tracking structures.

        The label_ctx is stored on the Scope object (via enter_scope),
        but we also keep _all_labels and _scope_labels for sibling/uncle
        visibility after the scope exits.

        Args:
            ctx: LabelContext to register
        """
        self._all_labels[ctx.name] = ctx
        parent_depth = ctx.parent_scope_depth
        if parent_depth not in self._scope_labels:
            self._scope_labels[parent_depth] = []
        self._scope_labels[parent_depth].append(ctx)

    def unregister_label(self, ctx: Any):
        """No-op: label cleanup is handled by scope exit.

        Kept for API compatibility. _all_labels and _scope_labels persist
        for sibling/uncle access.
        """
        pass

    def has_label(self, name: str) -> bool:
        """Check if a label with the given name exists."""
        return name in self._all_labels

    def find_label(self, name: str) -> Optional[Any]:
        """Find label for goto. Checks ancestors and siblings/uncles.

        goto can target:
        - Self (inside the label)
        - Ancestors (containing labels)
        - Siblings (labels at same parent_depth)
        - Uncles (ancestor's siblings)

        Returns:
            LabelContext if found and visible, None otherwise
        """
        # 1. Check ancestor chain via scope stack
        for scope in self._scopes:
            if scope.label_ctx is not None and scope.label_ctx.name == name:
                return scope.label_ctx

        # 2. Check siblings and uncles via persistent tracking
        # Include all scope depths in the ancestor chain so labels at any
        # ancestor level (including the function body level) are visible.
        ancestor_depths = set()
        for scope in self._scopes:
            ancestor_depths.add(scope.depth)
            if scope.label_ctx is not None:
                ancestor_depths.add(scope.label_ctx.parent_scope_depth)

        for depth in ancestor_depths:
            if depth in self._scope_labels:
                for ctx in self._scope_labels[depth]:
                    if ctx.name == name:
                        return ctx

        return None

    def find_label_for_end(self, name: str) -> Optional[Any]:
        """Find label for goto_end. Only checks ancestors (must be inside target).

        goto_end can ONLY target:
        - Self (inside the label)
        - Ancestors (containing labels)

        Returns:
            LabelContext if found in ancestor chain, None otherwise
        """
        for scope in self._scopes:
            if scope.label_ctx is not None and scope.label_ctx.name == name:
                return scope.label_ctx
        return None

    def is_ancestor_label(self, ctx: Any) -> bool:
        """Check if a label is in the current ancestor chain"""
        for scope in self._scopes:
            if scope.label_ctx is ctx:
                return True
        return False

    def capture_defer_snapshot(self) -> List:
        """Capture current defer state for forward goto resolution.

        Returns a list of (depth, callable_obj, func_ref, args, node) tuples
        in unwind order: inner scopes first, and within a scope the most recent
        defer comes first.
        """
        snapshot = []
        current_depth = self.current_depth
        for scope in reversed(self._scopes):
            if scope.depth <= current_depth:
                for defer_info in reversed(scope.defers):
                    snapshot.append((
                        scope.depth,
                        defer_info.callable_obj,
                        defer_info.func_ref,
                        defer_info.args,
                        defer_info.node
                    ))
        return snapshot

    def _get_defer_executor(self):
        """Create a defer executor function for pending goto resolution."""
        from pythoc.builtin_entities.defer import _execute_single_defer
        visitor = self._visitor

        def defer_executor(callable_obj, func_ref, args, defer_node):
            _execute_single_defer(visitor, callable_obj, func_ref, args, defer_node)

        return defer_executor

    def _check_sibling_crossing(self, goto_node: ast.AST, target_node: ast.AST,
                                parent_depth: int, goto_scope_depth: int):
        """Check that a forward sibling goto does not cross variable definitions
        or defer statements.

        A forward goto from label A to label B (both at parent_depth) must not
        skip over any variable declarations or defer registrations that occur
        at the same scope depth between the two labels.

        The check only applies when the goto was issued from the same depth as
        the target label's parent (true sibling jump). If the goto originates
        from a deeper scope, the hazards at parent_depth were already in scope
        before the goto, so no crossing occurs.

        Args:
            goto_node: The AST node of the goto() call.
            target_node: The AST node of the target label's 'with' statement.
            parent_depth: The parent scope depth where both labels live.
            goto_scope_depth: The scope depth where the goto was issued.
        """
        # Only check true sibling gotos. A goto from a deeper child scope
        # cannot cross declarations at the parent level — those declarations
        # were already in scope when the child scope was entered.
        if goto_scope_depth != parent_depth:
            return

        goto_line = getattr(goto_node, 'lineno', None)
        target_line = getattr(target_node, 'lineno', None)
        if goto_line is None or target_line is None:
            return

        hazards = self._scope_hazards.get(parent_depth, [])
        for lineno, desc in hazards:
            if goto_line < lineno < target_line:
                logger.error(
                    f"forward goto crosses {desc} at line {lineno}. "
                    f"Forward sibling goto must not skip over variable "
                    f"definitions or defer statements.",
                    node=goto_node, exc_type=SyntaxError,
                )

    def create_label_scope(self, label_name: str, cf, node: ast.AST) -> LabelContext:
        """Create a label scope: check duplicates, create blocks, register, resolve pending gotos.

        Args:
            label_name: The label name string.
            cf: ControlFlowBuilder instance.
            node: The AST With node.

        Returns:
            The newly created LabelContext.
        """
        if self.has_label(label_name):
            logger.error(f"Label '{label_name}' already defined in this function",
                        node=node, exc_type=SyntaxError)

        begin_block = cf.create_block(f"label_{label_name}_begin")
        end_block = cf.create_block(f"label_{label_name}_end")

        parent_depth = self.current_depth
        ctx = LabelContext(
            name=label_name,
            parent_scope_depth=parent_depth,
            begin_block=begin_block,
            end_block=end_block,
            node=node,
        )

        self.register_label(ctx)

        if not cf.is_terminated():
            cf.branch(begin_block)
        cf.position_at_end(begin_block)

        # Check forward sibling gotos for visibility and crossing hazards
        for pending in cf.get_pending_gotos_for_label(label_name):
            # Visibility check: the target label must be at a parent_depth
            # that was visible from the goto site
            if pending.is_goto_end:
                # goto_end can only target ancestors - forward goto_end is
                # always invalid (the target is not yet entered)
                logger.error(
                    f"goto_end: label '{label_name}' not visible. "
                    f"goto_end can only target self or ancestors "
                    f"(must be inside the label).",
                    node=pending.node, exc_type=SyntaxError,
                )
            elif parent_depth not in pending.visible_parent_depths:
                logger.error(
                    f"goto: label '{label_name}' not visible from goto site. "
                    f"Forward goto can only target siblings or uncles.",
                    node=pending.node, exc_type=SyntaxError,
                )
            self._check_sibling_crossing(
                pending.node, node, parent_depth, pending.goto_scope_depth)

        # Resolve any pending forward gotos to this label
        cf.resolve_pending_gotos(
            ctx.name, ctx.begin_block, ctx.parent_scope_depth,
            self._get_defer_executor(),
        )

        logger.debug(f"Entered label scope '{label_name}' at depth {ctx.scope_depth} "
                     f"(parent_depth={parent_depth})")
        return ctx

    def exit_label_scope(self, ctx: LabelContext, cf):
        """Exit a label scope: emit defers, branch to end block.

        IMPORTANT: Does NOT call position_at_end - that must be done AFTER
        scope_manager.exit_scope() so that is_terminated() check works correctly.

        Args:
            ctx: The LabelContext being exited.
            cf: ControlFlowBuilder instance.
        """
        if not cf.is_terminated():
            self.emit_defers_to_depth(self.current_depth - 1, cf)
            cf.branch(ctx.end_block)

        logger.debug(f"Exited label scope '{ctx.name}'")

    def register_forward_goto(self, label_name: str, cf, node: ast.AST,
                              is_goto_end: bool = False):
        """Register a forward goto (label not yet defined).

        Captures defer snapshot and visible parent depths, then delegates
        to cfbuilder. The visibility info is used at resolution time to
        verify the target label is actually visible from the goto site.

        Args:
            label_name: Target label name.
            cf: ControlFlowBuilder instance.
            node: The AST node for error reporting.
            is_goto_end: True for goto_end, False for goto.
        """
        current_block = self._visitor.builder.block
        goto_scope_depth = self.current_depth
        defer_snapshot = self.capture_defer_snapshot()

        # Capture visible parent depths for visibility check at resolution time.
        # A forward goto can target labels at any depth that is an ancestor
        # of the goto site (including the function body level). This includes
        # every depth in the scope chain, not just depths where labels exist.
        visible_parent_depths = set()
        for scope in self._scopes:
            visible_parent_depths.add(scope.depth)
            if scope.label_ctx is not None:
                visible_parent_depths.add(scope.label_ctx.parent_scope_depth)

        logger.debug(f"Forward goto to '{label_name}': captured {len(defer_snapshot)} "
                     f"defers at scope {goto_scope_depth}")

        cf.register_pending_goto(
            block=current_block,
            label_name=label_name,
            goto_scope_depth=goto_scope_depth,
            defer_snapshot=defer_snapshot,
            node=node,
            is_goto_end=is_goto_end,
            visible_parent_depths=visible_parent_depths,
        )

    def resolve_backward_goto(self, ctx: LabelContext, cf):
        """Resolve a backward goto (label already defined).

        Emits defers down to target's parent depth and branches to begin_block.

        Args:
            ctx: The target LabelContext.
            cf: ControlFlowBuilder instance.
        """
        self.emit_defers_to_depth(ctx.parent_scope_depth, cf)
        cf.branch(ctx.begin_block, kind='goto')

    def resolve_backward_goto_end(self, ctx: LabelContext, cf):
        """Resolve a backward goto_end (label already defined).

        Emits defers down to target's parent depth and branches to end_block.

        Args:
            ctx: The target LabelContext.
            cf: ControlFlowBuilder instance.
        """
        self.emit_defers_to_depth(ctx.parent_scope_depth, cf)
        cf.branch(ctx.end_block, kind='goto_end')

    def check_goto_consistency(self, func_node: Any):
        """Check that all scoped gotos have been resolved.

        Called at the end of function compilation.

        Args:
            func_node: The function AST node (for error location)
        """
        if not self._visitor:
            return
        cf = self._visitor._get_cf_builder()
        if cf is None:
            return
        unresolved = cf.get_unresolved_pending_gotos()
        if unresolved:
            for pending in unresolved:
                goto_type = "goto_end" if pending.is_goto_end else "goto"
                logger.error(f"Undefined label '{pending.label_name}' in {goto_type} statement",
                            node=pending.node, exc_type=SyntaxError)

    def clear(self):
        """Clear all scopes (for testing or function end)"""
        self._scopes.clear()


class _ScopeContext:
    """Context manager for ScopeManager.scope()"""

    def __init__(self, manager: ScopeManager, scope_type: ScopeType,
                 cf: Any, continue_target: Any, break_target: Any,
                 label_ctx: Any = None,
                 node: Optional[ast.AST] = None):
        self._manager = manager
        self._scope_type = scope_type
        self._cf = cf
        self._continue_target = continue_target
        self._break_target = break_target
        self._label_ctx = label_ctx
        self._scope: Optional[Scope] = None
        self._node = node

    def __enter__(self) -> Scope:
        self._scope = self._manager.enter_scope(
            self._scope_type,
            continue_target=self._continue_target,
            break_target=self._break_target,
            label_ctx=self._label_ctx
        )
        return self._scope
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager.exit_scope(self._cf, node=self._node)
        return False
