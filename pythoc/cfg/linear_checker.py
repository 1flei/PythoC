"""
Linear Type Checker on CFG

This module provides CFG-based linear type checking using forward dataflow analysis.
Linear type checking ensures that linear resources (tokens) are used exactly once.

Key concepts:
- LinearEvent: Events emitted by AST visitor describing linear operations
- LinearSnapshot: State of all linear variables at a program point
- LinearChecker: Performs forward dataflow analysis on CFG to check linear constraints

Architecture:
- AST visitor emits LinearRegister/LinearTransition events to CFG
- CFG checker simulates execution by processing events
- All linear state checking happens in CFG checker (single source of truth)

Rules:
1. At merge points: all incoming paths must have compatible linear states
2. At loop back edges: state must equal loop header entry state (invariant)
3. At function exit: all linear tokens must be consumed
"""

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Any, Union, TYPE_CHECKING
from ..logger import logger
from ..source import ast_location
import ast

from .graph import CFG, CFGBlock, CFGEdge
from .linear_state import LinearState

if TYPE_CHECKING:
    from ..registry import VariableInfo


# =============================================================================
# Linear Events - emitted by AST visitor, processed by CFG checker
# =============================================================================

@dataclass
class LinearRegister:
    """Event: Variable with linear type enters scope
    
    Emitted when:
    - Variable declaration: `t: linear` -> initial_state='invalid'
    - Function parameter: `def f(t: linear)` -> initial_state='valid'
    - Assignment to new var: `t = linear()` -> initial_state='invalid' (then LinearTransition)
    
    Uses var_id (unique ID) instead of var_name to distinguish shadowed variables.
    The source node and owner type support diagnostic names and locations.
    """
    var_id: int            # Binding or temporary owner ID
    var_name: str          # Variable name for error messages
    path: Tuple[int, ...]  # Index path for composite types, () for simple
    initial_state: str     # 'valid' (function param) or 'invalid' (declaration)
    node: Optional[ast.AST] = None
    type_hint: Any = None


@dataclass
class LinearTransition:
    """Event: Linear state change
    
    Emitted when:
    - `t = linear()` -> ('invalid', 'valid') - create token
    - `consume(t)` -> ('valid', 'invalid') - consume token
    - `f(t)` where f takes linear -> ('valid', 'invalid') - transfer ownership
    - `return t` -> ('valid', 'invalid') - return transfers ownership
    
    Uses var_id (unique ID) instead of var_name to distinguish shadowed variables.
    The source node identifies the ownership change.
    """
    var_id: int            # Binding or temporary owner ID
    var_name: str          # Variable name for error messages
    path: Tuple[int, ...]  # Index path for composite types, () for simple
    old_state: str         # 'valid' or 'invalid' - expected current state
    new_state: str         # 'valid' or 'invalid' - new state after transition
    node: Optional[ast.AST] = None


@dataclass
class LinearCopy:
    """A borrowed linear reference was used where an owned value is required."""
    var_id: int
    var_name: str
    path: Tuple[int, ...]
    node: Optional[ast.AST] = None


LinearEvent = Union[LinearRegister, LinearTransition, LinearCopy]


# Type alias: owner ID -> {path -> state}
# Bindings and temporary values share the VariableInfo ID allocator.
# path is a tuple of integers representing field access path
# state is 'valid' (active/usable) or 'invalid' (consumed/undefined)
LinearSnapshot = Dict[int, Dict[Tuple[int, ...], str]]

# Mapping from var_id to its source name and registration node.
VarIdInfo = Dict[int, Tuple[str, Optional[ast.AST]]]


@dataclass
class LinearError:
    """Represents a linear type error detected during CFG analysis"""
    kind: str  # 'merge_inconsistent', 'loop_invariant_violated', 
               # 'unconsumed_at_exit', 'use_after_consume', 'leak'
    block_id: int
    message: str
    details: Any = None
    source_node: Optional[ast.AST] = None
    related: Tuple[Tuple[Optional[ast.AST], str], ...] = ()
    
    def format(self, source=(None, 0)) -> str:
        """Format the failure and its related source locations."""
        lines = [f"Error: {self.message}"]
        if self.kind == 'merge_inconsistent':
            for path, branches in self.details or ():
                lines.append(f"  {path}:")
                for index, (node, state) in enumerate(branches, 1):
                    location = ast_location(node, source)
                    label = str(location) if location else f'incoming path {index}'
                    lines.append(f'    - {label}: {_ownership_state(state)}')
        notes = dict.fromkeys((ast_location(node, source), message)
                              for node, message in self.related)
        lines.extend(f'  {location}: {message}' for location, message in notes
                     if location is not None)
        return '\n'.join(lines)


def _ownership_state(state):
    return 'held' if state in ('valid', 'active') else 'unavailable'


def find_snapshot_diffs(
    s1: LinearSnapshot, s2: LinearSnapshot, 
    var_id_info: Optional[VarIdInfo] = None
) -> List[Dict]:
    """Find differences between two snapshots
    
    Args:
        s1: First snapshot (typically 'before' or 'expected')
        s2: Second snapshot (typically 'after' or 'actual')
        var_id_info: Optional mapping from var_id to (var_name, origin)
        
    Returns:
        List of diffs, each with 'path_str' and 'states'
    """
    diffs = []
    all_var_ids = set(s1.keys()) | set(s2.keys())
    
    for var_id in sorted(all_var_ids):
        var_name = f"<id={var_id}>"
        if var_id_info and var_id in var_id_info:
            var_name, _ = var_id_info[var_id]
        
        paths1 = s1.get(var_id, {})
        paths2 = s2.get(var_id, {})
        all_paths = set(paths1.keys()) | set(paths2.keys())
        
        for path in sorted(all_paths):
            state1 = paths1.get(path, 'invalid')
            state2 = paths2.get(path, 'invalid')
            
            # Check if valid status differs (handle both old and new naming)
            is_valid1 = (state1 in ('valid', 'active'))
            is_valid2 = (state2 in ('valid', 'active'))
            if is_valid1 != is_valid2:
                path_str = f"{var_name}[{']['.join(map(str, path))}]" if path else var_name
                diffs.append({
                    'path_str': path_str,
                    'states': [('s1', state1), ('s2', state2)]
                })
    
    return diffs


class LinearChecker:
    """Check linear types on CFG using forward dataflow analysis
    
    This checker performs forward dataflow analysis on the CFG to verify
    that linear resources are used correctly:
    
    1. At merge points (blocks with multiple predecessors), all incoming
       paths must have compatible linear states
    2. At loop back edges, the exit state must match the header entry state
    3. At function exit, all linear tokens must be consumed
    
    Usage:
        checker = LinearChecker()
        errors = checker.check(cfg, initial_snapshot)
        if errors:
            for err in errors:
                print(err.format())
    """

    def __init__(self):
        """Initialize the linear checker"""
        self.errors: List[LinearError] = []
        
        self._owners: Dict[int, LinearRegister] = {}
        self._unavailable_uses = []
        
        # Snapshots at block entry/exit (computed during dataflow analysis)
        self.entry_snapshots: Dict[int, LinearState] = {}
        self.exit_snapshots: Dict[int, LinearState] = {}
        self._token_ids = {}
        self._token_paths = []

    def _token_id(self, var_id, path):
        key = (var_id, path)
        if key not in self._token_ids:
            self._token_ids[key] = len(self._token_paths)
            self._token_paths.append(key)
        return self._token_ids[key]

    def _describe_state(self, state):
        snapshot = {}
        for token in state:
            var_id, path = self._token_paths[token]
            snapshot.setdefault(var_id, {})[path] = 'valid'
        return snapshot

    def _owner_path(self, var_id, path, fallback='temporary value'):
        from ..schema_protocol import iter_schema_fields

        owner = self._owners.get(var_id)
        parts = [owner.var_name if owner is not None else fallback]
        type_hint = owner.type_hint if owner is not None else None
        for index in path:
            name, type_hint = next(
                ((field_name, field_type) for field_index, field_name, field_type
                 in iter_schema_fields(type_hint) if field_index == index), (None, None))
            parts.append(f'.{name}' if name is not None else f'[{index}]')
        return ''.join(parts)
    
    def check(
        self, 
        cfg: CFG, 
        initial_snapshot: LinearSnapshot
    ) -> List[LinearError]:
        """Run linear type checking on CFG
        
        Args:
            cfg: The control flow graph to check
            initial_snapshot: Initial linear state (from function parameters)
            
        Returns:
            List of LinearError objects describing any violations
        """
        self.errors = []
        self._owners = {}
        self._unavailable_uses = []
        self._token_ids = {}
        self._token_paths = []
        initial = LinearState()
        for var_id, paths in initial_snapshot.items():
            for path, state in paths.items():
                initial = initial.with_token(
                    self._token_id(var_id, path), state in ('valid', 'active'))
        self.entry_snapshots = {cfg.entry_id: initial}
        self.exit_snapshots = {}
        
        # Process blocks in topological order
        for block in cfg.topological_order():
            block_id = block.id
            
            # Skip unreachable blocks (no entry snapshot computed)
            if block_id != cfg.entry_id and block_id not in self.entry_snapshots:
                # Try to compute entry from predecessors
                entry = self._compute_entry_snapshot(cfg, block_id)
                if entry is None:
                    continue
                self.entry_snapshots[block_id] = entry
            
            # Get entry snapshot
            entry_snapshot = self.entry_snapshots.get(block_id)
            if entry_snapshot is None:
                continue
            
            # Virtual exit block: no simulation needed, just propagate entry as exit
            # The exit block is a merge point - its entry snapshot is the merged
            # state from all return/fallthrough paths
            if block_id == cfg.exit_id:
                self.exit_snapshots[block_id] = entry_snapshot
                continue
            
            # Simulate block execution to get exit snapshot
            exit_snapshot = self._simulate_block(cfg, block, entry_snapshot)
            if exit_snapshot is not None:
                self.exit_snapshots[block_id] = exit_snapshot
            
            # Propagate to successors
            for edge in cfg.get_successors(block_id):
                if edge.kind == 'loop_back':
                    # Check loop invariant
                    self._check_loop_invariant(cfg, edge, exit_snapshot)
                else:
                    # Propagate to successor
                    target_id = edge.target_id
                    if target_id not in self.entry_snapshots:
                        # First predecessor to reach this block
                        self.entry_snapshots[target_id] = exit_snapshot
        
        # After processing all blocks, check merge points for consistency
        self._check_merge_points(cfg)
        self._explain_unavailable_uses(cfg)
        
        return self.errors
    
    def _check_merge_points(self, cfg: CFG):
        """Check all merge points for linear state consistency
        
        A merge point is a block with multiple predecessors (excluding back edges).
        All predecessors must have compatible linear states.
        
        Check order:
        1. First check all non-exit merge points (more specific errors)
        2. Then check exit block (less specific, catch-all error)
        """
        # First pass: check all non-exit merge points
        for block in cfg.iter_blocks():
            block_id = block.id
            
            # Skip exit block for first pass
            if block_id == cfg.exit_id:
                continue
            
            # Get predecessors (excluding back edges)
            preds = [e for e in cfg.get_predecessors(block_id) if e.kind != 'loop_back']
            
            if len(preds) <= 1:
                continue  # Not a merge point
            
            # Collect exit snapshots from predecessors
            pred_info: List[Tuple[CFGEdge, LinearState]] = []
            for edge in preds:
                if edge.source_id in self.exit_snapshots:
                    pred_info.append((edge, self.exit_snapshots[edge.source_id]))
            
            if len(pred_info) <= 1:
                continue  # Not enough predecessors with snapshots
            
            # Check consistency
            first_edge, first_snapshot = pred_info[0]
            for edge, snapshot in pred_info[1:]:
                if first_snapshot != snapshot:
                    self._error_merge_inconsistent(cfg, block_id, pred_info)
                    break  # Only report once per merge point
        
        # Second pass: check exit block
        # This implements semantic 2: variables must not hold linear state at lifetime end
        block_id = cfg.exit_id
        preds = [e for e in cfg.get_predecessors(block_id) if e.kind != 'loop_back']
        if len(preds) >= 1:
            self._check_exit_block_consumed(cfg, block_id)

    def _check_exit_block_consumed(self, cfg: CFG, block_id: int):
        """Check that all linear tokens are consumed at function exit
        
        This implements semantic 2: variables must not hold linear state at lifetime end.
        At function exit (virtual exit block), all linear tokens must be consumed.
        
        The exit block is a merge point for all return/fallthrough paths.  We must
        inspect every predecessor's exit snapshot, not just the first one that was
        propagated during the forward pass.  If any path reaches the exit with a
        still-active linear token, that token is leaked.
        
        Args:
            cfg: The CFG
            block_id: Exit block ID (should be cfg.exit_id)
        """
        for edge in cfg.get_predecessors(block_id):
            if edge.kind == 'loop_back':
                continue
            predecessor = edge.source_id
            state = self.exit_snapshots.get(predecessor)
            if not state:
                continue
            unconsumed = []
            related = []
            for token in sorted(state, key=self._token_paths.__getitem__):
                path = self._owner_path(*self._token_paths[token])
                unconsumed.append(path)
                events, _ = self._reaching_ownership_events(cfg, token, predecessor, None, True)
                related.extend((event.node, f'ownership of {path} starts here') for event in events)
            statements = cfg.blocks[predecessor].stmts
            node = statements[-1] if statements else self._get_block_source_node(cfg, predecessor)
            self.errors.append(LinearError(
                kind='unconsumed_at_exit',
                block_id=block_id,
                message='Linear tokens not consumed before function exit: ' + ', '.join(unconsumed),
                source_node=node,
                related=tuple(related),
            ))
    
    def _compute_entry_snapshot(
        self, cfg: CFG, block_id: int
    ) -> Optional[LinearState]:
        """Compute entry snapshot from predecessors
        
        For blocks with multiple predecessors (merge points), checks that
        all incoming paths have compatible linear states.
        
        Args:
            cfg: The CFG
            block_id: Block to compute entry for
            
        Returns:
            Entry snapshot, or None if not yet computable
        """
        # Collect snapshots from predecessors (excluding back edges)
        pred_info: List[Tuple[CFGEdge, LinearState]] = []
        for edge in cfg.get_predecessors(block_id):
            if edge.kind == 'loop_back':
                continue
            if edge.source_id in self.exit_snapshots:
                pred_info.append((edge, self.exit_snapshots[edge.source_id]))
        
        if not pred_info:
            return None
        
        if len(pred_info) == 1:
            return pred_info[0][1]
        
        # Multiple predecessors - MERGE POINT
        return self._merge_snapshots(cfg, block_id, pred_info)
    
    def _merge_snapshots(
        self, cfg: CFG, block_id: int,
        pred_info: List[Tuple[CFGEdge, LinearState]]
    ) -> LinearState:
        """Merge snapshots at merge point - all must be compatible
        
        Args:
            cfg: The CFG
            block_id: The merge block ID
            pred_info: List of (edge, snapshot) from predecessors
            
        Returns:
            Merged snapshot (uses first snapshot as base)
        """
        first_edge, first_snapshot = pred_info[0]
        
        for edge, snapshot in pred_info[1:]:
            if first_snapshot != snapshot:
                self._error_merge_inconsistent(cfg, block_id, pred_info)
                return first_snapshot
        
        return first_snapshot
    
    def _get_block_source_node(self, cfg: CFG, block_id: int) -> Optional[ast.AST]:
        """Get a source AST node for a block for error reporting
        
        Tries to find the first statement in the block, or falls back to
        predecessor blocks if the target block is empty. Uses BFS to search
        predecessors recursively until a block with statements is found.
        """
        block = cfg.get_block(block_id)
        if block and block.stmts:
            return block.stmts[0]
        
        # BFS to find nearest predecessor with statements
        visited = {block_id}
        queue = deque([block_id])
        
        while queue:
            current_id = queue.popleft()
            for edge in cfg.get_predecessors(current_id):
                pred_id = edge.source_id
                if pred_id in visited:
                    continue
                visited.add(pred_id)
                
                pred_block = cfg.get_block(pred_id)
                if pred_block and pred_block.stmts:
                    return pred_block.stmts[-1]
                
                queue.append(pred_id)
        
        # If no predecessor has statements, try successors
        visited = {block_id}
        queue = deque([block_id])
        
        while queue:
            current_id = queue.popleft()
            for edge in cfg.get_successors(current_id):
                succ_id = edge.target_id
                if succ_id in visited:
                    continue
                visited.add(succ_id)
                
                succ_block = cfg.get_block(succ_id)
                if succ_block and succ_block.stmts:
                    return succ_block.stmts[0]
                
                queue.append(succ_id)
        
        return None
    
    def _error_merge_inconsistent(
        self, cfg: CFG, block_id: int,
        pred_info: List[Tuple[CFGEdge, LinearState]]
    ):
        """Report inconsistent snapshots at merge point"""
        pred_info = [(edge, self._describe_state(state))
                     for edge, state in pred_info]
        # Find which (var_id, path) pairs have different valid status
        all_var_ids: Set[int] = set()
        for _, snapshot in pred_info:
            all_var_ids.update(snapshot.keys())
        
        diffs = []
        for var_id in sorted(all_var_ids):
            # Collect all paths for this variable
            all_paths: Set[Tuple[int, ...]] = set()
            for _, snapshot in pred_info:
                if var_id in snapshot:
                    all_paths.update(snapshot[var_id].keys())
            
            for path in sorted(all_paths):
                states_for_path = []
                for edge, snapshot in pred_info:
                    state = snapshot.get(var_id, {}).get(path, 'invalid')
                    states_for_path.append((edge, state))
                
                # Check if valid status differs (handle both old and new naming)
                valid_values = set(s in ('valid', 'active') for _, s in states_for_path)
                if len(valid_values) > 1:
                    path_str = self._owner_path(var_id, path)
                    branches = [(self._branch_ownership_source(cfg, edge, var_id, path), state)
                                for edge, state in states_for_path]
                    diffs.append((path_str, branches))
        
        self.errors.append(LinearError(
            kind='merge_inconsistent',
            block_id=block_id,
            message='Inconsistent linear states at merge point',
            details=diffs,
            source_node=self._get_block_source_node(cfg, block_id)
        ))

    def _branch_ownership_source(self, cfg, edge, var_id, path):
        block = cfg.get_block(edge.source_id)
        for event in reversed(block.linear_events):
            if (event.var_id == var_id and event.path == path and event.node is not None
                    and isinstance(event, (LinearRegister, LinearTransition))):
                return event.node
        if block.stmts:
            return block.stmts[-1]
        return self._get_block_source_node(cfg, edge.source_id)
    
    def _simulate_block(
        self, cfg: CFG, block: CFGBlock, entry_snapshot: LinearState
    ) -> LinearState:
        """Simulate block execution by processing linear events
        
        Processes LinearRegister and LinearTransition events recorded in the block.
        This is the core of event-based linear type checking.
        
        Args:
            cfg: The CFG
            block: The block to simulate
            entry_snapshot: Snapshot at block entry
            
        Returns:
            Snapshot at block exit
        """
        result = entry_snapshot
        
        # Process each linear event in the block
        for index, event in enumerate(block.linear_events):
            if isinstance(event, LinearCopy):
                path_str = self._owner_path(event.var_id, event.path, event.var_name)
                self.errors.append(LinearError(
                    kind='illegal_copy', block_id=block.id,
                    message=f"Cannot assign linear token '{path_str}' "
                            '(use move() to transfer ownership)',
                    source_node=event.node))
            elif isinstance(event, LinearRegister):
                # Variable enters scope with initial state
                var_id = event.var_id
                path = event.path
                result = result.with_token(
                    self._token_id(var_id, path), event.initial_state == 'valid')
                # Store var info for error reporting
                self._owners[var_id] = event
                logger.debug(f"LinearChecker: Register id={var_id} {event.var_name}{path} = {event.initial_state}")
            
            elif isinstance(event, LinearTransition):
                # State transition
                var_id = event.var_id
                path = event.path
                
                # Get current state
                token = self._token_id(var_id, path)
                current_state = 'valid' if token in result else 'invalid'
                
                # Validate transition: current state must match expected old_state
                if current_state != event.old_state:
                    # State mismatch - report error
                    path_str = self._owner_path(var_id, path, event.var_name)
                    
                    if event.old_state == 'valid' and current_state == 'invalid':
                        # Expected valid but got invalid -> use after consume or use undefined
                        error = LinearError(
                            kind='use_after_consume',
                            block_id=block.id,
                            message=f"Linear token '{path_str}' already consumed or undefined",
                            source_node=event.node
                        )
                        self.errors.append(error)
                        self._unavailable_uses.append((error, path_str, event, index))
                    elif event.old_state == 'invalid' and current_state == 'valid':
                        # Expected invalid but got valid -> reassigning valid token
                        self.errors.append(LinearError(
                            kind='reassign_valid',
                            block_id=block.id,
                            message=f"Cannot reassign '{path_str}': linear token not consumed",
                            source_node=event.node
                        ))
                
                # Apply transition regardless (to continue analysis)
                result = result.with_token(token, event.new_state == 'valid')
                logger.debug(f"LinearChecker: Transition id={var_id} {event.var_name}{path}: {current_state} -> {event.new_state}")
        
        return result

    def _explain_unavailable_uses(self, cfg):
        for error, path, event, index in self._unavailable_uses:
            token = self._token_ids[event.var_id, event.path]
            events, complete = self._reaching_ownership_events(
                cfg, token, error.block_id, index, False)
            reasons = {'undefined' if isinstance(origin, LinearRegister) else 'consumed'
                       for origin in events}
            if complete and reasons == {'consumed'}:
                error.message = f"Linear token '{path}' already consumed"
            elif complete and reasons == {'undefined'}:
                error.message = f"Linear token '{path}' is undefined"
            error.related = tuple(
                (origin.node, 'declared here without an owned value'
                 if isinstance(origin, LinearRegister) else 'ownership was consumed or transferred here')
                for origin in events)

    def _reaching_ownership_events(self, cfg, token, block_id, index, wanted):
        """Find the events that established the checked state on incoming paths."""
        owner_path = self._token_paths[token]
        pending = deque([(block_id, index)])
        visited = set()
        origins = []
        complete = True
        while pending:
            source = pending.popleft()
            if source in visited:
                continue
            visited.add(source)
            block, end = source
            held = token in self.entry_snapshots[block]
            origin = None
            for event in cfg.blocks[block].linear_events[:end]:
                if (event.var_id, event.path) != owner_path:
                    continue
                if isinstance(event, LinearRegister):
                    held = event.initial_state == 'valid'
                    origin = event
                elif isinstance(event, LinearTransition):
                    new_state = event.new_state == 'valid'
                    if new_state != held:
                        origin = event
                    held = new_state
            if held != wanted:
                complete = False
                continue
            if origin is not None:
                origins.append(origin)
                continue

            predecessors = [edge.source_id for edge in cfg.get_predecessors(block)
                            if edge.kind != 'loop_back' and edge.source_id in self.exit_snapshots]
            if not predecessors:
                complete = False
                continue
            for predecessor in predecessors:
                if (token in self.exit_snapshots[predecessor]) == wanted:
                    pending.append((predecessor, None))
                else:
                    complete = False
        return origins, complete
    
    def _check_loop_invariant(
        self, cfg: CFG, back_edge: CFGEdge, exit_snapshot: LinearState
    ):
        """Check that loop body preserves linear state
        
        At a loop back edge, the exit snapshot must match the header entry
        snapshot (loop invariant).
        
        Args:
            cfg: The CFG
            back_edge: The loop back edge
            exit_snapshot: Snapshot at the source of the back edge
        """
        header_id = back_edge.target_id
        header_entry = self.entry_snapshots.get(header_id)
        
        if header_entry is None:
            return
        
        if exit_snapshot != header_entry:
            diff_strs = []
            changed = set(header_entry) ^ set(exit_snapshot)
            for token in sorted(changed, key=self._token_paths.__getitem__):
                path = self._owner_path(*self._token_paths[token])
                before = 'valid' if token in header_entry else 'invalid'
                after = 'valid' if token in exit_snapshot else 'invalid'
                diff_strs.append(f"{path}: {_ownership_state(before)} at loop entry, "
                                 f"{_ownership_state(after)} at the end of the iteration")
            
            self.errors.append(LinearError(
                kind='loop_invariant_violated',
                block_id=back_edge.source_id,
                message=f"Loop body changes linear state: {'; '.join(diff_strs)}",
                source_node=self._get_block_source_node(cfg, back_edge.source_id)
            ))
    


def check_linear_types_on_cfg(
    cfg: CFG,
    initial_snapshot: Optional[LinearSnapshot] = None
) -> List[LinearError]:
    """Convenience function to run linear type checking on CFG

    Args:
        cfg: The control flow graph
        initial_snapshot: Optional initial snapshot (defaults to capturing current state)

    Returns:
        List of linear type errors
    """
    if initial_snapshot is None:
        initial_snapshot = {}

    checker = LinearChecker()
    return checker.check(cfg, initial_snapshot)
