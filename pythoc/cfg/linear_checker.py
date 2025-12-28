"""
Linear Type Checker on CFG

This module provides CFG-based linear type checking using forward dataflow analysis.
Linear type checking ensures that linear resources (tokens) are used exactly once.

Key concepts:
- LinearSnapshot: Captures the linear state of all variables at a program point
- LinearChecker: Performs forward dataflow analysis on CFG to check linear constraints

Rules:
1. At merge points: all incoming paths must have compatible linear states
2. At loop back edges: state must equal loop header entry state (invariant)
3. At function exit: all linear tokens must be consumed
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING
from ..logger import logger
import ast
import copy

from .graph import CFG, CFGBlock, CFGEdge

if TYPE_CHECKING:
    from ..registry import VariableRegistry, VariableInfo


# Type alias: var_name -> {path -> state}
# path is a tuple of integers representing field access path
# state is 'active', 'consumed', or 'undefined'
LinearSnapshot = Dict[str, Dict[Tuple[int, ...], str]]


@dataclass
class LinearError:
    """Represents a linear type error detected during CFG analysis"""
    kind: str  # 'merge_inconsistent', 'loop_invariant_violated', 
               # 'unconsumed_at_exit', 'use_after_consume', 'leak'
    block_id: int
    message: str
    details: Any = None
    source_node: Optional[ast.AST] = None
    
    def format(self) -> str:
        """Format error for display"""
        if self.kind == 'merge_inconsistent':
            lines = [f"Error: {self.message}"]
            if self.details:
                for diff in self.details:
                    lines.append(f"  {diff['path_str']}:")
                    for block_id, state in diff['states']:
                        lines.append(f"    - {state} (from block {block_id})")
            return '\n'.join(lines)
        return f"Error: {self.message}"


def capture_linear_snapshot(var_registry: "VariableRegistry") -> LinearSnapshot:
    """Capture current linear states of all visible variables
    
    Args:
        var_registry: The variable registry to capture from
        
    Returns:
        LinearSnapshot mapping var_name -> {path -> state}
    """
    snapshot: LinearSnapshot = {}
    for name, var_info in var_registry.list_all_visible().items():
        if var_info.linear_states:
            snapshot[name] = dict(var_info.linear_states)
    return snapshot


def restore_linear_snapshot(var_registry: "VariableRegistry", snapshot: LinearSnapshot):
    """Restore linear states from snapshot
    
    Args:
        var_registry: The variable registry to restore to
        snapshot: The snapshot to restore from
    """
    # Clear all current linear states
    for name, var_info in var_registry.list_all_visible().items():
        if var_info.linear_states:
            var_info.linear_states.clear()
    
    # Restore from snapshot
    for name, states_dict in snapshot.items():
        var_info = var_registry.lookup(name)
        if var_info:
            var_info.linear_states = dict(states_dict)


def copy_snapshot(snapshot: LinearSnapshot) -> LinearSnapshot:
    """Deep copy a snapshot"""
    return {var: dict(paths) for var, paths in snapshot.items()}


def snapshots_compatible(s1: LinearSnapshot, s2: LinearSnapshot) -> bool:
    """Check if two snapshots are compatible for merging
    
    Two snapshots are compatible if for every (var_name, path):
    - Both have 'active', OR
    - Both have non-active (consumed/undefined)
    
    We don't require exact state match - just whether it's usable or not.
    
    Args:
        s1: First snapshot
        s2: Second snapshot
        
    Returns:
        True if snapshots are compatible
    """
    all_vars = set(s1.keys()) | set(s2.keys())
    for var_name in all_vars:
        paths1 = s1.get(var_name, {})
        paths2 = s2.get(var_name, {})
        all_paths = set(paths1.keys()) | set(paths2.keys())
        
        for path in all_paths:
            state1 = paths1.get(path, 'undefined')
            state2 = paths2.get(path, 'undefined')
            # Only compare active vs non-active
            is_active1 = (state1 == 'active')
            is_active2 = (state2 == 'active')
            if is_active1 != is_active2:
                return False
    return True


def find_snapshot_diffs(s1: LinearSnapshot, s2: LinearSnapshot) -> List[Dict]:
    """Find differences between two snapshots
    
    Args:
        s1: First snapshot (typically 'before' or 'expected')
        s2: Second snapshot (typically 'after' or 'actual')
        
    Returns:
        List of diffs, each with 'path_str' and 'states'
    """
    diffs = []
    all_vars = set(s1.keys()) | set(s2.keys())
    
    for var_name in sorted(all_vars):
        paths1 = s1.get(var_name, {})
        paths2 = s2.get(var_name, {})
        all_paths = set(paths1.keys()) | set(paths2.keys())
        
        for path in sorted(all_paths):
            state1 = paths1.get(path, 'undefined')
            state2 = paths2.get(path, 'undefined')
            
            # Check if active status differs
            is_active1 = (state1 == 'active')
            is_active2 = (state2 == 'active')
            if is_active1 != is_active2:
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
        checker = LinearChecker(var_registry)
        errors = checker.check(cfg, initial_snapshot)
        if errors:
            for err in errors:
                print(err.format())
                
    For CFG-based checking with pre-recorded snapshots:
        checker = LinearChecker(var_registry)
        errors = checker.check(cfg, initial_snapshot, 
                               recorded_entry_snapshots=cf._entry_snapshots,
                               recorded_exit_snapshots=cf._exit_snapshots)
    """
    
    def __init__(self, var_registry: "VariableRegistry"):
        """Initialize the linear checker
        
        Args:
            var_registry: Variable registry for looking up variable info
        """
        self.var_registry = var_registry
        self.errors: List[LinearError] = []
        
        # Snapshots at block entry/exit
        self.entry_snapshots: Dict[int, LinearSnapshot] = {}
        self.exit_snapshots: Dict[int, LinearSnapshot] = {}
        
        # Pre-recorded snapshots from ControlFlowBuilder (if provided)
        self._recorded_entry_snapshots: Optional[Dict[int, LinearSnapshot]] = None
        self._recorded_exit_snapshots: Optional[Dict[int, LinearSnapshot]] = None
    
    def check(
        self, 
        cfg: CFG, 
        initial_snapshot: LinearSnapshot,
        recorded_entry_snapshots: Optional[Dict[int, LinearSnapshot]] = None,
        recorded_exit_snapshots: Optional[Dict[int, LinearSnapshot]] = None
    ) -> List[LinearError]:
        """Run linear type checking on CFG
        
        Args:
            cfg: The control flow graph to check
            initial_snapshot: Initial linear state (from function parameters)
            recorded_entry_snapshots: Pre-recorded entry snapshots from ControlFlowBuilder
            recorded_exit_snapshots: Pre-recorded exit snapshots from ControlFlowBuilder
            
        Returns:
            List of LinearError objects describing any violations
        """
        self.errors = []
        self._recorded_entry_snapshots = recorded_entry_snapshots
        self._recorded_exit_snapshots = recorded_exit_snapshots
        
        self.entry_snapshots = {cfg.entry_id: copy_snapshot(initial_snapshot)}
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
            
            # Simulate block execution to get exit snapshot
            exit_snapshot = self._simulate_block(cfg, block, entry_snapshot)
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
                        self.entry_snapshots[target_id] = copy_snapshot(exit_snapshot)
        
        # After processing all blocks, check merge points for consistency
        self._check_merge_points(cfg)
        
        # Check function exit
        self._check_function_exit(cfg)
        
        return self.errors
    
    def _check_merge_points(self, cfg: CFG):
        """Check all merge points for linear state consistency
        
        A merge point is a block with multiple predecessors (excluding back edges).
        All predecessors must have compatible linear states.
        """
        from ..logger import logger
        
        for block in cfg.iter_blocks():
            block_id = block.id
            
            # Get predecessors (excluding back edges)
            preds = [e for e in cfg.get_predecessors(block_id) if e.kind != 'loop_back']
            
            if len(preds) <= 1:
                continue  # Not a merge point
            
            # Collect exit snapshots from predecessors
            pred_info: List[Tuple[CFGEdge, LinearSnapshot]] = []
            for edge in preds:
                if edge.source_id in self.exit_snapshots:
                    pred_info.append((edge, self.exit_snapshots[edge.source_id]))
            
            if len(pred_info) <= 1:
                continue  # Not enough predecessors with snapshots
            
            # Check consistency
            first_edge, first_snapshot = pred_info[0]
            for edge, snapshot in pred_info[1:]:
                if not snapshots_compatible(first_snapshot, snapshot):
                    logger.debug(f"_check_merge_points: block {block_id} has inconsistent predecessors")
                    self._error_merge_inconsistent(cfg, block_id, pred_info)
                    break  # Only report once per merge point

    def _compute_entry_snapshot(
        self, cfg: CFG, block_id: int
    ) -> Optional[LinearSnapshot]:
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
        pred_info: List[Tuple[CFGEdge, LinearSnapshot]] = []
        for edge in cfg.get_predecessors(block_id):
            if edge.kind == 'loop_back':
                continue
            if edge.source_id in self.exit_snapshots:
                pred_info.append((edge, self.exit_snapshots[edge.source_id]))
        
        if not pred_info:
            return None
        
        if len(pred_info) == 1:
            return copy_snapshot(pred_info[0][1])
        
        # Multiple predecessors - MERGE POINT
        return self._merge_snapshots(cfg, block_id, pred_info)
    
    def _merge_snapshots(
        self, cfg: CFG, block_id: int,
        pred_info: List[Tuple[CFGEdge, LinearSnapshot]]
    ) -> LinearSnapshot:
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
            if not snapshots_compatible(first_snapshot, snapshot):
                self._error_merge_inconsistent(cfg, block_id, pred_info)
                return copy_snapshot(first_snapshot)
        
        return copy_snapshot(first_snapshot)
    
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
        queue = [block_id]
        
        while queue:
            current_id = queue.pop(0)
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
        queue = [block_id]
        
        while queue:
            current_id = queue.pop(0)
            for edge in cfg.get_successors(current_id):
                succ_id = edge.target_id
                if succ_id in visited:
                    continue
                visited.add(succ_id)
                
                succ_block = cfg.get_block(succ_id)
                if succ_block and succ_block.stmts:
                    return succ_block.stmts[0]
                
                queue.append(succ_id)
        
        logger.error(f"No source node found for block {block_id}")
        return None
    
    def _error_merge_inconsistent(
        self, cfg: CFG, block_id: int,
        pred_info: List[Tuple[CFGEdge, LinearSnapshot]]
    ):
        """Report inconsistent snapshots at merge point"""
        # Find which (var, path) pairs have different active status
        all_vars: Set[str] = set()
        for _, snapshot in pred_info:
            all_vars.update(snapshot.keys())
        
        diffs = []
        for var_name in sorted(all_vars):
            # Collect all paths for this variable
            all_paths: Set[Tuple[int, ...]] = set()
            for _, snapshot in pred_info:
                if var_name in snapshot:
                    all_paths.update(snapshot[var_name].keys())
            
            for path in sorted(all_paths):
                states_for_path = []
                for edge, snapshot in pred_info:
                    state = snapshot.get(var_name, {}).get(path, 'undefined')
                    states_for_path.append((edge.source_id, state))
                
                # Check if active status differs
                active_values = set(s == 'active' for _, s in states_for_path)
                if len(active_values) > 1:
                    path_str = f"{var_name}[{']['.join(map(str, path))}]" if path else var_name
                    diffs.append({
                        'path_str': path_str,
                        'states': states_for_path
                    })
        
        self.errors.append(LinearError(
            kind='merge_inconsistent',
            block_id=block_id,
            message=f"Inconsistent linear states at merge point (block {block_id})",
            details=diffs,
            source_node=self._get_block_source_node(cfg, block_id)
        ))
    
    def _simulate_block(
        self, cfg: CFG, block: CFGBlock, entry_snapshot: LinearSnapshot
    ) -> LinearSnapshot:
        """Simulate block execution, return exit snapshot
        
        Uses pre-recorded exit snapshots from ControlFlowBuilder.
        The actual linear state tracking is done by the AST visitor during
        code generation. This checker validates the recorded snapshots.
        
        Args:
            cfg: The CFG
            block: The block to simulate
            entry_snapshot: Snapshot at block entry
            
        Returns:
            Snapshot at block exit
        """
        from ..logger import logger
        
        # Use pre-recorded exit snapshot if available
        if self._recorded_exit_snapshots and block.id in self._recorded_exit_snapshots:
            return copy_snapshot(self._recorded_exit_snapshots[block.id])
        
        # No recorded exit snapshot - this indicates a bug in ControlFlowBuilder
        # that failed to record the exit snapshot for this block
        logger.error(
            None,
            f"Internal error: no recorded exit snapshot for block {block.id}. "
            f"ControlFlowBuilder should record exit snapshots for all blocks."
        )
    
    def _check_loop_invariant(
        self, cfg: CFG, back_edge: CFGEdge, exit_snapshot: LinearSnapshot
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
        
        if not snapshots_compatible(exit_snapshot, header_entry):
            diffs = find_snapshot_diffs(header_entry, exit_snapshot)
            diff_strs = []
            for diff in diffs:
                states_str = ', '.join(f"{s}" for _, s in diff['states'])
                diff_strs.append(f"{diff['path_str']}: {states_str}")
            
            self.errors.append(LinearError(
                kind='loop_invariant_violated',
                block_id=back_edge.source_id,
                message=f"Loop body changes linear state: {'; '.join(diff_strs)}",
                details=diffs,
                source_node=self._get_block_source_node(cfg, back_edge.source_id)
            ))
    
    def _get_effective_exit_snapshot(
        self, cfg: CFG, block_id: int
    ) -> Optional[LinearSnapshot]:
        """Get effective exit snapshot for a block
        
        For exit points (blocks with no successors or return blocks), we need
        to determine the correct linear state. The challenge is that AST visitor
        executes sequentially, so recorded exit_snapshots may be incorrect for
        blocks that are only reachable from specific branches.
        
        Strategy:
        1. entry_snapshot (from dataflow) is the ground truth for block entry
        2. exit_snapshot may be wrong if AST visitor recorded it at wrong time
        3. Key insight: a block can only consume tokens (active -> consumed),
           never resurrect them (consumed -> active). If exit shows active
           but entry shows consumed, exit is wrong.
        4. Merge entry and exit: for each token, if entry is consumed, use consumed;
           otherwise use exit state (if available) or entry state.
        
        Args:
            cfg: The CFG
            block_id: Block ID
            
        Returns:
            The effective exit snapshot, or None if not available
        """
        from ..logger import logger
        
        # Check if we have both entry and exit snapshots
        has_entry = block_id in self.entry_snapshots
        has_exit = block_id in self.exit_snapshots
        
        if has_entry and has_exit:
            entry = self.entry_snapshots[block_id]
            exit_snap = self.exit_snapshots[block_id]
            
            # Merge: for each token, take the "more consumed" state
            # If entry is consumed, it stays consumed (can't resurrect)
            # If entry is active and exit is consumed, use consumed (block consumed it)
            # If entry is active and exit is active, use active
            merged: LinearSnapshot = {}
            all_vars = set(entry.keys()) | set(exit_snap.keys())
            
            for var_name in all_vars:
                entry_paths = entry.get(var_name, {})
                exit_paths = exit_snap.get(var_name, {})
                all_paths = set(entry_paths.keys()) | set(exit_paths.keys())
                
                merged[var_name] = {}
                for path in all_paths:
                    entry_state = entry_paths.get(path, 'undefined')
                    exit_state = exit_paths.get(path, 'undefined')
                    
                    # If entry is consumed, stay consumed (can't resurrect)
                    if entry_state == 'consumed':
                        merged[var_name][path] = 'consumed'
                    # If entry is active and exit is consumed, use consumed
                    elif entry_state == 'active' and exit_state == 'consumed':
                        merged[var_name][path] = 'consumed'
                    # If entry is active and exit is active, use active
                    elif entry_state == 'active' and exit_state == 'active':
                        merged[var_name][path] = 'active'
                    # If entry is undefined and exit has a state, use exit state
                    # (variable was created in this block)
                    elif entry_state == 'undefined' and exit_state != 'undefined':
                        merged[var_name][path] = exit_state
                    # Otherwise use entry state
                    else:
                        merged[var_name][path] = entry_state
            
            logger.debug(f"_get_effective_exit_snapshot: block {block_id} merged, "
                       f"entry={entry}, exit={exit_snap}, merged={merged}")
            return merged
        
        if has_entry:
            entry = self.entry_snapshots[block_id]
            logger.debug(f"_get_effective_exit_snapshot: block {block_id} only has entry={entry}")
            return entry
        
        if has_exit:
            exit_snap = self.exit_snapshots[block_id]
            logger.debug(f"_get_effective_exit_snapshot: block {block_id} only has exit={exit_snap}")
            return exit_snap
        
        # No entry or exit snapshot found - this indicates a bug
        logger.error(
            None,
            f"Internal error: no entry or exit snapshot for block {block_id}. "
            f"This block should have been processed during dataflow analysis."
        )
        return None

    def _check_function_exit(self, cfg: CFG):
        """Check all linear tokens consumed at function exit points
        
        Function exit points are:
        1. All return blocks (explicit return statements)
        2. Blocks with no successors that are reachable (implicit return for void functions)
        
        For each exit point, we check:
        - All linear tokens must be consumed (no active tokens at exit)
        
        If there are multiple exit points, we also check:
        - All exit points must have consistent linear states
        
        Note: Unreachable code (e.g., code after `while True` without break) is not checked.
        
        Args:
            cfg: The CFG
        """
        from ..logger import logger
        
        # Find all function exit points
        # Exit points are: return blocks + blocks with no successors (excluding unreachable)
        reachable = cfg.get_reachable_blocks()
        exit_points: List[int] = []
        
        # Add return blocks
        for ret_block_id in cfg.return_blocks:
            if ret_block_id in reachable and ret_block_id not in exit_points:
                exit_points.append(ret_block_id)
        
        # Add blocks with no successors (implicit return)
        for block_id in reachable:
            successors = cfg.get_successors(block_id)
            # No successors and not already in exit_points
            if not successors and block_id not in exit_points:
                exit_points.append(block_id)
        
        logger.debug(f"_check_function_exit: exit_points={exit_points}, exit_snapshots keys={list(self.exit_snapshots.keys())}")
        
        if not exit_points:
            # No exit points means infinite loop or unreachable - no linear check needed
            logger.debug("_check_function_exit: no exit points (infinite loop or unreachable)")
            return
        
        # Collect exit snapshots for all exit points
        # For blocks without exit_snapshot (e.g., empty blocks after break),
        # use the predecessor's exit_snapshot (the block is empty, state unchanged)
        exit_point_snapshots: List[Tuple[int, LinearSnapshot]] = []
        for block_id in exit_points:
            snapshot = self._get_effective_exit_snapshot(cfg, block_id)
            if snapshot is not None:
                exit_point_snapshots.append((block_id, snapshot))
            else:
                logger.debug(f"_check_function_exit: exit point {block_id} has no effective exit snapshot")
        
        if not exit_point_snapshots:
            # No exit snapshots available
            return
        
        # Check each exit point for unconsumed tokens
        for block_id, snapshot in exit_point_snapshots:
            logger.debug(f"_check_function_exit: checking block {block_id}, snapshot = {snapshot}")
            unconsumed = []
            
            for var_name, paths in snapshot.items():
                for path, state in paths.items():
                    if state == 'active':
                        path_str = f"{var_name}[{']['.join(map(str, path))}]" if path else var_name
                        unconsumed.append(path_str)
            
            if unconsumed:
                self.errors.append(LinearError(
                    kind='unconsumed_at_exit',
                    block_id=block_id,
                    message=f"Linear tokens not consumed at function exit (block {block_id}): {', '.join(unconsumed)}",
                    source_node=self._get_block_source_node(cfg, block_id)
                ))
        
        # If multiple exit points, check consistency
        if len(exit_point_snapshots) > 1:
            first_block_id, first_snapshot = exit_point_snapshots[0]
            for block_id, snapshot in exit_point_snapshots[1:]:
                if not snapshots_compatible(first_snapshot, snapshot):
                    # Find differences
                    diffs = find_snapshot_diffs(first_snapshot, snapshot)
                    self.errors.append(LinearError(
                        kind='exit_inconsistent',
                        block_id=block_id,
                        message=f"Inconsistent linear states at function exit points (blocks {first_block_id} vs {block_id})",
                        details=[{
                            'path_str': d['path_str'],
                            'states': [(first_block_id, d['states'][0][1]), (block_id, d['states'][1][1])]
                        } for d in diffs],
                        source_node=self._get_block_source_node(cfg, block_id)
                    ))


def check_linear_types_on_cfg(
    cfg: CFG,
    var_registry: "VariableRegistry",
    initial_snapshot: Optional[LinearSnapshot] = None,
    recorded_entry_snapshots: Optional[Dict[int, LinearSnapshot]] = None,
    recorded_exit_snapshots: Optional[Dict[int, LinearSnapshot]] = None
) -> List[LinearError]:
    """Convenience function to run linear type checking on CFG
    
    Args:
        cfg: The control flow graph
        var_registry: Variable registry
        initial_snapshot: Optional initial snapshot (defaults to capturing current state)
        recorded_entry_snapshots: Pre-recorded entry snapshots from ControlFlowBuilder
        recorded_exit_snapshots: Pre-recorded exit snapshots from ControlFlowBuilder
        
    Returns:
        List of linear type errors
    """
    if initial_snapshot is None:
        initial_snapshot = capture_linear_snapshot(var_registry)
    
    checker = LinearChecker(var_registry)
    return checker.check(
        cfg, 
        initial_snapshot,
        recorded_entry_snapshots=recorded_entry_snapshots,
        recorded_exit_snapshots=recorded_exit_snapshots
    )
