"""
Compile-time pattern analysis for regex optimizations.

Analyzes a parsed regex AST and DFA to extract properties used by codegen
to select optimized code paths:
  - AcceptDepthInfo: per-accept-state depth for 1-pass search
  - LinearChain: maximal no-branch paths in the DFA for memcmp
  - PatternInfo: literal suffix for early rejection guard
  - SkipInfo: Boyer-Moore bad-character skip table from DFA structure
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from . import parse as ast
from .dfa import DFA


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternInfo:
    """Static properties of a regex pattern, computed at compile time."""
    literal_suffix: Optional[bytes] = None  # e.g. b"efg" for "abcd.*efg"
    stripped_ast: object = None  # AST with literal suffix removed (e.g. a.* for a.*efg)


@dataclass(frozen=True)
class SkipInfo:
    """Boyer-Moore bad-character skip table derived from DFA structure.

    The skip table enables O(n/W) scanning for patterns with minimum
    match length W >= 2. At each position, we probe data[i + W-1] and
    use the table to determine how many bytes can be safely skipped.

    When skip > 0: advance i by skip, stay in fast-skip loop.
    When skip == 0: the probe byte could be at the last position of a
    match; fall through to byte-by-byte DFA verification.
    """
    window: int              # W = min match length
    skip_table: List[int]    # 256-entry table: byte -> skip distance


@dataclass
class LinearChain:
    """A maximal no-branch path in the DFA.

    A sequence of states where each has exactly one non-dead transition
    to a single byte value. The DFA state machine can replace byte-by-byte
    simulation of these states with a single memcmp-like comparison.
    """
    start_state: int       # DFA state where chain begins
    end_state: int         # DFA state after the last byte in the chain
    byte_sequence: bytes   # the literal bytes along the chain


@dataclass
class AcceptDepthInfo:
    """Per-accept-state depth information for 1-pass search.

    depth = number of pattern bytes consumed to reach an accept state
    (BFS distance from DFA start to that accept state).

    If all accept states have unique depths, the reverse DFA is unnecessary:
    match_start = match_end - depth.
    """
    depths: Dict[int, int]          # accept_state -> depth
    all_unique: bool                # True if every accept state has one depth
    max_depth: Optional[int]        # max depth (bounds reverse scan)


# ---------------------------------------------------------------------------
# AST-level analysis: literal suffix extraction
# ---------------------------------------------------------------------------

def _extract_literal_suffix(node) -> Optional[bytes]:
    """Extract literal suffix from a regex AST.

    Walks the AST from the right, collecting consecutive Literal nodes.
    Stops at the first non-Literal (Dot, CharClass, Repeat with variable
    count, Alternate, etc.). Skips Anchors (they have zero width).

    Returns None if suffix is empty.
    """
    parts = _collect_suffix_bytes(node)
    if parts is None or len(parts) < 1:
        return None
    return bytes(parts)


def _collect_suffix_bytes(node) -> Optional[List[int]]:
    """Recursively collect literal bytes from the right side of the AST.

    Returns a list of byte values (rightmost first, reversed at the end)
    or None if the rightmost element is not a literal.
    """
    if isinstance(node, ast.Literal):
        return [node.byte]

    if isinstance(node, ast.Group):
        return _collect_suffix_bytes(node.child)

    if isinstance(node, ast.Anchor):
        # Anchors have zero width; skip them but don't contribute bytes
        return []

    if isinstance(node, ast.Concat):
        # Walk children from right to left
        result = []
        for child in reversed(node.children):
            child_bytes = _collect_suffix_bytes(child)
            if child_bytes is None:
                break
            result = child_bytes + result
            # If this child could match variable length, stop
            if not _is_fixed_single(child):
                break
        return result if result else None

    # Dot, CharClass, Alternate, Repeat with variable count -> stop
    return None


def _is_fixed_single(node) -> bool:
    """Check if node always matches exactly one byte (fixed single char)."""
    if isinstance(node, ast.Literal):
        return True
    if isinstance(node, ast.Dot):
        return True
    if isinstance(node, ast.CharClass):
        return True
    if isinstance(node, ast.Group):
        return _is_fixed_single(node.child)
    if isinstance(node, ast.Anchor):
        return True  # zero width, fine to continue
    return False


def _suffix_preceded_by_dotstar(node, suffix_len: int) -> bool:
    """Check if the literal suffix is immediately preceded by .* in the AST.

    This is the condition under which suffix stripping is safe: the .*
    can match any content including the suffix itself, so the DFA for
    the prefix pattern (without suffix) will accept whenever the full
    pattern would accept (given the suffix exists somewhere in the data).
    """
    if not isinstance(node, ast.Concat):
        return False

    # Find where the suffix starts in the children (from right)
    remaining = suffix_len
    suffix_start_idx = len(node.children)
    for idx in range(len(node.children) - 1, -1, -1):
        child = node.children[idx]
        child_bytes = _collect_suffix_bytes(child)
        if child_bytes is None:
            break
        child_len = len(child_bytes)
        if child_len <= remaining:
            remaining -= child_len
            suffix_start_idx = idx
            if remaining == 0:
                break
        else:
            # Partial child consumed — the preceding part is inside this child
            return False

    if suffix_start_idx <= 0:
        return False

    # The child just before the suffix must be .* (Repeat(Dot(), 0, None))
    # or .+ (Repeat(Dot(), 1, None)) — any unbounded dot repeat
    prev = node.children[suffix_start_idx - 1]
    if isinstance(prev, ast.Group):
        prev = prev.child
    return (isinstance(prev, ast.Repeat) and
            isinstance(prev.child, ast.Dot) and
            prev.max_count is None)


def analyze_pattern(ast_node) -> PatternInfo:
    """Analyze a regex AST and return static pattern properties."""
    suffix = _extract_literal_suffix(ast_node)
    stripped = None
    if suffix and len(suffix) >= 1 and _suffix_preceded_by_dotstar(ast_node, len(suffix)):
        stripped = _strip_literal_suffix(ast_node, len(suffix))
    return PatternInfo(literal_suffix=suffix, stripped_ast=stripped)


def _strip_literal_suffix(node, num_bytes: int):
    """Remove num_bytes literal bytes from the right of the AST.

    Returns a new AST node with the suffix stripped, or None if the
    entire AST is consumed.
    """
    if num_bytes <= 0:
        return node

    if isinstance(node, ast.Literal):
        # This literal IS part of the suffix; consume it
        return None

    if isinstance(node, ast.Group):
        child = _strip_literal_suffix(node.child, num_bytes)
        if child is None:
            return None
        return ast.Group(child=child)

    if isinstance(node, ast.Anchor):
        # Zero width, skip
        return None

    if isinstance(node, ast.Concat):
        # Walk from right, strip suffix bytes from rightmost children
        remaining = num_bytes
        keep_up_to = len(node.children)  # index: keep children[:keep_up_to]
        for idx in range(len(node.children) - 1, -1, -1):
            child = node.children[idx]
            if remaining <= 0:
                break
            child_suffix = _collect_suffix_bytes(child)
            if child_suffix is None:
                # Non-literal child; can't strip further
                break
            child_len = len(child_suffix)
            if child_len <= remaining:
                # Entire child is part of the suffix
                remaining -= child_len
                keep_up_to = idx
            else:
                # Partial strip within this child
                stripped_child = _strip_literal_suffix(child, remaining)
                remaining = 0
                new_children = list(node.children[:idx]) + (
                    [stripped_child] if stripped_child else [])
                if not new_children:
                    return None
                if len(new_children) == 1:
                    return new_children[0]
                return ast.Concat(children=new_children)

        new_children = list(node.children[:keep_up_to])
        if not new_children:
            return None
        if len(new_children) == 1:
            return new_children[0]
        return ast.Concat(children=new_children)

    # For other node types, return as-is (shouldn't reach here
    # since suffix extraction stops at non-literal nodes)
    return node


# ---------------------------------------------------------------------------
# DFA-level analysis: accept depths
# ---------------------------------------------------------------------------

def compute_accept_depths(dfa: DFA) -> AcceptDepthInfo:
    """Compute BFS depth from start to each accept state in the DFA.

    For accept states reachable via multiple paths of different lengths,
    the depth is not unique (all_unique will be False). This happens with
    patterns like a.*z or a+b where loops create variable-length paths.

    When cycles exist on paths to accept states, max_depth is None
    (unbounded), meaning the full reverse scan is needed for search.

    Only considers real byte classes (not sentinel 256/257 classes)
    for depth computation and cycle detection.
    """
    if not dfa.accept_states:
        return AcceptDepthInfo(depths={}, all_unique=True, max_depth=None)

    # Sentinel classes to skip (they don't consume input bytes)
    sentinel_classes = set()
    if dfa.sentinel_256_class >= 0:
        sentinel_classes.add(dfa.sentinel_256_class)
    if dfa.sentinel_257_class >= 0:
        sentinel_classes.add(dfa.sentinel_257_class)

    # Step 1: BFS from start state to compute min depth per state
    visited: Dict[int, int] = {}  # state -> min depth
    queue = deque()
    queue.append((dfa.start_state, 0))
    visited[dfa.start_state] = 0

    while queue:
        state, depth = queue.popleft()
        next_depth = depth + 1
        seen_targets: Set[int] = set()
        for cls in range(dfa.num_classes):
            if cls in sentinel_classes:
                continue
            target = dfa.transitions[state][cls]
            if target == dfa.dead_state or target in seen_targets:
                continue
            seen_targets.add(target)
            if target not in visited:
                visited[target] = next_depth
                queue.append((target, next_depth))

    # Step 2: Detect cycles (back edges in BFS tree)
    has_cycle = False
    for state in visited:
        if state == dfa.dead_state:
            continue
        state_depth = visited[state]
        seen: Set[int] = set()
        for cls in range(dfa.num_classes):
            if cls in sentinel_classes:
                continue
            target = dfa.transitions[state][cls]
            if target == dfa.dead_state or target in seen:
                continue
            seen.add(target)
            if target in visited and visited[target] <= state_depth:
                has_cycle = True
                break
        if has_cycle:
            break

    # Step 3: For each accept state, determine if it is reachable
    # through a cycle. If the DFA has a cycle and that cycle is on
    # a path to an accept state, the depth is variable.
    # Conservative approach: if any cycle exists anywhere in the
    # reachable graph AND an accept state is reachable from a cyclic
    # state, treat those accept states as variable-depth.
    #
    # Simpler: if any cycle exists and accept states have min_depth > 0,
    # check if a cyclic state (one with a back edge) can reach an accept
    # state. For now, use the simple heuristic: if the graph has cycles,
    # check each accept state's reachability from cyclic states.

    # Find states that are part of or after a cycle
    cyclic_states: Set[int] = set()
    if has_cycle:
        for state in visited:
            if state == dfa.dead_state:
                continue
            state_depth = visited[state]
            seen_t: Set[int] = set()
            for cls in range(dfa.num_classes):
                if cls in sentinel_classes:
                    continue
                target = dfa.transitions[state][cls]
                if target == dfa.dead_state or target in seen_t:
                    continue
                seen_t.add(target)
                if target in visited and visited[target] <= state_depth:
                    cyclic_states.add(state)
                    cyclic_states.add(target)

    # Step 4: Compute which accept states have variable depth
    # An accept state has variable depth if it's reachable from a
    # cyclic state (via forward BFS from cyclic states).
    variable_depth_accepts: Set[int] = set()
    if cyclic_states:
        # BFS forward from all cyclic states
        reachable_from_cycle: Set[int] = set(cyclic_states)
        q2 = deque(list(cyclic_states))
        while q2:
            s = q2.popleft()
            seen_t2: Set[int] = set()
            for cls in range(dfa.num_classes):
                if cls in sentinel_classes:
                    continue
                t = dfa.transitions[s][cls]
                if t == dfa.dead_state or t in seen_t2:
                    continue
                seen_t2.add(t)
                if t not in reachable_from_cycle:
                    reachable_from_cycle.add(t)
                    q2.append(t)

        for a in dfa.accept_states:
            if a in reachable_from_cycle:
                variable_depth_accepts.add(a)

    # Step 5: Build result
    depths: Dict[int, int] = {}
    all_unique = True
    max_depth_val = 0

    for accept_state in dfa.accept_states:
        if accept_state not in visited:
            all_unique = False
            continue
        depth = visited[accept_state]
        depths[accept_state] = depth
        if depth > max_depth_val:
            max_depth_val = depth
        if accept_state in variable_depth_accepts:
            all_unique = False

    return AcceptDepthInfo(
        depths=depths,
        all_unique=all_unique,
        max_depth=None if variable_depth_accepts else (
            max_depth_val if max_depth_val > 0 else None),
    )


# ---------------------------------------------------------------------------
# DFA-level analysis: guaranteed-accept states
# ---------------------------------------------------------------------------

def find_guaranteed_accept_states(dfa: DFA) -> FrozenSet[int]:
    """Find DFA states where a match is guaranteed regardless of remaining input.

    A state is guaranteed-accept if it is an accept state AND every
    non-dead transition leads to another guaranteed-accept state (or itself).
    This captures the .* tail pattern: once the DFA enters such a state,
    no future byte can prevent acceptance.

    Uses fixed-point iteration: start with accept states that self-loop
    on all bytes, then expand to states whose all transitions stay within
    the guaranteed set.
    """
    if not dfa.accept_states:
        return frozenset()

    # Start: candidate set = all accept states
    candidates: Set[int] = set(dfa.accept_states)

    # Fixed-point: shrink candidates until stable
    changed = True
    while changed:
        changed = False
        for s in list(candidates):
            # Every non-dead transition from s must go to a candidate
            ok = True
            has_any = False
            for cls in range(dfa.num_classes):
                target = dfa.transitions[s][cls]
                if target == dfa.dead_state:
                    continue
                has_any = True
                if target not in candidates:
                    ok = False
                    break
            # A state with no non-dead transitions can't be guaranteed
            # (it's effectively dead after current byte)
            if not ok or not has_any:
                candidates.discard(s)
                changed = True

    return frozenset(candidates)


# ---------------------------------------------------------------------------
# DFA-level analysis: linear chain detection
# ---------------------------------------------------------------------------

def _state_single_byte_target(dfa: DFA, state: int) -> Optional[Tuple[int, int]]:
    """Check if a DFA state has exactly one non-dead transition to one byte.

    Returns (byte_value, target_state) if the state transitions to exactly
    one target state on exactly one byte value. Returns None otherwise
    (multiple targets, multiple bytes, self-loop, or no transitions).
    """
    if state == dfa.dead_state:
        return None

    # Collect all non-dead transitions: target -> set of byte values
    target_bytes: Dict[int, List[int]] = {}
    for byte_val in range(256):
        cls = dfa.class_map[byte_val]
        target = dfa.transitions[state][cls]
        if target != dfa.dead_state:
            target_bytes.setdefault(target, []).append(byte_val)

    # Must have exactly one non-dead target
    if len(target_bytes) != 1:
        return None

    target, byte_vals = next(iter(target_bytes.items()))

    # Must transition on exactly one byte value (not a range/class)
    if len(byte_vals) != 1:
        return None

    # Must not be a self-loop
    if target == state:
        return None

    return (byte_vals[0], target)


def find_linear_chains(dfa: DFA) -> List[LinearChain]:
    """Find all maximal linear chains in a DFA.

    A linear chain is a maximal sequence of states where each has exactly
    one non-dead transition to a single byte value (no branching, no
    self-loops, no character classes).

    Only returns chains of length >= 2 (at least 2 bytes). Single-byte
    chains aren't worth the memcmp overhead.
    """
    # Track which states are internal to a chain (to avoid starting
    # a chain from the middle)
    has_incoming_chain: Set[int] = set()

    # First pass: find all chain-eligible edges
    chain_edges: Dict[int, Tuple[int, int]] = {}  # state -> (byte, target)
    for state in range(dfa.num_states):
        result = _state_single_byte_target(dfa, state)
        if result is not None:
            byte_val, target = result
            chain_edges[state] = (byte_val, target)
            has_incoming_chain.add(target)

    # Second pass: build chains starting from states that are NOT
    # the target of another chain edge (i.e., chain heads)
    chains: List[LinearChain] = []
    visited: Set[int] = set()

    for start_state in sorted(chain_edges.keys()):
        # Skip if this state is in the middle of another chain
        if start_state in has_incoming_chain and start_state in chain_edges:
            # Check if the predecessor is also a chain edge pointing here
            # (this state is internal to a chain, not a head)
            is_internal = False
            for s, (_, t) in chain_edges.items():
                if t == start_state and s != start_state:
                    is_internal = True
                    break
            if is_internal:
                continue

        if start_state in visited:
            continue

        # Extend chain forward
        byte_seq = []
        current = start_state
        while current in chain_edges and current not in visited:
            visited.add(current)
            byte_val, target = chain_edges[current]
            byte_seq.append(byte_val)
            current = target
            # Stop at accept states so in-loop accept checks can fire
            if current in dfa.accept_states:
                break

        if len(byte_seq) >= 2:
            chains.append(LinearChain(
                start_state=start_state,
                end_state=current,
                byte_sequence=bytes(byte_seq),
            ))

    return chains


# ---------------------------------------------------------------------------
# DFA-level analysis: Boyer-Moore skip table
# ---------------------------------------------------------------------------

def compute_skip_table(dfa: DFA) -> Optional[SkipInfo]:
    """Compute Boyer-Moore bad-character skip table from DFA structure.

    Uses the *original* (non-search) DFA to derive a skip table that
    enables O(n/W) scanning, where W is the minimum match length.

    Algorithm:
      1. BFS from start to compute min depth per state.
      2. W = min(depth of accept states) = minimum match length (window).
      3. For each depth d in [0, W), collect bytes with non-dead transitions
         from states at that depth.
      4. For each byte b: skip[b] = W - 1 - max_d where max_d is the
         rightmost depth at which b can appear. If b never appears in any
         depth: skip[b] = W.

    Returns None if W < 2 (skip not beneficial).
    """
    if not dfa.accept_states:
        return None

    # Sentinel classes to skip
    sentinel_classes = set()
    if dfa.sentinel_256_class >= 0:
        sentinel_classes.add(dfa.sentinel_256_class)
    if dfa.sentinel_257_class >= 0:
        sentinel_classes.add(dfa.sentinel_257_class)

    # Step 1: BFS from start to compute min depth per state
    depth_of: Dict[int, int] = {}
    queue = deque()
    queue.append((dfa.start_state, 0))
    depth_of[dfa.start_state] = 0

    while queue:
        state, depth = queue.popleft()
        next_depth = depth + 1
        seen_targets: Set[int] = set()
        for cls in range(dfa.num_classes):
            if cls in sentinel_classes:
                continue
            target = dfa.transitions[state][cls]
            if target == dfa.dead_state or target in seen_targets:
                continue
            seen_targets.add(target)
            if target not in depth_of:
                depth_of[target] = next_depth
                queue.append((target, next_depth))

    # Step 2: W = min depth of accept states
    min_accept_depth = None
    for a in dfa.accept_states:
        if a in depth_of:
            d = depth_of[a]
            if min_accept_depth is None or d < min_accept_depth:
                min_accept_depth = d
    if min_accept_depth is None or min_accept_depth < 2:
        return None
    window = min_accept_depth

    # Step 3: For each depth d in [0, W), collect bytes with non-dead
    # transitions from states at that depth.
    # Build depth -> set of states at that depth
    states_at_depth: Dict[int, List[int]] = {}
    for state, d in depth_of.items():
        if d < window and state != dfa.dead_state:
            states_at_depth.setdefault(d, []).append(state)

    # For each byte, compute the rightmost depth where it has a non-dead
    # transition from a state at that depth.
    rightmost_depth = [-1] * 256  # byte -> max depth where it appears

    for d in range(window):
        for state in states_at_depth.get(d, []):
            for byte_val in range(256):
                cls = dfa.class_map[byte_val]
                target = dfa.transitions[state][cls]
                if target != dfa.dead_state:
                    if d > rightmost_depth[byte_val]:
                        rightmost_depth[byte_val] = d

    # Step 4: Compute skip table
    skip_table = [0] * 256
    for byte_val in range(256):
        rd = rightmost_depth[byte_val]
        if rd < 0:
            # Byte never appears in any depth position -> max skip
            skip_table[byte_val] = window
        else:
            skip_table[byte_val] = window - 1 - rd

    return SkipInfo(window=window, skip_table=skip_table)


# ---------------------------------------------------------------------------
# Per-state skip optimization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateSkipInfo:
    """Per-state skip probe information.

    For a given DFA state, this describes a lookahead probe that can
    safely skip bytes when the probe byte is not in live_bytes.

    state: the DFA state this info applies to
    offset: lookahead distance from current position
    live_bytes: bytes that could appear at this offset on any path
                from this state to an accept state within W steps
    shift: safe jump distance when probe byte not in live_bytes
    """
    state: int
    offset: int
    live_bytes: FrozenSet[int]
    shift: int


def compute_state_skips(dfa: DFA, max_window: int = 32,
                         restart_states: Set[int] = None,
                         ) -> Dict[int, StateSkipInfo]:
    """Compute per-state skip probes for a DFA.

    For each non-dead, non-accept state: BFS forward up to the
    minimum accept depth from that state (W). At each depth d,
    collect the set of live bytes. Select the best probe offset
    that maximizes: shift * (256 - |live_bytes|) / 256.

    restart_states: optional set of states to exclude from BFS.
        In a search DFA, states with broad self-loops (the .* prefix)
        should be excluded because transitions to them represent
        "restart" rather than forward progress toward a match.
        Bytes leading only to restart states are not counted as live.

    States that can reach accept through a cycle (back edge in BFS)
    are excluded, since variable-depth paths make skip probes unsafe.

    Returns a mapping from DFA state to its best StateSkipInfo,
    only for states where a useful skip (shift > 0, live_bytes < 256)
    exists.
    """
    if not dfa.accept_states:
        return {}

    if restart_states is None:
        restart_states = set()

    # Sentinel classes to skip
    sentinel_classes = set()
    if dfa.sentinel_256_class >= 0:
        sentinel_classes.add(dfa.sentinel_256_class)
    if dfa.sentinel_257_class >= 0:
        sentinel_classes.add(dfa.sentinel_257_class)

    result: Dict[int, StateSkipInfo] = {}

    for start_state in range(dfa.num_states):
        if start_state == dfa.dead_state:
            continue
        if start_state in dfa.accept_states:
            continue
        if start_state in restart_states:
            continue

        # BFS from start_state to find min accept depth (W)
        # Also detect cycles: if any back edge exists, skip this state
        visited: Dict[int, int] = {start_state: 0}
        queue = deque([(start_state, 0)])
        min_accept_depth = None
        has_cycle = False

        # Collect bytes at each depth
        bytes_at_depth: Dict[int, Set[int]] = {}

        while queue:
            s, depth = queue.popleft()
            if min_accept_depth is not None and depth >= min_accept_depth:
                continue
            if depth >= max_window:
                continue

            next_depth = depth + 1
            live_at_depth = bytes_at_depth.setdefault(depth, set())
            seen_targets: Set[int] = set()

            for byte_val in range(256):
                cls = dfa.class_map[byte_val]
                target = dfa.transitions[s][cls]
                if target == dfa.dead_state:
                    continue
                if target in restart_states:
                    # Transition to restart state: don't count as live,
                    # don't follow in BFS
                    continue
                live_at_depth.add(byte_val)
                if target not in seen_targets:
                    seen_targets.add(target)
                    if target in dfa.accept_states:
                        if min_accept_depth is None or next_depth < min_accept_depth:
                            min_accept_depth = next_depth
                    elif target not in visited or visited[target] > next_depth:
                        visited[target] = next_depth
                        queue.append((target, next_depth))

        # Skip states that can reach accept via variable-length paths.
        # A state with a self-loop (excluding restart states) on the path
        # to accept creates variable accept depths, making skip unsafe.
        has_variable_path = False
        if min_accept_depth is not None:
            for s in visited:
                if s in restart_states:
                    continue
                s_depth = visited[s]
                if s_depth >= min_accept_depth:
                    continue
                for bv in range(256):
                    cls = dfa.class_map[bv]
                    t = dfa.transitions[s][cls]
                    if t == s and t not in restart_states:
                        has_variable_path = True
                        break
                if has_variable_path:
                    break

        if has_variable_path:
            continue

        if min_accept_depth is None or min_accept_depth < 2:
            continue

        W = min_accept_depth

        # Select best probe offset: maximize shift * (256 - |live|) / 256
        best_score = 0
        best_info = None

        for d in range(W):
            live = bytes_at_depth.get(d, set())
            live_count = len(live)
            if live_count >= 256:
                continue
            # shift = how far we can advance if probe byte not in live
            shift = W - 1 - d
            if shift <= 0:
                continue
            score = shift * (256 - live_count)
            if score > best_score:
                best_score = score
                best_info = StateSkipInfo(
                    state=start_state,
                    offset=d,
                    live_bytes=frozenset(live),
                    shift=shift,
                )

        if best_info is not None:
            result[start_state] = best_info

    return result
