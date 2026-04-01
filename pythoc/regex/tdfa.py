"""
Leftmost TDFA frontend built on top of the explicit TNFA.

This module follows the broad re2c decomposition:

  TNFA -> closure items -> kernel buffers -> tag versions / tcmds -> TDFA

It intentionally implements only the leftmost core for PythoC's current
surface syntax. The schema is shaped so a later ``TDFA -> TaggedOPCSBMA``
bridge can consume explicit control transitions and command lists instead of
reconstructing tag writes from an untagged DFA closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .tnfa import (
    INTERNAL_TAG_PREFIX,
    SENTINEL_END,
    SENTINEL_START,
    TNFA,
    TagInfo,
)


HROOT = 0
TAGVER_ZERO = 0
NO_RULE = -1


@dataclass(frozen=True)
class TDFACommand:
    """One re2c-shaped tag command on a TDFA edge."""

    kind: str  # "copy", "set", or "add"
    lhs: int
    rhs: int = 0
    history: Tuple[int, ...] = ()


@dataclass(frozen=True)
class TDFAKernelItem:
    """One canonical kernel item in a TDFA state."""

    state_id: int
    tvers: int
    thist: int = HROOT


@dataclass(frozen=True)
class TDFAState:
    """One canonical TDFA state."""

    id: int
    kernel: Tuple[TDFAKernelItem, ...]
    rule: int
    accepting: bool


@dataclass(frozen=True)
class TDFA:
    """Canonical leftmost TDFA artifact."""

    states: Tuple[TDFAState, ...]
    num_states: int
    num_classes: int
    start_state: int
    dead_state: int
    accept_states: FrozenSet[int]
    transitions: Tuple[Tuple[int, ...], ...]
    class_map: Tuple[int, ...]  # real bytes only
    class_representatives: Tuple[int, ...]
    sentinel_256_class: int
    sentinel_257_class: int
    tag_names: Tuple[str, ...]
    public_tag_names: Tuple[str, ...]
    tag_slots: Dict[str, int]
    initial_commands: Tuple[TDFACommand, ...]
    transition_commands: Dict[Tuple[int, int], Tuple[TDFACommand, ...]]
    accept_commands: Dict[int, Tuple[TDFACommand, ...]]
    output_registers: Tuple[int, ...]
    register_count: int
    tagver_rows: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class _HistoryNode:
    info: TagInfo
    pred: int


@dataclass
class _ClosItem:
    state_id: int
    origin: int
    thist: int


@dataclass
class _KernelBufferItem:
    state_id: int
    origin: int
    thist: int


@dataclass
class _KernelBufferState:
    state_id: int
    paths: List[_KernelBufferItem]


@dataclass(frozen=True)
class _ConcreteConfig:
    state_id: int
    tag_values: Tuple[int, ...]


class _LeftmostHistory:
    """Minimal leftmost history chain for the Python frontend."""

    def __init__(self):
        self.nodes: List[_HistoryNode] = [_HistoryNode(TagInfo(idx=-1, neg=0), -1)]
        self._cache: Dict[int, Tuple[int, ...]] = {HROOT: ()}

    def node(self, idx: int) -> _HistoryNode:
        return self.nodes[idx]

    def link(self, info: TagInfo, pred: int) -> int:
        idx = len(self.nodes)
        self.nodes.append(_HistoryNode(info=info, pred=pred))
        return idx

    def tag_path(self, idx: int) -> Tuple[int, ...]:
        path: List[int] = []
        while idx != HROOT:
            node = self.node(idx)
            path.append(node.info.idx)
            idx = node.pred
        return tuple(reversed(path))

    def tag_indices(self, idx: int) -> Tuple[int, ...]:
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        original_idx = idx
        ordered = list(self.tag_path(idx))

        seen: Set[int] = set()
        unique: List[int] = []
        for tag_idx in ordered:
            if tag_idx in seen:
                continue
            seen.add(tag_idx)
            unique.append(tag_idx)

        result = tuple(unique)
        self._cache[original_idx] = result
        return result

    def merge(self, existing: int, incoming: int) -> Tuple[int, bool]:
        merged = existing
        existing_tags = set(self.tag_indices(existing))
        changed = False
        for tag_idx in self.tag_indices(incoming):
            if tag_idx in existing_tags:
                continue
            merged = self.link(TagInfo(idx=tag_idx, neg=0), merged)
            existing_tags.add(tag_idx)
            changed = True
        return merged, changed

class _TagVersionTable:
    """Interned rows of per-tag version numbers."""

    def __init__(self, ntags: int):
        self.ntags = ntags
        zero_row = tuple([TAGVER_ZERO] * ntags)
        self.rows: List[Tuple[int, ...]] = [zero_row]
        self.index: Dict[Tuple[int, ...], int] = {zero_row: 0}

    def insert_const(self, ver: int) -> int:
        return self.insert(tuple([ver] * self.ntags))

    def insert_succ(self, first: int) -> int:
        if self.ntags == 0:
            return 0
        return self.insert(tuple(range(first, first + self.ntags)))

    def insert(self, row: Sequence[int]) -> int:
        key = tuple(row)
        existing = self.index.get(key)
        if existing is not None:
            return existing
        idx = len(self.rows)
        self.rows.append(key)
        self.index[key] = idx
        return idx

    def __getitem__(self, idx: int) -> Tuple[int, ...]:
        return self.rows[idx]


def format_search_result(tags: Dict[str, int]) -> Dict[str, int]:
    """Convert internal begin/end tags into the public search result shape."""
    beg = INTERNAL_TAG_PREFIX + "beg"
    end = INTERNAL_TAG_PREFIX + "end"
    if beg not in tags or end not in tags:
        raise ValueError("search result requires internal begin/end tags")

    result = {
        "start": tags[beg],
        "end": tags[end],
    }
    for name, value in tags.items():
        if name.startswith(INTERNAL_TAG_PREFIX):
            continue
        result[name] = value
    return result


def _materialize_tag_dict(tag_names: Tuple[str, ...],
                          values: Sequence[int],
                          include_internal: bool) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for name, value in zip(tag_names, values):
        if value < 0:
            continue
        if not include_internal and name.startswith(INTERNAL_TAG_PREFIX):
            continue
        result[name] = value
    return result


def _matches_symbol(state, symbol: int) -> bool:
    if state.kind != "RAN":
        return False
    for lo, hi in state.ranges:
        if lo <= symbol <= hi:
            return True
    return False


def _compute_symbol_classes(tnfa: TNFA) -> Tuple[
    Tuple[int, ...],
    int,
    int,
    int,
    Tuple[int, ...],
]:
    signatures: Dict[int, Tuple[Tuple[int, int], ...]] = {}
    for symbol in range(SENTINEL_END + 1):
        sig: List[Tuple[int, int]] = []
        for state in tnfa.states:
            if state.kind != "RAN" or not _matches_symbol(state, symbol):
                continue
            sig.append((state.id, state.out1 if state.out1 is not None else -1))
        signatures[symbol] = tuple(sig)

    full_class_map = [0] * (SENTINEL_END + 1)
    sig_to_class: Dict[Tuple[Tuple[int, int], ...], int] = {(): 0}
    next_class = 1

    for symbol in range(256):
        sig = signatures[symbol]
        class_id = sig_to_class.get(sig)
        if class_id is None:
            class_id = next_class
            sig_to_class[sig] = class_id
            next_class += 1
        full_class_map[symbol] = class_id

    sentinel_256_class = next_class
    full_class_map[SENTINEL_START] = sentinel_256_class
    next_class += 1

    sentinel_257_class = next_class
    full_class_map[SENTINEL_END] = sentinel_257_class
    next_class += 1

    class_representatives = [-1] * next_class
    for symbol in range(256):
        class_id = full_class_map[symbol]
        if class_representatives[class_id] < 0:
            class_representatives[class_id] = symbol
    class_representatives[sentinel_256_class] = SENTINEL_START
    class_representatives[sentinel_257_class] = SENTINEL_END

    return (
        tuple(full_class_map[:256]),
        next_class,
        sentinel_256_class,
        sentinel_257_class,
        tuple(class_representatives),
    )


def _topsort_copy_commands(commands: Sequence[TDFACommand]) -> Tuple[Tuple[TDFACommand, ...], bool]:
    if not commands:
        return (), False

    indegree: Dict[int, int] = {}
    for command in commands:
        indegree.setdefault(command.lhs, 0)
        indegree.setdefault(command.rhs, 0)
    for command in commands:
        indegree[command.rhs] += 1

    ordered: List[TDFACommand] = []
    pending = list(commands)
    while pending:
        progressed = False
        remaining: List[TDFACommand] = []
        for command in pending:
            if indegree[command.lhs] == 0:
                indegree[command.rhs] -= 1
                ordered.append(command)
                progressed = True
            else:
                remaining.append(command)
        if not progressed:
            non_trivial_cycle = any(command.lhs != command.rhs for command in remaining)
            return tuple(ordered + remaining), non_trivial_cycle
        pending = remaining

    return tuple(ordered), False


class _TDFACompiler:
    def __init__(self, tnfa: TNFA):
        self.tnfa = tnfa
        self.history = _LeftmostHistory()

        self.tag_names = tuple(tag.name for tag in tnfa.tags)
        self.public_tag_names = tuple(
            name for name in self.tag_names
            if not name.startswith(INTERNAL_TAG_PREFIX)
        )
        self.tag_slots = {
            name: idx
            for idx, name in enumerate(self.tag_names)
        }
        self.num_tags = len(self.tag_names)
        self.initial_row = tuple(range(1, self.num_tags + 1))

        self.tagver_table = _TagVersionTable(self.num_tags)
        self.initial_tvers = self.tagver_table.insert_succ(1)
        self.maxtagver = self.num_tags
        self.output_registers = tuple(
            self._alloc_version()
            for _ in range(self.num_tags)
        )

        (self.class_map,
         self.num_classes,
         self.sentinel_256_class,
         self.sentinel_257_class,
         self.class_representatives) = _compute_symbol_classes(tnfa)

        self.states: List[TDFAState] = []
        self.state_map: Dict[Tuple[Tuple[int, int, int], ...], int] = {}
        self.transitions: List[List[int]] = []
        self.transition_commands: Dict[Tuple[int, int], Tuple[TDFACommand, ...]] = {}
        self.accept_commands: Dict[int, Tuple[TDFACommand, ...]] = {}
        self.accept_states: Set[int] = set()

    def _alloc_version(self) -> int:
        self.maxtagver += 1
        return self.maxtagver

    def _kernel_key(self, kernel: Sequence[TDFAKernelItem]) -> Tuple[Tuple[int, int, Tuple[int, ...]], ...]:
        return tuple(
            (item.state_id, item.tvers, self._history_signature(item.thist))
            for item in kernel
        )

    def _history_tags(self, thist: int) -> Tuple[int, ...]:
        return self.history.tag_indices(thist)

    def _history_signature(self, thist: int) -> Tuple[int, ...]:
        return self.history.tag_path(thist)

    def _history_has(self, thist: int, tag_idx: int) -> bool:
        return tag_idx in self._history_tags(thist)

    def _preserve_existing_version(self, tag_idx: int, old_ver: int) -> bool:
        tag = self.tnfa.tags[tag_idx]
        return (
            not tag.history and
            not tag.fixed and
            old_ver != self.initial_row[tag_idx]
        )

    def _closure(self, reach: Sequence[_ClosItem]) -> Tuple[_ClosItem, ...]:
        state: List[_ClosItem] = []
        seen_index: Dict[int, int] = {}
        seen_origin: Dict[Tuple[int, int], int] = {}
        seen_fin_index: Dict[Tuple[int, int], int] = {}

        def visit(current: _ClosItem) -> None:
            pair_key = (current.state_id, current.origin)
            existing_pair = seen_origin.get(pair_key)
            if existing_pair is not None:
                merged_thist, changed = self.history.merge(existing_pair, current.thist)
                if not changed:
                    return
                current = _ClosItem(current.state_id, current.origin, merged_thist)
                seen_origin[pair_key] = merged_thist
            else:
                seen_origin[pair_key] = current.thist

            node = self.tnfa.states[current.state_id]
            if node.kind == "FIN":
                fin_idx = seen_fin_index.get(pair_key)
                if fin_idx is None:
                    seen_fin_index[pair_key] = len(state)
                    state.append(current)
                else:
                    state[fin_idx] = current
            else:
                idx = seen_index.get(current.state_id)
                if idx is None:
                    seen_index[current.state_id] = len(state)
                    state.append(current)

            if node.kind == "ALT":
                if node.out1 is not None:
                    visit(_ClosItem(node.out1, current.origin, current.thist))
                if node.out2 is not None:
                    visit(_ClosItem(node.out2, current.origin, current.thist))
            elif node.kind == "TAG":
                next_thist = current.thist
                if node.tag is not None:
                    next_thist = self.history.link(node.tag, current.thist)
                if node.out1 is not None:
                    visit(_ClosItem(node.out1, current.origin, next_thist))

        for item in reach:
            visit(item)

        return tuple(state)

    def _prune_closure(self, closure: Sequence[_ClosItem]) -> List[_KernelBufferItem]:
        kernel: List[_KernelBufferItem] = []
        for item in closure:
            node = self.tnfa.states[item.state_id]
            if node.kind == "RAN":
                kernel.append(_KernelBufferItem(
                    state_id=item.state_id,
                    origin=item.origin,
                    thist=item.thist,
                ))
            elif node.kind == "FIN":
                kernel.append(_KernelBufferItem(
                    state_id=item.state_id,
                    origin=item.origin,
                    thist=item.thist,
                ))
        return kernel

    def _newver_key(self, tag_idx: int, base_ver: int, thist: int) -> Tuple[int, int, int]:
        return (tag_idx, base_ver, thist)

    def _build_new_versions(self,
                            origin_state_id: int,
                            kernel: Sequence[_KernelBufferItem]) -> Tuple[
                                Dict[Tuple[int, int, int], int],
                                Tuple[TDFACommand, ...],
                            ]:
        origin_state = self.states[origin_state_id]
        new_versions: Dict[Tuple[int, int, int], int] = {}
        action_keys: Set[Tuple[int, int, int]] = set()

        for item in kernel:
            origin_item = origin_state.kernel[item.origin]
            orig_thist = origin_item.thist
            if orig_thist == HROOT:
                continue

            old_row = self.tagver_table[origin_item.tvers]
            for tag_idx in self._history_tags(orig_thist):
                tag = self.tnfa.tags[tag_idx]
                if self._preserve_existing_version(tag_idx, old_row[tag_idx]):
                    continue
                base_ver = old_row[tag_idx] if tag.history else TAGVER_ZERO
                key = self._newver_key(tag_idx, base_ver, orig_thist)
                if key not in new_versions:
                    new_versions[key] = self._alloc_version()
                if not tag.fixed and (tag.history or not self._history_has(item.thist, tag_idx)):
                    action_keys.add(key)

        commands: List[TDFACommand] = []
        for key in sorted(action_keys):
            tag_idx, base_ver, thist = key
            ver = new_versions[key]
            tag = self.tnfa.tags[tag_idx]
            if tag.history:
                commands.append(TDFACommand(
                    kind="add",
                    lhs=ver,
                    rhs=abs(base_ver),
                    history=self._history_tags(thist),
                ))
            else:
                commands.append(TDFACommand(kind="set", lhs=abs(ver)))
        return new_versions, tuple(commands)

    def _materialize_initial_kernel(self,
                                    kernel: Sequence[_KernelBufferItem]) -> Tuple[
                                        Tuple[TDFAKernelItem, ...],
                                        Tuple[TDFACommand, ...],
                                    ]:
        if not kernel:
            return (), ()

        return (
            tuple(
                TDFAKernelItem(
                    state_id=item.state_id,
                    tvers=self.initial_tvers,
                    thist=item.thist,
                )
                for item in kernel
            ),
            (),
        )

    def _materialize_transition(self,
                                origin_state_id: int,
                                kernel: Sequence[_KernelBufferItem]) -> Tuple[
                                    Tuple[TDFAKernelItem, ...],
                                    Tuple[TDFACommand, ...],
                                ]:
        if not kernel:
            return (), ()

        origin_state = self.states[origin_state_id]
        new_versions, commands = self._build_new_versions(origin_state_id, kernel)
        items: List[TDFAKernelItem] = []
        for item in kernel:
            origin_item = origin_state.kernel[item.origin]
            orig_thist = origin_item.thist
            if orig_thist == HROOT:
                tvers = origin_item.tvers
            else:
                old_row = list(self.tagver_table[origin_item.tvers])
                for tag_idx in self._history_tags(orig_thist):
                    tag = self.tnfa.tags[tag_idx]
                    if self._preserve_existing_version(tag_idx, old_row[tag_idx]):
                        continue
                    base_ver = old_row[tag_idx] if tag.history else TAGVER_ZERO
                    old_row[tag_idx] = new_versions[self._newver_key(tag_idx, base_ver, orig_thist)]
                tvers = self.tagver_table.insert(old_row)
            items.append(TDFAKernelItem(
                state_id=item.state_id,
                tvers=tvers,
                thist=item.thist,
            ))
        return tuple(items), commands

    def _build_accept_commands(self,
                               kernel: Sequence[TDFAKernelItem]) -> Tuple[TDFACommand, ...]:
        if not self.output_registers:
            return ()

        fin_items = [
            item
            for item in kernel
            if self.tnfa.states[item.state_id].kind == "FIN"
        ]
        if not fin_items:
            return ()

        commands: List[TDFACommand] = []
        for tag_idx in range(self.num_tags):
            tag = self.tnfa.tags[tag_idx]
            if tag.fixed:
                continue
            dst = self.output_registers[tag_idx]
            handled = False
            for fin_item in fin_items:
                row = self.tagver_table[fin_item.tvers]
                if not self._history_has(fin_item.thist, tag_idx):
                    continue
                if tag.history:
                    commands.append(TDFACommand(
                        kind="add",
                        lhs=dst,
                        rhs=abs(row[tag_idx]),
                        history=(tag_idx,),
                    ))
                else:
                    commands.append(TDFACommand(kind="set", lhs=dst))
                handled = True
                break
            if handled:
                continue

            copy_src = None
            for fin_item in fin_items:
                row = self.tagver_table[fin_item.tvers]
                if row[tag_idx] != self.initial_row[tag_idx]:
                    copy_src = row[tag_idx]
                    break
            if copy_src is None:
                copy_src = self.tagver_table[fin_items[0].tvers][tag_idx]
            commands.append(TDFACommand(kind="copy", lhs=dst, rhs=copy_src))
        return tuple(commands)

    def _add_state(self, kernel: Sequence[TDFAKernelItem]) -> int:
        state_id = len(self.states)
        accepting = any(
            self.tnfa.states[item.state_id].kind == "FIN"
            for item in kernel
        )
        state = TDFAState(
            id=state_id,
            kernel=tuple(kernel),
            rule=0 if accepting else NO_RULE,
            accepting=accepting,
        )
        self.states.append(state)
        self.transitions.append([-1] * self.num_classes)
        self.state_map[self._kernel_key(kernel)] = state_id

        if accepting:
            self.accept_states.add(state_id)
            commands = self._build_accept_commands(kernel)
            if commands:
                self.accept_commands[state_id] = commands

        return state_id

    def _try_map_kernel(self,
                        kernel: Sequence[TDFAKernelItem],
                        target_state: TDFAState,
                        commands: Sequence[TDFACommand]) -> Optional[Tuple[TDFACommand, ...]]:
        if len(kernel) != len(target_state.kernel):
            return None

        x_to_y: Dict[int, int] = {}
        y_to_x: Dict[int, int] = {}
        for candidate_item, existing_item in zip(kernel, target_state.kernel):
            if (candidate_item.state_id != existing_item.state_id or
                    self._history_signature(candidate_item.thist) != self._history_signature(existing_item.thist)):
                return None

            candidate_row = self.tagver_table[candidate_item.tvers]
            existing_row = self.tagver_table[existing_item.tvers]
            for tag_idx, (x_ver, y_ver) in enumerate(zip(candidate_row, existing_row)):
                if x_ver == y_ver:
                    continue
                mapped_y = x_to_y.get(x_ver)
                mapped_x = y_to_x.get(y_ver)
                if mapped_y is None and mapped_x is None:
                    x_to_y[x_ver] = y_ver
                    y_to_x[y_ver] = x_ver
                elif mapped_y != y_ver or mapped_x != x_ver:
                    return None

        if not x_to_y:
            return tuple(commands)

        candidate_set_lhs = {
            command.lhs
            for command in commands
            if command.kind in ("set", "add")
        }

        rewritten_commands = tuple(
            TDFACommand(
                kind=command.kind,
                lhs=x_to_y.get(command.lhs, command.lhs),
                rhs=x_to_y.get(command.rhs, command.rhs) if command.rhs else command.rhs,
                history=command.history,
            )
            for command in commands
        )

        copy_commands = [
            TDFACommand(kind="copy", lhs=y_ver, rhs=x_ver)
            for x_ver, y_ver in sorted(x_to_y.items())
            if x_ver != y_ver and x_ver not in candidate_set_lhs
        ]
        ordered_copies, has_cycle = _topsort_copy_commands(copy_commands)
        if has_cycle:
            return None
        return ordered_copies + rewritten_commands

    def _find_or_create_state(self,
                              kernel: Sequence[TDFAKernelItem],
                              commands: Sequence[TDFACommand]) -> Tuple[int, Tuple[TDFACommand, ...], bool]:
        key = self._kernel_key(kernel)
        exact = self.state_map.get(key)
        if exact is not None:
            return exact, tuple(commands), False

        for target_state in self.states:
            mapped = self._try_map_kernel(kernel, target_state, commands)
            if mapped is not None:
                return target_state.id, mapped, False

        state_id = self._add_state(kernel)
        return state_id, tuple(commands), True

    def _reach_on_symbol(self,
                         state: TDFAState,
                         symbol: int) -> Tuple[_ClosItem, ...]:
        transparent = symbol in (SENTINEL_START, SENTINEL_END)
        reach: List[_ClosItem] = []
        for origin, item in enumerate(state.kernel):
            item = state.kernel[origin]
            node = self.tnfa.states[item.state_id]
            if node.kind == "RAN":
                if _matches_symbol(node, symbol):
                    if node.out1 is not None:
                        reach.append(_ClosItem(node.out1, origin, HROOT))
                elif transparent:
                    reach.append(_ClosItem(item.state_id, origin, HROOT))
            elif transparent:
                reach.append(_ClosItem(item.state_id, origin, HROOT))
        return tuple(reach)

    def build(self) -> TDFA:
        initial_closure = self._closure((_ClosItem(self.tnfa.root, 0, HROOT),))
        initial_kernel = self._prune_closure(initial_closure)
        start_kernel, initial_commands = self._materialize_initial_kernel(initial_kernel)
        start_state = self._add_state(start_kernel)

        index = 0
        while index < len(self.states):
            state = self.states[index]
            for class_id, symbol in enumerate(self.class_representatives):
                reach = self._reach_on_symbol(state, symbol)
                closure = self._closure(reach) if reach else ()
                kernel_buffers = self._prune_closure(closure)
                candidate_kernel, commands = self._materialize_transition(
                    state.id,
                    kernel_buffers,
                )
                target_id, edge_commands, _is_new = self._find_or_create_state(
                    candidate_kernel,
                    commands,
                )
                self.transitions[state.id][class_id] = target_id
                if edge_commands:
                    self.transition_commands[(state.id, class_id)] = edge_commands
            index += 1

        dead_state = self.state_map.get(())
        if dead_state is None:
            dead_state = self._add_state(())
        self.transitions[dead_state] = [dead_state] * self.num_classes

        for row in self.transitions:
            for class_id, target in enumerate(row):
                if target < 0:
                    row[class_id] = dead_state

        return TDFA(
            states=tuple(self.states),
            num_states=len(self.states),
            num_classes=self.num_classes,
            start_state=start_state,
            dead_state=dead_state,
            accept_states=frozenset(self.accept_states),
            transitions=tuple(tuple(row) for row in self.transitions),
            class_map=self.class_map,
            class_representatives=self.class_representatives,
            sentinel_256_class=self.sentinel_256_class,
            sentinel_257_class=self.sentinel_257_class,
            tag_names=self.tag_names,
            public_tag_names=self.public_tag_names,
            tag_slots=self.tag_slots,
            initial_commands=initial_commands,
            transition_commands=self.transition_commands,
            accept_commands=self.accept_commands,
            output_registers=self.output_registers,
            register_count=self.maxtagver,
            tagver_rows=tuple(self.tagver_table.rows),
        )


def build_tdfa(tnfa: TNFA, include_internal: bool = False) -> TDFA:
    """Determinize an explicit TNFA into a leftmost TDFA."""
    _ = include_internal  # kept for source compatibility
    return _TDFACompiler(tnfa).build()


def _tnfa_closure(tnfa: TNFA,
                  start: Sequence[_ConcreteConfig],
                  pos: int) -> Tuple[_ConcreteConfig, ...]:
    result: List[_ConcreteConfig] = []
    seen_index: Dict[int, int] = {}
    seen_origin: Dict[Tuple[int, int], Tuple[int, ...]] = {}

    def merge_tag_values(existing: Tuple[int, ...],
                         incoming: Tuple[int, ...]) -> Tuple[Tuple[int, ...], bool]:
        merged = list(existing)
        changed = False
        for slot, value in enumerate(incoming):
            if merged[slot] < 0 and value >= 0:
                merged[slot] = value
                changed = True
        return tuple(merged), changed

    def visit(current: _ConcreteConfig, origin: int) -> None:
        pair_key = (current.state_id, origin)
        existing_pair = seen_origin.get(pair_key)
        if existing_pair is not None:
            merged, changed = merge_tag_values(existing_pair, current.tag_values)
            if not changed:
                return
            current = _ConcreteConfig(current.state_id, merged)
            seen_origin[pair_key] = merged
        else:
            seen_origin[pair_key] = current.tag_values

        idx = seen_index.get(current.state_id)
        if idx is not None:
            state = tnfa.states[current.state_id]
            if state.kind == "FIN":
                merged, changed = merge_tag_values(result[idx].tag_values, current.tag_values)
                if changed:
                    result[idx] = _ConcreteConfig(current.state_id, merged)
        else:
            seen_index[current.state_id] = len(result)
            result.append(current)

        state = tnfa.states[current.state_id]
        if state.kind == "ALT":
            if state.out1 is not None:
                visit(_ConcreteConfig(state.out1, current.tag_values), origin)
            if state.out2 is not None:
                visit(_ConcreteConfig(state.out2, current.tag_values), origin)
        elif state.kind == "TAG":
            next_tags = list(current.tag_values)
            if state.tag is not None:
                next_tags[state.tag.idx] = pos
            if state.out1 is not None:
                visit(_ConcreteConfig(state.out1, tuple(next_tags)), origin)

    for origin, config in enumerate(start):
        visit(config, origin)

    return tuple(result)


def _tnfa_move(tnfa: TNFA,
               configs: Sequence[_ConcreteConfig],
               symbol: int) -> Tuple[_ConcreteConfig, ...]:
    transparent = symbol in (SENTINEL_START, SENTINEL_END)
    moved: List[_ConcreteConfig] = []
    for config in configs:
        state = tnfa.states[config.state_id]
        if state.kind == "RAN":
            if _matches_symbol(state, symbol):
                if state.out1 is not None:
                    moved.append(_ConcreteConfig(state.out1, config.tag_values))
            elif transparent:
                moved.append(config)
        elif transparent:
            moved.append(config)
    return tuple(moved)


def _tnfa_accept_values(tnfa: TNFA,
                        configs: Sequence[_ConcreteConfig]) -> Optional[Tuple[int, ...]]:
    for config in configs:
        if tnfa.states[config.state_id].kind == "FIN":
            return config.tag_values
    return None


def run_tnfa(tnfa: TNFA,
             data: bytes,
             include_internal: bool = False) -> Tuple[bool, Dict[str, int]]:
    """Reference runner over the explicit TNFA."""
    tag_names = tuple(tag.name for tag in tnfa.tags)
    empty_tags = tuple([-1] * len(tag_names))

    configs = _tnfa_closure(
        tnfa,
        (_ConcreteConfig(tnfa.root, empty_tags),),
        pos=0,
    )
    configs = _tnfa_closure(
        tnfa,
        _tnfa_move(tnfa, configs, SENTINEL_START),
        pos=0,
    )

    accept_values = _tnfa_accept_values(tnfa, configs)
    if accept_values is not None:
        return True, _materialize_tag_dict(tag_names, accept_values, include_internal)

    for index, byte_val in enumerate(data):
        configs = _tnfa_move(tnfa, configs, byte_val)
        if configs:
            configs = _tnfa_closure(tnfa, configs, pos=index + 1)
        accept_values = _tnfa_accept_values(tnfa, configs)
        if accept_values is not None:
            return True, _materialize_tag_dict(tag_names, accept_values, include_internal)

    configs = _tnfa_move(tnfa, configs, SENTINEL_END)
    if configs:
        configs = _tnfa_closure(tnfa, configs, pos=len(data))
    accept_values = _tnfa_accept_values(tnfa, configs)
    if accept_values is not None:
        return True, _materialize_tag_dict(tag_names, accept_values, include_internal)

    return False, {}


def _apply_commands(registers: Sequence[int],
                    commands: Sequence[TDFACommand],
                    current_pos: int) -> List[int]:
    if not commands:
        return list(registers)

    next_registers = list(registers)
    for command in commands:
        if command.kind == "copy":
            next_registers[command.lhs] = registers[command.rhs]
        elif command.kind == "set":
            next_registers[command.lhs] = current_pos
        elif command.kind == "add":
            raise NotImplementedError("history-tag add commands are not supported yet")
        else:
            raise ValueError(f"Unknown TDFA command kind: {command.kind}")
    return next_registers


def _accept_values_from_registers(tdfa: TDFA,
                                  registers: Sequence[int]) -> Tuple[int, ...]:
    return tuple(
        registers[reg] if reg < len(registers) else -1
        for reg in tdfa.output_registers
    )


def _maybe_accept(tdfa: TDFA,
                  state_id: int,
                  registers: Sequence[int],
                  current_pos: int) -> Optional[Tuple[int, ...]]:
    if state_id not in tdfa.accept_states:
        return None
    final_registers = _apply_commands(
        registers,
        tdfa.accept_commands.get(state_id, ()),
        current_pos=current_pos,
    )
    return _accept_values_from_registers(tdfa, final_registers)


def run_tdfa(tdfa: TDFA,
             data: bytes,
             include_internal: bool = False) -> Tuple[bool, Dict[str, int]]:
    """Run the canonical TDFA interpreter over input bytes."""
    registers = [-1] * (tdfa.register_count + 1)
    registers = _apply_commands(registers, tdfa.initial_commands, current_pos=0)
    state = tdfa.start_state

    prev_state = state
    state = tdfa.transitions[state][tdfa.sentinel_256_class]
    registers = _apply_commands(
        registers,
        tdfa.transition_commands.get((prev_state, tdfa.sentinel_256_class), ()),
        current_pos=0,
    )
    accepted = _maybe_accept(tdfa, state, registers, 0)
    if accepted is not None:
        return True, _materialize_tag_dict(tdfa.tag_names, accepted, include_internal)

    for index, byte_val in enumerate(data):
        prev_state = state
        class_id = tdfa.class_map[byte_val]
        state = tdfa.transitions[state][class_id]
        registers = _apply_commands(
            registers,
            tdfa.transition_commands.get((prev_state, class_id), ()),
            current_pos=index,
        )
        accepted = _maybe_accept(tdfa, state, registers, index + 1)
        if accepted is not None:
            return True, _materialize_tag_dict(tdfa.tag_names, accepted, include_internal)

    prev_state = state
    state = tdfa.transitions[state][tdfa.sentinel_257_class]
    registers = _apply_commands(
        registers,
        tdfa.transition_commands.get((prev_state, tdfa.sentinel_257_class), ()),
        current_pos=len(data),
    )
    accepted = _maybe_accept(tdfa, state, registers, len(data))
    if accepted is not None:
        return True, _materialize_tag_dict(tdfa.tag_names, accepted, include_internal)

    return False, {}

