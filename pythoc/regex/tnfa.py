"""
Explicit TNFA for the tagged regex frontend.

This module is the single entry point from parse-time AST to automata
land. It builds a tagged NFA whose structure is close to re2c's TNFA,
which the leftmost TDFA frontend in ``tdfa.py`` then determinizes. The
result is the canonical control automaton consumed by the T-BMA builder
and codegen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import parse as ast


INTERNAL_TAG_PREFIX = "__pythoc_internal_"
SENTINEL_START = 256
SENTINEL_END = 257


@dataclass(frozen=True)
class TagInfo:
    idx: int
    neg: int = 0


@dataclass(frozen=True)
class TNFATag:
    name: str
    internal: bool = False
    history: bool = False
    fixed: bool = False


@dataclass(frozen=True)
class TNFARule:
    ltag: int
    htag: int
    ttag: int = 0
    ncap: int = 0


@dataclass
class TNFAState:
    id: int
    kind: str  # "ALT", "RAN", "TAG", "FIN"
    rule: int
    out1: Optional[int] = None
    out2: Optional[int] = None
    ranges: Tuple[Tuple[int, int], ...] = ()
    tag: Optional[TagInfo] = None
    eof: bool = False


@dataclass(frozen=True)
class TNFA:
    states: Tuple[TNFAState, ...]
    root: int
    rules: Tuple[TNFARule, ...]
    tags: Tuple[TNFATag, ...]
    ncores: int

    @property
    def accept(self) -> int:
        for state in self.states:
            if state.kind == "FIN":
                return state.id
        raise ValueError("TNFA has no final state")

    def state_count(self) -> int:
        return len(self.states)


def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    if not ranges:
        return ()

    merged: List[Tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _complement_ranges(ranges: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    merged = _merge_ranges(ranges)
    result: List[Tuple[int, int]] = []
    current = 0
    for lo, hi in merged:
        if current < lo:
            result.append((current, lo - 1))
        current = hi + 1
    if current <= 255:
        result.append((current, 255))
    return tuple(result)


def _collect_tag_names(ast_node) -> Tuple[str, ...]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def walk(node) -> None:
        if isinstance(node, ast.Tag):
            if node.name not in seen:
                seen.add(node.name)
                ordered.append(node.name)
            return
        if isinstance(node, ast.Group):
            walk(node.child)
            return
        if isinstance(node, ast.Repeat):
            walk(node.child)
            return
        if isinstance(node, (ast.Concat, ast.Alternate)):
            for child in node.children:
                walk(child)

    walk(ast_node)
    return tuple(ordered)


class TNFABuilder:
    """Build an explicit TNFA from the regex AST."""

    def __init__(self, tag_names: Sequence[str]):
        self.states: List[TNFAState] = []
        self.tag_slots = {
            name: idx
            for idx, name in enumerate(tag_names)
        }
        self.tags = tuple(
            TNFATag(
                name=name,
                internal=name.startswith(INTERNAL_TAG_PREFIX),
            )
            for name in tag_names
        )

    def _new_state(self,
                   kind: str,
                   *,
                   rule: int = 0,
                   out1: Optional[int] = None,
                   out2: Optional[int] = None,
                   ranges: Sequence[Tuple[int, int]] = (),
                   tag: Optional[TagInfo] = None,
                   eof: bool = False) -> int:
        sid = len(self.states)
        self.states.append(TNFAState(
            id=sid,
            kind=kind,
            rule=rule,
            out1=out1,
            out2=out2,
            ranges=tuple(ranges),
            tag=tag,
            eof=eof,
        ))
        return sid

    def _make_alt(self, out1: int, out2: int, rule: int = 0) -> int:
        return self._new_state("ALT", rule=rule, out1=out1, out2=out2)

    def _make_ran(self,
                  out1: int,
                  ranges: Sequence[Tuple[int, int]],
                  rule: int = 0) -> int:
        return self._new_state("RAN", rule=rule, out1=out1, ranges=ranges)

    def _make_tag(self, out1: int, tag: TagInfo, rule: int = 0) -> int:
        return self._new_state("TAG", rule=rule, out1=out1, tag=tag)

    def _make_fin(self, rule: int = 0) -> int:
        return self._new_state("FIN", rule=rule, eof=False)

    def build(self, ast_node) -> TNFA:
        accept = self._make_fin(rule=0)
        root = self._build_node(ast_node, accept, rule=0)
        return TNFA(
            states=tuple(self.states),
            root=root,
            rules=(TNFARule(ltag=0, htag=len(self.tags)),),
            tags=self.tags,
            ncores=self._count_cores(root),
        )

    def _build_node(self, node, end: int, rule: int) -> int:
        if isinstance(node, ast.Literal):
            return self._make_ran(end, ((node.byte, node.byte),), rule=rule)
        if isinstance(node, ast.Dot):
            return self._make_ran(end, ((0, 255),), rule=rule)
        if isinstance(node, ast.CharClass):
            ranges = _merge_ranges(node.ranges)
            if node.negated:
                ranges = _complement_ranges(ranges)
            return self._make_ran(end, ranges, rule=rule)
        if isinstance(node, ast.Concat):
            current_end = end
            for child in reversed(node.children):
                current_end = self._build_node(child, current_end, rule=rule)
            return current_end
        if isinstance(node, ast.Alternate):
            starts = [
                self._build_node(child, end, rule=rule)
                for child in node.children
            ]
            current = starts[-1]
            for start in reversed(starts[:-1]):
                current = self._make_alt(start, current, rule=rule)
            return current
        if isinstance(node, ast.Group):
            return self._build_node(node.child, end, rule=rule)
        if isinstance(node, ast.Tag):
            tag_slot = self.tag_slots[node.name]
            return self._make_tag(end, TagInfo(idx=tag_slot, neg=0), rule=rule)
        if isinstance(node, ast.Anchor):
            symbol = SENTINEL_START if node.kind == "start" else SENTINEL_END
            return self._make_ran(end, ((symbol, symbol),), rule=rule)
        if isinstance(node, ast.Repeat):
            return self._build_repeat(node, end, rule=rule)
        raise ValueError(f"Unknown AST node type: {type(node)}")

    def _build_repeat(self, node: ast.Repeat, end: int, rule: int) -> int:
        if node.min_count == 0 and node.max_count == 1:
            return self._build_optional(node.child, end, rule=rule, lazy=node.lazy)
        if node.min_count == 0 and node.max_count is None:
            return self._build_star(node.child, end, rule=rule, lazy=node.lazy)
        if node.min_count == 1 and node.max_count is None:
            return self._build_plus(node.child, end, rule=rule, lazy=node.lazy)

        children: List[object] = []
        for _ in range(node.min_count):
            children.append(node.child)
        if node.max_count is not None:
            for _ in range(node.max_count - node.min_count):
                children.append(ast.Repeat(
                    child=node.child,
                    min_count=0,
                    max_count=1,
                    lazy=node.lazy,
                ))
        else:
            children.append(ast.Repeat(
                child=node.child,
                min_count=0,
                max_count=None,
                lazy=node.lazy,
            ))
        return self._build_node(ast.Concat(children=children), end, rule=rule)

    def _build_optional(self, child, end: int, rule: int, lazy: bool) -> int:
        child_start = self._build_node(child, end, rule=rule)
        return (
            self._make_alt(end, child_start, rule=rule)
            if lazy else
            self._make_alt(child_start, end, rule=rule)
        )

    def _build_star(self, child, end: int, rule: int, lazy: bool) -> int:
        loop = self._new_state("ALT", rule=rule)
        child_start = self._build_node(child, loop, rule=rule)
        loop_state = self.states[loop]
        if lazy:
            loop_state.out1 = end
            loop_state.out2 = child_start
        else:
            loop_state.out1 = child_start
            loop_state.out2 = end
        return loop

    def _build_plus(self, child, end: int, rule: int, lazy: bool) -> int:
        loop = self._new_state("ALT", rule=rule)
        child_start = self._build_node(child, loop, rule=rule)
        loop_state = self.states[loop]
        if lazy:
            loop_state.out1 = end
            loop_state.out2 = child_start
        else:
            loop_state.out1 = child_start
            loop_state.out2 = end
        return child_start

    def _count_cores(self, root: int) -> int:
        seen: Set[int] = set()
        stack = [root]
        ncores = 0
        while stack:
            sid = stack.pop()
            if sid in seen:
                continue
            seen.add(sid)
            state = self.states[sid]
            if state.kind in ("RAN", "FIN"):
                ncores += 1
            if state.out1 is not None:
                stack.append(state.out1)
            if state.out2 is not None:
                stack.append(state.out2)
        return ncores


def build_tnfa(ast_node) -> TNFA:
    """Build an explicit TNFA from a regex AST node."""
    builder = TNFABuilder(_collect_tag_names(ast_node))
    return builder.build(ast_node)

