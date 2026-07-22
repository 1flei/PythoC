"""Effect specialization graph — standalone directed graph for effect propagation.

This module implements a standalone directed graph that tracks effect
specialization propagation. It replaces the BFS-based effect propagation
mechanism in the build pipeline with an incremental, structure-sharing
graph that records import edges and layer memberships as they happen.

Key design properties:
- Two fundamental operations: ``add_edge`` and ``add_layer`` (orthogonal
  and composable — order does not matter).
- Incremental propagation via worklist BFS (no batch ``compute_layers``).
- Structure sharing: nodes stored once, layer membership via sets.
- Thread-safe via ``threading.RLock``.
- No dynamic/static node distinction.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NodeID = Tuple[str, Optional[str], Optional[str]]
"""Identifies a compilation unit: ``(source_file, scope, compile_suffix)``."""

LayerID = Optional[str]
"""Identifies a specialization layer: ``None`` for base, suffix string for effect layers."""

GroupKey = Tuple[str, Optional[str], Optional[str], Optional[str]]
"""Full 4-tuple: ``(source_file, scope, compile_suffix, effect_suffix)`` = NodeID + LayerID."""


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def node_id_from_group_key(gk: GroupKey) -> NodeID:
    """Extract the NodeID (first 3 components) from a GroupKey."""
    return (gk[0], gk[1], gk[2])


def layer_id_from_group_key(gk: GroupKey) -> LayerID:
    """Extract the LayerID (4th component) from a GroupKey."""
    return gk[3]


def group_key_from_parts(nid: NodeID, lid: LayerID) -> GroupKey:
    """Compose a GroupKey from a NodeID and a LayerID."""
    return (nid[0], nid[1], nid[2], lid)


# ---------------------------------------------------------------------------
# EffectGraph
# ---------------------------------------------------------------------------

class EffectGraph:
    """Standalone directed graph for effect specialization propagation.

    The graph maintains two kinds of entities:

    - **Nodes** (NodeID): compilation units identified by
      ``(source_file, scope, compile_suffix)``.
    - **Layers** (LayerID): effect specialization contexts identified by
      a suffix string (or ``None`` for the implicit base layer).

    Two mutation operations change layer membership:

    - ``add_edge(source, target)`` — records a dependency edge and
      propagates all of source's layers to target (and its reachable
      closure).
    - ``add_layer(node, layer)`` — adds a node to a layer and propagates
      the layer through the node's existing edges.

    These operations are orthogonal and composable: ``add_layer(A, L)``
    followed by ``add_edge(A, B)`` produces the same final state as
    ``add_edge(A, B)`` followed by ``add_layer(A, L)``.
    """

    def __init__(self) -> None:
        # Node identity
        self._nodes: Set[NodeID] = set()

        # Edge adjacency (bidirectional)
        self._out: Dict[NodeID, Set[NodeID]] = {}
        self._in: Dict[NodeID, Set[NodeID]] = {}

        # Layer membership (bidirectional)
        self._node_layers: Dict[NodeID, Set[LayerID]] = {}
        self._layer_nodes: Dict[LayerID, Set[NodeID]] = {}
        self._layer_entries: Dict[LayerID, Set[NodeID]] = {}

        # Effect declarations
        self._node_effects: Dict[NodeID, Set[str]] = {}

        # Memoized query cache
        self._effects_cache: Dict[Tuple[NodeID, FrozenSet[str]], bool] = {}

        # Thread safety
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_edge(self, source: NodeID, target: NodeID) -> None:
        """Record a dependency edge ``source → target``.

        All layers that ``source`` currently belongs to are propagated to
        ``target`` and its reachable closure. If ``source == target``,
        the call is a no-op (self-edges are ignored).

        This is called at import time (module A imports a ``@compile``
        function from module B) and at codegen time (codegen of A emits
        a call to a function in B). Both cases use the same API.
        """
        if source == target:
            return
        with self._lock:
            self._nodes.add(source)
            self._nodes.add(target)
            self._out.setdefault(source, set()).add(target)
            self._in.setdefault(target, set()).add(source)
            self._node_layers.setdefault(source, set())
            self._node_layers.setdefault(target, set())
            # Propagate all of source's layers to target.
            # _propagate_layer only extends to seed's outgoing neighbors,
            # so we must add target to the layer first, then propagate.
            for layer in list(self._node_layers[source]):
                if layer not in self._node_layers[target]:
                    self._node_layers[target].add(layer)
                    self._layer_nodes.setdefault(layer, set()).add(target)
                    self._propagate_layer(layer, target)
            self._effects_cache.clear()

    def add_layer(self, node: NodeID, layer: LayerID) -> None:
        """Add ``node`` to ``layer``, propagating through existing edges.

        If the node is already in the layer, this is a no-op (aside from
        cache invalidation). Otherwise the node joins the layer as an
        *entry point*, and the layer propagates through the node's
        outgoing edges to all reachable nodes.
        """
        with self._lock:
            self._nodes.add(node)
            self._node_layers.setdefault(node, set())
            self._layer_nodes.setdefault(layer, set())
            self._layer_entries.setdefault(layer, set())
            if layer not in self._node_layers[node]:
                self._node_layers[node].add(layer)
                self._layer_nodes[layer].add(node)
                self._layer_entries[layer].add(node)
                self._propagate_layer(layer, node)
            self._effects_cache.clear()

    def set_node_effects(self, node: NodeID, effect_names: Set[str]) -> None:
        """Declare the set of effect names used by ``node``'s code.

        Used by ``transitive_uses_effects`` to prune unnecessary
        specialization. Each call replaces the previous effect set for
        the node (set semantics, not augment).
        """
        with self._lock:
            self._nodes.add(node)
            self._node_effects[node] = set(effect_names)
            self._effects_cache.clear()

    # ------------------------------------------------------------------
    # Propagation (internal)
    # ------------------------------------------------------------------

    def _propagate_layer(self, layer: LayerID, seed: NodeID) -> None:
        """Propagate ``layer`` from ``seed`` through outgoing edges.

        Uses a worklist BFS that only visits newly discovered nodes
        (nodes already in the layer are skipped). This makes the cost
        proportional to the number of *new* nodes discovered, not the
        total graph size.

        Must be called with ``self._lock`` held.
        """
        worklist: deque[NodeID] = deque([seed])
        while worklist:
            node = worklist.popleft()
            for target in self._out.get(node, ()):
                target_layers = self._node_layers.setdefault(target, set())
                if layer not in target_layers:
                    target_layers.add(layer)
                    self._layer_nodes.setdefault(layer, set()).add(target)
                    worklist.append(target)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_specializations(self, node: NodeID) -> Set[LayerID]:
        """Return the set of layers that ``node`` belongs to."""
        with self._lock:
            return self._node_layers.get(node, set()).copy()

    def get_layer_members(self, layer: LayerID) -> Set[NodeID]:
        """Return the set of all nodes in ``layer`` (including propagated)."""
        with self._lock:
            return self._layer_nodes.get(layer, set()).copy()

    def get_layer_entries(self, layer: LayerID) -> Set[NodeID]:
        """Return the set of entry-point nodes for ``layer``.

        Entry points are nodes explicitly added via ``add_layer``, as
        opposed to nodes that joined the layer via propagation.
        """
        with self._lock:
            return self._layer_entries.get(layer, set()).copy()

    def get_all_layers(self) -> Set[LayerID]:
        """Return the set of all layers that exist in the graph."""
        with self._lock:
            return set(self._layer_nodes.keys())

    def get_neighbors(self, node: NodeID) -> Set[NodeID]:
        """Return the set of nodes that ``node`` depends on (outgoing edges)."""
        with self._lock:
            return self._out.get(node, set()).copy()

    def get_reverse_neighbors(self, node: NodeID) -> Set[NodeID]:
        """Return the set of nodes that depend on ``node`` (incoming edges)."""
        with self._lock:
            return self._in.get(node, set()).copy()

    def has_node(self, node: NodeID) -> bool:
        """Return True if ``node`` has been registered in the graph."""
        with self._lock:
            return node in self._nodes

    def has_effects_declared(self, node: NodeID) -> bool:
        """Return True if ``set_node_effects`` has been called for ``node``.

        Callers use this to distinguish "node exists but effects not yet
        declared" (conservative: assume it uses effects) from "node exists
        and effects declared as empty" (definitive: does not use effects).
        """
        with self._lock:
            return node in self._node_effects

    def get_all_nodes(self) -> Set[NodeID]:
        """Return the set of all registered nodes."""
        with self._lock:
            return self._nodes.copy()

    def get_topological_order(self, layer: LayerID) -> List[NodeID]:
        """Return nodes in ``layer`` in topological order (dependencies first).

        Handles cycles by skipping back-edges (nodes currently on the
        DFS temp-mark stack are not re-visited). The result contains all
        layer members; the exact order among independent nodes depends on
        iteration order.
        """
        with self._lock:
            members = self._layer_nodes.get(layer, set())
            result: List[NodeID] = []
            visited: Set[NodeID] = set()
            temp: Set[NodeID] = set()

            def visit(n: NodeID) -> None:
                if n in visited or n in temp:
                    return
                temp.add(n)
                for t in self._out.get(n, ()):
                    if t in members:
                        visit(t)
                temp.discard(n)
                visited.add(n)
                result.append(n)

            for n in members:
                visit(n)
            return result

    def transitive_uses_effects(
        self, node: NodeID, effects: FrozenSet[str]
    ) -> bool:
        """Return True if ``node`` or any reachable node uses any of ``effects``.

        Results are memoized in ``_effects_cache``. The cache is cleared
        on any mutation (``add_edge``, ``add_layer``, ``set_node_effects``).
        """
        key = (node, effects)
        with self._lock:
            if key in self._effects_cache:
                return self._effects_cache[key]
            visited: Set[NodeID] = set()
            queue: deque[NodeID] = deque([node])
            result = False
            while queue:
                n = queue.popleft()
                if n in visited:
                    continue
                visited.add(n)
                if self._node_effects.get(n, set()) & effects:
                    result = True
                    break
                for t in self._out.get(n, ()):
                    queue.append(t)
            self._effects_cache[key] = result
            return result

    def get_specialization_path(
        self, source: NodeID, target: NodeID, layer: LayerID
    ) -> Optional[List[NodeID]]:
        """Return the shortest path from ``source`` to ``target`` within ``layer``.

        Returns ``None`` if no path exists or either node is not in the
        layer. Uses BFS for shortest-path guarantee.
        """
        with self._lock:
            members = self._layer_nodes.get(layer, set())
            if source not in members or target not in members:
                return None
            queue: deque[Tuple[NodeID, List[NodeID]]] = deque([(source, [source])])
            visited: Set[NodeID] = {source}
            while queue:
                node, path = queue.popleft()
                if node == target:
                    return path
                for t in self._out.get(node, ()):
                    if t in members and t not in visited:
                        visited.add(t)
                        queue.append((t, path + [t]))
            return None

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary for debugging/logging."""
        with self._lock:
            return {
                "nodes": len(self._nodes),
                "edges": sum(len(v) for v in self._out.values()),
                "layers": {str(l): len(v) for l, v in self._layer_nodes.items()},
                "entry_points": {
                    str(l): len(v) for l, v in self._layer_entries.items()
                },
                "effects_declared": len(self._node_effects),
                "cache_size": len(self._effects_cache),
            }

    def dump_dot(self, layer: Optional[LayerID] = None) -> str:
        """Return a Graphviz DOT representation of the graph.

        If ``layer`` is specified, only nodes and edges within that layer
        are included. Otherwise the full graph is dumped.
        """
        with self._lock:
            lines = ["digraph EffectGraph {"]
            if layer is not None:
                members = self._layer_nodes.get(layer, set())
                for n in members:
                    ls = self._node_layers.get(n, set())
                    lines.append(f'  "{n}" [label="{n}\\nlayers={ls}"];')
                for n in members:
                    for t in self._out.get(n, ()):
                        if t in members:
                            lines.append(f'  "{n}" -> "{t}";')
            else:
                for n in self._nodes:
                    ls = self._node_layers.get(n, set())
                    lines.append(f'  "{n}" [label="{n}\\nlayers={ls}"];')
                for n in self._nodes:
                    for t in self._out.get(n, ()):
                        lines.append(f'  "{n}" -> "{t}";')
            lines.append("}")
            return "\n".join(lines)

    def validate(self) -> List[str]:
        """Return a list of consistency issues (empty list means valid).

        Checks:
        - Bidirectional consistency between ``_node_layers`` and ``_layer_nodes``.
        - No self-edges.
        - All layer members reachable from layer entry points (no orphans).
        """
        with self._lock:
            issues: List[str] = []
            # Check _node_layers / _layer_nodes bidirectional consistency
            for node, layers in self._node_layers.items():
                for layer in layers:
                    if node not in self._layer_nodes.get(layer, set()):
                        issues.append(
                            f"_node_layers[{node}] has {layer} "
                            f"but _layer_nodes[{layer}] missing {node}"
                        )
            for layer, nodes in self._layer_nodes.items():
                for node in nodes:
                    if layer not in self._node_layers.get(node, set()):
                        issues.append(
                            f"_layer_nodes[{layer}] has {node} "
                            f"but _node_layers[{node}] missing {layer}"
                        )
            # Check no self-edges
            for node, targets in self._out.items():
                if node in targets:
                    issues.append(f"Self-edge: {node} -> {node}")
            # Check all layer members reachable from entries
            for layer, nodes in self._layer_nodes.items():
                entries = self._layer_entries.get(layer, set())
                if entries:
                    reachable: Set[NodeID] = set()
                    queue: deque[NodeID] = deque(entries)
                    while queue:
                        n = queue.popleft()
                        if n in reachable:
                            continue
                        reachable.add(n)
                        for t in self._out.get(n, ()):
                            if t in nodes:
                                queue.append(t)
                    orphans = nodes - reachable
                    if orphans:
                        issues.append(
                            f"Layer {layer} has orphan nodes "
                            f"(not reachable from entries): {orphans}"
                        )
            return issues

    # ------------------------------------------------------------------
    # Equality (for testing)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Compare two EffectGraph instances by structural equality.

        Two graphs are equal if they have the same nodes, edges, layer
        memberships, layer entries, and effect declarations. The effects
        cache and lock are not compared.
        """
        if not isinstance(other, EffectGraph):
            return NotImplemented
        with self._lock:
            return (
                self._nodes == other._nodes
                and self._out == other._out
                and self._in == other._in
                and self._node_layers == other._node_layers
                and self._layer_nodes == other._layer_nodes
                and self._layer_entries == other._layer_entries
                and self._node_effects == other._node_effects
            )

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"EffectGraph(nodes={len(self._nodes)}, "
                f"edges={sum(len(v) for v in self._out.values())}, "
                f"layers={len(self._layer_nodes)})"
            )
