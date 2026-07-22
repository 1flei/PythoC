"""Unit tests for EffectGraph — standalone directed graph for effect propagation.

Tests use synthetic NodeID tuples (e.g. ("A", None, None)) and LayerID
strings. No PythoC imports are needed.

Test categories (per design doc §10):
  1. Basic Edge and Layer Operations
  2. Composability (order independence)
  3. Multiple Layers
  4. Cycle Handling
  5. Diamond / Branch and Merge
  6. Effect Pruning
  7. Late Edge Addition (codegen-time)
  8. Realistic Scenarios
  9. Validation and Debugging
"""

import unittest

from pythoc.effect_graph import EffectGraph

# Convenience constants — short aliases for common NodeIDs
A = ("A", None, None)
B = ("B", None, None)
C = ("C", None, None)
D = ("D", None, None)


class TestBasicOperations(unittest.TestCase):
    """Category 1: Basic edge and layer operations."""

    def test_add_edge_no_layers(self):
        """add_edge with no layers: edge recorded, no propagation."""
        g = EffectGraph()
        g.add_edge(A, B)
        self.assertEqual(g.get_neighbors(A), {B})
        self.assertEqual(g.get_reverse_neighbors(B), {A})
        # Nodes are registered but no effect layers yet
        self.assertEqual(g.list_specializations(A), set())
        self.assertEqual(g.list_specializations(B), set())

    def test_add_layer_propagates(self):
        """add_layer propagates through existing edges."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_layer(A, "L1")
        self.assertEqual(
            g.get_layer_members("L1"), {A, B, C}
        )

    def test_add_edge_propagates_existing_layers(self):
        """add_edge after add_layer: target joins source's layers."""
        g = EffectGraph()
        g.add_layer(A, "L1")
        g.add_edge(A, B)
        self.assertIn("L1", g.list_specializations(B))


class TestComposability(unittest.TestCase):
    """Category 2: add_layer + add_edge == add_edge + add_layer."""

    def test_layer_then_edge_equals_edge_then_layer(self):
        """Order of add_layer and add_edge does not matter."""
        g1 = EffectGraph()
        g1.add_layer(A, "L1")
        g1.add_edge(A, B)

        g2 = EffectGraph()
        g2.add_edge(A, B)
        g2.add_layer(A, "L1")

        self.assertEqual(g1.get_layer_members("L1"), g2.get_layer_members("L1"))
        self.assertEqual(
            g1.list_specializations(B), g2.list_specializations(B)
        )
        self.assertEqual(g1, g2)


class TestMultipleLayers(unittest.TestCase):
    """Category 3: Independent layer propagation."""

    def test_multiple_layers_independent_propagation(self):
        """Different layers propagate independently."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_edge(C, D)

        g.add_layer(A, "L1")  # L1 = {A, B, C, D}
        g.add_layer(B, "L2")  # L2 = {B, C, D} (not A)

        self.assertEqual(g.get_layer_members("L1"), {A, B, C, D})
        self.assertEqual(g.get_layer_members("L2"), {B, C, D})
        self.assertEqual(g.list_specializations(A), {"L1"})
        self.assertEqual(g.list_specializations(B), {"L1", "L2"})


class TestCycleHandling(unittest.TestCase):
    """Category 4: Cycle handling — termination and topo order."""

    def test_cycle_does_not_infinite_loop(self):
        """A → B → A cycle: add_layer propagates without infinite recursion."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, A)  # cycle
        g.add_layer(A, "L1")
        self.assertEqual(g.get_layer_members("L1"), {A, B})

    def test_topological_order_with_cycle(self):
        """Topological order handles cycles by skipping back-edges."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_edge(C, A)  # cycle
        g.add_layer(A, "L1")
        order = g.get_topological_order("L1")
        self.assertEqual(len(order), 3)
        # All 3 nodes present; exact order depends on DFS entry point


class TestDiamond(unittest.TestCase):
    """Category 5: Diamond / branch and merge."""

    def test_diamond_propagation(self):
        """A → B → D, A → C → D: all nodes in layer."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(A, C)
        g.add_edge(B, D)
        g.add_edge(C, D)
        g.add_layer(A, "L1")
        self.assertEqual(g.get_layer_members("L1"), {A, B, C, D})

    def test_diamond_topological_order(self):
        """Diamond: D before B and C, B and C before A (dependencies first).

        Edge A→B means A depends on B, so B must be materialized first.
        Topological order (dependencies first): D, then B/C, then A.
        """
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(A, C)
        g.add_edge(B, D)
        g.add_edge(C, D)
        g.add_layer(A, "L1")
        order = g.get_topological_order("L1")
        self.assertLess(order.index(D), order.index(B))
        self.assertLess(order.index(D), order.index(C))
        self.assertLess(order.index(B), order.index(A))
        self.assertLess(order.index(C), order.index(A))


class TestEffectPruning(unittest.TestCase):
    """Category 6: Effect pruning via transitive_uses_effects."""

    def test_transitive_uses_effects(self):
        """transitive_uses_effects returns True when effect is reachable."""
        g = EffectGraph()
        g.set_node_effects(C, {"rng"})
        g.add_edge(A, B)
        g.add_edge(B, C)
        self.assertTrue(g.transitive_uses_effects(A, frozenset({"rng"})))
        self.assertFalse(g.transitive_uses_effects(A, frozenset({"io"})))

    def test_transitive_uses_effects_cache_invalidation(self):
        """Cache is invalidated on add_edge."""
        g = EffectGraph()
        g.set_node_effects(C, {"rng"})
        g.add_edge(A, B)
        # C not reachable yet
        self.assertFalse(g.transitive_uses_effects(A, frozenset({"rng"})))
        g.add_edge(B, C)  # now reachable
        self.assertTrue(g.transitive_uses_effects(A, frozenset({"rng"})))


class TestLateEdges(unittest.TestCase):
    """Category 7: Late edge addition (codegen-time)."""

    def test_late_edge_propagates_existing_layers(self):
        """Edge added after add_layer: target joins existing layers."""
        g = EffectGraph()
        g.add_layer(A, "L1")
        # Simulate codegen-time edge discovery
        g.add_edge(A, B)
        self.assertIn("L1", g.list_specializations(B))

    def test_late_edge_with_deep_closure(self):
        """Late edge discovers a chain: A→L1, then add_edge(A→B), B→C→D."""
        g = EffectGraph()
        g.add_layer(A, "L1")
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_edge(C, D)
        self.assertEqual(g.get_layer_members("L1"), {A, B, C, D})


class TestRealisticScenarios(unittest.TestCase):
    """Category 8: Realistic scenarios."""

    def test_typed_task_scenario(self):
        """Simulate _TypedTask: 3 @compile functions, each spawning 5 dynamic functions."""
        g = EffectGraph()
        test_file = ("test_file", None, None)
        future = ("future", None, None)
        runtime = ("runtime", None, None)
        trampoline = ("spawn_typed_trampoline", None, None)

        # Registration phase: 3 base functions
        g.set_node_effects(test_file, set())
        g.set_node_effects(future, {"spawn"})
        g.set_node_effects(runtime, set())

        # Import-time edges
        g.add_edge(test_file, future)
        g.add_edge(test_file, runtime)

        # Effect specialization
        g.add_layer(test_file, "mock")
        self.assertEqual(g.get_layer_members("mock"), {test_file, future, runtime})

        # Codegen-time: _TypedTask creates 5 new functions
        for i in range(5):
            node = (f"spawn_typed_{i}", None, None)
            g.add_edge(trampoline, node)
            g.add_edge(node, test_file)  # calls add_pair
            g.add_edge(node, runtime)    # calls runtime_spawn

        # trampoline was not in mock layer, so its children are not either
        # unless trampoline itself is specialized. Let's specialize it.
        g.add_layer(trampoline, "mock")
        # All new nodes should now be in "mock" layer
        for i in range(5):
            self.assertIn("mock", g.list_specializations((f"spawn_typed_{i}", None, None)))

    def test_multi_effect_scenario(self):
        """Two effects on same module: both layers coexist."""
        g = EffectGraph()
        app = ("app", None, None)
        lib = ("lib", None, None)
        g.add_edge(app, lib)
        g.add_layer(app, "mock")
        g.add_layer(app, "tracked")
        self.assertEqual(g.list_specializations(lib), {"mock", "tracked"})
        self.assertEqual(len(g.get_layer_members("mock")), 2)
        self.assertEqual(len(g.get_layer_members("tracked")), 2)


class TestValidationDebugging(unittest.TestCase):
    """Category 9: Validation and debugging."""

    def test_validate_clean_graph(self):
        """A well-formed graph has no validation issues."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(A, "L1")
        self.assertEqual(g.validate(), [])

    def test_dump_dot_contains_all_nodes(self):
        """dump_dot includes all nodes and layer info."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(A, "L1")
        dot = g.dump_dot()
        self.assertIn("A", dot)
        self.assertIn("B", dot)
        self.assertIn("L1", dot)


class TestAdditionalCoverage(unittest.TestCase):
    """Additional edge cases beyond the 17 design-doc tests."""

    def test_self_edge_ignored(self):
        """add_edge with source == target is a no-op."""
        g = EffectGraph()
        g.add_edge(A, A)
        self.assertFalse(g.has_node(A))  # node never registered

    def test_add_layer_idempotent(self):
        """Adding the same layer twice is a no-op."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(A, "L1")
        members_before = g.get_layer_members("L1")
        entries_before = g.get_layer_entries("L1")
        g.add_layer(A, "L1")  # idempotent
        self.assertEqual(g.get_layer_members("L1"), members_before)
        self.assertEqual(g.get_layer_entries("L1"), entries_before)

    def test_layer_entries_vs_members(self):
        """Entry points are a subset of layer members."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_layer(A, "L1")
        entries = g.get_layer_entries("L1")
        members = g.get_layer_members("L1")
        self.assertTrue(entries.issubset(members))
        self.assertEqual(entries, {A})  # only A is an entry point
        self.assertEqual(members, {A, B, C})

    def test_get_all_layers(self):
        """get_all_layers returns all layer IDs including None if used."""
        g = EffectGraph()
        g.add_layer(A, "L1")
        g.add_layer(B, "L2")
        self.assertEqual(g.get_all_layers(), {"L1", "L2"})

    def test_get_all_nodes(self):
        """get_all_nodes returns every registered node."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(C, "L1")
        self.assertEqual(g.get_all_nodes(), {A, B, C})

    def test_has_node(self):
        """has_node returns True for registered nodes, False otherwise."""
        g = EffectGraph()
        g.add_edge(A, B)
        self.assertTrue(g.has_node(A))
        self.assertTrue(g.has_node(B))
        self.assertFalse(g.has_node(C))

    def test_specialization_path_found(self):
        """get_specialization_path returns shortest path within a layer."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, D)
        g.add_edge(A, C)
        g.add_edge(C, D)
        g.add_layer(A, "L1")
        path = g.get_specialization_path(A, D, "L1")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], A)
        self.assertEqual(path[-1], D)
        self.assertEqual(len(path), 3)  # shortest: A → B → D or A → C → D

    def test_specialization_path_not_found(self):
        """get_specialization_path returns None when no path exists."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(C, D)
        g.add_layer(A, "L1")
        g.add_layer(C, "L1")
        # A and D are both in L1 but no path A → D
        path = g.get_specialization_path(A, D, "L1")
        self.assertIsNone(path)

    def test_specialization_path_node_not_in_layer(self):
        """Returns None if source or target not in the layer."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(A, "L1")
        # B is in L1 via propagation, but C is not in the graph at all
        path = g.get_specialization_path(A, C, "L1")
        self.assertIsNone(path)

    def test_set_node_effects_replaces(self):
        """set_node_effects replaces, not augments."""
        g = EffectGraph()
        g.set_node_effects(A, {"rng", "io"})
        g.set_node_effects(A, {"rng"})  # replace
        self.assertEqual(g._node_effects[A], {"rng"})

    def test_transitive_uses_effects_self(self):
        """Node itself uses effects: returns True without traversal."""
        g = EffectGraph()
        g.set_node_effects(A, {"rng"})
        self.assertTrue(g.transitive_uses_effects(A, frozenset({"rng"})))

    def test_transitive_uses_effects_no_effects_declared(self):
        """No effects declared anywhere: returns False."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        self.assertFalse(g.transitive_uses_effects(A, frozenset({"rng"})))

    def test_topological_order_empty_layer(self):
        """Topological order of non-existent layer is empty list."""
        g = EffectGraph()
        self.assertEqual(g.get_topological_order("nonexistent"), [])

    def test_topological_order_single_node(self):
        """Single node layer: order is just that node."""
        g = EffectGraph()
        g.add_layer(A, "L1")
        self.assertEqual(g.get_topological_order("L1"), [A])

    def test_dump_dot_filtered_by_layer(self):
        """dump_dot(layer=...) only includes nodes in that layer."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_layer(A, "L1")
        dot_full = g.dump_dot()
        dot_l1 = g.dump_dot("L1")
        # All nodes are in L1 via propagation
        self.assertIn("A", dot_l1)
        self.assertIn("B", dot_l1)
        self.assertIn("C", dot_l1)
        # Full dump should also contain all
        self.assertIn("A", dot_full)

    def test_summary(self):
        """summary returns expected counts."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_layer(A, "L1")
        g.set_node_effects(A, {"rng"})
        s = g.summary()
        self.assertEqual(s["nodes"], 3)
        self.assertEqual(s["edges"], 2)
        self.assertEqual(s["layers"], {"L1": 3})
        self.assertEqual(s["entry_points"], {"L1": 1})
        self.assertEqual(s["effects_declared"], 1)

    def test_validate_detects_orphan(self):
        """validate detects orphan nodes not reachable from entries.

        This is a synthetic scenario: normally orphans can't happen because
        add_layer always propagates. But if we manually corrupt the state,
        validate should catch it.
        """
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(A, "L1")
        # Manually add C to L1 without proper propagation path
        g._layer_nodes["L1"].add(C)
        g._node_layers.setdefault(C, set()).add("L1")
        issues = g.validate()
        self.assertTrue(any("orphan" in i.lower() for i in issues))

    def test_repr(self):
        """__repr__ produces a useful string."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(A, "L1")
        r = repr(g)
        self.assertIn("EffectGraph", r)
        self.assertIn("nodes=2", r)

    def test_isolated_node_via_add_layer(self):
        """add_layer on a node with no edges: node is in layer, no propagation."""
        g = EffectGraph()
        g.add_layer(A, "L1")
        self.assertEqual(g.get_layer_members("L1"), {A})
        self.assertEqual(g.get_layer_entries("L1"), {A})

    def test_chain_topological_order(self):
        """Linear chain A→B→C→D: topo order is reverse of edge direction."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_edge(B, C)
        g.add_edge(C, D)
        g.add_layer(A, "L1")
        order = g.get_topological_order("L1")
        self.assertEqual(order, [D, C, B, A])

    def test_late_edge_extends_existing_layer(self):
        """Late edge extends an existing layer's membership."""
        g = EffectGraph()
        g.add_edge(A, B)
        g.add_layer(A, "L1")
        # Now add a new edge from B to a new node C
        g.add_edge(B, C)
        self.assertIn("L1", g.list_specializations(C))
        self.assertIn(C, g.get_layer_members("L1"))

    def test_multi_entry_same_layer(self):
        """Multiple entry points in the same layer."""
        g = EffectGraph()
        g.add_edge(A, C)
        g.add_edge(B, C)
        g.add_layer(A, "L1")
        g.add_layer(B, "L1")
        self.assertEqual(g.get_layer_entries("L1"), {A, B})
        self.assertEqual(g.get_layer_members("L1"), {A, B, C})


if __name__ == "__main__":
    unittest.main()
