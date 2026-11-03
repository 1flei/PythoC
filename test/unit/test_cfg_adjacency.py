"""CFG adjacency invariants and deterministic traversal-cost regressions."""

import ast
import copy
import random
import unittest

from pythoc.ast_visitor.control_flow_builder import ControlFlowBuilder
from pythoc.cfg.graph import CFG, CFGEdge
from pythoc.cfg.linear_checker import LinearChecker


class CountingEdge(CFGEdge):
    reads = 0

    @property
    def source_id(self):
        type(self).reads += 1
        return super().source_id

    @property
    def target_id(self):
        type(self).reads += 1
        return super().target_id


class TestCFGAdjacency(unittest.TestCase):
    def assert_adjacency(self, cfg):
        for block_id in range(9):
            outgoing = [e for e in cfg.edges if e.source_id == block_id]
            incoming = [e for e in cfg.edges if e.target_id == block_id]
            self.assertEqual([id(e) for e in cfg.get_successors(block_id)],
                             [id(e) for e in outgoing])
            self.assertEqual([id(e) for e in cfg.get_predecessors(block_id)],
                             [id(e) for e in incoming])

    def test_constructor_owns_topology(self):
        edges = [CFGEdge(0, 1), CFGEdge(0, 2), CFGEdge(2, 1)]
        cfg = CFG('initial', edges=edges)
        edges.clear()
        self.assertEqual(len(cfg.edges), 3)
        self.assert_adjacency(cfg)
        with self.assertRaises(AttributeError):
            cfg.edges.append(CFGEdge(1, 2))
        with self.assertRaises(TypeError):
            cfg.edges[0] = CFGEdge(0, 3)
        with self.assertRaises(AttributeError):
            cfg.edges = []
        with self.assertRaises(AttributeError):
            cfg.edges[0].target_id = 3
        with self.assertRaises(AttributeError):
            cfg.edges[0].source_id = 3

    def test_duplicate_edges_and_independent_query_results(self):
        cfg = CFG('duplicates')
        first = cfg.add_edge(0, 1)
        second = cfg.add_edge(0, 1)
        cfg.get_successors(0).clear()
        cfg.get_predecessors(1).clear()
        cfg.remove_edge(second)
        self.assertEqual(len(cfg.edges), 1)
        self.assertIs(cfg.edges[0], second)
        self.assertIsNot(cfg.edges[0], first)
        self.assert_adjacency(cfg)
        cfg.remove_edge(CFGEdge(0, 1))
        cfg.remove_edge(CFGEdge(0, 1))
        self.assertEqual(len(cfg.edges), 0)
        self.assertEqual(cfg._incoming, {})
        self.assertEqual(cfg._outgoing, {})

    def test_same_length_edits_and_metadata(self):
        cfg = CFG('edits')
        edge = cfg.add_edge(0, 1)
        cfg.get_predecessors(1)
        cfg.remove_edge(edge)
        replacement = cfg.add_edge(2, 3)
        self.assert_adjacency(cfg)
        tail = cfg.add_edge(4, 3)
        replacement.kind = 'loop_back'
        cfg.compute_loop_headers()
        self.assertEqual(cfg.loop_headers, {3})
        self.assertFalse(cfg.is_merge_point(3))
        redirected = cfg.redirect_edge(replacement, 1, 2)
        self.assertIs(cfg.edges[0], redirected)
        self.assertIs(cfg.edges[1], tail)
        self.assertEqual(redirected.kind, 'loop_back')
        self.assert_adjacency(cfg)

    def test_random_edits_match_scan_oracle(self):
        rng = random.Random(64217)
        cfg = CFG('random')
        for _ in range(500):
            action = rng.randrange(3)
            if action == 0 or not cfg.edges:
                cfg.add_edge(rng.randrange(9), rng.randrange(9))
            elif action == 1:
                cfg.remove_edge(rng.choice(cfg.edges))
            else:
                cfg.redirect_edge(rng.choice(cfg.edges),
                                  rng.randrange(9), rng.randrange(9))
            self.assert_adjacency(cfg)

    def test_copy_and_splice_keep_indexes_consistent(self):
        caller = CFG('caller')
        entry, exit_block = caller.add_block(), caller.add_block()
        caller.entry_id, caller.exit_id = entry.id, exit_block.id
        caller.add_edge(entry.id, exit_block.id)
        callee = CFG('callee')
        callee_entry = callee.add_block(ast.parse('return 7').body)
        callee_exit = callee.add_block()
        callee.entry_id, callee.exit_id = callee_entry.id, callee_exit.id
        callee.return_blocks.append(callee_entry.id)
        callee.add_edge(callee_entry.id, callee_exit.id, 'return')
        caller.splice(entry.id, callee, 'result')
        self.assert_adjacency(caller)
        cloned = copy.deepcopy(caller)
        cloned.remove_edge(cloned.edges[0])
        cloned.add_edge(7, 8)
        self.assert_adjacency(cloned)
        self.assert_adjacency(caller)
        self.assertFalse(caller.get_successors(7))

    def test_adjacency_sweep_does_not_scan_graph_per_query(self):
        cfg = CFG('chain', edges=[CountingEdge(i, i + 1) for i in range(4095)])
        CountingEdge.reads = 0
        for block_id in range(4096):
            self.assertEqual(len(cfg.get_successors(block_id)), int(block_id < 4095))
            self.assertEqual(len(cfg.get_predecessors(block_id)), int(block_id > 0))
        self.assertLessEqual(CountingEdge.reads, 4 * 4096)

    def test_traversal_and_interleaved_finalization_are_linear(self):
        for size in (128, 256, 512):
            with self.subTest(size=size):
                cfg = CFG('fanout', edges=[CountingEdge(0, i) for i in range(1, size - 1)])
                for _ in range(size):
                    cfg.add_block()
                cfg.entry_id, cfg.exit_id = 0, size - 1
                builder = ControlFlowBuilder(None, None, 'fanout')
                builder._cfg = cfg
                builder._exit_block = cfg.blocks[cfg.exit_id]
                CountingEdge.reads = 0
                builder.finalize()
                self.assertFalse(LinearChecker().check(cfg, {}))
                self.assertEqual(len(cfg.get_reachable_blocks()), size)
                self.assertEqual(len(cfg.topological_order()), size)
                self.assertLessEqual(CountingEdge.reads, 20 * size)
                self.assertEqual(len(cfg.edges), 2 * (size - 2))


if __name__ == '__main__':
    unittest.main()
