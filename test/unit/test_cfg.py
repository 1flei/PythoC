"""
Unit tests for CFG module

Tests:
- CFG data structures (graph.py)
- Linear snapshot utilities (linear_checker.py)
"""

import unittest
import ast
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc.cfg.graph import CFGBlock, CFGEdge, CFG
from pythoc.cfg.linear_checker import (
    LinearSnapshot,
    LinearError,
    copy_snapshot,
    snapshots_compatible,
    find_snapshot_diffs,
)


class TestCFGBlock(unittest.TestCase):
    """Tests for CFGBlock"""
    
    def test_create_empty_block(self):
        block = CFGBlock(id=0)
        self.assertEqual(block.id, 0)
        self.assertEqual(block.stmts, [])
        self.assertTrue(block.is_empty())
    
    def test_create_block_with_stmts(self):
        stmt = ast.parse("x = 1").body[0]
        block = CFGBlock(id=1, stmts=[stmt])
        self.assertEqual(block.id, 1)
        self.assertEqual(len(block.stmts), 1)
        self.assertFalse(block.is_empty())
    
    def test_get_line_numbers(self):
        code = "x = 1\ny = 2"
        stmts = ast.parse(code).body
        block = CFGBlock(id=0, stmts=stmts)
        self.assertEqual(block.get_first_line(), 1)
        self.assertEqual(block.get_last_line(), 2)


class TestCFGEdge(unittest.TestCase):
    """Tests for CFGEdge"""
    
    def test_create_sequential_edge(self):
        edge = CFGEdge(source_id=0, target_id=1)
        self.assertEqual(edge.source_id, 0)
        self.assertEqual(edge.target_id, 1)
        self.assertEqual(edge.kind, 'sequential')
        self.assertFalse(edge.is_conditional())
        self.assertFalse(edge.is_back_edge())
    
    def test_create_branch_edge(self):
        cond = ast.parse("x > 0", mode='eval').body
        edge = CFGEdge(source_id=0, target_id=1, kind='branch_true', condition=cond)
        self.assertTrue(edge.is_conditional())
        self.assertIsNotNone(edge.condition)
    
    def test_create_back_edge(self):
        edge = CFGEdge(source_id=1, target_id=0, kind='loop_back')
        self.assertTrue(edge.is_back_edge())


class TestCFG(unittest.TestCase):
    """Tests for CFG"""
    
    def test_create_cfg(self):
        cfg = CFG(func_name="test")
        self.assertEqual(cfg.func_name, "test")
        self.assertEqual(len(cfg.blocks), 0)
        self.assertEqual(len(cfg.edges), 0)
    
    def test_add_block(self):
        cfg = CFG(func_name="test")
        block = cfg.add_block()
        self.assertEqual(block.id, 0)
        self.assertIn(0, cfg.blocks)
        
        block2 = cfg.add_block()
        self.assertEqual(block2.id, 1)
    
    def test_add_edge(self):
        cfg = CFG(func_name="test")
        cfg.add_block()
        cfg.add_block()
        
        edge = cfg.add_edge(0, 1)
        self.assertEqual(len(cfg.edges), 1)
        self.assertEqual(edge.source_id, 0)
        self.assertEqual(edge.target_id, 1)
    
    def test_get_successors_predecessors(self):
        cfg = CFG(func_name="test")
        cfg.add_block()  # 0
        cfg.add_block()  # 1
        cfg.add_block()  # 2
        
        cfg.add_edge(0, 1)
        cfg.add_edge(0, 2)
        cfg.add_edge(1, 2)
        
        succs = cfg.get_successors(0)
        self.assertEqual(len(succs), 2)
        
        preds = cfg.get_predecessors(2)
        self.assertEqual(len(preds), 2)
    
    def test_is_merge_point(self):
        cfg = CFG(func_name="test")
        cfg.add_block()  # 0
        cfg.add_block()  # 1
        cfg.add_block()  # 2 - merge point
        
        cfg.add_edge(0, 2)
        cfg.add_edge(1, 2)
        
        self.assertFalse(cfg.is_merge_point(0))
        self.assertFalse(cfg.is_merge_point(1))
        self.assertTrue(cfg.is_merge_point(2))
    
    def test_topological_order(self):
        cfg = CFG(func_name="test")
        entry = cfg.add_block()  # 0
        b1 = cfg.add_block()     # 1
        b2 = cfg.add_block()     # 2
        exit_b = cfg.add_block() # 3
        
        cfg.entry_id = entry.id
        cfg.exit_id = exit_b.id
        
        cfg.add_edge(0, 1)
        cfg.add_edge(0, 2)
        cfg.add_edge(1, 3)
        cfg.add_edge(2, 3)
        
        order = cfg.topological_order()
        ids = [b.id for b in order]
        
        # Entry must come first
        self.assertEqual(ids[0], 0)
        # Exit must come last
        self.assertEqual(ids[-1], 3)
        # 1 and 2 must come before 3
        self.assertLess(ids.index(1), ids.index(3))
        self.assertLess(ids.index(2), ids.index(3))
    
    def test_validate(self):
        cfg = CFG(func_name="test")
        entry = cfg.add_block()
        exit_b = cfg.add_block()
        cfg.entry_id = entry.id
        cfg.exit_id = exit_b.id
        cfg.add_edge(0, 1)
        
        errors = cfg.validate()
        self.assertEqual(len(errors), 0)
    
    def test_to_dot(self):
        cfg = CFG(func_name="test")
        entry = cfg.add_block([ast.parse("x = 1").body[0]])
        exit_b = cfg.add_block()
        cfg.entry_id = entry.id
        cfg.exit_id = exit_b.id
        cfg.add_edge(0, 1)
        
        dot = cfg.to_dot()
        self.assertIn('digraph "test"', dot)
        self.assertIn('B0', dot)
        self.assertIn('B1', dot)


class TestLinearSnapshotUtils(unittest.TestCase):
    """Tests for linear snapshot utility functions"""
    
    def test_copy_snapshot(self):
        original: LinearSnapshot = {'token': {(): 'active', (0,): 'consumed'}}
        copied = copy_snapshot(original)
        
        # Modify copy
        copied['token'][()] = 'consumed'
        
        # Original should be unchanged
        self.assertEqual(original['token'][()], 'active')
    
    def test_snapshots_compatible_both_active(self):
        s1: LinearSnapshot = {'token': {(): 'active'}}
        s2: LinearSnapshot = {'token': {(): 'active'}}
        
        self.assertTrue(snapshots_compatible(s1, s2))
    
    def test_snapshots_compatible_both_consumed(self):
        s1: LinearSnapshot = {'token': {(): 'consumed'}}
        s2: LinearSnapshot = {'token': {(): 'consumed'}}
        
        self.assertTrue(snapshots_compatible(s1, s2))
    
    def test_snapshots_compatible_consumed_undefined(self):
        # consumed and undefined are both "not active", so compatible
        s1: LinearSnapshot = {'token': {(): 'consumed'}}
        s2: LinearSnapshot = {'token': {(): 'undefined'}}
        
        self.assertTrue(snapshots_compatible(s1, s2))
    
    def test_snapshots_incompatible(self):
        # active vs consumed - incompatible
        s1: LinearSnapshot = {'token': {(): 'active'}}
        s2: LinearSnapshot = {'token': {(): 'consumed'}}
        
        self.assertFalse(snapshots_compatible(s1, s2))
    
    def test_snapshots_compatible_multiple_paths(self):
        # Struct with two linear fields
        s1: LinearSnapshot = {'s': {(0,): 'active', (1,): 'consumed'}}
        s2: LinearSnapshot = {'s': {(0,): 'active', (1,): 'consumed'}}
        
        self.assertTrue(snapshots_compatible(s1, s2))
    
    def test_snapshots_incompatible_one_path(self):
        # One path differs
        s1: LinearSnapshot = {'s': {(0,): 'active', (1,): 'consumed'}}
        s2: LinearSnapshot = {'s': {(0,): 'consumed', (1,): 'consumed'}}
        
        self.assertFalse(snapshots_compatible(s1, s2))
    
    def test_find_snapshot_diffs_no_diff(self):
        s1: LinearSnapshot = {'token': {(): 'active'}}
        s2: LinearSnapshot = {'token': {(): 'active'}}
        
        diffs = find_snapshot_diffs(s1, s2)
        self.assertEqual(len(diffs), 0)
    
    def test_find_snapshot_diffs_with_diff(self):
        s1: LinearSnapshot = {'token': {(): 'active'}}
        s2: LinearSnapshot = {'token': {(): 'consumed'}}
        
        diffs = find_snapshot_diffs(s1, s2)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]['path_str'], 'token')
    
    def test_find_snapshot_diffs_multiple_vars(self):
        s1: LinearSnapshot = {'a': {(): 'active'}, 'b': {(): 'consumed'}}
        s2: LinearSnapshot = {'a': {(): 'consumed'}, 'b': {(): 'consumed'}}
        
        diffs = find_snapshot_diffs(s1, s2)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]['path_str'], 'a')


class TestLinearError(unittest.TestCase):
    """Tests for LinearError"""
    
    def test_create_error(self):
        error = LinearError(
            kind='merge_inconsistent',
            block_id=3,
            message='Test error'
        )
        self.assertEqual(error.kind, 'merge_inconsistent')
        self.assertEqual(error.block_id, 3)
        self.assertEqual(error.message, 'Test error')
    
    def test_format_simple_error(self):
        error = LinearError(
            kind='unconsumed_at_exit',
            block_id=5,
            message='Token not consumed'
        )
        formatted = error.format()
        self.assertIn('Token not consumed', formatted)


if __name__ == '__main__':
    unittest.main()
