"""Owned temporaries obey the same CFG rules as named resource bindings."""

import ast
import unittest

from pythoc.cfg.graph import CFG
from pythoc.cfg.linear_checker import (
    LinearChecker, LinearCopy, LinearRegister, LinearTransition,
)


class OwnedExpressionCFGTest(unittest.TestCase):
    def make_selection(self, consume_result):
        cfg = CFG('owned_selection')
        entry, left, right, merge, exit_block = [cfg.add_block() for _ in range(5)]
        cfg.entry_id, cfg.exit_id = entry.id, exit_block.id
        entry.stmts.append(ast.parse('token = create()').body[0])
        cfg.add_edge(entry.id, left.id)
        cfg.add_edge(entry.id, right.id)
        cfg.add_edge(left.id, merge.id)
        cfg.add_edge(right.id, merge.id)
        cfg.add_edge(merge.id, exit_block.id, kind='return')
        entry.linear_events.append(LinearRegister(1, 'token', (), 'valid'))
        for branch in (left, right):
            branch.linear_events.append(
                LinearTransition(1, 'token', (), 'valid', 'invalid'))
        left.linear_events.append(LinearRegister(2, '<owned result>', (), 'valid'))
        if consume_result:
            left.linear_events.append(
                LinearTransition(2, '<owned result>', (), 'valid', 'invalid'))
        return cfg

    def test_source_invalidation_does_not_discharge_moved_result(self):
        errors = LinearChecker().check(self.make_selection(False), {})
        self.assertIn('merge_inconsistent', {error.kind for error in errors})

    def test_consumed_result_balances_the_same_branch(self):
        self.assertEqual(LinearChecker().check(self.make_selection(True), {}), [])

    def test_copy_check_is_owned_by_cfg_and_ignores_unreachable_blocks(self):
        cfg = self.make_selection(True)
        cfg.blocks[cfg.entry_id].linear_events.append(LinearCopy(1, 'token', ()))
        self.assertIn('illegal_copy',
                      {error.kind for error in LinearChecker().check(cfg, {})})
        cfg.blocks[cfg.entry_id].linear_events.pop()
        unreachable = cfg.add_block()
        unreachable.linear_events.append(LinearCopy(1, 'token', ()))
        self.assertEqual(LinearChecker().check(cfg, {}), [])

    def test_function_exit_rejects_unconsumed_owned_temporary(self):
        cfg = CFG('temporary_exit')
        entry, exit_block = cfg.add_block(), cfg.add_block()
        cfg.entry_id, cfg.exit_id = entry.id, exit_block.id
        entry.stmts.append(ast.parse('create()').body[0])
        entry.linear_events.append(LinearRegister(2, '<owned result>', (), 'valid'))
        cfg.add_edge(entry.id, exit_block.id, kind='return')
        self.assertIn('unconsumed_at_exit',
                      {error.kind for error in LinearChecker().check(cfg, {})})

    def test_second_consumption_of_temporary_is_rejected(self):
        cfg = self.make_selection(True)
        cfg.blocks[1].linear_events.append(
            LinearTransition(2, '<owned result>', (), 'valid', 'invalid'))
        self.assertIn('use_after_consume',
                      {error.kind for error in LinearChecker().check(cfg, {})})


if __name__ == '__main__':
    unittest.main()
