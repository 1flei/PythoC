"""Sparse ownership snapshots share unchanged paths as CFGs grow."""

import ast
import random
import unittest

from pythoc.cfg.graph import CFG
from pythoc.cfg.linear_checker import LinearChecker, LinearRegister, LinearTransition
from pythoc.cfg.linear_state import LinearState, _Branch


class TestLinearState(unittest.TestCase):
    def test_updates_preserve_old_snapshots_and_match_a_set(self):
        randomizer = random.Random(715)
        state, expected, history = LinearState(), set(), []
        for _ in range(1000):
            token, present = randomizer.randrange(256), bool(randomizer.randrange(2))
            history.append((state, expected.copy()))
            state = state.with_token(token, present)
            if present:
                expected.add(token)
            else:
                expected.discard(token)
            self.assertEqual(set(state), expected)
            self.assertEqual(len(state), len(expected))
            self.assertEqual(token in state, present)
        for previous, values in history:
            self.assertEqual(set(previous), values)

    def test_equal_states_do_not_depend_on_update_order(self):
        forward, reverse = LinearState(), LinearState()
        for token in range(128):
            forward = forward.with_token(token, True)
        for token in reversed(range(128)):
            reverse = reverse.with_token(token, True)
        self.assertEqual(forward, reverse)
        for token in range(128):
            reverse = reverse.with_token(token, False)
        self.assertEqual(reverse, LinearState())
        self.assertEqual(len(forward), 128)

    def test_unchanged_updates_share_the_state(self):
        state = LinearState().with_token(3, True)
        self.assertIs(state.with_token(3, True), state)
        self.assertIs(state.with_token(7, False), state)

    def test_retained_state_nodes_grow_with_changes_not_snapshot_sizes(self):
        for size in (128, 256, 512):
            with self.subTest(size=size):
                state, snapshots = LinearState(), []
                for token in range(size):
                    state = state.with_token(token, True)
                    snapshots.append(state)
                for token in range(size):
                    state = state.with_token(token, False)
                    snapshots.append(state)
                visited, pending = set(), [state._root for state in snapshots]
                while pending:
                    node = pending.pop()
                    if not isinstance(node, _Branch) or id(node) in visited:
                        continue
                    visited.add(id(node))
                    pending.extend((node.left, node.right))
                self.assertLessEqual(len(visited), 2 * size * size.bit_length())

    def test_cfg_forwards_live_states_without_copying_them(self):
        cfg = CFG('live_chain')
        blocks = [cfg.add_block() for _ in range(130)]
        cfg.entry_id, cfg.exit_id = blocks[0].id, blocks[-1].id
        for left, right in zip(blocks, blocks[1:]):
            cfg.add_edge(left.id, right.id)
        blocks[0].linear_events.extend(
            LinearRegister(1, 'record', (field,), 'valid') for field in range(128))
        blocks[-2].linear_events.extend(
            LinearTransition(1, 'record', (field,), 'valid', 'invalid')
            for field in range(128))
        checker = LinearChecker()
        self.assertEqual(checker.check(cfg, {}), [])
        state = checker.exit_snapshots[blocks[0].id]
        for block in blocks[1:-2]:
            self.assertIs(checker.entry_snapshots[block.id], state)
            self.assertIs(checker.exit_snapshots[block.id], state)
        self.assertFalse(checker.exit_snapshots[blocks[-1].id])

    def test_pruning_consumed_paths_still_rejects_double_consumption(self):
        cfg = CFG('double_consume')
        entry, exit_block = cfg.add_block(), cfg.add_block()
        cfg.entry_id, cfg.exit_id = entry.id, exit_block.id
        entry.stmts = ast.parse('consume(record)').body
        cfg.add_edge(entry.id, exit_block.id)
        entry.linear_events = [
            LinearRegister(1, 'record', (0,), 'valid'),
            LinearTransition(1, 'record', (0,), 'valid', 'invalid'),
            LinearTransition(1, 'record', (0,), 'valid', 'invalid'),
        ]
        self.assertIn('use_after_consume',
                      {error.kind for error in LinearChecker().check(cfg, {})})


if __name__ == '__main__':
    unittest.main()
