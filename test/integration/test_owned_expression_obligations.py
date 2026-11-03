"""CFG owns expression-result obligations; aggregate proofs protect projections."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pythoc
from pythoc import (
    array, compile, consume, defer, i32, inline, label, linear, move, ptr, refined, struct, void,
)


from pythoc.logger import set_raise_on_error

set_raise_on_error(True)

Record = struct['count': i32]
Checked = refined[Record, 'positive_record']
Pair = struct[linear, linear]
Single = struct['token': linear]


@compile
def make_token() -> linear:
    return linear()


@inline
def relay_token(token: linear) -> linear:
    return move(token)


@compile
def make_single() -> Single:
    return linear(),


@compile
def make_pair() -> Pair:
    return linear(), linear()


@compile
def consume_call_and_carrier_results() -> i32:
    consume(make_token())
    consume(make_single().token)
    consume((linear(), linear()))
    token = linear()
    carrier = move((token,))
    consume(carrier[0])
    pair = make_pair()
    consume(pair[0])
    consume(pair[1])
    return 41


@compile
def select_owned(flag: i32) -> i32:
    token = make_token()
    selected = move(token) if flag else move(token)
    consume(relay_token(move(selected)))

    def relay(value: linear) -> linear:
        return move(value)

    consume(relay(make_token()))
    return 42


@compile
def consume_loop_results(count: i32) -> i32:
    index: i32 = 0
    while index < count:
        consume(make_token())
        index += 1
    return count


@compile
def deferred_owned_argument() -> i32:
    counter: i32 = 0

    def finish(value: ptr[i32], token: linear) -> void:
        consume(token)
        value[0] += 1

    with label('owned_cleanup'):
        defer(finish, ptr(counter), make_token())
    return counter


@compile
def partial_record_initialization() -> i32:
    pair = Pair()
    pair[0] = linear()
    pair[1] = linear()
    consume(pair[0])
    consume(pair[1])
    return 43


@compile
def read_and_replace_proven_record() -> i32:
    value: Checked = Checked((1,))
    value = Checked((2,))
    copy: Record = value
    copy.count = 40
    return value.count + copy.count


@compile
def refined_pointer_keeps_mutable_pointee() -> i32:
    NonNull = refined[ptr[i32], 'nonnull']
    value: i32 = 1
    address: NonNull = NonNull(ptr(value))
    address[0] = 42
    return value


class OwnedExpressionObligationsTest(unittest.TestCase):
    def test_owned_results_transfer_through_calls_and_literal_carriers(self):
        self.assertEqual(consume_call_and_carrier_results(), 41)

    def test_conditional_and_inline_results_transfer_on_both_paths(self):
        for flag in (0, 1):
            with self.subTest(flag=flag):
                self.assertEqual(select_owned(flag), 42)

    def test_loop_results_are_consumed_per_iteration(self):
        for count in (0, 1, 5):
            with self.subTest(count=count):
                self.assertEqual(consume_loop_results(count), count)

    def test_deferred_argument_is_consumed_once_at_scope_exit(self):
        self.assertEqual(deferred_owned_argument(), 1)

    def test_plain_linear_record_supports_partial_initialization(self):
        self.assertEqual(partial_record_initialization(), 43)

    def test_whole_proof_replacement_and_mutable_copy(self):
        self.assertEqual(read_and_replace_proven_record(), 42)

    def test_refined_pointer_does_not_freeze_its_pointee(self):
        self.assertEqual(refined_pointer_keeps_mutable_pointee(), 42)

    def test_invalid_owned_and_proof_operations_are_rejected(self):
        cases = (
            ('void_selection', 'CFG linear check'),
            ('void_selection_reversed', 'CFG linear check'),
            ('nested_selection', 'CFG linear check'),
            ('call_result_discard', 'CFG linear check'),
            ('deferred_result_discard', 'CFG linear check'),
            ('literal_discard', 'CFG linear check'),
            ('carrier_move_discard', 'CFG linear check'),
            ('loop_discard', 'CFG linear check'),
            ('double_transfer', 'CFG linear check'),
            ('copy_selection', 'CFG linear check'),
            ('proof_default', 'default-construct an aggregate with proof-bearing'),
            ('nested_proof_default', 'default-construct an aggregate with proof-bearing'),
            ('proof_member_store', 'Cannot reassign to const'),
            ('proof_index_store', 'Cannot reassign to const'),
            ('proof_nested_store', 'Cannot reassign to const'),
            ('proof_member_address', 'Cannot implicitly convert'),
            ('proof_pointer_store', 'Cannot reassign to const'),
            ('proof_array_store', 'Cannot reassign to const'),
            ('proof_array_decay', 'Cannot implicitly convert'),
        )
        fixture = Path(__file__).parent / 'fixtures' / 'owned_expression_failure.py'
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(pythoc.__file__).resolve().parent.parent)
        env['PC_RAISE_ON_ERROR'] = '1'
        for case, diagnostic in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory(prefix='owned-expression-') as temp:
                    result = subprocess.run(
                        [sys.executable, str(fixture), case], cwd=temp, env=env,
                        capture_output=True, text=True, timeout=120)
                output = result.stdout + result.stderr
                self.assertNotEqual(result.returncode, 0, output)
                self.assertIn(diagnostic, output)
                self.assertRegex(output, '(?:TypeError|RuntimeError):')


if __name__ == '__main__':
    unittest.main()
