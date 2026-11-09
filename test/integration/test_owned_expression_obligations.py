"""CFG owns expression-result obligations; aggregate proofs protect projections."""

import unittest

from pythoc import (
    array, compile, const, consume, defer, i32, inline, label, linear, move,
    ptr, refined, struct, void,
)
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


Record = struct['count': i32]
Checked = refined[Record, 'positive_record']
Pair = struct[linear, linear]
Single = struct['token': linear]
ScalarProof = refined[i32, 'field_checked']
ProofRecord = struct['checked': ScalarProof]
NestedProofRecord = struct['inner': ProofRecord]
Nested = struct['inner': Record]
CheckedNested = refined[Nested, 'nested_checked']
CheckedArray = refined[array[i32, 2], 'array_checked']


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


class OwnedExpressionObligationsTest(DeferredTestCase):
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

    def test_rejects_void_selection(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def candidate(flag: i32) -> i32:
                token = linear()
                value = move(token) if flag else consume(token)
                return 0

            flush_all_pending_outputs()

    def test_rejects_void_selection_reversed(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def candidate(flag: i32) -> i32:
                token = linear()
                value = consume(token) if flag else move(token)
                return 0

            flush_all_pending_outputs()

    def test_rejects_nested_selection(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def candidate(first: i32, second: i32) -> i32:
                token = linear()
                value = ((move(token) if first else move(token))
                         if second else consume(token))
                return 0

            flush_all_pending_outputs()

    def test_rejects_call_result_discard(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def make_token() -> linear:
                return linear()

            @compile
            def candidate() -> i32:
                make_token()
                return 0

            flush_all_pending_outputs()

    def test_rejects_deferred_result_discard(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def make_token() -> linear:
                return linear()

            @compile
            def candidate() -> i32:
                defer(make_token)
                return 0

            flush_all_pending_outputs()

    def test_rejects_literal_discard(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def candidate() -> i32:
                (linear(),)
                return 0

            flush_all_pending_outputs()

    def test_rejects_carrier_move_discard(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def candidate() -> i32:
                token = linear()
                move((token,))
                return 0

            flush_all_pending_outputs()

    def test_rejects_loop_discard(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def make_token() -> linear:
                return linear()

            @compile
            def candidate(count: i32) -> i32:
                index: i32 = 0
                while index < count:
                    make_token()
                    index += 1
                return 0

            flush_all_pending_outputs()

    def test_rejects_double_transfer(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def make_token() -> linear:
                return linear()

            @compile
            def candidate() -> i32:
                token = make_token()
                consume(token)
                consume(token)
                return 0

            flush_all_pending_outputs()

    def test_rejects_copy_selection(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'CFG linear check'):
            @compile
            def candidate(flag: i32) -> i32:
                token = linear()
                value = token if flag else token
                consume(token)
                consume(value)
                return 0

            flush_all_pending_outputs()

    def test_rejects_proof_default(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'default-construct an aggregate with proof-bearing'):
            @compile
            def candidate() -> i32:
                value = ProofRecord()
                return value.checked

            flush_all_pending_outputs()

    def test_rejects_nested_proof_default(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'default-construct an aggregate with proof-bearing'):
            @compile
            def candidate() -> i32:
                value = NestedProofRecord()
                return value.inner.checked

            flush_all_pending_outputs()

    def test_rejects_proof_member_store(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            @compile
            def candidate() -> i32:
                value: Checked = Checked((1,))
                value.count = -2
                return value.count

            flush_all_pending_outputs()

    def test_rejects_proof_index_store(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            @compile
            def candidate() -> i32:
                value: Checked = Checked((1,))
                value[0] = -2
                return value[0]

            flush_all_pending_outputs()

    def test_rejects_proof_nested_store(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            @compile
            def candidate() -> i32:
                value: CheckedNested = CheckedNested(((1,),))
                value.inner.count = -2
                return value.inner.count

            flush_all_pending_outputs()

    def test_rejects_proof_member_address(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'Cannot implicitly convert'):
            @compile
            def candidate() -> i32:
                value: Checked = Checked((1,))
                address: ptr[i32] = ptr(value.count)
                address[0] = -2
                return value.count

            flush_all_pending_outputs()

    def test_rejects_proof_pointer_store(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            @compile
            def candidate() -> i32:
                value: Checked = Checked((1,))
                address = ptr(value)
                address.count = -2
                return address.count

            flush_all_pending_outputs()

    def test_rejects_proof_array_store(self):
        with CompileSession(), self.assertRaisesRegex(
                RuntimeError, 'Cannot reassign to const'):
            @compile
            def candidate(address: ptr[CheckedArray]) -> i32:
                address[0][0] = -2
                return address[0][0]

            flush_all_pending_outputs()

    def test_rejects_proof_array_decay(self):
        with CompileSession(), self.assertRaisesRegex(
                TypeError, 'Cannot implicitly convert'):
            @compile
            def candidate(source: ptr[CheckedArray]) -> i32:
                address: ptr[i32] = source[0]
                return address[0]

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
