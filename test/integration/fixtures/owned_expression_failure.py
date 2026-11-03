"""Rejected ownership and proof operations, selected in isolated processes."""

import sys

from pythoc import (
    array, compile, const, consume, defer, i32, linear, move, ptr, refined, struct, void,
)
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.logger import set_raise_on_error


set_raise_on_error(True)
Case = sys.argv[1]
ScalarProof = refined[i32, 'field_checked']
Record = struct['count': i32]
ProofRecord = struct['checked': ScalarProof]
NestedProofRecord = struct['inner': ProofRecord]
Checked = refined[Record, 'record_checked']
Nested = struct['inner': Record]
CheckedNested = refined[Nested, 'nested_checked']
CheckedArray = refined[array[i32, 2], 'array_checked']


@compile
def make_token() -> linear:
    return linear()


if Case == 'void_selection':
    @compile
    def candidate(flag: i32) -> i32:
        token = linear()
        value = move(token) if flag else consume(token)
        return 0
elif Case == 'void_selection_reversed':
    @compile
    def candidate(flag: i32) -> i32:
        token = linear()
        value = consume(token) if flag else move(token)
        return 0
elif Case == 'nested_selection':
    @compile
    def candidate(first: i32, second: i32) -> i32:
        token = linear()
        value = (move(token) if first else move(token)) if second else consume(token)
        return 0
elif Case == 'call_result_discard':
    @compile
    def candidate() -> i32:
        make_token()
        return 0
elif Case == 'deferred_result_discard':
    @compile
    def candidate() -> i32:
        defer(make_token)
        return 0
elif Case == 'literal_discard':
    @compile
    def candidate() -> i32:
        (linear(),)
        return 0
elif Case == 'carrier_move_discard':
    @compile
    def candidate() -> i32:
        token = linear()
        move((token,))
        return 0
elif Case == 'loop_discard':
    @compile
    def candidate(count: i32) -> i32:
        index: i32 = 0
        while index < count:
            make_token()
            index += 1
        return 0
elif Case == 'double_transfer':
    @compile
    def candidate() -> i32:
        token = make_token()
        consume(token)
        consume(token)
        return 0
elif Case == 'copy_selection':
    @compile
    def candidate(flag: i32) -> i32:
        token = linear()
        value = token if flag else token
        consume(token)
        consume(value)
        return 0
elif Case == 'proof_default':
    @compile
    def candidate() -> i32:
        value = ProofRecord()
        return value.checked
elif Case == 'nested_proof_default':
    @compile
    def candidate() -> i32:
        value = NestedProofRecord()
        return value.inner.checked
elif Case == 'proof_member_store':
    @compile
    def candidate() -> i32:
        value: Checked = Checked((1,))
        value.count = -2
        return value.count
elif Case == 'proof_index_store':
    @compile
    def candidate() -> i32:
        value: Checked = Checked((1,))
        value[0] = -2
        return value[0]
elif Case == 'proof_nested_store':
    @compile
    def candidate() -> i32:
        value: CheckedNested = CheckedNested(((1,),))
        value.inner.count = -2
        return value.inner.count
elif Case == 'proof_member_address':
    @compile
    def candidate() -> i32:
        value: Checked = Checked((1,))
        address: ptr[i32] = ptr(value.count)
        address[0] = -2
        return value.count
elif Case == 'proof_pointer_store':
    @compile
    def candidate() -> i32:
        value: Checked = Checked((1,))
        address = ptr(value)
        address.count = -2
        return address.count
elif Case == 'proof_array_store':
    @compile
    def candidate(address: ptr[CheckedArray]) -> i32:
        address[0][0] = -2
        return address[0][0]
elif Case == 'proof_array_decay':
    @compile
    def candidate(source: ptr[CheckedArray]) -> i32:
        address: ptr[i32] = source[0]
        return address[0]
else:
    raise ValueError(Case)


flush_all_pending_outputs()
