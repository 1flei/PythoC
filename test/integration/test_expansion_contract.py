"""Expansion preserves argument values, effect order and nested bindings."""

import unittest
from unittest.mock import patch

from pythoc import (
    compile, consume, effect, i32, i64, bool, inline, linear, ptr, refine,
    static, struct,
)
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


factory_calls = []


@compile
class ExpansionState:
    trace: static[i64]
    calls: static[i32]


@compile
def mark(digit: i32, value: i32) -> i32:
    ExpansionState.trace = ExpansionState.trace * 10 + digit
    ExpansionState.calls += 1
    return value


@compile
def positive(value: i32) -> bool:
    ExpansionState.trace = ExpansionState.trace * 10 + 7
    return value > 0


@compile
def even(value: i32) -> bool:
    ExpansionState.trace = ExpansionState.trace * 10 + 8
    return value % 2 == 0


@compile
def positive_sum(left: i32, right: i32) -> bool:
    ExpansionState.trace = ExpansionState.trace * 10 + 7
    return left + right > 0


@compile
def unstable() -> i32:
    ExpansionState.trace = ExpansionState.trace * 10 + 1
    ExpansionState.calls += 1
    if ExpansionState.calls <= 2:
        return 1
    return -1


@compile
def read_trace() -> i64:
    return ExpansionState.trace


@compile
def read_calls() -> i32:
    return ExpansionState.calls


def predicate_factory():
    factory_calls.append(1)
    return positive


@compile
def refine_once() -> i32:
    for value in refine(unstable(), positive):
        return value
    else:
        return -99


@compile
def refine_many() -> i32:
    for value in refine(mark(1, 4), positive, even):
        return value
    else:
        return -99


@compile
def refine_failure() -> i32:
    for value in refine(mark(1, -4), positive, even):
        return value
    else:
        return -99


@compile
def refine_multi() -> i32:
    for value in refine(mark(1, 3), mark(2, 4), positive_sum):
        return value.left * 10 + value.right
    else:
        return -99


@compile
def refine_starred_pair() -> struct[i32, i32]:
    return (mark(1, 3), mark(2, 4))


@compile
def refine_starred() -> i32:
    for value in refine(*refine_starred_pair(), positive_sum):
        return value.left * 10 + value.right
    else:
        return -99


@compile
def refine_factory() -> i32:
    for value in refine(mark(1, 4), predicate_factory()):
        return value
    else:
        return -99


@compile
def refine_nested() -> i32:
    total: i32 = 0
    for left in refine(mark(1, 3), positive):
        for right in refine(mark(2, 4), positive):
            total = left * 10 + right
    return total


@compile
def refine_tag() -> i32:
    for value in refine(mark(1, 4), 'checked'):
        return value


@inline
def inline_keyword_digits(left: i32, right: i32 = 8) -> i32:
    return left * 10 + right


@compile
def inline_keyword() -> i32:
    return inline_keyword_digits(right=mark(1, 3), left=mark(2, 4))


@inline
def inline_once_duplicate(value: i32) -> i32:
    return value * 10 + value


@compile
def inline_once() -> i32:
    return inline_once_duplicate(mark(1, 4))


@inline
def nested_shadow_outer(value: i32) -> i32:

    def inner(value: i32) -> i32:
        return value + 1
    return inner(4)


@compile
def nested_shadow() -> i32:
    return nested_shadow_outer(99)


@inline
def nested_capture_outer(value: i32) -> i32:

    def middle(other: i32) -> i32:

        def inner(value: i32) -> i32:
            return value + other
        return inner(4)
    return middle(3)


@compile
def nested_capture() -> i32:
    return nested_capture_outer(99)


@inline
def nested_default_outer(value: i32) -> i32:

    def inner(value: i32=value + 1) -> i32:
        return value
    return inner()


@compile
def nested_default() -> i32:
    return nested_default_outer(mark(1, 4))


@compile
def yield_once_digits(left: i32, right: i32) -> i32:
    yield (left * 10 + right)


@compile
def yield_once() -> i32:
    for value in yield_once_digits(right=mark(1, 3), left=mark(2, 4)):
        return value


@compile
def closure_once() -> i32:

    def duplicate(value: i32) -> i32:
        return value * 10 + value
    return duplicate(mark(1, 4))


@compile
def refine_break() -> i32:
    result: i32 = 0
    for value in refine(mark(1, 4), positive):
        result = value
        break
    else:
        result = -99
    return result


@compile
def yield_carrier_one(value: i32) -> i32:
    yield value


@compile
def yield_carrier() -> i32:
    values = yield_carrier_one(mark(1, 4))
    result: i32 = 0
    for value in values:
        result += value
    for value in values:
        result += value
    return result


@compile
def generator_replay_one(value: i32) -> i32:
    yield value


@compile
def generator_replay() -> i32:
    values = (value for value in generator_replay_one(mark(1, 4)))
    result: i32 = 0
    for value in values:
        result += value
    for value in values:
        result += value
    return result


@compile
def chain_once_source(value: i32) -> i32:
    yield value


@compile
def chain_once_producer() -> i32:
    for value in chain_once_source(mark(1, 4)):
        yield value + 1


@compile
def chain_once() -> i32:
    for value in chain_once_producer():
        return value


@inline
def inline_snapshot_modify(value: i32, address: ptr[i32]) -> i32:
    address[0] = 99
    return value * 10 + value


@compile
def inline_snapshot() -> i32:
    value: i32 = mark(1, 4)
    return inline_snapshot_modify(value, ptr(value))


@inline
def inline_unused_ignore(value: i32) -> i32:
    return 5


@compile
def inline_unused() -> i32:
    return inline_unused_ignore(mark(1, 4))


@inline
def inline_collectors_digits(*values: struct[i32, i32]) -> i32:
    return values[0] * 10 + values[1]


@compile
def inline_collectors() -> i32:
    return inline_collectors_digits(mark(1, 3), mark(2, 4))


@inline
def inline_default_digits(left: i32, /, *, right: i32 = 8) -> i32:
    return left * 10 + right


@compile
def inline_default() -> i32:
    return inline_default_digits(mark(1, 4))


@compile
def linear_refine() -> i32:
    token = linear()
    for proof in refine(token, 'checked'):
        consume(proof)
    return 4


@compile
def reset_state() -> None:
    ExpansionState.trace = 0
    ExpansionState.calls = 0


class ExpansionContractTest(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Host-side factory counts require code generation even on warm runs.
        with patch('pythoc.build.cache.BuildCache.check_obj_uptodate',
                   return_value=False):
            flush_all_pending_outputs()

    def setUp(self):
        reset_state()

    def observe(self, function, value, trace, calls):
        self.assertEqual(function(), value)
        self.assertEqual(read_trace(), trace)
        self.assertEqual(read_calls(), calls)

    def test_refinement_checks_the_value_it_returns(self):
        self.observe(refine_once, 1, 17, 1)

    def test_refinement_predicates_share_one_input(self):
        self.observe(refine_many, 4, 178, 1)

    def test_refinement_failure_short_circuits_predicates(self):
        self.observe(refine_failure, -99, 17, 1)

    def test_multi_parameter_refinement_preserves_order(self):
        self.observe(refine_multi, 34, 127, 2)

    def test_refinement_consumes_expanded_starred_values(self):
        self.observe(refine_starred, 34, 127, 2)

    def test_predicate_factory_is_evaluated_once(self):
        self.observe(refine_factory, 4, 17, 1)
        self.assertEqual(factory_calls, [1])

    def test_nested_refinements_preserve_order(self):
        self.observe(refine_nested, 34, 1727, 2)

    def test_tag_only_refinement_does_not_replay_input(self):
        self.observe(refine_tag, 4, 1, 1)

    def test_inline_keywords_bind_without_reordering_effects(self):
        self.observe(inline_keyword, 43, 12, 2)

    def test_inline_repeated_parameter_uses_one_value(self):
        self.observe(inline_once, 44, 1, 1)

    def test_nested_parameter_shadows_outer_parameter(self):
        self.observe(nested_shadow, 5, 0, 0)

    def test_nested_capture_and_shadow_are_distinct(self):
        self.observe(nested_capture, 7, 0, 0)

    def test_nested_default_uses_definition_environment(self):
        self.observe(nested_default, 5, 1, 1)

    def test_yield_arguments_share_the_call_contract(self):
        self.observe(yield_once, 43, 12, 2)

    def test_closure_arguments_share_the_call_contract(self):
        self.observe(closure_once, 44, 1, 1)

    def test_refinement_break_skips_else(self):
        self.observe(refine_break, 4, 17, 1)

    def test_stored_yield_call_reuses_the_evaluated_argument(self):
        self.observe(yield_carrier, 8, 1, 1)

    def test_generator_syntax_keeps_its_explicit_replay_semantics(self):
        self.observe(generator_replay, 8, 11, 2)

    def test_yield_chain_preserves_argument_evaluation(self):
        self.observe(chain_once, 5, 1, 1)

    def test_refinement_transfers_a_linear_owner_once(self):
        self.observe(linear_refine, 4, 0, 0)

    def test_inline_parameter_is_a_snapshot_before_body_mutation(self):
        self.observe(inline_snapshot, 44, 1, 1)

    def test_unused_inline_argument_still_runs_once(self):
        self.observe(inline_unused, 5, 1, 1)

    def test_inline_typed_collectors_use_the_shared_binder(self):
        self.observe(inline_collectors, 34, 12, 2)

    def test_inline_parameter_kinds_keep_definition_time_defaults(self):
        self.observe(inline_default, 48, 1, 1)

    def test_refinement_rejects_reusing_the_transferred_owner(self):
        with CompileSession(), self.assertRaisesRegex(RuntimeError, 'already consumed'):
            @compile(suffix='refine_use_after_transfer')
            def invalid_refinement() -> i32:
                token = linear()
                for proof in refine(token, 'checked'):
                    consume(proof)
                consume(token)
                return 4

            flush_all_pending_outputs()

    def test_refinement_keeps_the_returned_owner_obligation(self):
        with CompileSession(), self.assertRaisesRegex(RuntimeError, 'not consumed'):
            @compile(suffix='refine_returned_owner_leak')
            def invalid_refinement() -> i32:
                token = linear()
                for proof in refine(token, 'checked'):
                    pass
                return 4

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
