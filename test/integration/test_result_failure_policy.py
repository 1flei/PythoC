"""Result propagation is local; a global error slot is an explicit policy."""

import unittest
from contextlib import redirect_stderr
from io import StringIO

from pythoc import compile, effect, enum, i32, i64, ptr, static, u32
from pythoc.build.output_manager import clear_failed_group, flush_all_pending_outputs
from pythoc.logger import set_raise_on_error
from pythoc.std.result import ErrnoSlotProvider, Option, Result as ResultType


set_raise_on_error(True)
Result = ResultType(i32, i32, name='ExplicitFailurePolicyResult')
provider = ErrnoSlotProvider()
explicit_slot = provider.get_slot(i32)


@compile
def checked(value: i32) -> Result:
    if value >= 0:
        return Result.ok(value)
    return Result.err(-value * 10)


OtherOk = ResultType(i64, i32)
Maybe = Option(i32)
bind_other = OtherOk.bind


@compile
class Calls:
    checked: static[i32]


@compile
def first(value: i32) -> OtherOk:
    Calls.checked += 1
    if value < 0:
        return OtherOk.err(value)
    return OtherOk.ok(i64(value))


@compile
def counted_second(value: i32) -> Result:
    Calls.checked += 1
    return checked(value)


@compile
def counted_chain(a: i32, b: i32) -> i32:
    Calls.checked = 0
    composed = Result.do(
        i32(x) + y for x in bind_other(first(a)) for y in Result.bind(counted_second(b))
    )
    match composed:
        case (Result.Ok, answer):
            return Calls.checked * 100 + answer
        case (Result.Err, error):
            return Calls.checked * 100 + error


@compile
def empty_if_zero(value: i32) -> i32:
    if value != 0:
        yield value


@compile
def return_if_zero(value: i32) -> i32:
    if value == 0:
        return
    yield value


@compile
def with_fallback(value: i32) -> i32:
    for x in Result.bind(checked(value)):
        yield x
    Calls.checked += 1
    yield 99


@compile
def nested_do(value: i32) -> i32:
    Calls.checked = 0
    composed = Result.do(
        x for x in Result.bind(Result.do(y for y in with_fallback(value)))
    )
    match composed:
        case (Result.Ok, answer):
            return answer + 100 * Calls.checked
        case (Result.Err, error):
            return -error + 100 * Calls.checked


@compile
def compose(first: i32, second: i32) -> Result:
    return Result.do(
        left + right
        for left in Result.bind(checked(first))
        for right in Result.bind(checked(second))
    )


@compile
def propagate(first: i32, second: i32) -> i32:
    value = compose(first, second)
    match value:
        case (Result.Ok, answer):
            return answer
        case (Result.Err, error):
            return -error


@compile
def nested(value: i32) -> i32:
    composed = Result.do(
        item + 1 for item in Result.bind(compose(value, 3))
    )
    match composed:
        case (Result.Ok, answer):
            return answer
        case (Result.Err, error):
            return -error


with effect(errno=provider, suffix='explicit_result_errno'):
    @compile
    def record_failure(value: i32) -> i32:
        for item in Result.bind(checked(value)):
            return item
        return i32(-1)

    @compile
    def local_composition(value: i32) -> i32:
        composed = Result.do(item for item in Result.bind(checked(value)))
        match composed:
            case (Result.Ok, answer):
                return answer
            case (Result.Err, error):
                return -error


@compile
def reset_slot(value: i32) -> None:
    ptr[i32](explicit_slot())[0] = value


@compile
def read_slot() -> i32:
    return ptr[i32](explicit_slot())[0]


class TestResultFailurePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        flush_all_pending_outputs()

    def test_import_and_construction_do_not_register_errno(self):
        self.assertIsNone(effect.get_effect_impl('errno'))

    def test_default_composition_propagates_the_error_value(self):
        self.assertEqual(propagate(3, 4), 7)
        self.assertEqual(propagate(-2, 4), -20)
        self.assertEqual(propagate(3, -5), -50)
        self.assertEqual(nested(-2), -20)
        self.assertEqual(nested(2), 6)

    def test_mixed_success_types_and_first_failure(self):
        self.assertEqual(counted_chain(3, 4), 207)
        self.assertEqual(counted_chain(-2, 4), 98)
        self.assertEqual(counted_chain(3, -5), 250)

    def test_nested_do_exits_the_current_producer_on_first_failure(self):
        self.assertEqual(nested_do(3), 3)
        self.assertEqual(nested_do(-2), -20)

    def test_error_type_must_match_even_at_the_same_width(self):
        for err_type in (i64, u32):
            with self.subTest(err_type=err_type):
                other = ResultType(i32, err_type)

                @compile(suffix=("result_error_mismatch", err_type))
                def bad() -> Result:
                    return Result.do(
                        x for x in other.bind(other.err(err_type(7)))
                    )

                output = StringIO()
                with redirect_stderr(output):
                    self.reject(bad, rf"same error type.*expected i32, got {err_type.get_name()}")
                line = bad.__wrapped__.__code__.co_firstlineno + 3
                self.assertIn(f'[test_result_failure_policy.py:{line}:19] Result.do ',
                              output.getvalue())

    def test_option_conversion_errors_point_to_the_call(self):
        malformed = enum['Some': i32, 'NoneVal': i32]
        cases = (
            (i32, 'Expected an Option source, got i32'),
            (Option(i64), 'same value type.*expected i32, got i64'),
            (malformed, 'Option NoneVal must have no payload'),
        )
        for source, message in cases:
            with self.subTest(message=message):
                @compile(suffix=('invalid_option_source', source))
                def bad(value: source) -> Result:
                    return Result.from_option(value, 1)

                output = StringIO()
                with redirect_stderr(output):
                    self.reject(bad, message)
                line = bad.__wrapped__.__code__.co_firstlineno + 2
                self.assertIn(f'[test_result_failure_policy.py:{line}:11]', output.getvalue())

    def reject(self, function, message, exception=TypeError):
        try:
            with self.assertRaisesRegex(exception, message):
                flush_all_pending_outputs()
        finally:
            clear_failed_group(function._binding.group_key)

    def test_option_exhaustion_requires_an_explicit_error(self):
        @compile(suffix="result_option_empty")
        def bad() -> Result:
            return Result.do(x for x in Maybe.bind(Maybe.none()))

        self.reject(bad, "without yielding a value or propagating an error")

    def test_custom_producer_must_cover_every_reachable_exit(self):
        @compile(suffix="result_empty_producer")
        def bad(v: i32) -> Result:
            return Result.do(x for x in empty_if_zero(v))

        output = StringIO()
        with redirect_stderr(output):
            self.reject(bad, "without yielding a value or propagating an error")
        line = bad.__wrapped__.__code__.co_firstlineno + 2
        self.assertIn(f'[test_result_failure_policy.py:{line}:11]', output.getvalue())
        branch = empty_if_zero._yield_func_obj.__code__.co_firstlineno + 2
        self.assertIn(f'test_result_failure_policy.py:{branch}:7: the false branch can reach this point',
                      output.getvalue())

    def test_reachable_producer_return_is_reported(self):
        @compile(suffix="result_producer_return")
        def bad(v: i32) -> Result:
            return Result.do(x for x in return_if_zero(v))

        output = StringIO()
        with redirect_stderr(output):
            self.reject(bad, "without yielding a value or propagating an error")
        line = return_if_zero._yield_func_obj.__code__.co_firstlineno + 3
        self.assertIn(f'test_result_failure_policy.py:{line}:8: the producer can return',
                      output.getvalue())

    def test_filtered_success_requires_an_explicit_error(self):
        @compile(suffix="result_filtered_success")
        def bad(v: i32) -> Result:
            return Result.do(x for x in Result.bind(checked(v)) if False)

        output = StringIO()
        with redirect_stderr(output):
            self.reject(bad, "without yielding a value or propagating an error")
        line = bad.__wrapped__.__code__.co_firstlineno + 2
        self.assertRegex(output.getvalue(),
                         rf'test_result_failure_policy.py:{line}:\d+: this generator filter evaluates to false')

    def test_errno_policy_is_explicit(self):
        reset_slot(73)
        self.assertEqual(record_failure(5), 5)
        self.assertEqual(read_slot(), 73)
        self.assertEqual(record_failure(-4), -1)
        self.assertEqual(read_slot(), 40)

    def test_do_does_not_write_an_explicit_global_channel(self):
        reset_slot(73)
        self.assertEqual(local_composition(-4), -40)
        self.assertEqual(read_slot(), 73)

    def test_unhandled_bind_is_rejected_at_compilation(self):
        @compile(suffix='unhandled_result_bind')
        def unhandled() -> i32:
            for value in Result.bind(Result.err(31)):
                return value
            return i32(-1)

        try:
            with self.assertRaisesRegex(AttributeError, "Effect 'errno'.*no implementation"):
                flush_all_pending_outputs()
        finally:
            clear_failed_group(unhandled._binding.group_key)


if __name__ == '__main__':
    unittest.main()
