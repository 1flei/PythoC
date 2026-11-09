"""Result propagation is local; a global error slot is an explicit policy."""

import unittest

from pythoc import compile, effect, i32, ptr
from pythoc.build.output_manager import clear_failed_group, flush_all_pending_outputs
from pythoc.logger import set_raise_on_error
from pythoc.std.result import ErrnoSlotProvider, result_wrap


set_raise_on_error(True)
Result, result = result_wrap(i32, i32, name='ExplicitFailurePolicyResult')
provider = ErrnoSlotProvider()
explicit_slot = provider.get_slot(i32)


@compile
def checked(value: i32) -> Result:
    if value >= 0:
        return result.ok(value)
    return result.err(-value * 10)


@compile
def compose(first: i32, second: i32) -> Result:
    return result.do(
        left + right
        for left in result.bind(checked(first))
        for right in result.bind(checked(second))
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
    composed = result.do(
        item + 1 for item in result.bind(compose(value, 3))
    )
    match composed:
        case (Result.Ok, answer):
            return answer
        case (Result.Err, error):
            return -error


with effect(errno=provider, suffix='explicit_result_errno'):
    @compile
    def record_failure(value: i32) -> i32:
        for item in result.bind(checked(value)):
            return item
        return i32(-1)

    @compile
    def local_composition(value: i32) -> i32:
        composed = result.do(item for item in result.bind(checked(value)))
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
            for value in result.bind(result.err(31)):
                return value
            return i32(-1)

        try:
            with self.assertRaisesRegex(AttributeError, "Effect 'errno'.*no implementation"):
                flush_all_pending_outputs()
        finally:
            clear_failed_group(unhandled._binding.group_key)


if __name__ == '__main__':
    unittest.main()
