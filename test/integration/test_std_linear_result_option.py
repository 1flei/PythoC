"""Shared sum helpers preserve owners instead of discarding losing values."""

import unittest

from pythoc import compile, consume, i32, linear, static
from pythoc.std.result import Option as OptionType, Result as ResultType


@compile
class Owner:
    value: i32
    token: linear


@compile
class Counts:
    made: static[i32] = 0
    released: static[i32] = 0


@compile
def reset() -> None:
    Counts.made = 0
    Counts.released = 0


@compile
def acquire(value: i32) -> Owner:
    resource: Owner
    resource.value = value
    resource.token = linear()
    Counts.made = Counts.made + 1
    return resource


@compile
def release(resource: Owner) -> i32:
    value: i32 = resource.value
    consume(resource.token)
    Counts.released = Counts.released + 1
    return value


@compile
def drop(resource: Owner) -> None:
    release(resource)


Owned = ResultType(Owner, i32, name='LinearSuccess')
Both = ResultType(Owner, Owner, name='LinearBoth')
Optional = OptionType(Owner, name='LinearOptional')
Scalar = ResultType(i32, i32, name='LinearCompositionValue')
ErrorOwned = ResultType(i32, Owner, name='LinearErrorOnly')
PlainOption = OptionType(i32, name='PlainLazyOption')


@compile
def make_error() -> Owner:
    return acquire(99)


@compile
def checked(value: i32) -> Owned:
    if value > 0:
        return Owned.ok(acquire(value))
    return Owned.err(41)


@compile
def tag_inspection() -> i32:
    reset()
    result = checked(7)
    valid: i32 = i32(result[0] == Owned.Ok)
    valid = valid + i32(result[0] != Owned.Err)
    value: i32
    match result:
        case (Owned.Ok, resource):
            value = release(resource)
        case (Owned.Err, error):
            value = -error
    return value * 1000 + valid * 100 + Counts.made * 10 + Counts.released


@compile
def composed(value: i32) -> i32:
    result = Scalar.do(release(resource) + 1
                       for resource in Owned.bind(checked(value)))
    match result:
        case (Scalar.Ok, value):
            return value
        case (Scalar.Err, error):
            return -error


@compile
def option_fallback(present: i32) -> i32:
    reset()
    result: Optional
    if present != 0:
        result = Optional.some(acquire(7))
    else:
        result = Optional.none()

    def fallback() -> Owner:
        return acquire(9)

    resource = Optional.unwrap_or_else(result, fallback)
    value: i32 = release(resource)
    return value * 100 + Counts.made * 10 + Counts.released


@compile
def result_fallback(present: i32) -> i32:
    reset()
    result = Both.ok(acquire(7))
    if present == 0:
        match result:
            case (Both.Ok, resource):
                drop(resource)
            case (Both.Err, resource):
                drop(resource)
        result = Both.err(acquire(3))

    def fallback(error: Owner) -> Owner:
        value: i32 = release(error)
        return acquire(value + 10)

    resource = Both.unwrap_or_else(result, fallback)
    value: i32 = release(resource)
    return value * 100 + Counts.made * 10 + Counts.released


@compile
def converted(present: i32) -> i32:
    reset()
    result: Optional
    if present != 0:
        result = Optional.some(acquire(7))
    else:
        result = Optional.none()
    converted_result = Both.from_option_else(result, make_error)
    final = Optional.from_result(converted_result, drop_error=drop)
    value: i32 = 0
    match final:
        case (Optional.Some, resource):
            value = release(resource)
        case Optional.NoneVal:
            value = -1
    return value * 100 + Counts.made * 10 + Counts.released


@compile
def option_composed(present: i32) -> i32:
    reset()
    source: Optional
    if present != 0:
        source = Optional.some(acquire(17))
    else:
        source = Optional.none()
    result = Optional.do(resource for resource in Optional.bind(source))
    value: i32 = -1
    match result:
        case (Optional.Some, resource):
            value = release(resource)
        case Optional.NoneVal:
            pass
    return value * 100 + Counts.made * 10 + Counts.released


@compile
def error_only_fallback(success: i32) -> i32:
    reset()
    result: ErrorOwned
    if success != 0:
        result = ErrorOwned.ok(5)
    else:
        result = ErrorOwned.err(acquire(9))

    def fallback(error: Owner) -> i32:
        return release(error) + 10

    value = ErrorOwned.unwrap_or_else(result, fallback)
    return value * 100 + Counts.made * 10 + Counts.released


@compile
def plain_fallbacks(present: i32) -> i32:
    calls: i32 = 0

    def option_default() -> i32:
        calls += 1
        return 9

    def result_default(error: i32) -> i32:
        calls += 1
        return error + 10

    option: PlainOption
    result: Scalar
    if present != 0:
        option = PlainOption.some(3)
        result = Scalar.ok(4)
    else:
        option = PlainOption.none()
        result = Scalar.err(7)
    a = PlainOption.unwrap_or_else(option, option_default)
    b = Scalar.unwrap_or_else(result, result_default)
    return a * 1000 + b * 10 + calls


class LinearResultOptionTest(unittest.TestCase):
    def test_factory_does_not_register_invalid_unused_helpers(self):
        self.assertFalse(hasattr(Owned, 'unwrap_or'))
        self.assertFalse(hasattr(Optional, 'unwrap_or'))
        self.assertFalse(hasattr(Owned, 'is_ok'))
        self.assertFalse(hasattr(Both, 'bind'))
        self.assertFalse(hasattr(Both, 'do'))
        self.assertTrue(hasattr(Scalar, 'unwrap_or'))

    def test_tag_inspection_retains_the_owner(self):
        self.assertEqual(tag_inspection(), 7211)

    def test_owned_success_composes_with_result_do(self):
        self.assertEqual(composed(7), 8)
        self.assertEqual(composed(0), -41)

    def test_option_fallback_is_lazy(self):
        self.assertEqual(option_fallback(1), 711)
        self.assertEqual(option_fallback(0), 911)

    def test_result_fallback_consumes_owned_error(self):
        self.assertEqual(result_fallback(1), 711)
        self.assertEqual(result_fallback(0), 1333)

    def test_lazy_note_and_explicit_error_consumer(self):
        self.assertEqual(converted(1), 711)
        self.assertEqual(converted(0), -89)

    def test_linear_option_bind_do_preserves_exact_resource_counts(self):
        self.assertEqual(option_composed(1), 1711)
        self.assertEqual(option_composed(0), -100)

    def test_copyable_success_with_owned_error(self):
        for helper in ('bind', 'do', 'unwrap_or', 'is_ok', 'is_err'):
            self.assertFalse(hasattr(ErrorOwned, helper), helper)
        self.assertEqual(error_only_fallback(1), 500)
        self.assertEqual(error_only_fallback(0), 1911)

    def test_plain_fallbacks_are_lazy_and_receive_the_error(self):
        self.assertEqual(plain_fallbacks(1), 3040)
        self.assertEqual(plain_fallbacks(0), 9172)

    def test_implicit_owned_error_loss_is_rejected(self):
        from pythoc.build.output_manager import clear_failed_group
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)

        @compile(suffix='invalid_result_conversion_drop')
        def lose_error() -> i32:
            value = Optional.from_result(Both.err(acquire(3)))
            return 0

        with self.assertRaisesRegex(Exception, 'explicit consumer'):
            lose_error()
        clear_failed_group(lose_error._binding.group_key)

        @compile(suffix='invalid_option_conversion_eager_error')
        def lose_eager_error() -> i32:
            value = Both.from_option(Optional.some(acquire(3)), acquire(4))
            return 0

        with self.assertRaisesRegex(Exception, 'from_option_else'):
            lose_eager_error()
        clear_failed_group(lose_eager_error._binding.group_key)

    def test_z_fallback_cannot_discard_an_owned_error(self):
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)

        @compile(suffix='invalid_owned_error_fallback')
        def losing_error() -> i32:
            def fallback(error: Owner) -> Owner:
                return acquire(5)

            resource = Both.unwrap_or_else(Both.err(acquire(3)), fallback)
            return release(resource)

        with self.assertRaisesRegex(Exception, '[Ll]inear'):
            losing_error()


if __name__ == '__main__':
    unittest.main()
