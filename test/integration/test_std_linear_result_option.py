"""Shared sum helpers preserve owners instead of discarding losing values."""

import unittest

from pythoc import compile, consume, i32, linear, static
from pythoc.std.result import (
    make_hush, make_note, make_note_lazy, option_wrap, result_wrap,
)


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


Owned, owned = result_wrap(Owner, i32, name='LinearSuccess')
Both, both = result_wrap(Owner, Owner, name='LinearBoth')
Optional, optional = option_wrap(Owner, name='LinearOptional')
Scalar, scalar = result_wrap(i32, i32, name='LinearCompositionValue')
ErrorOwned, error_owned = result_wrap(i32, Owner, name='LinearErrorOnly')
PlainOption, plain_option = option_wrap(i32, name='PlainLazyOption')


@compile
def make_error() -> Owner:
    return acquire(99)


note = make_note_lazy(optional, both, make_error)
hush = make_hush(both, optional, drop)


@compile
def checked(value: i32) -> Owned:
    if value > 0:
        return owned.ok(acquire(value))
    return owned.err(41)


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
    result = scalar.do(release(resource) + 1
                       for resource in owned.bind(checked(value)))
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
        result = optional.some(acquire(7))
    else:
        result = optional.none()

    def fallback() -> Owner:
        return acquire(9)

    resource = optional.unwrap_or_else(result, fallback)
    value: i32 = release(resource)
    return value * 100 + Counts.made * 10 + Counts.released


@compile
def result_fallback(present: i32) -> i32:
    reset()
    result = both.ok(acquire(7))
    if present == 0:
        match result:
            case (Both.Ok, resource):
                drop(resource)
            case (Both.Err, resource):
                drop(resource)
        result = both.err(acquire(3))

    def fallback(error: Owner) -> Owner:
        value: i32 = release(error)
        return acquire(value + 10)

    resource = both.unwrap_or_else(result, fallback)
    value: i32 = release(resource)
    return value * 100 + Counts.made * 10 + Counts.released


@compile
def converted(present: i32) -> i32:
    reset()
    result: Optional
    if present != 0:
        result = optional.some(acquire(7))
    else:
        result = optional.none()
    converted_result = note(result)
    final = hush(converted_result)
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
        source = optional.some(acquire(17))
    else:
        source = optional.none()
    result = optional.do(resource for resource in optional.bind(source))
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
        result = error_owned.ok(5)
    else:
        result = error_owned.err(acquire(9))

    def fallback(error: Owner) -> i32:
        return release(error) + 10

    value = error_owned.unwrap_or_else(result, fallback)
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
        option = plain_option.some(3)
        result = scalar.ok(4)
    else:
        option = plain_option.none()
        result = scalar.err(7)
    a = plain_option.unwrap_or_else(option, option_default)
    b = scalar.unwrap_or_else(result, result_default)
    return a * 1000 + b * 10 + calls


class LinearResultOptionTest(unittest.TestCase):
    def test_factory_does_not_register_invalid_unused_helpers(self):
        self.assertFalse(hasattr(owned, 'unwrap_or'))
        self.assertFalse(hasattr(optional, 'unwrap_or'))
        self.assertFalse(hasattr(owned, 'is_ok'))
        self.assertFalse(hasattr(both, 'bind'))
        self.assertFalse(hasattr(both, 'do'))
        self.assertTrue(hasattr(scalar, 'unwrap_or'))

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
            self.assertFalse(hasattr(error_owned, helper), helper)
        self.assertEqual(error_only_fallback(1), 500)
        self.assertEqual(error_only_fallback(0), 1911)

    def test_plain_fallbacks_are_lazy_and_receive_the_error(self):
        self.assertEqual(plain_fallbacks(1), 3040)
        self.assertEqual(plain_fallbacks(0), 9172)

    def test_implicit_owned_error_loss_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'explicit consumer'):
            make_hush(both, optional)
        with self.assertRaisesRegex(TypeError, 'make_note_lazy'):
            make_note(optional, both)

    def test_z_fallback_cannot_discard_an_owned_error(self):
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)

        @compile(suffix='invalid_owned_error_fallback')
        def losing_error() -> i32:
            def fallback(error: Owner) -> Owner:
                return acquire(5)

            resource = both.unwrap_or_else(both.err(acquire(3)), fallback)
            return release(resource)

        with self.assertRaisesRegex(Exception, '[Ll]inear'):
            losing_error()


if __name__ == '__main__':
    unittest.main()
