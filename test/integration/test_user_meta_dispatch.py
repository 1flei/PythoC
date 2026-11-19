"""User-defined dispatch uses public type reflection and quoted programs."""

import unittest

from pythoc import (
    CompileSession, compile, consume, enum, i32, inline, linear, meta, ptr, static, typeof,
)
from pythoc.std.poly import Poly


@meta.quote
def invoke(target, /, *args):
    return target(*args)


@meta.quote
def invoke_twice(target, value):
    return target(value) + target(value)


@meta.quote
def address_of(value):
    return ptr(value)


@compile
class State:
    calls: static[i32] = 0


@compile
def bump() -> i32:
    State.calls += 1
    return State.calls


@compile
def weighted(a: i32, b: i32) -> i32:
    return a * 10 + b


@compile
class Owner:
    value: i32
    token: linear


@compile
def release(value: Owner) -> i32:
    consume(value.token)
    return value.value


class Dispatcher:
    defer_linear_transfer = True

    def __init__(self, target):
        self.target = target

    def handle_call(self, context, reference, args, location):
        return invoke(self.target, *args)


weighted_dispatch = Dispatcher(weighted)
release_dispatch = Dispatcher(release)


class ExplicitProtocol(Dispatcher):
    @inline
    def __call__(self, a, b):
        return i32(-100)


explicit_dispatch = ExplicitProtocol(weighted)


class PythonCallable:
    @inline
    def __call__(self, value):
        return value + 1


python_callable = PythonCallable()


class DuplicateDispatcher:
    defer_linear_transfer = True

    def handle_call(self, context, reference, args, location):
        return invoke_twice(release, args[0])


duplicate_dispatch = DuplicateDispatcher()


class QuotedAddress:
    @staticmethod
    def evaluate_call_argument(visitor, node):
        return visitor.visit_lvalue(node)

    def handle_call(self, visitor, reference, args, node):
        return address_of(args[0])


quoted_address = QuotedAddress()


@compile
def ordered_call() -> i32:
    State.calls = 0
    return weighted_dispatch(bump(), bump()) * 10 + State.calls


@compile
def explicit_protocol_wins() -> i32:
    return explicit_dispatch(3, 4)


@compile
def consume_once() -> i32:
    item: Owner = (42, linear())
    return release_dispatch(item)


@compile
def borrow_indirect(p: ptr[linear]) -> i32:
    return i32(quoted_address(p[0]) == p)


@compile
def borrow_linear_place() -> i32:
    token = linear()
    same_address = borrow_indirect(ptr(token))
    consume(token)
    return same_address


@compile
def native_signature_reflection() -> i32:
    signature = typeof(weighted)
    return i32(signature.return_type is i32)


@enum(i32)
class Input:
    Value: i32


unprepared = Poly(bump)


class UserMetaDispatchTest(unittest.TestCase):
    def test_evaluated_arguments_are_not_replayed(self):
        self.assertEqual(ordered_call(), 122)

    def test_explicit_protocol_is_not_replaced_by_inline_dunder(self):
        self.assertEqual(explicit_protocol_wins(), 34)

    def test_quoted_argument_preserves_owner(self):
        self.assertEqual(consume_once(), 42)

    def test_quoted_indirect_place_retains_its_address_without_a_new_owner(self):
        self.assertEqual(borrow_linear_place(), 1)

    def test_function_types_are_public_metadata(self):
        self.assertEqual(tuple(typeof(weighted).param_types), (i32, i32))
        self.assertIs(typeof(weighted).return_type, i32)
        self.assertEqual(native_signature_reflection(), 1)

    def test_invalid_resolution_does_not_generate_a_function(self):
        with self.assertRaisesRegex(TypeError, 'missing function overloads'):
            unprepared.resolve(Input)

    def test_inline_dunder_has_only_ordinary_python_call_behavior(self):
        from pythoc.build.output_manager import flush_all_pending_outputs
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)
        self.assertEqual(python_callable(3), 4)
        with CompileSession(), self.assertRaisesRegex(Exception, 'Python call failed'):
            @compile(suffix='unsupported_implicit_inline_callable')
            def invalid(value: i32) -> i32:
                return python_callable(value)

            flush_all_pending_outputs()

    def test_quoted_values_cannot_duplicate_ownership(self):
        from pythoc.build.output_manager import flush_all_pending_outputs
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)
        with CompileSession(), self.assertRaisesRegex(Exception, 'already consumed'):
            @compile(suffix='invalid_meta_argument_duplication')
            def invalid() -> i32:
                item: Owner = (42, linear())
                return duplicate_dispatch(item)

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
