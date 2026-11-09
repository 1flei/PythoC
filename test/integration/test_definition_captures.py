"""Inline expansion preserves definition-site bindings and capture modes."""

import unittest

from pythoc import compile, consume, i32, inline, linear, ptr, void
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.builtin_entities.instantiate import instantiate
from pythoc.session import CompileSession
from test.utils.test_utils import DeferredTestCase


scope_value = 7
consumer_value = 41


@inline
def inline_local() -> i32:
    name = 7
    return name


@inline
def inline_global() -> i32:
    return i32(scope_value)


@inline
def increment_pointer(target: ptr[i32]) -> void:
    target[0] += 1


@inline
def unresolved_inline() -> i32:
    return undefined_at_definition


@inline
def inline_comprehension() -> i32:
    values = (item + i32(scope_value) for item in (1, 2))
    total: i32 = 0
    for item in values:
        total += item
    return total


@compile
def global_values() -> i32:
    yield i32(scope_value)


def make_values(consumer_value):
    @compile
    def values() -> i32:
        yield i32(consumer_value)
    return values


factory_values = make_values(7)


@compile
def inline_caller(value: i32) -> i32:
    name: i32 = value
    result = inline_local()
    return name * 100 + result


@compile
def inline_global_caller(scope_value: i32) -> i32:
    return inline_global() + scope_value


@compile
def inline_comprehension_caller(scope_value: i32) -> i32:
    return inline_comprehension() + scope_value


@compile
def yield_global_caller(scope_value: i32) -> i32:
    for current in global_values():
        return current + scope_value
    return 0


@compile
def consumer_globals() -> i32:
    for current in factory_values():
        return current + i32(consumer_value)
    return 0


@compile
def reference_write(value: i32) -> i32:
    name: i32 = 1

    def increment() -> i32:
        name += 1
        return name

    name = 10
    result: i32 = 0
    if value > 0:
        name: i32 = 100
        result = increment() * 1000 + name
    return result + name


@compile
def yield_reference_write(value: i32) -> i32:
    name: i32 = 1

    def increment() -> i32:
        name += 1
        yield name

    pending = increment()
    name = 10
    result: i32 = 0
    if value > 0:
        name: i32 = 100
        for current in pending:
            result = current * 1000 + name
    return result + name


@compile
def lambda_reference(value: i32) -> i32:
    name: i32 = 1
    read = lambda: name
    name = 10
    if value > 0:
        name: i32 = 100
        return read() * 1000 + name
    return 0


@compile
def later_local_does_not_capture() -> i32:
    def read() -> i32:
        return i32(scope_value)

    scope_value: i32 = 41
    return read() + scope_value


@compile
def nested_reference(value: i32) -> i32:
    name: i32 = 1

    def outer() -> i32:
        def inner() -> i32:
            return name
        return inner()

    name = 10
    if value > 0:
        name: i32 = 100
        return outer() * 1000 + name
    return 0


@compile
def eager_default_and_reference() -> i32:
    name: i32 = 1

    def read(saved: i32 = name) -> i32:
        return saved * 100 + name

    name = 10
    if name > 0:
        name: i32 = 100
        return read()
    return 0


@compile
def instantiated_value_and_direct_reference() -> i32:
    name: i32 = 1

    def read() -> i32:
        return name

    api = instantiate(read)
    state = api.init()
    name = 10
    return read() * 100 + api.call(ptr(state))


@compile
def expression_reference(value: i32) -> i32:
    name: i32 = 3
    values = (name + item for item in (1, 2))
    name = 10
    total: i32 = 0
    if value > 0:
        name: i32 = 100
        for item in values:
            total += item
    return total


@compile
def expression_target_shadows_iterable() -> i32:
    item = (1, 2)
    values = (item for item in item)
    total: i32 = 0
    for current in values:
        total += current
    return total + i32(len(item))


@compile
def void_capture() -> i32:
    count: i32 = 0

    def increment() -> void:
        count += 1
        return

    increment()
    increment_pointer(ptr(count))
    return count


@compile
def compiletime_reference() -> i32:
    value = 1

    def read() -> i32:
        return i32(value)

    api = instantiate(read)
    state = api.init()
    value = 7
    return read() * 100 + api.call(ptr(state))


class DefinitionCaptureTest(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_module_inline_locals_do_not_write_caller_locals(self):
        self.assertEqual(inline_caller(41), 4107)

    def test_module_inline_globals_do_not_read_caller_parameters(self):
        self.assertEqual(inline_global_caller(41), 48)

    def test_inline_comprehension_retains_its_local_bindings(self):
        self.assertEqual(inline_comprehension_caller(41), 58)

    def test_module_yield_globals_do_not_read_caller_parameters(self):
        self.assertEqual(yield_global_caller(41), 48)

    def test_producer_and_consumer_keep_their_own_global_bindings(self):
        self.assertEqual(consumer_globals(), 48)

    def test_closure_writes_its_definition_binding_through_call_site_shadow(self):
        self.assertEqual(reference_write(1), 11111)

    def test_stored_yield_closure_keeps_the_same_reference(self):
        self.assertEqual(yield_reference_write(1), 11111)

    def test_lambda_reads_updated_definition_binding(self):
        self.assertEqual(lambda_reference(1), 10100)

    def test_new_call_site_local_cannot_become_a_capture(self):
        self.assertEqual(later_local_does_not_capture(), 48)

    def test_nested_closure_preserves_a_transitive_capture(self):
        self.assertEqual(nested_reference(1), 10100)

    def test_default_is_a_value_while_free_variable_is_a_reference(self):
        self.assertEqual(eager_default_and_reference(), 110)

    def test_instantiation_still_copies_values_into_independent_state(self):
        self.assertEqual(instantiated_value_and_direct_reference(), 1001)

    def test_stored_generator_expression_retains_definition_bindings(self):
        self.assertEqual(expression_reference(1), 23)

    def test_generator_target_does_not_replace_the_outer_iterable_binding(self):
        self.assertEqual(expression_target_shadows_iterable(), 5)

    def test_void_returns_survive_definition_global_aliasing(self):
        self.assertEqual(void_capture(), 2)

    def test_compiletime_binding_updates_remain_visible_to_direct_calls(self):
        self.assertEqual(compiletime_reference(), 701)

    def test_unresolved_global_cannot_capture_a_new_caller_local(self):
        with CompileSession(), self.assertRaisesRegex(NameError, 'undefined_at_definition'):
            @compile(suffix='unbound_definition_name')
            def invalid() -> i32:
                undefined_at_definition: i32 = 41
                return unresolved_inline()

            flush_all_pending_outputs()

    def test_reference_aliases_cannot_duplicate_a_linear_owner(self):
        with CompileSession(), self.assertRaisesRegex(RuntimeError, 'already consumed'):
            @compile(suffix='definition_linear_double_use')
            def invalid() -> i32:
                token: linear = linear()

                def take() -> void:
                    consume(token)

                take()
                take()
                return 0

            flush_all_pending_outputs()

    def test_linear_reference_retains_the_original_owner(self):
        with CompileSession():
            @compile(suffix='definition_linear_capture')
            def use_original(value: i32) -> i32:
                token: linear = linear()

                def take() -> i32:
                    consume(token)
                    return 7

                result: i32 = 0
                if value > 0:
                    token: linear = linear()
                    result = take()
                    consume(token)
                else:
                    result = take()
                return result

            flush_all_pending_outputs()
            self.assertEqual(use_original(0), 7)
            self.assertEqual(use_original(1), 7)


if __name__ == '__main__':
    unittest.main()
