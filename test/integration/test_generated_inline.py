"""Generated bodies use ordinary inline binding, control flow and ownership."""

import ast
import inspect
import unittest
from unittest.mock import patch

from pythoc import (
    CompileSession, compile, consume, func, i32, inline, linear, meta, ptr, typeof, void,
)
from pythoc.build.output_manager import get_output_manager, flush_all_pending_outputs


@meta.quote
def choose_body(value):
    local: i32 = value + 10
    if value < 0:
        return -local
    return local


choice_definition = meta.func(
    'generated_choice', [('value', i32)], i32, choose_body('value'))
choose = meta.inline_generated(choice_definition)
native_choice = meta.compile_generated(choice_definition)
void_definition = meta.func(
    'generated_default_void', [('value', i32)], None, choose_body('value'))
expanded_void = meta.inline_generated(void_definition)
native_void = meta.compile_generated(void_definition)


@compile
class Owner:
    value: i32
    token: linear


@meta.quote
def owner_body(value):
    resource: Owner
    if value < 0:
        resource = (-value, linear())
        return resource
    resource = (value + 1, linear())
    return resource


make_owner = meta.inline_generated(meta.func(
    'generated_owner', [('value', i32)], Owner, owner_body('value'),
    required_globals={'Owner': Owner, 'linear': linear}))


@compile
def select_twice() -> i32:
    local: i32 = 500
    value: i32 = 100
    return choose(value=2) * 100 + choose(-3) + local + value


@compile
def owned_return(value: i32) -> i32:
    resource = make_owner(value)
    consume(resource.token)
    return resource.value


@compile
def increment(target: ptr[i32]) -> void:
    target[0] += 1


@inline
def forward_void(target: ptr[i32]) -> None:
    return increment(target)


@compile
def forwarded_void() -> i32:
    value: i32 = 0
    forward_void(ptr(value))
    return value


@compile
def default_void_returns() -> i32:
    expanded_void(2)
    expanded_void(-3)
    native_void(2)
    native_void(-3)
    return 0


class GeneratedInlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manager = get_output_manager()
        with patch.object(manager, 'queue_compilation',
                          side_effect=AssertionError('Generated inline registered native work')):
            flush_all_pending_outputs()

    def test_branches_keywords_and_local_names(self):
        self.assertEqual(select_twice(), 1793)

    def test_owned_results_cross_return_merges_once(self):
        self.assertEqual(owned_return(-7), 7)
        self.assertEqual(owned_return(7), 8)

    def test_definition_can_be_compiled_and_expanded(self):
        self.assertEqual(native_choice(2), 12)
        self.assertEqual(native_choice(-3), -7)

    def test_python_inline_shares_void_return_handling(self):
        self.assertEqual(forwarded_void(), 1)

    def test_none_return_metadata_means_void_for_both_paths(self):
        self.assertIs(inspect.signature(expanded_void).return_annotation,
                      typeof(native_void).return_type)
        self.assertEqual(default_void_returns(), 0)

    def test_generated_inline_is_not_a_python_or_native_function(self):
        self.assertTrue(choose._pc_compile_level_callable)
        self.assertFalse(getattr(choose, '_is_compiled', False))
        with self.assertRaisesRegex(TypeError, 'only be called in compiled code'):
            choose(3)

    def test_native_attributes_are_rejected(self):
        with self.assertRaisesRegex(TypeError, 'Native function attributes'):
            meta.inline_generated(meta.func(
                'invalid_attrs', [('value', i32)], i32,
                choose_body('value'), attrs={'noinline'}))

    def test_inline_body_cannot_be_stored_as_a_native_function(self):
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)
        with CompileSession(), self.assertRaisesRegex(TypeError, 'Compile-level callable'):
            @compile(suffix='invalid_generated_inline_pointer')
            def invalid() -> i32:
                callback: func[i32, i32] = choose
                return callback(3)

            flush_all_pending_outputs()

    def test_void_return_cannot_discard_owned_values(self):
        from pythoc.logger import set_raise_on_error
        set_raise_on_error(True)
        discard = meta.inline_generated(meta.func(
            'invalid_void_owner', [('resource', Owner)], void,
            ast.parse('return resource').body))
        with CompileSession(), self.assertRaisesRegex(TypeError, 'Cannot discard a linear'):
            @compile(suffix='invalid_generated_void_discard')
            def invalid() -> i32:
                resource: Owner = (7, linear())
                discard(resource)
                return 0

            flush_all_pending_outputs()


if __name__ == '__main__':
    unittest.main()
