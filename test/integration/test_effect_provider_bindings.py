"""Compile and execute provider bindings without callable-wide role state."""

import os
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import compile, effect, i32
from effect_lib import import_scope_roles as source
from effect_lib import import_scope_role_bridge as bridge


original = source.read_role
with effect(role_scalar=11, suffix='role_before'):
    from effect_lib.import_scope_roles import read_role as before
    from effect_lib.import_scope_roles import read_unregistered as before_temporary

with effect(unused_provider=SimpleNamespace(value=source.read_unregistered),
            suffix='role_unused'):
    pass

with effect(role_scalar=22, suffix='role_after'):
    from effect_lib.import_scope_roles import read_role as after
    from effect_lib.import_scope_roles import read_helper as after_helper
    from effect_lib.import_scope_roles import read_provider as default_provider
    from effect_lib.import_scope_roles import yield_role as after_yield
    from effect_lib.import_scope_roles import read_unregistered as after_temporary

with effect(role_provider=SimpleNamespace(value=original),
            role_scalar=33, suffix='role_simultaneous'):
    from effect_lib.import_scope_roles import read_role as simultaneous
    from effect_lib.import_scope_roles import read_provider as fixed_provider
    captured_provider = effect.get_effect_impl('role_provider')

effect.pinned_role_provider = SimpleNamespace(value=original)
effect.default(other_role_provider=SimpleNamespace(value=original))
with effect(role_scalar=44, suffix='role_registered'):
    from effect_lib.import_scope_roles import read_role as registered

with effect(role_provider=source.DefaultProvider,
            role_scalar=55, suffix='role_cross_group'):
    from effect_lib.import_scope_role_bridge import (
        read_imported, read_namespace, read_provider as cross_group_provider,
        yield_namespace,
    )


@compile
def lexical_call() -> i32:
    return original()


@compile
def specialized_call() -> i32:
    return after() + after_helper() + simultaneous() + fixed_provider()


@compile
def yield_call() -> i32:
    result: i32 = 0
    for value in after_yield():
        result += value
    return result


@compile
def namespace_yield_call() -> i32:
    result: i32 = 0
    for value in yield_namespace():
        result += value
    return result


class TestEffectProviderBindings(unittest.TestCase):
    def test_temporary_provider_has_no_lasting_role(self):
        self.assertEqual(before(), 11)
        self.assertEqual(after(), 22)
        self.assertEqual(before_temporary(), 11)
        self.assertEqual(after_temporary(), 22)
        self.assertEqual(registered(), 44)
        self.assertIs(source.read_role, original)

    def test_provider_edge_keeps_lexical_binding(self):
        self.assertEqual(simultaneous(), 33)
        self.assertEqual(fixed_provider(), 7)
        self.assertEqual(default_provider(), 7)
        self.assertIs(captured_provider.value, original)

    def test_ordinary_dependency_and_provider_can_share_declaration(self):
        self.assertEqual(after_helper(), 22)
        self.assertEqual(lexical_call(), 7)
        self.assertEqual(specialized_call(), 84)

    def test_yield_dependency_can_specialize_a_provider_declaration(self):
        self.assertEqual(yield_call(), 22)

    def test_cross_group_namespace_and_provider_edges_are_independent(self):
        self.assertEqual(read_imported(), 55)
        self.assertEqual(read_namespace(), 55)
        self.assertEqual(cross_group_provider(), 7)
        self.assertIs(bridge.Provider, source.DefaultProvider)
        self.assertIs(bridge.Provider.value, original)

    def test_yield_namespace_can_specialize_a_provider_declaration(self):
        self.assertEqual(namespace_yield_call(), 55)

    def test_captured_provider_bindings_are_not_rewritten(self):
        for wrapper in (fixed_provider, cross_group_provider):
            binding = wrapper._binding.captured_effect_context['role_provider']
            self.assertIs(binding.value, original)


if __name__ == '__main__':
    unittest.main()
