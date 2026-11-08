"""Compile and execute fixed effect bindings across real module imports."""

import builtins
import os
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pythoc import CompileSession, compile, effect, i32
from effect_lib import import_scope_lib as source
from effect_lib.import_scope_lib import read_scalar as default_scalar
from effect_lib.import_scope_lib import read_value as default_value


@compile
def arena_value() -> i32:
    return i32(11)


@compile
def other_value() -> i32:
    return i32(22)


ArenaProvider = SimpleNamespace(value=arena_value)
OtherProvider = SimpleNamespace(value=other_value)
import_entry = builtins.__import__

with effect(import_provider=ArenaProvider, import_scalar=11, suffix='hook_arena'):
    from effect_lib.import_scope_lib import read_value as arena_read
    from effect_lib.import_scope_lib import read_with_compile_suffix as arena_suffix
    from effect_lib.import_scope_lib import yield_value as arena_yield
    assert source.read_value is default_value
    assert builtins.__import__ is import_entry

    with effect(import_provider=OtherProvider, import_scalar=22, suffix='hook_other'):
        from effect_lib.import_scope_lib import read_value as other_read
        from effect_lib.import_scope_lib import read_scalar as other_scalar

    from effect_lib.import_scope_facade import read_value as arena_again
    from effect_lib.import_scope_cold import (
        observed_provider, observed_suffix, read_cold as arena_cold, read_pinned,
    )

    @compile
    def lexical_call() -> i32:
        return default_value()

assert arena_read is arena_again
assert observed_suffix is None
assert observed_provider is source.DefaultProvider
assert read_pinned._binding.effect_suffix == 'module_owned'
assert builtins.__import__ is import_entry

from effect_lib.import_scope_cold import read_cold as default_cold

with effect(import_pin=11, suffix='hook_pinned'):
    from effect_lib.import_scope_pinned import read_pin


@compile
def block_scalar() -> i32:
    with effect(import_scalar=99):
        return i32(effect.import_scalar)


@compile
def bound_calls() -> i32:
    return default_value() + arena_read() + other_read() + lexical_call()


@compile
def bound_yield() -> i32:
    value: i32 = 0
    for item in arena_yield():
        value += item
    return value


session = CompileSession.current()
barrier = threading.Barrier(2)
thread_bindings = {}
thread_errors = []


def import_in_thread(label, provider):
    token = session.activate()
    try:
        with effect(import_provider=provider, suffix=label):
            barrier.wait(timeout=10)
            from effect_lib.import_scope_lib import read_value
            barrier.wait(timeout=10)
            thread_bindings[label] = read_value
    except BaseException as error:
        thread_errors.append(error)
    finally:
        session.deactivate(token)


threads = [threading.Thread(target=import_in_thread, args=arguments) for arguments in (
    ('thread_arena', ArenaProvider), ('thread_other', OtherProvider),
)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=30)
assert all(not thread.is_alive() for thread in threads)
assert not thread_errors, thread_errors


class TestEffectImportScope(unittest.TestCase):
    def test_python_calls_use_fixed_versions(self):
        self.assertEqual(default_value(), 7)
        self.assertEqual(arena_read(), 11)
        self.assertEqual(other_read(), 22)
        self.assertEqual(arena_suffix(), 11)

    def test_compiled_calls_preserve_lexical_bindings(self):
        self.assertEqual(lexical_call(), 7)
        self.assertEqual(bound_calls(), 47)

    def test_scalar_resolution_uses_module_and_block_bindings(self):
        self.assertEqual(default_scalar(), 7)
        self.assertEqual(other_scalar(), 22)
        self.assertEqual(block_scalar(), 99)

    def test_cold_import_and_module_owned_specialization(self):
        self.assertEqual(arena_cold(), 11)
        self.assertEqual(default_cold(), 13)
        self.assertEqual(read_pinned(), 31)

    def test_yield_import_has_a_fixed_specialization(self):
        self.assertEqual(bound_yield(), 11)

    def test_library_pinning_wins_during_cold_import(self):
        self.assertEqual(read_pin(), 37)

    def test_parallel_imports_produce_isolated_native_bindings(self):
        self.assertEqual(thread_bindings['thread_arena'](), 11)
        self.assertEqual(thread_bindings['thread_other'](), 22)
        self.assertIs(source.read_value, default_value)


if __name__ == '__main__':
    unittest.main()
