"""Effect import binding and execution-context isolation regressions."""

import asyncio
import builtins
import importlib
from importlib.util import spec_from_loader
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
from pathlib import Path
import unittest

from pythoc.effect import (
    Effect, _install_import_hook, _make_import_hook, capture_effect_context, effect,
    restore_effect_context,
)
from pythoc.session import CompileSession


def make_callable(value=0, suffix=None):
    """A compiler-free stand-in for the specialization entry point."""
    def function():
        return value

    function._is_compiled = True
    function._binding = SimpleNamespace(
        effect_suffix=suffix, group_key=None, original_name='function',
    )
    function._mangled_name = 'function'
    specializations = {}

    def specialize(target_suffix, bindings):
        if target_suffix not in specializations:
            specializations[target_suffix] = make_callable(
                bindings['import_value'], target_suffix,
            )
        return specializations[target_suffix]

    function.get_effect_specialized = specialize
    return function


class TestEffectImportScope(unittest.TestCase):
    def setUp(self):
        self.session = CompileSession()
        self.session_token = self.session.activate()
        self.original_import = builtins.__import__
        self.module = ModuleType('_pc_effect_scope_fixture')
        self.module.__spec__ = spec_from_loader(self.module.__name__, None)
        self.module.function = make_callable()
        sys.modules[self.module.__name__] = self.module
        self.addCleanup(sys.modules.pop, self.module.__name__, None)
        self.addCleanup(self.session.deactivate, self.session_token)
        self.addCleanup(setattr, builtins, '__import__', self.original_import)

    def test_context_keeps_import_entry_stable(self):
        with effect(import_value=11, suffix='arena'):
            self.assertIs(builtins.__import__, self.original_import)
            with effect(import_value=22, suffix='other'):
                self.assertIs(builtins.__import__, self.original_import)
        self.assertIs(builtins.__import__, self.original_import)

    def test_installation_is_idempotent_and_preserves_later_importers(self):
        self.assertIs(_install_import_hook(), self.original_import)

        def observer(*args, **kwargs):
            return self.original_import(*args, **kwargs)

        builtins.__import__ = observer
        self.assertIs(_install_import_hook(), self.original_import)
        self.assertIs(builtins.__import__, observer)

    def test_original_module_and_existing_reference_are_unchanged(self):
        from _pc_effect_scope_fixture import function as default
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function as arena
            self.assertIs(self.module.function, default)
            self.assertEqual(default(), 0)
        with effect(import_value=22, suffix='other'):
            self.assertEqual(arena(), 11)
        self.assertIs(sys.modules[self.module.__name__], self.module)

    def test_nested_bindings_and_cached_aliases_are_fixed(self):
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function as outer
            with effect(import_value=22, suffix='other'):
                from _pc_effect_scope_fixture import function as inner
            from _pc_effect_scope_fixture import function as again
        self.assertIs(outer, again)
        self.assertEqual((outer(), inner()), (11, 22))

    def test_existing_specialization_is_not_rebound(self):
        self.module.function = make_callable(33, 'pinned')
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function
        self.assertIs(function, self.module.function)
        self.assertEqual(function(), 33)

    def test_suffix_reuse_rejects_different_bindings_before_cache_lookup(self):
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function as first
        with effect(import_value=22, suffix='arena'):
            with self.assertRaisesRegex(ValueError, 'different import bindings'):
                from _pc_effect_scope_fixture import function
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function as again
        self.assertIs(first, again)
        self.assertEqual(first(), 11)
        self.assertEqual(self.module.function(), 0)

    def test_suffix_bindings_preserve_python_value_types(self):
        with effect(import_value=True, suffix='typed'):
            from _pc_effect_scope_fixture import function
        self.assertIs(function(), True)
        with effect(import_value=1, suffix='typed'):
            with self.assertRaisesRegex(ValueError, 'different import bindings'):
                from _pc_effect_scope_fixture import function

    def test_compile_suffix_substring_does_not_skip_specialization(self):
        self.module.function._mangled_name = 'function_arena_compile_suffix'
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function
        self.assertEqual(function(), 11)

    def test_python_dynamic_export_is_not_probed_twice(self):
        reads = []

        def lookup(name):
            if name != 'value':
                raise AttributeError(name)
            reads.append(name)
            return len(reads)

        self.module.__getattr__ = lookup
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import value
        self.assertEqual(value, 1)
        self.assertEqual(reads, ['value'])

    def test_mixed_dynamic_and_compiled_exports(self):
        reads = []

        def lookup(name):
            if name != 'value':
                raise AttributeError(name)
            reads.append(name)
            return len(reads)

        self.module.__getattr__ = lookup
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function, value
        self.assertEqual((function(), value), (11, 1))
        self.assertEqual(reads, ['value'])

    def test_dynamic_compiled_export_is_bound_once(self):
        reads = []
        declaration = self.module.function

        def lookup(name):
            if name != 'dynamic':
                raise AttributeError(name)
            reads.append(name)
            return declaration

        self.module.__getattr__ = lookup
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import dynamic
        self.assertEqual(dynamic(), 11)
        self.assertEqual(reads, ['dynamic'])

    def test_cold_import_respects_pinning_registered_by_library(self):
        name = '_pc_effect_cold_pin_fixture'
        self.module.make_callable = make_callable
        self.addCleanup(sys.modules.pop, name, None)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, name + '.py').write_text(
                'from pythoc import effect\n'
                'from _pc_effect_scope_fixture import function\n'
                'from _pc_effect_scope_fixture import make_callable\n'
                'effect.import_value = 37\n'
                'def __getattr__(name):\n'
                '    if name == "dynamic":\n'
                '        return make_callable(effect.get_effect_impl("import_value"), "generated")\n'
                '    raise AttributeError(name)\n'
            )
            sys.path.insert(0, directory)
            try:
                with restore_effect_context({'import_value': 11}):
                    with effect(import_value=11, suffix='cold_pin'):
                        namespace = {}
                        exec(f'from {name} import function, dynamic', namespace)
            finally:
                sys.path.remove(directory)
        self.assertEqual(namespace['function'](), 37)
        self.assertEqual(namespace['dynamic'](), 37)

    def test_private_exports_do_not_collide_with_view_state(self):
        self.module._request = 'ordinary request'
        self.module._source_module = 'ordinary source'
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import (
                function, _request, _source_module, __dict__,
            )
        self.assertEqual(function(), 11)
        self.assertEqual(_request, 'ordinary request')
        self.assertEqual(_source_module, 'ordinary source')
        self.assertIs(__dict__, vars(self.module))

    def test_provider_registration_does_not_change_callable_identity(self):
        for registration in ('scoped', 'default', 'direct'):
            for container in ('function', 'namespace', 'class'):
                with self.subTest(registration=registration, container=container):
                    original = make_callable()
                    self.module.function = original
                    metadata = dict(vars(original))
                    if container == 'namespace':
                        provider = SimpleNamespace(value=original)
                    elif container == 'class':
                        provider = type('Provider', (), {'value': original})
                    else:
                        provider = original

                    def check_binding():
                        with effect(import_value=11,
                                    suffix=f'role_{registration}_{container}'):
                            from _pc_effect_scope_fixture import function
                        self.assertIsNot(function, original)
                        self.assertEqual(function(), 11)
                        self.assertIs(effect.get_effect_impl('provider'), provider)
                        self.assertEqual(vars(original), metadata)

                    if registration == 'scoped':
                        with effect(provider=provider, suffix='provider_scope'):
                            check_binding()
                    elif registration == 'default':
                        effect.default(provider=provider)
                        check_binding()
                    else:
                        effect.provider = provider
                        check_binding()

    def test_temporary_provider_does_not_affect_later_imports(self):
        original = self.module.function
        with effect(import_value=11, suffix='before_provider'):
            from _pc_effect_scope_fixture import function as before
        with effect(provider=SimpleNamespace(value=original), suffix='temporary'):
            pass
        with effect(import_value=22, suffix='after_provider'):
            from _pc_effect_scope_fixture import function as after
        self.assertEqual(before(), 11)
        self.assertEqual(after(), 22)
        self.assertIs(self.module.function, original)

    def test_provider_role_does_not_cross_sessions(self):
        original = self.module.function
        with effect(provider=SimpleNamespace(value=original), suffix='temporary'):
            captured = capture_effect_context()
        other_session = CompileSession()
        token = other_session.activate()
        try:
            with effect(import_value=22, suffix='other_session'):
                from _pc_effect_scope_fixture import function
            self.assertEqual(function(), 22)
            self.assertIsNone(effect.get_effect_impl('provider'))
        finally:
            other_session.deactivate(token)
        with restore_effect_context(captured):
            self.assertIs(effect.get_effect_impl('provider').value, original)

    def test_provider_role_does_not_cross_threads(self):
        original = self.module.function
        entered = threading.Barrier(2)
        leave = threading.Barrier(2)
        results = []
        errors = []

        def register_provider():
            token = self.session.activate()
            try:
                with effect(provider=SimpleNamespace(value=original), suffix='provider'):
                    entered.wait(timeout=10)
                    leave.wait(timeout=10)
            except BaseException as error:
                errors.append(error)
            finally:
                self.session.deactivate(token)

        thread = threading.Thread(target=register_provider)
        thread.start()
        try:
            entered.wait(timeout=10)
            with effect(import_value=22, suffix='other_thread'):
                from _pc_effect_scope_fixture import function
            results.append(function())
            leave.wait(timeout=10)
        finally:
            thread.join(timeout=15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [22])

    def test_custom_module_lookup_is_preserved(self):
        def ordinary():
            return 55

        class CustomModule(ModuleType):
            def __getattribute__(self, name):
                if name == 'function':
                    return ordinary
                return super().__getattribute__(name)

        self.module.__class__ = CustomModule
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function
        self.assertIs(function, ordinary)

    def test_python_callable_metadata_is_not_evaluated(self):
        class OrdinaryCallable:
            @property
            def __dict__(self):
                raise AssertionError('compiler inspected a Python descriptor')

            def __call__(self):
                return 55

        self.module.ordinary = OrdinaryCallable()
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function, ordinary
        self.assertEqual(function(), 11)
        self.assertIs(ordinary, self.module.ordinary)

    def test_plain_reexport_and_relative_import(self):
        package = ModuleType('_pc_effect_scope_package')
        package.__path__ = []
        target = ModuleType(package.__name__ + '.target')
        target.function = self.module.function
        package.target = target
        for module in (package, target):
            sys.modules[module.__name__] = module
            self.addCleanup(sys.modules.pop, module.__name__, None)
        namespace = {'__package__': package.__name__}
        with effect(import_value=11, suffix='arena'):
            exec('from .target import function', namespace)
        self.assertEqual(namespace['function'](), 11)
        self.assertIs(target.function, self.module.function)

    def test_other_importer_receives_cached_requests(self):
        calls = []

        def observer(name, globals=None, locals=None, fromlist=(), level=0):
            calls.append(name)
            return self.original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = observer
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function
        self.assertIn(self.module.__name__, calls)
        self.assertIs(builtins.__import__, observer)
        self.assertEqual(function(), 11)

    def test_preceding_importer_is_called_by_the_adapter(self):
        calls = []

        def observer(name, globals=None, locals=None, fromlist=(), level=0):
            calls.append(name)
            return self.original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _make_import_hook(observer)
        with effect(import_value=11, suffix='arena'):
            from _pc_effect_scope_fixture import function
        self.assertIn(self.module.__name__, calls)
        self.assertEqual(function(), 11)

    def test_importer_installed_inside_context_is_not_removed(self):
        def observer(*args, **kwargs):
            return self.original_import(*args, **kwargs)

        with effect(import_value=11, suffix='arena'):
            builtins.__import__ = observer
        self.assertIs(builtins.__import__, observer)

    def test_import_failure_does_not_leak_context(self):
        with effect(import_value=11, suffix='arena'):
            with self.assertRaises(ImportError):
                from _pc_effect_scope_fixture import missing
            with self.assertRaises(ModuleNotFoundError):
                import _pc_effect_scope_missing
            self.assertEqual(effect.get_effect_impl('import_value'), 11)
        self.assertIsNone(effect.get_effect_impl('import_value'))
        self.assertIs(builtins.__import__, self.original_import)

    def test_foreign_session_does_not_inherit_import_request(self):
        with effect(import_value=11, suffix='arena'):
            with CompileSession():
                from _pc_effect_scope_fixture import function as foreign
            from _pc_effect_scope_fixture import function as local
        self.assertEqual((foreign(), local()), (0, 11))

    def test_process_session_fallback_also_owns_its_import_request(self):
        observations = []
        errors = []

        def worker():
            try:
                with effect(import_value=11, suffix='fallback_owner'):
                    with CompileSession():
                        from _pc_effect_scope_fixture import function
                        observations.append(function())
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(observations, [0])

    def test_unrelated_module_is_returned_unchanged(self):
        with effect(import_value=11, suffix='arena'):
            module = builtins.__import__('threading', fromlist=('Thread',))
        self.assertIs(module, threading)

    def test_importlib_keeps_standard_module_identity(self):
        with effect(import_value=11, suffix='arena'):
            module = importlib.import_module(self.module.__name__)
        self.assertIs(module, self.module)
        self.assertEqual(module.function(), 0)

    def test_child_task_can_import_after_parent_context_exits(self):
        async def run():
            ready = asyncio.Event()

            async def child():
                await ready.wait()
                from _pc_effect_scope_fixture import function
                return function()

            with effect(import_value=11, suffix='arena'):
                task = asyncio.create_task(child())
            self.assertIsNone(effect.get_effect_impl('import_value'))
            ready.set()
            return await task

        self.assertEqual(asyncio.run(run()), 11)


class TestEffectProviderIsolation(unittest.TestCase):
    def test_context_object_can_be_reused_and_nested(self):
        manager = Effect()
        manager.default(value=0)
        context = manager(value=11, suffix='arena')
        for unused in range(2):
            with context:
                with context:
                    self.assertEqual(manager.get_effect_impl('value'), 11)
                self.assertEqual(manager.get_effect_impl('value'), 11)
            self.assertEqual(manager.get_effect_impl('value'), 0)

    def test_provider_bindings_are_isolated_between_threads(self):
        manager = Effect()
        manager.default(value=0)
        barrier = threading.Barrier(2)
        observed = {}
        errors = []

        def worker(number):
            try:
                with manager(value=number, suffix=str(number)):
                    barrier.wait(timeout=5)
                    observed[number] = manager.get_effect_impl('value')
                    barrier.wait(timeout=5)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(number,))
                   for number in (11, 22)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertFalse(errors)
        self.assertEqual(observed, {11: 11, 22: 22})
        self.assertEqual(manager.get_effect_impl('value'), 0)

    def test_provider_bindings_are_isolated_between_async_tasks(self):
        manager = Effect()
        manager.default(value=0)

        async def worker(number):
            with manager(value=number, suffix=str(number)):
                await asyncio.sleep(0)
                result = manager.get_effect_impl('value')
                await asyncio.sleep(0)
                return result

        async def run():
            return await asyncio.gather(worker(11), worker(22))

        self.assertEqual(asyncio.run(run()), [11, 22])
        self.assertEqual(manager.get_effect_impl('value'), 0)

    def test_restore_snapshot_does_not_replace_shared_provider(self):
        with CompileSession():
            effect.default(value=0)
            original_namespace = effect.value
            snapshot = {'value': 11}
            with restore_effect_context(snapshot):
                self.assertEqual(effect.get_effect_impl('value'), 11)
                self.assertEqual(
                    object.__getattribute__(original_namespace, '_impl'), 0,
                )
                self.assertEqual(capture_effect_context()['value'], 11)
            self.assertEqual(effect.get_effect_impl('value'), 0)

    def test_direct_binding_stays_pinned_inside_replayed_context(self):
        with CompileSession():
            effect.value = 7
            captured = capture_effect_context()
            with restore_effect_context(captured):
                with effect(value=11, suffix='arena'):
                    self.assertEqual(effect.get_effect_impl('value'), 7)

    def test_default_registered_during_override_survives_exit(self):
        manager = Effect()
        with manager(value=11, suffix='arena'):
            manager.default(value=7)
            self.assertEqual(manager.get_effect_impl('value'), 11)
        self.assertEqual(manager.get_effect_impl('value'), 7)


if __name__ == '__main__':
    unittest.main()
