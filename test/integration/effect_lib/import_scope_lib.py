"""Compiled declarations for import-scope integration tests."""

from types import SimpleNamespace

from pythoc import compile, effect, i32


@compile
def default_value() -> i32:
    return i32(7)


DefaultProvider = SimpleNamespace(value=default_value)
effect.default(import_provider=DefaultProvider, import_scalar=7)


@compile
def read_value() -> i32:
    return effect.import_provider.value()


@compile(suffix='existing_hook_arena_compile')
def read_with_compile_suffix() -> i32:
    return effect.import_provider.value()


@compile
def read_scalar() -> i32:
    return i32(effect.import_scalar)


@compile
def yield_value():
    yield read_value()
