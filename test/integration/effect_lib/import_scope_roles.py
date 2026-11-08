"""A declaration can be both an ordinary dependency and a fixed provider."""

from types import SimpleNamespace

from pythoc import compile, effect, i32


effect.default(role_scalar=7)


@compile
def read_role() -> i32:
    return i32(effect.role_scalar)


@compile
def read_unregistered() -> i32:
    return i32(effect.role_scalar)


DefaultProvider = SimpleNamespace(value=read_role)
effect.default(role_provider=DefaultProvider)


@compile
def read_provider() -> i32:
    return effect.role_provider.value()


@compile
def read_helper() -> i32:
    return read_role()


@compile
def yield_role():
    yield read_helper()
