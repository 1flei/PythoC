"""Use one provider namespace through both ordinary and effect edges."""

from pythoc import compile, effect, i32
from .import_scope_roles import DefaultProvider as Provider, read_role


@compile
def read_imported() -> i32:
    return read_role()


@compile
def read_namespace() -> i32:
    return Provider.value()


@compile
def read_provider() -> i32:
    return effect.role_provider.value()


@compile
def yield_namespace():
    yield Provider.value()
