"""
Parametric parameter type.

``param`` is a special annotation that marks a function parameter as a
compile-time parameter.  It has no runtime representation; at call sites the
compiler binds all ``param`` arguments first, produces a specialized PythoC
callable, and then invokes it with the remaining runtime arguments.
"""

from llvmlite import ir

from .base import BuiltinType
from ..logger import logger


class param(BuiltinType):
    """Compile-time parameter type for parametric polymorphism.

    ``param`` may only appear in ``@compile`` function parameter annotations.
    It is not a value type and has no LLVM representation.
    """

    _llvm_type = None
    _size_bytes = None
    _is_signed = False
    _is_param = True

    @classmethod
    def get_name(cls) -> str:
        return 'param'

    @classmethod
    def get_llvm_type(cls, module_context=None) -> None:
        """No runtime LLVM type."""
        return None

    @classmethod
    def get_size_bytes(cls) -> None:
        """No runtime size."""
        return None

    @classmethod
    def can_be_called(cls) -> bool:
        return False

    @classmethod
    def can_be_type(cls) -> bool:
        """``param`` is not a usable runtime type.

        It may only appear as a function parameter annotation to mark a
        compile-time parameter.  Any attempt to use it as a local variable,
        field, return, or pointee type will be rejected by the resolver.
        """
        return False

    @classmethod
    def is_param_type(cls) -> bool:
        """Distinguish ``param`` from real runtime types."""
        return True
