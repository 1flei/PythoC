"""
pythoc.meta - Unified compile-time program generation.

This module is the single public home for compile-time code generation
in PythoC. After the "fragment as universal currency" simplification it
exposes a small surface:

Core types:
    Fragment            - the universal AST currency (list[ast.stmt] inside)
    MetaTemplate        - a compile-time template captured from Python code
    GeneratedFunction   - a thin structured function that lowers to ast.FunctionDef
    MetaArtifact        - a multi-function output from a meta factory

Public API:
    Quote decorator:    quote
    Const helper:       const                    (only when you need a string *literal*)
    Builders:           func, artifact
    Compilation:        compile_ast, compile_generated, compile_artifact
    Factories:          factory, compile_factory
    Normalization:      normalize_factory_key
"""

from .fragment import Fragment

from .template import (
    MetaTemplate,
    quote,
    const,
)

from .generated import (
    GeneratedFunction,
    MetaArtifact,
    func,
    artifact,
)

from .compile_api import (
    compile_ast,
    compile_generated,
    compile_artifact,
)

from .normalize import (
    normalize_factory_key,
)

from .factory import (
    factory,
    compile_factory,
)


def struct_type(*args, **kwargs):
    """Create a struct type programmatically.

    Accepts multiple calling conventions for convenience::

        struct_type(x=i32, y=i32)                    # keyword pairs
        struct_type([("x", i32), ("y", i32)])         # list of (name, type) pairs
        struct_type([i32, i32], ["x", "y"])            # separate lists (legacy)

    Returns a StructType (same as ``struct['x': i32, 'y': i32]``).
    """
    from ..builtin_entities.struct import create_struct_type

    if kwargs and not args:
        # struct_type(x=i32, y=i32)
        names = list(kwargs.keys())
        types = list(kwargs.values())
        return create_struct_type(types, names)
    elif len(args) == 1 and isinstance(args[0], list):
        items = args[0]
        if items and isinstance(items[0], (list, tuple)):
            # struct_type([("x", i32), ("y", i32)])
            names = [n for n, _ in items]
            types = [t for _, t in items]
            return create_struct_type(types, names)
        else:
            # struct_type([i32, i32]) — unnamed
            return create_struct_type(items)
    elif len(args) == 2 and isinstance(args[0], list) and isinstance(args[1], list):
        # struct_type([i32, i32], ["x", "y"]) — legacy
        return create_struct_type(args[0], args[1])
    else:
        raise TypeError(
            "struct_type() expects keyword args, list of (name, type) pairs, "
            "or (types, names) lists. Got: {}".format(args)
        )

__all__ = [
    # Core types
    'Fragment',
    'MetaTemplate',
    'GeneratedFunction',
    'MetaArtifact',
    # Quote decorator + helpers
    'quote',
    'const',
    # Builders
    'func',
    'artifact',
    # Compilation
    'compile_ast',
    'compile_generated',
    'compile_artifact',
    # Factories
    'factory',
    'compile_factory',
    # Normalization
    'normalize_factory_key',
    # Struct builder
    'struct_type',
]
