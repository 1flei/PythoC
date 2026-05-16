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
]
