"""
pythoc.meta - Unified compile-time program generation.

This module is the single public home for compile-time code generation
in PythoC. It unifies quasi-quote templates, generated functions, and
AST compilation under one roof.

Core concepts:
    MetaFragment   - a typed fragment of Python AST
    MetaTemplate   - a compile-time template captured from Python code
    GeneratedFunction - a thin structured function that lowers to ast.FunctionDef
    MetaArtifact   - a multi-function output from a meta factory

Public API:
    Quote decorators:  quote_expr, quote_stmt, quote_stmts, quote_func, quote_module
    Binding helpers:   ref, ident, const, type_expr, splice_stmt, splice_stmts
    Builders:          func, artifact
    Compilation:       compile_ast, compile_generated, compile_artifact
"""

from .fragment import MetaFragment, FragmentKind

from .template import (
    MetaTemplate,
    quote_expr,
    quote_stmt,
    quote_stmts,
    quote_func,
    quote_module,
    ref,
    ident,
    const,
    type_expr,
    splice_stmt,
    splice_stmts,
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

__all__ = [
    # Core types
    'MetaFragment',
    'FragmentKind',
    'MetaTemplate',
    'GeneratedFunction',
    'MetaArtifact',
    # Quote decorators
    'quote_expr',
    'quote_stmt',
    'quote_stmts',
    'quote_func',
    'quote_module',
    # Binding helpers
    'ref',
    'ident',
    'const',
    'type_expr',
    'splice_stmt',
    'splice_stmts',
    # Builders
    'func',
    'artifact',
    # Compilation
    'compile_ast',
    'compile_generated',
    'compile_artifact',
]
