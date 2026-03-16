"""
GeneratedFunction and MetaArtifact - structured builders for meta code generation.

GeneratedFunction is a thin front-end object that lowers directly to
ast.FunctionDef. It is NOT a new compiler IR -- it is just a convenient
way to assemble function AST without raw ast constructors.

MetaArtifact groups a primary function with optional helpers, suitable
for meta factories that emit more than one function (e.g., regex, format).
"""

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .fragment import MetaFragment, FragmentKind


@dataclass
class GeneratedFunction:
    """A thin structured function object that lowers to ast.FunctionDef.

    This is intentionally minimal. The purpose is to avoid forcing
    everything through raw AST constructors, not to invent a second
    language.

    Attributes:
        name: The function name.
        params: List of (param_name, type_hint) tuples.
        return_type: A pythoc type object (e.g., i32, void) or None.
        body: Either a list of ast.stmt or a MetaFragment(kind=STMTS).
        attrs: Optional set of LLVM function-level attributes.
        required_globals: Additional globals needed for compilation.
        source_file: Diagnostic source file path.
        start_line: Diagnostic start line number.
        debug_source: Optional human-readable source for diagnostics.
    """
    name: str
    params: List[Tuple[str, Any]]
    return_type: Optional[Any]
    body: Union[List[ast.stmt], MetaFragment]
    attrs: Optional[Set[str]] = None
    required_globals: Optional[Dict[str, Any]] = None
    source_file: Optional[str] = None
    start_line: int = 1
    debug_source: Optional[str] = None

    def to_func_def(self) -> ast.FunctionDef:
        """Lower to an ast.FunctionDef node.

        Type hints are NOT placed in AST annotations. They are passed
        separately via get_param_type_hints()/get_return_type_hint()
        to the compile pipeline, which already supports pre-parsed hints.
        """
        args = [ast.arg(arg=pname) for pname, _ in self.params]

        body_stmts = self.body
        if isinstance(body_stmts, MetaFragment):
            body_stmts = body_stmts.as_stmts

        if not body_stmts:
            body_stmts = [ast.Pass()]

        func_def = ast.FunctionDef(
            name=self.name,
            args=ast.arguments(
                posonlyargs=[],
                args=args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=body_stmts,
            decorator_list=[],
            returns=None,
            lineno=self.start_line,
            col_offset=0,
        )
        ast.fix_missing_locations(func_def)
        return func_def

    def get_param_type_hints(self) -> Dict[str, Any]:
        """Extract parameter type hints as a dict."""
        return {name: hint for name, hint in self.params}

    def get_return_type_hint(self) -> Optional[Any]:
        """Return the return type hint."""
        return self.return_type


@dataclass
class MetaArtifact:
    """A multi-function output from a meta factory.

    Attributes:
        primary: The main generated function.
        helpers: Additional helper functions (compiled into the same group).
        required_globals: Shared globals for all functions in the artifact.
        suffix_seed: Data used to derive a specialization suffix.
        debug_source: Optional diagnostic source text.
    """
    primary: Union[GeneratedFunction, MetaFragment, ast.FunctionDef]
    helpers: Tuple[Union[GeneratedFunction, ast.FunctionDef], ...] = ()
    required_globals: Optional[Dict[str, Any]] = None
    suffix_seed: Optional[Any] = None
    debug_source: Optional[str] = None


def func(name, params, return_type, body, *,
         attrs=None, required_globals=None, source_file=None,
         start_line=1, debug_source=None):
    """Thin builder for GeneratedFunction.

    Example::

        gf = meta.func(
            name="add",
            params=[("a", i32), ("b", i32)],
            return_type=i32,
            body=[...],  # list of ast.stmt
        )
    """
    return GeneratedFunction(
        name=name,
        params=params,
        return_type=return_type,
        body=body,
        attrs=attrs,
        required_globals=required_globals,
        source_file=source_file,
        start_line=start_line,
        debug_source=debug_source,
    )


def artifact(primary, *, helpers=(), required_globals=None,
             suffix_seed=None, debug_source=None):
    """Thin builder for MetaArtifact.

    Example::

        art = meta.artifact(
            primary_func,
            helpers=(helper1, helper2),
            suffix_seed=("my_spec_key",),
        )
    """
    return MetaArtifact(
        primary=primary,
        helpers=tuple(helpers),
        required_globals=required_globals,
        suffix_seed=suffix_seed,
        debug_source=debug_source,
    )
