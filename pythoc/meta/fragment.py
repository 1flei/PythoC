"""
MetaFragment - typed fragment of Python AST.

A MetaFragment is the universal result type for quasi-quote instantiation
and template expansion. It carries enough metadata for diagnostics but
introduces no new semantic IR layer.
"""

import ast
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Union


class FragmentKind(Enum):
    """The kind of AST fragment."""
    EXPR = "expr"
    STMT = "stmt"
    STMTS = "stmts"
    FUNC = "func"
    MODULE = "module"


@dataclass(frozen=True)
class MetaFragment:
    """A typed fragment of Python AST produced by template instantiation.

    Attributes:
        kind: What kind of AST fragment this holds.
        node: The actual AST node(s).
        origin_file: Source file that produced this fragment (for diagnostics).
        origin_line: Source line that produced this fragment (for diagnostics).
        debug_source: Optional reconstructed source text (for diagnostics only).
    """
    kind: FragmentKind
    node: Union[ast.expr, ast.stmt, List[ast.stmt], ast.FunctionDef, ast.Module]
    origin_file: Optional[str] = None
    origin_line: Optional[int] = None
    debug_source: Optional[str] = None

    @property
    def as_expr(self) -> ast.expr:
        if self.kind != FragmentKind.EXPR:
            raise TypeError(
                "MetaFragment.as_expr requires kind=EXPR, got {}".format(self.kind)
            )
        return self.node

    @property
    def as_stmt(self) -> ast.stmt:
        if self.kind != FragmentKind.STMT:
            raise TypeError(
                "MetaFragment.as_stmt requires kind=STMT, got {}".format(self.kind)
            )
        return self.node

    @property
    def as_stmts(self) -> List[ast.stmt]:
        if self.kind != FragmentKind.STMTS:
            raise TypeError(
                "MetaFragment.as_stmts requires kind=STMTS, got {}".format(self.kind)
            )
        return self.node

    @property
    def as_func(self) -> ast.FunctionDef:
        if self.kind != FragmentKind.FUNC:
            raise TypeError(
                "MetaFragment.as_func requires kind=FUNC, got {}".format(self.kind)
            )
        return self.node

    def to_ast(self) -> Union[ast.AST, List[ast.AST]]:
        """Return the raw AST node(s)."""
        return self.node

    def with_name(self, name: str) -> 'MetaFragment':
        """Return a new FUNC fragment with the function renamed.

        Only valid for kind=FUNC.
        """
        if self.kind != FragmentKind.FUNC:
            raise TypeError(
                "with_name() requires kind=FUNC, got {}".format(self.kind)
            )
        new_node = copy.deepcopy(self.node)
        new_node.name = name
        return MetaFragment(
            kind=self.kind,
            node=new_node,
            origin_file=self.origin_file,
            origin_line=self.origin_line,
            debug_source=self.debug_source,
        )
