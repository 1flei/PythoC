"""
Fragment - the universal AST currency in pythoc.meta.

A Fragment is the unified result type for all quasi-quote template
instantiations. Internally it is always a ``list[ast.stmt]`` (the
function body of the template). The consumer decides what shape they
want via the three cast properties:

  * ``.stmts`` -- always succeeds (it IS a list).
  * ``.stmt``  -- succeeds when the body has exactly one statement.
  * ``.expr``  -- succeeds when the body is exactly one
                   ``Return(value=X)`` or ``Expr(value=X)``; returns ``X``.

This replaces the older ``MetaFragment`` + ``FragmentKind`` design,
which forced the producer (the quote decorator) to commit to a kind.
With Fragment-as-universal-currency the role is decided at the
consumption point.
"""

import ast
import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Fragment:
    """A universal fragment of Python AST produced by template instantiation.

    Attributes:
        body: The fragment's canonical form: a list of ast.stmt.
        origin_file: Source file that produced this fragment (diagnostics).
        origin_line: Source line that produced this fragment (diagnostics).
        debug_source: Optional reconstructed source text (diagnostics).
    """
    body: List[ast.stmt]
    origin_file: Optional[str] = None
    origin_line: Optional[int] = None
    debug_source: Optional[str] = None

    # --- Cast properties --------------------------------------------------

    @property
    def stmts(self) -> List[ast.stmt]:
        """Return the fragment as a list of statements (always succeeds)."""
        return self.body

    @property
    def stmt(self) -> ast.stmt:
        """Return the single statement in the fragment.

        Raises:
            TypeError: if the body does not have exactly one statement.
        """
        if len(self.body) != 1:
            raise TypeError(
                "Fragment with {} stmts has no single-statement form; "
                "use .stmts".format(len(self.body))
            )
        return self.body[0]

    @property
    def expr(self) -> ast.expr:
        """Return the inner expression of a one-line return/expression body.

        Succeeds iff ``body`` is exactly ``[Return(value=X)]`` or
        ``[Expr(value=X)]``; returns ``X``.

        Raises:
            TypeError: otherwise.
        """
        if len(self.body) != 1:
            raise TypeError(
                "Fragment with {} stmts has no single-expression form; "
                "use .stmts".format(len(self.body))
            )
        only = self.body[0]
        if isinstance(only, ast.Return) and only.value is not None:
            return only.value
        if isinstance(only, ast.Expr):
            return only.value
        raise TypeError(
            "Fragment's only stmt is {} which has no expression form; "
            "use .stmt or .stmts".format(type(only).__name__)
        )

    # --- Helpers ----------------------------------------------------------

    def with_func_name(self, name: str) -> 'Fragment':
        """Return a new Fragment with the leading FunctionDef renamed.

        Useful for "function template" patterns where the body's first
        statement is a ``def`` and the caller wants a specific name.

        Raises:
            TypeError: if ``body[0]`` is not an ``ast.FunctionDef``.
        """
        if not self.body or not isinstance(self.body[0], ast.FunctionDef):
            raise TypeError(
                "with_func_name() requires body[0] to be FunctionDef; "
                "got {}".format(
                    type(self.body[0]).__name__ if self.body else "empty"
                )
            )
        new_body = list(self.body)
        new_body[0] = copy.deepcopy(new_body[0])
        new_body[0].name = name
        return Fragment(
            body=new_body,
            origin_file=self.origin_file,
            origin_line=self.origin_line,
            debug_source=self.debug_source,
        )

    def __repr__(self):
        return "Fragment(stmts={})".format(len(self.body))
