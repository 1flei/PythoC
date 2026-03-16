"""
Quasi-quote templates and binding helpers.

This module provides the quote decorators (quote_expr, quote_stmts, etc.)
and binding helpers (ref, ident, const, type_expr) that form the main
authoring surface for compile-time templates in pythoc.meta.

The core idea: write code-shaped Python, get back legal AST fragments.

Hole model:
    Template function parameters define the holes. When you write::

        @meta.quote_expr
        def add_expr(x, y):
            return x + y

    ``x`` and ``y`` are the template holes. You instantiate by calling::

        expr = add_expr(meta.ref("lhs"), meta.ref("rhs"))

    Every occurrence of ``x`` in the body AST is replaced with the AST
    produced by ``meta.ref("lhs")``, and similarly for ``y``.

Automatic coercion:
    - Plain scalars (int, float, str, bytes, bool, None) auto-lower to
      ast.Constant.
    - PythoC type objects auto-lower to ast.Name with the type's name,
      and the type is added to extra_globals.
    - Binding helpers (ref, ident, const, type_expr, splice_stmt,
      splice_stmts) produce explicit AST-coercible values for cases
      where automatic coercion would be ambiguous.
"""

import ast
import copy
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .fragment import MetaFragment, FragmentKind


# ---------------------------------------------------------------------------
# Binding helpers
# ---------------------------------------------------------------------------
# These return thin marker objects that the substitution engine recognizes.
# They are passed as *arguments* to a template call, not written inside
# the template body.

class _Ref:
    """Marker: produces ast.Name(id=name) in expression position."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "ref({!r})".format(self.name)


class _Ident:
    """Marker: bare identifier for renaming positions."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "ident({!r})".format(self.name)


class _Const:
    """Marker: compile-time constant -> ast.Constant."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "const({!r})".format(self.value)


class _TypeExpr:
    """Marker: type expression for annotation positions."""
    __slots__ = ('type_obj',)

    def __init__(self, type_obj):
        self.type_obj = type_obj

    def __repr__(self):
        return "type_expr({!r})".format(self.type_obj)


class _SpliceStmt:
    """Marker: single statement splice."""
    __slots__ = ('stmt',)

    def __init__(self, stmt):
        self.stmt = stmt

    def __repr__(self):
        return "splice_stmt(...)".format()


class _SpliceStmts:
    """Marker: statement-list splice."""
    __slots__ = ('stmts',)

    def __init__(self, stmts):
        self.stmts = stmts

    def __repr__(self):
        return "splice_stmts(...)".format()


def ref(name):
    """Produce a name reference expression.

    Use in expression position. At substitution, produces
    ``ast.Name(id=name)``.

    Example::

        expr = add_expr(meta.ref("lhs"), meta.ref("rhs"))
    """
    return _Ref(name)


def ident(name):
    """Produce a bare identifier for renaming positions.

    Use for argument names, field names, or other identifier-position
    holes.
    """
    return _Ident(name)


def const(value):
    """Produce a compile-time constant.

    At substitution, produces ``ast.Constant(value=value)``.

    Example::

        body = template(meta.const(42))
    """
    return _Const(value)


def type_expr(type_obj):
    """Produce a type expression for annotation positions.

    The type object is added to extra_globals so the compiler can
    resolve it.

    Example::

        fn = func_template(meta.type_expr(i32))
    """
    return _TypeExpr(type_obj)


def splice_stmt(stmt):
    """Splice a single ast.stmt into a statement-list position."""
    if not isinstance(stmt, ast.stmt):
        raise TypeError(
            "splice_stmt expects ast.stmt, got {}".format(type(stmt).__name__)
        )
    return _SpliceStmt(stmt)


def splice_stmts(stmts):
    """Splice a list of ast.stmt into a statement-list position."""
    if not isinstance(stmts, list):
        raise TypeError(
            "splice_stmts expects list, got {}".format(type(stmts).__name__)
        )
    return _SpliceStmts(stmts)


# ---------------------------------------------------------------------------
# Value-to-AST coercion
# ---------------------------------------------------------------------------

_AUTO_CONST_TYPES = (int, float, str, bytes, bool, type(None))


def _coerce_to_ast(value, extra_globals):
    """Coerce a binding value to an AST node.

    Coercion rules:
    1. Binding helpers (_Ref, _Const, _TypeExpr, etc.) -> explicit AST.
    2. ast.AST nodes -> used as-is.
    3. Plain scalars (int, float, str, bytes, bool, None) -> ast.Constant.
    4. PythoC type objects (has get_name/get_llvm_type) -> ast.Name + extra_globals.
    5. Other callables with __name__ -> ast.Name + extra_globals.

    Returns:
        An ast.expr node (or _SpliceMarker for statement splices).
    """
    if isinstance(value, _Ref):
        return ast.Name(id=value.name, ctx=ast.Load())

    if isinstance(value, _Ident):
        return ast.Name(id=value.name, ctx=ast.Load())

    if isinstance(value, _Const):
        return ast.Constant(value=value.value)

    if isinstance(value, _TypeExpr):
        obj = value.type_obj
        type_name = _get_type_name(obj)
        extra_globals[type_name] = obj
        return ast.Name(id=type_name, ctx=ast.Load())

    if isinstance(value, _SpliceStmt):
        return _SpliceMarker(stmts=[value.stmt])

    if isinstance(value, _SpliceStmts):
        return _SpliceMarker(stmts=value.stmts)

    # ast.AST nodes pass through
    if isinstance(value, ast.AST):
        return value

    # Auto-coerce scalars
    if isinstance(value, _AUTO_CONST_TYPES):
        return ast.Constant(value=value)

    # PythoC type objects
    if hasattr(value, 'get_name') and hasattr(value, 'get_llvm_type'):
        type_name = value.get_name()
        extra_globals[type_name] = value
        return ast.Name(id=type_name, ctx=ast.Load())

    # Other named objects (compiled wrappers, functions, etc.)
    if hasattr(value, '__name__'):
        name = value.__name__
        extra_globals[name] = value
        return ast.Name(id=name, ctx=ast.Load())

    if hasattr(value, '_original_name'):
        name = value._original_name
        extra_globals[name] = value
        return ast.Name(id=name, ctx=ast.Load())

    raise TypeError(
        "Cannot coerce {} to AST node. Use an explicit binding helper "
        "(ref, const, type_expr, etc.)".format(type(value).__name__)
    )


def _get_type_name(obj):
    """Extract a canonical name for a type-like object."""
    if hasattr(obj, 'get_name'):
        return obj.get_name()
    if hasattr(obj, '__name__'):
        return obj.__name__
    return str(obj)


# ---------------------------------------------------------------------------
# Splice marker (internal)
# ---------------------------------------------------------------------------

class _SpliceMarker:
    """Internal marker for statement-list splicing.

    Not a real AST node. Expanded by _expand_splices() after substitution.
    """
    __slots__ = ('stmts',)

    def __init__(self, stmts):
        self.stmts = stmts


def _expand_splices(stmts):
    """Expand _SpliceMarker placeholders in a statement list.

    An Expr(value=_SpliceMarker) is replaced with the spliced statements.
    Recurses into compound statement bodies (with, if, for, while, try, etc.)
    so that splice_stmts works at any nesting level.
    """
    result = []
    for stmt in stmts:
        if (isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, _SpliceMarker)):
            result.extend(stmt.value.stmts)
        else:
            _expand_splices_in_node(stmt)
            result.append(stmt)
    return result


def _expand_splices_in_node(node):
    """Recursively expand splices in nested statement bodies of a node."""
    # All AST node fields that hold statement lists
    for field_name in ('body', 'orelse', 'finalbody', 'handlers'):
        body = getattr(node, field_name, None)
        if isinstance(body, list) and body:
            expanded = _expand_splices(body)
            setattr(node, field_name, expanded)
    # ExceptHandler nodes inside handlers
    if hasattr(node, 'handlers'):
        for handler in getattr(node, 'handlers', []):
            if hasattr(handler, 'body'):
                handler.body = _expand_splices(handler.body)


# ---------------------------------------------------------------------------
# Parameter substitution transformer
# ---------------------------------------------------------------------------

class _ParamSubstituter(ast.NodeTransformer):
    """Replace ast.Name nodes matching template param names with bound AST.

    This is the core of quasi-quote instantiation. Every occurrence of
    a template parameter name in the body AST is replaced with the
    coerced AST node for the corresponding argument.
    """

    def __init__(self, param_bindings):
        """
        Args:
            param_bindings: dict mapping param_name -> ast.expr (coerced).
        """
        self.param_bindings = param_bindings

    def visit_Name(self, node):
        if node.id in self.param_bindings:
            replacement = self.param_bindings[node.id]
            if isinstance(replacement, _SpliceMarker):
                # Statement splicing in expression context is an error;
                # this will be caught at the statement level instead.
                return replacement
            return copy.deepcopy(replacement)
        return node


# ---------------------------------------------------------------------------
# MetaTemplate
# ---------------------------------------------------------------------------

class MetaTemplate:
    """A compile-time template captured from a Python function.

    Created by quote decorators. Calling the template with positional
    and/or keyword arguments instantiates it, producing a MetaFragment.

    The function parameters define the template holes. Arguments are
    matched to parameters by position or name, coerced to AST, and
    substituted into the body.

    Attributes:
        kind: FragmentKind of the output.
        source: Original source text (for diagnostics).
        template_ast: The AST to substitute into.
        param_names: Ordered tuple of parameter names (= hole names).
        user_globals: Captured globals from the template definition site.
        origin_file: Source file of the template definition.
        origin_line: Start line of the template definition.
    """

    def __init__(self, kind, source, template_ast, param_names,
                 user_globals, origin_file=None, origin_line=None):
        self.kind = kind
        self.source = source
        self.template_ast = template_ast
        self.param_names = tuple(param_names)
        self.user_globals = dict(user_globals) if user_globals else {}
        self.origin_file = origin_file
        self.origin_line = origin_line

    def instantiate(self, *args, **kwargs):
        """Produce a MetaFragment by substituting holes with arguments.

        Arguments are matched to template parameters by position first,
        then by keyword. Each argument is coerced to an AST node via
        _coerce_to_ast().

        Returns:
            A MetaFragment with the substituted AST.
        """
        # Bind args to param names
        extra_globals = {}
        param_bindings = {}

        for i, val in enumerate(args):
            if i >= len(self.param_names):
                raise TypeError(
                    "Too many positional arguments: template has {} params, "
                    "got {} args".format(len(self.param_names), len(args))
                )
            name = self.param_names[i]
            param_bindings[name] = _coerce_to_ast(val, extra_globals)

        for name, val in kwargs.items():
            if name in param_bindings:
                raise TypeError(
                    "Duplicate binding for parameter {!r}".format(name)
                )
            if name not in self.param_names:
                raise TypeError(
                    "Unknown template parameter {!r}. "
                    "Available: {}".format(name, list(self.param_names))
                )
            param_bindings[name] = _coerce_to_ast(val, extra_globals)

        # Perform substitution
        substituter = _ParamSubstituter(param_bindings)

        if isinstance(self.template_ast, list):
            result = [
                substituter.visit(copy.deepcopy(node))
                for node in self.template_ast
            ]
            result = _expand_splices(result)
        else:
            result = substituter.visit(copy.deepcopy(self.template_ast))
            _expand_splices_in_node(result)

        # Fix locations
        if isinstance(result, list):
            for node in result:
                ast.fix_missing_locations(node)
        else:
            ast.fix_missing_locations(result)

        # Merge globals
        merged_globals = dict(self.user_globals)
        merged_globals.update(extra_globals)

        # Generate debug source
        debug_source = None
        try:
            if isinstance(result, list):
                debug_source = '\n'.join(ast.unparse(n) for n in result)
            else:
                debug_source = ast.unparse(result)
        except Exception:
            pass

        fragment = MetaFragment(
            kind=self.kind,
            node=result,
            origin_file=self.origin_file,
            origin_line=self.origin_line,
            debug_source=debug_source,
        )
        # Attach merged globals for compile_ast to use.
        # Use object.__setattr__ because MetaFragment is frozen.
        object.__setattr__(fragment, '_user_globals', merged_globals)
        return fragment

    def __call__(self, *args, **kwargs):
        """Shorthand for instantiate()."""
        return self.instantiate(*args, **kwargs)

    def __repr__(self):
        return "MetaTemplate(kind={}, params={})".format(
            self.kind, self.param_names,
        )


# ---------------------------------------------------------------------------
# Quote decorators
# ---------------------------------------------------------------------------

def _capture_caller_globals(depth=2):
    """Capture globals+locals from the caller's frame.

    depth=2 because: caller -> quote_xxx -> _make_template -> here.
    """
    frame = inspect.currentframe()
    try:
        for _ in range(depth + 1):
            if frame is None:
                return {}
            frame = frame.f_back
        if frame is None:
            return {}
        symbols = dict(frame.f_globals)
        symbols.update(frame.f_locals)
        return symbols
    finally:
        del frame


def _make_template(func, kind):
    """Internal: create a MetaTemplate from a decorated function.

    The function's parameters define the template holes. The body
    (or inner function for quote_func) becomes the template AST.
    Source is captured from the Python function object.
    """
    source = inspect.getsource(func)
    source = textwrap.dedent(source)

    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise TypeError(
            "quote decorator expects a function, got {}".format(
                type(func_def).__name__
            )
        )

    param_names = [arg.arg for arg in func_def.args.args]

    # Get origin info
    origin_file = None
    origin_line = None
    try:
        origin_file = inspect.getfile(func)
        _, origin_line = inspect.getsourcelines(func)
    except (OSError, TypeError):
        pass

    # Capture user globals from the decorator call site
    user_globals = _capture_caller_globals(depth=2)

    # Extract template AST based on kind
    if kind == FragmentKind.EXPR:
        # Function should have a single return statement
        body = func_def.body
        if (len(body) == 1
                and isinstance(body[0], ast.Return)
                and body[0].value is not None):
            template_ast = body[0].value
        else:
            raise ValueError(
                "quote_expr function must contain exactly one "
                "'return <expr>' statement"
            )

    elif kind == FragmentKind.STMT:
        body = func_def.body
        if len(body) == 1:
            template_ast = body[0]
        else:
            raise ValueError(
                "quote_stmt function must contain exactly one statement"
            )

    elif kind == FragmentKind.STMTS:
        template_ast = func_def.body

    elif kind == FragmentKind.FUNC:
        # The body should contain a nested function def.
        # The outer function's params are template holes; the inner
        # function def is the template AST.
        inner_func = None
        for stmt in func_def.body:
            if isinstance(stmt, ast.FunctionDef):
                inner_func = stmt
                break
        if inner_func is None:
            raise ValueError(
                "quote_func function must contain a nested function definition"
            )
        template_ast = inner_func

    elif kind == FragmentKind.MODULE:
        template_ast = ast.Module(body=func_def.body, type_ignores=[])

    else:
        raise ValueError("Unsupported fragment kind: {}".format(kind))

    return MetaTemplate(
        kind=kind,
        source=source,
        template_ast=template_ast,
        param_names=param_names,
        user_globals=user_globals,
        origin_file=origin_file,
        origin_line=origin_line,
    )


def quote_expr(func=None, *, debug_source=True):
    """Capture a function body as an expression template.

    The decorated function must contain exactly one ``return <expr>``
    statement. The expression becomes the template AST. The function
    parameters define the template holes.

    Example::

        @meta.quote_expr
        def add_expr(x, y):
            return x + y

        expr = add_expr(meta.ref("lhs"), meta.ref("rhs"))
    """
    if func is None:
        def decorator(f):
            return _make_template(f, FragmentKind.EXPR)
        return decorator
    return _make_template(func, FragmentKind.EXPR)


def quote_stmt(func=None, *, debug_source=True):
    """Capture a function body as a single-statement template.

    The decorated function must contain exactly one statement.
    """
    if func is None:
        def decorator(f):
            return _make_template(f, FragmentKind.STMT)
        return decorator
    return _make_template(func, FragmentKind.STMT)


def quote_stmts(func=None, *, debug_source=True):
    """Capture a function body as a statement-list template.

    All statements in the body become the template AST.

    Example::

        @meta.quote_stmts
        def body_template(init_val, offset):
            x = init_val
            result = x + offset
    """
    if func is None:
        def decorator(f):
            return _make_template(f, FragmentKind.STMTS)
        return decorator
    return _make_template(func, FragmentKind.STMTS)


def quote_func(func=None, *, debug_source=True):
    """Capture a nested function definition as a function template.

    The outer function's parameters become template holes.
    The inner function definition becomes the template AST.

    Example::

        @meta.quote_func
        def add_template(ret_ty):
            def generated(x: i32, y: i32) -> ret_ty:
                return x + y

        fn = add_template(i32).with_name("add_i32")
    """
    if func is None:
        def decorator(f):
            return _make_template(f, FragmentKind.FUNC)
        return decorator
    return _make_template(func, FragmentKind.FUNC)


def quote_module(func=None, *, debug_source=True):
    """Capture a function body as a module-level template.

    All statements in the body become a Module AST node.
    """
    if func is None:
        def decorator(f):
            return _make_template(f, FragmentKind.MODULE)
        return decorator
    return _make_template(func, FragmentKind.MODULE)
