"""
Quasi-quote templates and the single binding helper.

This module provides the unified ``@meta.quote`` decorator and the
single explicit binding helper ``meta.const(value)``. The core idea is
"fragment as universal currency + consumer-driven cast":

  - There is exactly one decorator: ``@meta.quote``. The decorated
    function's body becomes the template (always captured as
    ``list[ast.stmt]``).
  - Calling a template returns a :class:`Fragment`. The consumer picks
    a shape via ``.expr`` / ``.stmt`` / ``.stmts``.
  - Each template parameter is statically classified at decoration
    time into one of three positions: ``'expr'``, ``'splice'``, or
    ``'store'``. At instantiation, the bound value is coerced to AST
    according to the position:

        * ``str``                 -> ``Name`` (or ``Expr(Name)`` for splice)
        * ``int/float/bool/None`` -> ``Constant``
        * PythoC type             -> ``Name`` with its definition binding
        * ``ast.expr``            -> as-is
        * ``ast.stmt``            -> only valid in splice position
        * ``list[ast.stmt]``      -> only valid in splice position (auto-spliced)
        * ``Fragment``            -> auto-cast (``.expr`` in expr/store; ``.stmts`` in splice)

  - The only explicit helper is ``meta.const(value)`` which forces a
    string (or any value) to become a ``Constant`` rather than the
    default ``str -> Name`` mapping.
"""

import ast
import copy
import inspect
import textwrap
from typing import List, Set

from .fragment import Fragment


# ---------------------------------------------------------------------------
# Public binding helper
# ---------------------------------------------------------------------------

class _Const:
    """Marker: compile-time constant -> ``ast.Constant``.

    This is the only remaining explicit binding helper. It exists
    because the default coercion of ``str`` is ``Name`` (variable name);
    if you want a string literal you must wrap with ``const("...")``.
    """
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "const({!r})".format(self.value)


def const(value):
    """Produce an explicit compile-time constant.

    Use this to disambiguate string *literals* from variable names::

        meta.const("hello")     # -> Constant("hello")
        "hello"                  # -> Name("hello") (variable reference)
    """
    return _Const(value)


# ---------------------------------------------------------------------------
# Internal: splice marker (preserves existing _expand_splices machinery)
# ---------------------------------------------------------------------------

class _SpliceMarker:
    """Internal marker placed at a splice position during substitution.

    Not a real AST node. ``_expand_splices`` recognises
    ``Expr(value=_SpliceMarker)`` and replaces the wrapping ``Expr`` with
    the contained statements.
    """
    __slots__ = ('stmts',)

    def __init__(self, stmts):
        self.stmts = stmts


def _expand_splices(stmts):
    """Replace ``Expr(value=_SpliceMarker)`` placeholders in a stmt list."""
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
    """Recursively expand splices in nested stmt-list bodies of a node."""
    for field_name in ('body', 'orelse', 'finalbody'):
        body = getattr(node, field_name, None)
        if isinstance(body, list) and body:

            setattr(node, field_name, _expand_splices(body))
    handlers = getattr(node, 'handlers', None)
    if isinstance(handlers, list):
        for handler in handlers:
            if hasattr(handler, 'body'):
                handler.body = _expand_splices(handler.body)


# ---------------------------------------------------------------------------
# Constant-condition if-folding
# ---------------------------------------------------------------------------
# After template instantiation, ``if <const(True)>:`` / ``if <const(False)>:``
# branches are statically resolved at the AST level:
#   * ``if True:  body`` → inline *body*
#   * ``if False: body`` → drop entirely
#   * ``if True: body else: orelse`` → inline *body*
#   * ``if False: body else: orelse`` → inline *orelse*
# This mirrors what the PythoC compiler does at the visit stage but performs
# it eagerly so that downstream consumers (tests, further templates) see a
# clean AST without dead branches.

def _is_const_true(test):
    """Check if an if-test is a compile-time True (Constant(True) or not Constant(False))."""
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Constant)):
        return not test.operand.value
    return None  # not a constant


def _fold_const_if(stmts):
    """Fold ``if True``/``if False`` in a statement list (recursive).

    Also handles ``if not True:``/``if not False:`` patterns.
    """
    result = []
    for stmt in stmts:
        if isinstance(stmt, ast.If):
            truth = _is_const_true(stmt.test)
            if truth is True:
                result.extend(_fold_const_if(stmt.body))
                continue
            elif truth is False:
                if stmt.orelse:
                    result.extend(_fold_const_if(stmt.orelse))
                continue
        _fold_const_if_in_node(stmt)
        result.append(stmt)
    return result


def _fold_const_if_in_node(node):
    """Recursively fold constant-condition ifs in nested stmt-list fields."""
    for field_name in ('body', 'orelse', 'finalbody'):
        body = getattr(node, field_name, None)
        if isinstance(body, list) and body:
            setattr(node, field_name, _fold_const_if(body))
    handlers = getattr(node, 'handlers', None)
    if isinstance(handlers, list):
        for handler in handlers:
            if hasattr(handler, 'body'):
                handler.body = _fold_const_if(handler.body)


# ---------------------------------------------------------------------------
# Value-to-AST coercion (position-aware)
# ---------------------------------------------------------------------------

_AUTO_CONST_TYPES = (int, float, bool, type(None))   # str excluded on purpose


def _is_pythoc_type(obj) -> bool:
    return hasattr(obj, 'get_name') and hasattr(obj, 'get_llvm_type')


def _get_type_name(obj):
    if hasattr(obj, 'get_name'):
        return obj.get_name()
    if hasattr(obj, '__name__'):
        return obj.__name__
    return str(obj)


def _coerce_value_to_expr(value):
    """Coerce a bound value to an ``ast.expr``.

    Used for ``'expr'`` position and as the inner step for ``'splice'``
    position (whose result is then wrapped in ``Expr(...)``).

    Returns:
        ``ast.expr``.
    Raises:
        TypeError: when the value cannot meaningfully appear as an
        expression.
    """
    if isinstance(value, _Const):
        return ast.Constant(value=value.value)

    if isinstance(value, ast.expr):
        return value

    if isinstance(value, str):
        # Default: string -> variable name. Use meta.const(s) for literal.
        return ast.Name(id=value, ctx=ast.Load())

    if isinstance(value, _AUTO_CONST_TYPES):
        return ast.Constant(value=value)

    if isinstance(value, Fragment):
        return value.expr   # may raise TypeError with a clear message

    if _is_pythoc_type(value):
        from .._inline.definition_bindings import symbol_expression
        type_name = _get_type_name(value)
        return symbol_expression(type_name, value)

    # Other named callables / wrappers
    if hasattr(value, '__name__'):
        from .._inline.definition_bindings import symbol_expression
        name = value.__name__
        return symbol_expression(name, value)

    if hasattr(value, '_original_name'):
        from .._inline.definition_bindings import symbol_expression
        name = value._original_name
        return symbol_expression(name, value)

    raise TypeError(
        "Cannot coerce {} to an expression node.".format(type(value).__name__)
    )


def _coerce_to_ast(value, position='expr'):
    """Coerce a bound value to AST according to its template position.

    Args:
        value: The argument passed at the template call site.
        position: One of ``'expr'``, ``'splice'``, ``'store'``.
            Defaults to ``'expr'`` (most common; lets legacy callers
            that don't know about position still get sensible coercion).

    Returns:
        Either an ``ast.AST`` node or a ``_SpliceMarker``.
    """
    if position == 'expr':
        return _coerce_value_to_expr(value)

    if position == 'store':
        if isinstance(value, str):
            # ctx is fixed up by _ParamSubstituter to match the template
            # site's original Store context.
            return ast.Name(id=value, ctx=ast.Store())
        if isinstance(value, (ast.Name, ast.Tuple, ast.List, ast.Subscript,
                              ast.Attribute, ast.Starred)):
            return value
        raise TypeError(
            "In store position, expected str or assign-target ast node, "
            "got {}".format(type(value).__name__)
        )

    if position == 'splice':
        if isinstance(value, list):
            flat = []
            for item in value:
                if isinstance(item, Fragment):
                    flat.extend(item.stmts)
                elif isinstance(item, ast.stmt):
                    flat.append(item)
                else:
                    raise TypeError(
                        "In splice position, list elements must be "
                        "ast.stmt or Fragment, got {}".format(
                            type(item).__name__)
                    )
            return _SpliceMarker(stmts=flat)

        if isinstance(value, Fragment):
            return _SpliceMarker(stmts=list(value.stmts))

        if isinstance(value, ast.stmt):
            return _SpliceMarker(stmts=[value])

        # Single-expression-ish values: wrap as Expr(...)
        expr = _coerce_value_to_expr(value)
        return _SpliceMarker(stmts=[ast.Expr(value=expr)])

    raise ValueError("Unknown position: {!r}".format(position))


# ---------------------------------------------------------------------------
# Position analyzer
# ---------------------------------------------------------------------------

class _PositionAnalyzer:
    """One-shot static analyser of parameter ``Name`` positions.

    Walks the template body once and tags:

      * Each ``ast.Name(id=p)`` (where ``p`` is a template param) with
        attribute ``_meta_position`` in ``{'expr', 'splice', 'store'}``.
      * Each ``ast.Expr(value=Name(id=p))`` element of a stmt-list field
        with attribute ``_meta_splice_param = p``.
    """

    def __init__(self, params: Set[str]):
        self.params = set(params)

    def analyze(self, body_list):
        self._visit_stmt_list(body_list)

    def _visit_stmt_list(self, stmts):
        for s in stmts:
            if (isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Name)
                    and s.value.id in self.params):
                p = s.value.id
                s._meta_splice_param = p
                s.value._meta_position = 'splice'
                # Don't recurse: the Name is fully classified.
                continue
            self._visit_node(s)

    def _visit_node(self, node):
        if isinstance(node, ast.Name) and node.id in self.params:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                node._meta_position = 'store'
            else:
                node._meta_position = 'expr'
            return

        # Recurse: descend by field, treating known stmt-list fields
        # specially so splice patterns can be detected at any depth.
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                if field in ('body', 'orelse', 'finalbody') and value \
                        and all(isinstance(v, ast.stmt) for v in value):
                    self._visit_stmt_list(value)
                else:
                    for item in value:
                        if isinstance(item, ast.AST):
                            self._visit_node(item)
            elif isinstance(value, ast.AST):
                self._visit_node(value)


# ---------------------------------------------------------------------------
# Parameter substitution transformer
# ---------------------------------------------------------------------------

class _ParamSubstituter(ast.NodeTransformer):
    """Replace tagged ``ast.Name`` nodes with their bound values.

    For each parameter ``Name`` occurrence, reads the ``_meta_position``
    attribute placed by ``_PositionAnalyzer`` (defaults to 'expr') and
    coerces the bound *raw value* per-occurrence. This ensures correct
    behaviour when the same parameter is used in multiple positions
    inside a template.
    """

    def __init__(self, raw_bindings):
        """
        Args:
            raw_bindings: dict mapping param_name -> raw Python value as
                provided at the call site (no coercion yet).
        """
        self.raw_bindings = raw_bindings

    def _bound_unpack(self, node, container_type):
        if isinstance(node, ast.Name) and node.id in self.raw_bindings:
            value = self.raw_bindings[node.id]
            if isinstance(value, container_type):
                return value
        return None

    def visit_Call(self, node):
        node.func = self.visit(node.func)
        args = []
        for arg in node.args:
            values = (self._bound_unpack(arg.value, (tuple, list))
                      if isinstance(arg, ast.Starred) else None)
            if values is None:
                args.append(self.visit(arg))
            else:
                args.extend(copy.deepcopy(_coerce_value_to_expr(
                    value)) for value in values)
        node.args = args

        keywords = []
        names = set()
        for entry in node.keywords:
            values = (self._bound_unpack(entry.value, dict)
                      if entry.arg is None else None)
            if values is None:
                expanded = [self.visit(entry)]
            else:
                expanded = []
                for name, value in values.items():
                    if not isinstance(name, str):
                        raise TypeError(
                            f"Invalid quoted keyword argument name: {name!r}")
                    expanded.append(ast.keyword(
                        arg=name, value=copy.deepcopy(_coerce_value_to_expr(
                            value))))
            for entry in expanded:
                if entry.arg is not None:
                    if entry.arg in names:
                        raise TypeError(
                            f"Duplicate quoted keyword argument: {entry.arg!r}")
                    names.add(entry.arg)
                keywords.append(entry)
        node.keywords = keywords
        return node

    def visit_Name(self, node):
        if node.id not in self.raw_bindings:
            return node
        raw = self.raw_bindings[node.id]
        position = getattr(node, '_meta_position', 'expr')
        coerced = _coerce_to_ast(raw, position)
        if isinstance(coerced, _SpliceMarker):
            # Splice context: leave as-is so the wrapping Expr can be
            # picked up by _expand_splices.
            return _SpliceMarker(stmts=copy.deepcopy(coerced.stmts))
        result = copy.deepcopy(coerced)
        # Preserve the syntactic context declared by the template site
        # when substituting Name -> Name. The template position dictates
        # Load/Store/Del; the binding only supplies the identifier.
        if isinstance(result, ast.Name):
            result.ctx = copy.copy(node.ctx)
        return result


# ---------------------------------------------------------------------------
# MetaTemplate
# ---------------------------------------------------------------------------

class MetaTemplate:
    """A compile-time template captured from a Python function.

    Attributes:
        source: Original source text (diagnostics).
        template_body: The captured ``list[ast.stmt]``.
        param_names: Ordered tuple of parameter names.
        param_positions: Mapping ``param_name -> position``. The position
            is the *primary* one (used for sanity checks); the actual
            substitution reads each Name node's own ``_meta_position``
            tag, so a parameter may be used in multiple positions.
        origin_file / origin_line: Diagnostics.
    """

    def __init__(self, source, definition, syntax, signature,
                 param_positions,
                 origin_file=None, origin_line=None,
                 debug_source_enabled=True):
        self.source = source
        self.definition = definition
        self.syntax = syntax
        self.symbol_names = frozenset(node.id for node in ast.walk(syntax)
                                      if isinstance(node, ast.Name))
        self.signature = signature
        self.param_names = tuple(signature.parameters)
        self.param_positions = dict(param_positions)
        self.origin_file = origin_file
        self.origin_line = origin_line
        self.debug_source_enabled = debug_source_enabled

    @property
    def template_body(self):
        return self.syntax.body

    def instantiate(self, /, *args, **kwargs) -> Fragment:
        """Instantiate the template and return a :class:`Fragment`."""
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        provided = bound.arguments

        # 2. Substitute into a deep copy of the template body. The
        #    substituter coerces each Name occurrence per its tagged
        #    position, so a single parameter used in multiple positions
        #    coerces correctly per-site.
        from ..decorators.visible import get_all_accessible_symbols
        from .._inline.definition_bindings import bind_definition
        from .._inline.names import ExpansionNames
        from .._inline.renaming import ScopedVariableRenamer, _parameters

        names = ExpansionNames()
        names.reserve_ast(self.syntax)
        parameter_aliases = {name: names.fresh('_quote_' + name)
                             for name in self.param_names}
        syntax = copy.deepcopy(self.syntax)
        renamer = ScopedVariableRenamer(parameter_aliases)
        syntax.body = [renamer.visit(stmt) for stmt in syntax.body]
        for argument in _parameters(syntax.args):
            argument.arg = parameter_aliases[argument.arg]
        definition = bind_definition(
            syntax, get_all_accessible_symbols(
                self.definition, names=self.symbol_names), names=names)
        _PositionAnalyzer(set(parameter_aliases.values())).analyze(
            definition.syntax.body)
        substituter = _ParamSubstituter({parameter_aliases[name]: value
                                        for name, value in provided.items()})
        result_body: List[ast.stmt] = [
            substituter.visit(stmt) for stmt in definition.syntax.body
        ]
        result_body = _expand_splices(result_body)
        result_body = _fold_const_if(result_body)

        # 3. Fix locations
        for node in result_body:
            ast.fix_missing_locations(node)

        # 4. Optional debug source
        debug_source = None
        if self.debug_source_enabled:
            try:
                debug_source = '\n'.join(ast.unparse(n) for n in result_body)
            except Exception:
                pass

        fragment = Fragment(
            body=result_body,
            origin_file=self.origin_file,
            origin_line=self.origin_line,
            debug_source=debug_source,
        )
        return fragment

    def __call__(self, /, *args, **kwargs) -> Fragment:
        return self.instantiate(*args, **kwargs)

    def __repr__(self):
        return "MetaTemplate(params={}, positions={})".format(
            self.param_names, self.param_positions,
        )


# ---------------------------------------------------------------------------
# Quote decorator
# ---------------------------------------------------------------------------

def _primary_position_for(param_name: str, body: List[ast.stmt]) -> str:
    """Return one canonical position for a parameter.

    Priority: splice > store > expr. The first occurrence with the
    highest priority wins. Used for the initial coercion choice; the
    substituter still reads each individual Name's own tag.
    """
    found = {'expr': False, 'store': False, 'splice': False}

    class _Scan(ast.NodeVisitor):
        def visit_Name(self, n):
            if n.id == param_name:
                pos = getattr(n, '_meta_position', 'expr')
                found[pos] = True

    for s in body:
        _Scan().visit(s)

    if found['splice']:
        return 'splice'
    if found['store']:
        return 'store'
    return 'expr'


def _make_template(func, debug_source=True) -> MetaTemplate:
    """Create a :class:`MetaTemplate` from a decorated Python function."""
    from ..utils.inspect_utils import get_object_source, get_object_sourcelines
    source = textwrap.dedent(get_object_source(func))
    tree = ast.parse(source)
    # The first top-level stmt is the function. If the user wrote a
    # decorated function, ``inspect.getsource`` returns the source with
    # decorators; ``ast.parse`` makes the first ``FunctionDef`` the
    # decorated def. Skip past any leading non-FunctionDef noise just in
    # case.
    func_def = None
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            func_def = stmt
            break
    if func_def is None:
        raise TypeError(
            "@meta.quote expects a function definition, got {}".format(
                type(tree.body[0]).__name__
            )
        )

    signature = inspect.signature(func)
    param_names = tuple(signature.parameters)
    template_body = func_def.body

    # Static position analysis
    _PositionAnalyzer(set(param_names)).analyze(template_body)
    param_positions = {
        p: _primary_position_for(p, template_body) for p in param_names
    }

    # Origin info for diagnostics
    origin_file = None
    origin_line = None
    try:
        origin_file = inspect.getfile(func)
        _, origin_line = get_object_sourcelines(func)
    except (OSError, TypeError):
        pass

    return MetaTemplate(
        source=source,
        definition=func,
        syntax=func_def,
        signature=signature,
        param_positions=param_positions,
        origin_file=origin_file,
        origin_line=origin_line,
        debug_source_enabled=debug_source,
    )


def quote(func=None, *, debug_source=True):
    """Capture a Python function body as a quasi-quote template.

    The decorated function's parameters define the template holes; its
    body is captured (always as ``list[ast.stmt]``) and substituted at
    instantiation. Calling the template returns a :class:`Fragment`.

    Template symbols bind in the definition environment. References supplied
    through template parameters retain their own compilation context.

    Example::

        @meta.quote
        def add(a, b):
            return a + b

        frag = add("x", "y")     # str -> Name in expr position
        frag.expr                 # BinOp(Name(x), Add, Name(y))
        frag.stmts                # [Return(value=BinOp(...))]
    """
    if func is None:
        def decorator(f):
            return _make_template(f, debug_source=debug_source)
        return decorator
    return _make_template(func, debug_source=debug_source)
