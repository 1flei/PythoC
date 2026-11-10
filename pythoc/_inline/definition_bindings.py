"""Prepare lexical references before an AST body enters a consumer's scope."""

import ast
import builtins
import copy
from dataclasses import dataclass

from .names import ExpansionNames, visitor_expansion_names
from .renaming import ScopedVariableRenamer, _parameters
from .scope_analyzer import ScopeAnalyzer, ScopeContext


@dataclass
class BoundDefinition:
    syntax: ast.FunctionDef
    globals: dict
    references: dict
    reference_names: dict


def _global_aliases(syntax, globals_, protected, names):
    from ..builtin_entities import get_builtin_entity
    from ._intrinsics import _PC_INTRINSICS

    referenced = {node.id for node in ast.walk(syntax)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    aliases = {name: names.fresh('_definition_' + name)
               for name in sorted(referenced - protected)}
    values = {}
    for name, alias in aliases.items():
        if name in globals_:
            values[alias] = globals_[name]
        elif name == '__pc_intrinsics':
            values[alias] = _PC_INTRINSICS
        else:
            builtin = get_builtin_entity(name)
            if builtin is not None:
                values[alias] = builtin
            elif name in vars(builtins):
                values[alias] = vars(builtins)[name]
        # An unresolved name retains a fresh unbound identifier. It cannot
        # accidentally resolve to a same-named consumer local, even later.
    return aliases, values


def bind_definition(syntax, globals_, references=None, *, names=None):
    """Rename accesses to original bindings; never snapshot their values."""
    references = references or {}
    globals_ = dict(globals_ or {})
    names = names if names is not None else ExpansionNames()
    names.reserve_ast(syntax)
    names.reserve(globals_)
    _, local, parameters = ScopeAnalyzer(ScopeContext(set(references))).analyze(
        syntax.body, _parameters(syntax.args))
    bound = local | parameters
    captures = {name: names.fresh('_capture_' + name)
                for name in sorted(references.keys() - bound)}
    global_names, global_values = _global_aliases(
        syntax, globals_, bound | references.keys(), names)
    renamer = ScopedVariableRenamer(
        {**{name: name for name in bound}, **captures}, global_names)
    prepared = copy.deepcopy(syntax)
    prepared.body = [renamer.visit(statement) for statement in prepared.body]
    for parameter in _parameters(prepared.args):
        if parameter.annotation is not None:
            parameter.annotation = renamer.visit(parameter.annotation)
    if prepared.returns is not None:
        prepared.returns = renamer.visit(prepared.returns)
    globals_.update(global_values)
    used = {node.id for node in ast.walk(prepared)
            if isinstance(node, ast.Name)}
    used.update(node.name for node in ast.walk(prepared)
                if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name)
    used.update(node.rest for node in ast.walk(prepared)
                if isinstance(node, ast.MatchMapping) and node.rest)
    capture_names = {alias: name for name, alias in captures.items()
                     if alias in used}
    return BoundDefinition(
        prepared, globals_,
        {alias: references[name] for alias, name in capture_names.items()},
        capture_names)


def bind_consumer_globals(visitor, body, target):
    """Keep consumer globals independent of the producer's definition scope."""
    scope = set(visitor.scope_manager.get_all_visible())
    scope.update(node.id for node in ast.walk(target) if isinstance(node, ast.Name))
    _, local, _ = ScopeAnalyzer(ScopeContext(scope)).analyze(body, [])
    scope.update(local)
    globals_ = visitor.ctx.user_globals or {}
    names = visitor_expansion_names(visitor)
    syntax = ast.Module(body=body, type_ignores=[])
    global_names, global_values = _global_aliases(syntax, globals_, scope, names)
    renamer = ScopedVariableRenamer({name: name for name in scope}, global_names)
    prepared = [renamer.visit(copy.deepcopy(statement)) for statement in body]
    return prepared, global_values
