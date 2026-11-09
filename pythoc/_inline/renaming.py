"""Rename expansion bindings while respecting nested function scopes."""

import ast
import copy

from .scope_analyzer import ScopeAnalyzer, ScopeContext


def _parameters(arguments):
    fixed = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    return fixed + [arg for arg in (arguments.vararg, arguments.kwarg)
                    if arg is not None]


class ScopedVariableRenamer(ast.NodeTransformer):
    def __init__(self, rename_map, global_renames=None):
        self.rename_map = rename_map
        self.global_renames = global_renames or {}

    def visit_Name(self, node):
        name = self.rename_map.get(node.id)
        if name is None and isinstance(node.ctx, ast.Load):
            name = self.global_renames.get(node.id)
        if name is None:
            return node
        return ast.copy_location(ast.Name(id=name, ctx=node.ctx), node)

    def visit_MatchAs(self, node):
        self.generic_visit(node)
        if node.name:
            node.name = self.rename_map.get(node.name, node.name)
        return node

    def visit_MatchStar(self, node):
        if node.name:
            node.name = self.rename_map.get(node.name, node.name)
        return node

    def visit_MatchMapping(self, node):
        self.generic_visit(node)
        if node.rest:
            node.rest = self.rename_map.get(node.rest, node.rest)
        return node

    def _definition_arguments(self, arguments):
        """Defaults and annotations resolve outside the new function scope."""
        arguments = copy.deepcopy(arguments)
        arguments.defaults = [self.visit(value)
                              for value in arguments.defaults]
        arguments.kw_defaults = [
            self.visit(value) if value is not None else None
            for value in arguments.kw_defaults
        ]
        for argument in _parameters(arguments):
            if argument.annotation is not None:
                argument.annotation = self.visit(argument.annotation)
        return arguments

    def visit_FunctionDef(self, node):
        node = copy.copy(node)
        node.name = self.rename_map.get(node.name, node.name)
        node.args = self._definition_arguments(node.args)
        if node.returns is not None:
            node.returns = self.visit(copy.deepcopy(node.returns))
        node.decorator_list = [self.visit(copy.deepcopy(value))
                               for value in node.decorator_list]

        analyzer = ScopeAnalyzer(ScopeContext(set(self.rename_map)))
        _, locals_, parameters_ = analyzer.analyze(
            node.body, _parameters(node.args))
        nested_map = {name: target for name, target in self.rename_map.items()
                      if name not in locals_ and name not in parameters_}
        hidden = locals_ | parameters_
        nested_globals = {name: target for name, target in self.global_renames.items()
                          if name not in hidden}
        renamer = ScopedVariableRenamer(nested_map, nested_globals)
        node.body = [renamer.visit(copy.deepcopy(stmt)) for stmt in node.body]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        node = copy.deepcopy(node)
        node.args = self._definition_arguments(node.args)
        shadowed = {parameter.arg for parameter in _parameters(node.args)}
        renamer = ScopedVariableRenamer({
            name: target for name, target in self.rename_map.items()
            if name not in shadowed
        }, {name: target for name, target in self.global_renames.items()
            if name not in shadowed})
        node.body = renamer.visit(node.body)
        return node

    def _visit_comprehension(self, node):
        node = copy.deepcopy(node)
        renamer = self
        for generator in node.generators:
            generator.iter = renamer.visit(generator.iter)
            bound = {name.id for name in ast.walk(generator.target)
                     if isinstance(name, ast.Name)}
            renamer = ScopedVariableRenamer(
                {**renamer.rename_map, **{name: name for name in bound}},
                renamer.global_renames)
            generator.target = renamer.visit(generator.target)
            generator.ifs = [renamer.visit(value) for value in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = renamer.visit(node.key)
            node.value = renamer.visit(node.value)
        else:
            node.elt = renamer.visit(node.elt)
        return node

    visit_GeneratorExp = _visit_comprehension
    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
