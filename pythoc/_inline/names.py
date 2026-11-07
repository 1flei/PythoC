"""Collision-free names for compiler-created expansion bindings."""

import ast


class ExpansionNames:
    """Reserve source identifiers separately from capture visibility."""

    def __init__(self):
        self.reserved = set()
        self._next_suffix = {}

    def reserve(self, names):
        self.reserved.update(names)

    def reserve_ast(self, *nodes):
        for root in nodes:
            if isinstance(root, str):
                self.reserved.add(root)
                continue
            for node in ast.walk(root):
                if isinstance(node, ast.Name):
                    self.reserved.add(node.id)
                elif isinstance(node, ast.arg):
                    self.reserved.add(node.arg)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef)):
                    self.reserved.add(node.name)
                elif (isinstance(node, ast.Constant)
                      and isinstance(node.value, str)):
                    # Scoped labels use string literals as their identifiers.
                    self.reserved.add(node.value)
                elif (isinstance(node, (ast.MatchAs, ast.MatchStar))
                      and node.name):
                    self.reserved.add(node.name)

    def fresh(self, preferred):
        index = self._next_suffix.get(preferred, 0)
        name = f'{preferred}_{index}' if index else preferred
        while name in self.reserved:
            index += 1
            name = f'{preferred}_{index}'
        self.reserved.add(name)
        self._next_suffix[preferred] = index + 1
        return name


def visitor_expansion_names(visitor):
    """Include visible bindings and globals before allocating an alias."""
    names = visitor.expansion_names
    names.reserve(visitor.scope_manager.get_all_visible())
    names.reserve(visitor.ctx.user_globals or {})
    return names
