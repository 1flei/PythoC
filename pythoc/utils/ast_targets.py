"""Shared syntax rules for assignment targets."""

import ast


def unpack_target_items(target):
    """Return the children of a tuple assignment target, or None for a leaf."""
    return target.elts if isinstance(target, ast.Tuple) else None


def iter_unpack_leaves(target, path=()):
    """Yield each leaf and its field path, in assignment order."""
    items = unpack_target_items(target)
    if items is None:
        yield path, target
        return
    for index, item in enumerate(items):
        yield from iter_unpack_leaves(item, path + (index,))


def is_rebinding_target(target):
    """Check for an underscore assignment target, including after renaming."""
    return (isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
            and (target.id == '_' or getattr(target, '_pc_rebind', False)))


def iter_rebinding_names(target):
    """Yield rebinding names in an assignment target tree."""
    yield from (node.id for _, node in iter_unpack_leaves(target) if is_rebinding_target(node))


def iter_bound_names(target):
    """Yield the names introduced by an assignment target."""
    for _, leaf in iter_unpack_leaves(target):
        if isinstance(leaf, ast.Name):
            yield leaf.id
        elif isinstance(leaf, ast.List):
            for item in leaf.elts:
                yield from iter_bound_names(item)
        elif isinstance(leaf, ast.Starred):
            yield from iter_bound_names(leaf.value)
