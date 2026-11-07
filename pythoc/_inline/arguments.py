"""Bridge evaluated values to syntax without replaying source arguments."""

import ast
import copy
from typing import Iterable

from ..context import VariableInfo
from ..valueref import ValueRef
from .names import visitor_expansion_names


def bind_argument_values(visitor, values: Iterable[ValueRef]) -> list[str]:
    """Expose values under fresh names, retaining existing CFG ownership.

    Expanded parameter bindings perform the ownership transfer. These names
    are aliases, not new resources. Bare AST nodes are a protocol error;
    explicit syntax carriers must arrive inside a Python-valued ValueRef.
    """
    values = tuple(values)
    if any(not isinstance(value, ValueRef) for value in values):
        raise TypeError('AST expansion requires evaluated call arguments (ValueRef)')
    allocator = visitor_expansion_names(visitor)
    names = []
    for value in values:
        name = allocator.fresh('_arg_inline')
        info = VariableInfo(
            name=name, value_ref=value, alloca=None,
            source='inline_arg_temp', is_parameter=False,
        )
        visitor.scope_manager.declare_variable(info, allow_shadow=False)
        names.append(name)
    return names


def bind_evaluated_call(visitor, values, node):
    """Replace a source call's arguments with aliases to normalized values."""
    names = bind_argument_values(visitor, values)
    bound = copy.copy(node)
    bound.args = [ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
                  for name in names]
    bound.keywords = []
    return bound
