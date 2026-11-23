"""Turn generated function definitions into compile-only inline callables."""

import inspect
import sys

from ..logger import logger
from ..source import SourceLocation
from .generated import GeneratedFunction


def inline_generated(fn, *, user_globals=None):
    """Expand a GeneratedFunction in its caller without registering native work.

    The returned callable uses the same argument binding, hygiene, return
    handling and CFG ownership rules as @inline. It cannot be converted to a
    native function pointer. Native implementations referenced by its body
    must already be declared through the ordinary initialization path.
    A None return type denotes void, as it does in compile_generated.
    """
    call_site = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
    with logger.at_node(call_site):
        if not isinstance(fn, GeneratedFunction):
            logger.error('inline_generated expects GeneratedFunction', exc_type=TypeError)
        if fn.attrs:
            logger.error('Native function attributes do not apply to inline bodies',
                         exc_type=TypeError)

    from .._inline.definition_bindings import symbol_expression
    from ..builtin_entities.types import void
    from ..decorators.inline import _make_inline_callable

    syntax = fn.to_func_def(call_site=call_site)
    hints = fn.get_param_type_hints()
    for argument in syntax.args.args:
        hint = hints[argument.arg]
        if hint is not None:
            argument.annotation = symbol_expression('_inline_parameter_type', hint)
    return_type = void if fn.return_type is None else fn.return_type
    syntax.returns = symbol_expression('_inline_return_type', return_type)
    globals_ = dict(user_globals or {})
    globals_.update(fn.required_globals or {})
    result = _make_inline_callable(syntax, lambda: dict(globals_))
    result.__signature__ = inspect.Signature(
        [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           annotation=hint if hint is not None else inspect.Parameter.empty)
         for name, hint in fn.params],
        return_annotation=return_type)
    return result
