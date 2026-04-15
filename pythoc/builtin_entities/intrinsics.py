from llvmlite import ir
from .types import ptr, i8, void
from ..valueref import wrap_value

from .typeof import typeof
from .sizeof import sizeof
from .char import char
from .seq import seq
from .consume import consume
from .assume import assume
from .refine import refine
from .defer import defer
from .scoped_label import label, goto, goto_begin, goto_end
from .va_builtins import va_start, va_arg, va_end

nullptr = wrap_value(ir.Constant(ir.PointerType(ir.IntType(8)), None), kind="value", type_hint=ptr[void])

__all__ = [
    'typeof',
    'sizeof',
    'char',
    'seq',
    'consume',
    'assume',
    'refine',
    'nullptr',
    'defer',
    # Scoped goto/label
    'label',
    'goto',
    'goto_begin',  # Backward compatibility alias
    'goto_end',
    # C ABI varargs
    'va_start',
    'va_arg',
    'va_end',
]
