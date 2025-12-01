from llvmlite import ir
from .types import ptr, i8
from ..valueref import wrap_value

from .typeof import typeof
from .sizeof import sizeof
from .char import char
from .seq import seq
from .consume import consume
from .assume import assume
from .refine import refine

nullptr = wrap_value(ir.Constant(ir.PointerType(ir.IntType(8)), None), kind="value", type_hint=ptr[i8])

__all__ = [
    'typeof',
    'sizeof',
    'char',
    'seq',
    'consume',
    'assume',
    'refine',
    'nullptr',
]
