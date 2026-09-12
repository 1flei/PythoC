"""IEEE-754 float constants: infinity and quiet NaN, in both widths.

These lower to LLVM constant bit patterns; the C11 macro ``INFINITY``
(math.h) denotes the same value as ``inf``.
"""

from llvmlite import ir

from .types import f32, f64
from ..valueref import wrap_value

# llvmlite float types are singleton classes: instantiate without args.
_F32 = ir.FloatType()
_F64 = ir.DoubleType()

# +inf: exponent all ones, mantissa zero; quiet NaN: exponent all ones,
# MSB of mantissa set.  Python float('inf')/float('nan') carry exactly
# these IEEE-754 bit patterns.
_F32_INF = ir.Constant(_F32, float("inf"))
_F64_INF = ir.Constant(_F64, float("inf"))
_F32_NAN = ir.Constant(_F32, float("nan"))
_F64_NAN = ir.Constant(_F64, float("nan"))

inf = wrap_value(_F64_INF, kind="value", type_hint=f64)
inff = wrap_value(_F32_INF, kind="value", type_hint=f32)
nan = wrap_value(_F64_NAN, kind="value", type_hint=f64)
nanf = wrap_value(_F32_NAN, kind="value", type_hint=f32)

__all__ = [
    'inf',
    'inff',
    'nan',
    'nanf',
]
