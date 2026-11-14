"""Shared LLVM floating types, exact constants, conversions and intrinsics.

All formats use round-to-nearest, ties-to-even at the constant boundary.
Dynamic arithmetic remains LLVM instructions. Materialized constants work
with both LLVM 15 and LLVM 20 without floating ConstantExpr dependencies.
"""

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
import math
import operator
import struct

from llvmlite import ir


@dataclass(frozen=True)
class FloatFormat:
    llvm_type: ir.Type
    exponent_bits: int
    fraction_bits: int
    prefix: str
    native_pack: str = ''

    @property
    def bits(self):
        return 1 + self.exponent_bits + self.fraction_bits

    @property
    def exponent_mask(self):
        return (1 << self.exponent_bits) - 1

    @property
    def bias(self):
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def sign_mask(self):
        return 1 << (self.bits - 1)

    def contains(self, other):
        return (self.exponent_bits >= other.exponent_bits
                and self.fraction_bits >= other.fraction_bits)

    def encode(self, value):
        """Round an exact numeric input once to this binary format."""
        if isinstance(value, str):
            value = Decimal(value)
        if isinstance(value, FloatValue):
            if value.format == self:
                return value
            sign = int(value.negative)
            if value.is_special:
                payload = value.fraction
                if payload:
                    shift = self.fraction_bits - value.format.fraction_bits
                    payload = payload << shift if shift >= 0 else payload >> -shift
                    payload |= 1 << (self.fraction_bits - 1)
                return FloatValue(self, (sign << (self.bits - 1))
                                  | (self.exponent_mask << self.fraction_bits)
                                  | payload)
            numerator, denominator = value.as_integer_ratio()
        elif isinstance(value, float):
            if not math.isfinite(value):
                bits = struct.unpack('>Q', struct.pack('>d', value))[0]
                return self.encode(FloatValue(F64, bits))
            numerator, denominator = value.as_integer_ratio()
            sign = int(math.copysign(1.0, value) < 0)
        elif isinstance(value, int):
            numerator, denominator = value, 1
            sign = int(value < 0)
        elif isinstance(value, Decimal):
            if not value.is_finite():
                return self.encode(float(value))
            numerator, denominator = value.as_integer_ratio()
            sign = int(value.is_signed())
        else:
            numerator, denominator = value.as_integer_ratio()
            sign = int(numerator < 0)
        numerator = abs(numerator)
        if not numerator:
            return FloatValue(self, sign << (self.bits - 1))
        exponent = numerator.bit_length() - denominator.bit_length()
        if ((numerator < denominator << exponent) if exponent >= 0
                else (numerator << -exponent < denominator)):
            exponent -= 1
        minimum = 1 - self.bias
        shift = self.fraction_bits - max(exponent, minimum)
        scaled, divisor = ((numerator << shift, denominator) if shift >= 0
                           else (numerator, denominator << -shift))
        significand, remainder = divmod(scaled, divisor)
        halfway = 2 * remainder - divisor
        significand += halfway > 0 or (halfway == 0 and significand & 1)
        if significand >= 1 << (self.fraction_bits + 1):
            significand >>= 1
            exponent += 1
        encoded_exponent = max(exponent, minimum) + self.bias
        if encoded_exponent >= self.exponent_mask:
            encoded_exponent, significand = self.exponent_mask, 0
        elif significand < 1 << self.fraction_bits:
            encoded_exponent = 0
        fraction = significand & ((1 << self.fraction_bits) - 1)
        bits = ((sign << (self.bits - 1))
                | (encoded_exponent << self.fraction_bits) | fraction)
        return FloatValue(self, bits)

    def normalize(self, value):
        """Keep native Python floats directly when they represent this format."""
        if self.native_pack == 'd' and isinstance(value, (int, float)):
            try:
                return float(value)
            except OverflowError:
                return -math.inf if value < 0 else math.inf
        if self.native_pack and isinstance(value, float):
            try:
                return struct.unpack('>' + self.native_pack,
                                     struct.pack('>' + self.native_pack, value))[0]
            except OverflowError:
                return math.copysign(math.inf, value)
        rounded = self.encode(value)
        return float(rounded) if F64.contains(self) else rounded

    def literal(self, value):
        if self.native_pack:
            return self.llvm_type.format_constant(self.normalize(value))
        bits = self.encode(value).bits
        if self.bits > 64:
            # LLVM writes the low 64-bit word first in wide literals.
            return f'0x{self.prefix}{bits & ((1 << 64) - 1):016X}{bits >> 64:016X}'
        return f'0x{self.prefix}{bits:0{self.bits // 4}X}'


@dataclass(frozen=True, eq=False)
class FloatValue:
    """Exact target bits shared by constant construction and operations."""

    format: FloatFormat
    bits: int

    @property
    def negative(self):
        return bool(self.bits & self.format.sign_mask)

    @property
    def fraction(self):
        return self.bits & ((1 << self.format.fraction_bits) - 1)

    @property
    def exponent(self):
        return ((self.bits >> self.format.fraction_bits)
                & self.format.exponent_mask)

    @property
    def is_special(self):
        return self.exponent == self.format.exponent_mask

    @property
    def is_nan(self):
        return self.is_special and bool(self.fraction)

    def as_integer_ratio(self):
        if self.is_special:
            raise ValueError('Non-finite floating constant has no integer ratio')
        fraction = self.fraction
        exponent = 1 - self.format.bias - self.format.fraction_bits
        if self.exponent:
            fraction |= 1 << self.format.fraction_bits
            exponent += self.exponent - 1
        if not fraction:
            return 0, 1
        numerator, denominator = ((fraction << exponent, 1) if exponent >= 0
                                  else (fraction, 1 << -exponent))
        return (-numerator if self.negative else numerator), denominator

    def __neg__(self):
        return FloatValue(self.format, self.bits ^ self.format.sign_mask)

    def __pos__(self):
        return self

    def __abs__(self):
        return FloatValue(self.format, self.bits & ~self.format.sign_mask)

    def _binary(self, other, operation):
        other = self.format.encode(other)
        if (operation is operator.mod and other.is_special and not other.is_nan
                and not self.is_special):
            if self.negative == other.negative:
                return self
            return other if self else self.format.encode(-0.0 if other.negative else 0.0)
        if self.is_special or other.is_special:
            def representative(value):
                if value.is_special:
                    return float(value)
                return math.copysign(float(bool(value)), -1 if value.negative else 1)
            result = operation(representative(self), representative(other))
        else:
            result = operation(Fraction(*self.as_integer_ratio()),
                               Fraction(*other.as_integer_ratio()))
            if not result:
                if operation is operator.add:
                    negative = self.negative and other.negative
                elif operation is operator.sub:
                    negative = self.negative and not other.negative
                elif operation is operator.mod:
                    negative = other.negative
                else:
                    negative = self.negative != other.negative
                result = -0.0 if negative else 0.0
        return self.format.encode(result)

    def __add__(self, other):
        return self._binary(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary(other, operator.sub)

    def __rsub__(self, other):
        return self.format.encode(other) - self

    def __mul__(self, other):
        return self._binary(other, operator.mul)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.format.encode(other) / self

    def __floordiv__(self, other):
        return self._binary(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self.format.encode(other) // self

    def __mod__(self, other):
        return self._binary(other, operator.mod)

    def __rmod__(self, other):
        return self.format.encode(other) % self

    def _compare(self, other, operation):
        if self.is_nan:
            return False
        if isinstance(other, FloatValue):
            if other.is_nan:
                return False
            other = float(other) if other.is_special else Fraction(*other.as_integer_ratio())
        value = float(self) if self.is_special else Fraction(*self.as_integer_ratio())
        return operation(value, other)

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    def __bool__(self):
        return bool(self.bits & (self.format.sign_mask - 1))

    def __int__(self):
        numerator, denominator = self.as_integer_ratio()
        result = abs(numerator) // denominator
        return -result if numerator < 0 else result

    def __float__(self):
        return struct.unpack('>d', struct.pack('>Q', F64.encode(self).bits))[0]

    def __eq__(self, other):
        if not isinstance(other, (FloatValue, int, float)):
            return NotImplemented
        if self.is_nan or (isinstance(other, FloatValue) and other.is_nan):
            return False
        if self.is_special:
            if isinstance(other, FloatValue):
                return other.is_special and self.negative == other.negative
            return other == float(self)
        if isinstance(other, float):
            if not math.isfinite(other):
                return False
            other = Fraction(*other.as_integer_ratio())
        elif isinstance(other, FloatValue):
            if other.is_special:
                return False
            other = Fraction(*other.as_integer_ratio())
        return Fraction(*self.as_integer_ratio()) == other

    def __hash__(self):
        if self.is_special:
            return hash(float(self)) if not self.is_nan else hash((self.format, self.bits))
        return hash(Fraction(*self.as_integer_ratio()))

    def __repr__(self):
        if self.is_special:
            return repr(float(self))
        if not self:
            return '-0.0' if self.negative else '0.0'
        numerator, denominator = self.as_integer_ratio()
        context = Context(prec=len(str(1 << self.format.fraction_bits)) + 2,
                          rounding=ROUND_HALF_EVEN, traps=[])
        with localcontext(context):
            return str(Decimal(numerator) / Decimal(denominator))


class _ExtendedFloatType(ir.Type):
    def __str__(self):
        return self.llvm_name

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def format_constant(self, value):
        return format_for_type(self).literal(value)


class BFloatType(_ExtendedFloatType):
    llvm_name = 'bfloat'
    intrinsic_name = 'bf16'
    null = '0xR0000'


class FP128Type(_ExtendedFloatType):
    llvm_name = 'fp128'
    intrinsic_name = 'f128'
    null = '0xL00000000000000000000000000000000'


F16 = FloatFormat(ir.HalfType(), 5, 10, 'H', 'e')
BF16 = FloatFormat(BFloatType(), 8, 7, 'R')
F32 = FloatFormat(ir.FloatType(), 8, 23, '', 'f')
F64 = FloatFormat(ir.DoubleType(), 11, 52, '', 'd')
F128 = FloatFormat(FP128Type(), 15, 112, 'L')
FLOAT_FORMATS = (F16, BF16, F32, F64, F128)


def format_for_type(llvm_type):
    return next((fmt for fmt in FLOAT_FORMATS if fmt.llvm_type == llvm_type), None)


def is_float_type(llvm_type):
    return format_for_type(llvm_type) is not None


def float_type_classes():
    return tuple(type(fmt.llvm_type) for fmt in FLOAT_FORMATS)


def constant(llvm_type, value):
    fmt = format_for_type(llvm_type)
    return ir.Constant(llvm_type, fmt.normalize(value) if fmt is not None else value)


def constant_operation(llvm_type, operation, left, right):
    fmt = format_for_type(llvm_type)
    if fmt is None:
        return operation(left, right)
    if F64.contains(fmt):
        return fmt.normalize(operation(fmt.normalize(left), fmt.normalize(right)))
    return fmt.normalize(fmt.encode(left)._binary(right, operation))


def _constant_value(value):
    if isinstance(value, ir.Constant) and value.constant is None:
        return format_for_type(value.type).normalize(0)
    if (isinstance(value, ir.Constant)
            and isinstance(value.constant, (int, float, FloatValue))):
        return format_for_type(value.type).normalize(value.constant)
    return None


def negate(builder, value, name=''):
    folded = _constant_value(value) if is_float_type(value.type) else None
    return (constant(value.type, -folded) if folded is not None
            else builder.fneg(value, name=name))


def binary(builder, operation, left, right, name=''):
    fmt = format_for_type(left.type)
    a = _constant_value(left) if fmt is not None else None
    b = _constant_value(right) if fmt is not None else None
    if a is not None and b is not None:
        operations = {'fadd': operator.add, 'fsub': operator.sub,
                      'fmul': operator.mul, 'fdiv': operator.truediv}
        try:
            result = constant_operation(left.type, operations[operation], a, b)
        except ZeroDivisionError:
            a, b = fmt.encode(a), fmt.encode(b)
            if a.is_nan or not a:
                result = math.nan
            else:
                result = -math.inf if a.negative != b.negative else math.inf
        return constant(left.type, result)
    return getattr(builder, operation)(left, right, name=name)


def compare(builder, operation, left, right, *, ordered, name=''):
    fmt = format_for_type(left.type)
    a = _constant_value(left) if fmt is not None else None
    b = _constant_value(right) if fmt is not None else None
    if a is not None and b is not None:
        operations = {'==': operator.eq, '!=': operator.ne, '<': operator.lt,
                      '<=': operator.le, '>': operator.gt, '>=': operator.ge}
        unordered = ((a.is_nan if isinstance(a, FloatValue) else math.isnan(a))
                     or (b.is_nan if isinstance(b, FloatValue) else math.isnan(b)))
        if operation in ('true', 'false', 'ord', 'uno'):
            result = {'true': True, 'false': False,
                      'ord': not unordered, 'uno': unordered}[operation]
        else:
            if operation not in operations:
                ordered = operation[0] == 'o'
                operation = {'eq': '==', 'ne': '!=', 'lt': '<',
                             'le': '<=', 'gt': '>', 'ge': '>='}[operation[1:]]
            result = not ordered if unordered else operations[operation](a, b)
        return ir.Constant(ir.IntType(1), int(result))
    method = builder.fcmp_ordered if ordered else builder.fcmp_unordered
    return method(operation, left, right, name=name)


def convert(builder, value, target, *, source_unsigned=False,
            target_unsigned=False, target_bool=False):
    """Fold a conversion using the source format, or emit runtime instructions."""
    source_format = format_for_type(value.type)
    target_format = format_for_type(target)
    if target_bool:
        folded = _constant_value(value)
        return (ir.Constant(target, int(bool(folded))) if folded is not None
                else builder.fcmp_unordered('!=', value, constant(value.type, 0)))
    if source_format is not None:
        folded = _constant_value(value)
        if folded is not None:
            if target_format is not None:
                return constant(target, folded)
            if ((isinstance(folded, float) and not math.isfinite(folded))
                    or (isinstance(folded, FloatValue) and folded.is_special)):
                return ir.FormattedConstant(target, 'poison')
            integer = int(folded)
            lower = 0 if target_unsigned else -(1 << (target.width - 1))
            upper = 1 << (target.width - int(not target_unsigned))
            if not lower <= integer < upper:
                return ir.FormattedConstant(target, 'poison')
            return ir.Constant(target, integer)
        if target_format is None:
            operation = builder.fptoui if target_unsigned else builder.fptosi
            return operation(value, target)
        if source_format == target_format:
            return value
        if target_format.contains(source_format):
            return builder.fpext(value, target)
        if source_format.contains(target_format):
            return builder.fptrunc(value, target)
        # Incomparable formats meet at a format containing both.
        common = next(fmt for fmt in FLOAT_FORMATS
                      if fmt.contains(source_format) and fmt.contains(target_format))
        return builder.fptrunc(builder.fpext(value, common.llvm_type), target)
    if isinstance(value, ir.Constant) and isinstance(value.constant, int):
        integer = value.constant & ((1 << value.type.width) - 1)
        if not source_unsigned and integer & (1 << (value.type.width - 1)):
            integer -= 1 << value.type.width
        return constant(target, integer)
    operation = builder.uitofp if source_unsigned else builder.sitofp
    return operation(value, target)


def intrinsic(module, name, llvm_type):
    return module.declare_intrinsic(name, [llvm_type])


def fold_instruction(operation, args):
    """Fold before a captured builder operation becomes a runtime register."""
    if operation in ('fadd', 'fsub', 'fmul', 'fdiv'):
        left, right = args[:2]
        if (is_float_type(left.type) and _constant_value(left) is not None
                and _constant_value(right) is not None):
            return binary(None, operation, left, right)
    elif operation in ('fcmp_ordered', 'fcmp_unordered'):
        predicate, left, right = args[:3]
        if (is_float_type(left.type) and _constant_value(left) is not None
                and _constant_value(right) is not None):
            return compare(None, predicate, left, right,
                           ordered=operation == 'fcmp_ordered')
    elif operation == 'fneg':
        value = args[0]
        if is_float_type(value.type) and _constant_value(value) is not None:
            return negate(None, value)
    elif operation in ('fpext', 'fptrunc'):
        value, target = args[:2]
        if (is_float_type(value.type) and is_float_type(target)
                and _constant_value(value) is not None):
            return convert(None, value, target)
    elif operation in ('fptosi', 'fptoui'):
        value, target = args[:2]
        if is_float_type(value.type) and _constant_value(value) is not None:
            return convert(None, value, target, target_unsigned=operation == 'fptoui')
    elif operation in ('sitofp', 'uitofp'):
        value, target = args[:2]
        if (is_float_type(target) and isinstance(value, ir.Constant)
                and isinstance(value.constant, int)):
            return convert(None, value, target, source_unsigned=operation == 'uitofp')
    return None
