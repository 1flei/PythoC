"""pc_literal: Python-level tagged value carrier for PythoC types.

pc_literal wraps a Python value together with a PythoC type tag so that
``u64(42)`` called outside ``@compile`` returns something meaningful:

    x = u64(42)          # pc_literal(42, u64)
    y = x + u64(8)       # pc_literal(50, u64)  -- Python-level arithmetic
    @compile
    def foo():
        a = x            # lowered to ir.Constant(IntType(64), 42)

Design differences from pc_list / pc_tuple / pc_dict:
- **Instance-level**, not type-level (users interact with it as a value)
- Supports Python dunder arithmetic and comparisons
- Bridges Python and compile worlds via the ``get_value()`` protocol
  already present in ``_wrap_python_name_binding``
"""

from __future__ import annotations

import operator
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _get_type_width(pc_type) -> int:
    return getattr(pc_type, '_size_bytes', 0) * 8


def _is_float(pc_type) -> bool:
    return getattr(pc_type, '_is_float', False)


def _is_signed(pc_type) -> bool:
    return getattr(pc_type, '_is_signed', True)


def _is_integer(pc_type) -> bool:
    return getattr(pc_type, '_is_integer', False)


def _is_bool_type(pc_type) -> bool:
    return getattr(pc_type, '_is_bool', False)


def _coerce_types(left_type, right_type):
    """Determine the result type for a binary operation between two PC types.

    Rules (mirror compile-time ``unify_binop_types``):
    - Float always wins over int
    - Wider type wins
    - Mixed sign at same width -> signed
    """
    l_float = _is_float(left_type)
    r_float = _is_float(right_type)

    if l_float and not r_float:
        return left_type
    if r_float and not l_float:
        return right_type
    if l_float and r_float:
        return left_type if _get_type_width(left_type) >= _get_type_width(right_type) else right_type

    l_w = _get_type_width(left_type)
    r_w = _get_type_width(right_type)
    if l_w != r_w:
        return left_type if l_w > r_w else right_type

    # Same width: signed wins
    if _is_signed(left_type):
        return left_type
    if _is_signed(right_type):
        return right_type
    return left_type


def _mask_int(value: int, pc_type) -> int:
    """Mask/truncate an integer to the width of *pc_type*."""
    width = _get_type_width(pc_type)
    if width <= 0 or width > 128:
        return value
    mask = (1 << width) - 1
    value = value & mask
    if _is_signed(pc_type) and value >= (1 << (width - 1)):
        value -= (1 << width)
    return value


class pc_literal:
    """Tagged Python value carrying a PythoC type.

    Scalars store a plain ``int`` / ``float`` / ``bool`` in ``_value``.
    Compound types (struct, array, enum) store richer payloads; see
    ``_from_composite_call`` and ``_from_ctypes_result``.
    """

    __slots__ = ('_value', '_pc_type', '_fields', '_field_names')

    def __init__(self, value: Any, pc_type: Any):
        self._value = value
        self._pc_type = pc_type
        self._fields: Optional[dict] = None
        self._field_names: Optional[list] = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def _from_type_call(cls, pc_type, *args, **kwargs):
        """Called by ``BuiltinEntityMeta.__call__`` for scalar types."""
        if len(args) != 1 or kwargs:
            from ..logger import logger
            logger.error(
                f"{pc_type.get_name()}() takes exactly 1 argument "
                f"({len(args)} given)",
                node=None, exc_type=TypeError,
            )
        value = args[0]
        if isinstance(value, pc_literal):
            value = value._value
        if _is_float(pc_type):
            value = float(value)
        elif _is_integer(pc_type) or _is_bool_type(pc_type):
            value = _mask_int(int(value), pc_type)
        return cls(value, pc_type)

    @classmethod
    def _from_composite_call(cls, unified_type, *args, **kwargs):
        """Called by injected ``__new__`` on @compile class / @union / @enum."""
        inst = cls.__new__(cls)
        inst._pc_type = unified_type
        field_names = getattr(unified_type, '_field_names', None) or []
        inst._field_names = list(field_names)

        if not args and not kwargs:
            # Zero-arg construction: uninitialised fields
            inst._fields = {n: None for n in field_names}
            inst._value = None
        else:
            # Positional arg construction (Python-level convenience)
            inst._fields = {}
            for i, name in enumerate(field_names):
                if i < len(args):
                    inst._fields[name] = args[i]
                elif name in kwargs:
                    inst._fields[name] = kwargs[name]
                else:
                    inst._fields[name] = None
            inst._value = inst._fields
        return inst

    @classmethod
    def _from_ctypes_result(cls, ctypes_result, pc_type):
        """Wrap a ctypes return value from native execution."""
        import ctypes as ct

        if pc_type is None:
            return ctypes_result

        if _is_bool_type(pc_type):
            return cls(builtins_bool(ctypes_result), pc_type)

        if _is_integer(pc_type):
            return cls(int(ctypes_result), pc_type)

        if _is_float(pc_type):
            return cls(float(ctypes_result), pc_type)

        if getattr(pc_type, '_is_pointer', False):
            raw = ctypes_result
            if hasattr(raw, 'value'):
                raw = raw.value
            elif not isinstance(raw, int):
                raw = ct.cast(raw, ct.c_void_p).value or 0
            return cls(int(raw), pc_type)

        if isinstance(ctypes_result, ct.Structure):
            return cls._struct_from_ctypes(ctypes_result, pc_type)

        return ctypes_result

    @classmethod
    def _struct_from_ctypes(cls, ct_struct, pc_type):
        """Recursively convert a ctypes.Structure to a struct pc_literal."""
        field_names = getattr(pc_type, '_field_names', None) or []
        field_types = getattr(pc_type, '_field_types', None) or []
        inst = cls.__new__(cls)
        inst._pc_type = pc_type
        inst._field_names = list(field_names)
        inst._fields = {}
        for i, name in enumerate(field_names):
            raw = getattr(ct_struct, name, None)
            ft = field_types[i] if i < len(field_types) else None
            if ft is not None and raw is not None:
                inst._fields[name] = cls._from_ctypes_result(raw, ft)
            else:
                inst._fields[name] = raw
        inst._value = inst._fields
        return inst

    # ------------------------------------------------------------------
    # Compile-time lowering (get_value protocol)
    # ------------------------------------------------------------------

    def get_value(self):
        """Return a ``ValueRef`` for use inside ``@compile``.

        Uses ``PythonType`` with ``preferred_pc_type`` so that
        ``TypeConverter`` can lazily produce the correct ``ir.Constant``.
        """
        from .python_type import PythonType
        from ..valueref import wrap_value

        if self._fields is not None:
            # Compound type: wrap the dict/list as python constant
            pt = PythonType.wrap(self._value, is_constant=True,
                                 preferred_pc_type=self._pc_type)
            return wrap_value(self._value, kind='python', type_hint=pt)

        pt = PythonType.wrap(self._value, is_constant=True,
                             preferred_pc_type=self._pc_type)
        return wrap_value(self._value, kind='python', type_hint=pt)

    # ------------------------------------------------------------------
    # ctypes interop
    # ------------------------------------------------------------------

    def _to_ctypes(self, param_type):
        """Convert this pc_literal to a ctypes value for native execution."""
        import ctypes as ct

        if self._fields is not None and issubclass(param_type, ct.Structure):
            vals = []
            for name in (self._field_names or []):
                v = self._fields.get(name)
                if isinstance(v, pc_literal):
                    if _is_integer(v._pc_type) or _is_bool_type(v._pc_type):
                        vals.append(int(v))
                    elif _is_float(v._pc_type):
                        vals.append(float(v))
                    else:
                        vals.append(v._value)
                elif v is not None:
                    vals.append(v)
                else:
                    vals.append(0)
            return param_type(*vals)

        if _is_integer(self._pc_type) or _is_bool_type(self._pc_type):
            return param_type(int(self._value))
        if _is_float(self._pc_type):
            return param_type(float(self._value))
        return param_type(self._value)

    # ------------------------------------------------------------------
    # Python numeric protocol
    # ------------------------------------------------------------------

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)

    def __bool__(self):
        return builtins_bool(self._value)

    def __index__(self):
        return operator.index(self._value)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        name = 'pc_literal'
        if self._pc_type is not None and hasattr(self._pc_type, 'get_name'):
            name = self._pc_type.get_name()
        if self._fields is not None:
            field_str = ', '.join(
                f'{k}={v!r}' for k, v in (self._fields or {}).items()
            )
            return f'{name}({field_str})'
        return f'{name}({self._value!r})'

    def __str__(self):
        return repr(self)

    # ------------------------------------------------------------------
    # Hashing / equality
    # ------------------------------------------------------------------

    def __hash__(self):
        return hash((self._value, id(self._pc_type)))

    def __eq__(self, other):
        if isinstance(other, pc_literal):
            return self._value == other._value
        return self._value == other

    def __ne__(self, other):
        if isinstance(other, pc_literal):
            return self._value != other._value
        return self._value != other

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def __lt__(self, other):
        return self._value < (other._value if isinstance(other, pc_literal) else other)

    def __le__(self, other):
        return self._value <= (other._value if isinstance(other, pc_literal) else other)

    def __gt__(self, other):
        return self._value > (other._value if isinstance(other, pc_literal) else other)

    def __ge__(self, other):
        return self._value >= (other._value if isinstance(other, pc_literal) else other)

    # ------------------------------------------------------------------
    # Struct field access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        fields = object.__getattribute__(self, '_fields')
        if fields is not None and name in fields:
            return fields[name]
        raise AttributeError(
            f"'{self._pc_type.get_name() if self._pc_type else 'pc_literal'}' "
            f"has no field '{name}'"
        )

    def __setattr__(self, name: str, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        fields = object.__getattribute__(self, '_fields')
        if fields is not None and name in fields:
            fields[name] = value
            return
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Arithmetic helpers
    # ------------------------------------------------------------------

    def _binop(self, other, op):
        if isinstance(other, pc_literal):
            result_type = _coerce_types(self._pc_type, other._pc_type)
            raw = op(self._value, other._value)
        elif isinstance(other, (int, float)):
            result_type = self._pc_type
            raw = op(self._value, other)
        else:
            return NotImplemented
        if _is_integer(result_type) and isinstance(raw, (int, float)):
            raw = _mask_int(int(raw), result_type)
        return pc_literal(raw, result_type)

    def _rbinop(self, other, op):
        if isinstance(other, (int, float)):
            result_type = self._pc_type
            raw = op(other, self._value)
        else:
            return NotImplemented
        if _is_integer(result_type) and isinstance(raw, (int, float)):
            raw = _mask_int(int(raw), result_type)
        return pc_literal(raw, result_type)

    def _unaryop(self, op):
        raw = op(self._value)
        if _is_integer(self._pc_type) and isinstance(raw, int):
            raw = _mask_int(raw, self._pc_type)
        return pc_literal(raw, self._pc_type)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other):       return self._binop(other, operator.add)
    def __radd__(self, other):      return self._rbinop(other, operator.add)
    def __sub__(self, other):       return self._binop(other, operator.sub)
    def __rsub__(self, other):      return self._rbinop(other, operator.sub)
    def __mul__(self, other):       return self._binop(other, operator.mul)
    def __rmul__(self, other):      return self._rbinop(other, operator.mul)
    def __truediv__(self, other):   return self._binop(other, operator.truediv)
    def __rtruediv__(self, other):  return self._rbinop(other, operator.truediv)
    def __floordiv__(self, other):  return self._binop(other, operator.floordiv)
    def __rfloordiv__(self, other): return self._rbinop(other, operator.floordiv)
    def __mod__(self, other):       return self._binop(other, operator.mod)
    def __rmod__(self, other):      return self._rbinop(other, operator.mod)
    def __lshift__(self, other):    return self._binop(other, operator.lshift)
    def __rlshift__(self, other):   return self._rbinop(other, operator.lshift)
    def __rshift__(self, other):    return self._binop(other, operator.rshift)
    def __rrshift__(self, other):   return self._rbinop(other, operator.rshift)
    def __and__(self, other):       return self._binop(other, operator.and_)
    def __rand__(self, other):      return self._rbinop(other, operator.and_)
    def __or__(self, other):        return self._binop(other, operator.or_)
    def __ror__(self, other):       return self._rbinop(other, operator.or_)
    def __xor__(self, other):       return self._binop(other, operator.xor)
    def __rxor__(self, other):      return self._rbinop(other, operator.xor)

    def __neg__(self):              return self._unaryop(operator.neg)
    def __invert__(self):           return self._unaryop(operator.invert)
    def __abs__(self):              return self._unaryop(operator.abs)
    def __pos__(self):              return self._unaryop(operator.pos)


# Keep a reference to Python's bool to avoid shadowing by pythoc.bool
import builtins as _builtins
builtins_bool = _builtins.bool

__all__ = ['pc_literal']
