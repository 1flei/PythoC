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


def _lower_field_to_ir_constant(value, field_pc_type):
    """Produce an IR constant for a struct field during get_value().

    Used by ``pc_literal.get_value()`` to lower a ctypes-backed struct
    pc_literal as an aggregate IR constant.  Each field is converted
    according to its declared PC type:

    - integer / bool: ``ir.Constant(i_n, int(value))``
    - float:          ``ir.Constant(fN,  float(value))``
    - pointer:        ``inttoptr`` ConstantExpr on the address int
    - nested struct:  recurse via the nested pc_literal's get_value()
    """
    from llvmlite import ir

    if isinstance(value, pc_literal):
        # Pointer / nested-struct fields have ctypes-backed pc_literals.
        if value._ctypes_owner is not None:
            # Reuse get_value() so pointer/struct lowering stays in one
            # place.  Its result is a ValueRef whose .value is already
            # an IR constant of the correct field type.
            return value.get_value().value
        raw = value._value
    else:
        raw = value

    if field_pc_type is None:
        return raw

    if _is_bool_type(field_pc_type) or _is_integer(field_pc_type):
        llvm_ty = field_pc_type.get_llvm_type()
        return ir.Constant(llvm_ty, int(raw))
    if _is_float(field_pc_type):
        llvm_ty = field_pc_type.get_llvm_type()
        return ir.Constant(llvm_ty, float(raw))
    if getattr(field_pc_type, '_is_pointer', False):
        from .types import i64
        llvm_ty = field_pc_type.get_llvm_type()
        addr = int(raw) if not hasattr(raw, 'value') else (raw.value or 0)
        return ir.Constant(i64.get_llvm_type(), addr).inttoptr(llvm_ty)

    # Fallback: assume already an IR constant compatible with the field.
    return raw


class pc_literal:
    """Tagged Python value carrying a PythoC type.

    Scalars store a plain ``int`` / ``float`` / ``bool`` in ``_value``.
    Compound types (struct, array, enum) store richer payloads; see
    ``_from_composite_call`` and ``_from_ctypes_result``.
    """

    __slots__ = ('_value', '_pc_type', '_fields', '_field_names',
                 '_ctypes_owner')

    def __init__(self, value: Any, pc_type: Any):
        self._value = value
        self._pc_type = pc_type
        self._fields: Optional[dict] = None
        self._field_names: Optional[list] = None
        # Optional ctypes object kept alive to preserve native memory
        # semantics (pointers, by-value structs).  When non-None this
        # instance behaves as a thin tag layer over a live ctypes value;
        # _to_ctypes() will hand back the same object zero-copy.
        self._ctypes_owner = None

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
        inst._ctypes_owner = None
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
            # Preserve the original ctypes handle so that native memory
            # semantics (cast, lifetime, dereferencing) are not lost when
            # the value flows back into another native call.
            inst = cls.__new__(cls)
            inst._pc_type = pc_type
            inst._fields = None
            inst._field_names = None
            inst._ctypes_owner = ctypes_result
            if hasattr(ctypes_result, 'value'):
                inst._value = ctypes_result.value or 0
            elif isinstance(ctypes_result, int):
                inst._value = ctypes_result
            else:
                inst._value = ct.cast(ctypes_result, ct.c_void_p).value or 0
            return inst

        if isinstance(ctypes_result, ct.Structure):
            return cls._struct_from_ctypes(ctypes_result, pc_type)

        return ctypes_result

    @classmethod
    def _struct_from_ctypes(cls, ct_struct, pc_type):
        """Wrap a ctypes.Structure as a struct pc_literal.

        Field access is lazy: the original ``ct_struct`` is rooted via
        ``_ctypes_owner`` and field reads go through ``__getattr__`` /
        ``_materialize_fields`` which re-read directly from the live
        Structure.  This keeps native memory semantics intact (no
        premature copy of pointer fields, no detached snapshots whose
        backing buffer may be freed by ctypes shortly after).
        """
        inst = cls.__new__(cls)
        inst._pc_type = pc_type
        inst._field_names = list(getattr(pc_type, '_field_names', None) or [])
        # Sentinel dict signalling "compound, fields readable lazily".
        # Real values are fetched on demand from _ctypes_owner.
        inst._fields = {}
        inst._ctypes_owner = ct_struct
        inst._value = ct_struct
        return inst

    def _materialize_field(self, name):
        """Read a single field from the live ctypes Structure on demand."""
        owner = object.__getattribute__(self, '_ctypes_owner')
        if owner is None:
            raise AttributeError(name)
        raw = getattr(owner, name)
        field_types = getattr(self._pc_type, '_field_types', None) or []
        field_names = self._field_names or []
        ft = None
        if name in field_names:
            i = field_names.index(name)
            if i < len(field_types):
                ft = field_types[i]
        if ft is None:
            return raw
        return pc_literal._from_ctypes_result(raw, ft)

    @staticmethod
    def _reject_pointer_fields(pc_type, _path=()):
        """Raise via logger.error if any (transitive) field is a pointer.

        Capture into IR is restricted to value-typed compounds:
        scalars, struct/union/enum aggregates of value types, and
        nested aggregates of those.  Any pointer field would force us
        to bake a process-local heap address into a cacheable artefact,
        which is precisely the contract pythoc refuses to honour.
        """
        if pc_type is None:
            return
        if getattr(pc_type, '_is_pointer', False):
            from ..logger import logger
            owner_name = '.'.join(_path) if _path else '<value>'
            type_name = (pc_type.get_name()
                         if hasattr(pc_type, 'get_name') else str(pc_type))
            logger.error(
                f"Cannot capture a struct with a pointer field "
                f"('{owner_name}: {type_name}') into a @compile function. "
                f"Pass the value as an explicit argument instead.",
                node=None, exc_type=TypeError,
            )
        field_names = getattr(pc_type, '_field_names', None) or []
        field_types = getattr(pc_type, '_field_types', None) or []
        for i, ft in enumerate(field_types):
            fname = field_names[i] if i < len(field_names) else f'_{i}'
            pc_literal._reject_pointer_fields(ft, _path + (fname,))

    # ------------------------------------------------------------------
    # Compile-time lowering (get_value protocol)
    # ------------------------------------------------------------------

    def get_value(self):
        """Return a ``ValueRef`` for use inside ``@compile``.

        Capture into IR is supported for **value-typed** payloads:

        1. Plain scalar pc_literal (int / float / bool): lowered to an
           ``ir.Constant`` of the preferred PC type.
        2. ctypes-backed by-value struct / union / enum pc_literal,
           where every (transitive) field is itself a value type:
           returns the pc_literal itself as the ``PythonType`` payload
           so that ``PythonType.handle_attribute`` delegates back to
           ``pc_literal.handle_attribute``, which lowers each field at
           IR build time.

        Capture is **rejected** when the payload contains a runtime
        pointer (whether the pc_literal itself is pointer-shaped or a
        struct field is).  Embedding a runtime heap address into IR
        would couple the cached compile artefact to one specific
        process and provides no semantic that cannot be expressed by
        passing the pointer as an explicit argument.  The argument
        path is fully supported via ``_to_ctypes`` (see
        ``_from_ctypes_result``).

        Escape hatch: if the user really wants a runtime address
        baked into IR (e.g. for ad-hoc inspection of a value known to
        outlive the build cache), they may explicitly cast the
        pc_literal to an integer first -- ``u64(int(p))`` produces a
        plain value-typed pc_literal that captures fine.  At that
        point pythoc no longer claims any guarantees about the
        captured address: it is just an integer, and dereferencing it
        is at the caller's risk (precisely the same risk profile as
        capturing any other Python ``int`` that happens to be a heap
        address).
        """
        from .python_type import PythonType
        from ..valueref import wrap_value

        owner = object.__getattribute__(self, '_ctypes_owner')
        if owner is not None:
            field_names = object.__getattribute__(self, '_field_names') or []
            if field_names:
                # Live struct owner.  Capture is supported only when
                # every field is a value type that can be safely pinned
                # into IR.  A pointer field would mean baking a
                # process-local heap address into a cacheable artefact
                # -- a contract pythoc deliberately refuses (capture a
                # pointer back into Python-side state instead, then
                # pass it as an explicit argument).
                self._reject_pointer_fields(self._pc_type)
                pt = PythonType.wrap(self, is_constant=True,
                                     preferred_pc_type=self._pc_type)
                return wrap_value(self, kind='python', type_hint=pt)

            # Pointer-shaped owner: capturing a runtime pointer value
            # into a compiled function is intentionally unsupported.
            # The address is a process-local heap handle; pinning it
            # into IR couples the cached artefact to one specific
            # process.  Pass the pointer as an explicit argument
            # instead -- that path is fully supported via _to_ctypes.
            from ..logger import logger
            type_name = (self._pc_type.get_name()
                         if hasattr(self._pc_type, 'get_name')
                         else str(self._pc_type))
            logger.error(
                f"Cannot capture a runtime pointer ({type_name}) into "
                f"a @compile function. Pass it as an explicit argument, "
                f"or cast to an integer first (e.g. u64(int(p))) if you "
                f"really need to bake the current address into IR -- "
                f"address validity then becomes the caller's "
                f"responsibility.",
                node=None, exc_type=TypeError,
            )

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

        # Owner-backed path: we already hold a live ctypes object created
        # by a previous native call.  Preserve native memory semantics
        # instead of round-tripping through Python field copies.
        owner = self._ctypes_owner
        if owner is not None:
            try:
                if isinstance(owner, param_type):
                    return owner
            except TypeError:
                pass
            # Pointer-shaped target: hand back the integer address while
            # the owner reference held on self keeps the buffer alive
            # for the duration of the native call.
            if param_type is ct.c_void_p:
                if isinstance(owner, int):
                    return owner
                if hasattr(owner, 'value'):
                    return owner.value
                return ct.cast(owner, ct.c_void_p).value
            # Same-shape structure produced by a different ctypes class
            # (e.g. two compile groups generated independent Structure
            # subclasses for the same PC type).  Reinterpret in place.
            if (isinstance(owner, ct.Structure)
                    and isinstance(param_type, type)
                    and issubclass(param_type, ct.Structure)):
                src = type(owner)
                if (getattr(src, '_fields_', None)
                        == getattr(param_type, '_fields_', None)
                        and ct.sizeof(src) == ct.sizeof(param_type)):
                    return ct.cast(ct.pointer(owner),
                                   ct.POINTER(param_type))[0]
            from ..logger import logger
            logger.error(
                f"Cannot pass ctypes-backed pc_literal "
                f"({type(owner).__name__}) as {param_type!r}",
                node=None, exc_type=TypeError,
            )

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
        owner = object.__getattribute__(self, '_ctypes_owner')
        field_names = object.__getattribute__(self, '_field_names') or []
        if owner is not None and field_names:
            field_str = ', '.join(
                f'{k}={self._materialize_field(k)!r}' for k in field_names
            )
            return f'{name}({field_str})'
        if self._fields is not None:
            field_str = ', '.join(
                f'{k}={v!r}' for k, v in (self._fields or {}).items()
            )
            return f'{name}({field_str})'
        return f'{name}({self._value!r})'

    def __str__(self):
        return repr(self)

    # ------------------------------------------------------------------
    # IR-aware lowering for ctypes-backed compound pc_literal
    # ------------------------------------------------------------------

    def handle_attribute(self, visitor, base, attr_name, node):
        """Lower a struct-shaped, ctypes-backed pc_literal field access
        directly to an IR constant ``extractvalue``.

        This hook is dispatched by ``PythonType.handle_attribute`` when
        the wrapped ``_python_object`` defines its own protocol.  It is
        only meaningful for ctypes-backed compound pc_literal: at this
        point we have the ``visitor`` (and therefore ``module.context``)
        and can emit the proper aggregate constant + field projection
        without needing a builder.
        """
        from llvmlite import ir
        from ..valueref import wrap_value
        from ..logger import logger

        owner = object.__getattribute__(self, '_ctypes_owner')
        field_names = object.__getattribute__(self, '_field_names') or []
        if owner is None or not field_names:
            logger.error(
                f"pc_literal of type "
                f"'{self._pc_type.get_name() if self._pc_type else '?'}'"
                f" has no attribute '{attr_name}'",
                node=node, exc_type=AttributeError,
            )

        if attr_name not in field_names:
            logger.error(
                f"struct '{self._pc_type.get_name()}' has no field "
                f"named '{attr_name}'",
                node=node, exc_type=AttributeError,
            )

        idx = field_names.index(attr_name)
        field_types = getattr(self._pc_type, '_field_types', None) or []
        field_pc_type = field_types[idx] if idx < len(field_types) else None

        # Materialise the live field value through the IR-constant path.
        fval = self._materialize_field(attr_name)
        ir_const = _lower_field_to_ir_constant(fval, field_pc_type)
        return wrap_value(ir_const, kind='value', type_hint=field_pc_type)

    # ------------------------------------------------------------------
    # Hashing / equality
    # ------------------------------------------------------------------

    def __hash__(self):
        owner = object.__getattribute__(self, '_ctypes_owner')
        if owner is not None:
            # ctypes Structure / pointer instances are not hashable; use
            # object identity of the owner as a stable surrogate.
            return hash((id(owner), id(self._pc_type)))
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
        if fields is not None:
            if name in fields:
                return fields[name]
            owner = object.__getattribute__(self, '_ctypes_owner')
            field_names = object.__getattribute__(self, '_field_names') or []
            if owner is not None and name in field_names:
                return self._materialize_field(name)
        raise AttributeError(
            f"'{self._pc_type.get_name() if self._pc_type else 'pc_literal'}' "
            f"has no field '{name}'"
        )

    def __setattr__(self, name: str, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        fields = object.__getattribute__(self, '_fields')
        owner = object.__getattribute__(self, '_ctypes_owner') \
            if hasattr(type(self), '_ctypes_owner') else None
        if owner is not None:
            field_names = object.__getattribute__(self, '_field_names') or []
            if name in field_names:
                # Write through to the live ctypes Structure so that
                # subsequent native calls observe the update.
                if isinstance(value, pc_literal):
                    if value._ctypes_owner is not None:
                        setattr(owner, name, value._ctypes_owner)
                    else:
                        setattr(owner, name, value._value)
                else:
                    setattr(owner, name, value)
                return
        if fields is not None and name in fields:
            fields[name] = value
            return
        object.__setattr__(self, name, value)

    def __iter__(self):
        """Allow *struct_val unpacking by iterating over field values."""
        fields = object.__getattribute__(self, '_fields')
        owner = object.__getattribute__(self, '_ctypes_owner')
        field_names = object.__getattribute__(self, '_field_names') or []
        if owner is not None and field_names:
            for name in field_names:
                yield self._materialize_field(name)
            return
        if fields is not None:
            for name in field_names:
                yield fields[name]
            return
        raise TypeError(
            f"'{self._pc_type.get_name() if self._pc_type else 'pc_literal'}'"
            f" is not iterable (scalar value)"
        )

    def __len__(self):
        """Return number of fields for struct pc_literal."""
        fields = object.__getattribute__(self, '_fields')
        owner = object.__getattribute__(self, '_ctypes_owner')
        if owner is not None or fields is not None:
            return len(object.__getattribute__(self, '_field_names') or [])
        raise TypeError(
            f"'{self._pc_type.get_name() if self._pc_type else 'pc_literal'}'"
            f" has no len() (scalar value)"
        )

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
