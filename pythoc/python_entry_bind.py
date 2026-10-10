"""Python entry for one known @compile function.

The entry template is the vectorcall function. Holes are the known
parameter types and the direct kernel call. Conversion lives in
python_entry.
"""

import ast
import os

from pythoc import array, f64, i8, i32, i64, linear, meta, ptr, u64, void
from pythoc.builtin_entities.func import func as func_type
from pythoc.builtin_entities.offsetof import offsetof
from pythoc.cpy_api import (
    PyBytes_AsString,
    PyBytes_FromStringAndSize,
    PyDict_New,
    PyFloat_FromDouble,
    PyLong_FromLongLong,
    PyLong_FromUnsignedLongLong,
    PyTuple_GetItem,
    PyTuple_New,
    PyTuple_SetItem,
    PyUnicode_CompareWithASCIIString,
    PyUnicode_FromString,
    _PyLong_FromByteArray,
    memcpy,
)
from pythoc.python_entry import (
    PyErr_Occurred,
    bind_args,
    bool_obj,
    box_bool,
    box_f64,
    box_fields,
    box_i64,
    box_u64,
    box_value,
    box_wide,
    errno_store,
    export_item,
    field_item,
    fill_bytes,
    fill_wide,
    none_ref,
    pythoc_install_runtime,
    pythoc_make_module,
    release,
    take_bool,
    take_f64,
    take_ptr,
    take_signed,
    take_u64,
    type_error,
    value_error,
    zero_bytes,
)

_INTS = {
    'i8': (True, 8),
    'i16': (True, 16),
    'i32': (True, 32),
    'i64': (True, 64),
    'i128': (True, 128),
    'u8': (False, 8),
    'u16': (False, 16),
    'u32': (False, 32),
    'u64': (False, 64),
    'u128': (False, 128),
}

_SCALAR = ('int', 'u64', 'f32', 'f64', 'bool', 'ptr', 'wide', 'blob')


def _bounds(signed, bits):
    if not signed:
        return 0, (1 << bits) - 1
    if bits >= 64:
        return -9223372036854775808, 9223372036854775807
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _form(pc_type):
    from .python_adapter import (
        _peel_type,
        _resolve_named_type,
        _size_of,
        _type_flag,
    )

    original = pc_type
    pc_type = _peel_type(pc_type)
    if pc_type is None or not hasattr(pc_type, 'get_name'):
        raise RuntimeError('Python entry type is not lowerable')
    name = pc_type.get_name()
    from .schema_protocol import get_linear_schema_paths
    base = {
        'type': original,
        'peeled': pc_type,
        'name': name,
        'holds_linear': bool(get_linear_schema_paths(pc_type)),
    }
    if name == 'void' or getattr(pc_type, '_is_linear', False):
        base['kind'] = 'skip'
        base['linear'] = bool(getattr(pc_type, '_is_linear', False))
        return base
    if getattr(pc_type, '_is_bool', False):
        base['kind'] = 'bool'
        return base
    if getattr(pc_type, '_is_float', False):
        if name not in ('f32', 'f64'):
            raise RuntimeError('unsupported float {}'.format(name))
        base['kind'] = name
        return base
    if getattr(pc_type, '_is_integer', False):
        if name not in _INTS:
            raise RuntimeError('unsupported integer {}'.format(name))
        signed, bits = _INTS[name]
        if bits >= 128 or (not signed and bits >= 64):
            base['kind'] = 'wide' if bits >= 128 else 'u64'
            base['signed'] = signed
            return base
        lo, hi = _bounds(signed, bits)
        base.update(kind='int', signed=signed, bits=bits, lo=lo, hi=hi)
        return base
    if isinstance(pc_type, type) and issubclass(pc_type, func_type):
        base['kind'] = 'ptr'
        return base
    if getattr(pc_type, '_is_pointer', False):
        base['kind'] = 'ptr'
        return base
    size = _size_of(pc_type) or 0
    if size == 0:
        base['kind'] = 'skip'
        base['linear'] = False
        return base
    base['size'] = int(size)
    if _type_flag(pc_type, 'is_enum_type') or getattr(pc_type, '_is_enum', False):
        tag = _form(pc_type._field_types[0])
        payload = pc_type._field_types[1]
        payload_size = _size_of(payload) or 0
        offset = 0
        if payload_size:
            offset = int(offsetof._get_field_offset(pc_type, 'payload'))
        base.update(
            kind='enum',
            tag=tag,
            payload_size=int(payload_size),
            payload_offset=offset,
        )
        return base
    if _type_flag(pc_type, 'is_array') or name == 'array':
        dims = pc_type.dimensions
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(int(dim) for dim in dims)
        elem = _form(_resolve_named_type(pc_type.element_type))
        base.update(kind='array', dims=dims, elem=elem)
        return base
    if getattr(pc_type, '_is_union', False) or name == 'union':
        base['kind'] = 'blob'
        return base
    if _type_flag(pc_type, 'is_struct_type') or getattr(pc_type, '_is_struct', False):
        fields = []
        types = getattr(pc_type, '_field_types', None) or []
        names = getattr(pc_type, '_field_names', None) or []
        for index, field_type in enumerate(types):
            field_type = _resolve_named_type(field_type)
            field = _form(field_type)
            raw = names[index] if index < len(names) else None
            if not isinstance(raw, str) or not raw:
                raw = None
            field['field'] = raw
            field['index'] = index
            field['offset'] = _field_offset(pc_type, index)
            fields.append(field)
        base.update(kind='struct', fields=fields)
        return base
    raise RuntimeError('unsupported Python entry type {}'.format(name))


def _field_offset(pc_type, index):
    _names, types = offsetof._get_struct_fields(pc_type)
    offset = 0
    for i, field_type in enumerate(types):
        align = offsetof._get_type_alignment(field_type)
        offset = offsetof._align_to(offset, align)
        if i == index:
            return int(offset)
        offset += offsetof._get_type_size(field_type)
    raise RuntimeError('struct field index is out of range')


def _named_fields(form):
    if form['kind'] != 'struct':
        return None
    names = []
    for field in form['fields']:
        if field['kind'] == 'skip':
            continue
        if not field.get('field'):
            return None
        names.append(field['field'])
    return names or None


def _nbytes(form):
    if form.get('size'):
        return int(form['size'])
    from .python_adapter import _size_of
    return int(_size_of(form['peeled']) or 0)


def _flat_offset(indices, dims, elem_size):
    flat = 0
    stride = 1
    for index, dim in zip(reversed(indices), reversed(dims)):
        flat += index * stride
        stride *= dim
    return flat * elem_size


def _nd_indices(dims):
    if not dims:
        yield ()
        return
    total = 1
    for dim in dims:
        total *= dim
    for flat in range(total):
        rest = []
        value = flat
        for dim in reversed(dims):
            rest.append(value % dim)
            value //= dim
        yield tuple(reversed(rest))


def _kind_flags(form):
    kind = form['kind']
    if kind in ('f32', 'f64'):
        kind = 'float'
    names = ('int', 'u64', 'float', 'bool', 'ptr', 'wide', 'blob')
    flags = {'as_{}'.format(name): meta.const(kind == name) for name in names}
    flags['lo'] = int(form.get('lo', 0))
    flags['hi'] = int(form.get('hi', 0))
    flags['signed_flag'] = 1 if form.get('signed') else 0
    flags['size'] = _nbytes(form) if kind in ('blob', 'wide') else int(form.get('size', 0))
    return flags


# The vectorcall entry. bind_names and tail are the only statement holes.
@meta.quote
def _entry(slots_n, count, bind_names, tail):
    slots: array[ptr[void], slots_n]
    names: array[ptr[i8], slots_n]
    bind_names
    if bind_args(
        args,
        nargsf,
        kwnames,
        ptr[ptr[i8]](ptr(names)),
        i64(count),
        ptr[ptr[void]](ptr(slots)),
    ) != i32(0):
        return ptr[void](0)
    tail


@meta.quote
def _name_at(index, text):
    names[index] = text


# Python values are loaded first. Linear tokens are created only on the
# path that reaches the kernel call.
@meta.quote
def _tail(loads, linears, returned):
    loads
    if i64(PyErr_Occurred()) != i64(0):
        return ptr[void](0)
    linears
    returned


@meta.quote
def _returned(ret, call, box):
    errno_store(i32(0))
    result: ret = call
    box


@meta.quote
def _void_returned(call):
    errno_store(i32(0))
    call
    return none_ref()


@meta.quote
def _linear(dst, ty):
    dst: ty = linear()


@meta.quote
def _decl(dst, ty):
    dst: ty


@meta.quote
def _load_required(index, src, loaded):
    if i64(slots[index]) == i64(0):
        type_error("missing required argument")
        return ptr[void](0)
    src: ptr[void] = slots[index]
    loaded


@meta.quote
def _load_optional(dst, ty, index, size, src, parts):
    dst: ty
    zero_bytes(ptr[i8](ptr(dst)), i64(size))
    if i64(slots[index]) != i64(0):
        src: ptr[void] = slots[index]
        parts


# One scalar template. Constant flags fold away the other branches.
@meta.quote
def _from_python(
    dst, ty, src,
    as_int, as_u64, as_float, as_bool, as_ptr, as_wide, as_blob,
    lo, hi, signed_flag, size,
):
    if as_int:
        dst: ty = ty(take_signed(src, i64(lo), i64(hi)))
    if as_u64:
        dst: ty = take_u64(src)
    if as_float:
        dst: ty = ty(take_f64(src))
    if as_bool:
        dst: ty = ty(take_bool(src))
    if as_ptr:
        dst: ty = ty(take_ptr(src))
    if as_wide:
        dst: ty
        zero_bytes(ptr[i8](ptr(dst)), i64(16))
        if fill_wide(src, ptr[i8](ptr(dst)), i32(signed_flag)) != i32(0):
            return ptr[void](0)
    if as_blob:
        dst: ty
        if fill_bytes(src, ptr[i8](ptr(dst)), i64(size)) != i32(0):
            return ptr[void](0)


@meta.quote
def _from_python_default(
    dst, ty, index,
    as_int, as_u64, as_float, as_bool, as_ptr,
    lo, hi, default,
):
    dst: ty
    if i64(slots[index]) == i64(0):
        if as_int:
            dst = ty(default)
        if as_u64:
            dst = ty(default)
        if as_float:
            dst = ty(default)
        if as_bool:
            dst = ty(default)
        if as_ptr:
            dst = ty(default)
    else:
        if as_int:
            dst = ty(take_signed(slots[index], i64(lo), i64(hi)))
        if as_u64:
            dst = take_u64(slots[index])
        if as_float:
            dst = ty(take_f64(slots[index]))
        if as_bool:
            dst = ty(take_bool(slots[index]))
        if as_ptr:
            dst = ty(take_ptr(slots[index]))


@meta.quote
def _aggregate(dst, ty, size, parts):
    dst: ty
    zero_bytes(ptr[i8](ptr(dst)), i64(size))
    parts


@meta.quote
def _one_field(item, src, index, label, loaded, store):
    item: ptr[void] = field_item(src, i64(index), label)
    loaded
    store
    release(item)


@meta.quote
def _assign_index(obj, index, src):
    obj[index] = src


@meta.quote
def _copy_into(dst, offset, src, size):
    memcpy(
        ptr[i8](ptr(dst)) + i64(offset),
        ptr[i8](ptr(src)),
        u64(size),
    )


@meta.quote
def _step(dst, src, index):
    dst: ptr[void] = field_item(src, i64(index), "")


@meta.quote
def _one_elem(walk, loaded, store, cleanup):
    walk
    loaded
    store
    cleanup


@meta.quote
def _release_one(obj):
    release(obj)


@meta.quote
def _payload(item, dst, src, offset, size):
    item: ptr[void] = field_item(src, i64(1), "payload")
    if i64(item) != i64(0):
        if fill_bytes(item, ptr[i8](ptr(dst)) + i64(offset), i64(size)) != i32(0):
            release(item)
            return ptr[void](0)
        release(item)


@meta.quote
def _read_field(dst, ty, obj, index):
    dst: ty = obj[index]


@meta.quote
def _read_bytes(dst, ty, obj, offset, size):
    dst: ty
    memcpy(
        ptr[i8](ptr(dst)),
        ptr[i8](ptr(obj)) + i64(offset),
        u64(size),
    )


@meta.quote
def _plain_scalar(
    dst, src,
    as_int, as_u64, as_float, as_bool, as_ptr, as_wide, as_blob,
    signed_flag, size,
):
    if as_int:
        dst = PyLong_FromLongLong(i64(src))
    if as_u64:
        dst = PyLong_FromUnsignedLongLong(src)
    if as_float:
        dst = PyFloat_FromDouble(f64(src))
    if as_bool:
        dst = bool_obj(i32(src))
    if as_ptr:
        dst = PyLong_FromUnsignedLongLong(u64(src))
    if as_wide:
        dst = _PyLong_FromByteArray(
            ptr[i8](ptr(src)), u64(16), i32(1), i32(signed_flag),
        )
    if as_blob:
        dst = PyBytes_FromStringAndSize(ptr[i8](ptr(src)), i64(size))


@meta.quote
def _box_result(
    type_name,
    as_int, as_u64, as_float, as_bool, as_ptr, as_wide, as_blob,
    signed_flag, size,
):
    if as_int:
        return box_i64(i64(result), type_name)
    if as_u64:
        return box_u64(result, type_name)
    if as_float:
        return box_f64(f64(result), type_name)
    if as_bool:
        return box_bool(i32(result), type_name)
    if as_ptr:
        return box_u64(u64(result), type_name)
    if as_wide:
        return box_wide(ptr[i8](ptr(result)), i32(signed_flag), type_name)
    if as_blob:
        return box_value(
            type_name,
            PyBytes_FromStringAndSize(ptr[i8](ptr(result)), i64(size)),
        )


@meta.quote
def _box_enum(tag, tag_ty, type_name):
    tag: tag_ty = result[0]
    return box_i64(i64(tag), type_name)


# A linear enum is one resource. Taking its address consumes it and copies
# the payload. The Python object is a pc_literal whose value is
# (tag, payload bytes), so a Python sequence match does not unpack it.
@meta.quote
def _box_linear_enum(tag, tag_ty, type_name, payload_off, payload_size):
    tag: tag_ty = result[0]
    packed: ptr[void] = PyTuple_New(i64(2))
    PyTuple_SetItem(packed, i64(0), PyLong_FromLongLong(i64(tag)))
    PyTuple_SetItem(
        packed,
        i64(1),
        PyBytes_FromStringAndSize(
            ptr[i8](ptr(result)) + i64(payload_off),
            i64(payload_size),
        ),
    )
    return box_value(type_name, packed)


@meta.quote
def _rebuild_enum(dst, ty, tag, src, lo, hi, buf, arms):
    dst: ty
    item: ptr[void] = field_item(src, i64(0), "tag")
    tag: i64 = take_signed(item, i64(lo), i64(hi))
    release(item)
    if i64(PyErr_Occurred()) != i64(0):
        return ptr[void](0)
    buf: ptr[void] = field_item(src, i64(1), "payload")
    arms
    release(buf)


@meta.quote
def _if_i64(tag, expected, yes, no):
    if tag == i64(expected):
        yes
    else:
        no


@meta.quote
def _bad_tag(buf):
    release(buf)
    type_error("unknown enum tag")
    return ptr[void](0)


@meta.quote
def _from_bytes(dst, ty, buf, offset, size):
    dst: ty
    if i64(buf) == i64(0):
        return ptr[void](0)
    data: ptr[i8] = PyBytes_AsString(buf)
    if i64(data) == i64(0):
        release(buf)
        return ptr[void](0)
    memcpy(ptr[i8](ptr(dst)), data + i64(offset), u64(size))


@meta.quote
def _set_call(dst, call):
    dst = call


@meta.quote
def _stmt(stmt):
    stmt


@meta.quote
def _new_tuple(dst, n):
    dst = PyTuple_New(i64(n))


@meta.quote
def _new_dict(dst):
    dst = PyDict_New()


@meta.quote
def _null_ptr(dst):
    dst: ptr[void] = ptr[void](0)


@meta.quote
def _put(tup, index, obj):
    PyTuple_SetItem(tup, i64(index), obj)


@meta.quote
def _export(tup, index, dct, key, obj):
    export_item(tup, i64(index), dct, key, obj)


@meta.quote
def _name_item(tup, index, text):
    PyTuple_SetItem(tup, i64(index), PyUnicode_FromString(text))


@meta.quote
def _return_boxed(type_name, value, fields, names):
    boxed: ptr[void] = box_value(type_name, value)
    if i64(fields) != i64(0):
        box_fields(boxed, fields, names)
    return boxed


@meta.quote
def _adapter_address(hits):
    if i64(arg) == i64(0):
        type_error("adapter_address expected a string")
        return ptr[void](0)
    hits
    value_error("unknown PythoC export")
    return ptr[void](0)


@meta.quote
def _address_hit(export_id, fn):
    if PyUnicode_CompareWithASCIIString(arg, export_id) == i32(0):
        return PyLong_FromUnsignedLongLong(u64(fn))


@meta.quote
def _install_entry():
    if i64(arg) == i64(0):
        type_error("install_runtime expected a tuple")
        return ptr[void](0)
    lit: ptr[void] = PyTuple_GetItem(arg, i64(0))
    call: ptr[void] = PyTuple_GetItem(arg, i64(1))
    types: ptr[void] = PyTuple_GetItem(arg, i64(2))
    pythoc_install_runtime(lit, call, types)
    return none_ref()


@meta.quote
def _init(module_name):
    return pythoc_make_module(
        module_name,
        ptr[void](pythoc_adapter_address),
        ptr[void](pythoc_install_entry),
    )


class _Bind:
    def __init__(self):
        self.count = 0
        self.aliases = {}
        self.globals = {}
        self.captured = {}

    def fresh(self, prefix):
        name = '{}{}'.format(prefix, self.count)
        self.count += 1
        return name

    def alias(self, pc_type):
        key = id(pc_type)
        if key not in self.aliases:
            name = 'T{}'.format(len(self.aliases))
            self.aliases[key] = name
            self.globals[name] = pc_type
        return self.aliases[key]

    def take(self, frag):
        self.captured.update(getattr(frag, '_user_globals', {}))
        return frag


def _fill(bind, dst, src, form):
    kind = form['kind']
    if kind in _SCALAR:
        return bind.take(_from_python(
            dst, bind.alias(form['type']), src, **_kind_flags(form),
        ))
    return bind.take(_aggregate(
        dst, bind.alias(form['type']), form['size'], _parts(bind, dst, src, form),
    ))


def _parts(bind, dst, src, form):
    kind = form['kind']
    if kind == 'struct':
        parts = []
        for field in form['fields']:
            if field['kind'] == 'skip':
                continue
            parts.append(_field_part(bind, dst, src, field))
        return parts
    if kind == 'array':
        return [
            _elem_part(bind, dst, src, form, indices, offset)
            for indices, offset in _array_slots(form)
        ]
    if kind == 'enum':
        parts = [_field_part(bind, dst, src, _tag_field(form))]
        if form['payload_size']:
            parts.append(bind.take(_payload(
                bind.fresh('p'),
                dst,
                src,
                form['payload_offset'],
                form['payload_size'],
            )))
        return parts
    raise RuntimeError('cannot unbox {}'.format(kind))


def _tag_field(form):
    tag = dict(form['tag'])
    tag['field'] = 'tag'
    tag['index'] = 0
    tag['offset'] = 0
    return tag


def _field_part(bind, dst, src, field):
    item = bind.fresh('p')
    native = bind.fresh('a')
    label = field['field'] or 'f{}'.format(field['index'])
    return bind.take(_one_field(
        item,
        src,
        field['index'],
        meta.const(label),
        _fill(bind, native, item, field),
        _store_field(bind, dst, field, native),
    ))


def _store_field(bind, dst, field, src):
    if field['kind'] == 'array':
        return bind.take(_copy_into(dst, field['offset'], src, field['size']))
    return bind.take(_assign_index(dst, field['index'], src))


def _array_slots(form):
    elem_size = _nbytes(form['elem'])
    for indices in _nd_indices(form['dims']):
        yield indices, _flat_offset(indices, form['dims'], elem_size)


def _elem_part(bind, dst, src, form, indices, offset):
    current = src
    walk = []
    owned = []
    for depth, index in enumerate(indices):
        nxt = bind.fresh('p')
        walk.append(bind.take(_step(nxt, current, index)))
        if depth:
            owned.append(current)
        current = nxt
    native = bind.fresh('a')
    cleanup = [
        bind.take(_release_one(name))
        for name in owned + [current]
    ]
    return bind.take(_one_elem(
        walk,
        _fill(bind, native, current, form['elem']),
        bind.take(_copy_into(dst, offset, native, _nbytes(form['elem']))),
        cleanup,
    ))


def _load_param(bind, dst, index, form, default):
    if default is None:
        src = bind.fresh('p')
        return _load_required(index, src, _fill(bind, dst, src, form))
    kind = form['kind']
    if default[0] == 'none' and kind in ('int', 'u64', 'f32', 'f64', 'bool', 'ptr'):
        return _scalar_default(bind, dst, index, form, _default_number(default, kind))
    if default[0] == 'none':
        src = bind.fresh('p')
        return _load_optional(
            dst, bind.alias(form['type']), index, form.get('size', 0),
            src, _parts(bind, dst, src, form),
        )
    if kind in ('int', 'u64', 'f32', 'f64', 'bool', 'ptr'):
        return _scalar_default(bind, dst, index, form, _default_number(default, kind))
    raise RuntimeError('default is not valid for {}'.format(kind))


def _default_number(default, kind):
    value = default[1]
    if kind in ('f32', 'f64'):
        return float(value)
    if kind == 'bool':
        return 1 if value else 0
    return int(value)


def _scalar_default(bind, dst, index, form, default):
    flags = _kind_flags(form)
    return _from_python_default(
        dst,
        bind.alias(form['type']),
        index,
        flags['as_int'],
        flags['as_u64'],
        flags['as_float'],
        flags['as_bool'],
        flags['as_ptr'],
        flags['lo'],
        flags['hi'],
        default,
    )


def _read_native(bind, dst, field, src):
    if field['kind'] == 'array':
        return bind.take(_read_bytes(
            dst, bind.alias(field['type']), src, field['offset'], field['size'],
        ))
    return bind.take(_read_field(
        dst, bind.alias(field['type']), src, field['index'],
    ))


def _plain(bind, src, form):
    kind = form['kind']
    if kind in _SCALAR:
        dst = bind.fresh('o')
        return dst, [bind.take(_plain_scalar(dst, src, **_plain_flags(form)))]
    if kind == 'struct':
        return _plain_struct(bind, src, form)
    if kind == 'array':
        return _plain_array(bind, src, form)
    if kind == 'enum':
        native = bind.fresh('a')
        head = bind.take(_read_field(
            native, bind.alias(form['tag']['type']), src, 0,
        ))
        obj, more = _plain(bind, native, form['tag'])
        return obj, [head] + more
    raise RuntimeError('cannot box {}'.format(kind))


def _plain_flags(form):
    flags = _kind_flags(form)
    return {
        'as_int': flags['as_int'],
        'as_u64': flags['as_u64'],
        'as_float': flags['as_float'],
        'as_bool': flags['as_bool'],
        'as_ptr': flags['as_ptr'],
        'as_wide': flags['as_wide'],
        'as_blob': flags['as_blob'],
        'signed_flag': flags['signed_flag'],
        'size': flags['size'],
    }


def _plain_struct(bind, src, form):
    visible = [field for field in form['fields'] if field['kind'] != 'skip']
    tup = bind.fresh('t')
    parts = [bind.take(_new_tuple(tup, len(visible)))]
    for slot, field in enumerate(visible):
        native = bind.fresh('a')
        parts.append(_read_native(bind, native, field, src))
        obj, more = _plain(bind, native, field)
        parts.extend(more)
        parts.append(bind.take(_put(tup, slot, obj)))
    return tup, parts


def _plain_array(bind, src, form):
    return _plain_dims(bind, src, form, form['dims'], ())


def _plain_dims(bind, src, form, dims, prefix):
    tup = bind.fresh('t')
    parts = [bind.take(_new_tuple(tup, dims[0]))]
    if len(dims) == 1:
        elem_size = _nbytes(form['elem'])
        for index in range(dims[0]):
            native = bind.fresh('a')
            offset = _flat_offset(prefix + (index,), form['dims'], elem_size)
            parts.append(bind.take(_read_bytes(
                native, bind.alias(form['elem']['type']), src, offset, elem_size,
            )))
            obj, more = _plain(bind, native, form['elem'])
            parts.extend(more)
            parts.append(bind.take(_put(tup, index, obj)))
        return tup, parts
    for index in range(dims[0]):
        sub, more = _plain_dims(bind, src, form, dims[1:], prefix + (index,))
        parts.extend(more)
        parts.append(bind.take(_put(tup, index, sub)))
    return tup, parts


def _box_return(bind, form):
    if form['kind'] == 'void':
        return []
    type_name = meta.const(form['peeled'].get_name())
    kind = form['kind']
    if kind in _SCALAR:
        flags = _kind_flags(form)
        return bind.take(_box_result(
            type_name,
            flags['as_int'], flags['as_u64'], flags['as_float'],
            flags['as_bool'], flags['as_ptr'], flags['as_wide'],
            flags['as_blob'], flags['signed_flag'], flags['size'],
        ))
    if kind == 'enum':
        tag = bind.fresh('a')
        if form.get('holds_linear'):
            return bind.take(_box_linear_enum(
                tag,
                bind.alias(form['tag']['type']),
                type_name,
                int(form['payload_offset']),
                int(form['payload_size']),
            ))
        return bind.take(_box_enum(
            tag, bind.alias(form['tag']['type']), type_name,
        ))
    if kind == 'struct':
        return _box_struct(bind, form, type_name)
    if kind == 'array':
        return _box_array(bind, form, type_name)
    raise RuntimeError('cannot return {}'.format(kind))


def _box_struct(bind, form, type_name):
    visible = [field for field in form['fields'] if field['kind'] != 'skip']
    named = _named_fields(form)
    tup = bind.fresh('t')
    fields = bind.fresh('d')
    name_tup = bind.fresh('n')
    parts = [bind.take(_new_tuple(tup, len(visible)))]
    if named:
        parts.append(bind.take(_new_dict(fields)))
        parts.append(bind.take(_new_tuple(name_tup, len(named))))
    else:
        parts.append(bind.take(_null_ptr(fields)))
        parts.append(bind.take(_null_ptr(name_tup)))
    for slot, field in enumerate(visible):
        native = bind.fresh('a')
        parts.append(_read_native(bind, native, field, 'result'))
        obj, more = _plain(bind, native, field)
        parts.extend(more)
        label = field['field'] or 'f'
        parts.append(bind.take(_export(
            tup, slot, fields, meta.const(label), obj,
        )))
        if named:
            parts.append(bind.take(_name_item(
                name_tup, slot, meta.const(label),
            )))
    parts.append(bind.take(_return_boxed(type_name, tup, fields, name_tup)))
    return parts


def _box_array(bind, form, type_name):
    tup, parts = _plain_array(bind, 'result', form)
    fields = bind.fresh('d')
    names = bind.fresh('n')
    parts.append(bind.take(_null_ptr(fields)))
    parts.append(bind.take(_null_ptr(names)))
    parts.append(bind.take(_return_boxed(type_name, tup, fields, names)))
    return parts


def _enum_variants(pc_type):
    names = list(getattr(pc_type, '_variant_names', None) or [])
    types = list(getattr(pc_type, '_variant_types', None) or [])
    tags = getattr(pc_type, '_tag_values', None)
    found = []
    for index, variant_name in enumerate(names):
        payload = types[index] if index < len(types) else None
        if isinstance(tags, dict):
            tag = int(tags[variant_name])
        else:
            tag = int(tags[index])
        found.append((variant_name, tag, payload))
    return found


def _as_stmts(items):
    stmts = []
    for item in items:
        if isinstance(item, list):
            stmts.extend(_as_stmts(item))
        elif isinstance(item, ast.stmt):
            stmts.append(item)
        else:
            stmts.extend(item.stmts)
    return stmts


def _store_named(bind, obj, field, src):
    label = field.get('field')
    if not label:
        return bind.take(_assign_index(obj, field['index'], src))
    stmt = ast.Assign(
        targets=[ast.Attribute(
            value=ast.Name(id=obj, ctx=ast.Load()),
            attr=label,
            ctx=ast.Store(),
        )],
        value=ast.Name(id=src, ctx=ast.Load()),
    )
    return bind.take(_stmt(stmt))


def _linear_call(bind):
    bind.globals['linear'] = linear
    return ast.Call(
        func=ast.Name(id='linear', ctx=ast.Load()),
        args=[],
        keywords=[],
    )


def _fill_struct_payload(bind, name, buf, form):
    stmts = [bind.take(_decl(name, bind.alias(form['type'])))]
    for field in form['fields']:
        if field['kind'] == 'skip' and field.get('linear'):
            label = field.get('field')
            if not label:
                raise RuntimeError('linear struct field needs a name')
            stmt = ast.Assign(
                targets=[ast.Attribute(
                    value=ast.Name(id=name, ctx=ast.Load()),
                    attr=label,
                    ctx=ast.Store(),
                )],
                value=_linear_call(bind),
            )
            stmts.append(bind.take(_stmt(stmt)))
            continue
        if field['kind'] == 'skip':
            continue
        if field.get('holds_linear'):
            raise RuntimeError(
                'nested linear field {} is not a Python boundary value'.format(
                    field.get('field') or field['index'],
                )
            )
        native = bind.fresh('a')
        stmts.append(bind.take(_from_bytes(
            native,
            bind.alias(field['type']),
            buf,
            int(field['offset']),
            _nbytes(field),
        )))
        stmts.append(_store_named(bind, name, field, native))
    return stmts


def _payload_expr(bind, buf, payload_type):
    if payload_type is None:
        return None, []
    form = _form(payload_type)
    if form['kind'] == 'skip' and not form.get('linear'):
        return None, []
    if form['kind'] == 'skip' and form.get('linear'):
        return _linear_call(bind), []
    name = bind.fresh('a')
    if form['kind'] == 'struct':
        return name, _fill_struct_payload(bind, name, buf, form)
    if form.get('holds_linear'):
        raise RuntimeError(
            'nested linear payload {} is not a Python boundary value'.format(
                form['name'],
            )
        )
    return name, [bind.take(_from_bytes(
        name, bind.alias(form['type']), buf, 0, _nbytes(form),
    ))]


def _enum_call(bind, enum_type, tag_type, tag_value, payload):
    args = [ast.Call(
        func=ast.Name(id=bind.alias(tag_type), ctx=ast.Load()),
        args=[ast.Constant(value=int(tag_value))],
        keywords=[],
    )]
    if isinstance(payload, ast.AST):
        args.append(payload)
    elif payload is not None:
        args.append(ast.Name(id=payload, ctx=ast.Load()))
    return ast.Call(
        func=ast.Name(id=bind.alias(enum_type), ctx=ast.Load()),
        args=args,
        keywords=[],
    )


def _variant_arm(bind, dst, buf, form, payload_type, tag_value):
    payload, stmts = _payload_expr(bind, buf, payload_type)
    call = _enum_call(
        bind, form['peeled'], form['tag']['type'], tag_value, payload,
    )
    stmts.append(bind.take(_set_call(dst, call)))
    return _as_stmts(stmts)


def _enum_arms(bind, dst, tag, buf, form):
    if form['tag']['kind'] != 'int':
        raise RuntimeError('linear enum tag must be a machine integer')
    node = bind.take(_bad_tag(buf))
    for _variant_name, tag_value, payload_type in reversed(
        _enum_variants(form['peeled'])
    ):
        arm = _variant_arm(bind, dst, buf, form, payload_type, tag_value)
        node = bind.take(_if_i64(tag, int(tag_value), arm, node))
    return node


def _kernel_call(arg_names):
    return ast.Call(
        func=ast.Name(id='kern', ctx=ast.Load()),
        args=[ast.Name(id=name, ctx=ast.Load()) for name in arg_names],
        keywords=[],
    )


def build_entry(spec):
    """Instantiate the entry template for one known signature."""
    bind = _Bind()
    params = [_form(pc_type) for pc_type in spec['param_types']]
    returns = _form(spec['return_type'])
    if returns['kind'] == 'skip':
        returns = {
            'kind': 'void',
            'type': spec['return_type'],
            'peeled': spec['return_type'],
        }
    arg_names = []
    loads = []
    rebuilds = []
    mints = []
    for index, form in enumerate(params):
        name = bind.fresh('a')
        arg_names.append(name)
        default = spec['defaults'][index]
        if form['kind'] == 'enum' and form.get('holds_linear'):
            if default is not None:
                raise RuntimeError('linear enum parameters have no Python default')
            src = bind.fresh('p')
            tag = bind.fresh('a')
            buf = bind.fresh('p')
            loads.append(bind.take(_load_required(index, src, [])))
            rebuilds.append(bind.take(_rebuild_enum(
                name,
                bind.alias(form['type']),
                tag,
                src,
                int(form['tag']['lo']),
                int(form['tag']['hi']),
                buf,
                _enum_arms(bind, name, tag, buf, form),
            )))
            continue
        if form['kind'] == 'skip' and form.get('linear'):
            mints.append(bind.take(_linear(name, bind.alias(form['type']))))
            continue
        if form['kind'] == 'skip':
            loads.append(bind.take(_decl(name, bind.alias(form['type']))))
            continue
        loads.append(bind.take(_load_param(bind, name, index, form, default)))
    linears = rebuilds + mints
    call = _kernel_call(arg_names)
    if returns['kind'] == 'void':
        returned = bind.take(_void_returned(call))
    else:
        returned = bind.take(_returned(
            bind.alias(returns['type']),
            call,
            _box_return(bind, returns),
        ))
    count = len(spec['names'])
    bind_names = [
        bind.take(_name_at(index, meta.const(text)))
        for index, text in enumerate(spec['names'])
    ]
    tail = bind.take(_tail(loads, linears, returned))
    body = bind.take(_entry(count or 1, count, bind_names, tail))
    return body.stmts, bind


def _generated_function(
    name,
    params,
    return_type,
    body,
    user_globals,
    source_file,
):
    return meta.func(
        name=name,
        params=params,
        return_type=return_type,
        body=body,
        required_globals=user_globals,
        source_file=source_file,
    )


def _compile_one(
    name,
    params,
    return_type,
    body,
    user_globals,
    source_file,
    group_key,
):
    gf = _generated_function(
        name,
        params,
        return_type,
        body,
        user_globals,
        source_file,
    )
    return meta.compile_generated(
        gf,
        user_globals=user_globals,
        group_key=group_key,
        source_file=source_file,
    )


def _entry_params():
    return [
        ('self_', ptr[void]),
        ('args', ptr[ptr[void]]),
        ('nargsf', i64),
        ('kwnames', ptr[void]),
    ]


def _object_args():
    return [
        ('self_', ptr[void]),
        ('arg', ptr[void]),
    ]


def _development_entry(spec, source_file):
    body, bind = build_entry(spec)
    user_globals = dict(bind.captured)
    user_globals.update(bind.globals)
    user_globals['kern'] = spec['callee']
    user_globals['__name__'] = 'pythoc.python_entry_bind'
    generated = _generated_function(
        spec['adapter'],
        _entry_params(),
        ptr[void],
        body,
        user_globals,
        source_file,
    )
    return generated, user_globals


def compile_adapter_object(specs, group_key, init_symbol, module_name):
    """Compile Python entries for known kernels into one object file."""
    source_file = group_key[0]
    wrappers = []
    for spec in specs:
        generated, user_globals = _development_entry(spec, source_file)
        wrappers.append(meta.compile_generated(
            generated,
            user_globals=user_globals,
            group_key=group_key,
            source_file=source_file,
        ))

    if init_symbol:
        hits = []
        captured = {}
        for spec, wrapper in zip(specs, wrappers):
            hit = _address_hit(meta.const(spec['export_id']), wrapper)
            captured.update(getattr(hit, '_user_globals', {}))
            hits.append(hit)
        address_body = _adapter_address(hits)
        captured.update(getattr(address_body, '_user_globals', {}))
        address_globals = dict(captured)
        address_globals['__name__'] = 'pythoc.python_entry_bind'
        address = _compile_one(
            'pythoc_adapter_address',
            _object_args(),
            ptr[void],
            address_body.stmts,
            address_globals,
            source_file,
            group_key,
        )
        install = _install_entry()
        install_globals = dict(getattr(install, '_user_globals', {}))
        install_globals['__name__'] = 'pythoc.python_entry_bind'
        install_wrapper = _compile_one(
            'pythoc_install_entry',
            _object_args(),
            ptr[void],
            install.stmts,
            install_globals,
            source_file,
            group_key,
        )
        init = _init(meta.const(module_name))
        init_globals = dict(getattr(init, '_user_globals', {}))
        init_globals.update({
            'pythoc_adapter_address': address,
            'pythoc_install_entry': install_wrapper,
            '__name__': 'pythoc.python_entry_bind',
        })
        _compile_one(
            init_symbol,
            [],
            ptr[void],
            init.stmts,
            init_globals,
            source_file,
            group_key,
        )

    from .build.output_manager import flush_all_pending_outputs
    from .build.output_manager import get_output_manager
    from .build.deps import get_dependency_tracker
    from .artifact import ArtifactRole

    manager = get_output_manager()
    manager.set_group_artifact_role(
        group_key,
        ArtifactRole.PYTHON_ADAPTER,
    )
    from . import python_entry
    runtime_source = os.path.realpath(python_entry.__file__)
    for runtime_key, group in manager.get_all_groups().items():
        if os.path.realpath(group.get('source_file') or '') == runtime_source:
            manager.set_group_artifact_role(
                runtime_key,
                ArtifactRole.PYTHON_RUNTIME,
            )
    flush_all_pending_outputs()
    return get_dependency_tracker().derive_obj_file_from_group_key(group_key)
