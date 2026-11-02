"""Shared Python boundary helpers.

One copy of the unbox/box logic. Each function's Python entry calls these
and then calls its own kernel directly.
"""

from pythoc import array, compile, f64, i8, i32, i64, ptr, static, struct, u64, void
from pythoc.artifact import ArtifactRole
from pythoc.build.output_manager import get_output_manager
from pythoc.cpy_api import (
    PyByteArray_AsString,
    PyByteArray_Size,
    PyBytes_AsString,
    PyBytes_Size,
    PyDict_GetItemString,
    PyDict_New,
    PyDict_SetItemString,
    PyBuffer_Release,
    PyErr_Clear,
    PyErr_Occurred,
    PyErr_SetString,
    PyObject_GetBuffer,
    PyFloat_FromDouble,
    PyList_GetItem,
    PyList_Size,
    PyFloat_AsDouble,
    PyLong_AsDouble,
    PyLong_AsLongLong,
    PyLong_AsUnsignedLongLong,
    PyLong_FromLongLong,
    PyLong_FromUnsignedLongLong,
    PyModule_Create2,
    PyObject_CallNoArgs,
    PyObject_GetAttrString,
    PyTuple_GetItem,
    PyTuple_New,
    PyTuple_SetItem,
    PyTuple_Size,
    PyType_GenericAlloc,
    PyUnicode_CompareWithASCIIString,
    PyUnicode_FromString,
    Py_DecRef,
    Py_IncRef,
    _PyLong_AsByteArray,
    _PyLong_FromByteArray,
    memcpy,
)
from pythoc.decorators.extern import extern_global
from pythoc.libc.errno import errno_slot

PyLong_Type = extern_global(i8, 'PyLong_Type', lib='')
PyFloat_Type = extern_global(i8, 'PyFloat_Type', lib='')
PyBool_Type = extern_global(i8, 'PyBool_Type', lib='')
PyTuple_Type = extern_global(i8, 'PyTuple_Type', lib='')
PyList_Type = extern_global(i8, 'PyList_Type', lib='')
PyDict_Type = extern_global(i8, 'PyDict_Type', lib='')
PyBytes_Type = extern_global(i8, 'PyBytes_Type', lib='')
PyByteArray_Type = extern_global(i8, 'PyByteArray_Type', lib='')
PyUnicode_Type = extern_global(i8, 'PyUnicode_Type', lib='')
_Py_NoneStruct = extern_global(i8, '_Py_NoneStruct', lib='')
_Py_TrueStruct = extern_global(i8, '_Py_TrueStruct', lib='')
_Py_FalseStruct = extern_global(i8, '_Py_FalseStruct', lib='')
PyExc_TypeError = extern_global(ptr[void], 'PyExc_TypeError', lib='')
PyExc_OverflowError = extern_global(ptr[void], 'PyExc_OverflowError', lib='')
PyExc_ValueError = extern_global(ptr[void], 'PyExc_ValueError', lib='')
PyExc_AttributeError = extern_global(ptr[void], 'PyExc_AttributeError', lib='')

Meth = struct[ptr[i8], ptr[void], i32, ptr[i8]]
ModuleDef = struct[
    i64, ptr[void], ptr[void], i64, ptr[void],
    ptr[i8], ptr[i8], i64, ptr[void],
    ptr[void], ptr[void], ptr[void], ptr[void],
]


@compile(linkage='internal')
class PyRuntime:
    literal_type: static[ptr[void]]
    callable_type: static[ptr[void]]
    types: static[ptr[void]]


@compile
def pythoc_make_module(
    name: ptr[i8],
    address: ptr[void],
    install: ptr[void],
) -> ptr[void]:
    methods: static[array[Meth, 3]]
    methods[0][0] = "adapter_address"
    methods[0][1] = address
    methods[0][2] = i32(8)
    methods[0][3] = "adapter address"
    methods[1][0] = "install_runtime"
    methods[1][1] = install
    methods[1][2] = i32(8)
    methods[1][3] = "install runtime"
    defn: static[ModuleDef]
    defn[0] = i64(1)
    defn[5] = name
    defn[7] = i64(-1)
    defn[8] = ptr[void](ptr(methods))
    return PyModule_Create2(ptr(defn), i32(1013))


@compile
def pythoc_install_runtime(
    literal_ty: ptr[void],
    callable_ty: ptr[void],
    types: ptr[void],
) -> void:
    # IncRef the new values before DecRef-ing the old ones: reinstalling
    # the same object must not transiently drop it to refcount zero.
    previous: ptr[void] = PyRuntime.literal_type
    Py_IncRef(literal_ty)
    PyRuntime.literal_type = literal_ty
    if i64(previous) != i64(0):
        Py_DecRef(previous)

    previous = PyRuntime.callable_type
    Py_IncRef(callable_ty)
    PyRuntime.callable_type = callable_ty
    if i64(previous) != i64(0):
        Py_DecRef(previous)

    previous = PyRuntime.types
    Py_IncRef(types)
    PyRuntime.types = types
    if i64(previous) != i64(0):
        Py_DecRef(previous)


@compile
def pythoc_py_load(obj: ptr[void], offset: i64) -> ptr[void]:
    raw: ptr[i8] = ptr[i8](obj)
    at: ptr[i8] = raw + offset
    slot: ptr[ptr[void]] = ptr[ptr[void]](at)
    return slot[0]


@compile
def pythoc_py_store(obj: ptr[void], offset: i64, value: ptr[void]) -> void:
    raw: ptr[i8] = ptr[i8](obj)
    at: ptr[i8] = raw + offset
    slot: ptr[ptr[void]] = ptr[ptr[void]](at)
    slot[0] = value


@compile
def pythoc_py_type_id(obj: ptr[void]) -> i64:
    return i64(pythoc_py_load(obj, i64(8)))


@compile
def pythoc_none_ptr() -> ptr[void]:
    return ptr[void](ptr(_Py_NoneStruct))


@compile
def pythoc_none_ref() -> ptr[void]:
    obj: ptr[void] = pythoc_none_ptr()
    Py_IncRef(obj)
    return obj


@compile
def pythoc_type_error(msg: ptr[i8]) -> void:
    PyErr_SetString(PyExc_TypeError, msg)


@compile
def pythoc_errno_store(value: i32) -> void:
    errno_slot()[0] = value


@compile
def pythoc_overflow_error(msg: ptr[i8]) -> void:
    PyErr_SetString(PyExc_OverflowError, msg)


@compile
def pythoc_value_error(msg: ptr[i8]) -> void:
    PyErr_SetString(PyExc_ValueError, msg)


@compile
def pythoc_release(obj: ptr[void]) -> void:
    if i64(obj) != i64(0):
        Py_DecRef(obj)


@compile
def pythoc_own(obj: ptr[void]) -> ptr[void]:
    if i64(obj) != i64(0):
        Py_IncRef(obj)
    return obj


@compile
def pythoc_long_id() -> i64:
    return i64(ptr(PyLong_Type))


@compile
def pythoc_float_id() -> i64:
    return i64(ptr(PyFloat_Type))


@compile
def pythoc_bool_id() -> i64:
    return i64(ptr(PyBool_Type))


@compile
def pythoc_tuple_id() -> i64:
    return i64(ptr(PyTuple_Type))


@compile
def pythoc_list_id() -> i64:
    return i64(ptr(PyList_Type))


@compile
def pythoc_dict_id() -> i64:
    return i64(ptr(PyDict_Type))


@compile
def pythoc_bytes_id() -> i64:
    return i64(ptr(PyBytes_Type))


@compile
def pythoc_bytearray_id() -> i64:
    return i64(ptr(PyByteArray_Type))


@compile
def pythoc_unwrap(obj: ptr[void]) -> ptr[void]:
    if i64(obj) == i64(0):
        return obj
    if pythoc_py_type_id(obj) == i64(PyRuntime.literal_type):
        return pythoc_py_load(obj, i64(48))
    return obj


@compile
def pythoc_is_none(obj: ptr[void]) -> i32:
    if i64(obj) == i64(0):
        return i32(0)
    if i64(obj) == i64(pythoc_none_ptr()):
        return i32(1)
    return i32(0)


@compile
def pythoc_is_true(obj: ptr[void]) -> i32:
    if i64(obj) == i64(ptr(_Py_TrueStruct)):
        return i32(1)
    return i32(0)


@compile
def pythoc_bind_args(
    args: ptr[ptr[void]],
    nargsf: i64,
    kwnames: ptr[void],
    names: ptr[ptr[i8]],
    nparam: i64,
    out: ptr[ptr[void]],
) -> i32:
    raw: i64 = nargsf
    nargs: i64 = raw
    if raw < i64(0):
        nargs = raw + i64(9223372036854775807) + i64(1)
    index: i64 = i64(0)
    while index < nparam:
        out[index] = ptr[void](0)
        index = index + i64(1)
    if nargs > nparam:
        pythoc_type_error("too many positional arguments")
        return i32(1)
    npos: i64 = nargs
    index = i64(0)
    while index < npos:
        out[index] = args[index]
        index = index + i64(1)
    if i64(kwnames) != i64(0):
        nkw: i64 = PyTuple_Size(kwnames)
        kw: i64 = i64(0)
        while kw < nkw:
            label: ptr[void] = PyTuple_GetItem(kwnames, kw)
            hit: i64 = i64(-1)
            slot: i64 = i64(0)
            while slot < nparam:
                if PyUnicode_CompareWithASCIIString(label, names[slot]) == i32(0):
                    hit = slot
                slot = slot + i64(1)
            if hit < i64(0):
                pythoc_type_error("unexpected keyword argument")
                return i32(1)
            if i64(out[hit]) != i64(0):
                pythoc_type_error("multiple values for argument")
                return i32(1)
            out[hit] = args[nargs + kw]
            kw = kw + i64(1)
    return i32(0)


@compile
def pythoc__as_i64(obj: ptr[void]) -> i64:
    kind: i64 = pythoc_py_type_id(obj)
    if kind == pythoc_bool_id():
        if pythoc_is_true(obj) == i32(1):
            return i64(1)
        return i64(0)
    if kind != pythoc_long_id():
        pythoc_type_error("an integer is required")
        return i64(0)
    return PyLong_AsLongLong(obj)


@compile
def pythoc_take_i64(obj: ptr[void]) -> i64:
    if i64(obj) == i64(0):
        pythoc_type_error("missing required argument")
        return i64(0)
    return pythoc__as_i64(pythoc_unwrap(obj))


@compile
def pythoc_take_signed(obj: ptr[void], lo: i64, hi: i64) -> i64:
    value: i64 = pythoc_take_i64(obj)
    if i64(PyErr_Occurred()) != i64(0):
        return i64(0)
    if value < lo:
        pythoc_overflow_error("integer out of range")
        return i64(0)
    if value > hi:
        pythoc_overflow_error("integer out of range")
        return i64(0)
    return value


@compile
def pythoc_take_u64(obj: ptr[void]) -> u64:
    if i64(obj) == i64(0):
        pythoc_type_error("missing required argument")
        return u64(0)
    inner: ptr[void] = pythoc_unwrap(obj)
    kind: i64 = pythoc_py_type_id(inner)
    if kind == pythoc_bool_id():
        if pythoc_is_true(inner) == i32(1):
            return u64(1)
        return u64(0)
    if kind != pythoc_long_id():
        pythoc_type_error("an integer is required")
        return u64(0)
    return PyLong_AsUnsignedLongLong(inner)


@compile
def pythoc_take_f64(obj: ptr[void]) -> f64:
    if i64(obj) == i64(0):
        pythoc_type_error("missing required argument")
        return f64(0.0)
    inner: ptr[void] = pythoc_unwrap(obj)
    kind: i64 = pythoc_py_type_id(inner)
    if kind == pythoc_bool_id():
        if pythoc_is_true(inner) == i32(1):
            return f64(1.0)
        return f64(0.0)
    if kind == pythoc_float_id():
        return PyFloat_AsDouble(inner)
    if kind == pythoc_long_id():
        return PyLong_AsDouble(inner)
    pythoc_type_error("expected a real number")
    return f64(0.0)


@compile
def pythoc_take_bool(obj: ptr[void]) -> i32:
    if i64(obj) == i64(0):
        pythoc_type_error("missing required argument")
        return i32(0)
    inner: ptr[void] = pythoc_unwrap(obj)
    kind: i64 = pythoc_py_type_id(inner)
    if kind == pythoc_bool_id():
        return pythoc_is_true(inner)
    if kind == pythoc_long_id():
        if PyLong_AsLongLong(inner) != i64(0):
            return i32(1)
        return i32(0)
    pythoc_type_error("expected a bool")
    return i32(0)


@compile
def pythoc_take_ptr(obj: ptr[void]) -> i64:
    if i64(obj) == i64(0):
        pythoc_type_error("expected a pointer")
        return i64(0)
    inner: ptr[void] = pythoc_unwrap(obj)
    if pythoc_is_none(inner) == i32(1):
        return i64(0)
    kind: i64 = pythoc_py_type_id(inner)
    if kind == pythoc_bool_id():
        return i64(pythoc_is_true(inner))
    if kind == pythoc_long_id():
        return PyLong_AsLongLong(inner)
    if kind == pythoc_bytes_id():
        return i64(PyBytes_AsString(inner))
    if kind == pythoc_bytearray_id():
        return i64(PyByteArray_AsString(inner))
    pointer_method: ptr[void] = PyObject_GetAttrString(
        inner, "_pythoc_pointer")
    if i64(pointer_method) != i64(0):
        pointer_value: ptr[void] = PyObject_CallNoArgs(pointer_method)
        pythoc_release(pointer_method)
        if i64(pointer_value) == i64(0):
            return i64(0)
        address: u64 = PyLong_AsUnsignedLongLong(pointer_value)
        pythoc_release(pointer_value)
        return i64(address)
    if i64(PyErr_Occurred()) != i64(PyExc_AttributeError):
        return i64(0)
    PyErr_Clear()
    if kind == i64(PyRuntime.callable_type):
        kern: ptr[void] = PyObject_GetAttrString(inner, "_pythoc_kernel")
        if i64(kern) == i64(0):
            pythoc_type_error("expected a pointer")
            return i64(0)
        addr: u64 = PyLong_AsUnsignedLongLong(kern)
        Py_DecRef(kern)
        return i64(addr)
    buf: i64 = pythoc_buffer_addr(inner)
    if i64(PyErr_Occurred()) == i64(0):
        return buf
    PyErr_Clear()
    pythoc_type_error("expected a pointer")
    return i64(0)


@compile
def pythoc__buffer_data(obj: ptr[void]) -> i64:
    view: array[i8, 80]
    if PyObject_GetBuffer(obj, ptr[void](ptr(view)), i32(0)) != i32(0):
        return i64(0)
    raw: ptr[i8] = ptr[i8](ptr(view))
    slot: ptr[i64] = ptr[i64](raw)
    addr: i64 = slot[0]
    PyBuffer_Release(ptr[void](ptr(view)))
    return addr


@compile
def pythoc__type_code_is(obj: ptr[void], code: ptr[i8]) -> i32:
    raw: ptr[i8] = ptr[i8](obj)
    at: ptr[i8] = raw + i64(8)
    slot: ptr[ptr[void]] = ptr[ptr[void]](at)
    ty: ptr[void] = slot[0]
    marker: ptr[void] = PyObject_GetAttrString(ty, "_type_")
    if i64(marker) == i64(0):
        PyErr_Clear()
        return i32(0)
    if pythoc_py_type_id(marker) != i64(ptr(PyUnicode_Type)):
        pythoc_release(marker)
        return i32(0)
    eq: i32 = PyUnicode_CompareWithASCIIString(marker, code)
    pythoc_release(marker)
    PyErr_Clear()
    if eq == i32(0):
        return i32(1)
    return i32(0)


@compile
def pythoc__contained_address(obj: ptr[void]) -> i64:
    held: ptr[void] = PyObject_GetAttrString(obj, "value")
    if i64(held) == i64(0):
        return i64(0)
    if pythoc_is_none(held) == i32(1):
        pythoc_release(held)
        return i64(0)
    kind: i64 = pythoc_py_type_id(held)
    if kind == pythoc_long_id():
        addr: i64 = PyLong_AsLongLong(held)
        pythoc_release(held)
        return addr
    if kind == pythoc_bytes_id():
        addr2: i64 = i64(PyBytes_AsString(held))
        pythoc_release(held)
        return addr2
    if kind == pythoc_bytearray_id():
        addr3: i64 = i64(PyByteArray_AsString(held))
        pythoc_release(held)
        return addr3
    pythoc_release(held)
    pythoc_type_error("expected a pointer")
    return i64(0)


@compile
def pythoc_buffer_addr(obj: ptr[void]) -> i64:
    # ctypes.pointer: buffer protocol exposes the pointer slot. contents is
    # the pointee, and its buffer is the address the kernel should receive.
    pointee: ptr[void] = PyObject_GetAttrString(obj, "contents")
    if i64(pointee) != i64(0):
        addr: i64 = pythoc__buffer_data(pointee)
        pythoc_release(pointee)
        return addr
    if i64(PyErr_Occurred()) != i64(PyExc_AttributeError):
        PyErr_Clear()
        return i64(0)
    PyErr_Clear()

    # c_void_p / c_char_p store the address in .value. Their buffer is the
    # slot that holds that address, which is not the pointer itself.
    if pythoc__type_code_is(obj, "P") == i32(1):
        return pythoc__contained_address(obj)
    if pythoc__type_code_is(obj, "z") == i32(1):
        return pythoc__contained_address(obj)

    data: i64 = pythoc__buffer_data(obj)
    if i64(PyErr_Occurred()) == i64(0):
        return data
    PyErr_Clear()
    owned: ptr[void] = PyObject_GetAttrString(obj, "_obj")
    if i64(owned) == i64(0):
        return i64(0)
    data2: i64 = pythoc__buffer_data(owned)
    pythoc_release(owned)
    if i64(PyErr_Occurred()) == i64(0):
        return data2
    return i64(0)


@compile
def pythoc_fill_wide(obj: ptr[void], dest: ptr[i8], is_signed: i32) -> i32:
    if i64(obj) == i64(0):
        pythoc_type_error("missing required argument")
        return i32(1)
    inner: ptr[void] = pythoc_unwrap(obj)
    kind: i64 = pythoc_py_type_id(inner)
    if kind == pythoc_bool_id():
        if pythoc_is_true(inner) == i32(1):
            dest[0] = i8(1)
        else:
            dest[0] = i8(0)
        return i32(0)
    if kind != pythoc_long_id():
        pythoc_type_error("an integer is required")
        return i32(1)
    return _PyLong_AsByteArray(inner, dest, u64(16), i32(1), is_signed)


@compile
def pythoc_fill_bytes(obj: ptr[void], dest: ptr[i8], n: i64) -> i32:
    if i64(obj) == i64(0):
        pythoc_type_error("expected bytes")
        return i32(1)
    inner: ptr[void] = pythoc_unwrap(obj)
    kind: i64 = pythoc_py_type_id(inner)
    src: ptr[i8] = ptr[i8](0)
    size: i64 = i64(0)
    if kind == pythoc_bytes_id():
        src = PyBytes_AsString(inner)
        size = PyBytes_Size(inner)
    else:
        if kind == pythoc_bytearray_id():
            src = PyByteArray_AsString(inner)
            size = PyByteArray_Size(inner)
        else:
            pythoc_type_error("expected bytes")
            return i32(1)
    if size < n:
        pythoc_type_error("buffer too small")
        return i32(1)
    memcpy(dest, src, u64(n))
    return i32(0)


@compile
def pythoc_zero_bytes(dest: ptr[i8], n: i64) -> void:
    index: i64 = i64(0)
    while index < n:
        dest[index] = i8(0)
        index = index + i64(1)


@compile
def pythoc_field_item(src: ptr[void], index: i64, name: ptr[i8]) -> ptr[void]:
    if i64(src) == i64(0):
        return src
    if pythoc_py_type_id(src) == i64(PyRuntime.literal_type):
        fields: ptr[void] = pythoc_py_load(src, i64(32))
        if i64(fields) != i64(0):
            if i64(fields) != i64(pythoc_none_ptr()):
                if pythoc_py_type_id(fields) == pythoc_dict_id():
                    got: ptr[void] = PyDict_GetItemString(fields, name)
                    if i64(got) != i64(0):
                        return pythoc_own(got)
        return pythoc_field_item(pythoc_py_load(src, i64(48)), index, name)
    if pythoc_py_type_id(src) == pythoc_tuple_id():
        if index < PyTuple_Size(src):
            return pythoc_own(PyTuple_GetItem(src, index))
        pythoc_type_error("not enough values")
        return ptr[void](0)
    if pythoc_py_type_id(src) == pythoc_list_id():
        if index < PyList_Size(src):
            return pythoc_own(PyList_GetItem(src, index))
        pythoc_type_error("not enough values")
        return ptr[void](0)
    if pythoc_py_type_id(src) == pythoc_dict_id():
        got: ptr[void] = PyDict_GetItemString(src, name)
        if i64(got) == i64(0):
            pythoc_type_error("missing field")
            return ptr[void](0)
        return pythoc_own(got)
    return PyObject_GetAttrString(src, name)


@compile
def pythoc_box_new(type_name: ptr[i8]) -> ptr[void]:
    ty: ptr[void] = PyDict_GetItemString(PyRuntime.types, type_name)
    if i64(ty) == i64(0):
        pythoc_type_error("unknown PythoC type")
        return ptr[void](0)
    obj: ptr[void] = PyType_GenericAlloc(PyRuntime.literal_type, i64(0))
    if i64(obj) == i64(0):
        return obj
    Py_IncRef(ty)
    pythoc_py_store(obj, i64(40), ty)
    blank: ptr[void] = pythoc_none_ptr()
    Py_IncRef(blank)
    pythoc_py_store(obj, i64(32), blank)
    Py_IncRef(blank)
    pythoc_py_store(obj, i64(24), blank)
    Py_IncRef(blank)
    pythoc_py_store(obj, i64(16), blank)
    return obj


@compile
def pythoc_box_value(type_name: ptr[i8], value: ptr[void]) -> ptr[void]:
    obj: ptr[void] = pythoc_box_new(type_name)
    if i64(obj) == i64(0):
        return obj
    pythoc_py_store(obj, i64(48), value)
    return obj


@compile
def pythoc_box_fields(obj: ptr[void], fields: ptr[void], names: ptr[void]) -> void:
    previous: ptr[void] = pythoc_py_load(obj, i64(32))
    pythoc_release(previous)
    pythoc_py_store(obj, i64(32), fields)
    previous = pythoc_py_load(obj, i64(24))
    pythoc_release(previous)
    pythoc_py_store(obj, i64(24), names)


@compile
def pythoc_box_i64(value: i64, type_name: ptr[i8]) -> ptr[void]:
    return pythoc_box_value(type_name, PyLong_FromLongLong(value))


@compile
def pythoc_box_u64(value: u64, type_name: ptr[i8]) -> ptr[void]:
    return pythoc_box_value(type_name, PyLong_FromUnsignedLongLong(value))


@compile
def pythoc_box_f64(value: f64, type_name: ptr[i8]) -> ptr[void]:
    return pythoc_box_value(type_name, PyFloat_FromDouble(value))


@compile
def pythoc_bool_obj(value: i32) -> ptr[void]:
    obj: ptr[void] = ptr[void](ptr(_Py_FalseStruct))
    if value != i32(0):
        obj = ptr[void](ptr(_Py_TrueStruct))
    Py_IncRef(obj)
    return obj


@compile
def pythoc_box_bool(value: i32, type_name: ptr[i8]) -> ptr[void]:
    obj: ptr[void] = ptr[void](ptr(_Py_FalseStruct))
    if value != i32(0):
        obj = ptr[void](ptr(_Py_TrueStruct))
    Py_IncRef(obj)
    return pythoc_box_value(type_name, obj)


@compile
def pythoc_export_item(
    tup: ptr[void],
    index: i64,
    dct: ptr[void],
    key: ptr[i8],
    obj: ptr[void],
) -> void:
    if i64(dct) != i64(0):
        Py_IncRef(obj)
    PyTuple_SetItem(tup, index, obj)
    if i64(dct) != i64(0):
        PyDict_SetItemString(dct, key, obj)
        Py_DecRef(obj)


@compile
def pythoc_box_wide(src: ptr[i8], is_signed: i32, type_name: ptr[i8]) -> ptr[void]:
    num: ptr[void] = _PyLong_FromByteArray(src, u64(16), i32(1), is_signed)
    if i64(num) == i64(0):
        return num
    return pythoc_box_value(type_name, num)


def _check_literal_layout():
    """Pin the pc_literal instance-slot offsets the compiled helpers use.

    ``unwrap``/``box_new`` read ``_value`` at byte offset 48 of the
    object.  That offset is an artifact of how CPython lays out
    ``pc_literal.__slots__`` members; a silent reorder would corrupt
    every boundary crossing, so verify it eagerly at import time.
    """
    import ctypes

    from .builtin_entities.pc_literal import pc_literal

    expected = (
        (16, '_ctypes_owner'),
        (24, '_field_names'),
        (32, '_fields'),
        (40, '_pc_type'),
        (48, '_value'),
    )
    probe = pc_literal(0, None)
    base = id(probe)
    for offset, name in expected:
        value = ctypes.cast(
            base + offset, ctypes.POINTER(ctypes.py_object),
        )[0]
        if value is not getattr(probe, name):
            raise RuntimeError(
                'pc_literal slot layout changed: offset {} no longer '
                'holds {!r}; update the offsets in python_entry.py'.format(
                    offset, name,
                )
            )


_check_literal_layout()

# Classify the producer before a caller requests an adapter or a link plan.
get_output_manager().set_group_artifact_role(
    pythoc_make_module._binding.group_key, ArtifactRole.PYTHON_RUNTIME,
)
