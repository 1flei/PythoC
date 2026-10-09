"""PythoCCallable, compiled from PythoC for the running interpreter.

The instance layout and type-object offsets come from callable_layout,
which reads this interpreter's C API headers. The extension exports
PyInit__callable. User kernels are not part of this object.
"""

from pythoc import (
    array,
    compile,
    extern,
    extern_global,
    func,
    i32,
    i64,
    i8,
    ptr,
    static,
    struct,
    u32,
    u64,
    void,
)
from pythoc.callable_layout import python_abi_layout

LAYOUT = python_abi_layout()

OFF_VECTORCALL = LAYOUT['vectorcall']
OFF_DICT = LAYOUT['dict']
OFF_WRAPPED = LAYOUT['wrapped']
OFF_SLOW = LAYOUT['slow']
OFF_LOCK = LAYOUT['lock']
OFF_STATE = LAYOUT['state']
OFF_TP_DICTS = LAYOUT['tp_dictoffset']
OFF_TP_FLAGS = LAYOUT['tp_flags']
OFF_TP_VECTORCALL = LAYOUT.get('tp_vectorcall_offset', 0)
OFF_M_NAME = LAYOUT['m_name']
OFF_M_SIZE = LAYOUT['m_size']
BASICSIZE = LAYOUT['basicsize']
MODULE_BYTES = LAYOUT['module_size']
MODULE_WORDS = (MODULE_BYTES + 7) // 8
API_VERSION = LAYOUT['api_version']
HAVE_VECTORCALL = LAYOUT['have_vectorcall']
FLAG_BITS = (
    LAYOUT['flag_default'] | LAYOUT['flag_gc'] | LAYOUT['flag_vectorcall']
)
TP_FLAGS_SIZE = LAYOUT['tp_flags_size']
SLOT_DEALLOC = LAYOUT['slot_dealloc']
SLOT_TRAVERSE = LAYOUT['slot_traverse']
SLOT_CLEAR = LAYOUT['slot_clear']
SLOT_REPR = LAYOUT['slot_repr']
SLOT_CALL = LAYOUT['slot_call']
SLOT_METHODS = LAYOUT['slot_methods']
SLOT_GETSET = LAYOUT['slot_getset']
SLOT_INIT = LAYOUT['slot_init']
SLOT_NEW = LAYOUT['slot_new']
SLOT_DOC = LAYOUT['slot_doc']
METH_O = LAYOUT['meth_o']
METH_NOARGS = LAYOUT['meth_noargs']
NARGS_MASK = 9223372036854775807

TypeSlot = struct[i32, ptr[void]]
TypeSpec = struct[ptr[i8], i32, i32, u32, ptr[void]]
MethodDef = struct[ptr[i8], ptr[void], i32, ptr[i8]]
GetSetDef = struct[ptr[i8], ptr[void], ptr[void], ptr[i8], ptr[void]]

Vectorcall = func[ptr[void], ptr[ptr[void]], i64, ptr[void], ptr[void]]
Visit = func[ptr[void], ptr[void], i32]

if TypeSlot.get_size_bytes() != LAYOUT['type_slot_size']:
    raise RuntimeError('PyType_Slot layout does not match this interpreter')
if TypeSpec.get_size_bytes() != LAYOUT['type_spec_size']:
    raise RuntimeError('PyType_Spec layout does not match this interpreter')
if MethodDef.get_size_bytes() != LAYOUT['method_size']:
    raise RuntimeError('PyMethodDef layout does not match this interpreter')
if GetSetDef.get_size_bytes() != LAYOUT['getset_size']:
    raise RuntimeError('PyGetSetDef layout does not match this interpreter')


@extern(lib='')
def PyType_FromSpec(spec: ptr[void]) -> ptr[void]:
    pass


@extern(lib='')
def PyType_GenericAlloc(typ: ptr[void], nitems: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyModuleDef_Init(defn: ptr[void]) -> ptr[void]:
    pass


@extern(lib='')
def PyModule_Create2(defn: ptr[void], apiver: i32) -> ptr[void]:
    pass


@extern(lib='')
def PyModule_AddObject(mod: ptr[void], name: ptr[i8], value: ptr[void]) -> i32:
    pass


@extern(lib='')
def Py_IncRef(obj: ptr[void]) -> void:
    pass


@extern(lib='')
def Py_DecRef(obj: ptr[void]) -> void:
    pass


@extern(lib='')
def PyErr_SetString(exc: ptr[void], msg: ptr[i8]) -> void:
    pass


@extern(lib='')
def PyErr_Occurred() -> ptr[void]:
    pass


@extern(lib='')
def PyErr_Clear() -> void:
    pass


@extern(lib='')
def PyErr_NoMemory() -> ptr[void]:
    pass


@extern(lib='')
def PyObject_GetAttrString(obj: ptr[void], name: ptr[i8]) -> ptr[void]:
    pass


@extern(lib='')
def PyObject_Call(
    callee: ptr[void], args: ptr[void], kwargs: ptr[void],
) -> ptr[void]:
    pass


@extern(lib='')
def PyObject_GC_UnTrack(obj: ptr[void]) -> void:
    pass


@extern(lib='')
def PyObject_GC_Del(obj: ptr[void]) -> void:
    pass


@extern(lib='')
def PyTuple_New(size: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyTuple_Size(tup: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyTuple_GetItem(tup: ptr[void], index: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyTuple_SetItem(tup: ptr[void], index: i64, value: ptr[void]) -> i32:
    pass


@extern(lib='')
def PyDict_New() -> ptr[void]:
    pass


@extern(lib='')
def PyDict_Size(dct: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyDict_SetItem(dct: ptr[void], key: ptr[void], value: ptr[void]) -> i32:
    pass


@extern(lib='')
def PyDict_Next(
    dct: ptr[void],
    pos: ptr[i64],
    key: ptr[ptr[void]],
    value: ptr[ptr[void]],
) -> i32:
    pass


@extern(lib='')
def PyMem_Malloc(size: u64) -> ptr[void]:
    pass


@extern(lib='')
def PyMem_Free(block: ptr[void]) -> void:
    pass


@extern(lib='')
def PyUnicode_FromString(text: ptr[i8]) -> ptr[void]:
    pass


@extern(lib='')
def PyUnicode_Concat(left: ptr[void], right: ptr[void]) -> ptr[void]:
    pass


@extern(lib='')
def PyBool_FromLong(value: i32) -> ptr[void]:
    pass


@extern(lib='')
def PyLong_AsUnsignedLongLong(obj: ptr[void]) -> u64:
    pass


@extern(lib='')
def PyCallable_Check(obj: ptr[void]) -> i32:
    pass


@extern(lib='')
def PyThread_allocate_lock() -> ptr[void]:
    pass


@extern(lib='')
def PyThread_free_lock(lock: ptr[void]) -> void:
    pass


@extern(lib='')
def PyThread_acquire_lock(lock: ptr[void], wait: i32) -> i32:
    pass


@extern(lib='')
def PyThread_release_lock(lock: ptr[void]) -> void:
    pass


PyExc_RuntimeError = extern_global(ptr[void], 'PyExc_RuntimeError', lib='')
PyExc_TypeError = extern_global(ptr[void], 'PyExc_TypeError', lib='')
PyExc_ValueError = extern_global(ptr[void], 'PyExc_ValueError', lib='')
_Py_NoneStruct = extern_global(i8, '_Py_NoneStruct', lib='')


@compile
def none_ptr() -> ptr[void]:
    return ptr[void](ptr(_Py_NoneStruct))


@compile
def load_ptr(base: ptr[void], offset: i64) -> ptr[void]:
    raw: ptr[i8] = ptr[i8](base)
    slot: ptr[ptr[void]] = ptr[ptr[void]](raw + offset)
    return slot[0]


@compile
def store_ptr(base: ptr[void], offset: i64, value: ptr[void]) -> void:
    raw: ptr[i8] = ptr[i8](base)
    slot: ptr[ptr[void]] = ptr[ptr[void]](raw + offset)
    slot[0] = value


@compile
def load_i32(base: ptr[void], offset: i64) -> i32:
    raw: ptr[i8] = ptr[i8](base)
    slot: ptr[i32] = ptr[i32](raw + offset)
    return slot[0]


@compile
def store_i32(base: ptr[void], offset: i64, value: i32) -> void:
    raw: ptr[i8] = ptr[i8](base)
    slot: ptr[i32] = ptr[i32](raw + offset)
    slot[0] = value


@compile
def load_i64(base: ptr[void], offset: i64) -> i64:
    raw: ptr[i8] = ptr[i8](base)
    slot: ptr[i64] = ptr[i64](raw + offset)
    return slot[0]


@compile
def store_i64(base: ptr[void], offset: i64, value: i64) -> void:
    raw: ptr[i8] = ptr[i8](base)
    slot: ptr[i64] = ptr[i64](raw + offset)
    slot[0] = value


@compile
def load_state(obj: ptr[void]) -> i32:
    return load_i32(obj, i64(OFF_STATE))


@compile
def store_state(obj: ptr[void], value: i32) -> void:
    store_i32(obj, i64(OFF_STATE), value)


@compile
def lock_ptr(obj: ptr[void]) -> ptr[void]:
    return load_ptr(obj, i64(OFF_LOCK))


@compile
def lock_enter(obj: ptr[void]) -> void:
    PyThread_acquire_lock(lock_ptr(obj), i32(1))


@compile
def lock_leave(obj: ptr[void]) -> void:
    PyThread_release_lock(lock_ptr(obj))


@compile
def vectorcall_ptr(obj: ptr[void]) -> ptr[void]:
    return load_ptr(obj, i64(OFF_VECTORCALL))


@compile
def call_vectorcall(
    fn: Vectorcall,
    obj: ptr[void],
    args: ptr[ptr[void]],
    nargsf: i64,
    kwnames: ptr[void],
) -> ptr[void]:
    return fn(obj, args, nargsf, kwnames)


@compile
def as_vectorcall(raw: ptr[void]) -> Vectorcall:
    return Vectorcall(raw)


@compile
def nargs_of(nargsf: i64) -> i64:
    return nargsf & i64(NARGS_MASK)


@compile
def zero_bytes(base: ptr[i8], size: i64) -> void:
    index: i64 = i64(0)
    while index < size:
        base[index] = i8(0)
        index = index + i64(1)


@compile
def release(obj: ptr[void]) -> void:
    if i64(obj) != i64(0):
        Py_DecRef(obj)


@compile
def visit_field(obj: ptr[void], offset: i64, visit: Visit, arg: ptr[void]) -> i32:
    field: ptr[void] = load_ptr(obj, offset)
    if i64(field) == i64(0):
        return i32(0)
    return visit(field, arg)


@compile
def clear_field(obj: ptr[void], offset: i64) -> void:
    field: ptr[void] = load_ptr(obj, offset)
    store_ptr(obj, offset, ptr[void](0))
    release(field)


@compile
def call_noargs_attr(obj: ptr[void], name: ptr[i8]) -> ptr[void]:
    method: ptr[void] = PyObject_GetAttrString(obj, name)
    if i64(method) == i64(0):
        return ptr[void](0)
    empty: ptr[void] = PyTuple_New(i64(0))
    if i64(empty) == i64(0):
        release(method)
        return ptr[void](0)
    result: ptr[void] = PyObject_Call(method, empty, ptr[void](0))
    release(empty)
    release(method)
    return result


@compile
def callable_traverse(obj: ptr[void], visit: Visit, arg: ptr[void]) -> i32:
    code: i32 = visit_field(obj, i64(OFF_DICT), visit, arg)
    if code != i32(0):
        return code
    code = visit_field(obj, i64(OFF_WRAPPED), visit, arg)
    if code != i32(0):
        return code
    return visit_field(obj, i64(OFF_SLOW), visit, arg)


@compile
def callable_clear(obj: ptr[void]) -> i32:
    clear_field(obj, i64(OFF_DICT))
    clear_field(obj, i64(OFF_WRAPPED))
    clear_field(obj, i64(OFF_SLOW))
    return i32(0)


@compile
def callable_dealloc(obj: ptr[void]) -> void:
    PyObject_GC_UnTrack(obj)
    callable_clear(obj)
    lock: ptr[void] = lock_ptr(obj)
    if i64(lock) != i64(0):
        PyThread_free_lock(lock)
        store_ptr(obj, i64(OFF_LOCK), ptr[void](0))
    PyObject_GC_Del(obj)


@compile
def initial_vectorcall(
    obj: ptr[void],
    args: ptr[ptr[void]],
    nargsf: i64,
    kwnames: ptr[void],
) -> ptr[void]:
    lock_enter(obj)
    adapter: ptr[void] = vectorcall_ptr(obj)
    if i64(adapter) != i64(ptr[void](initial_vectorcall)):
        lock_leave(obj)
        return call_vectorcall(as_vectorcall(adapter), obj, args, nargsf, kwnames)
    state: i32 = load_state(obj)
    if state == i32(1):
        lock_leave(obj)
        waited: ptr[void] = call_noargs_attr(obj, "_pythoc_wait")
        if i64(waited) == i64(0):
            return ptr[void](0)
        release(waited)
        adapter = vectorcall_ptr(obj)
        if i64(adapter) == i64(ptr[void](initial_vectorcall)):
            PyErr_SetString(
                PyExc_RuntimeError,
                "PythoC callable resolve did not bind an adapter",
            )
            return ptr[void](0)
        return call_vectorcall(
            as_vectorcall(adapter), obj, args, nargsf, kwnames,
        )
    if state == i32(3):
        lock_leave(obj)
        failed: ptr[void] = call_noargs_attr(obj, "_pythoc_raise_error")
        release(failed)
        return ptr[void](0)
    store_state(obj, i32(1))
    lock_leave(obj)

    resolved: ptr[void] = call_noargs_attr(obj, "_pythoc_resolve")
    if i64(resolved) == i64(0):
        lock_enter(obj)
        store_state(obj, i32(3))
        lock_leave(obj)
        return ptr[void](0)
    release(resolved)

    adapter = vectorcall_ptr(obj)
    if i64(adapter) == i64(ptr[void](initial_vectorcall)):
        lock_enter(obj)
        store_state(obj, i32(3))
        lock_leave(obj)
        PyErr_SetString(
            PyExc_RuntimeError,
            "PythoC callable resolve did not bind an adapter",
        )
        return ptr[void](0)
    return call_vectorcall(as_vectorcall(adapter), obj, args, nargsf, kwnames)


@compile
def slow_vectorcall(
    obj: ptr[void],
    args: ptr[ptr[void]],
    nargsf: i64,
    kwnames: ptr[void],
) -> ptr[void]:
    nargs: i64 = nargs_of(nargsf)
    nkw: i64 = i64(0)
    if i64(kwnames) != i64(0):
        nkw = PyTuple_Size(kwnames)
    tup: ptr[void] = PyTuple_New(nargs)
    if i64(tup) == i64(0):
        return ptr[void](0)
    index: i64 = i64(0)
    while index < nargs:
        Py_IncRef(args[index])
        if PyTuple_SetItem(tup, index, args[index]) != i32(0):
            Py_DecRef(args[index])
            release(tup)
            return ptr[void](0)
        index = index + i64(1)
    kwargs: ptr[void] = ptr[void](0)
    if nkw > i64(0):
        kwargs = PyDict_New()
        if i64(kwargs) == i64(0):
            release(tup)
            return ptr[void](0)
        index = i64(0)
        while index < nkw:
            if PyDict_SetItem(
                kwargs, PyTuple_GetItem(kwnames, index), args[nargs + index],
            ) != i32(0):
                release(kwargs)
                release(tup)
                return ptr[void](0)
            index = index + i64(1)
    result: ptr[void] = PyObject_Call(load_ptr(obj, i64(OFF_SLOW)), tup, kwargs)
    release(tup)
    release(kwargs)
    return result


@compile
def callable_repr(obj: ptr[void]) -> ptr[void]:
    name: ptr[void] = PyObject_GetAttrString(obj, "__name__")
    if i64(name) == i64(0):
        PyErr_Clear()
        return PyUnicode_FromString("<PythoC function>")
    left: ptr[void] = PyUnicode_FromString("<PythoC function ")
    if i64(left) == i64(0):
        release(name)
        return ptr[void](0)
    mid: ptr[void] = PyUnicode_Concat(left, name)
    release(left)
    release(name)
    if i64(mid) == i64(0):
        return ptr[void](0)
    right: ptr[void] = PyUnicode_FromString(">")
    if i64(right) == i64(0):
        release(mid)
        return ptr[void](0)
    text: ptr[void] = PyUnicode_Concat(mid, right)
    release(mid)
    release(right)
    return text


@compile
def callable_new(typ: ptr[void], args: ptr[void], kwds: ptr[void]) -> ptr[void]:
    obj: ptr[void] = PyType_GenericAlloc(typ, i64(0))
    if i64(obj) == i64(0):
        return ptr[void](0)
    store_ptr(obj, i64(OFF_VECTORCALL), ptr[void](initial_vectorcall))
    store_state(obj, i32(0))
    lock: ptr[void] = PyThread_allocate_lock()
    if i64(lock) == i64(0):
        PyErr_NoMemory()
        Py_DecRef(obj)
        return ptr[void](0)
    store_ptr(obj, i64(OFF_LOCK), lock)
    return obj


@compile
def callable_init(obj: ptr[void], args: ptr[void], kwds: ptr[void]) -> i32:
    if PyTuple_Size(args) != i64(1):
        PyErr_SetString(
            PyExc_TypeError,
            "PythoCCallable() takes exactly one argument",
        )
        return i32(-1)
    wrapped: ptr[void] = PyTuple_GetItem(args, i64(0))
    if i64(wrapped) == i64(0):
        return i32(-1)
    Py_IncRef(wrapped)
    previous: ptr[void] = load_ptr(obj, i64(OFF_WRAPPED))
    store_ptr(obj, i64(OFF_WRAPPED), wrapped)
    release(previous)
    return i32(0)


@compile
def callable_get_dict(obj: ptr[void], closure: ptr[void]) -> ptr[void]:
    dct: ptr[void] = load_ptr(obj, i64(OFF_DICT))
    if i64(dct) == i64(0):
        dct = PyDict_New()
        if i64(dct) == i64(0):
            return ptr[void](0)
        store_ptr(obj, i64(OFF_DICT), dct)
    Py_IncRef(dct)
    return dct


@compile
def callable_bind_adapter(obj: ptr[void], arg: ptr[void]) -> ptr[void]:
    addr: u64 = PyLong_AsUnsignedLongLong(arg)
    if i64(PyErr_Occurred()) != i64(0):
        return ptr[void](0)
    if addr == u64(0):
        PyErr_SetString(PyExc_ValueError, "null PythoC adapter address")
        return ptr[void](0)
    lock_enter(obj)
    store_ptr(obj, i64(OFF_VECTORCALL), ptr[void](i64(addr)))
    store_state(obj, i32(2))
    lock_leave(obj)
    result: ptr[void] = none_ptr()
    Py_IncRef(result)
    return result


@compile
def callable_bind_slow(obj: ptr[void], impl: ptr[void]) -> ptr[void]:
    if PyCallable_Check(impl) == i32(0):
        PyErr_SetString(PyExc_TypeError, "slow implementation must be callable")
        return ptr[void](0)
    Py_IncRef(impl)
    lock_enter(obj)
    previous: ptr[void] = load_ptr(obj, i64(OFF_SLOW))
    store_ptr(obj, i64(OFF_SLOW), impl)
    store_ptr(obj, i64(OFF_VECTORCALL), ptr[void](slow_vectorcall))
    store_state(obj, i32(2))
    lock_leave(obj)
    release(previous)
    result: ptr[void] = none_ptr()
    Py_IncRef(result)
    return result


@compile
def callable_is_fast_bound(obj: ptr[void], unused: ptr[void]) -> ptr[void]:
    adapter: ptr[void] = vectorcall_ptr(obj)
    if i64(adapter) != i64(ptr[void](initial_vectorcall)):
        if i64(adapter) != i64(ptr[void](slow_vectorcall)):
            return PyBool_FromLong(i32(1))
    return PyBool_FromLong(i32(0))


@compile
def callable_tp_call(obj: ptr[void], args: ptr[void], kwargs: ptr[void]) -> ptr[void]:
    npos: i64 = PyTuple_Size(args)
    if npos < i64(0):
        return ptr[void](0)
    nkw: i64 = i64(0)
    if i64(kwargs) != i64(0):
        nkw = PyDict_Size(kwargs)
        if nkw < i64(0):
            return ptr[void](0)
    total: i64 = npos + nkw
    raw: ptr[void] = ptr[void](0)
    if total > i64(0):
        raw = PyMem_Malloc(u64(total) * u64(8))
        if i64(raw) == i64(0):
            PyErr_NoMemory()
            return ptr[void](0)
    arr: ptr[ptr[void]] = ptr[ptr[void]](raw)
    index: i64 = i64(0)
    while index < npos:
        item: ptr[void] = PyTuple_GetItem(args, index)
        if i64(item) == i64(0):
            PyMem_Free(raw)
            return ptr[void](0)
        arr[index] = item
        index = index + i64(1)
    names: ptr[void] = ptr[void](0)
    if nkw > i64(0):
        names = PyTuple_New(nkw)
        if i64(names) == i64(0):
            PyMem_Free(raw)
            return ptr[void](0)
        pos: i64 = i64(0)
        key: ptr[void] = ptr[void](0)
        val: ptr[void] = ptr[void](0)
        filled: i64 = i64(0)
        while PyDict_Next(kwargs, ptr(pos), ptr(key), ptr(val)) != i32(0):
            Py_IncRef(key)
            if PyTuple_SetItem(names, filled, key) != i32(0):
                Py_DecRef(key)
                release(names)
                PyMem_Free(raw)
                return ptr[void](0)
            arr[npos + filled] = val
            filled = filled + i64(1)
    result: ptr[void] = call_vectorcall(
        as_vectorcall(vectorcall_ptr(obj)),
        obj,
        arr,
        npos,
        names,
    )
    release(names)
    if total > i64(0):
        PyMem_Free(raw)
    return result


@compile
def load_flags(base: ptr[void]) -> i64:
    if TP_FLAGS_SIZE == 8:
        return load_i64(base, i64(OFF_TP_FLAGS))
    return i64(load_i32(base, i64(OFF_TP_FLAGS)))


@compile
def store_flags(base: ptr[void], value: i64) -> void:
    if TP_FLAGS_SIZE == 8:
        store_i64(base, i64(OFF_TP_FLAGS), value)
        return
    store_i32(base, i64(OFF_TP_FLAGS), i32(value))


@compile
def install_type_offsets(typ: ptr[void]) -> void:
    store_i64(typ, i64(OFF_TP_DICTS), i64(OFF_DICT))
    if HAVE_VECTORCALL != 0:
        current: i64 = load_flags(typ)
        store_flags(typ, current | i64(FLAG_BITS))
        store_i64(typ, i64(OFF_TP_VECTORCALL), i64(OFF_VECTORCALL))


@compile
def PyInit__callable() -> ptr[void]:
    methods: static[array[MethodDef, 4]]
    getsets: static[array[GetSetDef, 2]]
    slots: static[array[TypeSlot, 11]]
    spec: static[TypeSpec]
    raw: static[array[i64, MODULE_WORDS]]

    slots[0][0] = i32(SLOT_DEALLOC)
    slots[0][1] = ptr[void](callable_dealloc)
    slots[1][0] = i32(SLOT_TRAVERSE)
    slots[1][1] = ptr[void](callable_traverse)
    slots[2][0] = i32(SLOT_CLEAR)
    slots[2][1] = ptr[void](callable_clear)
    slots[3][0] = i32(SLOT_REPR)
    slots[3][1] = ptr[void](callable_repr)
    slots[4][0] = i32(SLOT_CALL)
    slots[4][1] = ptr[void](callable_tp_call)
    slots[5][0] = i32(SLOT_METHODS)
    slots[5][1] = ptr[void](ptr(methods))
    slots[6][0] = i32(SLOT_GETSET)
    slots[6][1] = ptr[void](ptr(getsets))
    slots[7][0] = i32(SLOT_INIT)
    slots[7][1] = ptr[void](callable_init)
    slots[8][0] = i32(SLOT_NEW)
    slots[8][1] = ptr[void](callable_new)
    slots[9][0] = i32(SLOT_DOC)
    slots[9][1] = ptr[void]("Native callable for a PythoC function.")
    slots[10][0] = i32(0)
    slots[10][1] = ptr[void](0)

    methods[0][0] = "bind_adapter"
    methods[0][1] = ptr[void](callable_bind_adapter)
    methods[0][2] = i32(METH_O)
    methods[0][3] = "Install a generated vectorcall adapter address."
    methods[1][0] = "bind_slow"
    methods[1][1] = ptr[void](callable_bind_slow)
    methods[1][2] = i32(METH_O)
    methods[1][3] = "Install a Python fallback for signatures without a fast adapter."
    methods[2][0] = "is_fast_bound"
    methods[2][1] = ptr[void](callable_is_fast_bound)
    methods[2][2] = i32(METH_NOARGS)
    methods[2][3] = "Return whether steady-state calls enter a generated adapter."
    methods[3][0] = ptr[i8](0)
    methods[3][1] = ptr[void](0)
    methods[3][2] = i32(0)
    methods[3][3] = ptr[i8](0)

    getsets[0][0] = "__dict__"
    getsets[0][1] = ptr[void](callable_get_dict)
    getsets[0][2] = ptr[void](0)
    getsets[0][3] = ptr[i8](0)
    getsets[0][4] = ptr[void](0)
    getsets[1][0] = ptr[i8](0)
    getsets[1][1] = ptr[void](0)
    getsets[1][2] = ptr[void](0)
    getsets[1][3] = ptr[i8](0)
    getsets[1][4] = ptr[void](0)

    spec[0] = "pythoc._callable.PythoCCallable"
    spec[1] = i32(BASICSIZE)
    spec[2] = i32(0)
    spec[3] = u32(FLAG_BITS)
    spec[4] = ptr[void](ptr(slots))

    typ: ptr[void] = PyType_FromSpec(ptr[void](ptr(spec)))
    if i64(typ) == i64(0):
        return ptr[void](0)
    install_type_offsets(typ)

    base: ptr[i8] = ptr[i8](ptr(raw))
    zero_bytes(base, i64(MODULE_BYTES))
    store_ptr(ptr[void](base), i64(OFF_M_NAME), ptr[void]("pythoc._callable"))
    store_i64(ptr[void](base), i64(OFF_M_SIZE), i64(-1))
    if i64(PyModuleDef_Init(ptr[void](base))) == i64(0):
        release(typ)
        return ptr[void](0)
    module: ptr[void] = PyModule_Create2(ptr[void](base), i32(API_VERSION))
    if i64(module) == i64(0):
        release(typ)
        return ptr[void](0)
    if PyModule_AddObject(module, "PythoCCallable", typ) != i32(0):
        release(typ)
        release(module)
        return ptr[void](0)
    return module
