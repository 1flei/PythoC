"""CPython C-API declarations for the Python call trampoline.

Same arrangement pcpy uses: the symbols are already in the running
interpreter, so the declarations do not add a -lpython dependency.
``lib=''`` keeps them out of the global link registry. The adapter
object resolves them from the process when it is dlopened.
"""

from pythoc import f64, i8, i32, i64, ptr, u64, void
from pythoc.decorators import extern


@extern(lib='')
def PyLong_AsLongLong(o: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyLong_AsUnsignedLongLong(o: ptr[void]) -> u64:
    pass


@extern(lib='')
def PyLong_FromLongLong(v: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyLong_FromUnsignedLongLong(v: u64) -> ptr[void]:
    pass


@extern(lib='')
def PyLong_AsDouble(o: ptr[void]) -> f64:
    pass


@extern(lib='')
def PyFloat_AsDouble(o: ptr[void]) -> f64:
    pass


@extern(lib='')
def PyFloat_FromDouble(v: f64) -> ptr[void]:
    pass


@extern(lib='')
def Py_IncRef(o: ptr[void]) -> void:
    pass


@extern(lib='')
def Py_DecRef(o: ptr[void]) -> void:
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
def PyObject_GetBuffer(obj: ptr[void], view: ptr[void], flags: i32) -> i32:
    pass


@extern(lib='')
def PyBuffer_Release(view: ptr[void]) -> void:
    pass


@extern(lib='')
def PyType_GenericAlloc(tp: ptr[void], nitems: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyTuple_Size(t: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyTuple_GetItem(t: ptr[void], i: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyTuple_New(size: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyTuple_SetItem(t: ptr[void], i: i64, v: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyList_GetItem(lst: ptr[void], i: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyList_Size(lst: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyUnicode_FromString(s: ptr[i8]) -> ptr[void]:
    pass


@extern(lib='')
def PyDict_New() -> ptr[void]:
    pass


@extern(lib='')
def PyDict_SetItemString(d: ptr[void], key: ptr[i8], v: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyDict_GetItemString(d: ptr[void], key: ptr[i8]) -> ptr[void]:
    pass


@extern(lib='')
def PyObject_GetAttrString(o: ptr[void], name: ptr[i8]) -> ptr[void]:
    pass


@extern(lib='')
def PyUnicode_CompareWithASCIIString(left: ptr[void], right: ptr[i8]) -> i32:
    pass


@extern(lib='')
def PyBytes_AsString(o: ptr[void]) -> ptr[i8]:
    pass


@extern(lib='')
def PyBytes_Size(o: ptr[void]) -> i64:
    pass


@extern(lib='')
def PyBytes_FromStringAndSize(s: ptr[i8], n: i64) -> ptr[void]:
    pass


@extern(lib='')
def PyByteArray_AsString(o: ptr[void]) -> ptr[i8]:
    pass


@extern(lib='')
def PyByteArray_Size(o: ptr[void]) -> i64:
    pass


@extern(lib='')
def _PyLong_AsByteArray(
    v: ptr[void],
    dest: ptr[i8],
    n: u64,
    little_endian: i32,
    is_signed: i32,
) -> i32:
    pass


@extern(lib='')
def _PyLong_FromByteArray(
    bytes: ptr[i8],
    n: u64,
    little_endian: i32,
    is_signed: i32,
) -> ptr[void]:
    pass


@extern(lib='')
def memcpy(dest: ptr[i8], src: ptr[i8], n: u64) -> ptr[i8]:
    pass


@extern(lib='')
def PyModule_New(name: ptr[i8]) -> ptr[void]:
    pass


@extern(lib='')
def PyCFunction_New(ml: ptr[void], self_: ptr[void]) -> ptr[void]:
    pass


@extern(lib='')
def PyModule_AddObject(mod: ptr[void], name: ptr[i8], value: ptr[void]) -> i32:
    pass


@extern(lib='')
def PyModule_Create2(module: ptr[void], apiver: i32) -> ptr[void]:
    pass
