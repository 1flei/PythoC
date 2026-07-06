from __future__ import annotations
from pythoc import *
from pythoc.libc.stdlib import malloc, free, realloc
from pythoc.libc.string import memset, memcpy


def Vector(element_type, inline_capacity=0, size_type=u64):
    """
    Factory that generates a small-vector type specialized by element type and
    inline capacity, plus C-style methods operating on it.

    The returned object is the compiled ``_Vector`` class itself, with compiled
    helper methods declared directly inside the class body.  Callers write::

        IntVec = Vector(i32, 4)
        v: IntVec
        vp = ptr(v)
        IntVec.init(vp)
        IntVec.push_back(vp, 42)

    Methods do not receive a synthetic ``self``; the first argument is the
    pointer to the vector instance, just like the previous ``api`` pattern.
    Instance-level attribute access resolves struct fields and class-level
    access resolves class attributes (methods), so a field and a method may
    share the same name without ambiguity.
    """
    if not isinstance(inline_capacity, int) or inline_capacity < 0:
        raise TypeError("inline_capacity must be a non-negative integer")
    if not (hasattr(size_type, '_is_integer') and size_type._is_integer):
        raise TypeError(
            f"Vector: size_type must be a PythoC integer type, got {size_type}"
        )

    type_suffix = (element_type, inline_capacity, size_type)

    @compile(suffix=type_suffix)
    class _Vector:
        size: size_type
        storage: union[
            heap_data: struct[capacity: size_type, heap_buf: ptr[element_type]],
            inline_buffer: array[element_type, inline_capacity],
        ]

        def init(v: ptr[_Vector]) -> None:
            memset(v, 0, sizeof(_Vector))

        def destroy(v: ptr[_Vector]) -> None:
            if v.size > inline_capacity:
                free(v.storage.heap_data.heap_buf)
            v.size = 0

        def size(v: ptr[_Vector]) -> size_type:
            return v.size

        def capacity(v: ptr[_Vector]) -> size_type:
            if v.size <= inline_capacity:
                return size_type(inline_capacity)
            return v.storage.heap_data.capacity

        def get(v: ptr[_Vector], index: size_type) -> element_type:
            if v.size <= inline_capacity:
                return v.storage.inline_buffer[index]
            return v.storage.heap_data.heap_buf[index]

        def set(v: ptr[_Vector], index: size_type, value: element_type) -> None:
            if v.size <= inline_capacity:
                v.storage.inline_buffer[index] = value
            else:
                v.storage.heap_data.heap_buf[index] = value

        def data(v: ptr[_Vector]) -> ptr[element_type]:
            if v.size <= inline_capacity:
                return ptr(v.storage.inline_buffer[0])
            return v.storage.heap_data.heap_buf

        def push_back(v: ptr[_Vector], value: element_type) -> None:
            current_cap: size_type = _Vector.capacity(v)

            if v.size == inline_capacity:
                new_capacity: size_type = 4
                if inline_capacity > 0:
                    new_capacity = inline_capacity * 2
                new_heap: ptr[element_type] = malloc(new_capacity * sizeof(element_type))
                memcpy(new_heap, ptr(v.storage.inline_buffer[0]), v.size * sizeof(element_type))
                v.storage.heap_data.capacity = new_capacity
                v.storage.heap_data.heap_buf = new_heap
            elif v.size >= current_cap:
                new_capacity: size_type = 4
                if current_cap > 0:
                    new_capacity = current_cap * 2
                new_mem_i8: ptr[i8] = realloc(v.storage.heap_data.heap_buf, new_capacity * sizeof(element_type))
                v.storage.heap_data.heap_buf = ptr[element_type](new_mem_i8)
                v.storage.heap_data.capacity = new_capacity
            v.size += 1
            _Vector.set(v, v.size - 1, value)

        def pop_back(v: ptr[_Vector]) -> None:
            if v.size > 0:
                v.size = v.size - 1

    return _Vector
