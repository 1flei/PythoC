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

    Inline/heap discrimination uses a smallvec-style dual-purpose word: the
    ``size_or_cap`` field holds the element count while inline (always
    ``<= inline_capacity``) and the heap capacity once spilled (always
    ``> inline_capacity``).  Spilling is one-way, so the predicate
    ``size_or_cap > inline_capacity`` stays correct no matter how far
    ``pop_back`` shrinks the element count; the spilled element count lives
    in the heap branch of the union instead.
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
        size_or_cap: size_type
        storage: union[
            heap_data: struct[size: size_type, heap_buf: ptr[element_type]],
            inline_buffer: array[element_type, inline_capacity],
        ]

        def init(v: ptr[_Vector]) -> None:
            memset(v, 0, sizeof(_Vector))

        def destroy(v: ptr[_Vector]) -> None:
            if v.size_or_cap > inline_capacity:
                free(v.storage.heap_data.heap_buf)
            v.size_or_cap = 0

        def size(v: ptr[_Vector]) -> size_type:
            if v.size_or_cap > inline_capacity:
                return v.storage.heap_data.size
            return v.size_or_cap

        def capacity(v: ptr[_Vector]) -> size_type:
            if v.size_or_cap > inline_capacity:
                return v.size_or_cap
            return size_type(inline_capacity)

        def get(v: ptr[_Vector], index: size_type) -> element_type:
            if v.size_or_cap > inline_capacity:
                return v.storage.heap_data.heap_buf[index]
            return v.storage.inline_buffer[index]

        def set(v: ptr[_Vector], index: size_type, value: element_type) -> None:
            if v.size_or_cap > inline_capacity:
                v.storage.heap_data.heap_buf[index] = value
            else:
                v.storage.inline_buffer[index] = value

        def data(v: ptr[_Vector]) -> ptr[element_type]:
            if v.size_or_cap > inline_capacity:
                return v.storage.heap_data.heap_buf
            return ptr(v.storage.inline_buffer[0])

        def push_back(v: ptr[_Vector], value: element_type) -> None:
            if v.size_or_cap > inline_capacity:
                # Spilled: grow the heap allocation if it is full.
                if v.storage.heap_data.size == v.size_or_cap:
                    grown_capacity: size_type = v.size_or_cap * 2
                    new_mem_i8: ptr[i8] = realloc(
                        v.storage.heap_data.heap_buf,
                        grown_capacity * sizeof(element_type),
                    )
                    v.storage.heap_data.heap_buf = ptr[element_type](new_mem_i8)
                    v.size_or_cap = grown_capacity
                v.storage.heap_data.heap_buf[v.storage.heap_data.size] = value
                v.storage.heap_data.size = v.storage.heap_data.size + 1
            elif v.size_or_cap == inline_capacity:
                # Inline buffer is full: spill onto the heap.
                new_capacity: size_type = 4
                if inline_capacity > 0:
                    new_capacity = inline_capacity * 2
                new_heap: ptr[element_type] = malloc(
                    new_capacity * sizeof(element_type))
                if inline_capacity > 0:
                    memcpy(new_heap, ptr(v.storage.inline_buffer[0]),
                           v.size_or_cap * sizeof(element_type))
                new_heap[v.size_or_cap] = value
                # Publish the heap branch of the union only after the inline
                # buffer has been copied out; they share storage.
                v.storage.heap_data.size = v.size_or_cap + 1
                v.storage.heap_data.heap_buf = new_heap
                v.size_or_cap = new_capacity
            else:
                v.storage.inline_buffer[v.size_or_cap] = value
                v.size_or_cap = v.size_or_cap + 1

        def pop_back(v: ptr[_Vector]) -> None:
            if v.size_or_cap > inline_capacity:
                if v.storage.heap_data.size > 0:
                    v.storage.heap_data.size = v.storage.heap_data.size - 1
            else:
                if v.size_or_cap > 0:
                    v.size_or_cap = v.size_or_cap - 1

        def set_size(v: ptr[_Vector], new_size: size_type) -> None:
            """Set the logical element count. Caller must keep it <= capacity."""
            if v.size_or_cap > inline_capacity:
                v.storage.heap_data.size = new_size
            else:
                v.size_or_cap = new_size

    return _Vector
