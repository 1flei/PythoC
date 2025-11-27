from __future__ import annotations
from pythoc import *
from pythoc.libc.stdlib import malloc, free, realloc
from pythoc.libc.string import memset, memcpy

def Vector(element_type, inline_capacity = 0, size_type = u64):
    """
    Factory that generates a small-vector type specialized by element type and
    inline capacity, plus a set of C-style functions operating on it.

    Memory layout optimization:
    - size: current number of elements
    - storage: union of inline_buffer and heap metadata
      * When size < inline_capacity: use inline_buffer
      * When size >= inline_capacity: use heap (capacity + heap_buf)

    Returns a VectorApi with `.type` for the struct and functions like `.init`.
    """
    if not isinstance(inline_capacity, int) or inline_capacity < 0:
        raise TypeError("inline_capacity must be a non-negative integer")

    # Use tuple suffix for automatic deduplication
    # compile() will normalize this to a string like "int_0_u64"
    type_suffix = (element_type, inline_capacity, size_type)

    # Define vector struct with union storage
    @compile(suffix=type_suffix)
    class _HeapData:
        capacity: size_type
        heap_buf: ptr[element_type]
    
    @compile(suffix=type_suffix)
    class _Vector:
        size: size_type
        storage: union[heap_data: _HeapData, inline_buffer: array[element_type, inline_capacity]]

    # Define C-style functions bound to this _Vector type
    @compile(suffix=type_suffix)
    def vector_init(v: ptr[_Vector]) -> None:
        memset(v, 0, sizeof(_Vector))

    @compile(suffix=type_suffix)
    def vector_destroy(v: ptr[_Vector]) -> None:
        if v.size >= inline_capacity:
            # Free heap storage
            free(v.storage.heap_data.heap_buf)
        v.size = 0

    @compile(suffix=type_suffix)
    def vector_size(v: ptr[_Vector]) -> size_type:
        return v.size

    @compile(suffix=type_suffix)
    def vector_capacity(v: ptr[_Vector]) -> size_type:
        if v.size < inline_capacity:
            return inline_capacity
        else:
            return v.storage.heap_data.capacity

    @compile(suffix=type_suffix)
    def vector_get(v: ptr[_Vector], index: size_type) -> element_type:
        if v.size < inline_capacity:
            return v.storage.inline_buffer[index]
        else:
            return v.storage.heap_data.heap_buf[index]

    @compile(suffix=type_suffix)
    def vector_set(v: ptr[_Vector], index: size_type, value: element_type):
        if v.size < inline_capacity:
            v.storage.inline_buffer[index] = value
        else:
            v.storage.heap_data.heap_buf[index] = value

    @compile(suffix=type_suffix)
    def vector_push_back(v: ptr[_Vector], value: element_type):
        current_cap: size_type = vector_capacity(v)
        
        if v.size == inline_capacity:
            # Transition from inline to heap
            # new_capacity: size_type = inline_capacity * 2 if inline_capacity > 0 else 4
            new_capacity: size_type = 4
            if inline_capacity > 0:
                new_capacity = inline_capacity * 2
            new_heap: ptr[element_type] = malloc(new_capacity * sizeof(element_type))
            memcpy(new_heap, ptr(v.storage.inline_buffer[0]), v.size * sizeof(element_type))
            v.storage.heap_data.capacity = new_capacity
            v.storage.heap_data.heap_buf = new_heap
        elif v.size >= current_cap:
            # Already in heap mode, reallocate
            # old_buf: ptr[i8] = ptr[i8](v.storage.heap_data.heap_buf)
            new_capacity: size_type = 4
            if current_cap > 0:
                new_capacity = current_cap * 2
            new_mem_i8: ptr[i8] = realloc(v.storage.heap_data.heap_buf, new_capacity * sizeof(element_type))
            v.storage.heap_data.heap_buf = ptr[element_type](new_mem_i8)
            v.storage.heap_data.capacity = new_capacity
        # Write element
        vector_set(v, v.size, value)
        v.size += 1

    @compile(suffix=type_suffix)
    def vector_pop_back(v: ptr[_Vector]) -> None:
        if v.size > 0:
            v.size = v.size - 1

    # Return an API class containing functions and the Vector type itself
    class api:
        init = vector_init
        destroy = vector_destroy
        size = vector_size
        capacity = vector_capacity
        get = vector_get
        set = vector_set
        push_back = vector_push_back
        pop_back = vector_pop_back
        type = _Vector
        
        @classmethod
        def get_name(cls):
            """Return a unique name based on the vector's element type and capacity"""
            elem_name = element_type.get_name() if hasattr(element_type, 'get_name') else str(element_type)
            size_name = size_type.get_name() if hasattr(size_type, 'get_name') else str(size_type)
            return f'Vector_{elem_name}_{inline_capacity}_{size_name}'

    return api
