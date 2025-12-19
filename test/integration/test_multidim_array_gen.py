#!/usr/bin/env python3
"""
Test dynamic generation of compile functions to initialize multi-dimensional arrays.

Tests:
- Generate a function that creates static multi-dimensional arrays
- Initialize array elements with sum of indices: xs[i, j, k, ...] = i + j + k + ...
"""
from pythoc import i32, compile, seq, static, array, ptr, bool
from pythoc.libc.stdio import printf

import itertools

def iter_comb(*args):
 # args /
    for combination in itertools.product(*args):
        yield combination


def iter_range_comb(shape):
    shape_range = [range(dim_size) for dim_size in shape]
    for comb in iter_comb(*shape_range):
        yield comb



def tuple_to_struct_code(comb):
    """Generate code string to create struct from tuple at compile time."""
    from pythoc import struct, pyconst
    from pythoc.builtin_entities.python_type import PythonType
    
    # Create struct type with pyconst fields
    field_specs = []
    for elem in comb:
        elem_type = PythonType.wrap(elem, is_constant=True)
        field_specs.append((None, elem_type))
    
    struct_type = struct.handle_type_subscript(tuple(field_specs))
    return struct_type


def generate_multidim_array_initializer(shape):
    # for comb in iter_range_comb(shape):
    #     print(comb)
    shape_tuple = tuple(shape)
    # Use tuple unpacking compatible with Python 3.10+
    ArrayType = array[(i32,) + shape_tuple]
    if len(shape_tuple) == 1:
        DecayPtrType = ptr[i32]
    else:
        DecayPtrType = ptr[array[(i32,) + shape_tuple[1:]]]
    
    @compile(suffix=shape_tuple)
    def init_multidim_array() -> DecayPtrType:
        is_init: static[bool] = False
        xs: static[ArrayType]
        if is_init:
            return xs

        # Generate assignment statements for each combination
        for comb in iter_range_comb(shape):
            # Create struct type from tuple
            idx_type = tuple_to_struct_code(comb)
            # Create struct instance (zero-sized for pyconst fields)
            xs[idx_type()] = i32(sum(comb))
        is_init = True
        return xs
    return init_multidim_array

func_5 = generate_multidim_array_initializer([5])
func_34 = generate_multidim_array_initializer([3, 4])
func_357 = generate_multidim_array_initializer([2, 3, 4])
func_3579 = generate_multidim_array_initializer([2, 3, 4, 5])

# Test with various dimension configurations
@compile
def test_1d() -> i32:
    xs = func_5()
    sum: i32 = 0
    for i in seq(5):
        sum += xs[i]
    return sum


@compile
def test_2d() -> i32:
    xs = func_34()
    sum: i32 = 0
    for i in seq(3):
        for j in seq(4):
            sum += xs[i, j]
    return sum

@compile
def test_3d() -> i32:
    xs = func_357()
    sum: i32 = 0
    for i in seq(2):
        for j in seq(3):
            for k in seq(4):
                sum += xs[i, j, k]
    return sum


@compile
def test_4d() -> i32:
    xs = func_3579()
    sum: i32 = 0
    for i in seq(2):
        for j in seq(3):
            for k in seq(4):
                for l in seq(5):
                    sum += xs[i, j, k, l]
    return sum


if __name__ == "__main__":
    print("test_1d", test_1d())
    print("test_2d", test_2d())
    print("test_3d", test_3d())
    print("test_4d", test_4d())
    print("\n=== All tests completed ===")
