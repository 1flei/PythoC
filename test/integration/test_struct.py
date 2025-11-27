from pythoc import i32, i64, f64, bool, ptr, compile, struct, array
from pythoc.libc.stdio import printf

@compile
def test_struct() -> i32:
    """Struct with different field naming styles"""
    # Named fields
    named: struct[a: i32, b: f64, c: i32] = (42, 3.14, 1)
    a: i32 = named.a
    b: f64 = named.b
    c: i32 = named.c

    # Unnamed fields - must use same type
    unnamed: struct[i32, f64, i32] = (42, 3.14, 1)
    aa = unnamed[0]
    bb = unnamed[1]
    cc = unnamed[2]

    # Mixed fields - must use same type
    mixed: struct[i32, b: f64, i32] = (42, 3.14, 1)
    aaa = mixed[0]
    bbb = mixed[1]
    # Should be able to access by name
    bbb2 = mixed.b
    ccc = mixed[2]
    
    printf("Named types: (%d, %f, %d)\n", a, b, c)
    printf("Unnamed types: (%d, %f, %d)\n", aa, bb, cc)
    printf("Mixed types: (%d, %f, %f, %d)\n", aaa, bbb, bbb2, ccc)
    return 0

named_struct = struct["a": i32, "b": f64, "c": i32]
unnamed_struct = struct[i32, f64, i32]
mixed_struct = struct[i32, "b": f64, i32]

@compile
def test_struct_global() -> i32:
    """Struct with different field naming styles"""
    # Named fields
    named: named_struct = (42, 3.14, 1)
    a: i32 = named.a
    b: f64 = named.b
    c: i32 = named.c

    # Unnamed fields - must use same type
    unnamed: unnamed_struct = (42, 3.14, 1)
    aa = unnamed[0]
    bb = unnamed[1]
    cc = unnamed[2]

    # Mixed fields - must use same type
    mixed: mixed_struct = (42, 3.14, 1)
    aaa = mixed[0]
    bbb = mixed[1]
    # Should be able to access by name
    bbb2 = mixed.b
    ccc = mixed[2]
    
    printf("Named types: (%d, %f, %d)\n", a, b, c)
    printf("Unnamed types: (%d, %f, %d)\n", aa, bb, cc)
    printf("Mixed types: (%d, %f, %f, %d)\n", aaa, bbb, bbb2, ccc)
    return 0

@compile
def test_struct_string() -> i32:
    """Struct with different field naming styles"""
    # Named fields
    named: struct["a": i32, "b": f64, "c": i32] = (42, 3.14, 1)
    a: i32 = named.a
    b: f64 = named.b
    c: i32 = named.c

    # Unnamed fields - must use same type
    unnamed: struct[i32, f64, i32] = (42, 3.14, 1)
    aa = unnamed[0]
    bb = unnamed[1]
    cc = unnamed[2]

    # Mixed fields - must use same type
    mixed: struct[i32, "b": f64, i32] = (42, 3.14, 1)
    aaa = mixed[0]
    bbb = mixed[1]
    # Should be able to access by name
    bbb2 = mixed.b
    ccc = mixed[2]
    
    printf("Named types: (%d, %f, %d)\n", a, b, c)
    printf("Unnamed types: (%d, %f, %d)\n", aa, bb, cc)
    printf("Mixed types: (%d, %f, %f, %d)\n", aaa, bbb, bbb2, ccc)
    return 0

@compile
def test_basic_struct() -> i32:
    """Basic struct usage with type annotation"""
    # Declaration and initialization
    point: struct[i32, i32] = (10, 20)
    
    # Access by index
    x: i32 = point[0]
    y: i32 = point[1]
    
    printf("Basic struct: point=(%d, %d)\n", x, y)
    return 0

@compile
def test_mixed_types() -> i32:
    """struct with different types"""
    mixed: struct[i32, f64, i32] = (42, 3.14, 1)
    a: i32 = mixed[0]
    b: f64 = mixed[1]
    c: i32 = mixed[2]
    
    printf("Mixed types: (%d, %f, %d)\n", a, b, c)
    return 0

@compile
def test_struct_return() -> struct[i32, i32]:
    """Function returning struct"""
    result: struct[i32, i32] = (100, 200)
    return result

@compile
def struct_parameter(t: struct[i32, i32]) -> i32:
    """Pass struct as function parameter"""
    sum: i32 = t[0] + t[1]
    printf("struct parameter sum: %d\n", sum)
    return sum

@compile
def test_nested_struct() -> i32:
    """Nested structs"""
    nested: struct[i32, struct[f64, f64]] = (1, (2.0, 3.0))
    
    outer: i32 = nested[0]
    inner: struct[f64, f64] = nested[1]
    x: f64 = inner[0]
    y: f64 = inner[1]
    
    printf("Nested: outer=%d, inner=(%f, %f)\n", outer, x, y)
    return 0

@compile
def test_struct_modification() -> i32:
    """Modify struct elements"""
    point: struct[i32, i32] = (10, 20)
    
    # Modify elements
    point[0] = 30
    point[1] = 40
    
    printf("Modified struct: (%d, %d)\n", point[0], point[1])
    return 0

@compile
def test_struct_with_pointer() -> i32:
    """struct containing pointers"""
    arr: array[i32, 3] = [1, 2, 3]
    p: ptr[i32] = arr
    
    # struct with pointer
    data: struct[ptr[i32], i32] = (p, 3)
    
    ptr_val: ptr[i32] = data[0]
    len_val: i32 = data[1]
    
    printf("struct with pointer: len=%d, first=%d\n", len_val, ptr_val[0])
    return 0


@compile
def main() -> i32:
    test_struct()
    test_struct_string()
    test_struct_global()

    test_basic_struct()
    test_mixed_types()
    
    result: struct[i32, i32] = test_struct_return()
    printf("Returned struct: (%d, %d)\n", result[0], result[1])
    struct_parameter(result)
    
    test_nested_struct()
    test_struct_modification()
    test_struct_with_pointer()
    return 0

main()