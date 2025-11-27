from __future__ import annotations
from pythoc import i32, i64, f64, bool, ptr, compile, union, struct
from pythoc.libc.stdio import printf

@compile
def test_union_basic() -> i32:
    """Basic union usage - all fields share same memory"""
    # Union with named fields
    data = union[i: i32, f: f64]()
    data.i = 42
    
    # Access as integer
    i_val: i32 = data.i
    printf("Union as i32: %d\n", i_val)
    
    # Now store a float (overwrites the integer)
    data.f = 3.14
    f_val: f64 = data.f
    printf("Union as f64: %f\n", f_val)
    
    # Reading as integer now gives the bit pattern of the float
    i_val2: i32 = data.i
    printf("Union as i32 (after f64 write): %d\n", i_val2)
    
    return 0

@compile
def test_union_unnamed() -> i32:
    """Union with unnamed fields"""
    data: union[i32, f64]
    
    # Access by index
    i_val: i32 = data[0]
    printf("Union[0] as i32: %d\n", i_val)
    
    # Write to second field
    data[1] = 2.718
    f_val: f64 = data[1]
    printf("Union[1] as f64: %f\n", f_val)
    
    return 0

@compile
def test_union_mixed() -> i32:
    """Union with mixed named/unnamed fields"""
    data: union[i32, f: f64, i64]
    
    # Access first field by index
    i_val: i32 = data[0]
    printf("Mixed union[0]: %d\n", i_val)
    
    # Access second field by name
    data.f = 1.414
    f_val: f64 = data.f
    printf("Mixed union.f: %f\n", f_val)
    
    # Access third field by index
    data[2] = 999
    l_val: i64 = data[2]
    printf("Mixed union[2]: %ld\n", l_val)
    
    return 0

@compile
def test_union_size() -> i32:
    """Union size is the size of largest member"""
    # i32 is 4 bytes, f64 is 8 bytes
    # Union should be 8 bytes (size of largest member)
    data: union[i32, f64]
    
    # Store a 64-bit value
    data[1] = 123.456
    
    # Read it back
    val: f64 = data[1]
    printf("Union stores largest type: %f\n", val)
    
    return 0

@compile
def test_union_in_struct() -> i32:
    """Union as a field in struct"""
    # Struct containing a union
    record: struct[id: i32, value: union[i: i32, f: f64]] = (1, union[i: i32, f: f64]())
    
    id_val: i32 = record.id
    union_val: union[i: i32, f: f64] = record.value
    i_val: i32 = union_val.i
    
    printf("Struct with union: id=%d, value.i=%d\n", id_val, i_val)
    
    # Change union to float
    union_val.f = 99.9
    f_val: f64 = union_val.f
    printf("After change: value.f=%f\n", f_val)
    
    return 0

@compile
def main() -> i32:
    printf("=== Union Tests ===\n")
    
    test_union_basic()
    printf("\n")
    
    test_union_unnamed()
    printf("\n")
    
    test_union_mixed()
    printf("\n")
    
    test_union_size()
    printf("\n")
    
    test_union_in_struct()
    
    return 0

main()
