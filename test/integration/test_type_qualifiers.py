from pythoc import i32, i64, f64, ptr, array, compile, const, static, volatile, nullptr, u32
from pythoc.libc.stdio import printf
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests that expect to catch exceptions
set_raise_on_error(True)

@compile
def test_const_basic() -> i32:
    """Test basic const qualifier - values cannot be reassigned"""
    x: const[i32] = 42
    printf("Const i32: %d\n", x)
    
    y: const[f64] = 3.14
    printf("Const f64: %f\n", y)
    
    # Note: Attempting to reassign x or y would cause a compile-time error
    # x = 100  # TypeError: Cannot reassign to const variable
    
    return 0

@compile
def test_const_read_multiple() -> i32:
    """Test that const variables can be read multiple times"""
    x: const[i32] = 100
    
    # Reading const values multiple times is OK
    a: i32 = x
    b: i32 = x
    c: i32 = x
    
    printf("Const read three times: %d, %d, %d\n", a, b, c)
    
    return 0

@compile
def test_volatile_basic() -> i32:
    """Test basic volatile qualifier - generates volatile load/store"""
    counter: volatile[i32] = 100
    printf("Volatile i32: %d\n", counter)
    
    # Each access generates a volatile load (prevents optimization)
    flag: volatile[i32] = 1
    val1: i32 = flag  # volatile load
    val2: i32 = flag  # another volatile load (not optimized away)
    printf("Volatile reads: %d, %d\n", val1, val2)
    
    return 0

@compile
def test_volatile_multiple_access() -> i32:
    """Test multiple volatile accesses generate separate instructions"""
    reg: volatile[i32] = 42
    
    # Each read should generate a separate volatile load
    a: i32 = reg
    b: i32 = reg
    c: i32 = reg
    
    sum: i32 = a + b + c
    printf("Sum of three volatile reads: %d\n", sum)
    
    # Each write should generate a volatile store
    reg = 10
    reg = 20
    reg = 30
    
    printf("Final volatile value: %d\n", reg)
    
    return 0

@compile
def test_volatile_modification() -> i32:
    """Test volatile variable modification"""
    status: volatile[u32] = u32(0)
    
    # Volatile store
    status = u32(0xFF)
    printf("Status after write: %u\n", status)
    
    # Multiple stores should all occur
    status = u32(1)
    status = u32(2)
    status = u32(3)
    
    printf("Status final: %u\n", status)
    
    return 0

@compile
def test_static_basic() -> i32:
    """Test basic static qualifier - static storage duration"""
    value: static[i32] = 999
    printf("Static i32: %d\n", value)
    
    # Modify static variable
    value = 1000
    printf("Modified static: %d\n", value)
    
    return 0

@compile
def test_static_counter() -> i32:
    """Test static counter that persists across calls"""
    counter: static[i32] = 0
    counter = counter + 1
    printf("Static counter: %d\n", counter)
    return counter

@compile
def test_static_persistence() -> i32:
    """Test that static variables persist across function calls"""
    printf("Calling static counter 3 times:\n")
    test_static_counter()
    test_static_counter()
    test_static_counter()
    return 0

@compile
def test_const_pointer() -> i32:
    """Test const with pointer types"""
    # Const pointer (pointer itself is const)
    p: const[ptr[i32]] = nullptr
    
    if p == nullptr:
        printf("Const pointer is null\n")
    
    return 0

@compile
def test_pointer_to_const() -> i32:
    """Test pointer to const"""
    # Pointer to const integer
    p: ptr[const[i32]] = nullptr
    
    if p == nullptr:
        printf("Pointer to const is null\n")
    
    return 0

@compile
def test_volatile_pointer() -> i32:
    """Test volatile pointer (for memory-mapped I/O)"""
    p: volatile[ptr[i32]] = nullptr
    
    if p == nullptr:
        printf("Volatile pointer is null\n")
    
    return 0

@compile
def test_const_array() -> i32:
    """Test const with array"""
    # Const array
    arr: const[array[i32, 5]] = [1, 2, 3, 4, 5]
    
    printf("Const array[0]: %d\n", arr[0])
    printf("Const array[4]: %d\n", arr[4])
    
    return 0

@compile
def test_volatile_array() -> i32:
    """Test volatile array access"""
    # Volatile array (each access generates volatile load/store)
    arr: volatile[array[i32, 3]] = [10, 20, 30]
    
    # Each array access should be volatile
    printf("Volatile array[0]: %d\n", arr[0])
    printf("Volatile array[1]: %d\n", arr[1])
    printf("Volatile array[2]: %d\n", arr[2])
    
    return 0

@compile
def test_mixed_qualifiers() -> i32:
    """Test mixing different qualifiers"""
    # Static and const
    x: static[const[i32]] = 100
    printf("Static const: %d\n", x)
    
    # Static and volatile
    y: static[volatile[i32]] = 200
    printf("Static volatile: %d\n", y)
    y = 250
    printf("Modified static volatile: %d\n", y)
    
    return 0


def test_error_const_reassignment():
    """Test that reassigning const variable raises TypeError"""
    try:
        @compile
        def bad_const_reassign() -> i32:
            x: const[i32] = 42
            x = 100  # ERROR: Cannot reassign to const variable
            return x
        
        print("FAIL test_error_const_reassignment failed - should have raised RuntimeError")
        return False
    except RuntimeError as e:
        if "Cannot reassign to const variable" in str(e):
            print(f"OK test_error_const_reassignment passed: {e}")
            return True
        else:
            print(f"FAIL test_error_const_reassignment failed - wrong error: {e}")
            return False


def test_error_const_array_modification():
    """Test that modifying const array element raises TypeError"""
    try:
        @compile
        def bad_const_array_modify() -> i32:
            arr: const[array[i32, 3]] = [1, 2, 3]
            arr[0] = 99  # ERROR: Cannot modify const array
            return arr[0]
        
        print("FAIL test_error_const_array_modification failed - should have raised RuntimeError")
        return False
    except RuntimeError as e:
        if "const" in str(e).lower():
            print(f"OK test_error_const_array_modification passed: {e}")
            return True
        else:
            print(f"FAIL test_error_const_array_modification failed - wrong error: {e}")
            return False


def test_error_const_pointer_reassignment():
    """Test that reassigning const pointer raises TypeError"""
    try:
        @compile
        def bad_const_ptr_reassign() -> i32:
            x: i32 = 42
            y: i32 = 100
            p: const[ptr[i32]] = ptr[i32](x)
            p = ptr[i32](y)  # ERROR: Cannot reassign const pointer
            return 0
        
        print("FAIL test_error_const_pointer_reassignment failed - should have raised RuntimeError")
        return False
    except RuntimeError as e:
        if "Cannot reassign to const variable" in str(e):
            print(f"OK test_error_const_pointer_reassignment passed: {e}")
            return True
        else:
            print(f"FAIL test_error_const_pointer_reassignment failed - wrong error: {e}")
            return False


def test_error_static_const_modification():
    """Test that modifying static const variable raises TypeError"""
    try:
        @compile
        def bad_static_const_modify() -> i32:
            x: static[const[i32]] = 100
            x = 200  # ERROR: Cannot modify static const
            return x
        
        print("FAIL test_error_static_const_modification failed - should have raised RuntimeError")
        return False
    except RuntimeError as e:
        if "Cannot reassign to const variable" in str(e):
            print(f"OK test_error_static_const_modification passed: {e}")
            return True
        else:
            print(f"FAIL test_error_static_const_modification failed - wrong error: {e}")
            return False


def test_error_const_in_loop():
    """Test that modifying const variable in loop raises TypeError"""
    try:
        @compile
        def bad_const_in_loop() -> i32:
            x: const[i32] = 0
            i: i32 = 0
            while i < 10:
                x = x + 1  # ERROR: Cannot modify const
                i = i + 1
            return x
        
        print("FAIL test_error_const_in_loop failed - should have raised RuntimeError")
        return False
    except RuntimeError as e:
        if "Cannot reassign to const variable" in str(e):
            print(f"OK test_error_const_in_loop passed: {e}")
            return True
        else:
            print(f"FAIL test_error_const_in_loop failed - wrong error: {e}")
            return False

@compile
def main() -> i32:
    printf("=== Type Qualifiers Comprehensive Test ===\n\n")
    
    printf("--- Basic const ---\n")
    test_const_basic()
    printf("\n")
    
    printf("--- Const multiple reads ---\n")
    test_const_read_multiple()
    printf("\n")
    
    printf("--- Basic volatile ---\n")
    test_volatile_basic()
    printf("\n")
    
    printf("--- Volatile multiple access ---\n")
    test_volatile_multiple_access()
    printf("\n")
    
    printf("--- Volatile modification ---\n")
    test_volatile_modification()
    printf("\n")
    
    printf("--- Basic static ---\n")
    test_static_basic()
    printf("\n")
    
    printf("--- Static persistence ---\n")
    test_static_persistence()
    printf("\n")
    
    printf("--- Const pointer ---\n")
    test_const_pointer()
    printf("\n")
    
    printf("--- Pointer to const ---\n")
    test_pointer_to_const()
    printf("\n")
    
    printf("--- Volatile pointer ---\n")
    test_volatile_pointer()
    printf("\n")
    
    printf("--- Const array ---\n")
    test_const_array()
    printf("\n")
    
    printf("--- Volatile array ---\n")
    test_volatile_array()
    printf("\n")
    
    printf("--- Mixed qualifiers ---\n")
    test_mixed_qualifiers()
    printf("\n")
    
    printf("=== All Tests Complete ===\n")
    
    return 0


def run_error_tests():
    """Run all error tests (Python side)"""
    print("\n=== Const Error Tests ===\n")
    
    print("--- Test const reassignment error ---")
    test_error_const_reassignment()
    print()
    
    print("--- Test const array modification error ---")
    test_error_const_array_modification()
    print()
    
    print("--- Test const pointer reassignment error ---")
    test_error_const_pointer_reassignment()
    print()
    
    print("--- Test static const modification error ---")
    test_error_static_const_modification()
    print()
    
    print("--- Test const in loop error ---")
    test_error_const_in_loop()
    print()
    
    print("=== Error Tests Complete ===\n")


# Run positive tests
main()

# Run error tests
run_error_tests()
