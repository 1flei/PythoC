# C Parity: Full C Capabilities in PythoC

PythoC provides complete C runtime capabilities with Python syntax. This document covers all C-equivalent features available in PythoC.

## Table of Contents

- [Primitive Types](#primitive-types)
- [Composite Types](#composite-types)
- [Control Flow](#control-flow)
- [Operators](#operators)
- [Functions](#functions)
- [C Standard Library](#c-standard-library)
- [Differences from C](#differences-from-c)

---

## Primitive Types

### Integer Types

PythoC supports all standard integer types with explicit bit widths:

| PythoC | C Equivalent | Size | Range |
|--------|--------------|------|-------|
| `i8` | `int8_t` / `signed char` | 1 byte | -128 to 127 |
| `i16` | `int16_t` / `short` | 2 bytes | -32,768 to 32,767 |
| `i32` | `int32_t` / `int` | 4 bytes | `-2**31` to `2**31-1` |
| `i64` | `int64_t` / `long long` | 8 bytes | `-2**63` to `2**63-1` |
| `u8` | `uint8_t` / `unsigned char` | 1 byte | 0 to 255 |
| `u16` | `uint16_t` / `unsigned short` | 2 bytes | 0 to 65,535 |
| `u32` | `uint32_t` / `unsigned int` | 4 bytes | 0 to `2**32-1` |
| `u64` | `uint64_t` / `unsigned long long` | 8 bytes | 0 to `2**64-1` |

```python
from pythoc import i8, i16, i32, i64, u8, u16, u32, u64, compile

@compile
def integer_types() -> i32:
    a: i8 = 127
    b: i16 = 32767
    c: i32 = 100
    d: i64 = 1000

    ua: u8 = 255
    ub: u16 = 65535
    uc: u32 = 4294967295
    ud: u64 = 18446744073709551615

    return c
```

### Floating Point Types

| PythoC | C Equivalent | Size | Precision |
|--------|--------------|------|-----------|
| `f32` | `float` | 4 bytes | ~7 digits |
| `f64` | `double` | 8 bytes | ~15 digits |
| `f16` | `_Float16` | 2 bytes | ~3 digits |
| `bf16` | `bfloat16` | 2 bytes | ~3 digits |
| `f128` | `__float128` | 16 bytes | ~33 digits |

> **Note**: `f16`, `bf16`, and `f128` support depends on your compiler and target platform. These types may not be fully supported on all systems.

```python
from pythoc import f32, f64, compile, i32

@compile
def float_types() -> i32:
    a: f32 = f32(3.14)      # Explicit f32 conversion
    b: f64 = 2.718          # Default is f64

    threshold: f32 = f32(3.0)
    if a > threshold:
        return 1
    return 0
```

### Boolean Type

```python
from pythoc import bool, compile, i32

@compile
def boolean_type() -> i32:
    t: bool = True
    f: bool = False

    if t and not f:
        return 1
    return 0
```

### Type Conversions

PythoC supports explicit type conversions similar to C casts:

```python
from pythoc import i32, i64, f64, ptr, compile

@compile
def type_conversions() -> i32:
    # Integer to integer
    a: i8 = 100
    b: i32 = i32(a)         # Widening
    c: i16 = i16(b)         # Narrowing

    # Integer to float
    d: i32 = 42
    e: f64 = f64(d)

    # Float to integer (truncates)
    f: f64 = 3.14
    g: i32 = i32(f)         # g = 3

    # Pointer to integer and back
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    addr: i64 = i64(p)
    p2: ptr[i32] = ptr[i32](addr)

    return p2[0]
```

---

## Composite Types

### Pointers (`ptr[T]`)

Pointers work exactly like C pointers with full pointer arithmetic:

```python
from pythoc import i32, i64, ptr, nullptr, array, compile

@compile
def pointer_basics() -> i32:
    # Get pointer to variable
    x: i32 = 42
    p: ptr[i32] = ptr(x)

    # Dereference
    val: i32 = p[0]         # Same as *p in C

    # Modify through pointer
    p[0] = 100              # x is now 100

    return x

@compile
def pointer_arithmetic() -> i32:
    arr: array[i32, 5] = [10, 20, 30, 40, 50]

    # Array decays to pointer
    p: ptr[i32] = arr

    # Pointer arithmetic
    p1: ptr[i32] = p + 2    # Points to arr[2]
    p2: ptr[i32] = p1 - 1   # Points to arr[1]

    # Subscript access
    val: i32 = p[3]         # Same as arr[3]

    return val

@compile
def nullptr_check() -> i32:
    p: ptr[i32] = ptr[i32](nullptr)

    if p == ptr[i32](nullptr):
        return 1
    return 0

@compile
def pointer_to_pointer() -> i32:
    x: i32 = 42
    p: ptr[i32] = ptr(x)
    pp: ptr[ptr[i32]] = ptr(p)

    return pp[0][0]         # 42
```

### Arrays (`array[T, N]`)

Fixed-size arrays with C-compatible memory layout:

```python
from pythoc import i32, f64, ptr, array, compile

@compile
def array_1d() -> i32:
    # Declaration and initialization
    arr: array[i32, 5] = [1, 2, 3, 4, 5]

    # Partial initialization (remaining = 0)
    arr2: array[i32, 5] = [10, 20, 30]

    # Zero initialization
    zeros: array[f64, 10] = array[f64, 10]()

    # Access and modify
    x: i32 = arr[0]
    arr[1] = 42

    return arr[1]

@compile
def array_2d() -> i32:
    # 2D array (row-major like C)
    matrix: array[i32, 2, 3] = [[1, 2, 3], [4, 5, 6]]

    # Access methods
    a: i32 = matrix[0][0]   # Traditional
    b: i32 = matrix[1, 2]   # Tuple indexing (PythoC extension)

    # Modify
    matrix[1][1] = 99
    matrix[0, 2] = 88

    return matrix[1, 2]

@compile
def array_3d() -> i32:
    # 3D array
    cube: array[i32, 2, 3, 4] = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]
    ]

    val: i32 = cube[1, 2, 3]    # 24
    return val

@compile
def array_decay() -> i32:
    arr: array[i32, 5] = [10, 20, 30, 40, 50]

    # Array decays to pointer (like C)
    p: ptr[i32] = arr

    # Access through pointer
    return p[2]             # 30
```

### Structs

PythoC supports multiple struct syntaxes:

#### Anonymous Struct Syntax

```python
from pythoc import i32, f64, ptr, struct, array, compile

@compile
def struct_unnamed() -> i32:
    # Unnamed fields (access by index)
    point: struct[i32, i32] = (10, 20)
    x: i32 = point[0]
    y: i32 = point[1]

    # Modify
    point[0] = 30

    return point[0]

@compile
def struct_named() -> i32:
    # Named fields (access by name)
    point: struct[x: i32, y: i32] = (10, 20)

    # Access by name
    x: i32 = point.x
    y: i32 = point.y

    # Or by index
    x2: i32 = point[0]

    # Modify
    point.x = 30

    return point.x

@compile
def struct_mixed() -> i32:
    # Mixed named/unnamed (like C)
    data: struct[i32, name: f64, i32] = (1, 3.14, 2)

    a: i32 = data[0]
    b: f64 = data.name      # Named access
    c: i32 = data[2]

    return a
```

#### Class-based Struct Syntax

```python
from pythoc import i32, f64, ptr, compile

@compile
class Point:
    x: i32
    y: i32

@compile
class Rect:
    top_left: Point
    width: i32
    height: i32

@compile
def class_struct() -> i32:
    p: Point
    p.x = 10
    p.y = 20

    r: Rect
    r.top_left.x = 0
    r.top_left.y = 0
    r.width = 100
    r.height = 200

    return p.x + r.width
```

#### Nested Structs

```python
from pythoc import i32, f64, struct, compile

@compile
def nested_struct() -> i32:
    nested: struct[i32, struct[f64, f64]] = (1, (2.0, 3.0))

    outer: i32 = nested[0]
    inner: struct[f64, f64] = nested[1]
    x: f64 = inner[0]
    y: f64 = inner[1]

    return outer

@compile
def struct_with_pointer() -> i32:
    arr: array[i32, 3] = [1, 2, 3]
    p: ptr[i32] = arr

    # Struct with pointer field
    data: struct[ptr[i32], i32] = (p, 3)

    ptr_val: ptr[i32] = data[0]
    len_val: i32 = data[1]

    return ptr_val[0]
```

### Unions (`union[...]`)

All fields share the same memory location:

```python
from pythoc import i32, i64, f64, union, struct, compile

@compile
def union_basic() -> i32:
    # Named fields
    data = union[i: i32, f: f64]()

    # Write as integer
    data.i = 42
    i_val: i32 = data.i

    # Write as float (overwrites)
    data.f = 3.14
    f_val: f64 = data.f

    # Reading as int shows bit pattern
    i_val2: i32 = data.i

    return 0

@compile
def union_unnamed() -> i32:
    # Unnamed fields (access by index)
    data: union[i32, f64]

    data[0] = 42            # Write as i32
    data[1] = 2.718         # Write as f64

    return data[0]

@compile
def union_in_struct() -> i32:
    # Tagged union pattern
    record: struct[tag: i32, value: union[i: i32, f: f64]]
    record.tag = 0
    record.value.i = 42

    return record.value.i
```

### Enums

PythoC supports Rust-style tagged unions (ADTs):

```python
from pythoc import i32, i8, f64, ptr, struct, enum, compile

# Simple enum (no payload)
@enum
class Color:
    Red: None
    Green = 5           # Explicit tag value
    Blue: None          # Auto tag (6)

# Enum with payloads
@enum(i32)              # Tag type
class Result:
    Ok: i32             # Has payload
    Err: ptr[i8]        # Has payload

# Mixed (some with payload, some without)
@enum(i16)
class Mixed:
    None_Variant: None              # No payload, auto tag (0)
    Int_Variant: i32 = 10           # Has payload, explicit tag
    Float_Variant: f64              # Has payload, auto tag (11)
    Struct_Variant: struct[i32, f64]
    Another_None: None = 20         # No payload, explicit tag

@compile
def enum_usage() -> i32:
    # Access tag values
    r: i32 = Color.Red              # 0
    g: i32 = Color.Green            # 5

    # Create instances
    ok: Result = Result(Result.Ok, 42)
    err: Result = Result(Result.Err, "Error message")

    # Void variants (single argument)
    c: Color = Color(Color.Red)

    return Color.Green
```

### Function Pointers (`func[...]`)

```python
from pythoc import i32, func, compile

@compile
def add(x: i32, y: i32) -> i32:
    return x + y

@compile
def subtract(x: i32, y: i32) -> i32:
    return x - y

@compile
def multiply(x: i32, y: i32) -> i32:
    return x * y

# Pass function as argument
@compile
def apply_op(op: func[[i32, i32], i32], x: i32, y: i32) -> i32:
    return op(x, y)

# Return function pointer
@compile
def select_op(code: i32) -> func[[i32, i32], i32]:
    if code == 0:
        return add
    elif code == 1:
        return subtract
    else:
        return multiply

@compile
def calculator(op_code: i32, x: i32, y: i32) -> i32:
    operation: func[[i32, i32], i32] = select_op(op_code)
    return operation(x, y)

@compile
def main() -> i32:
    # Direct function pointer call
    result1: i32 = apply_op(add, 10, 5)      # 15
    result2: i32 = apply_op(multiply, 3, 4)   # 12

    # Via function table
    calc1: i32 = calculator(0, 20, 10)        # add: 30
    calc2: i32 = calculator(1, 30, 10)        # subtract: 20

    return result1 + result2 + calc1 + calc2
```

---

## Control Flow

### If/Else

```python
from pythoc import i32, compile

@compile
def if_else() -> i32:
    x: i32 = 10

    if x > 5:
        return 1
    else:
        return 2

@compile
def nested_if() -> i32:
    x: i32 = 10
    y: i32 = 20

    if x > 5:
        if y > 15:
            return 1
        else:
            return 2
    else:
        return 3
```

### While Loop

```python
from pythoc import i32, compile

@compile
def while_loop() -> i32:
    sum: i32 = 0
    i: i32 = 0

    while i < 10:
        sum = sum + i
        i = i + 1

    return sum  # 45
```

### For Loop

PythoC supports Python-style for loops that compile to efficient C loops:

```python
from pythoc import i32, compile

@compile
def for_range() -> i32:
    sum: i32 = 0

    for i in range(10):         # 0 to 9
        sum = sum + i

    return sum  # 45

@compile
def for_range_start() -> i32:
    sum: i32 = 0

    for i in range(5, 10):      # 5 to 9
        sum = sum + i

    return sum  # 35

@compile
def for_range_step() -> i32:
    sum: i32 = 0

    for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
        sum = sum + i

    return sum  # 20
```

### Break and Continue

```python
from pythoc import i32, compile

@compile
def break_example() -> i32:
    sum: i32 = 0
    i: i32 = 0

    while i < 100:
        if i >= 10:
            break
        sum = sum + i
        i = i + 1

    return sum  # 45

@compile
def continue_example() -> i32:
    sum: i32 = 0
    i: i32 = 0

    while i < 10:
        i = i + 1
        if i % 2 == 0:
            continue
        sum = sum + i

    return sum  # 25 (sum of odd numbers 1-9)
```

### Match/Case (Pattern Matching)

PythoC extends C's switch with Rust-style pattern matching:

```python
from pythoc import i32, compile

@compile
class Point:
    x: i32
    y: i32

@compile
def match_struct(px: i32, py: i32) -> i32:
    p: Point
    p.x = px
    p.y = py

    match p:
        case (0, 0):
            return 100          # Origin
        case (x, 0):
            return x            # On x-axis
        case (0, y):
            return y            # On y-axis
        case (x, y):
            return x + y        # General case

@compile
class Rect:
    top_left: Point
    width: i32
    height: i32

@compile
def match_nested(x: i32, y: i32, w: i32, h: i32) -> i32:
    r: Rect
    r.top_left.x = x
    r.top_left.y = y
    r.width = w
    r.height = h

    match r:
        case ((0, 0), w, h):
            return w * h        # At origin
        case ((x, y), w, h):
            return x + y + w + h
```

---

## Operators

### Arithmetic Operators

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division (integer or float) |
| `%` | Modulo |
| `//` | Floor division |

```python
from pythoc import i32, compile

@compile
def arithmetic() -> i32:
    a: i32 = 10
    b: i32 = 3

    add: i32 = a + b        # 13
    sub: i32 = a - b        # 7
    mul: i32 = a * b        # 30
    div: i32 = a / b        # 3
    mod: i32 = a % b        # 1

    return add + sub + mul + div + mod  # 54
```

### Bitwise Operators

| Operator | Description |
|----------|-------------|
| `&` | Bitwise AND |
| `\|` | Bitwise OR |
| `^` | Bitwise XOR |
| `~` | Bitwise NOT |
| `<<` | Left shift |
| `>>` | Right shift |

```python
from pythoc import i32, compile

@compile
def bitwise() -> i32:
    a: i32 = 12             # 0b1100
    b: i32 = 10             # 0b1010

    and_r: i32 = a & b      # 8  (0b1000)
    or_r: i32 = a | b       # 14 (0b1110)
    xor_r: i32 = a ^ b      # 6  (0b0110)
    not_r: i32 = ~a         # -13
    shl_r: i32 = a << 2     # 48
    shr_r: i32 = a >> 1     # 6

    return and_r + or_r + xor_r  # 28
```

### Comparison Operators

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `>` | Greater than |
| `<=` | Less than or equal |
| `>=` | Greater than or equal |

```python
from pythoc import i32, compile

@compile
def comparison() -> i32:
    a: i32 = 10
    b: i32 = 20
    c: i32 = 10

    result: i32 = 0

    if a < b:  result = result + 1
    if a <= c: result = result + 1
    if b > a:  result = result + 1
    if c >= a: result = result + 1
    if a == c: result = result + 1
    if a != b: result = result + 1

    return result  # 6
```

### Logical Operators

| Operator | Description |
|----------|-------------|
| `and` | Logical AND |
| `or` | Logical OR |
| `not` | Logical NOT |

```python
from pythoc import i32, bool, compile

@compile
def logical() -> i32:
    t: bool = True
    f: bool = False

    result: i32 = 0

    if t and t: result = result + 1
    if t or f:  result = result + 1
    if not f:   result = result + 1

    return result  # 3
```

### Unary Operators

```python
from pythoc import i32, compile

@compile
def unary() -> i32:
    a: i32 = 10
    b: i32 = -a             # Negation: -10
    c: i32 = +a             # Positive: 10

    return b + c            # 0
```

---

## Functions

### Basic Functions

```python
from pythoc import i32, f64, compile

@compile
def add(x: i32, y: i32) -> i32:
    return x + y

@compile
def no_return(x: i32):
    # void function
    pass

@compile
def with_locals() -> i32:
    a: i32 = 10
    b: i32 = 20
    result: i32 = a + b
    return result
```

### External Functions

```python
from pythoc import i32, ptr, i8, extern

@extern(lib='c')
def printf(format: ptr[i8], *args) -> i32:
    pass

@extern(lib='c')
def puts(s: ptr[i8]) -> i32:
    pass
```

### Variadic Functions

```python
from pythoc.libc.stdio import printf

@compile
def print_values() -> i32:
    printf("Integer: %d\n", 42)
    printf("Float: %f\n", 3.14)
    printf("Multiple: %d, %s, %f\n", 10, "hello", 2.5)
    return 0
```

---

## C Standard Library

PythoC provides bindings to the C standard library:

### stdio.h (`pythoc.libc.stdio`)

```python
from pythoc.libc.stdio import (
    # Console I/O
    printf, scanf, puts, getchar, putchar,
    # File I/O
    fopen, fclose, fread, fwrite, fgets, fputs,
    fprintf, fscanf, fflush,
    # File positioning
    fseek, ftell, rewind,
    # Error handling
    ferror, feof, clearerr,
    # Buffering
    setvbuf, setbuf,
    # Character I/O
    fgetc, fputc, ungetc,
    # String I/O
    sprintf, snprintf
)
```

### stdlib.h (`pythoc.libc.stdlib`)

```python
from pythoc.libc.stdlib import (
    # Memory management
    malloc, free, calloc, realloc,
    # Program control
    exit, abort,
    # String conversion
    atoi, atol, atof, strtol, strtod, strtoul,
    # Random numbers
    rand, srand,
    # Environment
    getenv, system,
    # Searching and sorting
    qsort, bsearch
)
```

### string.h (`pythoc.libc.string`)

```python
from pythoc.libc.string import (
    # String length
    strlen,
    # String copy
    strcpy, strncpy,
    # String concatenation
    strcat, strncat,
    # String comparison
    strcmp, strncmp,
    # String searching
    strchr, strrchr, strstr, strpbrk,
    # Memory operations
    memcpy, memset, memcmp, memmove,
    # Tokenization
    strtok, strtok_r,
    # Span functions
    strspn, strcspn
)
```

### math.h (`pythoc.libc.math`)

```python
from pythoc.libc.math import (
    # Trigonometric
    sin, cos, tan, asin, acos, atan, atan2,
    # Hyperbolic
    sinh, cosh, tanh,
    # Exponential and logarithmic
    exp, log, log10, log1p, expm1,
    # Power
    pow, sqrt, hypot,
    # Rounding
    ceil, floor, trunc, round,
    # Other
    fabs, fmod, modf
)
```

### Example: Memory Management

```python
from pythoc import i32, ptr, sizeof, nullptr, compile
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.string import memset

@compile
class Node:
    value: i32
    next: ptr[Node]

@compile
def create_node(value: i32) -> ptr[Node]:
    node: ptr[Node] = ptr[Node](malloc(sizeof(Node)))
    if node != nullptr:
        node.value = value
        node.next = nullptr
    return node

@compile
def free_list(head: ptr[Node]):
    while head != nullptr:
        next_node: ptr[Node] = head.next
        free(head)
        head = next_node

@compile
def main() -> i32:
    # Allocate array
    arr: ptr[i32] = ptr[i32](malloc(10 * sizeof(i32)))

    if arr == nullptr:
        return 1

    # Initialize
    for i in range(10):
        arr[i] = i * 10

    result: i32 = arr[5]    # 50

    free(arr)
    return result
```

---

## Differences from C

### Not Supported

| Feature | Status | Workaround |
|---------|--------|------------|
| Fall-through `switch` | Not supported | Use `match`/`case` with explicit branches |
| Variable-length arrays (VLA) | Not supported | Use `malloc` or fixed-size arrays |
| Flexible array members | Not supported | Use separate size tracking |
| Inline assembly | Not yet | Planned |

### Enhanced Features

| Feature | PythoC | C |
|---------|--------|---|
| Pattern matching | `match`/`case` with destructuring | `switch` (integers only) |
| Type inference | Limited (`x = func()`) | None |
| Generic types | Via Python metaprogramming | Templates (C++) |
| Multi-dimensional indexing | `arr[i, j, k]` | `arr[i][j][k]` only |
| Tagged unions | `@enum` with payloads | Manual implementation |

### Semantic Differences

1. goto: Limited goto with labels
2. Global variable initialization: Prefer effect with local static state. Global states are modeled explicitly using the effect system.
3. C-Preprocessor and macros: Simply use Python functions to do meta-programming and compile-time computation.
