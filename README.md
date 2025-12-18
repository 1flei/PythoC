# PythoC: Write C in Python

PythoC is a Python DSL compiler that compiles statically-typed Python to LLVM IR, providing C-equivalent runtime capabilities with Python syntax and compile-time metaprogramming.

## Design Philosophy

**Core principle**: C-level runtime + Python-powered compile-time

1. **C-compatible runtime**: Compiled code maps directly to native machine code with C-level control and performance
   - Full access to low-level operations (pointers, manual memory, inline assembly)
   - C calling conventions for seamless interoperability
   - No runtime overhead beyond what C would have
2. **Compile-time = Python**: Full Python power for metaprogramming, generics, and code generation
3. **Zero-cost abstractions**: Python-level abstractions compile away completely
4. **Explicit typing**: All types must be annotated (like C, unlike Python)
5. **Explicit control flow**: No implicit control flow (no exceptions, no RAII, no destructors)
   - Structs are plain data (no methods, no constructors)
   - Manual resource management like C
6. **Optional safety features**: Linear types and refinement types provide memory safety guarantees without introducing hidden control flow
   - Prevent memory leaks, use-after-free, null pointer dereference, array out-of-bounds
   - Completely optional - use only when needed
   - No extra runtime overhead
7. **Convenient Python-C interoperability**: 
   - Python can call PythoC compiled functions at runtime (via ctypes/cffi)
   - PythoC can invoke Python code at compile-time for metaprogramming

## Quick Start

### Installation

```bash
pip install pythoc
```

### Hello World

```python
from pythoc import compile, i32

@compile
def add(x: i32, y: i32) -> i32:
    return x + y

# Can compile to native code
@compile
def main() -> i32:
    return add(10, 20)

# Or call the compiled dynamic library from Python directly
result = main()
```

### Run Tests

```bash
# Run all tests
python test/run_all_tests.py

# Run specific test suites
python test/run_integration_tests.py
python test/run_examples.py
```

## Example: Binary Tree Benchmark

This example demonstrates PythoC's direct mapping to C - compare with `test/example/base_binary_tree_test.c`:

```python
from __future__ import annotations
from pythoc import i32, ptr, compile, nullptr, seq, sizeof
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import printf

# C: typedef struct tn { struct tn* left; struct tn* right; } treeNode;
@compile
class TreeNode:
    left: ptr[TreeNode]
    right: ptr[TreeNode]

# C: treeNode* NewTreeNode(treeNode* left, treeNode* right)
@compile
def NewTreeNode(left: ptr[TreeNode], right: ptr[TreeNode]) -> ptr[TreeNode]:
    new: ptr[TreeNode] = ptr[TreeNode](malloc(sizeof(TreeNode)))
    new.left = left
    new.right = right
    return new

# C: long ItemCheck(treeNode* tree)
@compile
def ItemCheck(tree: ptr[TreeNode]) -> i32:
    if tree.left == nullptr:
        return 1
    else:
        return 1 + ItemCheck(tree.left) + ItemCheck(tree.right)

# C: treeNode* BottomUpTree(unsigned depth)
@compile
def BottomUpTree(depth: i32) -> ptr[TreeNode]:
    if depth > 0:
        return NewTreeNode(BottomUpTree(depth - 1), BottomUpTree(depth - 1))
    else:
        return NewTreeNode(nullptr, nullptr)

# C: void DeleteTree(treeNode* tree)
@compile
def DeleteTree(tree: ptr[TreeNode]):
    if tree.left != nullptr:
        DeleteTree(tree.left)
        DeleteTree(tree.right)
    free(tree)
```

## C Parity: Full C Capabilities

### Supported Features

PythoC provides complete C runtime capabilities:

**Primitive types**:
- Integers: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- Floats: `f16`, `f32`, `f64`, `bf16`, `f128`
- Boolean: `bool`

**Composite types**:
- Pointers: `ptr[T]`
- Arrays: `array[T, N]` or `array[T, N, M, ...]` for multi-dimensional
- Structs: `@compile class`, `@struct class`, `struct[x: i32, y: i32]` (named) or `struct[i32, i32]` (unnamed)
- Unions: `@union class` or `union[T1, T2, ...]`
- Enums: `@enum class`
- Function pointers: `func[[arg_types], return_type]`

**Control flow**:
- `if`/`else`, `while`, `for` loops
- `break`, `continue`, `return`
- Pattern matching: `match`/`case` (enhanced switch)

**Operations**:
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `//`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `and`, `or`, `not`
- Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`
- Pointer arithmetic and dereferencing

**C Standard Library**:
```python
from pythoc.libc.stdio import printf, scanf
from pythoc.libc.stdlib import malloc, free, atoi
from pythoc.libc.string import memcpy, strlen
from pythoc.libc.math import sin, cos, pow
```

### Not Yet Supported

Features deliberately excluded or pending:

- **`goto`**: Use structured control flow instead
- **Fall-through `switch`**: Use `match`/`case` with explicit branches
- **Global variable initialization**: Workaround with init functions
- **Variable-length arrays (VLA)**: Use fixed-size arrays or malloc
- **Flexible array members**: Use separate size tracking

## PythoC Language Core

Beyond C parity, PythoC adds modern type system features (all optional, minimal support):

### Algebraic Data Types (ADT) and Pattern Matching

Provide Rust-like enums with payload for type-safe tagged unions:

```python
from pythoc import enum, compile, i32

@enum
class Result:
    Ok: i32
    Err: i32

@compile
def handle_result(r: Result) -> i32:
    match r:
        case (Result.Ok, value):
            return value
        case (Result.Err, code):
            return -code
```

### Linear Types (Optional)

Prevent use-after-free and resource leaks without RAII or destructors:

```python
from pythoc import compile, linear, consume, void, ptr, i8, i32, struct
from pythoc.libc.stdlib import malloc, free

# Allocator returns resource + linear proof
@compile
def lmalloc(size: i32) -> struct[ptr[i8], linear]:
    return malloc(size), linear()

# Only the paired deallocator can consume the proof
@compile
def lfree(ptr: ptr[i8], prf: linear) -> void:
    free(ptr)
    consume(prf)  # Proof consumed - resource released

@compile
def safe_usage() -> void:
    mem, prf = lmalloc(100)
    # ... use mem ...
    lfree(mem, prf)  # Must call lfree to consume prf
    # Compile error if prf not consumed!
```

**Motivation**: Proof-carrying code pattern. The linear proof proves resource was allocated and must be deallocated. Compiler enforces pairing of alloc/free at compile-time.

### Refinement Types (Optional)

Check once, use safely everywhere without runtime overhead:

```python
from pythoc import compile, i32, bool, refined, refine, array, ptr, nullptr

# Non-null pointer - check once, use safely everywhere
@compile
def is_nonnull(p: ptr[i32]) -> bool:
    return p != nullptr

NonNull = refined[is_nonnull]

@compile
def process_data(p: ptr[i32]) -> i32:
    for ptr_checked in refine(p, is_nonnull):
        # Type system knows ptr_checked is non-null
        return access_ptr(ptr_checked)
    else:
        return -1  # Handle null case

@compile
def access_ptr(p: refined[is_nonnull]) -> i32:
    return p[0]

# Array bounds - check once, access many arrays safely
@compile
def is_valid_index(idx: i32) -> bool:
    return 0 <= idx < 10

@compile
def process_arrays(i: i32, arr1: array[i32, 10], arr2: array[i32, 10]) -> i32:
    for idx in refine(i, is_valid_index):
        # Type system remembers idx is valid
        a = arr1[idx[0]]  # Safe: no bounds check
        b = arr2[idx[0]]  # Safe: no bounds check
        c = arr1[idx[0]]  # Safe: reuse safely
        return a + b + c
    else:
        return -1
```

**Motivation**: Check once, encode in type system, use safely everywhere. Typical examples: non-null pointers, valid array indices, non-zero divisors. The type system remembers the property, eliminating redundant runtime checks. Zero overhead for subsequent uses.

## Python as Preprocessor

Use Python's full power at compile-time for metaprogramming:

### Compile-Time Constants: Zero-Size Fields

```python
from pythoc import compile, i32, pyconst, struct, typeof, ptr

def Vec(T, size_type):
    return struct['size': size_type, 'data': ptr[T]]

def make_vec(T, size):
    ret: Vec(T, typeof(size))
    ret.size = size
    ret.data = malloc(size * sizeof(T))
    return ret

@compile
def test()-> i32:
    static_vec = make_vec(i32, 100) # sizeof(static_vec) == sizeof(ptr[T])
    len: i32 = 100
    dynamic_vec = make_vec(i32, len) # sizeof(dynamic_vec) == sizeof(struct[i32, ptr[T])
```

**Use case**: One generic definition, two instantiations with different memory layouts - static saves 4 bytes per instance.

### Generic Types via Python Functions

```python
from pythoc import compile, struct, i32, f64

# Python function generates specialized types and functions
def make_point(T):
    @struct(suffix=T)  # suffix creates unique type name
    class Point:
        x: T
        y: T
    
    @compile(suffix=T)  # suffix creates unique function name
    def add_points(p1: Point, p2: Point) -> Point:
        result: Point = Point()
        result.x = p1.x + p2.x
        result.y = p1.y + p2.y
        return result
    
    return Point, add_points

# Generate specialized versions at compile-time
Point_i32, add_i32 = make_point(i32)
Point_f64, add_f64 = make_point(f64)

@compile
def test() -> i32:
    p1: Point_i32 = Point_i32()
    p1.x = 10
    p2: Point_i32 = Point_i32()
    p2.x = 20
    result: Point_i32 = add_i32(p1, p2)
    return result.x
```

### Code Generation at Compile-Time

```python
# Python computes values and generates code before compilation
UNROLL_COUNT = 4

def make_unrolled_sum(n):
    @compile
    def sum_unrolled(arr: array[i32, n]) -> i32:
        result: i32 = 0
        # Python loop runs at compile-time, generates n add operations
        for i in range(n):
            result = result + arr[i]
        return result
    return sum_unrolled

# Generate specialized unrolled versions
sum_4 = make_unrolled_sum(4)
sum_8 = make_unrolled_sum(8)
```


## Library: Polymorphism

PythoC provides compile-time and runtime polymorphism through the `Poly` library:

```python
from pythoc import compile, i32, f64, ptr, i8
from pythoc.std.poly import Poly

# Define specialized implementations (must have same return type)
@compile
def add_i32(a: i32, b: i32) -> i32:
    return a + b

@compile
def add_f64(a: f64, b: f64) -> i32:
    return i32(a + b)  # Convert to i32 for uniform return type

@compile
def add_mixed(a: i32, b: f64) -> i32:
    return i32(f64(a) + b)

# Create polymorphic function - dispatches based on argument types
add = Poly(add_i32, add_f64, add_mixed)

# Static dispatch - type known at compile-time (zero overhead)
@compile
def test_static():
    x = add(10, 20)        # Calls add_i32
    y = add(1.5, 2.5)      # Calls add_f64
    z = add(10, 2.5)       # Calls add_mixed

# Extensible - add implementations dynamically
@compile
def add_str(a: ptr[i8], b: ptr[i8]) -> i32:
    return 42

add.append(add_str)

# Runtime dispatch via enum boxing
@enum(i32)
class Number:
    Int: i32
    Float: f64

@compile
def test_runtime():
    a: Number = Number(Number.Int, 42)
    b: Number = Number(Number.Float, 3.14)
    add(a, b)  # Dispatches based on runtime enum tags
```

**Key features**:
- All implementations must have the same return type
- Compile-time dispatch when argument types are statically known
- Runtime dispatch via enum boxing for dynamic polymorphism
- Extensible - add new implementations via `append()`

## AST Transformations Features: Inline, Yield, Closures

These features share the same underlying implementation: AST-level code transformation and inlining.

### Inline Functions

Zero-overhead abstraction - function body is expanded at call site:

```python
from pythoc import compile, inline, i32

@inline
def clamp(value: i32, min_val: i32, max_val: i32) -> i32:
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value

@compile
def process(x: i32) -> i32:
    # clamp body is inlined here - no function call
    result: i32 = clamp(x, 0, 100)
    return result
```

### Closures

Nested functions with variable capture - inlined automatically:

```python
from pythoc import compile, i32

@compile
def make_adder(base: i32) -> i32:
    offset: i32 = 10
    
    # Closure captures base and offset
    def add_both(x: i32) -> i32:
        return x + base + offset
    
    return add_both(5)  # Closure body inlined at call site
```

**Note**: Current implementation has limitations - closures cannot be called inside loops or nested (requires architecture redesign).

### Yield-based Iterators

Generator functions using `yield` - inlined automatically at call sites:

```python
from pythoc import compile, i32

@compile
def fibonacci(limit: i32) -> i32:
    """Generator yielding Fibonacci numbers < limit"""
    a: i32 = 0
    b: i32 = 1
    while a < limit:
        yield a
        new_a: i32 = b
        new_b: i32 = a + b
        a = new_a
        b = new_b

@compile
def sum_fibonacci(n: i32) -> i32:
    total: i32 = 0
    for num in fibonacci(n):  # fibonacci body inlined here
        total = total + num
    return total
```

**Common kernel**: All three features use the same AST transformation and inlining kernel, maintaining C-level performance with high-level abstractions.

## More Examples
More examples can be found in the `test/` directory.
- Full examples: `test/example/`
- PythoC Integration tests: `test/integration/`

## License

MIT License - see LICENSE file for details

Contact: yfleiii@gmail.com
