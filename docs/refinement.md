# Refinement Types: Predicate-Checked Values in PythoC

PythoC provides an *optional* refinement type system for carrying lightweight
value invariants through the type checker.

A refinement type is a normal value plus a set of constraints:

- **Predicates**: runtime-checkable boolean functions (compiled with `@compile`).
- **Tags**: string markers that are always true at runtime, used to separate
  otherwise identical values (e.g., avoid mixing ownership states).

Refinement types are designed to be **zero-overhead** when refining a single
value: the LLVM representation is the same as the underlying base type.

This document describes how to define refined types, how to construct them
safely (`refine`) or unsafely (`assume`), and what restrictions the compiler
enforces.

## Table of Contents

- [Overview](#overview)
- [Core Constructs](#core-constructs)
- [Type Syntax](#type-syntax)
- [Single-Value Refinement](#single-value-refinement)
- [Multi-Parameter Refinement](#multi-parameter-refinement)
- [Runtime Checking with `refine` (for-else)](#runtime-checking-with-refine-for-else)
- [Unchecked Construction with `assume`](#unchecked-construction-with-assume)
- [Construction via Calling the Type](#construction-via-calling-the-type)
- [Conversions and Operation Boundaries](#conversions-and-operation-boundaries)
- [refine_wrapper: Common Predicate Generators](#common-predicate-generators)

---

## Overview

A refinement type describes a type where only a specific subset of values are valid.

By using refinement types, you can encode a condition into the type system and avoid
re-checking the same condition multiple times across the program.

Typical use cases:

- **Validated inputs**: only proceed if `x > 0`, `ptr != nullptr`, etc.
- **State separation**: attach tags like `"owned"`, `"validated"`, `"init"`.
- **Proof typing**: create distinct linear proof tokens (see `docs/linear.md`).

Refinement types are intentionally lightweight:

- For **single-value** refined types, the runtime representation is the same
  as the base value.
- For **multi-parameter** refined types, the runtime representation is a struct
  that carries the input tuple.

---

## Core Constructs

| Construct | Purpose | Runtime behavior |
|----------|---------|------------------|
| `refined[...]` | Define a refined type | no runtime code |
| `assume(...)` | Construct refined value without checking | no checks |
| `refine(...)` | Runtime-check and bind a refined value | checks then yields |

Related helpers:

- Predicates are ordinary functions that return `pythoc.bool`.
- Tags are string constants.

---

## Type Syntax

PythoC supports multiple surface forms for refinement types.

### Base type + constraints

These forms refine a single value and are **zero-overhead**.

- `refined[T, "tag"]`
- `refined[T, pred]`
- `refined[T, pred, "tag"]`
- `refined[T, "tag1", "tag2"]`
- `refined[T, pred1, pred2, "tag"]`

Rules:

- The base type `T` must appear at position 0.
- Predicates must have exactly one parameter.
- Tags must be string constants.

### Predicate-only

- `refined[pred]`

This form infers the refined "shape" from the predicate signature:

- If `pred` has 1 parameter: it becomes a single-value refined type.
- If `pred` has N parameters (N > 1): it becomes a multi-parameter refined type
  (see below).

---

## Single-Value Refinement

### Defining a predicate

A single-value predicate is a `@compile` function:

- Signature: `(T) -> bool`
- It can use ordinary PythoC expressions.

```python
from pythoc import compile, i32, bool

@compile
def is_positive(x: i32) -> bool:
    return x > 0
```

### Defining a refined type

You can define refined types either via predicate-only syntax:

```python
from pythoc import refined

PositiveInt = refined[is_positive]
```

Or via base type + constraints syntax (useful when you want tags or multiple
predicates):

```python
from pythoc import refined, i32

CheckedPositive = refined[i32, is_positive, "checked"]
```

### Using a single-value refined value

A single-value refined value behaves like the base value for most operations.

```python
from pythoc import compile, i32, refined, assume

@compile
def f(x: i32) -> i32:
    return x + 1

@compile
def demo() -> i32:
    x = assume(10, is_positive)   # unchecked construction
    return f(x)                   # refined -> base is allowed
```

---

## Multi-Parameter Refinement

Multi-parameter refined types represent a tuple of values validated together.

### Defining a multi-parameter predicate

A multi-parameter predicate is a `@compile` function:

- Signature: `(T0, T1, ..., TN) -> bool`

```python
from pythoc import compile, i32, bool

@compile
def is_valid_range(start: i32, end: i32) -> bool:
    return start <= end and start >= 0
```

### Defining the refined type

Use predicate-only syntax:

```python
from pythoc import refined

ValidRange = refined[is_valid_range]
```

The refined value is represented as a struct with field names taken from the
predicate parameters (`start`, `end`, ...). You can access fields:

- by name: `r.start`, `r.end`
- by index: `r[0]`, `r[1]`

```python
from pythoc import compile, i32, assume

@compile
def use_range() -> i32:
    r = assume(10, 20, is_valid_range)
    return r.start + r.end
```

### Current restriction: tags with auto multi-arg form

The auto multi-arg form:

- `assume(v0, v1, ..., pred)`
- `refine(v0, v1, ..., pred)`

is intended to be a concise encoding of "values + one multi-arg predicate".

In the current implementation, **do not mix tags with this form**.

---

## Runtime Checking with `refine` (for-else)

`refine(...)` performs runtime checking, but it is not a normal function call.
It is a yield-based construct that must be used inside a `for` loop.

### Single value

```python
from pythoc import compile, i32, bool, refine

@compile
def is_nonzero(x: i32) -> bool:
    return x != 0

@compile
def safe_divide(dividend: i32, divisor: i32) -> i32:
    for d in refine(divisor, is_nonzero):
        return dividend / d
    else:
        return 0
```

Semantics:

- `refine(...)` yields **0 or 1** value(s). If checks pass, it yields one refined binding; otherwise it yields nothing.
- The `else` clause follows normal Python `for`-`else` semantics: it runs only if the loop completes without `break`.
- Therefore, interpreting `else` as the "failure" path is only valid when the success path exits the loop (e.g., `break` or `return`).

Canonical pattern:

```python
for x in refine(value, pred):
    # success path
    use(x)
    break
else:
    # failure path
    handle_error()
```

### Multiple predicates

Predicates are combined with logical AND:

```python
from pythoc import compile, i32, bool, refine

@compile
def is_positive(x: i32) -> bool:
    return x > 0

@compile
def is_small(x: i32) -> bool:
    return x < 100

@compile
def guarded(x: i32) -> i32:
    for v in refine(x, is_positive, is_small):
        return v
    else:
        return -1
```

### Tags only

Tags are always true at runtime. A tags-only refine always succeeds:

```python
from pythoc import compile, i32, refine

@compile
def tags_only(x: i32) -> i32:
    for v in refine(x, "owned", "initialized"):
        return v
    else:
        return -1  # unreachable
```

### Multi-parameter

```python
from pythoc import compile, i32, refine

@compile
def range_sum(a: i32, b: i32) -> i32:
    for r in refine(a, b, is_valid_range):
        return r.start + r.end
    else:
        return -999
```

### Important: `refine` must be used in a `for`

`refine()` is lowered by the compiler into an inline function and integrated
into the `for` lowering.

Using it outside a `for` loop is rejected (or will fail at runtime).

---

## Unchecked Construction with `assume`

`assume(...)` constructs a refined value **without checking any predicate**.

### Single value

```python
from pythoc import compile, i32, assume

@compile
def unchecked() -> i32:
    x = assume(5, is_positive)
    return x
```

You can attach multiple constraints:

```python
from pythoc import compile, i32, assume

@compile
def unchecked_tags_and_preds() -> i32:
    x = assume(42, is_positive, is_small, "validated", "trusted")
    return x
```

### Multi-parameter

```python
from pythoc import compile, i32, assume

@compile
def unchecked_range() -> i32:
    r = assume(10, 20, is_valid_range)
    return r.start + r.end
```

---

## Construction via Calling the Type (Not Recommended)

A refined type can also be called like a constructor. This is **unchecked** and
is equivalent to using `assume`.

```python
from pythoc import compile, i32, refined

PositiveInt = refined[is_positive]

@compile
def ctor() -> i32:
    x = PositiveInt(42)  # unchecked
    return x
```

For multi-parameter refined types, call with N arguments:

```python
from pythoc import compile, i32

ValidRange = refined[is_valid_range]

@compile
def ctor_multi() -> i32:
    r = ValidRange(1, 100)  # unchecked
    return r.start + r.end
```

---

## Conversions and Operation Boundaries

### Base -> refined

Direct conversion from a base value to a refined type is not allowed.
You must use `assume` (unchecked) or `refine` (checked).

```python
from pythoc import compile, i32, bool, refined

@compile
def is_positive(x: i32) -> bool:
    return x > 0

PositiveInt = refined[is_positive]

@compile
def bad(x: i32) -> i32:
    y: PositiveInt = x  # error
    return y
```

### Refined -> base

A refined value can be used where the base type is expected.

This is intentional: refinement is extra information, and it is safe to forget.

### Refined -> refined

Refined-to-refined conversions are permitted when the refinement tag of the
target type is a subset of the source type's refinement tag (meaning source type is more restrictive).

Recommended style:

- Use `assume(x, ...)` to explicitly change tags or attach additional
  constraints.
- Use `refine(x, ...)` when you need runtime-checked strengthening.

### Operations "forget" refinement

When a refined value participates in arithmetic/bitwise/unary operations, the
result generally does not preserve refinement information.

---

## Common Predicate Generators

`pythoc.std.refine_wrapper` provides small helpers to generate common predicates
and refined types for a specific base type.

Available wrappers include:

- `nonnull_wrap(T)`
- `positive_wrap(T)`
- `nonnegative_wrap(T)`
- `nonzero_wrap(T)`
- `in_range_wrap(T, lower, upper)`

Example:

```python
from pythoc.std.refine_wrapper import nonnull_wrap, positive_wrap
from pythoc import compile, i32, ptr

is_valid_ptr, NonNullI32Ptr = nonnull_wrap(ptr[i32])
is_positive, PositiveI32 = positive_wrap(i32)

@compile
def use(p: NonNullI32Ptr, n: PositiveI32) -> i32:
    return p[0] * n
```
