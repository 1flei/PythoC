# More Control Flow: PythoC vs C

PythoC aims for C-level runtime semantics; on top of a C-like core, it also introduces a few modern control-flow.

This document focuses on control-flow features that are *not* in C or that behave differently from typical C patterns:

- `match` / `case` pattern matching (an enhanced, checked `switch`)
- scoped `label` / `goto` / `goto_end` (structured low-level jumps)
- `defer` (scope-exit actions; Go/Zig-like, but FIFO within a scope)
- `yield` functions used as zero-overhead iterators for `for` loops (compile-time inlining)
- compile-time known constants used as control flow (`if`, `while`, `for` unrolling/special-casing)
- `for ... else` and `while ... else`

All statements below are based on the compiler implementation and integration tests in this repository.

## 1. Pattern matching: `match` / `case`

### 1.1. What it is (vs C `switch`)

C `switch` is restricted to integral (and enum) selectors, has fall-through, and does not destructure data.

PythoC `match`:

- supports Python 3.10+ `match` syntax
- supports destructuring of structs, arrays/tuples, and enums
- supports guards (`case pat if cond:`)
- performs *exhaustiveness checking* when subject types are known
- does not have implicit fall-through

### 1.2. Supported subjects

- **Single subject**: `match x:`
- **Multiple subjects**: `match x, y:`
  - in the Python AST this is represented as a tuple-like subject; PythoC evaluates each subject expression and matches them as a product.

### 1.3. Supported patterns (current implementation)

The matcher supports these pattern forms:

- **Literal / value**: `case 0:`, `case -1:`, `case SOME_CONST:`
- **Wildcard**: `case _:`
- **Binding**: `case n:` (binds the subject to `n`; also acts as a catch-all)
- **OR patterns**: `case 1 | 2 | 3:`
- **Struct destructuring** (keyword field patterns):
  - `case MyStruct(x=0, y=y): ...`
- **Sequence destructuring** for types that support `subject[i]`:
  - arrays, structs (field order), enums (`tag` / `payload`), tuples
  - examples: `case (0, 0):`, `case (Tag, payload):`

### 1.4. Guards

Guards are supported:

```python
match x:
    case n if n > 0:
        ...
    case _:
        ...
```

Notes:

- guards disable certain optimizations (see below)
- for exhaustiveness, guards are treated as *potentially False*; add a final wildcard (`case _:`) if you want an exhaustive match.

### 1.5. Exhaustiveness checking

PythoC can reject non-exhaustive matches at compile time.

Behavior (high-level):

- if there is an unguarded catch-all (`case _:` or `case name:` without `if`), the match is immediately considered exhaustive
- otherwise, the compiler checks coverage using a pattern-matrix algorithm
- if not exhaustive, compilation fails with an error like "Non-exhaustive match statement"

When exhaustiveness is meaningful:

- `bool` is finite and can be proven exhaustive with `True` and `False`
- `@enum` types are finite in their *constructors* (variants); payload coverage may matter when payload types are finite
- `@struct` product types can be proven exhaustive if all fields are finite

For infinite types (`i32`, pointers, etc.), you typically need a catch-all arm.

### 1.6. Lowering strategy

The implementation uses two lowering paths:

- **LLVM `switch`**: only when all cases are simple integer literal patterns (including ORs of integer literals), and there are no guards
- **if/elif chain**: used for guards, destructuring patterns, multiple subjects, and other non-trivial patterns

This matters because the "switch path" is closer to C `switch`, while the if-chain path behaves like a structured decision tree without fall-through.

### 1.7. Examples

#### (a) Integer-literal `match` (can lower to LLVM `switch`)

```python
from pythoc import i32, compile

@compile
def classify(x: i32) -> i32:
    match x:
        case 1 | 2 | 3:
            return 10
        case 4:
            return 20
        case _:
            return 0
```

#### (b) Exhaustive `match` over a finite type (`bool`)

```python
from pythoc import bool, i32, compile

@compile
def bool_to_i32(b: bool) -> i32:
    match b:
        case True:
            return 1
        case False:
            return 0
```

#### (c) Destructuring with a guard (typically lowers to an if/elif chain)

```python
from pythoc import i32, compile

@compile
class Point:
    x: i32
    y: i32

@compile
def score(p: Point) -> i32:
    match p:
        case (x, y):
            return y
        case (x, y) if x == y:
            return 100
        case _:
            return -1
```

## 2. Scoped labels and structured gotos: `label`, `goto`, `goto_end`

C `goto` is unstructured and can jump to any visible label. PythoC provides a *scoped* alternative that is powerful enough for low-level patterns (loops, state machines, error handling), but intentionally restricts visibility.

### 2.1. Syntax

A label is a scope introduced by a `with` statement:

```python
from pythoc import compile, i32
from pythoc.builtin_entities import label, goto, goto_end

@compile
def f() -> i32:
    x: i32 = 0
    with label("loop"):
        x = x + 1
        if x < 10:
            goto("loop")
    return x
```

There is also `goto_begin`, which is currently an alias for `goto`.

### 2.2. Two jump targets per label

Each label `X` conceptually defines two internal jump points:

- `X.begin`: at the `with label("X"):` level
- `X.end`: inside the label body, after PythoC has emitted scope-exit behavior

Accordingly:

- `goto("X")` jumps to `X.begin`
- `goto_end("X")` jumps to `X.end`

A useful mental model is:

- `X.begin` is "outside" the label body (at the scope-entry boundary)
- `X.end` is "inside" the label body (after scope-exit behavior has been emitted)

### 2.3. Visibility rules

The compiler uses a single, uniform visibility rule for label names:

- From a given point, you may only jump to:
  - the current label (self)
  - ancestor labels
  - "uncle" labels (a sibling of an ancestor)

This applies to both `goto("X")` and `goto_end("X")`. The difference is which internal target you choose within the same label:

- `goto("X")` (and `goto_begin("X")`) jumps to `X.begin`, which you can think of as being *outside* the `X` scope (before `with label("X"):` is entered)
- `goto_end("X")` jumps to `X.end`, which is *inside* the `X` scope (in the label body, after scope-exit behavior has been emitted). So it is not possible to goto_end of a "uncle" label.

### 2.4. Interaction with scopes and `defer`

A label is also a scope.

When you jump with `goto` or `goto_end`, PythoC conceptually exits scopes until it reaches the *parent scope depth* of the target label, and it executes all deferred actions for the exited scopes.

After that:

- `goto` branches to `X.begin` (re-entering `X`)
- `goto_end` branches to `X.end` (skipping the remainder of `X`)

This is the key difference from C `goto`: in PythoC, a jump has well-defined scope-exit behavior.

### 2.5. Forward gotos

`goto("name")` supports forward references to labels defined later in the function.

The compiler records a pending jump and resolves it when it later enters the matching `label("name")` scope.

### 2.6. Practical patterns

#### (a) Loop / break / continue without `while`

You can express loops explicitly:

- "continue" is typically `goto("loop")`
- "break" is typically `goto_end("loop")`

This is similar in spirit to C `goto cleanup` patterns, but scoped.

Compared to "labeled" `break` / `continue` in other languages, the design here is intentionally close to the usual idea (jumping to an enclosing control point). The *only* extra capability is that the name-visibility rule allows jumps to an "uncle" label, which makes it possible to express state-machine-like transitions between sibling blocks under a shared outer scope.

Example (loop via `label` + `goto`):

```python
from pythoc import compile, i32
from pythoc.builtin_entities import label, goto, goto_end

@compile
def count_to_5() -> i32:
    i: i32 = 0
    with label("loop"):
        if i >= 5:
            goto_end("loop")
        i = i + 1
        goto("loop")
    return i
```

#### (b) State machines

Sibling label jumps are useful for state machines:

- each `with label("state"):` block is a state
- `goto("next_state")` transitions

Example (forward jump to a sibling state):

```python
from pythoc import compile, i32
from pythoc.builtin_entities import label, goto

@compile
def two_states() -> i32:
    result: i32 = 0
    with label("state_a"):
        result = 1
        goto("state_b")
        result = 100  # skipped
    with label("state_b"):
        result = result + 10
    return result
```

Example ("uncle" jump from an inner scope to an outer sibling label):

```python
from pythoc import compile, i32
from pythoc.builtin_entities import label, goto

@compile
def uncle_jump() -> i32:
    x: i32 = 0
    with label("outer"):
        with label("inner"):
            x = 1
            goto("after_outer")
            x = 100  # skipped
    with label("after_outer"):
        x = x + 10
    return x
```

#### (c) Error handling / cleanup

A common C idiom is `goto cleanup;` with manual cleanup blocks.

In PythoC, you typically combine `defer` with `goto_end("function")` to exit early while keeping cleanup explicit.

### 2.7. Current limitation: crossing checks are minimal

For non-ancestor jumps (siblings/uncles), a robust compiler would normally reject jumps that skip variable initialization or skip `defer(...)` registration.

The current implementation keeps this checking minimal. As a result:

- you should avoid writing sibling/uncle `goto` that "jumps over" important declarations or `defer` registrations
- future versions may add stricter checks and start rejecting code that currently compiles

## 3. `defer`: explicit scope-exit actions

C does not have built-in scope-exit hooks. Cleanup is typically done with manual `goto cleanup` or carefully structured `return` paths.

PythoC provides a `defer` intrinsic:

```python
defer(f, a, b, c)
```

which registers a call `f(a, b, c)` to run when the *current scope* exits.

### 3.1. When defers execute

Defers registered in a scope execute when that scope exits via:

- normal fallthrough (end of the scope)
- `return` (executes defers from the current scope outward)
- `break` / `continue` (executes defers for the loop iteration / loop scope as appropriate)
- `goto` / `goto_end` (executes defers for scopes being exited by the jump)

### 3.2. Order

This project now uses the following ordering rules:

- **within the same scope**: defers execute in **FIFO** order (first registered, first executed)
- **across nested scopes**: when unwinding, inner scopes execute before outer scopes

### 3.3. Return value semantics

PythoC follows Go/Zig-like semantics:

- the return value is evaluated / captured **before** defers run
- therefore, a defer cannot modify the returned value by mutating locals after the `return` expression is captured

However, defers inside loops execute at the end of each iteration, so they *can* affect subsequent iterations.

Example (return value captured before defers):

```python
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, defer, ptr

@compile
def return_capture_demo() -> i32:
    result: i32 = 0

    def inc(p: ptr[i32]) -> void:
        p[0] = p[0] + 1

    defer(inc, ptr(result))
    return result
```

### 3.4. Argument capture

Arguments are evaluated at `defer(...)` registration time.

That means:

- the callee and all arguments must be valid at registration
- later modifications to the original variables do not retroactively change what was registered

Example (argument capture uses the value at registration):

```python
@compile
def arg_capture_demo() -> i32:
    result: i32 = 0

    if True:
        def add_n(p: ptr[i32], n: i32) -> void:
            p[0] = p[0] + n

        n: i32 = 1
        defer(add_n, ptr(result), n)
        n = 100
    return result   # result is 1
```

### 3.5. Linear types interaction

If you pass a linear token to a deferred call:

- linear ownership transfer is performed when the deferred call executes (at scope exit)
- not when you register the defer

This allows you to "hold" a linear resource until the end of the scope without adding hidden control flow.

This becomes especially important when a `yield` function interacts with linear state. In yield-based iteration, the *caller side* (the `for` loop body) effectively decides the subsequent control flow: it may keep consuming yields, `break` early, or otherwise leave the inlined yield region. The yield function itself cannot reliably predict or observe which yield will be the last one, so without `defer` it is very hard to manage linear state correctly inside the yield function. In practice, `defer` is the mechanism that lets you bind linear state transitions to scope exit in a way that stays correct under early-exit control flow.

Example (defer inside a yield-based loop still runs on `break`):

```python
from pythoc.decorators.compile import compile
from pythoc.builtin_entities import void, i32, ptr, defer, linear, consume

@compile
def consume_token(counter: ptr[i32], t: linear) -> void:
    counter[0] = counter[0] + 1
    consume(t)

@compile
def gen_with_linear(counter: ptr[i32], n: i32) -> i32:
    i: i32 = 0
    while i < n:
        t = linear()
        # Linear ownership transfer happens when the deferred call executes.
        defer(consume_token, counter, t)
        yield i
        i = i + 1

@compile
def break_early_linear() -> i32:
    c: i32 = 0
    for x in gen_with_linear(ptr(c), 10):
        if x == 3:
            break
    return c
```

## 4. `yield` functions and `for` loops (zero-overhead iteration)

C has no built-in generators. Iteration is expressed with explicit indexing, pointer iteration, or hand-written iterator structs.

PythoC supports `yield` inside `@compile` functions, and the semantics are intentionally designed around *compile-time inlining*.

### 4.1. The key idea

A `yield` function is not a runtime generator object.

Instead:

- the compiler inlines the yield function body into the `for` loop
- each `yield expr` is transformed into:
  - an assignment to the loop variable(s)
  - execution of the loop body

This can be truly zero-overhead: no allocation, no virtual calls, no iterator objects.

### 4.2. `break` / `continue` in yield-based loops

When the loop body contains `break` or `continue`, the compiler rewrites them using scoped labels:

- `break` becomes a jump that skips all remaining yields and also skips the `else` clause
- `continue` becomes a jump that exits the current yield-scope, proceeding to the next yield

This rewriting is why yield-based loops integrate correctly with `defer` and with `for ... else` semantics.

### 4.3. Tuple unpacking

Yield-based loops can bind either:

- a single loop variable: `for x in gen(): ...`
- a tuple of variables: `for a, b in gen_pairs(): ...`

For tuple unpacking, the yield function should annotate its return type accordingly (e.g. `struct[i32, i32]`) and yield tuples of matching shape.

### 4.4. Inlining limitations

Because yield iteration is implemented via AST inlining, yield functions must be "inlinable". This means you cannot store the yield function in a pythoc variable and call it later.

- a yield function must contain at least one `yield`
- `return value` is not allowed (plain `return` is ok)

### 4.6. Examples

#### (a) A simple yield-based iterator

```python
from pythoc import compile, i32

@compile
def simple_seq(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1

@compile
def sum_seq() -> i32:
    total: i32 = 0
    for x in simple_seq(10):
        total = total + x
    return total
```

#### (b) Yielding a linear token

```python
from pythoc import compile, i32
from pythoc.builtin_entities import linear, consume

@compile
def yield_linear_once() -> linear:
    prf = linear()
    yield prf

@compile
def consume_linear() -> i32:
    for prf in yield_linear_once():
        consume(prf)
    return 0
```

## 5. `for ... else` and `while ... else`

C has no `else` clause on loops.

In PythoC, loop-`else` follows Python semantics in the places where it is implemented:

- `else` executes only if the loop completes normally
- `else` is skipped if the loop exits via `break`

### 5.1. Examples

#### (a) `for ... else`

```python
from pythoc import compile, i32

@compile
def for_else_demo() -> i32:
    s: i32 = 0
    for i in [1, 2, 3]:
        s = s + i
    else:
        s = s + 100
    return s    # 106
```

#### (b) `while ... else`

```python
from pythoc import compile, i32

@compile
def while_else_demo(n: i32) -> i32:
    i: i32 = 0
    s: i32 = 0
    while i < n:
        if i == 3:
            break
        s = s + 1
        i = i + 1
    else:
        s = s + 100
    return s    # while_else_demo(5) == 3, while_else_demo(3) == 103
```

## 6. Compile-time known constants as control flow

In several places, the compiler can treat an expression as a compile-time Python value. When that happens, control flow becomes a compile-time decision rather than a runtime branch.

### 6.1. `if` with a compile-time condition

If the `if` condition is a compile-time Python value, only the taken branch is emitted (the other branch is not executed).

Example:

```python
from pythoc import compile, i32

@compile
def if_const_demo() -> i32:
    x: i32 = 0
    if True:
        x = 1
    else:
        x = 2  # not lowering to llvm IR
    return x
```

### 6.2. `while True` / `while False`

`while False` never executes and is skipped entirely.

`while True` uses a special lowering without a runtime condition check; the loop either breaks/returns or loops back.

Example:

```python
from pythoc import compile, i32

@compile
def while_const_demo() -> i32:
    x: i32 = 0
    while False:
        x = 100  # not lowering to llvm IR
    return x
```

### 6.3. `for` over a compile-time constant iterable

If the iterable is a compile-time Python value (e.g. a list/tuple/range known at compile time), PythoC can unroll the loop by transforming the AST into repeated scoped blocks.

The current implementation does this via an AST transform (see `inline/constant_loop_adapter.py`). The transform is designed to keep interactions with `defer`, `break`/`continue`, and loop-`else` explicit.

Example:

```python
from pythoc import compile, i32

@compile
def const_for_demo() -> i32:
    s: i32 = 0
    for i in [1, 2, 3, 4, 5]:   # unrolled into 5 blocks
        s = s + i
        if i == 3:
            break
    return s
