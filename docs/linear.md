# Linear Types: Compile-Time Resource Tracking in PythoC

PythoC provides an *optional* linear type system for enforcing explicit resource cleanup
without introducing hidden control flow (no RAII, no destructors).

A `linear` value is a zero-size token (a proof marker). The compiler enforces that each
linear token is **consumed exactly once** before leaving its lifetime.

This document describes how to use linear tokens, how ownership moves through code,
and what control-flow restrictions are enforced.

## Table of Contents

- [Overview](#overview)
- [Core API](#core-api)
- [Ownership Rules](#ownership-rules)
- [Control Flow Rules](#control-flow-rules)
- [Composite Types and Field Paths](#composite-types-and-field-paths)
- [Defer + Linear](#defer--linear)
- [Scoped goto/label + Linear](#scoped-gotolabel--linear)
- [linear_wrap: Generate Linear-Safe Resource APIs](#linear_wrap-generate-linear-safe-resource-apis)

---

## Overview

Linear tokens are designed for proof-carrying resource management patterns.

A common pattern is to pair an acquire function with a release function using a
linear token as a proof that the resource must be released:

```python
from pythoc import compile, linear, consume, void, ptr, i8, i32, struct
from pythoc.libc.stdlib import malloc, free

@compile
def lmalloc(size: i32) -> struct[ptr[i8], linear]:
    return malloc(size), linear()

@compile
def lfree(p: ptr[i8], prf: linear) -> void:
    free(p)
    consume(prf)

@compile
def safe_usage() -> void:
    p, prf = lmalloc(100)
    # ... use p ...
    lfree(p, prf)  # must consume prf
```

The token has **no runtime cost**: it is a zero-size value used only for compile-time
tracking.

It is also possible to create a borrowed view of a resource that cannot outlive the resource:

```python
from pythoc import compile, linear, consume, void, ptr, i8, i32, struct, refined, assume

@compile
class Resource:
    x: i32

# A linear proof type that is distinct from other linear types
ResourcePrf = refined[linear, "ResourcePrf"]

@compile
def new_owned_res(size: i32) -> struct[Resource, ResourcePrf]:
    r: Resource
    r.x = size
    prf = assume(linear(), "ResourcePrf")
    return r, prf

@compile
def del_owned_res(r: Resource, prf: ResourcePrf) -> void:
    consume(prf)

# Create a view of the resource that cannot outlive the resource
ViewPrf = refined[linear, "ViewPrf"]

@compile
def get_res_view(r: Resource, prf: ResourcePrf) -> struct[ptr[Resource], ResourcePrf, ViewPrf]:
    pr = ptr(r)
    view_prf = assume(linear(), "ViewPrf")
    return pr, prf, view_prf

@compile
def release_res_view(pr: ptr[Resource], prf: ResourcePrf, view_prf: ViewPrf) -> ResourcePrf:
    consume(view_prf)
    return prf

@compile
def safe_usage() -> void:
    r, prf = new_owned_res(100)
    pr, prf, view_prf = get_res_view(r, prf)
    # del_owned_res(r, prf)   # If prf is consumed before view_prf, view_prf cannot be consumed properly
    prf = release_res_view(pr, prf, view_prf)
    del_owned_res(r, prf)
```

---

## Core API

| Construct | Meaning | Ownership effect |
|----------|---------|------------------|
| `linear()` | Create a new linear token | produces a fresh token |
| `consume(t)` | Consume a linear token | consumes `t` (makes it invalid) |
| `move(t)` | Explicitly transfer ownership | consumes `t` and returns a new owner handle |
| `f(t)` where `f` takes `linear` | Pass token to another function | consumes `t` in the caller |
| `return t` | Return token to caller | consumes `t` in the callee |

### `linear` (type)

- `linear` is a builtin type.
- It is represented as an empty LLVM struct (zero size).

### `consume(t: linear)`

- `consume` is a builtin function.
- It is a no-op at the IR level; its purpose is to trigger the ownership transition.

### `move(t: linear) -> linear`

`move` is a small helper in `pythoc.std.utility`:

```python
from pythoc.std.utility import move

@compile
def example() -> void:
    t = linear()
    t2 = move(t)
    consume(t2)
```

---

## Ownership Rules

### Tokens must be consumed exactly once

A token must be consumed before it leaves its lifetime (scope / function exit).

```python
from pythoc import compile, linear, consume, void

@compile
def ok() -> void:
    t = linear()
    consume(t)
```

Missing `consume` is an error:

```python
from pythoc import compile, linear, void

@compile
def bad() -> void:
    t = linear()
    # error: not consumed
```

### Tokens cannot be copied

Direct assignment copies are rejected. Use `move` instead.

```python
from pythoc import compile, linear, consume, void
from pythoc.std.utility import move

@compile
def ok_move() -> void:
    t = linear()
    t2 = move(t)
    consume(t2)

@compile
def bad_copy() -> void:
    t = linear()
    t2 = t  # error: cannot assign linear token (use move())
    consume(t2)
```

### Reassignment requires prior consumption

Reassigning an unconsumed token is rejected:

```python
from pythoc import compile, linear, consume, void

@compile
def bad_reassign() -> void:
    t = linear()
    t = linear()  # error: previous token not consumed
    consume(t)
```

After consumption, reassignment is allowed:

```python
from pythoc import compile, linear, consume, void

@compile
def ok_reassign() -> void:
    t = linear()
    consume(t)
    t = linear()
    consume(t)
```

### Declared-but-uninitialized linear is invalid

A declared linear variable is invalid until assigned a token.

```python
from pythoc import compile, linear, consume, void

@compile
def bad_undefined() -> void:
    t: linear
    consume(t)  # error: undefined / invalid

@compile
def ok_assign_then_consume() -> void:
    t: linear
    t = linear()
    consume(t)
```

---

## Control Flow Rules

Linear checking is CFG-based and path-sensitive.

At a high level:

1. **Merge points**: all incoming paths must have compatible linear states.
2. **Function exit**: all tokens must be consumed.

### If/Else

If a token is consumed in one branch, it must be consumed in all branches.

```python
from pythoc import compile, linear, consume, void, i32

@compile
def ok_ifelse(cond: i32) -> void:
    t = linear()
    if cond:
        consume(t)
    else:
        consume(t)
```

If only one branch consumes, it is rejected:

```python
from pythoc import compile, linear, consume, void, i32

@compile
def bad_ifelse(cond: i32) -> void:
    t = linear()
    if cond:
        consume(t)
    else:
        pass  # error: inconsistent states / not consumed at exit
```

If you want to consume after the `if`, keep the token active in all paths:

```python
from pythoc import compile, linear, consume, void, i32

@compile
def ok_consume_after_if(cond: i32) -> void:
    t = linear()
    if cond:
        pass
    consume(t)
```

### Loops

The same rule applies to loops:

- All merge points must have compatible linear states (including the loop back edge)
- That implies in generally tokens should not be consumed inside the loop unless there is no loop back edge

```python
from pythoc import compile, linear, consume, void, i32

@compile
def ok_loop_internal() -> void:
    i: i32 = 0
    while i < 3:
        t = linear()
        consume(t)
        i = i + 1
```

This pattern is rejected:

```python
from pythoc import compile, linear, consume, void, i32

@compile
def bad_loop_external() -> void:
    t = linear()
    i: i32 = 0
    while i < 3:
        consume(t)  # error: loop body changes linear state
        i = i + 1
```

OK, because the loop is exited without any condition:

```python
from pythoc import compile, linear, consume, void, i32

@compile
def bad_loop_external() -> void:
    t = linear()
    i: i32 = 0
    while i < 3:
        consume(t)  # error: loop body changes linear state
        i = i + 1
        break
    else:
        consume(t)
```

### Match/Case

`match`/`case` behaves like branching: all arms that reach a merge must be compatible.

### Return

Returning a value transfers ownership for any linear tokens contained in the return value.
That means:

- `return t` consumes `t` in the callee.
- Returning `struct[...]` that contains a linear field consumes those linear fields.

---

## Composite Types and Field Paths

Linear tokens can be nested inside structs and other composite values.

The checker tracks linear tokens by **field path**.

Example: `struct[ptr[i8], linear]` has one linear path at index `1`.

```python
from pythoc import compile, linear, consume, void, i32, struct, ptr, i8
from pythoc.libc.stdlib import malloc, free

@compile
def lmalloc(size: i32) -> struct[ptr[i8], linear]:
    return malloc(size), linear()

@compile
def lfree(p: ptr[i8], prf: linear) -> void:
    free(p)
    consume(prf)

@compile
def ok_unpack() -> void:
    p, prf = lmalloc(10)
    lfree(p, prf)
```

### Consuming nested tokens

You can consume tokens inside a struct using indexing:
All tokens in the struct must be consumed for the struct to be consumed.

```python
from pythoc import compile, linear, consume, void, struct

DualToken = struct[linear, linear]

@compile
def make_dual() -> DualToken:
    dt = DualToken()
    dt[0] = linear()
    dt[1] = linear()
    return dt

@compile
def destroy_dual(dt: DualToken) -> void:
    consume(dt[0])
    consume(dt[1])
```

You can also return a linear token from a struct field.

```python

@compile
def consume_first_return_second(dt: DualToken) -> linear:
    consume(dt[0])
    return move(dt[1])
```

---

## Defer + Linear

`defer(f, *args)` registers `f(*args)` to execute when the current scope exits.

Linear semantics:

- Linear arguments are **not** transferred at registration time.
- Linear arguments are transferred **when the deferred call executes**.

This allows a defer to hold a token until scope exit.

```python
from pythoc import compile, defer, linear, consume, void

@compile
def consumer(t: linear) -> void:
    consume(t)

@compile
def ok_defer() -> void:
    t = linear()
    defer(consumer, t)
    # do work...
    # consumer(t) executes at scope exit and consumes t
```

Important implication:

- If you manually consume or otherwise transfer the token before the defer runs,
  the deferred call will fail (use-after-consume).

Defer executes in FIFO order within the same scope.

---

## Scoped goto/label + Linear

PythoC supports scoped `goto`/`label` for low-level control flow.

Linear rules still apply:

- Any merge point (including label end targets) requires compatible linear states.
- If one path consumes a token before jumping to the same target, all paths must.

Example: consume before `goto_end` on all paths:

```python
from pythoc import compile, linear, consume, void, i32, label, goto_end

@compile
def ok_goto(cond: i32) -> void:
    t = linear()
    with label("main"):
        if cond:
            consume(t)
            goto_end("main")
        else:
            consume(t)
            goto_end("main")
```

Alternatively, keep token active across the label and consume after the merge point.

---

## linear_wrap: Generate Linear-Safe Resource APIs

`pythoc.std.linear_wrapper.linear_wrap` can generate wrappers around an
acquire/release function pair.

```python
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.libc.stdlib import malloc, free

lmalloc, lfree = linear_wrap(malloc, free)
```

Behavior (conceptual):

- `lmalloc(*args)` returns `struct[proof, resource]`.
- `lfree(proof, *args)` calls the original release function and then consumes `proof`.

Typical usage:

```python
from pythoc import compile
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.std.utility import move
from pythoc.libc.stdlib import malloc, free

lmalloc, lfree = linear_wrap(malloc, free)

@compile
def ok_linear_wrap() -> None:
    prf, p = lmalloc(100)
    lfree(prf, p)
```

### Named proof types via `struct_name`

You can request a named proof type (implemented as `refined[linear, "Name"]`):

```python
from pythoc.std.linear_wrapper import linear_wrap
from pythoc.libc.stdio import fopen, fclose

FileHandle, lfopen, lfclose = linear_wrap(fopen, fclose, struct_name="FileHandle")
```

This produces a distinct proof type that cannot be mixed with other wrappers.
