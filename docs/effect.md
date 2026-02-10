# Effect System: Compile-Time Dependency Injection in PythoC

PythoC provides an *effect system* for **compile-time dependency injection** with **zero runtime overhead**.

In compiled code, `effect.xxx.yyy(...)` is resolved to a concrete function call during compilation.
This gives you a structured way to:

- choose implementations (allocator, IO shim, RNG, etc.) without globals
- build multiple compiled variants (prod vs mock) side-by-side
- make side effects explicit and controllable across module boundaries

This document describes the user-facing API, symbol naming via suffixes, import override behavior,
transitive propagation across call graphs, and common patterns.

## Table of Contents

- [Overview](#overview)
- [Core API](#core-api)
- [Resolution Order](#resolution-order)
- [Two Kinds of Effects: Namespaces vs Values](#two-kinds-of-effects-namespaces-vs-values)
- [Suffixes and Variants](#suffixes-and-variants)
- [Import Override (Scoped Recompilation)](#import-override-scoped-recompilation)
- [Transitive Effect Propagation (Calls)](#transitive-effect-propagation-calls)
- [Common Effects](#common-effects)

---

## Overview

At a high level:

- `effect` is a global singleton (an instance of `pythoc.effect.Effect`).
- Each effect name (e.g., `rng`, `mem`) maps to an *implementation*.
- In compiled code, when the compiler sees `effect.rng.next()`, it resolves:
  - `rng` to the current implementation object
  - `next` to a concrete callable (usually a `@compile` function)
  - and generates a static call to that callable

There is no dynamic dispatch at runtime.

The effect system supports three main binding modes:

- **Module defaults**: `effect.default(...)` (overridable by callers)
- **Scoped override**: `with effect(..., suffix="S"):` (caller override + variant naming)
- **Direct assignment**: `effect.xxx = impl` (not overridable; "pinned")

---

## Core API

### `effect` (singleton)

Import it from `pythoc`:

```python
from pythoc import effect
```

Use attribute access to refer to effects:

- `effect.rng` is an effect named `rng`
- `effect.mem` is an effect named `mem`

If an effect is unbound, accessing attributes on it raises an error.

### `effect.default(**bindings)`

Set *module-level defaults* for effect implementations.

- Defaults are **overridable** by callers.
- Use this in libraries to provide reasonable behavior while allowing injection.

```python
from pythoc import effect
from types import SimpleNamespace

# example: RNG impl object with compiled callables
RNG = SimpleNamespace(next=..., seed=...)

effect.default(rng=RNG)
```

### `effect.xxx = impl` (direct assignment)

Direct assignment pins an effect name to an implementation.

- It is **not overridable** by caller contexts.
- Use this for security- or correctness-critical bindings that must not change.

```python
from pythoc import effect

effect.secure_rng = SecureRNG
```

### `with effect(..., suffix="S"):` (scoped override)

Use a context manager to apply overrides temporarily.

```python
from pythoc import effect

with effect(rng=MockRNG, suffix="mock"):
    # imports inside see the override
    from mylib import random_u64
```

Notes:

- If you pass overrides, `suffix` is **required**.
- You may use `with effect(suffix="S"):` without overrides; it only affects naming context, and can be used to **address the symbol conflict**.

---

## Resolution Order

When the compiler (or runtime helper) resolves `effect.NAME`, the priority order is:

1. **Direct assignment** in some module: `effect.NAME = impl` (highest priority, not overridable)
2. **Caller override** active in a `with effect(..., suffix=...)` context
3. **Module default** set by `effect.default(NAME=impl)`
4. **Unbound** (error on attribute access)

This is designed to make library defaults overridable, while allowing explicit pinning.

---

## Two Kinds of Effects: Namespaces vs Values

PythoC supports using effects as:

### 1) Namespaced implementations (typical)

An implementation is an object (often `types.SimpleNamespace`) that exposes methods.
Those methods are usually `@compile` functions.

```python
from pythoc import compile, effect, u64, void
from types import SimpleNamespace

@compile
def rng_next() -> u64:
    return u64(42)

@compile
def rng_seed(s: u64) -> void:
    pass

RNG = SimpleNamespace(next=rng_next, seed=rng_seed)

effect.default(rng=RNG)

@compile
def use_rng() -> u64:
    return effect.rng.next()
```

### 2) Simple compile-time values (flags)

If an effect implementation is a simple value (`int`, `float`, `bool`, `str`),
`effect.NAME` can be used as a constant.

```python
from pythoc import compile, effect, i32

effect.default(MULTIPLIER=2)

@compile
def scale(x: i32) -> i32:
    return x * i32(effect.MULTIPLIER)
```

---

## Suffixes and Variants

### Why suffixes exist

When you compile multiple variants of the same function, they must have distinct symbols.
The effect system uses a string **effect suffix** (e.g., `"mock"`, `"counted"`) to name those variants.

In general:

- **`compile_suffix`** comes from `@compile(suffix=...)` and is often used by metaprogramming wrappers.
- **`effect_suffix`** comes from `with effect(..., suffix=...)` and represents the effect-configuration variant.

Key semantic rule:

- `effect_suffix` is **contagious** across calls (propagates to callees when needed).
- `compile_suffix` is **not contagious** (it is local to the wrapper being compiled).

### How `@compile` picks up `effect_suffix`

`@compile` implicitly reads the *current* `effect_suffix` from the active `with effect(suffix=...)` stack.
That means you can do:

```python
from pythoc import compile, effect

with effect(suffix="mock"):
    @compile
    def f():
        ...
```

and `f` will be compiled as an effect-variant of `"mock"` even if you did not pass `@compile(suffix=...)`.

If you pass an explicit `@compile(suffix=...)`, it only sets `compile_suffix`; the current `effect_suffix`
still applies:

- `name_{compile_suffix}_{effect_suffix}`

### Stable meaning of a suffix

A suffix is treated as an ABI/semantic identity: "same suffix = same behavior".

If you reuse the same suffix with different effect bindings, that is a user error.

Guideline:

- Use descriptive suffixes (`"mock"`, `"crypto"`, `"counted"`, `"v2"`).
- Do not dynamically generate many suffixes unless you truly need many variants.

---

## Import Override (Scoped Recompilation)

A major feature is *import override*:

```python
from pythoc import effect

with effect(rng=MockRNG, suffix="mock"):
    from effect_lib.rng_lib import random as mock_random

# later, in compiled code:
# mock_random() uses MockRNG, while normal imports keep defaults
```

How it works conceptually:

- Inside `with effect(..., suffix="S")`, `builtins.__import__` is temporarily hooked.
- For `from module import name1, name2, ...`, any imported `@compile` functions are wrapped
  into new `@compile` variants that capture the current effect context.
- The wrapped function is cached by `(module, attr, suffix)` so the same suffix is reused.

Properties:

- Default and overridden versions can coexist in the same process.
- Overrides apply to imports inside the `with` block.

---

## Transitive Effect Propagation (Calls)

Import override handles the "entry point" (what you import). But once you have a compiled function,
its body may call other `@compile` functions.

PythoC enforces the following semantic:

- If a caller is compiled under `effect_suffix = S` with overrides `{name -> impl}`,
  and a callee (or its transitive dependencies) uses at least one overridden effect name,
  then the call resolves to a **callee variant compiled with the same `effect_suffix = S`**.

The compiler decides whether to propagate based on recorded dependencies (effects used by a group).

---

## Common Effects

This section describes *conventions* and *standard libraries*.

- The only effect with a standardized implementation shipped in `pythoc.std` today is `mem`.
- Other names (like `rng`, `io`, `executor`, and application-level flags) are supported by the core effect
  mechanism, but their standard libraries and conventions are still evolving.

### `mem` (available today)

`effect.mem` provides allocation APIs such as `malloc/free` (and optionally `lmalloc/lfree`).

The standard library module `pythoc.std.mem` installs an overridable default binding:

```python
from pythoc.std import mem  # registers: effect.default(mem=mem.DefaultMem)
```

You can override `mem` to get alternate allocators or tracking:

```python
from pythoc import compile, effect, u64, ptr, void
from types import SimpleNamespace

@compile
def counting_malloc(size: u64) -> ptr[void]:
    # update counters, then call libc malloc
    return ptr[void](0)

@compile
def counting_free(p: ptr[void]) -> void:
    pass

CountingMem = SimpleNamespace(malloc=counting_malloc, free=counting_free)

with effect(mem=CountingMem, suffix="counted"):
    from mylib import allocate_and_free
```

If you want compile-time resource tracking, `pythoc.std.mem` also defines `mem.MemProof`
(a refined linear token) and the default implementation supports `lmalloc/lfree`.

### Application-defined global state (flags)

The core effect system supports simple-value effects (`int`, `float`, `bool`, `str`) so you can
model compile-time constants and configuration.

There is no standardized set of well-known flags yet; treat these as application-defined names.

```python
from pythoc import compile, effect, i32

effect.default(MULTIPLIER=2)

effect.default(DEBUG=False)

@compile
def f(x: i32) -> i32:
    if effect.DEBUG:
        return x
    return x * i32(effect.MULTIPLIER)
```

### Planned: `rng`, `io`, `executor`, and other subsystems

You can already build these as normal effects (define your own `effect.rng`, `effect.io`, etc.),
but `pythoc.std` does not provide standardized implementations for them yet.

Examples of planned directions:

- `rng`: deterministic / crypto / test-double RNG implementations
- `io`: explicit IO boundaries (e.g., read/write, logging, clocks)
- `executor`: execution model hooks for async / concurrency experimentation
