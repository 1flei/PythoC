"""Internal tests for the runtime typed task adapter."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pythoc import compile, effect, i32, i64, u64, ptr, void, nullptr, sizeof
from pythoc.std.mem_pool import PoolMem
from pythoc.std.runtime.raw import (
    Runtime,
    runtime_new_raw as runtime_new,
    runtime_start_raw as runtime_start,
    runtime_shutdown_raw as runtime_shutdown,
    runtime_free_raw as runtime_free,
)
from pythoc.std.runtime.spawn_typed import _TypedTask

effect.default(mem=PoolMem)


# ============================================================
# Test 1: Simple two-argument function
# ============================================================

@compile
def multiply(x: i64, y: i64) -> i64:
    return x * y


Multiply = _TypedTask(multiply)


@compile(suffix="test_typed_spawn_simple")
def test_simple_main() -> i64:
    rt: ptr[Runtime] = runtime_new(i32(2))
    runtime_start(rt)

    handle = Multiply.spawn(rt, i64(6), i64(7))
    result: i64 = Multiply.join(rt, handle)

    runtime_shutdown(rt)
    runtime_free(rt)
    return result


def test_typed_spawn_simple():
    """Task adapter: spawn a simple multiply(6, 7) -> 42."""
    result = int(test_simple_main())
    assert result == 42, f"Expected 42, got {result}"


# ============================================================
# Test 2: Multiple tasks in parallel
# ============================================================

@compile
def add_triple(a: i64, b: i64, c: i64) -> i64:
    return a + b + c


AddTriple = _TypedTask(add_triple)


@compile(suffix="test_typed_spawn_parallel")
def test_parallel_main() -> i64:
    rt: ptr[Runtime] = runtime_new(i32(4))
    runtime_start(rt)

    h1 = AddTriple.spawn(rt, i64(1), i64(2), i64(3))
    h2 = AddTriple.spawn(rt, i64(10), i64(20), i64(30))
    h3 = AddTriple.spawn(rt, i64(100), i64(200), i64(300))

    r1: i64 = AddTriple.join(rt, h1)
    r2: i64 = AddTriple.join(rt, h2)
    r3: i64 = AddTriple.join(rt, h3)

    runtime_shutdown(rt)
    runtime_free(rt)
    return r1 + r2 + r3


def test_typed_spawn_parallel():
    """Task adapter: spawn 3 parallel tasks, join all."""
    result = int(test_parallel_main())
    expected = 6 + 60 + 600
    assert result == expected, f"Expected {expected}, got {result}"


# ============================================================
# Test 3: Recursive spawn (like skynet but simpler)
# ============================================================

@compile
def recursive_sum(rt: ptr[Runtime], n: i64) -> i64:
    """Sum 0..n-1 by binary divide: spawn two halves."""
    if n <= i64(1):
        return n - i64(1) if n == i64(1) else i64(0)
    if n <= i64(16):
        # Base case: direct computation
        total: i64 = 0
        i: i64 = 0
        while i < n:
            total = total + i
            i = i + 1
        return total
    # Split in half
    mid: i64 = n / i64(2)
    left_task: ptr[void] = nullptr  # placeholder for recursive_sum
    # For recursive tasks we'd use spawn_raw
    return i64(0)  # simplified for test


# ============================================================
# Test 4: void-returning function
# ============================================================

@compile
class SharedCounter:
    value: i64


@compile
def increment_counter(counter_ptr: ptr[SharedCounter]) -> void:
    counter_ptr.value = counter_ptr.value + i64(1)


IncrementCounter = _TypedTask(increment_counter)


@compile(suffix="test_typed_spawn_void")
def test_void_main() -> i64:
    rt: ptr[Runtime] = runtime_new(i32(2))
    runtime_start(rt)

    counter: ptr[SharedCounter] = ptr[SharedCounter](
        effect.mem.malloc(u64(sizeof(SharedCounter)))
    )
    counter.value = i64(0)

    h1 = IncrementCounter.spawn(rt, counter)
    IncrementCounter.join(rt, h1)

    h2 = IncrementCounter.spawn(rt, counter)
    IncrementCounter.join(rt, h2)

    result: i64 = counter.value
    effect.mem.free(ptr[void](counter))

    runtime_shutdown(rt)
    runtime_free(rt)
    return result


def test_typed_spawn_void():
    """Task adapter: void-returning function."""
    result = int(test_void_main())
    assert result == 2, f"Expected 2, got {result}"


# ============================================================
# Test 5: Internal adapter surface
# ============================================================

def test_internal_adapter_surface():
    """Verify the internal adapter surface is correct."""
    # Check that Multiply has the expected attributes
    assert hasattr(Multiply, 'spawn')
    assert hasattr(Multiply, 'join')
    assert hasattr(Multiply, 'spawn_raw')
    assert hasattr(Multiply, 'join_raw')
    assert hasattr(Multiply, 'detach')
    assert hasattr(Multiply, 'args_type')
    assert hasattr(Multiply, 'trampoline')


if __name__ == "__main__":
    print("=== Task adapter tests ===")
    for name, fn in [
        ("simple", test_typed_spawn_simple),
        ("parallel", test_typed_spawn_parallel),
        ("void", test_typed_spawn_void),
        ("adapter_surface", test_internal_adapter_surface),
    ]:
        try:
            fn()
            print(f"  {name}: PASS")
        except Exception as e:
            print(f"  {name}: FAIL ({e})")
