"""
Integration tests for instantiate() builtin.
"""
from __future__ import annotations
import unittest
from pythoc import compile, i32, ptr
from pythoc.builtin_entities.instantiate import instantiate


# ====================================================================
# Helpers: yield functions (Case 1a / 1)
# ====================================================================

@compile
def yield_three() -> i32:
    yield 1
    yield 2
    yield 3

@compile
def yield_single() -> i32:
    yield 42

@compile
def yield_while() -> i32:
    i: i32 = 0
    while i < 5:
        yield i
        i = i + 1

@compile
def yield_n(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        yield i
        i = i + 1


# ====================================================================
# Seal instantiators at module load time (bare yield fn + const iterable)
# ====================================================================

sealed_seq   = instantiate(yield_three)
sealed_one   = instantiate(yield_single)
sealed_loop  = instantiate(yield_while)
sealed_list  = instantiate([10, 20, 30])


# ====================================================================
# Case 1a: sequential yields
# ====================================================================

@compile
def run_seq_all() -> i32:
    it: sealed_seq.Iter; sealed_seq.init(ptr(it))
    v1 = sealed_seq.value(ptr(it)) if sealed_seq.next(ptr(it)) else i32(0)
    v2 = sealed_seq.value(ptr(it)) if sealed_seq.next(ptr(it)) else i32(0)
    v3 = sealed_seq.value(ptr(it)) if sealed_seq.next(ptr(it)) else i32(0)
    r4 = sealed_seq.next(ptr(it))
    return v1 + v2 + v3 + r4  # 1+2+3+0 = 6

@compile
def run_seq_single() -> i32:
    it: sealed_one.Iter; sealed_one.init(ptr(it))
    sealed_one.next(ptr(it))
    return sealed_one.value(ptr(it))  # 42


# ====================================================================
# Case 1: yield in while loop
# ====================================================================

@compile
def run_loop_all() -> i32:
    it: sealed_loop.Iter; sealed_loop.init(ptr(it))
    total: i32 = 0
    while sealed_loop.next(ptr(it)):
        total = total + sealed_loop.value(ptr(it))
    return total  # 0+1+2+3+4 = 10

@compile
def run_loop_partial() -> i32:
    it: sealed_loop.Iter; sealed_loop.init(ptr(it))
    count: i32 = 0
    while sealed_loop.next(ptr(it)):
        count = count + i32(1)
        if count >= i32(3):
            break
    return sealed_loop.value(ptr(it))  # 2


# ====================================================================
# Case 1b: yield function with call-time arguments
# ====================================================================

@compile
def run_n_all() -> i32:
    api = instantiate(yield_n(3))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+1+2 = 3

@compile
def run_n_partial() -> i32:
    api = instantiate(yield_n(3))
    it: api.Iter; api.init(ptr(it))
    api.next(ptr(it))
    api.next(ptr(it))
    return api.value(ptr(it))  # 1


# ====================================================================
# Case 2: generator expression
# ====================================================================

@compile
def run_genexpr_all() -> i32:
    api = instantiate(i32(x * 2) for x in range(5))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+2+4+6+8 = 20

@compile
def run_genexpr_partial() -> i32:
    api = instantiate(i32(x * 2) for x in range(5))
    it: api.Iter; api.init(ptr(it))
    api.next(ptr(it))
    api.next(ptr(it))
    return api.value(ptr(it))  # 2


# ====================================================================
# Case 3: constant iterable
# ====================================================================

@compile
def run_const_all() -> i32:
    it: sealed_list.Iter; sealed_list.init(ptr(it))
    v1 = sealed_list.value(ptr(it)) if sealed_list.next(ptr(it)) else i32(0)
    v2 = sealed_list.value(ptr(it)) if sealed_list.next(ptr(it)) else i32(0)
    v3 = sealed_list.value(ptr(it)) if sealed_list.next(ptr(it)) else i32(0)
    return v1 + v2 + v3  # 60


# ====================================================================
# Edge cases: constant iterable
# ====================================================================

sealed_empty = instantiate([])

@compile
def run_empty() -> i32:
    it: sealed_empty.Iter; sealed_empty.init(ptr(it))
    return sealed_empty.next(ptr(it))  # should be 0 (false)


# ====================================================================
# Unit-test runners
# ====================================================================

class TestYieldSeq(unittest.TestCase):
    def test_all(self):    self.assertEqual(run_seq_all(), 6)
    def test_single(self): self.assertEqual(run_seq_single(), 42)

class TestYieldLoop(unittest.TestCase):
    def test_all(self):    self.assertEqual(run_loop_all(), 10)
    def test_partial(self): self.assertEqual(run_loop_partial(), 2)

class TestYieldN(unittest.TestCase):
    def test_all(self):    self.assertEqual(run_n_all(), 3)
    def test_partial(self): self.assertEqual(run_n_partial(), 1)

class TestGenExpr(unittest.TestCase):
    def test_all(self):    self.assertEqual(run_genexpr_all(), 20)
    def test_partial(self): self.assertEqual(run_genexpr_partial(), 2)

class TestConst(unittest.TestCase):
    def test_all(self): self.assertEqual(run_const_all(), 60)

class TestEdgeCases(unittest.TestCase):
    def test_empty_const(self):
        self.assertEqual(run_empty(), 0)


if __name__ == '__main__':
    unittest.main()
