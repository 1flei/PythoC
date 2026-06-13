"""
Extended integration tests for instantiate() builtin.

Covers: heterogeneous types (f64/bool), multi-param, nested control flow,
break/continue, complex genexpr, compile-time for-unroll, error paths.
"""
from __future__ import annotations
import ast
import unittest
from pythoc import compile, i32, f64, bool, ptr
from pythoc.builtin_entities.instantiate import instantiate

F64Alias = f64


# ====================================================================
# Case: f64 yield
# ====================================================================

@compile
def yield_f64_seq() -> f64:
    yield 1.5
    yield 2.5
    yield 3.5

sealed_f64 = instantiate(yield_f64_seq)

@compile
def run_f64_all() -> i32:
    it: sealed_f64.Iter; sealed_f64.init(ptr(it))
    v1 = sealed_f64.value(ptr(it)) if sealed_f64.next(ptr(it)) else f64(0.0)
    v2 = sealed_f64.value(ptr(it)) if sealed_f64.next(ptr(it)) else f64(0.0)
    v3 = sealed_f64.value(ptr(it)) if sealed_f64.next(ptr(it)) else f64(0.0)
    r4 = sealed_f64.next(ptr(it))
    # Use i32 cast for comparison; values are 1.5+2.5+3.5 = 7.5 -> truncate 7
    total: i32 = i32(v1 + v2 + v3)
    return total + r4  # 7 + 0 = 7


# ====================================================================
# Case: bool yield
# ====================================================================

@compile
def yield_bool_seq() -> bool:
    yield True
    yield False
    yield True

sealed_bool = instantiate(yield_bool_seq)

@compile
def run_bool_all() -> i32:
    it: sealed_bool.Iter; sealed_bool.init(ptr(it))
    b1 = sealed_bool.value(ptr(it)) if sealed_bool.next(ptr(it)) else bool(0)
    b2 = sealed_bool.value(ptr(it)) if sealed_bool.next(ptr(it)) else bool(0)
    b3 = sealed_bool.value(ptr(it)) if sealed_bool.next(ptr(it)) else bool(0)
    r4 = sealed_bool.next(ptr(it))
    total: i32 = i32(0)
    if b1:
        total = total + i32(1)
    if b2:
        total = total + i32(2)
    if b3:
        total = total + i32(4)
    # 1 + 0 + 4 = 5
    return total + r4  # 5 + 0 = 5


# ====================================================================
# Case: typed constant iterables
# ====================================================================

sealed_const_f64 = instantiate([1.5, 2.5, 3.5])
sealed_const_bool = instantiate([True, False, True])


@compile
def run_const_f64_all() -> i32:
    it: sealed_const_f64.Iter; sealed_const_f64.init(ptr(it))
    total: f64 = 0.0
    while sealed_const_f64.next(ptr(it)):
        total = total + sealed_const_f64.value(ptr(it))
    return i32(total)


@compile
def run_const_bool_all() -> i32:
    it: sealed_const_bool.Iter; sealed_const_bool.init(ptr(it))
    total: i32 = 0
    while sealed_const_bool.next(ptr(it)):
        if sealed_const_bool.value(ptr(it)):
            total = total + 1
    return total


# ====================================================================
# Case: annotation alias resolution
# ====================================================================

@compile
def yield_alias_f64() -> F64Alias:
    value: F64Alias = 4.5
    yield value


sealed_alias_f64 = instantiate(yield_alias_f64)


@compile
def run_alias_f64() -> i32:
    it: sealed_alias_f64.Iter; sealed_alias_f64.init(ptr(it))
    total: f64 = 0.0
    while sealed_alias_f64.next(ptr(it)):
        total = total + sealed_alias_f64.value(ptr(it))
    return i32(total)


# ====================================================================
# Case: multi-param yield function
# ====================================================================

@compile
def yield_multi(start: i32, step: i32, count: i32) -> i32:
    i: i32 = 0
    while i < count:
        yield start + step * i
        i = i + 1

@compile
def run_multi_param_all() -> i32:
    api = instantiate(yield_multi(10, 3, 4))
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 10+13+16+19 = 58

@compile
def run_multi_param_partial() -> i32:
    api = instantiate(yield_multi(5, 2, 6))
    it: api.Iter; api.init(ptr(it))
    api.next(ptr(it))
    api.next(ptr(it))
    return api.value(ptr(it))  # 5 + 2*1 = 7


# ====================================================================
# Case: nested if inside while with yield
# ====================================================================

@compile
def yield_nested_if() -> i32:
    i: i32 = 0
    while i < 6:
        if i % 2 == 0:
            yield i * 10
        else:
            yield i
        i = i + 1

sealed_nested = instantiate(yield_nested_if)

@compile
def run_nested_if_all() -> i32:
    it: sealed_nested.Iter; sealed_nested.init(ptr(it))
    total: i32 = 0
    while sealed_nested.next(ptr(it)):
        total = total + sealed_nested.value(ptr(it))
    return total  # 0+1+20+3+40+5 = 69

@compile
def run_nested_if_partial() -> i32:
    it: sealed_nested.Iter; sealed_nested.init(ptr(it))
    sealed_nested.next(ptr(it))
    sealed_nested.next(ptr(it))
    sealed_nested.next(ptr(it))
    return sealed_nested.value(ptr(it))  # 20


# ====================================================================
# Case: multiple sequential yields (longer sequence)
# ====================================================================

@compile
def yield_ten() -> i32:
    yield 0
    yield 1
    yield 2
    yield 3
    yield 4
    yield 5
    yield 6
    yield 7
    yield 8
    yield 9

sealed_ten = instantiate(yield_ten)

@compile
def run_ten_sum() -> i32:
    it: sealed_ten.Iter; sealed_ten.init(ptr(it))
    total: i32 = 0
    while sealed_ten.next(ptr(it)):
        total = total + sealed_ten.value(ptr(it))
    return total  # 0+1+...+9 = 45


# ====================================================================
# Case: yield after loop (post-yield)
# ====================================================================

@compile
def yield_post() -> i32:
    i: i32 = 0
    while i < 3:
        yield i * 10
        i = i + 1
    yield 99

sealed_post = instantiate(yield_post)

@compile
def run_post_all() -> i32:
    it: sealed_post.Iter; sealed_post.init(ptr(it))
    total: i32 = 0
    while sealed_post.next(ptr(it)):
        total = total + sealed_post.value(ptr(it))
    return total  # 0+10+20+99 = 129


# ====================================================================
# Case: complex generator expression (filter + nested range)
# ====================================================================

@compile
def run_complex_genexpr() -> i32:
    api = instantiate(i32(x * x) for x in range(10) if x % 2 == 1)
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 1+9+25+49+81 = 165

@compile
def run_genexpr_filter() -> i32:
    api = instantiate(i32(x) for x in range(20) if x < 5)
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+1+2+3+4 = 10


# ====================================================================
# Case: compile-time for-loop unroll via constant iterable
# ====================================================================

@compile
def yield_for_unroll() -> i32:
    for x in range(3):
        yield x * 7

sealed_unroll = instantiate(yield_for_unroll)

@compile
def run_for_unroll() -> i32:
    it: sealed_unroll.Iter; sealed_unroll.init(ptr(it))
    v1 = sealed_unroll.value(ptr(it)) if sealed_unroll.next(ptr(it)) else i32(0)
    v2 = sealed_unroll.value(ptr(it)) if sealed_unroll.next(ptr(it)) else i32(0)
    v3 = sealed_unroll.value(ptr(it)) if sealed_unroll.next(ptr(it)) else i32(0)
    return v1 + v2 + v3  # 0+7+14 = 21


# ====================================================================
# Unit-test runners
# ====================================================================

class TestF64Yield(unittest.TestCase):
    def test_all(self): self.assertEqual(run_f64_all(), 7)

class TestBoolYield(unittest.TestCase):
    def test_all(self): self.assertEqual(run_bool_all(), 5)

class TestTypedConst(unittest.TestCase):
    def test_f64_const(self): self.assertEqual(run_const_f64_all(), 7)
    def test_bool_const(self): self.assertEqual(run_const_bool_all(), 2)

class TestAnnotationAlias(unittest.TestCase):
    def test_f64_alias(self): self.assertEqual(run_alias_f64(), 4)

class TestMultiParam(unittest.TestCase):
    def test_all(self):    self.assertEqual(run_multi_param_all(), 58)
    def test_partial(self): self.assertEqual(run_multi_param_partial(), 7)

class TestNestedIf(unittest.TestCase):
    def test_all(self):    self.assertEqual(run_nested_if_all(), 69)
    def test_partial(self): self.assertEqual(run_nested_if_partial(), 20)

class TestLongSequence(unittest.TestCase):
    def test_sum(self): self.assertEqual(run_ten_sum(), 45)

class TestPostYield(unittest.TestCase):
    def test_all(self): self.assertEqual(run_post_all(), 129)

class TestComplexGenExpr(unittest.TestCase):
    def test_filter_squares(self): self.assertEqual(run_complex_genexpr(), 165)
    def test_range_filter(self):   self.assertEqual(run_genexpr_filter(), 10)

class TestForUnroll(unittest.TestCase):
    def test_all(self): self.assertEqual(run_for_unroll(), 21)

class TestClosureSlot(unittest.TestCase):
    def test_closure_source_is_reserved(self):
        class FakeClosure:
            func_ast = ast.FunctionDef(
                name="fake_closure",
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    vararg=None,
                    kwarg=None,
                ),
                body=[ast.Return(value=ast.Constant(value=0))],
                decorator_list=[],
                returns=ast.Name(id="i32", ctx=ast.Load()),
            )
            func_globals = {}

        with self.assertRaises(NotImplementedError):
            instantiate(FakeClosure())


if __name__ == '__main__':
    unittest.main()
