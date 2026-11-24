"""
Integration tests for instantiate() builtin.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
import unittest

from pythoc import CompileSession, array, compile, f64, i32, i64, ptr, refined, struct
from pythoc.builtin_entities.instantiate import instantiate
from test.utils.test_utils import DeferredTestCase, expect_error


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


@compile
def run_n_init_return() -> i32:
    api = instantiate(yield_n(4))
    it = api.init()
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+1+2+3 = 6


# ====================================================================
# Typed generators with compile-time range expansion
# ====================================================================

@compile
def doubled_range() -> i32:
    x: i32
    for x in range(5):
        yield x * 2


@compile
def run_doubled_range_all() -> i32:
    api = instantiate(doubled_range)
    it: api.Iter; api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total  # 0+2+4+6+8 = 20

@compile
def run_doubled_range_partial() -> i32:
    api = instantiate(doubled_range)
    it: api.Iter; api.init(ptr(it))
    api.next(ptr(it))
    api.next(ptr(it))
    return api.value(ptr(it))  # 2


NESTED_CONSTANT_ROWS = (((1, 2), 3), ((4, 5), 6))


@compile
def nested_constant_sums() -> i32:
    a: i32
    b: i32
    c: i32
    for ((a, b), c) in NESTED_CONSTANT_ROWS:
        yield a + b + c
    for ((a, b), c) in (((7, 8), 9),):
        yield a + b + c


@compile
def run_nested_constant_sums() -> i32:
    api = instantiate(nested_constant_sums)
    it = api.init()
    total: i32 = 0
    while api.next(ptr(it)):
        total = total * 100 + api.value(ptr(it))
    return total


class CompileTimeData:
    def get_value(self):
        raise AssertionError('Iteration data must not be resolved as a global value provider')


CONSTANT_OBJECTS = (CompileTimeData(),)


@compile
def yield_for_constant_object() -> i32:
    for unused in CONSTANT_OBJECTS:
        yield 7


@compile
def run_constant_object() -> i32:
    api = instantiate(yield_for_constant_object)
    it = api.init()
    api.next(ptr(it))
    return api.value(ptr(it))


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


@compile
def run_const_init_return() -> i32:
    it = sealed_list.init()
    total: i32 = 0
    while sealed_list.next(ptr(it)):
        total = total + sealed_list.value(ptr(it))
    return total


# ====================================================================
# Edge cases: constant iterable
# ====================================================================

@compile
def yield_empty() -> i32:
    if False:
        yield 0


sealed_empty = instantiate(yield_empty)

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
    def test_init_return(self): self.assertEqual(run_n_init_return(), 6)

class TestTypedRange(unittest.TestCase):
    def test_all(self):    self.assertEqual(run_doubled_range_all(), 20)
    def test_partial(self): self.assertEqual(run_doubled_range_partial(), 2)

    def test_nested_constant_elements_use_ordinary_assignment(self):
        self.assertEqual(run_nested_constant_sums(), 61524)

    def test_iteration_objects_are_bound_as_data(self):
        self.assertEqual(run_constant_object(), 7)

class TestConst(unittest.TestCase):
    def test_all(self): self.assertEqual(run_const_all(), 60)
    def test_init_return(self): self.assertEqual(run_const_init_return(), 60)

class TestEdgeCases(DeferredTestCase):
    def test_typed_empty_source(self):
        self.assertEqual(run_empty(), 0)

    def test_empty_iterable_requires_a_type(self):
        with self.assertRaisesRegex(TypeError, 'empty iterable'):
            instantiate([])


# ====================================================================
# Nested typed yield-function iterators
# ====================================================================

@compile
def view_three() -> i32:
    yield 1
    yield 2
    yield 3


@compile
def view_xs() -> i32:
    yield 1
    yield 2


@compile
def view_ys() -> i32:
    yield 10
    yield 20


@compile
def doubled_view() -> i32:
    x: i32
    for x in view_three():
        yield x * 2


@compile
def underscore_view() -> i32:
    for _ in view_three():
        yield _ * 2


NestedIteratorValue = struct[struct[i32, f64], i32]


@compile
def nested_iterator_values() -> NestedIteratorValue:
    yield ((1, 2.5), 3)
    yield ((4, 5.5), 6)


@compile
def nested_iterator_view() -> i32:
    for ((a, b), c) in nested_iterator_values():
        yield a + i32(b * 10.0) + c


@compile
def combined_views() -> i32:
    x: i32
    y: i32
    for x in view_xs():
        for y in view_ys():
            yield x + y


@compile
def run_doubled_view() -> i32:
    api = instantiate(doubled_view)
    it: api.Iter
    api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total


@compile
def run_dynamic_nested() -> i32:
    api = instantiate(combined_views)
    it: api.Iter
    api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total = total + api.value(ptr(it))
    return total


@compile
def run_underscore_view() -> i32:
    api = instantiate(underscore_view)
    it: api.Iter
    api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total += api.value(ptr(it))
    return total


@compile
def run_nested_iterator_view() -> i32:
    api = instantiate(nested_iterator_view)
    it: api.Iter
    api.init(ptr(it))
    total: i32 = 0
    while api.next(ptr(it)):
        total += api.value(ptr(it))
    return total


@compile
def stateful_row_updates() -> i32:
    row: array[i32, 2] = [1, 10]
    index: i32 = 0
    row[index] += 2
    yield row[index]
    index = 1
    row[index] += 3
    yield row[index]


@compile
def run_stateful_row_updates() -> i32:
    api = instantiate(stateful_row_updates)
    it = api.init()
    total: i32 = 0
    while api.next(ptr(it)):
        total += api.value(ptr(it))
    return total


class TestDynamicIterator(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(run_doubled_view(), 12)

    def test_nested(self):
        self.assertEqual(run_dynamic_nested(), 66)

    def test_underscore_can_be_lifted_to_a_typed_state_slot(self):
        self.assertEqual(run_underscore_view(), 12)

    def test_nested_targets_are_lifted_to_typed_state_slots(self):
        self.assertEqual(run_nested_iterator_view(), 94)

    def test_augmented_assignment_rewrites_nested_state_references(self):
        self.assertEqual(run_stateful_row_updates(), 16)

    def test_instantiate_resolves_normal_callee_expressions(self):
        self.assertEqual(instantiate_dotted(), 6)
        self.assertEqual(instantiate_computed(), 6)

    def test_expanded_iterator_locals_keep_their_declared_types(self):
        self.assertEqual(instantiate_expanded_locals(), 9)


VALUE = 1.25


@compile
def typed_argument(value) -> f64:
    yield value


@compile
def argument_metadata() -> i32:
    api = instantiate(typed_argument(f64(1.25)))
    state = api.init()
    api.next(ptr(state))
    return i32(api.value(ptr(state)) * 100.0)


@compile
def wide_argument(value) -> i64:
    yield value


@compile
def wide_argument_metadata() -> i64:
    api = instantiate(wide_argument(i64(1099511627776)))
    state = api.init()
    api.next(ptr(state))
    return api.value(ptr(state))


@compile
def literal_generator() -> i32:
    api = instantiate(1.25 for item in range(2))
    state = api.init()
    api.next(ptr(state))
    return i32(api.value(ptr(state)) * 100.0)


@expect_error(['declared yield type'], suffix='instantiate_alias_type')
def unknown_generator_type():
    @compile(suffix='instantiate_alias_type')
    def invalid() -> i32:
        api = instantiate(VALUE for item in range(2))
        state = api.init()
        api.next(ptr(state))
        return i32(api.value(ptr(state)) * 100.0)


@compile
def unknown_local_source() -> f64:
    amount = f64(1.25)
    yield amount


@expect_error(["state type of 'amount'"], suffix='instantiate_local_type')
def unknown_state_type():
    @compile(suffix='instantiate_local_type')
    def invalid() -> i32:
        api = instantiate(unknown_local_source)
        state = api.init()
        api.next(ptr(state))
        return i32(api.value(ptr(state)) * 100.0)


class TestIteratorTypeMetadata(unittest.TestCase):
    def test_bound_argument_supplies_its_existing_type(self):
        self.assertEqual(argument_metadata(), 125)

    def test_bound_wide_argument_retains_its_storage_width(self):
        self.assertEqual(wide_argument_metadata(), 1099511627776)

    def test_literal_generator_keeps_its_known_type(self):
        self.assertEqual(literal_generator(), 125)

    def test_unknown_generator_type_is_rejected(self):
        passed, message = unknown_generator_type()
        self.assertTrue(passed, message)

    def test_unknown_state_type_is_rejected(self):
        passed, message = unknown_state_type()
        self.assertTrue(passed, message)


@compile
def numbers(limit: i32) -> i32:
    index: i32 = 0
    while index < limit:
        yield index
        index += 1


iterators = SimpleNamespace(inner=SimpleNamespace(numbers=numbers))


@compile
def floating_numbers(limit: i32) -> f64:
    index: i32 = 0
    value: f64 = 0.5
    while index < limit:
        yield value
        value += 1.0
        index += 1


iterators.inner.floating = floating_numbers


@compile
def integer_results(limit: i32) -> i32:
    for value in iterators.inner.floating(limit):
        yield i32(value * 2.0)


@compile
def instantiate_expanded_locals() -> i32:
    api = instantiate(integer_results(3))
    iterator = api.init()
    total: i32 = 0
    while api.next(ptr(iterator)):
        total += api.value(ptr(iterator))
    return total


def choose_numbers():
    return numbers


@compile
def dotted_numbers() -> i32:
    value: i32
    for value in iterators.inner.numbers(4):
        yield value


@compile
def computed_numbers() -> i32:
    value: i32
    for value in choose_numbers()(4):
        yield value


@compile
def instantiate_dotted() -> i32:
    api = instantiate(dotted_numbers)
    iterator = api.init()
    total: i32 = 0
    while api.next(ptr(iterator)):
        total += api.value(ptr(iterator))
    return total


@compile
def instantiate_computed() -> i32:
    api = instantiate(computed_numbers)
    iterator = api.init()
    total: i32 = 0
    while api.next(ptr(iterator)):
        total += api.value(ptr(iterator))
    return total


StateChecked = refined[i32, 'state_checked']


@compile
def uninitialized_refined_state() -> i32:
    slot: StateChecked
    slot = StateChecked(3)
    yield i32(slot)


@compile
def shadowed_refined_state() -> i32:
    slot: i32 = 1
    yield slot
    if slot:
        slot: StateChecked
        slot = StateChecked(3)
        yield i32(slot)


class TestIteratorStateValidation(DeferredTestCase):
    def test_lifted_refined_declaration_still_requires_initializer(self):
        for source in (uninitialized_refined_state, shadowed_refined_state):
            with self.subTest(source=source), self.assertRaisesRegex(
                    TypeError, "Local 'slot'.*requires an initializer"):
                instantiate(source)

    def test_unknown_state_local_is_rejected_before_evaluation(self):
        events = []

        def value():
            events.append(1)
            return 1.25

        output = StringIO()
        with CompileSession():
            @compile
            def source() -> f64:
                amount = value()
                yield amount

            with redirect_stderr(output), self.assertRaisesRegex(TypeError, "state type of 'amount'"):
                instantiate(source)
        line = source._yield_func_obj.__code__.co_firstlineno + 2
        self.assertIn(f'[test_instantiate.py:{line}:4]', output.getvalue())
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
