"""
Test pyconst and typeof features
"""
import ast
from unittest.mock import patch
from pythoc import CompileSession, LLVMCompiler
from pythoc.build.output_manager import clear_failed_group, flush_all_pending_outputs
from test.utils.test_utils import DeferredTestCase
from pythoc.logger import logger
import unittest
from pythoc import *


# Define struct outside @compile
TestStruct = struct[('a', i32), ('b', pyconst[42])]


@compile
def test_pyconst_field_access() -> i32:
    """Test accessing pyconst field in struct"""

    def get_b(x: TestStruct) -> i32:
        return i32(x.b)  # Should return constant 42
    
    s: TestStruct
    s.a = 10
    s.b = 42  # Should type-check and be no-op
    
    return get_b(s)


class TestPyconst(unittest.TestCase):
    def test_pyconst_basic(self):
        """Test basic pyconst type"""
        T = pyconst[42]
        self.assertTrue(hasattr(T, 'get_size_bytes'))
        self.assertEqual(T.get_size_bytes(), 0)
    
    def test_typeof_basic(self):
        """Test basic typeof functionality"""
        t1 = typeof(5)
        t2 = typeof(100)
        self.assertIsNot(t1, t2)  # Different values = different types
    
    def test_struct_with_pyconst(self):
        """Test struct with pyconst fields"""
        S = struct[('a', i32), ('b', pyconst[42]), ('c', i32)]
        # Size should exclude pyconst field (zero-sized)
        # sizeof(S) == sizeof(i32) + sizeof(i32) = 8
        expected_size = 8
        actual_size = S.get_size_bytes()
        self.assertEqual(actual_size, expected_size)
    
    def test_typeof_in_struct(self):
        """Test using typeof in struct definition"""
        def Vec(T, size_spec):
            return struct[('size', typeof(size_spec)), ('data', ptr[T])]
        
        # Static vector - size is pyconst[100]
        StaticVec = Vec(i32, 100)
        # Dynamic vector - size is i32
        DynamicVec = Vec(i32, i32)
        
        static_size = StaticVec.get_size_bytes()
        dynamic_size = DynamicVec.get_size_bytes()
        
        # Static: only ptr (8 bytes on 64-bit)
        # Dynamic: i32 + ptr (12-16 bytes depending on alignment)
        self.assertLess(static_size, dynamic_size)
    
    def test_pyconst_field_access(self):
        """Test compiled function with pyconst field"""
        result = test_pyconst_field_access()
        self.assertEqual(result, 42)

    def test_typed_constants_work_as_python_indices(self):
        self.assertEqual(typed_constant_indices(), 70)


TABLE = (10, 20)


MAPPING = {-1: 40, 1: 20}


@compile
def typed_constant_indices() -> i32:
    return TABLE[i32(1)] + TABLE[i32(0)] + MAPPING[i32(-1)]


@compile
def singleton_assignments(flag: i32) -> i32:
    inferred = 1
    declared: pyconst[1] = inferred
    implicit_value: pyconst[1]
    native: i32 = 1
    if flag:
        inferred = declared
        declared = implicit_value
        inferred += 0
        native = 2
    return inferred * 100 + declared * 10 + native


Singleton = pyconst[1]
SingletonRecord = struct['value': i32, 'tag': Singleton]


@compile
def make_singleton_record() -> SingletonRecord:
    return (41, 1)


@compile
def pass_singleton(value: Singleton) -> Singleton:
    address = ptr(value)
    address[0] = 1
    return value


@compile
def singleton_places() -> i32:
    local = 1
    record: SingletonRecord = (40, 1)
    local_ptr = ptr(local)
    field_ptr = ptr(record.tag)
    if i64(local_ptr) == 0 or i64(field_ptr) == 0:
        return -1
    local_ptr[0] = 1
    field_ptr[0] = 1
    record[1] += 0
    qualified: const[Singleton] = 1
    copied = qualified
    return (record.value + pass_singleton(local_ptr[0]) + field_ptr[0]
            + make_singleton_record().tag + copied)


@inline
def rebind_in_inline(_):
    _, _ = (1, 2)
    return _


@compile
def rebind_in_unpacking(counter: ptr[i32]) -> i32:
    def next_value() -> i32:
        counter[0] += 1
        return counter[0]

    _ = 7
    _, (_, kept) = (1, (2, 3))
    nested = _
    _, _ = (next_value(), next_value())
    return rebind_in_inline(_) * 100 + nested * 10 + _ + kept


@compile
def untyped_pairs():
    yield 1, 2
    yield 3, 4


@compile
def typed_values() -> struct[pyconst[1], pyconst[2], i32]:
    yield 1, 2, 3
    yield 1, 2, 4


@compile
def rebind_in_loops() -> i32:
    _ = 7
    total: i32 = 0
    for _, _ in untyped_pairs():
        total += _
    for _, _, kept in typed_values():
        total += _ + kept
    for _, _ in ((1, 2), (3, 4)):
        total += _
    for value in (_ + item for _, item in ((1, 5), (2, 6))):
        total += value
    for item in typed_values():
        total += _
        _ = item[2]
        total += _
    return total * 10 + _


@compile
def rebind_preserves_other_loop_bindings() -> i32:
    def values():
        yield 1, i32(40)

    _ = 7
    for _, other in values():
        pass
    return other + _


@compile
def rebind_with_closure(flag: i32) -> i32:
    _ = 1

    def read_first() -> i32:
        return _

    _ = 2
    inner: i32 = 0
    if flag:
        _ = 3
        inner = _
    match flag:
        case _:
            return read_first() * 100 + inner * 10 + _


@compile
def yield_owned_value() -> linear:
    token = linear()
    yield token


@compile
def rebind_after_consume() -> i32:
    _ = linear()
    consume(_)
    for _ in yield_owned_value():
        consume(_)
    _ = 7
    return _


class TestPyconstAssignments(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def reject(self, function, diagnostic, exc_type=TypeError):
        try:
            with self.assertRaisesRegex(exc_type, diagnostic):
                flush_all_pending_outputs()
        finally:
            clear_failed_group(function._binding.group_key)

    def test_same_singleton_and_native_reassignment(self):
        self.assertEqual(singleton_assignments(0), 111)
        self.assertEqual(singleton_assignments(1), 112)

    def test_singleton_places_have_real_addresses(self):
        self.assertEqual(singleton_places(), 44)

    def test_dereferenced_singleton_preserves_its_type(self):
        @compile(suffix='pyconst_pointer_assignment')
        def bad() -> i32:
            local = 1
            address = ptr(local)
            address[0] = 2
            return local

        self.reject(bad, r'Cannot assign pyconst\[2\] to pyconst\[1\]')

    def test_temporary_singleton_field_has_no_place(self):
        @compile(suffix='pyconst_temporary_field')
        def bad() -> i32:
            make_singleton_record().tag = 1
            return 0

        self.reject(bad, 'Assignment target is not an addressable value', ValueError)

    def test_const_singleton_obeys_the_same_write_check(self):
        @compile(suffix='pyconst_const_assignment')
        def bad() -> i32:
            local: const[Singleton] = 1
            local = 1
            return local

        self.reject(bad, 'Cannot reassign to const', RuntimeError)

    def test_unpacking_rebinds_in_order_and_evaluates_the_rhs_once(self):
        import ctypes

        counter = ctypes.c_int32()
        self.assertEqual(rebind_in_unpacking(ctypes.pointer(counter)), 225)
        self.assertEqual(counter.value, 2)

    def test_loop_rebinding_is_local_to_the_iteration(self):
        self.assertEqual(rebind_in_loops(), 587)
        self.assertEqual(rebind_preserves_other_loop_bindings(), 47)

    def test_rebinding_respects_scope_and_definition_site_capture(self):
        self.assertEqual(rebind_with_closure(0), 102)
        self.assertEqual(rebind_with_closure(1), 132)

    def test_rebinding_after_explicitly_consuming_the_old_value(self):
        self.assertEqual(rebind_after_consume(), 7)

    def test_unpacking_keeps_named_pyconst_bindings_strict(self):
        @compile(suffix="pyconst_named_unpack")
        def bad() -> i32:
            x = 1
            _, x = (0, 2)
            return x

        self.reject(bad, r"Cannot assign pyconst\[2\] to pyconst\[1\]")

    def test_rebinding_does_not_consume_linear_ownership(self):
        @compile(suffix="pyconst_rebind_linear")
        def bad() -> i32:
            _, _ = (linear(), 0)
            return 0

        self.reject(bad, "Linear tokens not consumed", RuntimeError)

    def test_different_value_fails_at_straight_line_assignment(self):
        @compile(suffix="pyconst_rebind")
        def bad() -> i32:
            x = 1
            x = 2
            return 0

        self.reject(bad, r"Cannot assign pyconst\[2\] to pyconst\[1\]")

    def test_branch_assignment_preserves_singleton_identity(self):
        for value in (2, True, 1.0):
            with self.subTest(value=value):
                @compile(suffix=("pyconst_branch", repr(value)))
                def bad(flag: i32) -> i32:
                    x = 1
                    if flag:
                        x = value
                    return x

                self.reject(bad, r"Cannot assign .* to pyconst\[1\]")

    def test_host_callable_cannot_be_selected_by_reassigning_a_pyconst(self):
        def first():
            return 1

        def second():
            return 2

        @compile(suffix="pyconst_callable_assignment")
        def bad(flag: i32) -> i32:
            target = first
            if flag:
                target = second
            return target()

        self.reject(bad, r"Cannot assign pyconst\[.*second.* to pyconst\[.*first")

    def test_native_parameter_cannot_replace_a_pyconst(self):
        @compile(suffix="pyconst_runtime_assignment")
        def bad(value: i32) -> i32:
            x = 1
            x = value
            return x

        self.reject(bad, r"Cannot assign i32 to pyconst\[1\]")

    def test_annotated_initializer_and_field_use_the_same_relation(self):
        Record = struct['tag': pyconst[1]]
        for field in (True, False):
            with self.subTest(field=field):
                @compile(suffix=("pyconst_explicit", field))
                def bad() -> i32:
                    if field:
                        record: Record
                        record.tag = True
                    else:
                        x: pyconst[1] = True
                    return 0

                self.reject(bad, r"Cannot assign pyconst\[True\] to pyconst\[1\]")


class TestPythonConstantIndexing(unittest.TestCase):
    def test_runtime_index_is_rejected_by_normal_compilation(self):
        syntax = ast.parse('def read(index):\n    return TABLE[index]').body[0]
        with CompileSession(), patch.object(logger, 'raise_on_error', True):
            compiler = LLVMCompiler(user_globals={'TABLE': (10, 20)})
            with self.assertRaisesRegex(TypeError, 'compile-time constant index'):
                compiler.compile_function_from_ast(
                    syntax, param_type_hints={'index': i32}, return_type_hint=i32)


if __name__ == "__main__":
    unittest.main()
