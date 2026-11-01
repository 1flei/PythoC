"""Opaque references preserve their PC binding and C pointer storage ABI."""

from pathlib import Path
import tempfile
import unittest

from llvmlite import ir
from pythoc import array, compile, extern, func, i32, i64, ptr, static, void
from pythoc.utils.cc_utils import compile_c_to_object
from pythoc.utils.link_utils import get_shared_lib_extension, link_files


@compile
class Holder:
    next: ptr['ForwardNode']


@compile
class ArrayHolder:
    links: array[ptr['ForwardNode'], 2]


@compile
class CallbackHolder:
    callback: func['x':i32, i32]


@compile
class AliasHolder:
    next: ptr['VisibleAlias']


# Exercise representations made before the semantic pointee is defined.
Holder.get_llvm_type(ir.global_context)
ArrayHolder.get_llvm_type(ir.global_context)


def node_type(value_type):
    @compile
    class ForwardNode:
        value: value_type

    return ForwardNode


Unrelated = node_type(i32)
Unrelated.get_llvm_type(ir.global_context)
Target = node_type(i64)
VisibleAlias = Target


_temporary = tempfile.TemporaryDirectory(prefix='pc-forward-reference-')
_root = Path(_temporary.name)
_source = Path(__file__).parent / 'fixtures' / 'forward_record_reference.c'
_object = compile_c_to_object(str(_source), str(_root / 'reference.o'))
_library = link_files(
    [_object], str(_root / ('reference' + get_shared_lib_extension())),
    shared=True, link_objects=[], link_libraries=[])


@extern(lib=_library)
def read_second(holder: Holder) -> i64:
    pass


@extern(lib=_library)
def set_holder(holder: ptr[Holder], node: ptr[Target]) -> void:
    pass


@compile(suffix=str(_root))
def field_and_copy() -> i64:
    values: array[Target, 2]
    values[0].value = 4294967297
    values[1].value = 8589934594
    original: Holder = (ptr(values[0]),)
    copied: Holder = original
    return copied.next[1].value


@compile(suffix=str(_root))
def pointer_array_fields() -> i64:
    values: array[Target, 2]
    values[0].value = 4294967297
    values[1].value = 8589934594
    holder: ArrayHolder
    holder.links[0] = ptr(values[0])
    holder.links[1] = ptr(values[1])
    return holder.links[0][0].value + holder.links[1].value


@compile(suffix=str(_root))
def visible_alias_arithmetic() -> i64:
    values: array[Target, 2]
    values[0].value = 4294967297
    values[1].value = 8589934594
    holder: AliasHolder
    holder.next = ptr(values[0])
    return holder.next[1].value


@compile(suffix=str(_root))
def visible_alias_reverse_and_subtract() -> i64:
    values: array[Target, 2]
    values[0].value = 4294967297
    values[1].value = 8589934594
    holder: AliasHolder
    holder.next = ptr(values[0])
    return (1 + holder.next)[0].value + (holder.next + 1 - 1)[0].value


@compile(suffix=str(_root))
def visible_alias_attribute() -> i64:
    value: Target
    value.value = 4294967297
    holder: AliasHolder
    holder.next = ptr(value)
    return holder.next.value


@compile(suffix=str(_root))
def increment(x: i32) -> i32:
    return x + 10


@compile(suffix=str(_root))
def static_callback_storage() -> i32:
    holder: static[CallbackHolder] = (increment,)
    return holder.callback(x=5)


@compile(suffix=str(_root))
def c_round_trip() -> i64:
    values: array[Target, 2]
    values[0].value = 4294967297
    values[1].value = 8589934594
    holder: Holder
    set_holder(ptr(holder), ptr(values[0]))
    return read_second(holder)


class ForwardRecordReferenceTest(unittest.TestCase):
    def test_same_name_type_does_not_capture_a_forward_reference(self):
        self.assertEqual(field_and_copy(), 8589934594)

    def test_arrays_of_forward_pointers_recover_their_semantic_view(self):
        self.assertEqual(pointer_array_fields(), 12884901891)

    def test_visible_alias_controls_pointer_arithmetic(self):
        self.assertEqual(visible_alias_arithmetic(), 8589934594)

    def test_visible_alias_controls_reverse_addition_and_subtraction(self):
        self.assertEqual(visible_alias_reverse_and_subtract(), 12884901891)

    def test_visible_alias_controls_pointer_attribute_access(self):
        self.assertEqual(visible_alias_attribute(), 4294967297)

    def test_static_function_pointer_fields_keep_typed_constants(self):
        self.assertEqual(static_callback_storage(), 15)

    def test_c_reads_and_writes_the_same_pointer_field_layout(self):
        self.assertEqual(c_round_trip(), 8589934594)


if __name__ == '__main__':
    unittest.main()
