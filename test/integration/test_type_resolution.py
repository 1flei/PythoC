"""Future annotations and explicit strings use the same native type graph."""

from __future__ import annotations

import unittest

from pythoc import (
    array, compile, enum, func, i32, i64, mark_type_defined, ptr, sizeof, static,
)


class SomeTypeConstructor:
    def __class_getitem__(cls, element):
        return ptr[element]


@compile
class RegistryPlaceholder:
    padding: i32


mark_type_defined('RegistryOnlyTarget', RegistryPlaceholder)
RegistryAlias = ptr['RegistryOnlyTarget']


@compile
def unquoted_registry_parameter(value: ptr[RegistryOnlyTarget]) -> i64:
    return value.value


@compile
def quoted_registry_parameter(value: RegistryAlias) -> i64:
    return value.value


@compile
class RegistryTarget:
    value: i64


mark_type_defined('RegistryOnlyTarget', RegistryTarget)

mark_type_defined('StorageBecomingInstance', static[i32])
mark_type_defined('StorageBecomingStatic', i64)


@compile
class DeferredInstance:
    selected: StorageBecomingInstance
    value: i32


@compile
class DeferredStatic:
    selected: StorageBecomingStatic
    value: i32


mark_type_defined('StorageBecomingInstance', i64)
mark_type_defined('StorageBecomingStatic', static[i32])


@compile
class FutureNode:
    next: ptr[FutureNode]
    links: array[ptr[FutureNode], 2]
    value: i32


@compile
class FutureHolder:
    values: array[LaterValue, 2]
    selected: 'SomeTypeConstructor[LaterValue]'
    links: array[ptr[LaterValue], 2]
    callback: func[i32, i32]


@compile
def pending_holder_parameter(value: ptr[FutureHolder]) -> i64:
    return value.values[1].value


@compile
class FutureStatics:
    selected: static[ptr[LaterValue]]
    value: i32


@enum(i32)
class FutureChoice:
    Node: ptr[LaterValue]
    Next: ptr[FutureChoice]


@compile
def deferred_parameter(value: SomeTypeConstructor[LaterValue]) -> i64:
    return value.value


@compile
class LaterValue:
    owner: ptr[FutureHolder]
    value: i64


@compile
def add_one(value: i32) -> i32:
    return value + 1


@compile
def returned_holder(value: ptr[LaterValue]) -> FutureHolder:
    holder: FutureHolder
    holder.values[0].value = 11
    holder.values[1].value = 31
    holder.selected = value
    holder.links[0] = value
    holder.links[1] = value
    holder.callback = add_one
    return holder


@compile
def use_returned_holder() -> i64:
    value: LaterValue
    value.value = 37
    selected = returned_holder(ptr(value)).selected
    selected.value += 5
    return deferred_parameter(ptr(value))


@compile
def use_returned_pointer_array() -> i64:
    value: LaterValue
    value.value = 19
    links: array[ptr[LaterValue], 2] = returned_holder(ptr(value)).links
    links[0].value += 4
    return links[0].value + links[1].value


@compile
def use_recursive_array() -> i32:
    first: FutureNode
    second: FutureNode
    second.value = 37
    first.next = ptr(second)
    first.links[0] = ptr(first)
    first.links[1] = ptr(second)
    return first.links[0].links[1].value + first.next.value


@compile
def use_pending_holder_parameter() -> i64:
    value: LaterValue
    holder = returned_holder(ptr(value))
    return pending_holder_parameter(ptr(holder))


@compile
def use_future_static() -> i64:
    value: LaterValue
    value.value = 37
    FutureStatics.selected = ptr(value)
    FutureStatics.selected.value += 5
    return value.value + i64(sizeof(FutureStatics))


@compile
def use_registry_replacement() -> i64:
    value: RegistryTarget
    value.value = 4294967333
    return (unquoted_registry_parameter(ptr(value))
            + quoted_registry_parameter(ptr(value)))


@compile
def use_local_type_alias() -> i32:
    Scalar = i32
    Pointer = ptr['Scalar']
    Values = array['Scalar', 2]
    values: Values = [19, 23]
    selected: Pointer = ptr(values[1])
    return values[0] + selected[0]


@compile
def use_reclassified_instance() -> i64:
    value: DeferredInstance
    value.selected = 4294967333
    value.value = 5
    return value.selected + i64(value.value)


@compile
def use_reclassified_static() -> i32:
    value: DeferredStatic
    value.value = 19
    DeferredStatic.selected = 23
    return value.value + DeferredStatic.selected + i32(sizeof(DeferredStatic))


@compile
def use_future_enum() -> i64:
    value: LaterValue
    value.value = 42
    first = FutureChoice(FutureChoice.Node, ptr(value))
    second = FutureChoice(FutureChoice.Next, ptr(first))
    match second:
        case (FutureChoice.Next, other):
            match other[0]:
                case (FutureChoice.Node, selected):
                    return selected.value
                case _:
                    return 0
        case _:
            return 0


@compile
def call_returned_callback() -> i32:
    value: LaterValue
    return returned_holder(ptr(value)).callback(41)


@compile
def static_typed_callback() -> i32:
    callbacks: static[array[func[i32, i32], 1]] = (add_one,)
    return callbacks[0](41)


def wrapped_compile(cls):
    return compile(cls)


def late_local_record(value_type):
    @wrapped_compile
    class LocalRecord:
        value: LocalAlias

    LocalAlias = value_type
    return LocalRecord


LocalSmall = late_local_record(i32)
LocalWide = late_local_record(i64)


@compile
def use_wrapped_local_records() -> i64:
    small: LocalSmall
    wide: LocalWide
    small.value = 19
    wide.value = 4294967319
    return i64(small.value) + wide.value


class ChoiceConstructor:
    def __class_getitem__(cls, payload):
        return enum['Value':payload, 'Empty':None]


@compile
class LocalChoiceHolder:
    choice: ChoiceConstructor[i64]


@compile
def use_constructed_anonymous_choice() -> i64:
    holder: LocalChoiceHolder
    Choice = ChoiceConstructor[i64]
    holder.choice = Choice(Choice.Value, i64(4294967338))
    match holder.choice:
        case (Choice.Value, value):
            return value
        case _:
            return 0


class UnifiedTypeResolutionTest(unittest.TestCase):
    def test_wrapped_definition_uses_its_own_late_local_alias(self):
        self.assertEqual(use_wrapped_local_records(), 4294967338)

    def test_anonymous_enum_constructor_keeps_payload_storage(self):
        self.assertEqual(use_constructed_anonymous_choice(), 4294967338)

    def test_returned_custom_constructor_field_updates_original_value(self):
        self.assertEqual(use_returned_holder(), 42)

    def test_returned_pointer_array_preserves_element_types_and_aliases(self):
        self.assertEqual(use_returned_pointer_array(), 46)

    def test_recursive_pointer_array_retains_all_aliases(self):
        self.assertEqual(use_recursive_array(), 74)

    def test_returned_function_pointer(self):
        self.assertEqual(call_returned_callback(), 42)

    def test_static_callback_initializer(self):
        self.assertEqual(static_typed_callback(), 42)

    def test_signature_can_reference_a_record_with_pending_value_fields(self):
        self.assertEqual(use_pending_holder_parameter(), 31)

    def test_future_static_is_independent_of_instance_storage(self):
        self.assertEqual(use_future_static(), 46)

    def test_both_registry_reference_forms_use_the_final_definition(self):
        self.assertEqual(use_registry_replacement(), 2 * 4294967333)

    def test_type_arguments_can_reference_compile_time_locals(self):
        self.assertEqual(use_local_type_alias(), 42)

    def test_deferred_static_alias_can_become_an_instance_field(self):
        self.assertEqual(use_reclassified_instance(), 4294967338)

    def test_deferred_instance_alias_can_become_static(self):
        self.assertEqual(use_reclassified_static(), 46)

    def test_future_enum_payloads_and_self_references(self):
        self.assertEqual(use_future_enum(), 42)


if __name__ == '__main__':
    unittest.main()
