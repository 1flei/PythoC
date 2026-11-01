"""Address-taking evaluates the base and index once, without reading the place."""

import unittest

from pythoc import array, compile, i32, ptr, struct, volatile


address = ptr


@struct
class Item:
    value: i32


@compile
def next_index(calls: ptr[i32]) -> i32:
    result: i32 = calls[0]
    calls[0] += 1
    return result


@compile
def base(items: ptr[Item], calls: ptr[i32]) -> ptr[Item]:
    calls[0] += 1
    return items


@compile
def address_index() -> i32:
    values: array[i32, 3] = (11, 22, 33)
    calls: i32 = 0
    selected: ptr[i32] = ptr(values[next_index(ptr(calls))])
    selected[0] = 99
    return calls * 1000 + values[0]


@compile
def address_member() -> i32:
    items: array[Item, 2] = ()
    calls: i32 = 0
    selected: ptr[i32] = address(base(items, ptr(calls))[0].value)
    selected[0] = 37
    return calls * 1000 + items[0].value


@compile
def address_computed_base() -> i32:
    values: array[i32, 3] = (11, 22, 33)
    calls: i32 = 0
    selected: ptr[i32] = ptr((values + next_index(ptr(calls)))[0])
    return calls * 1000 + selected[0]


@compile
def typed_pointer_cast() -> i32:
    items: array[Item, 2] = ()
    calls: i32 = 0
    selected: ptr[Item] = ptr[Item](base(items, ptr(calls)))
    selected[0].value = 41
    return calls * 1000 + items[0].value


@compile
def address_volatile() -> i32:
    value: volatile[i32] = 7
    selected: ptr[volatile[i32]] = ptr(value)
    selected[0] = 41
    return 0


class TestPointerAddressEvaluation(unittest.TestCase):
    def test_address_does_not_read_volatile_storage(self):
        from pythoc import flush_all_pending_outputs
        from pythoc.build.output_manager import get_output_manager

        flush_all_pending_outputs()
        ir = get_output_manager().get_ir_text(address_volatile._group_key)
        self.assertNotIn('load volatile', ir)

    def test_index(self):
        self.assertEqual(address_index(), 1099)

    def test_member_and_alias(self):
        self.assertEqual(address_member(), 1037)

    def test_computed_base(self):
        self.assertEqual(address_computed_base(), 1011)

    def test_typed_pointer_conversion(self):
        self.assertEqual(typed_pointer_cast(), 1041)


if __name__ == "__main__":
    unittest.main()
