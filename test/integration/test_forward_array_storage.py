"""Forward array elements support normal local storage and field access."""

import unittest

from pythoc import array, bool, compile, i32, i64, offsetof, ptr, sizeof


@compile
class ForwardArrayHolder:
    items: array['ForwardArrayElement', 2]
    matrix: array['ForwardArrayElement', 2, 3]
    tail: i32


@compile
class ForwardArrayElement:
    owner: ptr['ForwardArrayHolder']
    value: i32


@compile
def holder_size_matches_storage() -> bool:
    holders: array[ForwardArrayHolder, 2]
    stride = i64(ptr(holders[1])) - i64(ptr(holders[0]))
    return stride == sizeof(ForwardArrayHolder)


@compile
def holder_tail_offset_matches_storage() -> bool:
    holder: ForwardArrayHolder
    offset = i64(ptr(holder.tail)) - i64(ptr(holder))
    return offset == offsetof(ForwardArrayHolder, 'tail')


@compile
def element_size_matches_storage() -> bool:
    holder: ForwardArrayHolder
    stride = i64(ptr(holder.items[1])) - i64(ptr(holder.items[0]))
    return stride == sizeof(ForwardArrayElement)


@compile
def write_holder(holder: ptr[ForwardArrayHolder]) -> i32:
    holder.items[1].owner = holder
    holder.items[1].value = 37
    holder.matrix[1, 2].value = 41
    holder.tail = 43
    return holder.items[0].value


@compile
def access_through_pointer() -> bool:
    holder: ForwardArrayHolder
    holder.items[0].value = 31
    original = write_holder(ptr(holder))
    return (original == 31
            and holder.items[1].owner == ptr(holder)
            and holder.items[1].value == 37
            and holder.matrix[1, 2].value == 41
            and holder.tail == 43)


@compile
def local_holder() -> i32:
    holder: ForwardArrayHolder
    holder.items[0].value = 7
    holder.matrix[0, 2].value = 11
    return holder.items[0].value + holder.matrix[0, 2].value


class ForwardArrayStorageTest(unittest.TestCase):
    def test_holder_size_matches_its_array_stride(self):
        self.assertTrue(holder_size_matches_storage())

    def test_tail_offset_matches_its_address(self):
        self.assertTrue(holder_tail_offset_matches_storage())

    def test_forward_element_size_matches_its_array_stride(self):
        self.assertTrue(element_size_matches_storage())

    def test_local_array_fields(self):
        self.assertEqual(local_holder(), 18)

    def test_pointer_writes_are_visible_through_the_original_holder(self):
        self.assertTrue(access_through_pointer())


if __name__ == '__main__':
    unittest.main()
