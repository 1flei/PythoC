"""Already-formed method annotations bind to their enclosing declaration."""

import unittest

from pythoc import compile, enum, i32, ptr, union


@compile
class DirectRecord:
    value: i32

    def read(value: ptr['DirectRecord']) -> i32:
        return value.value


@union
class DirectUnion:
    value: i32

    def read(value: ptr['DirectUnion']) -> i32:
        return value.value


@enum(i32)
class DirectEnum:
    Value: i32
    Next: ptr['DirectEnum']

    def read(value: ptr['DirectEnum']) -> i32:
        match value[0]:
            case (DirectEnum.Value, number):
                return number
            case _:
                return 0


@compile
def read_all_methods() -> i32:
    record: DirectRecord
    record.value = 13
    choice: DirectUnion
    choice.value = 17
    value = DirectEnum(DirectEnum.Value, 12)
    return (DirectRecord.read(ptr(record)) + DirectUnion.read(ptr(choice))
            + DirectEnum.read(ptr(value)))


class EagerTypeResolutionTest(unittest.TestCase):
    def test_class_method_self_annotations(self):
        self.assertEqual(read_all_methods(), 42)


if __name__ == '__main__':
    unittest.main()
