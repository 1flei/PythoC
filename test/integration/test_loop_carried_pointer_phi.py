"""Distinct loop-carried pointers keep assignments from their own paths."""

import unittest

from pythoc import array, compile, i8, i32, i64, ptr


@compile
class PhiValue:
    tag: i8
    value: i64


@compile
class PhiChoice:
    table: ptr[ptr[PhiValue]]
    index: i32
    branch: i32


@compile
def load_choice(choice: ptr[PhiChoice]) -> i64:
    first: ptr[PhiValue]
    second: ptr[PhiValue]
    state: i32 = 0
    while True:
        match state:
            case 0:
                state = 1 if choice.branch else 3
                continue
            case 1:
                first = choice.table[choice.index]
                state = 2
                continue
            case 2:
                return first.value
            case 3:
                second = choice.table[choice.index]
                state = 4
                continue
            case 4:
                return second.value
            case _:
                return -999


@compile
def choose_value(branch: i32) -> i64:
    values: array[PhiValue, 2]
    # Only the initialized value field is read. The tag is indeterminate.
    values[0].value = 37
    values[1].value = 43
    table: array[ptr[PhiValue], 2]
    table[0] = ptr(values[0])
    table[1] = ptr(values[1])
    choice: PhiChoice = (ptr(table[0]), branch, branch)
    return load_choice(ptr(choice))


class LoopCarriedPointerPhiTest(unittest.TestCase):
    def test_each_branch_preserves_its_pointer_across_the_dispatch_loop(self):
        self.assertEqual([choose_value(branch) for branch in (1, 0, 1, 0)],
                         [43, 37, 43, 37])


if __name__ == '__main__':
    unittest.main()
