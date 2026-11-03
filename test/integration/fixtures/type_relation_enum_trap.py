"""A dynamic tag must select a variant that accepts the supplied payload."""

import resource
import sys

from pythoc import compile, i32
from pythoc.builtin_entities.enum import _create_enum_type


resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
Result = _create_enum_type([('Ok', i32, 0), ('Empty', None, 1)], i32)


@compile
def construct(tag: i32) -> i32:
    value: Result = Result(tag, 7)
    return i32(value[0])


if __name__ == '__main__':
    assert construct(int(sys.argv[1])) == 0
