"""Changing a captured type must invalidate an otherwise identical cache slot."""

import sys

from pythoc import compile, i32, i64, sizeof


Selected = i32 if sys.argv[1] == '32' else i64


@compile(suffix='same_cache_slot')
def read_size() -> i32:
    return sizeof(Selected)


if __name__ == '__main__':
    print('SIZE_RESULT', int(read_size()))
