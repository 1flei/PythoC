"""C ABI scratch storage must remain bounded in long-running loops."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import tempfile
import threading
import unittest

from pythoc import array, compile, extern, i32, i64, seq
from pythoc.utils.cc_utils import compile_c_to_object
from pythoc.utils.link_utils import get_shared_lib_extension, link_files


_temporary = tempfile.TemporaryDirectory(prefix='pc-abi-loop-')
_root = Path(_temporary.name)
_source = Path(__file__).parent / 'fixtures/abi_loop_stack.c'
_object = compile_c_to_object(str(_source), str(_root / 'oracle.o'))
_library = link_files([_object], str(_root / ('oracle' + get_shared_lib_extension())),
                      shared=True, link_objects=[])


@compile
class Record:
    fields: array[i64, 8]


@extern(lib=_library)
def make_stack_record(value: i32) -> Record:
    pass


@extern(lib=_library)
def sum_stack_record(value: Record) -> i64:
    pass


# Each temporary oracle has a distinct link dependency.
@compile(suffix=str(_root))
def loop(count: i32) -> i64:
    total: i64 = 0
    for index in seq(count):
        total += sum_stack_record(make_stack_record(index))
    return total


class AbiLoopStackTest(unittest.TestCase):
    def test_repeated_indirect_returns_and_by_value_arguments_use_bounded_stack(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child'],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def check_bounded_loop():
    count = 500000
    assert loop(count) == count * (count - 1) + count * 7


if __name__ == '__main__':
    if sys.argv[1:] == ['--child']:
        assert loop(1) == 7
        threading.stack_size(1024 * 1024)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(check_bounded_loop).result()
    else:
        unittest.main()
