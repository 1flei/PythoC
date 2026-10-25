"""AOT links transitive PythoC standard-library groups."""

import os
import subprocess
import sys

from pythoc import compile, compile_to_executable, i32, u64
from pythoc.std.set import _mix64
from pythoc.utils.link_utils import get_executable_extension


@compile
def main() -> i32:
    value: u64 = _mix64(u64(1), u64(2))
    if value == u64(0):
        return i32(1)
    return i32(0)


if __name__ == '__main__':
    output = os.path.abspath(os.path.join(
        'build',
        'test',
        'integration',
        'aot_std_dependency' + get_executable_extension(),
    ))
    compile_to_executable(output_path=output)
    result = subprocess.run([output], check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
