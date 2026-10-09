"""Check that the Python extension target stays separate from native AOT."""

import os
import sys

from pythoc import compile, compile_to_executable, compile_to_python_extension, i32, i64


@compile
def add(a: i64, b: i64) -> i64:
    return a + b


@compile
def main() -> i32:
    return 0 if add(20, 22) == 42 else 1


def run_checks():
    if os.environ.get('PYTHOC_NATIVE_USE') == '1':
        if not add.is_fast_bound():
            raise SystemExit('installed adapter was not bound at import')
        if add(20, 22) != 42:
            raise SystemExit('installed adapter returned the wrong result')
        return

    out_dir = os.path.join('build', 'python_ext_check')
    os.makedirs(out_dir, exist_ok=True)
    exe = os.path.join(out_dir, 'add_exe')
    # Exercise the Python call path first: the callable runtime and the
    # development adapter groups now exist in the output manager, and the
    # executable link below must still contain only user objects.
    if add(20, 22) != 42:
        raise SystemExit('development call returned the wrong result')
    compile_to_executable(output_path=exe)
    native_symbols = _symbols(exe)
    if any(name.startswith('PyInit_') or name.startswith('pythoc_pyadapter_')
           for name in native_symbols):
        raise SystemExit('native executable contains a Python adapter symbol')
    if 'add' not in native_symbols:
        raise SystemExit('native executable is missing the kernel symbol')

    compile_to_python_extension(
        add,
        output_path=os.path.join(out_dir, '_pythoc_native'),
        module_name='_pythoc_native',
    )
    print('AOT_OK')


def _symbols(path):
    import subprocess
    result = subprocess.run(
        ['nm', path],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    names = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts:
            # Mach-O prefixes external symbols with an underscore.
            names.add(parts[-1].lstrip('_'))
    return names


if __name__ == '__main__':
    run_checks()
