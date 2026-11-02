"""Hold native specializations while another process requests the same code."""

import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])

from pythoc import i8, i32, i64


def artifact_state(function):
    library = Path(function._binding.so_file).resolve()
    obj = library.with_suffix('.o')
    return {str(path): path.stat().st_mtime_ns for path in (obj, library)}


def main():
    root = Path(sys.argv[1])
    mode = sys.argv[2]
    offset = int(sys.argv[3])
    library = sys.argv[4]
    previous = json.loads((root / 'ready.json').read_text()) if mode == 'reader' else None

    if library == 'generic':
        from shared_definition import identity, increment

        first = identity(i64)
        assert first(41) == 41 + offset
        # Creating these after the first call must keep loaded groups closed.
        second = identity(i32)
        third = increment(i64)
        calls = [(first, (41,), 41 + offset),
                 (second, (17,), 17 + offset), (third, (40,), 42)]
    else:
        if library == 'set':
            from pythoc.std.set import _DEFAULT_HASH, _DEFAULT_EQ
        else:
            from pythoc.std.map import _DEFAULT_HASH, _DEFAULT_EQ
        hash_key = _DEFAULT_HASH[i8]
        same_key = _DEFAULT_EQ[i8]
        calls = [(hash_key, (5,), hash_key(5)),
                 (same_key, (5, 5), True), (same_key, (5, 6), False)]

    current = {}
    for function, arguments, expected in calls:
        assert function(*arguments) == expected
        current.update(artifact_state(function))

    if mode == 'holder':
        ready = root / 'ready.tmp'
        ready.write_text(json.dumps(current))
        ready.replace(root / 'ready.json')
        assert sys.stdin.readline().strip() == 'release'
        for function, arguments, expected in calls:
            assert function(*arguments) == expected
    elif previous is not None:
        assert current == previous, 'Rebuilt artifacts already loaded by another process'
    print('CACHE_STATE=' + json.dumps(current, sort_keys=True))


if __name__ == '__main__':
    main()
