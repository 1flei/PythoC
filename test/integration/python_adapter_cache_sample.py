"""Report development and AOT adapter artifacts for cache tests."""

import json
import os

from pythoc import compile, compile_to_python_extension, i64
from pythoc.python_adapter import (
    _adapter_group_key,
    _adapter_group_object,
    _adapter_object_path,
    _init_symbol,
    function_spec,
)


@compile
def cached_add(a: i64, b: i64) -> i64:
    return a + b


def _artifact_state(path):
    absolute = os.path.abspath(path)
    return {
        'path': absolute,
        'mtime_ns': os.stat(absolute).st_mtime_ns,
    }


def main():
    if cached_add(20, 22) != 42:
        raise SystemExit('development adapter returned the wrong result')

    development_so = cached_add._pythoc_adapter_lib._name
    specs = [function_spec(cached_add)]
    development_key = _adapter_object_path(development_so, specs)
    development_object = _adapter_group_object(
        _adapter_group_key(development_key)
    )

    extension = compile_to_python_extension(
        cached_add,
        output_path=os.path.join(
            'build',
            'python_adapter_cache',
            '_cache_sample',
        ),
        module_name='_cache_sample',
    )
    extension_key = _adapter_object_path(
        extension,
        specs,
        init_symbol=_init_symbol('_cache_sample'),
        module_name='_cache_sample',
    )
    extension_object = _adapter_group_object(
        _adapter_group_key(extension_key)
    )

    state = {
        'development_object': _artifact_state(development_object),
        'development_so': _artifact_state(development_so),
        'extension_object': _artifact_state(extension_object),
        'extension_so': _artifact_state(extension),
    }
    print('CACHE_STATE=' + json.dumps(state, sort_keys=True))


if __name__ == '__main__':
    main()
