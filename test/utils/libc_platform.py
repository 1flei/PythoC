"""Load real binding sources with isolated platform flags and registries."""

from contextlib import contextmanager, ExitStack
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

from pythoc import libc
from pythoc.libc import _platform
from pythoc.session import CompileSession
from pythoc.type_id import get_type_id


def signature_types(parameters):
    return [(name, get_type_id(value)) for name, value in parameters]


@contextmanager
def platform_bindings(platform, names):
    with CompileSession(), ExitStack() as stack:
        stack.enter_context(patch.multiple(
            _platform, IS_MACOS=platform == 'darwin',
            IS_LINUX=platform == 'linux', IS_WINDOWS=platform == 'win32',
        ))
        modules = {}
        for name in names:
            qualified = 'pythoc.libc.' + name
            source = Path(libc.__file__).with_name(name + '.py')
            spec = importlib.util.spec_from_file_location(qualified, source)
            module = importlib.util.module_from_spec(spec)
            stack.enter_context(patch.dict(sys.modules, {qualified: module}))
            stack.enter_context(patch.object(libc, name, module, create=True))
            spec.loader.exec_module(module)
            modules[name] = module
        yield modules
