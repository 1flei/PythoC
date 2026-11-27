import ast
import inspect
import sys
from types import ModuleType
from unittest.mock import patch

from pythoc.utils.inspect_utils import get_object_sourcelines


def test_class_sources_share_module_parse(tmp_path, monkeypatch):
    path = tmp_path / "indexed_classes.py"
    source = """\
def identity(value):
    return value

@identity
class Outer:
    class Nested:
        value = 1
    # Keep this final comment in the original class block.

def factory():
    @identity
    class Local:
        value = 2
    return Local
"""
    path.write_text(source)
    module = ModuleType("indexed_classes")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, str(path), "exec"), vars(module))
    classes = (module.Outer, module.Outer.Nested, module.factory())
    expected = [inspect.getsourcelines(cls) for cls in classes]

    with patch("pythoc.utils.inspect_utils.ast.parse", wraps=ast.parse) as parse:
        assert [get_object_sourcelines(cls) for cls in classes] == expected
        assert parse.call_count == int(not hasattr(classes[0], '__firstlineno__'))
