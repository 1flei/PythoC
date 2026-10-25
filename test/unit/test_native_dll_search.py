"""External DLL search follows the same persisted inputs as linking."""

from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pythoc.native_executor import MultiSOExecutor


class TestNativeDllSearch(unittest.TestCase):
    def test_external_library_directory_is_available_before_loading_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            kernel = Path(temp) / 'kernel' / 'main.dll'
            external = Path(temp) / 'external' / 'probe.dll'
            for path in (kernel, external):
                path.parent.mkdir()
                path.touch()
            executor = MultiSOExecutor()
            handle = object()
            with patch('sys.platform', 'win32'), patch.dict(os.environ), \
                    patch.object(executor, '_get_persisted_link_libraries',
                                 return_value=[str(external), 'c']) as inputs, \
                    patch('os.add_dll_directory', create=True) as add, \
                    patch('pythoc.utils.link_utils.file_lock',
                          return_value=nullcontext()), \
                    patch('pythoc.utils.link_utils.ensure_link_objects_loaded'), \
                    patch('ctypes.CDLL', return_value=handle) as load:
                def check_directories(*args, **kwargs):
                    self.assertEqual(
                        {call.args[0] for call in add.call_args_list},
                        {str(kernel.parent), str(external.parent)},
                    )
                    return handle

                load.side_effect = check_directories
                self.assertIs(executor._load_single_library(
                    str(kernel), str(kernel)), handle)
                inputs.assert_called_once_with(str(kernel.with_suffix('.o')))


if __name__ == '__main__':
    unittest.main()
