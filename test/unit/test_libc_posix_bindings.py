"""Exact signatures and availability of the new POSIX declarations."""

import unittest

from pythoc import func, i32, ptr, void
from test.utils.libc_platform import platform_bindings, signature_types


MODULES = ('stddef', 'sys_types', 'sys_stat', 'unistd', 'pthread',
           'time', 'sys_mman')


class TestPosixBindings(unittest.TestCase):
    def test_signatures_use_c_typedefs(self):
        for platform in ('darwin', 'linux'):
            with self.subTest(platform=platform):
                with platform_bindings(platform, MODULES) as modules:
                    stddef = modules['stddef']
                    types = modules['sys_types']
                    unistd = modules['unistd']
                    for name in ('pread', 'pwrite'):
                        function = getattr(unistd, name)
                        expected = [
                            ('fd', i32), ('buf', ptr[void]),
                            ('count', stddef.size_t), ('offset', types.off_t),
                        ]
                        self.assertEqual(signature_types(function.param_types),
                                         signature_types(expected))
                        self.assertIs(function.return_type, stddef.ssize_t)
                    self.assertIs(dict(unistd.ftruncate.param_types)['length'],
                                  types.off_t)
                    self.assertEqual(unistd.fchown.param_types, [
                        ('fd', i32), ('owner', types.uid_t),
                        ('group', types.gid_t),
                    ])
                    self.assertIs(unistd.geteuid.return_type, types.uid_t)
                    thread = modules['pthread']
                    create_parameters = [
                        ('thread', ptr[types.pthread_t]), ('attr', ptr[void]),
                        ('start', func[ptr[void], ptr[void]]),
                        ('arg', ptr[void]),
                    ]
                    self.assertEqual(
                        signature_types(thread.pthread_create.param_types),
                        signature_types(create_parameters),
                    )
                    join_parameters = [
                        ('thread', types.pthread_t),
                        ('retval', ptr[ptr[void]]),
                    ]
                    self.assertEqual(
                        signature_types(thread.pthread_join.param_types),
                        signature_types(join_parameters),
                    )
                    stat = modules['sys_stat']
                    self.assertIs(dict(stat.fchmod.param_types)['mode'],
                                  types.mode_t)
                    request = dict(modules['time'].nanosleep.param_types)['req']
                    self.assertIs(request.pointee_type, stat.timespec)

    def test_mremap_is_linux_only_with_unsigned_sizes(self):
        for platform in ('darwin', 'linux', 'win32'):
            with self.subTest(platform=platform):
                with platform_bindings(platform, MODULES) as modules:
                    memory = modules['sys_mman']
                    self.assertEqual(hasattr(memory, 'mremap'),
                                     platform == 'linux')
                    self.assertEqual('mremap' in memory.__all__,
                                     platform == 'linux')
                    if platform == 'linux':
                        arguments = dict(memory.mremap.param_types)
                        for name in ('old_size', 'new_size'):
                            self.assertIs(arguments[name],
                                          modules['stddef'].size_t)

    def test_windows_does_not_advertise_new_posix_symbols(self):
        with platform_bindings('win32', MODULES) as modules:
            expected_absent = {
                'unistd': ('_SC_PAGESIZE', '_SC_PAGE_SIZE', 'fsync',
                           'ftruncate', 'pread', 'pwrite', 'fchown',
                           'geteuid', 'rmdir'),
                'pthread': ('pthread_create', 'pthread_join'),
                'sys_stat': ('fchmod',),
                'time': ('nanosleep',),
            }
            for name, symbols in expected_absent.items():
                for symbol in symbols:
                    self.assertFalse(hasattr(modules[name], symbol),
                                     (name, symbol))

    def test_unistd_exports_only_public_api(self):
        for platform, pagesize in (('darwin', 29), ('linux', 30)):
            with platform_bindings(platform, MODULES) as modules:
                unistd = modules['unistd']
                self.assertEqual(unistd._SC_PAGESIZE, pagesize)
                self.assertEqual(unistd._SC_PAGE_SIZE, pagesize)
                for name in ('IS_LINUX', 'IS_MACOS', 'extern', 'ptr', 'uid_t'):
                    self.assertNotIn(name, unistd.__all__)
                for name in unistd.__all__:
                    self.assertTrue(hasattr(unistd, name), name)


if __name__ == '__main__':
    unittest.main()
