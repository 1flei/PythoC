import unittest

from llvmlite import binding

from pythoc.compiler import (
    _target_machine_cache_key,
    target_machine_identity,
)
from pythoc.config import config


class TestTargetMachineIdentity(unittest.TestCase):
    def tearDown(self):
        config.reset('target_cpu')
        config.reset('target_features')

    def test_generic_target_is_portable(self):
        with config.override(
            target_cpu='generic',
            target_features='generic',
        ):
            self.assertEqual(target_machine_identity(), ('', ''))

    def test_native_target_resolves_host_identity(self):
        with config.override(
            target_cpu='native',
            target_features='native',
        ):
            self.assertEqual(
                target_machine_identity(),
                (
                    binding.get_host_cpu_name(),
                    binding.get_host_cpu_features().flatten(),
                ),
            )

    def test_explicit_target_is_preserved(self):
        with config.override(
            target_cpu='x86-64-v3',
            target_features='+avx2,-avx512f',
        ):
            self.assertEqual(
                target_machine_identity(),
                ('x86-64-v3', '+avx2,-avx512f'),
            )

    def test_target_is_part_of_machine_cache_key(self):
        with config.override(
            target_cpu='generic',
            target_features='generic',
        ):
            generic = _target_machine_cache_key('test-triple')
        with config.override(
            target_cpu='native',
            target_features='native',
        ):
            native = _target_machine_cache_key('test-triple')
        self.assertNotEqual(generic, native)


if __name__ == '__main__':
    unittest.main()
