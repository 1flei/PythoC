"""Unit tests for pythoc.config -- the centralised configuration store."""

import os
import unittest
from unittest.mock import patch

from pythoc.config import config, _SCHEMA


class TestConfigSchema(unittest.TestCase):
    """Every advertised knob is registered and exposes the expected
    metadata."""

    def test_known_knobs_present(self):
        expected = {
            'log_level', 'log_modules', 'raise_on_error',
            'debug_ast', 'debug_ast_format', 'debug_ast_diff',
            'save_ir', 'save_unopt_ir', 'opt_level', 'debug_info',
            'cimport_backend',
            'cimport_target', 'cimport_sysroot', 'libclang_path',
            'cimport_clang_args',
        }
        self.assertEqual(expected, set(_SCHEMA))

    def test_each_knob_has_complete_metadata(self):
        for name, knob in _SCHEMA.items():
            self.assertTrue(knob.env.startswith('PC_'),
                            f"{name}: env var should start with PC_")
            self.assertTrue(knob.doc.strip(),
                            f"{name}: doc must be non-empty")
            self.assertTrue(knob.effective.strip(),
                            f"{name}: effective must be non-empty")


class TestEnvFallback(unittest.TestCase):
    """When no programmatic override is set, the env var wins."""

    def setUp(self):
        # Clear any in-memory overrides accumulated by other tests.
        config.reset()

    def tearDown(self):
        config.reset()

    def test_bool_truthy_strings(self):
        for truthy in ('1', 'true', 'yes', 'on', 'True', 'YES'):
            with patch.dict(os.environ, {'PC_SAVE_IR': truthy}):
                config.reset('save_ir')
                self.assertTrue(config.save_ir, f"expected truthy for {truthy!r}")

    def test_bool_falsy_strings(self):
        for falsy in ('0', 'false', 'no', 'off', '', 'OFF'):
            with patch.dict(os.environ, {'PC_SAVE_IR': falsy}):
                config.reset('save_ir')
                self.assertFalse(config.save_ir, f"expected falsy for {falsy!r}")

    def test_int_coercion(self):
        with patch.dict(os.environ, {'PC_OPT_LEVEL': '3'}):
            config.reset('opt_level')
            self.assertEqual(config.opt_level, 3)

    def test_int_bad_value_falls_back_to_default(self):
        with patch.dict(os.environ, {'PC_OPT_LEVEL': 'notanint'}):
            config.reset('opt_level')
            self.assertEqual(config.opt_level, 2)  # schema default

    def test_str_passthrough(self):
        with patch.dict(os.environ, {'PC_LOG_MODULES': 'build,-build.cache'}):
            config.reset('log_modules')
            self.assertEqual(config.log_modules, 'build,-build.cache')

    def test_empty_env_yields_default(self):
        os.environ.pop('PC_SAVE_IR', None)
        config.reset('save_ir')
        self.assertFalse(config.save_ir)


class TestProgrammaticOverride(unittest.TestCase):
    """Python-level set/override APIs."""

    def setUp(self):
        config.reset()

    def tearDown(self):
        config.reset()

    def test_attribute_set(self):
        config.save_ir = True
        self.assertTrue(config.save_ir)

    def test_explicit_set_wins_over_env(self):
        with patch.dict(os.environ, {'PC_SAVE_IR': '0'}):
            config.save_ir = True
            self.assertTrue(config.save_ir)

    def test_set_bulk(self):
        config.set(save_ir=True, opt_level=0)
        self.assertTrue(config.save_ir)
        self.assertEqual(config.opt_level, 0)

    def test_override_scope_restores_previously_set_value(self):
        config.opt_level = 3
        with config.override(opt_level=0):
            self.assertEqual(config.opt_level, 0)
        self.assertEqual(config.opt_level, 3)

    def test_override_scope_restores_env_fallback(self):
        # Knob was never explicitly set -> override exit must drop the
        # in-memory entry so env fallback resumes.
        with patch.dict(os.environ, {'PC_OPT_LEVEL': '2'}):
            config.reset('opt_level')
            with config.override(opt_level=0):
                self.assertEqual(config.opt_level, 0)
            self.assertEqual(config.opt_level, 2)  # back to env

    def test_override_restores_on_exception(self):
        config.opt_level = 3
        with self.assertRaises(RuntimeError):
            with config.override(opt_level=0):
                self.assertEqual(config.opt_level, 0)
                raise RuntimeError("simulated")
        self.assertEqual(config.opt_level, 3)

    def test_unknown_knob_raises(self):
        with self.assertRaises(AttributeError):
            config.no_such_knob
        with self.assertRaises(AttributeError):
            config.no_such_knob = True
        with self.assertRaises(AttributeError):
            with config.override(no_such_knob=1):
                pass


class TestReset(unittest.TestCase):

    def setUp(self):
        config.reset()

    def tearDown(self):
        config.reset()

    def test_reset_single(self):
        config.set(save_ir=True, opt_level=0)
        config.reset('save_ir')
        # save_ir falls back to env (or schema default), opt_level stays
        self.assertFalse(config.save_ir)
        self.assertEqual(config.opt_level, 0)

    def test_reset_all(self):
        config.set(save_ir=True, opt_level=0)
        config.reset()
        self.assertFalse(config.save_ir)
        self.assertEqual(config.opt_level, 2)


class TestDescribe(unittest.TestCase):

    def test_describe_one(self):
        text = config.describe('save_ir')
        self.assertIn('save_ir', text)
        self.assertIn('PC_SAVE_IR', text)

    def test_describe_all(self):
        text = config.describe()
        self.assertIn('save_ir', text)
        self.assertIn('opt_level', text)
        self.assertIn('log_level', text)


if __name__ == '__main__':
    unittest.main()
