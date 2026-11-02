"""Generic groups have physical provenance and independent specializations."""

from __future__ import annotations

from pathlib import Path
import unittest

from pythoc import compile, i32, i64, param
from pythoc.session import CompileSession


class ParametricGroupIdentityTest(unittest.TestCase):
    def setUp(self):
        session = CompileSession()
        session.__enter__()
        self.addCleanup(session.__exit__, None, None, None)

    def test_specialization_keeps_the_template_physical_source(self):
        @compile
        def identity(T: param, value: T) -> T:
            return value

        function = identity(i32)
        self.assertEqual(Path(function._binding.source_file), Path(__file__).resolve())
        self.assertEqual(function._binding.group_key[0], function._binding.source_file)

    def test_functions_and_type_arguments_have_distinct_groups(self):
        @compile
        def first(T: param, value: T) -> T:
            return value

        @compile
        def second(T: param, value: T) -> T:
            return value

        groups = {function._binding.group_key for function in
                  (first(i32), first(i64), second(i32))}
        self.assertEqual(len(groups), 3)


if __name__ == '__main__':
    unittest.main()
