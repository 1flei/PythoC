"""Yield detection respects nested function and lambda scopes."""

import ast
import inspect
import unittest

from pythoc.ast_visitor.yield_transform import analyze_yield_function


def nested_function():
    def inner():
        yield 1


def nested_async_function():
    async def inner():
        yield 1


def nested_lambda():
    inner = lambda: (yield 1)


def outer_and_nested_yields():
    def inner():
        yield 1

    async def another():
        yield 2

    expression = lambda: (yield 3)
    yield 4


class YieldAnalyzerTest(unittest.TestCase):
    def test_nested_producers_do_not_make_the_outer_function_a_generator(self):
        producers = (nested_function, nested_async_function, nested_lambda)
        for producer in producers:
            with self.subTest(producer=producer.__name__):
                function = ast.parse(inspect.getsource(producer)).body[0]
                self.assertIsNone(analyze_yield_function(function))

    def test_outer_yield_is_kept_and_nested_yields_are_not_counted(self):
        source = inspect.getsource(outer_and_nested_yields)
        function = ast.parse(source).body[0]
        analysis = analyze_yield_function(function)
        self.assertIsNotNone(analysis)
        self.assertEqual(len(analysis.yield_nodes), 1)
        self.assertEqual(analysis.yield_nodes[0].value.value, 4)


if __name__ == '__main__':
    unittest.main()
