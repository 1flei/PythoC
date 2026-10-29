"""Use Python signature binding as an independent parameter-kind oracle."""

import ast
import inspect
import unittest
from unittest.mock import patch

from pythoc.builtin_entities.python_type import PythonType
from pythoc.call_normalization import (
    _build_sig_from_ast, bind_call_args,
)
from pythoc.literal_protocol import get_mapping_entries, get_sequence_elements
from pythoc.logger import logger
from pythoc.valueref import wrap_value


def mixed_parameters(a, /, b=2, *items, c, d=4, **named):
    pass


def keyword_parameters(a, *, b=8):
    pass


def no_parameters():
    pass


def value_ref(value):
    return wrap_value(value, kind='python', type_hint=PythonType.wrap(value))


def bind(function, positional, keywords):
    arguments = ast.parse(inspect.getsource(function)).body[0].args
    defaults = {
        name: value_ref(parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    keyword_nodes = [ast.keyword(arg=name, value=ast.Name(id=name))
                     for name in keywords]
    return bind_call_args(
        _build_sig_from_ast(arguments, defaults),
        [value_ref(value) for value in positional], keyword_nodes,
        lambda expression: value_ref(keywords[expression.id]), None)


class CallBindingTest(unittest.TestCase):
    def test_mixed_parameter_binding_matches_python(self):
        cases = [
            ((1,), {'c': 3}),
            ((1, 2, 5, 6), {'c': 3, 'd': 7, 'extra': 9}),
            ((1,), {'b': 7, 'c': 3, 'a': 9}),
        ]
        for positional, keywords in cases:
            with self.subTest(positional=positional, keywords=keywords):
                expected = inspect.signature(mixed_parameters).bind(
                    *positional, **keywords)
                expected.apply_defaults()
                actual = bind(mixed_parameters, positional, keywords)
                self.assertEqual(
                    [value.get_python_value() for value in actual[:4]],
                    [expected.arguments[name]
                     for name in ('a', 'b', 'c', 'd')])
                self.assertEqual(
                    tuple(value.get_python_value() for value in
                          get_sequence_elements(actual[4].get_python_value())),
                    expected.arguments['items'])
                self.assertEqual(
                    {key.get_python_value(): value.get_python_value()
                     for key, value in
                     get_mapping_entries(actual[5].get_python_value())},
                    expected.arguments['named'])

    def test_invalid_parameter_binding_matches_python_rejection(self):
        cases = [
            (mixed_parameters, (), {'a': 1, 'c': 3}),
            (mixed_parameters, (1,), {}),
            (mixed_parameters, (1, 2), {'b': 3, 'c': 4}),
            (keyword_parameters, (1, 2), {}),
            (keyword_parameters, (), {'a': 1, 'unknown': 2}),
            (no_parameters, (1,), {}),
        ]
        for function, positional, keywords in cases:
            with self.subTest(function=function.__name__,
                              positional=positional, keywords=keywords):
                with self.assertRaises(TypeError):
                    inspect.signature(function).bind(*positional, **keywords)
                with patch.object(logger, 'raise_on_error', True):
                    with self.assertRaises(TypeError):
                        bind(function, positional, keywords)

    def test_empty_signature_is_still_bound(self):
        self.assertEqual(bind(no_parameters, (), {}), [])


if __name__ == '__main__':
    unittest.main()
