"""Direct noreturn calls terminate the caller's control-flow path."""

import unittest

from pythoc import bool, compile, defer, goto, i32, inline, label, void
from pythoc.libc.stdlib import abort
from pythoc.valueref import wrap_python_constant


@compile
def selected_value(terminate: bool) -> i32:
    if not terminate:
        return 17
    abort()


@compile
def selected_return(terminate: bool) -> void:
    if terminate:
        return abort()


@compile
def do_nothing() -> void:
    pass


@compile
def conditional_void(terminate: bool) -> i32:
    abort() if terminate else do_nothing()
    return 23


@compile
def conditional_value(terminate: bool) -> i32:
    return i32(31) if not terminate else abort()


@compile(attrs={'noreturn'})
def typed_stop() -> i32:
    abort()


@compile
def terminating_condition(terminate: bool) -> i32:
    if terminate:
        if abort():
            return 10
    if terminate:
        while abort():
            return 20
    return 42


@compile
def terminating_operand(terminate: bool) -> i32:
    if terminate:
        return (typed_stop() if terminate else abort()) + i32(1)
    return i32(42)


@compile
def short_circuit(terminate: bool) -> bool:
    return terminate and abort()


@compile
def short_circuit_nested(first: bool, second: bool) -> bool:
    return first or (second and abort())


@compile
def short_circuit_compare(first: i32, second: i32) -> bool:
    return first < second < typed_stop()


@compile
def short_circuit_compare_many(first: i32, second: i32, third: i32) -> bool:
    return first < second < third < typed_stop()


@inline
def inline_stop():
    abort()


@compile
def terminating_inline_operand(terminate: bool) -> i32:
    if terminate:
        return inline_stop() + i32(1)
    return i32(42)


class TerminalContext:
    @staticmethod
    def handle_attribute(visitor, base, name, node):
        return abort.handle_call(visitor, wrap_python_constant(abort), (), node)


@compile
def terminating_attribute(terminate: bool) -> i32:
    if terminate:
        return TerminalContext.stop + i32(1)
    return i32(42)


@compile
def terminating_with_context(terminate: bool) -> i32:
    if terminate:
        with abort():
            return 1
    return 42


@compile
def deferred_termination(terminate: bool) -> i32:
    if terminate:
        defer(abort)
        goto('finish')
    with label('finish'):
        return i32(42)


class NoreturnCallsTest(unittest.TestCase):
    def test_noreturn_call_does_not_require_a_return_value(self):
        self.assertEqual(selected_value(False), 17)
        selected_return(False)

    def test_conditional_expression_only_joins_reachable_arms(self):
        self.assertEqual(conditional_void(False), 23)
        self.assertEqual(conditional_value(False), 31)

    def test_expression_consumers_stop_when_the_operand_terminates(self):
        self.assertEqual(terminating_condition(False), 42)
        self.assertEqual(terminating_operand(False), 42)
        self.assertEqual(terminating_inline_operand(False), 42)
        self.assertEqual(terminating_attribute(False), 42)
        self.assertEqual(terminating_with_context(False), 42)
        self.assertFalse(short_circuit(False))
        self.assertTrue(short_circuit_nested(True, True))
        self.assertFalse(short_circuit_nested(False, False))
        self.assertFalse(short_circuit_compare(2, 1))
        self.assertFalse(short_circuit_compare_many(2, 1, 3))
        self.assertFalse(short_circuit_compare_many(1, 3, 2))

    def test_forward_goto_stops_after_terminating_defer(self):
        self.assertEqual(deferred_termination(False), 42)


if __name__ == '__main__':
    unittest.main()
