"""Generator locals remain lexical through match and nested functions."""

import unittest

from pythoc import compile, i8, i32, ptr
from pythoc.build.output_manager import flush_all_pending_outputs
from pythoc.libc.string import strlen
from pythoc.std.result import result_wrap
from test.utils.test_utils import DeferredTestCase


@compile
class Marker:
    value: i32


Outcome, outcome = result_wrap(i32, Marker, name='YieldHygieneOutcome')


@compile
def fallible_values() -> Outcome:
    result = outcome.ok(7)
    match result:
        case (Outcome.Err, name):
            yield outcome.err(name)
            return
        case Outcome.Ok:
            pass
    yield result


@compile
def pointer_consumer(name: ptr[i8]) -> i32:
    for result in fallible_values():
        return i32(strlen(name))
    return 0


@compile
def pattern_values() -> i32:
    result = outcome.ok(7)
    match result:
        case (Outcome.Ok, name):
            yield name
        case Outcome.Err:
            pass


@compile
def scalar_consumer(name: i32) -> i32:
    for current in pattern_values():
        return name + current
    return 0


@compile
def assignment_values() -> i32:
    name = 7
    yield name


@compile
def assignment_consumer(name: i32) -> i32:
    for current in assignment_values():
        return name + current
    return 0


@compile
def parameter_values() -> i32:
    result = outcome.ok(7)
    match result:
        case (Outcome.Ok, name):
            def inner(name: i32) -> i32:
                return name + 1
            yield inner(name=3) + name
        case Outcome.Err:
            pass


@compile
def parameter_consumer(name: i32) -> i32:
    for current in parameter_values():
        return name + current
    return 0


@compile
def annotated_values() -> i32:
    result = outcome.ok(7)
    match result:
        case (Outcome.Ok, name):
            def inner() -> i32:
                name: i32 = 3
                return name + 1
            yield inner() + name
        case Outcome.Err:
            pass


@compile
def annotated_consumer(name: i32) -> i32:
    for current in annotated_values():
        return name + current
    return 0


@compile
def captured_values() -> i32:
    result = outcome.ok(7)
    match result:
        case (Outcome.Ok, name):
            def inner() -> i32:
                name += 2
                return name
            current: i32 = inner()
            yield current + name
        case Outcome.Err:
            pass


@compile
def captured_consumer(name: i32) -> i32:
    for current in captured_values():
        return name + current
    return 0


@compile
def closure_consumer(name: i32) -> i32:
    def values() -> i32:
        name += 2
        result = outcome.ok(7)
        match result:
            case (Outcome.Ok, increment):
                name += increment
                yield name
            case Outcome.Err:
                pass
    for current in values():
        return name * 100 + current
    return 0


@compile
def closure_pattern_consumer(name: i32) -> i32:
    def values() -> i32:
        result = outcome.ok(7)
        match result:
            case (Outcome.Ok, name):
                yield name
            case Outcome.Err:
                pass
    for current in values():
        return name * 100 + current
    return 0


@compile
def reused_loop_target() -> i32:
    total: i32 = 0
    for current in assignment_values():
        total += current
    for current in assignment_values():
        total += current
    return total


class YieldBindingHygieneTest(DeferredTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        flush_all_pending_outputs()

    def test_unexecuted_pattern_cannot_replace_caller_pointer(self):
        self.assertEqual(pointer_consumer(b'hello'), 5)

    def test_pattern_cannot_overwrite_same_typed_caller_parameter(self):
        self.assertEqual(scalar_consumer(41), 48)

    def test_unannotated_generator_assignment_remains_local(self):
        self.assertEqual(assignment_consumer(41), 48)

    def test_nested_parameter_keeps_its_name_and_shadows_pattern(self):
        self.assertEqual(parameter_consumer(41), 52)

    def test_nested_annotated_local_shadows_pattern(self):
        self.assertEqual(annotated_consumer(41), 52)

    def test_nested_function_updates_its_lexical_generator_binding(self):
        self.assertEqual(captured_consumer(41), 59)

    def test_generator_closure_retains_by_reference_capture(self):
        self.assertEqual(closure_consumer(41), 5050)

    def test_closure_pattern_binding_does_not_replace_consumer_binding(self):
        self.assertEqual(closure_pattern_consumer(41), 4107)

    def test_consumer_loop_target_can_be_reused(self):
        self.assertEqual(reused_loop_target(), 14)


if __name__ == '__main__':
    unittest.main()
