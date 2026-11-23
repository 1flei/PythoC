"""Option/Result helpers that preserve native payload ownership.

Copyable sums support consuming predicates and eager defaults. Linear sums
use tag projection (value[0]) for inspection and lazy fallbacks for extraction.
Result bind/do use a copyable error channel; owned errors require an explicit
match or fallback consumer. Result.do installs a frame-local propagation
channel. Standalone bind requires an explicitly selected error-slot policy;
importing this module never installs a global error channel.
"""

import sys

from ..builtin_entities import i8, static, ptr, array, typeof
from ..builtin_entities.enum import _create_enum_type
from ..builtin_entities.types import bool as pc_bool
from ..builtin_entities.unreachable import _require_unreachable
from ..effect import effect
from ..logger import logger
from ..schema_protocol import is_linear_schema_type
from ..source import SourceLocation, _diagnose_expansion_at_call
from ..type_descriptor import format_type, same_type
from ..utils import get_next_id
from pythoc import compile, inline, label, goto_end


class ErrnoSlotProvider:
    """Optional error-slot policy using static (global) storage.

    Provides per-size errno slots: different err_types with the same
    sizeof share the same global storage -- like C errno but generalized.

    Select it explicitly with effect(errno=provider, suffix=...). It is not
    installed by importing Result. These are PythoC slots, not native C errno.
    A caller can instead supply thread-local or fiber-local slot storage.
    """

    def __init__(self):
        self._slots = {}

    def get_slot(self, err_type):
        """Get or create a slot function for the given err_type.

        Returns a @compile function: () -> ptr[array[i8, size]]
        """
        size = err_type.get_size_bytes()
        if size in self._slots:
            return self._slots[size]

        slot_type = array[i8, size]

        @compile(suffix=("errno_slot", size), attrs={'readnone', 'nounwind'})
        def _slot_fn() -> ptr[slot_type]:
            _slot: static[slot_type]
            return ptr(_slot)

        self._slots[size] = _slot_fn
        return _slot_fn


class _LocalSlotProvider:
    """Internal result propagation channel captured by a local closure.

    Used by result_do: the err channel lives in the composing function's
    frame (an alloca promoted to a register). The slot function is a
    compile-level closure capturing the channel by reference.
    """

    def __init__(self, err_type, slot_fn):
        self.err_type = err_type
        self.exit_label = f"__result_failure_{get_next_id()}"
        self._slot_fn = slot_fn

    @staticmethod
    def failure_exit(get_slot):
        provider = getattr(get_slot, '__self__', None)
        return provider.exit_label if isinstance(provider, _LocalSlotProvider) else None

    def get_slot(self, err_type):
        if not same_type(err_type, self.err_type):
            logger.error(
                'Result.do requires the same error type in every bind '
                f'(expected {format_type(self.err_type)}, got {format_type(err_type)})',
                node=logger.current_location(), exc_type=TypeError,
            )
        return self._slot_fn


def Option(value_type, tag_type=i8, name=None):
    """Create an Option type carrying its constructors and operations.

    Linear payloads support constructors, bind/do and unwrap_or_else.
    Copyable payloads also support is_some/is_none and unwrap_or.

    Returns:
        OptionType
    """
    node = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
    option_name = name or f"Option_{getattr(value_type, '__name__', str(value_type))}"
    option_type = _create_enum_type(
        [("Some", value_type, 0), ("NoneVal", None, 1)],
        tag_type,
        class_name=option_name, node=node,
    )

    suffix_base = (option_name, value_type, tag_type)

    @compile(suffix=(suffix_base, "some"))
    def option_some(x: value_type) -> option_type:
        return option_type(option_type.Some, x)

    @compile(suffix=(suffix_base, "none"))
    def option_none() -> option_type:
        return option_type(option_type.NoneVal)

    @compile(suffix=(suffix_base, "bind"))
    def option_bind(o: option_type) -> value_type:
        match o:
            case (option_type.Some, value):
                yield value
            case (option_type.NoneVal):
                pass

    @inline
    def option_do(genexp, _option_type=option_type) -> option_type:
        for value in genexp:
            return _option_type(_option_type.Some, value)
        return _option_type(_option_type.NoneVal)

    @inline
    def option_unwrap_or_else(o: option_type, fallback) -> value_type:
        match o:
            case (option_type.Some, value):
                return value
            case option_type.NoneVal:
                return fallback()

    @inline
    @_diagnose_expansion_at_call
    def option_from_result(r, drop_error=None) -> option_type:
        source = _conversion_source(typeof(r), 'result', value_type, drop_error)
        match r:
            case (source.Ok, value):
                return option_some(value)
            case (source.Err, error):
                if drop_error is not None:
                    drop_error(error)
                return option_none()

    option_type.some = option_some
    option_type.none = option_none
    option_type.bind = option_bind
    option_type.do = option_do
    option_type.unwrap_or_else = option_unwrap_or_else
    option_type.from_result = option_from_result

    if not is_linear_schema_type(option_type):
        @compile(suffix=(suffix_base, "is_some"))
        def option_is_some(o: option_type) -> pc_bool:
            match o:
                case (option_type.Some, _):
                    return True
                case (option_type.NoneVal):
                    return False

        @compile(suffix=(suffix_base, "is_none"))
        def option_is_none(o: option_type) -> pc_bool:
            return not option_is_some(o)

        @compile(suffix=(suffix_base, "unwrap_or"))
        def option_unwrap_or(o: option_type, default: value_type) -> value_type:
            match o:
                case (option_type.Some, v):
                    return v
                case (option_type.NoneVal):
                    return default

        option_type.is_some = option_is_some
        option_type.is_none = option_is_none
        option_type.unwrap_or = option_unwrap_or

    return option_type


def Result(ok_type, err_type, tag_type=i8, name=None):
    """Create a Result type carrying its constructors and operations.

    All payloads support constructors and unwrap_or_else. Bind/do require
    a copyable error. Every reachable path through a do producer must yield
    a value or propagate a bind error of exactly err_type. The first such
    error exits this do. is_ok/is_err and unwrap_or require a copyable sum.

    Returns:
        ResultType
    """
    node = logger.current_location() or SourceLocation.from_frame(sys._getframe(1))
    result_name = name or f"Result_{getattr(ok_type, '__name__', str(ok_type))}_{getattr(err_type, '__name__', str(err_type))}"
    result_type = _create_enum_type(
        [("Ok", ok_type, 0), ("Err", err_type, 1)],
        tag_type,
        class_name=result_name, node=node,
    )

    suffix_base = (result_name, ok_type, err_type, tag_type)

    @compile(suffix=(suffix_base, "ok"))
    def result_ok(x: ok_type) -> result_type:
        return result_type(result_type.Ok, x)

    @compile(suffix=(suffix_base, "err"))
    def result_err(e: err_type) -> result_type:
        return result_type(result_type.Err, e)

    @compile(suffix=(suffix_base, "bind"))
    @_diagnose_expansion_at_call
    def result_bind(r: result_type) -> ok_type:
        match r:
            case (result_type.Ok, value):
                yield value
            case (result_type.Err, err_value):
                # Result.do supplies a local propagation slot. Outside do,
                # the caller must explicitly bind a slot policy; an unbound
                # channel is a compile-time error, never a dropped error.
                get_slot = effect.errno.get_slot
                slot = get_slot(err_type)
                failure_exit = _LocalSlotProvider.failure_exit(get_slot)
                if failure_exit is None:
                    ptr[err_type](slot())[0] = err_value
                else:
                    slot()[0] = err_value
                    goto_end(failure_exit)

    @inline
    @_diagnose_expansion_at_call
    def result_do(
        genexp,
        _result_type=result_type,
    ) -> result_type:
        # The err channel is a frame local of the composing function: the
        # slot function is a compile-level closure capturing it by
        # reference, bound block-scoped so the inlined binds write into it.
        err: err_type

        def _slot() -> ptr[err_type]:
            return ptr(err)

        channel = _LocalSlotProvider(err_type, _slot)
        with label(channel.exit_label):
            with effect(errno=channel):
                for value in genexp:
                    return _result_type(_result_type.Ok, value)
            _require_unreachable("Result.do can finish without yielding a value or propagating an error")

        return _result_type(_result_type.Err, err)

    @inline
    def result_unwrap_or_else(r: result_type, fallback) -> ok_type:
        match r:
            case (result_type.Ok, value):
                return value
            case (result_type.Err, error):
                return fallback(error)

    @inline
    @_diagnose_expansion_at_call
    def result_from_option(o, error: err_type) -> result_type:
        _require_copyable_error(err_type)
        source = _conversion_source(typeof(o), 'option', ok_type)
        match o:
            case (source.Some, value):
                return result_ok(value)
            case source.NoneVal:
                return result_err(error)

    @inline
    @_diagnose_expansion_at_call
    def result_from_option_else(o, make_error) -> result_type:
        source = _conversion_source(typeof(o), 'option', ok_type)
        match o:
            case (source.Some, value):
                return result_ok(value)
            case source.NoneVal:
                return result_err(make_error())

    result_type.ok = result_ok
    result_type.err = result_err
    result_type.unwrap_or_else = result_unwrap_or_else
    result_type.from_option = result_from_option
    result_type.from_option_else = result_from_option_else

    # The local or explicitly selected channel contains copyable errors.
    # Owned errors stay in the sum until a match or fallback consumes them.
    if not is_linear_schema_type(err_type):
        result_type.bind = result_bind
        result_type.do = result_do

    if not is_linear_schema_type(result_type):
        @compile(suffix=(suffix_base, "is_ok"))
        def result_is_ok(r: result_type) -> pc_bool:
            match r:
                case (result_type.Ok, _):
                    return True
                case (result_type.Err, _):
                    return False

        @compile(suffix=(suffix_base, "is_err"))
        def result_is_err(r: result_type) -> pc_bool:
            return not result_is_ok(r)

        @compile(suffix=(suffix_base, "unwrap_or"))
        def result_unwrap_or(r: result_type, default: ok_type) -> ok_type:
            match r:
                case (result_type.Ok, v):
                    return v
                case (result_type.Err, _):
                    return default

        result_type.is_ok = result_is_ok
        result_type.is_err = result_is_err
        result_type.unwrap_or = result_unwrap_or

    return result_type


def _conversion_source(source, kind, value_type, drop_error=None):
    """Read conversion metadata from the actual source enum type."""
    node = logger.current_location()
    names = ['Some', 'NoneVal'] if kind == 'option' else ['Ok', 'Err']
    if not getattr(source, '_is_enum', False) or source._variant_names != names:
        expected = 'an Option' if kind == 'option' else 'a Result'
        logger.error(f'Expected {expected} source, got {format_type(source)}',
                     node=node, exc_type=TypeError)
    source_value_type = source._variant_types[0]
    if not same_type(source_value_type, value_type):
        logger.error(
            'Sum conversion requires the same value type '
            f'(expected {format_type(value_type)}, got {format_type(source_value_type)})',
            node=node, exc_type=TypeError,
        )
    if kind == 'option' and source._variant_types[1] is not None:
        logger.error('Option NoneVal must have no payload', node=node, exc_type=TypeError)
    if (kind == 'result' and is_linear_schema_type(source._variant_types[1])
            and drop_error is None):
        logger.error('from_result requires an explicit consumer for an owned error',
                     node=node, exc_type=TypeError)
    return source


def _require_copyable_error(err_type):
    if is_linear_schema_type(err_type):
        logger.error('from_option requires a copyable error. Use from_option_else.',
                     node=logger.current_location(), exc_type=TypeError)


__all__ = ['Option', 'Result', 'ErrnoSlotProvider']
