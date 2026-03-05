"""Option/Result std helpers with do-notation support."""

from ..builtin_entities import i8, static, ptr, array
from ..builtin_entities.enum import _create_enum_type
from ..builtin_entities.types import bool as pc_bool
from ..effect import effect
from ..logger import logger
from pythoc import compile, inline


class _UnsupportedHelper:
    def __init__(self, name):
        self.name = name

    def handle_call(self, visitor, func_ref, args, node):
        logger.error(f"{self.name} is not implemented in phase-1", node=node, exc_type=NotImplementedError)


class ErrnoSlotProvider:
    """Default errno slot provider using static (global) storage.

    Provides per-size errno slots: different err_types with the same
    sizeof share the same global storage -- like C errno but generalized.

    Users can replace this via the effect system to provide thread-local
    or fiber-local slot implementations.
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

        @compile(suffix=("errno_slot", size))
        def _slot_fn() -> ptr[slot_type]:
            _slot: static[slot_type]
            return ptr(_slot)

        self._slots[size] = _slot_fn
        return _slot_fn


effect.default(errno=ErrnoSlotProvider())


def option_wrap(value_type, tag_type=i8, name=None):
    """Create Option enum type and user-space helper namespace.

    Returns:
        (OptionType, option_api)
    """
    option_name = name or f"Option_{getattr(value_type, '__name__', str(value_type))}"
    option_type = _create_enum_type(
        [("Some", value_type, 0), ("NoneVal", None, 1)],
        tag_type,
        class_name=option_name,
    )

    suffix_base = (option_name, value_type, tag_type)

    @compile(suffix=(suffix_base, "some"))
    def option_some(x: value_type) -> option_type:
        return option_type(option_type.Some, x)

    @compile(suffix=(suffix_base, "none"))
    def option_none() -> option_type:
        return option_type(option_type.NoneVal)

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

    class option_api:
        type = option_type
        some = option_some
        none = option_none
        bind = option_bind
        do = option_do
        is_some = option_is_some
        is_none = option_is_none
        unwrap_or = _UnsupportedHelper("option.unwrap_or")
        note = _UnsupportedHelper("option.note")

    return option_type, option_api


def result_wrap(ok_type, err_type, tag_type=i8, name=None):
    """Create Result enum type and user-space helper namespace.

    Returns:
        (ResultType, result_api)
    """
    result_name = name or f"Result_{getattr(ok_type, '__name__', str(ok_type))}_{getattr(err_type, '__name__', str(err_type))}"
    result_type = _create_enum_type(
        [("Ok", ok_type, 0), ("Err", err_type, 1)],
        tag_type,
        class_name=result_name,
    )

    suffix_base = (result_name, ok_type, err_type, tag_type)

    from ..decorators import compile, inline

    @compile(suffix=(suffix_base, "ok"))
    def result_ok(x: ok_type) -> result_type:
        return result_type(result_type.Ok, x)

    @compile(suffix=(suffix_base, "err"))
    def result_err(e: err_type) -> result_type:
        return result_type(result_type.Err, e)

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

    errno_slot = effect.errno.get_slot(err_type)

    @compile(suffix=(suffix_base, "bind"))
    def result_bind(r: result_type) -> ok_type:
        match r:
            case (result_type.Ok, value):
                yield value
            case (result_type.Err, err_value):
                ptr[err_type](errno_slot())[0] = err_value
                pass

    @inline
    def result_do(
        genexp,
        _errno_slot=errno_slot,
        _result_type=result_type,
    ) -> result_type:
        for value in genexp:
            return _result_type(_result_type.Ok, value)
        return _result_type(_result_type.Err, ptr[err_type](_errno_slot())[0])

    class result_api:
        type = result_type
        ok = result_ok
        err = result_err
        bind = result_bind
        do = result_do
        is_ok = result_is_ok
        is_err = result_is_err
        unwrap_or = _UnsupportedHelper("result.unwrap_or")
        hush = _UnsupportedHelper("result.hush")

    return result_type, result_api


__all__ = ["option_wrap", "result_wrap"]
