"""A cold import owns its defaults and explicitly nested effect context."""

from pythoc import compile, effect, i32
from pythoc.effect import get_current_effect_suffix

observed_suffix = get_current_effect_suffix()
observed_provider = effect.get_effect_impl('import_provider')
effect.default(import_scalar=13)


@compile
def read_cold() -> i32:
    return i32(effect.import_scalar)


with effect(import_scalar=31, suffix='module_owned'):
    @compile
    def read_pinned() -> i32:
        return i32(effect.import_scalar)
