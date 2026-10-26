"""
Locale API (locale.h / langinfo.h).

Locale categories and nl_item values follow the selected host platform.
"""

from ..decorators import extern
from ..builtin_entities import i32, i8, ptr, void
from ._platform import IS_MACOS, IS_LINUX, IS_WINDOWS

if IS_MACOS or IS_WINDOWS:
    LC_ALL = 0
    LC_COLLATE = 1
    LC_CTYPE = 2
    LC_MONETARY = 3
    LC_NUMERIC = 4
    LC_TIME = 5
elif IS_LINUX:
    LC_CTYPE = 0
    LC_NUMERIC = 1
    LC_TIME = 2
    LC_COLLATE = 3
    LC_MONETARY = 4
    LC_ALL = 6
else:
    raise NotImplementedError(
        "<locale.h> bindings are only available on macOS, Linux and Windows"
    )

__all__ = [
    'setlocale',
    'LC_ALL', 'LC_COLLATE', 'LC_CTYPE', 'LC_MONETARY', 'LC_NUMERIC',
    'LC_TIME',
]

if not IS_WINDOWS:
    LC_MESSAGES = 6 if IS_MACOS else 5
    __all__.append('LC_MESSAGES')


@extern(lib='c')
def setlocale(category: i32, locale: ptr[i8]) -> ptr[i8]:
    """Set or query the current locale for a category."""
    pass


@extern(lib='c')
def localeconv() -> ptr[void]:
    """Return the lconv struct for the current locale (opaque layout)."""
    pass


__all__.append('localeconv')


if not IS_WINDOWS:
    @extern(lib='c')
    def nl_langinfo(item: i32) -> ptr[i8]:
        """Query a locale-dependent item (codeset, radix char, ...)."""
        pass

    __all__.append('nl_langinfo')


# glibc nl_item values, probed from the host <langinfo.h>.
# _NL_ITEM(category, index) == (category << 16) | index.
if IS_LINUX:
    CODESET = 14
    D_T_FMT = 131112
    D_FMT = 131113
    T_FMT = 131114
    T_FMT_AMPM = 131115
    AM_STR = 131110
    PM_STR = 131111
    DAY_1 = 131079
    DAY_2 = 131080
    DAY_3 = 131081
    DAY_4 = 131082
    DAY_5 = 131083
    DAY_6 = 131084
    DAY_7 = 131085
    ABDAY_1 = 131072
    ABDAY_2 = 131073
    ABDAY_3 = 131074
    ABDAY_4 = 131075
    ABDAY_5 = 131076
    ABDAY_6 = 131077
    ABDAY_7 = 131078
    MON_1 = 131098
    MON_2 = 131099
    MON_3 = 131100
    MON_4 = 131101
    MON_5 = 131102
    MON_6 = 131103
    MON_7 = 131104
    MON_8 = 131105
    MON_9 = 131106
    MON_10 = 131107
    MON_11 = 131108
    MON_12 = 131109
    ABMON_1 = 131086
    ABMON_2 = 131087
    ABMON_3 = 131088
    ABMON_4 = 131089
    ABMON_5 = 131090
    ABMON_6 = 131091
    ABMON_7 = 131092
    ABMON_8 = 131093
    ABMON_9 = 131094
    ABMON_10 = 131095
    ABMON_11 = 131096
    ABMON_12 = 131097
    RADIXCHAR = 65536
    THOUSEP = 65537
    YESEXPR = 327680
    NOEXPR = 327681
    CRNCYSTR = 262159
elif IS_MACOS:
    CODESET = 0
    D_T_FMT = 1
    D_FMT = 2
    T_FMT = 3
    T_FMT_AMPM = 4
    AM_STR = 5
    PM_STR = 6
    DAY_1 = 7
    DAY_2 = 8
    DAY_3 = 9
    DAY_4 = 10
    DAY_5 = 11
    DAY_6 = 12
    DAY_7 = 13
    ABDAY_1 = 14
    ABDAY_2 = 15
    ABDAY_3 = 16
    ABDAY_4 = 17
    ABDAY_5 = 18
    ABDAY_6 = 19
    ABDAY_7 = 20
    MON_1 = 21
    MON_2 = 22
    MON_3 = 23
    MON_4 = 24
    MON_5 = 25
    MON_6 = 26
    MON_7 = 27
    MON_8 = 28
    MON_9 = 29
    MON_10 = 30
    MON_11 = 31
    MON_12 = 32
    ABMON_1 = 33
    ABMON_2 = 34
    ABMON_3 = 35
    ABMON_4 = 36
    ABMON_5 = 37
    ABMON_6 = 38
    ABMON_7 = 39
    ABMON_8 = 40
    ABMON_9 = 41
    ABMON_10 = 42
    ABMON_11 = 43
    ABMON_12 = 44
    RADIXCHAR = 50
    THOUSEP = 51
    YESEXPR = 52
    NOEXPR = 53
    CRNCYSTR = 56

if IS_MACOS or IS_LINUX:
    __all__.extend([
        'CODESET', 'D_T_FMT', 'D_FMT', 'T_FMT', 'T_FMT_AMPM', 'AM_STR', 'PM_STR',
        'DAY_1', 'DAY_2', 'DAY_3', 'DAY_4', 'DAY_5', 'DAY_6', 'DAY_7',
        'ABDAY_1', 'ABDAY_2', 'ABDAY_3', 'ABDAY_4', 'ABDAY_5', 'ABDAY_6',
        'ABDAY_7',
        'MON_1', 'MON_2', 'MON_3', 'MON_4', 'MON_5', 'MON_6', 'MON_7', 'MON_8',
        'MON_9', 'MON_10', 'MON_11', 'MON_12',
        'ABMON_1', 'ABMON_2', 'ABMON_3', 'ABMON_4', 'ABMON_5', 'ABMON_6',
        'ABMON_7', 'ABMON_8', 'ABMON_9', 'ABMON_10', 'ABMON_11', 'ABMON_12',
        'RADIXCHAR', 'THOUSEP', 'YESEXPR', 'NOEXPR', 'CRNCYSTR',
    ])
