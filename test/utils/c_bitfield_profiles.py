"""Explicit configurations matching the C oracle's supported target ABIs."""


def c_bitfield_options(triple):
    microsoft = 'windows' in triple
    aapcs64 = triple.startswith(('aarch64-', 'arm64-')) and 'apple' not in triple
    return {
        'mixed': {
            'storage_unit': 16,
            'storage_unit_overrides': {'b': 32},
            'unit_break_before': {'b': 32, 'c': 32} if microsoft else {},
        },
        'boundary': {
            'storage_unit': 16,
            'unit_break_before': {'b': 32},
            'abi_alignment': 32 if microsoft or aapcs64 else 16,
        },
    }
