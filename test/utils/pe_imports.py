"""Read native PE imports without relying on platform SDK executables."""

from pathlib import Path
import struct


def read_pe_imports(path):
    data = Path(path).read_bytes()
    if data[:2] != b'MZ':
        raise ValueError('not a PE executable')
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe + 4] != b'PE\0\0':
        raise ValueError('invalid PE signature')
    count = struct.unpack_from('<H', data, pe + 6)[0]
    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
    optional = pe + 24
    magic = struct.unpack_from('<H', data, optional)[0]
    formats = {0x10b: (96, 4, '<I'), 0x20b: (112, 8, '<Q')}
    directory, width, fmt = formats[magic]
    import_rva = struct.unpack_from('<I', data, optional + directory + 8)[0]
    sections = [struct.unpack_from('<IIII', data,
                                  optional + optional_size + 40 * i + 8)
                for i in range(count)]

    def offset(rva):
        for virtual_size, address, raw_size, raw in sections:
            if address <= rva < address + max(virtual_size, raw_size):
                return raw + rva - address
        raise ValueError('PE import RVA is outside the section table')

    def string(rva, skip=0):
        start = offset(rva) + skip
        return data[start:data.index(b'\0', start)].decode('ascii')

    result = {}
    if import_rva == 0:
        return result
    descriptor = offset(import_rva)
    while True:
        original, stamp, chain, name, thunk = struct.unpack_from(
            '<IIIII', data, descriptor)
        if not any((original, stamp, chain, name, thunk)):
            return result
        cursor = offset(original or thunk)
        symbols = []
        entry = struct.unpack_from(fmt, data, cursor)[0]
        while entry:
            if entry & (1 << (width * 8 - 1)):
                symbols.append('#{}'.format(entry & 0xffff))
            else:
                symbols.append(string(entry, skip=2))
            cursor += width
            entry = struct.unpack_from(fmt, data, cursor)[0]
        result[string(name)] = tuple(symbols)
        descriptor += 20
