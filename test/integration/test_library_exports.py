import os
import subprocess
import sys
import unittest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from pythoc import (
    array,
    bool,
    compile,
    compile_to_dynamic_library,
    compile_to_static_library,
    export_c_headers,
    enum,
    f32,
    f64,
    func,
    i8,
    i32,
    i64,
    ptr,
    seq,
    struct,
    u16,
    union,
    void,
)
from pythoc.utils.cc_utils import find_available_cc
from pythoc.utils.link_utils import (
    get_platform_link_flags,
    get_shared_lib_extension,
    get_static_lib_extension,
)


@struct
class ExportSmall8:
    a: i32
    b: i32


@struct
class ExportMedium12:
    a: i32
    b: i32
    c: i32


@struct
class ExportLarge32:
    a: i64
    b: i64
    c: i64
    d: i64


@struct
class ExportArray4:
    data: array[i32, 4]


@struct
class ExportQuadI64:
    a: i64
    b: i64
    c: i64
    d: i64


@union
class ExportUnion32:
    arr: array[i64, 4]
    quad: ExportQuadI64


@enum(i32)
class ExportResult:
    Ok: i32
    Err: i32


@enum(i32)
class ExportExpr:
    Const: i64
    Pair: ExportSmall8


@compile
def public_add(left: i32, right: i32) -> i32:
    return left + right


@compile
def _private_mul(left: i32, right: i32) -> i32:
    return left * right


@compile
def export_mix_primitives(
    a: i8,
    b: u16,
    c: i32,
    d: i64,
    x: f32,
    y: f64,
    flag: bool,
) -> i64:
    result: i64 = i64(a) + i64(b) + i64(c) + d + i64(x) + i64(y)
    if flag:
        result = result + 10
    return result


@compile
def export_sum_ptr(values: ptr[i32], count: i32) -> i32:
    total: i32 = 0
    for i in seq(count):
        total = total + values[i]
    return total


@compile
def export_fill_sequence(values: ptr[i32], count: i32) -> void:
    for i in seq(count):
        values[i] = i * 3 + 1


@compile
def export_make_small(a: i32, b: i32) -> ExportSmall8:
    value: ExportSmall8 = ExportSmall8()
    value.a = a
    value.b = b
    return value


@compile
def export_sum_small(value: ExportSmall8) -> i32:
    return value.a + value.b


@compile
def export_double_small(value: ExportSmall8) -> ExportSmall8:
    result: ExportSmall8 = ExportSmall8()
    result.a = value.a * 2
    result.b = value.b * 2
    return result


@compile
def export_make_medium(a: i32, b: i32, c: i32) -> ExportMedium12:
    value: ExportMedium12 = ExportMedium12()
    value.a = a
    value.b = b
    value.c = c
    return value


@compile
def export_sum_medium(value: ExportMedium12) -> i32:
    return value.a + value.b + value.c


@compile
def export_make_large(base: i64) -> ExportLarge32:
    value: ExportLarge32 = ExportLarge32()
    value.a = base
    value.b = base + 1
    value.c = base + 2
    value.d = base + 3
    return value


@compile
def export_sum_large(value: ExportLarge32) -> i64:
    return value.a + value.b + value.c + value.d


@compile
def export_increment_large(value: ExportLarge32) -> ExportLarge32:
    result: ExportLarge32 = ExportLarge32()
    result.a = value.a + 1
    result.b = value.b + 1
    result.c = value.c + 1
    result.d = value.d + 1
    return result


@compile
def export_make_array4(a: i32, b: i32, c: i32, d: i32) -> ExportArray4:
    value: ExportArray4 = ExportArray4()
    value.data[0] = a
    value.data[1] = b
    value.data[2] = c
    value.data[3] = d
    return value


@compile
def export_sum_array4(value: ExportArray4) -> i32:
    return value.data[0] + value.data[1] + value.data[2] + value.data[3]


@compile
def export_make_union32(base: i64) -> ExportUnion32:
    value: ExportUnion32 = ExportUnion32()
    value.arr[0] = base
    value.arr[1] = base + 1
    value.arr[2] = base + 2
    value.arr[3] = base + 3
    return value


@compile
def export_sum_union32(value: ExportUnion32) -> i64:
    return value.arr[0] + value.arr[1] + value.arr[2] + value.arr[3]


@compile
def export_result_ok(value: i32) -> ExportResult:
    return ExportResult(ExportResult.Ok, value)


@compile
def export_result_err(value: i32) -> ExportResult:
    return ExportResult(ExportResult.Err, value)


@compile
def export_result_value(value: ExportResult) -> i32:
    match value:
        case (ExportResult.Ok, payload):
            return payload
        case (ExportResult.Err, payload):
            return 0 - payload


@compile
def export_expr_pair(left: i32, right: i32) -> ExportExpr:
    pair: ExportSmall8 = ExportSmall8()
    pair.a = left
    pair.b = right
    return ExportExpr(ExportExpr.Pair, pair)


@compile
def export_eval_expr(value: ExportExpr) -> i64:
    match value:
        case (ExportExpr.Const, payload):
            return payload
        case (ExportExpr.Pair, pair):
            return pair.a + pair.b


@compile
def export_call_i32(cb: func[i32, i32], value: i32) -> i32:
    return cb(value) + 1


@compile
def export_call_small(cb: func[ExportSmall8, i32], left: i32, right: i32) -> i32:
    value: ExportSmall8 = ExportSmall8()
    value.a = left
    value.b = right
    return cb(value)


@compile
def export_call_make_small(cb: func[i32, i32, ExportSmall8], left: i32, right: i32) -> i32:
    value: ExportSmall8 = cb(left, right)
    return value.a + value.b


@compile
def export_call_make_large(cb: func[i64, ExportLarge32], base: i64) -> i64:
    value: ExportLarge32 = cb(base)
    return value.a + value.b + value.c + value.d


@compile
def export_accept_ptr_callback(cb: ptr[func[i32, i32]]) -> i32:
    return 123


class TestLibraryExports(unittest.TestCase):
    def setUp(self):
        self.build_dir = os.path.join('build', 'test', 'integration', 'library_exports')
        os.makedirs(self.build_dir, exist_ok=True)

    def test_default_exports_visible_public_symbols(self):
        header_path = export_c_headers(
            output_path=os.path.join(self.build_dir, 'library_exports')
        )
        static_path = compile_to_static_library(
            output_path=os.path.join(self.build_dir, 'liblibrary_exports')
        )
        dynamic_path = compile_to_dynamic_library(
            output_path=os.path.join(self.build_dir, 'liblibrary_exports')
        )

        self.assertTrue(header_path.endswith('.h'))
        self.assertTrue(static_path.endswith(get_static_lib_extension()))
        self.assertTrue(dynamic_path.endswith(get_shared_lib_extension()))
        self.assertTrue(os.path.exists(header_path))
        self.assertTrue(os.path.exists(static_path))
        self.assertTrue(os.path.exists(dynamic_path))

        with open(header_path, 'r', encoding='ascii') as f:
            header_text = f.read()
        self.assertIn('int32_t public_add(int32_t left, int32_t right);', header_text)
        self.assertNotIn('_private_mul', header_text)

        explicit_header_path = export_c_headers(
            _private_mul,
            output_path=os.path.join(self.build_dir, 'private_exports')
        )
        with open(explicit_header_path, 'r', encoding='ascii') as f:
            explicit_header_text = f.read()
        self.assertIn('_private_mul', explicit_header_text)

    def _write_consumer(self, header_path, name):
        consumer_path = os.path.join(self.build_dir, name + '.c')
        header_name = os.path.basename(header_path)
        with open(consumer_path, 'w', encoding='ascii') as f:
            f.write(
                '#include <stdbool.h>\n'
                '#include <stdint.h>\n'
                f'#include "{header_name}"\n'
                '\n'
                'static int32_t c_square(int32_t value) {\n'
                '    return value * value;\n'
                '}\n'
                '\n'
                'static int32_t c_sum_small(ExportSmall8 value) {\n'
                '    return value.a + value.b;\n'
                '}\n'
                '\n'
                'static ExportSmall8 c_make_small(int32_t left, int32_t right) {\n'
                '    ExportSmall8 value;\n'
                '    value.a = left;\n'
                '    value.b = right;\n'
                '    return value;\n'
                '}\n'
                '\n'
                'static ExportLarge32 c_make_large(int64_t base) {\n'
                '    ExportLarge32 value;\n'
                '    value.a = base;\n'
                '    value.b = base + 1;\n'
                '    value.c = base + 2;\n'
                '    value.d = base + 3;\n'
                '    return value;\n'
                '}\n'
                '\n'
                'static int check_all(void) {\n'
                '    int32_t values[5] = {1, 2, 3, 4, 5};\n'
                '    int32_t out[4] = {0, 0, 0, 0};\n'
                '    int32_t (*square_ptr)(int32_t) = c_square;\n'
                '    if (public_add(2, 3) != 5) return __LINE__;\n'
                '    if (export_mix_primitives((int8_t)-2, (uint16_t)7, 11, 100, 3.25f, 4.75, true) != 133) return __LINE__;\n'
                '    if (export_sum_ptr(values, 5) != 15) return __LINE__;\n'
                '    export_fill_sequence(out, 4);\n'
                '    if (out[0] != 1 || out[1] != 4 || out[2] != 7 || out[3] != 10) return __LINE__;\n'
                '\n'
                '    ExportSmall8 small = export_make_small(10, 20);\n'
                '    if (small.a != 10 || small.b != 20) return __LINE__;\n'
                '    if (export_sum_small(small) != 30) return __LINE__;\n'
                '    ExportSmall8 small2 = export_double_small(small);\n'
                '    if (small2.a != 20 || small2.b != 40) return __LINE__;\n'
                '\n'
                '    ExportMedium12 medium = export_make_medium(1, 2, 3);\n'
                '    if (medium.a != 1 || medium.b != 2 || medium.c != 3) return __LINE__;\n'
                '    if (export_sum_medium(medium) != 6) return __LINE__;\n'
                '\n'
                '    ExportLarge32 large = export_make_large(10);\n'
                '    if (large.a != 10 || large.b != 11 || large.c != 12 || large.d != 13) return __LINE__;\n'
                '    if (export_sum_large(large) != 46) return __LINE__;\n'
                '    ExportLarge32 large2 = export_increment_large(large);\n'
                '    if (large2.a != 11 || large2.b != 12 || large2.c != 13 || large2.d != 14) return __LINE__;\n'
                '\n'
                '    ExportArray4 arr = export_make_array4(1, 2, 3, 4);\n'
                '    if (arr.data[0] != 1 || arr.data[1] != 2 || arr.data[2] != 3 || arr.data[3] != 4) return __LINE__;\n'
                '    if (export_sum_array4(arr) != 10) return __LINE__;\n'
                '\n'
                '    ExportUnion32 uni = export_make_union32(5);\n'
                '    if (uni.arr[0] != 5 || uni.arr[1] != 6 || uni.arr[2] != 7 || uni.arr[3] != 8) return __LINE__;\n'
                '    if (export_sum_union32(uni) != 26) return __LINE__;\n'
                '    ExportUnion32 uni2;\n'
                '    uni2.quad.a = 1;\n'
                '    uni2.quad.b = 2;\n'
                '    uni2.quad.c = 3;\n'
                '    uni2.quad.d = 4;\n'
                '    if (export_sum_union32(uni2) != 10) return __LINE__;\n'
                '\n'
                '    ExportResult ok = export_result_ok(42);\n'
                '    if (ok.tag != 0 || ok.payload.Ok != 42) return __LINE__;\n'
                '    ExportResult err = export_result_err(7);\n'
                '    if (err.tag != 1 || err.payload.Err != 7) return __LINE__;\n'
                '    if (export_result_value(ok) != 42) return __LINE__;\n'
                '    if (export_result_value(err) != -7) return __LINE__;\n'
                '    ExportExpr expr = export_expr_pair(9, 10);\n'
                '    if (expr.tag != 1 || expr.payload.Pair.a != 9 || expr.payload.Pair.b != 10) return __LINE__;\n'
                '    if (export_eval_expr(expr) != 19) return __LINE__;\n'
                '\n'
                '    if (export_call_i32(c_square, 7) != 50) return __LINE__;\n'
                '    if (export_call_small(c_sum_small, 13, 17) != 30) return __LINE__;\n'
                '    if (export_call_make_small(c_make_small, 4, 5) != 9) return __LINE__;\n'
                '    if (export_call_make_large(c_make_large, 20) != 86) return __LINE__;\n'
                '    if (export_accept_ptr_callback(&square_ptr) != 123) return __LINE__;\n'
                '    return 0;\n'
                '}\n'
                '\n'
                'int main(void) {\n'
                '    return check_all();\n'
                '}\n'
            )
        return consumer_path

    def _consumer_compile_command(self, header_path, source_path, link_library, exe_path):
        cc = find_available_cc()
        return cc.split() + get_platform_link_flags(shared=False, linker=cc) + [
            '-I', os.path.abspath(os.path.dirname(header_path)),
            os.path.abspath(source_path),
            os.path.abspath(link_library),
            '-o', os.path.abspath(exe_path),
        ]

    def _compile_consumer(self, header_path, library_path, exe_name, dynamic=False):
        consumer_path = self._write_consumer(header_path, exe_name)
        exe_path = os.path.join(self.build_dir, exe_name)
        if sys.platform == 'win32':
            exe_path += '.exe'

        link_library = library_path
        if dynamic and sys.platform == 'win32':
            implib = os.path.splitext(library_path)[0] + '.lib'
            if os.path.exists(implib):
                link_library = implib

        cmd = self._consumer_compile_command(
            header_path, consumer_path, link_library, exe_path
        )
        if dynamic and sys.platform in ('linux', 'darwin'):
            cmd.append('-Wl,-rpath,' + os.path.abspath(os.path.dirname(library_path)))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return exe_path

    def _run_consumer(self, exe_path, library_path=None):
        env = os.environ.copy()
        if library_path:
            lib_dir = os.path.abspath(os.path.dirname(library_path))
            if sys.platform == 'win32':
                env['PATH'] = lib_dir + os.pathsep + env.get('PATH', '')
            elif sys.platform == 'darwin':
                env['DYLD_LIBRARY_PATH'] = lib_dir + os.pathsep + env.get('DYLD_LIBRARY_PATH', '')
            else:
                env['LD_LIBRARY_PATH'] = lib_dir + os.pathsep + env.get('LD_LIBRARY_PATH', '')

        run_result = subprocess.run(
            [os.path.abspath(exe_path)], capture_output=True, text=True,
            timeout=30, stdin=subprocess.DEVNULL, env=env,
        )
        self.assertEqual(run_result.returncode, 0, run_result.stderr)

    def test_c_consumer_links_static_library(self):
        header_path = export_c_headers(
            public_add,
            output_path=os.path.join(self.build_dir, 'consumer_exports')
        )
        static_path = compile_to_static_library(
            public_add,
            output_path=os.path.join(self.build_dir, 'libconsumer_exports')
        )
        consumer_path = os.path.join(self.build_dir, 'consumer.c')
        exe_path = os.path.join(self.build_dir, 'consumer')

        with open(consumer_path, 'w', encoding='ascii') as f:
            f.write(
                '#include "consumer_exports.h"\n'
                'int main(void) {\n'
                '    return public_add(2, 3) == 5 ? 0 : 1;\n'
                '}\n'
            )

        cmd = self._consumer_compile_command(
            header_path, consumer_path, static_path, exe_path
        )
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        self._run_consumer(exe_path)

    def test_c_consumer_links_static_and_dynamic_libraries(self):
        symbols = (
            public_add,
            export_mix_primitives,
            export_sum_ptr,
            export_fill_sequence,
            export_make_small,
            export_sum_small,
            export_double_small,
            export_make_medium,
            export_sum_medium,
            export_make_large,
            export_sum_large,
            export_increment_large,
            export_make_array4,
            export_sum_array4,
            export_make_union32,
            export_sum_union32,
            export_result_ok,
            export_result_err,
            export_result_value,
            export_expr_pair,
            export_eval_expr,
            export_call_i32,
            export_call_small,
            export_call_make_small,
            export_call_make_large,
            export_accept_ptr_callback,
        )
        header_path = export_c_headers(
            *symbols,
            output_path=os.path.join(self.build_dir, 'full_exports')
        )
        static_path = compile_to_static_library(
            *symbols,
            output_path=os.path.join(self.build_dir, 'libfull_exports_static')
        )
        dynamic_path = compile_to_dynamic_library(
            *symbols,
            output_path=os.path.join(self.build_dir, 'libfull_exports_dynamic')
        )

        with open(header_path, 'r', encoding='ascii') as f:
            header_text = f.read()
        self.assertIn('typedef struct ExportSmall8 ExportSmall8;', header_text)
        self.assertIn('typedef union ExportUnion32 ExportUnion32;', header_text)
        self.assertIn('void export_fill_sequence(int32_t *values, int32_t count);', header_text)
        self.assertIn('ExportLarge32 export_increment_large(ExportLarge32 value);', header_text)
        self.assertIn('typedef struct ExportResult ExportResult;', header_text)
        self.assertIn('ExportResult export_result_ok(int32_t value);', header_text)
        self.assertIn('int32_t export_call_i32(int32_t (*cb)(int32_t), int32_t value);', header_text)
        self.assertIn('int32_t export_call_small(int32_t (*cb)(ExportSmall8), int32_t left, int32_t right);', header_text)
        self.assertIn('int32_t export_call_make_small(ExportSmall8 (*cb)(int32_t, int32_t), int32_t left, int32_t right);', header_text)
        self.assertIn('int64_t export_call_make_large(ExportLarge32 (*cb)(int64_t), int64_t base);', header_text)
        self.assertIn('int32_t export_accept_ptr_callback(int32_t (**cb)(int32_t));', header_text)

        static_exe = self._compile_consumer(
            header_path, static_path, 'full_static_consumer'
        )
        self._run_consumer(static_exe)

        dynamic_exe = self._compile_consumer(
            header_path, dynamic_path, 'full_dynamic_consumer', dynamic=True
        )
        self._run_consumer(dynamic_exe, dynamic_path)


if __name__ == '__main__':
    unittest.main()
