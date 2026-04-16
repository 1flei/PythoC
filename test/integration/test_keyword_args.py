"""
Integration tests for keyword argument support.

Tests:
1. f(key=value) - keyword arguments reordered to positional
2. f(positional, key=value) - mixed positional and keyword
3. **kwargs: StructType - pc_dict -> struct conversion
4. *args: T + **kwargs: T2 - coexistence of varargs and kwargs
5. Type coercion in kwargs (i32 -> f64, etc.)
6. Multiple callers sharing one kwargs function
"""
import unittest
import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from pythoc import compile, i32, i64, f64, struct


# ============================================================================
# Struct types
# ============================================================================

Rect = struct["width": i32, "height": i32]
Point3D = struct["x": i32, "y": i32, "z": i32]
WeightedPair = struct["a": i32, "b": i32, "weight": f64]
Vec2 = struct["x": f64, "y": f64]


# ============================================================================
# 1. Simple keyword -> positional reorder
# ============================================================================

@compile
def add_two(a: i32, b: i32) -> i32:
    return a + b


@compile
def sub_three(x: i32, y: i32, z: i32) -> i32:
    return x - y - z


@compile
def mixed_types(count: i32, scale: f64) -> f64:
    return f64(count) * scale


@compile
def test_kw_basic() -> i32:
    return add_two(a=i32(10), b=i32(20))


@compile
def test_kw_reversed() -> i32:
    return add_two(b=i32(20), a=i32(10))


@compile
def test_kw_mixed_positional() -> i32:
    return sub_three(i32(100), z=i32(3), y=i32(7))


@compile
def test_kw_float() -> i32:
    result: f64 = mixed_types(count=i32(5), scale=f64(2.0))
    return i32(result)


# ============================================================================
# 2. **kwargs: StructType
# ============================================================================

@compile
def make_rect(**kwargs: Rect) -> i32:
    return kwargs.width * kwargs.height


@compile
def make_point3d(**kwargs: Point3D) -> i32:
    return kwargs.x + kwargs.y + kwargs.z


@compile
def offset_rect(dx: i32, **kwargs: Rect) -> i32:
    return (kwargs.width + dx) * kwargs.height


@compile
def test_kwargs_rect() -> i32:
    return make_rect(width=i32(6), height=i32(7))


@compile
def test_kwargs_rect_reversed() -> i32:
    return make_rect(height=i32(7), width=i32(6))


@compile
def test_kwargs_point3d() -> i32:
    return make_point3d(x=i32(10), y=i32(20), z=i32(30))


@compile
def test_kwargs_with_positional() -> i32:
    return offset_rect(i32(5), width=i32(6), height=i32(7))


# ============================================================================
# 3. Type coercion within kwargs fields
# ============================================================================

@compile
def weighted_sum(**kwargs: WeightedPair) -> i32:
    result: f64 = f64(kwargs.a + kwargs.b) * kwargs.weight
    return i32(result)


@compile
def dot_product(**kwargs: Vec2) -> f64:
    return kwargs.x * kwargs.x + kwargs.y * kwargs.y


@compile
def test_kwargs_weighted() -> i32:
    return weighted_sum(a=i32(3), b=i32(7), weight=f64(2.0))


@compile
def test_kwargs_dot_product() -> i32:
    result: f64 = dot_product(x=f64(3.0), y=f64(4.0))
    return i32(result)


# ============================================================================
# 4. Multiple callers sharing one kwargs function
# ============================================================================

@compile
def test_kwargs_caller_a() -> i32:
    return make_rect(width=i32(10), height=i32(10))


@compile
def test_kwargs_caller_b() -> i32:
    return make_rect(width=i32(3), height=i32(5))


@compile
def test_kwargs_chain() -> i32:
    a: i32 = make_rect(width=i32(2), height=i32(3))
    b: i32 = make_rect(width=i32(4), height=i32(5))
    return a + b


# ============================================================================
# 5. Kwargs with field computation in the callee
# ============================================================================

@compile
def rect_perimeter(**kwargs: Rect) -> i32:
    return i32(2) * (kwargs.width + kwargs.height)


@compile
def test_kwargs_perimeter() -> i32:
    return rect_perimeter(width=i32(5), height=i32(3))


# ============================================================================
# 6. *args: T + **kwargs: T2 coexistence
# ============================================================================

@compile
def args_and_kwargs(*args: struct[i32, i32], **kwargs: Rect) -> i32:
    positional_sum: i32 = args[0] + args[1]
    area: i32 = kwargs.width * kwargs.height
    return positional_sum + area


@compile
def prefix_with_args_kwargs(tag: i32, *args: struct[i32, i32], **kwargs: Rect) -> i32:
    return tag + args[0] + args[1] + kwargs.width * kwargs.height


@compile
def test_args_and_kwargs() -> i32:
    return args_and_kwargs(i32(10), i32(20), width=i32(3), height=i32(4))


@compile
def test_prefix_args_kwargs() -> i32:
    return prefix_with_args_kwargs(
        i32(100), i32(10), i32(20), width=i32(3), height=i32(4)
    )


# ============================================================================
# 7. Multiple positional + kwargs
# ============================================================================

@compile
def scale_rect(factor: i32, offset: i32, **kwargs: Rect) -> i32:
    return factor * (kwargs.width + offset) * kwargs.height


@compile
def test_multi_positional_kwargs() -> i32:
    return scale_rect(i32(2), i32(1), width=i32(5), height=i32(3))


# ============================================================================
# Tests
# ============================================================================

class TestKeywordReorder(unittest.TestCase):
    """f(key=value) reordering to positional."""

    def test_basic(self):
        self.assertEqual(test_kw_basic(), 30)

    def test_reversed(self):
        self.assertEqual(test_kw_reversed(), 30)

    def test_mixed_positional(self):
        self.assertEqual(test_kw_mixed_positional(), 90)

    def test_float_types(self):
        self.assertEqual(test_kw_float(), 10)


class TestKwargsStruct(unittest.TestCase):
    """**kwargs: StructType end-to-end."""

    def test_rect(self):
        self.assertEqual(test_kwargs_rect(), 42)

    def test_rect_reversed(self):
        self.assertEqual(test_kwargs_rect_reversed(), 42)

    def test_point3d(self):
        self.assertEqual(test_kwargs_point3d(), 60)

    def test_with_positional(self):
        self.assertEqual(test_kwargs_with_positional(), 77)

    def test_weighted_pair(self):
        # (3+7)*2.0 = 20
        self.assertEqual(test_kwargs_weighted(), 20)

    def test_dot_product(self):
        # 3^2 + 4^2 = 25
        self.assertEqual(test_kwargs_dot_product(), 25)

    def test_perimeter(self):
        # 2*(5+3) = 16
        self.assertEqual(test_kwargs_perimeter(), 16)


class TestKwargsMultiCaller(unittest.TestCase):
    """Multiple callers sharing one **kwargs function."""

    def test_caller_a(self):
        self.assertEqual(test_kwargs_caller_a(), 100)

    def test_caller_b(self):
        self.assertEqual(test_kwargs_caller_b(), 15)

    def test_chain(self):
        # 2*3 + 4*5 = 6 + 20 = 26
        self.assertEqual(test_kwargs_chain(), 26)


class TestArgsAndKwargs(unittest.TestCase):
    """*args: T1 + **kwargs: T2 coexistence."""

    def test_args_and_kwargs(self):
        # (10+20) + (3*4) = 30 + 12 = 42
        self.assertEqual(test_args_and_kwargs(), 42)

    def test_prefix_args_kwargs(self):
        # 100 + 10 + 20 + 3*4 = 142
        self.assertEqual(test_prefix_args_kwargs(), 142)

    def test_multi_positional_kwargs(self):
        # 2 * (5+1) * 3 = 36
        self.assertEqual(test_multi_positional_kwargs(), 36)


if __name__ == "__main__":
    unittest.main(verbosity=2)
