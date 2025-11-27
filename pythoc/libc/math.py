"""
Mathematical Functions (math.h)
"""

from ..decorators import extern
from ..builtin_entities import f64

# Trigonometric functions
@extern(lib='m')
def sin(x: f64) -> f64:
    """Sine function"""
    pass

@extern(lib='m')
def cos(x: f64) -> f64:
    """Cosine function"""
    pass

@extern(lib='m')
def tan(x: f64) -> f64:
    """Tangent function"""
    pass

@extern(lib='m')
def asin(x: f64) -> f64:
    """Arc sine function"""
    pass

@extern(lib='m')
def acos(x: f64) -> f64:
    """Arc cosine function"""
    pass

@extern(lib='m')
def atan(x: f64) -> f64:
    """Arc tangent function"""
    pass

@extern(lib='m')
def atan2(y: f64, x: f64) -> f64:
    """Arc tangent of y/x"""
    pass

# Hyperbolic functions
@extern(lib='m')
def sinh(x: f64) -> f64:
    """Hyperbolic sine"""
    pass

@extern(lib='m')
def cosh(x: f64) -> f64:
    """Hyperbolic cosine"""
    pass

@extern(lib='m')
def tanh(x: f64) -> f64:
    """Hyperbolic tangent"""
    pass

# Exponential and logarithmic functions
@extern(lib='m')
def exp(x: f64) -> f64:
    """Exponential function"""
    pass

@extern(lib='m')
def log(x: f64) -> f64:
    """Natural logarithm"""
    pass

@extern(lib='m')
def log10(x: f64) -> f64:
    """Base-10 logarithm"""
    pass

@extern(lib='m')
def pow(x: f64, y: f64) -> f64:
    """Power function"""
    # Returns computed result at runtime
    # Will generate actual pow call at compile time
    import math
    try:
        return math.pow(float(x), float(y))
    except (TypeError, ValueError):
        return 0.0

@extern(lib='m')
def sqrt(x: f64) -> f64:
    """Square root"""
    pass

# Rounding and remainder functions
@extern(lib='m')
def ceil(x: f64) -> f64:
    """Ceiling function"""
    pass

@extern(lib='m')
def floor(x: f64) -> f64:
    """Floor function"""
    pass

@extern(lib='m')
def fabs(x: f64) -> f64:
    """Absolute value"""
    pass

@extern(lib='m')
def fmod(x: f64, y: f64) -> f64:
    """Floating-point remainder"""
    pass

# P0 additions
@extern(lib='m')
def trunc(x: f64) -> f64: pass

@extern(lib='m')
def round(x: f64) -> f64: pass

@extern(lib='m')
def modf(x: f64) -> f64: pass

@extern(lib='m')
def hypot(x: f64, y: f64) -> f64: pass

@extern(lib='m')
def log1p(x: f64) -> f64: pass

@extern(lib='m')
def expm1(x: f64) -> f64: pass