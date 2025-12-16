# -*- coding: utf-8 -*-
"""
Effect test library - demonstrates effect system usage.

This library uses effects to provide configurable behavior.
By default, it uses a simple LCG-based RNG, but callers can
override with their own implementation at import time.
"""

from .rng_lib import random, seed, DefaultRNG
