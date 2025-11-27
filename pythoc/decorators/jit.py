# -*- coding: utf-8 -*-
import os
from functools import wraps
from ..compiler import LLVMCompiler


def jit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pass
    return wrapper
