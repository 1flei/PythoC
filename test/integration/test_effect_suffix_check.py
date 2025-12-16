# -*- coding: utf-8 -*-
"""Test effect suffix propagation to @compile"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pythoc import compile, effect, i32, u64
from pythoc.decorators import clear_registry
from types import SimpleNamespace

clear_registry()

# Clean effect state
for name in list(effect._effects.keys()):
    try:
        delattr(effect, name)
    except:
        pass
effect._defaults.clear()
effect._direct_assignments.clear()
effect._suffix_stack.clear()

# Create RNG implementation
@compile
def rng_next() -> u64:
    return u64(42)

mock_rng = SimpleNamespace(next=rng_next)
effect.default(rng=mock_rng)

# Test 1: Check if effect suffix propagates to @compile
print('Test 1: effect suffix propagation')
with effect(suffix='test_suffix'):
    from pythoc.effect import get_current_effect_suffix
    print(f'  Inside context, suffix = {get_current_effect_suffix()}')
    
    # This function should get suffix from effect context
    @compile  # No explicit suffix - should it use effect's suffix?
    def func_in_effect_context() -> u64:
        return effect.rng.next()
    
    print(f'  func mangled_name = {func_in_effect_context._mangled_name}')
    print(f'  func so_file = {func_in_effect_context._so_file}')

print()
print('Test 2: explicit @compile suffix')
with effect(suffix='effect_suffix'):
    @compile(suffix='explicit_suffix')
    def func_with_explicit_suffix() -> u64:
        return effect.rng.next()
    
    print(f'  func mangled_name = {func_with_explicit_suffix._mangled_name}')
    print(f'  func so_file = {func_with_explicit_suffix._so_file}')

print()
print('Test 3: Force flush and check generated files')
from pythoc.build import flush_all_pending_outputs
flush_all_pending_outputs()

import os
build_dir = 'build/test/integration'
if os.path.exists(build_dir):
    suffix_files = [f for f in os.listdir(build_dir) if 'test_suffix' in f or 'explicit_suffix' in f]
    if suffix_files:
        for f in sorted(suffix_files):
            print(f'  Found: {f}')
    else:
        print('  No suffix files found, listing all effect files:')
        for f in sorted(os.listdir(build_dir)):
            if 'effect' in f:
                print(f'    {f}')

print()
print('Test 4: Check IR file content for function names')
for f in os.listdir(build_dir):
    if f.endswith('.ll') and ('test_suffix' in f or 'explicit_suffix' in f or 'effect_suffix_check' in f):
        path = os.path.join(build_dir, f)
        with open(path) as fp:
            content = fp.read()
            # Find function definitions
            import re
            funcs = re.findall(r'define.*@(\w+)\(', content)
            print(f'  {f}: functions = {funcs}')
