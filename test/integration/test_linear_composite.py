"""
Tests for linear tokens in composite types - comprehensive coverage

Tests cover:
1. Basic tuple returns with linear tokens
2. Nested struct with multiple linear fields
3. Function parameter passing with composite types
4. Control flow with composite types
5. Error cases (not consumed, double consume, etc.)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pythoc import compile, linear, consume, void, i32, struct, ptr, i8
from pythoc.std.utility import move
from pythoc.libc.stdlib import malloc, free
from pythoc.libc.stdio import fopen, fclose
from pythoc.logger import set_raise_on_error

# Enable exception raising for tests that expect to catch exceptions
set_raise_on_error(True)


# ============================================================================
# Basic tuple return patterns
# ============================================================================

@compile
def lmalloc(size: i32) -> struct[ptr[i8], linear]:
    return malloc(size), linear()

@compile
def lfree(ptr: ptr[i8], t: linear):
    free(ptr)
    consume(t)


# ============================================================================
# Struct with single linear field
# ============================================================================

FileToken = struct[linear]

@compile
def make_file_token() -> FileToken:
    f = FileToken()
    f[0] = linear()
    return f

@compile
def lfopen(path: ptr[i8], mode: ptr[i8]) -> struct[ptr[i8], FileToken]:
    return fopen(path, mode), make_file_token()

@compile
def lfclose(ptr: ptr[i8], t: FileToken):
    fclose(ptr)
    consume(t[0])


# ============================================================================
# Struct with multiple linear fields
# ============================================================================

DualToken = struct[linear, linear]

@compile
def make_dual_token() -> DualToken:
    dt = DualToken()
    dt[0] = linear()
    dt[1] = linear()
    return dt

@compile
def consume_dual_token(dt: DualToken):
    consume(dt[0])
    consume(dt[1])


TripleResource = struct[ptr[i8], linear, linear]

@compile
def make_triple() -> TripleResource:
    tr = TripleResource()
    tr[0] = malloc(100)
    tr[1] = linear()
    tr[2] = linear()
    return tr

@compile
def destroy_triple(tr: TripleResource):
    free(tr[0])
    consume(tr[1])
    consume(tr[2])


# ============================================================================
# Deeply nested struct
# ============================================================================

@struct
class NestedResource:
    mem: struct[ptr[i8], linear]
    file: struct[ptr[i8], FileToken]

@compile
def make_nested_resource() -> NestedResource:
    nr = NestedResource()
    nr.mem = lmalloc(10)
    nr.file = lfopen("/tmp/test.txt", "w")
    return nr

@compile
def destroy_nested_resource(nr: NestedResource):
    lfree(nr.mem[0], nr.mem[1])
    lfclose(nr.file[0], nr.file[1])

@compile
def partial_destroy_nested(nr: NestedResource) -> struct[ptr[i8], linear]:
    lfclose(nr.file[0], nr.file[1])
    return nr.mem


# ============================================================================
# Triple nested struct
# ============================================================================

@struct
class Level3:
    tokens: DualToken
    extra: linear

@compile
def make_level3() -> Level3:
    l3 = Level3()
    l3.tokens = make_dual_token()
    l3.extra = linear()
    return l3

@compile
def destroy_level3(l3: Level3):
    consume_dual_token(l3.tokens)
    consume(l3.extra)


# ============================================================================
# Control flow with composite types
# ============================================================================

@compile
def conditional_destroy_dual(cond: i32) -> void:
    """Create and conditionally destroy dual token in different orders"""
    dt = make_dual_token()
    if cond:
        consume(dt[0])
        consume(dt[1])
    else:
        consume(dt[1])
        consume(dt[0])

@compile
def conditional_return_resource(cond: i32) -> struct[ptr[i8], linear]:
    """Conditionally return different resources"""
    if cond:
        return lmalloc(10)
    else:
        return lmalloc(20)

@compile
def conditional_create_dual(cond: i32) -> DualToken:
    """Conditionally create dual token in different ways"""
    if cond:
        return make_dual_token()
    else:
        dt = DualToken()
        dt[0] = linear()
        dt[1] = linear()
        return dt


# ============================================================================
# Partial consumption and reconstruction
# ============================================================================

@compile
def consume_first_return_second(dt: DualToken) -> linear:
    """Consume first field and return second field"""
    consume(dt[0])
    return move(dt[1])

@compile
def swap_dual_tokens_via_temp() -> void:
    """Swap tokens by consuming and recreating in different order"""
    dt = make_dual_token()
    # Cannot directly swap in struct, so we consume and recreate
    t0 = move(dt[0])
    t1 = move(dt[1])
    result = DualToken()
    result[0] = move(t1)
    result[1] = move(t0)
    consume_dual_token(result)


# ============================================================================
# Return composite from composite
# ============================================================================

@compile
def extract_mem_from_nested(nr: NestedResource) -> struct[ptr[i8], linear]:
    lfclose(nr.file[0], nr.file[1])
    return nr.mem

@compile
def extract_file_from_nested(nr: NestedResource) -> struct[ptr[i8], FileToken]:
    lfree(nr.mem[0], nr.mem[1])
    return nr.file


# ============================================================================
# Test runner functions (must be defined before any execution)
# ============================================================================

@compile
def run_basic_tuple():
    p, t = lmalloc(10)
    lfree(p, t)

@compile
def run_single_linear_field():
    f, ft = lfopen("/tmp/test.txt", "w")
    lfclose(f, ft)

@compile
def run_dual_token():
    dt = make_dual_token()
    consume_dual_token(dt)

@compile
def run_triple_resource():
    tr = make_triple()
    destroy_triple(tr)

@compile
def run_nested_resource():
    nr = make_nested_resource()
    destroy_nested_resource(nr)

@compile
def run_partial_nested_destroy():
    nr = make_nested_resource()
    mem = partial_destroy_nested(nr)
    lfree(mem[0], mem[1])

@compile
def run_level3_nesting():
    l3 = make_level3()
    destroy_level3(l3)

@compile
def run_conditional_return():
    res = conditional_return_resource(1)
    lfree(res[0], res[1])

@compile
def run_conditional_create():
    dt = conditional_create_dual(0)
    consume_dual_token(dt)

@compile
def run_partial_consumption():
    dt = make_dual_token()
    t = consume_first_return_second(dt)
    consume(t)

@compile
def run_extract_mem():
    nr = make_nested_resource()
    mem = extract_mem_from_nested(nr)
    lfree(mem[0], mem[1])

@compile
def run_extract_file():
    nr = make_nested_resource()
    file = extract_file_from_nested(nr)
    lfclose(file[0], file[1])


# ============================================================================
# Test functions (successful cases)
# ============================================================================

def test_basic_tuple():
    """Test basic tuple with linear token"""
    run_basic_tuple()
    print("OK test_basic_tuple passed")


def test_single_linear_field():
    """Test struct with single linear field"""
    run_single_linear_field()
    print("OK test_single_linear_field passed")


def test_dual_token():
    """Test struct with two linear fields"""
    run_dual_token()
    print("OK test_dual_token passed")


def test_triple_resource():
    """Test struct with ptr and two linear fields"""
    run_triple_resource()
    print("OK test_triple_resource passed")


def test_nested_resource():
    """Test nested struct with multiple resources"""
    run_nested_resource()
    print("OK test_nested_resource passed")


def test_partial_nested_destroy():
    """Test partially destroying nested resource"""
    run_partial_nested_destroy()
    print("OK test_partial_nested_destroy passed")


def test_level3_nesting():
    """Test triple-level nesting"""
    run_level3_nesting()
    print("OK test_level3_nesting passed")


def test_conditional_destroy():
    """Test control flow with composite destruction in different orders"""
    conditional_destroy_dual(1)
    conditional_destroy_dual(0)
    print("OK test_conditional_destroy passed")


def test_conditional_return():
    """Test conditional return of composite"""
    run_conditional_return()
    print("OK test_conditional_return passed")


def test_conditional_create():
    """Test conditional creation of dual token"""
    run_conditional_create()
    print("OK test_conditional_create passed")


def test_partial_consumption():
    """Test consuming one field and returning another"""
    run_partial_consumption()
    print("OK test_partial_consumption passed")


def test_token_swap():
    """Test swapping tokens via temporary variables"""
    swap_dual_tokens_via_temp()
    print("OK test_token_swap passed")


def test_extract_mem():
    """Test extracting one composite from nested struct"""
    run_extract_mem()
    print("OK test_extract_mem passed")


def test_extract_file():
    """Test extracting another composite from nested struct"""
    run_extract_file()
    print("OK test_extract_file passed")


# ============================================================================
# Error cases
# ============================================================================

from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
import os

def test_error_dual_not_consumed():
    """Test error when not all tokens in dual are consumed"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_dual_not_consumed')
    try:
        @compile(suffix="bad_dual_not_consumed")
        def bad_dual_not_consumed():
            dt = make_dual_token()
            consume(dt[0])
            # ERROR: dt[1] not consumed
        
        flush_all_pending_outputs()  # Trigger deferred compilation
        print("FAIL test_error_dual_not_consumed failed - should have raised RuntimeError")
    except RuntimeError as e:
        if "not consumed" in str(e):
            print(f"OK test_error_dual_not_consumed passed: {e}")
        else:
            print(f"FAIL test_error_dual_not_consumed failed - wrong error: {e}")
    finally:
        # Clean up failed group
        clear_failed_group(group_key)


def test_error_nested_not_consumed():
    """Test error when nested resource not fully consumed"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_nested_not_consumed')
    try:
        @compile(suffix="bad_nested_not_consumed")
        def bad_nested_not_consumed():
            nr = make_nested_resource()
            lfree(nr.mem[0], nr.mem[1])
            # ERROR: nr.file not consumed
        
        flush_all_pending_outputs()  # Trigger deferred compilation
        print("FAIL test_error_nested_not_consumed failed - should have raised RuntimeError")
    except RuntimeError as e:
        if "not consumed" in str(e):
            print(f"OK test_error_nested_not_consumed passed: {e}")
        else:
            print(f"FAIL test_error_nested_not_consumed failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


def test_error_double_consume_in_dual():
    """Test error when consuming same field twice in dual"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_double_consume')
    try:
        @compile(suffix="bad_double_consume")
        def bad_double_consume():
            dt = make_dual_token()
            consume(dt[0])
            consume(dt[0])  # ERROR: already consumed
            consume(dt[1])
        
        flush_all_pending_outputs()  # Trigger deferred compilation
        print("FAIL test_error_double_consume_in_dual failed - should have raised RuntimeError")
    except RuntimeError as e:
        if "already consumed" in str(e):
            print(f"OK test_error_double_consume_in_dual passed: {e}")
        else:
            print(f"FAIL test_error_double_consume_in_dual failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


def test_error_inconsistent_if_composite():
    """Test error when if/else inconsistently consume composite"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_inconsistent_if')
    try:
        @compile(suffix="bad_inconsistent_if")
        def bad_inconsistent_if(cond: i32):
            dt = make_dual_token()
            if cond:
                consume(dt[0])
                consume(dt[1])
            else:
                consume(dt[0])
                # ERROR: dt[1] not consumed in this branch
        
        flush_all_pending_outputs()  # Trigger deferred compilation
        print("FAIL test_error_inconsistent_if_composite failed - should have raised RuntimeError")
    except RuntimeError as e:
        err_str = str(e).lower()
        if "consistently" in err_str or "not consumed" in err_str or "inconsistent" in err_str:
            print(f"OK test_error_inconsistent_if_composite passed: {e}")
        else:
            print(f"FAIL test_error_inconsistent_if_composite failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


def test_error_loop_external_composite():
    """Test error when loop tries to consume external composite token"""
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', 'bad_loop_external')
    try:
        @compile(suffix="bad_loop_external")
        def bad_loop_external():
            dt = make_dual_token()
            i: i32 = 0
            while i < 3:
                consume(dt[0])  # ERROR: external token
                i = i + 1
            consume(dt[1])
        
        flush_all_pending_outputs()  # Trigger deferred compilation
        print("FAIL test_error_loop_external_composite failed - should have raised RuntimeError")
    except RuntimeError as e:
        err_str = str(e).lower()
        if "external" in err_str or "scope" in err_str or "loop body changes" in err_str:
            print(f"OK test_error_loop_external_composite passed: {e}")
        else:
            print(f"FAIL test_error_loop_external_composite failed - wrong error: {e}")
    finally:
        clear_failed_group(group_key)


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    import sys
    
    failed = False
    print("Running composite linear type tests...")
    print()
    
    try:
        print("=== Error cases ===")
        test_error_dual_not_consumed()
        test_error_nested_not_consumed()
        test_error_double_consume_in_dual()
        test_error_inconsistent_if_composite()
        test_error_loop_external_composite()
        print()
        
        print("=== Basic patterns ===")
        test_basic_tuple()
        test_single_linear_field()
        test_dual_token()
        test_triple_resource()
        print()
        
        print("=== Nested structures ===")
        test_nested_resource()
        test_partial_nested_destroy()
        test_level3_nesting()
        print()
        
        print("=== Control flow ===")
        test_conditional_destroy()
        test_conditional_return()
        test_conditional_create()
        print()
        
        print("=== Partial consumption ===")
        test_partial_consumption()
        test_token_swap()
        test_extract_mem()
        test_extract_file()
        print()
        
        print("All composite linear type tests completed!")
    except Exception as e:
        print(f"FAIL: {e}")
        failed = True
    
    sys.exit(1 if failed else 0)
