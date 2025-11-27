#!/usr/bin/env python3
from pythoc import i32, i64, u64, compile

# Test tuple suffix normalization
@compile(suffix=(i32, 0, u64))
def test_func1(x: i32) -> i32:
    return x + 1

@compile(suffix=(i64, 2, u64))
def test_func2(x: i64) -> i64:
    return x + 2

def another_scope():
    # Deduplication test: same suffix should reuse files
    @compile(suffix=(i32, 0, u64))
    def test_func3(y: i32) -> i32:
        return y + 3
    return test_func3

test_func3 = another_scope()

print("Testing tuple suffix...")
print(f"test_func1(10) = {test_func1(10)}")
print(f"test_func2(20) = {test_func2(20)}")
print(f"test_func3(30) = {test_func3(30)}")
print("Success!")
