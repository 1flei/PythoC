"""Compile and execute the volatile cases in a fresh build directory."""

from pythoc import compile, const, i32, volatile
from pythoc.build.output_manager import flush_all_pending_outputs


@compile
def local(value: i32) -> i32:
    word: const[volatile[i32]] = value
    return word + word


@compile
def parameter(value: volatile[i32]) -> i32:
    return value + value


@compile
def later(value: i32) -> i32:
    word: volatile[i32]
    word = value
    return word + word


@compile
def plain(value: i32) -> i32:
    word: i32 = value
    return word + word


if __name__ == '__main__':
    flush_all_pending_outputs()
    for function, stores, loads in ((local, 1, 2), (parameter, 1, 2),
                                     (later, 1, 2), (plain, 0, 0)):
        ir = str(function._func_info.llvm_function)
        assert ir.count('store volatile i32') == stores, ir
        assert ir.count('load volatile i32,') == loads, ir
        assert function(21) == 42, function.__name__
