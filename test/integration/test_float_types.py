from pythoc import f16, bf16, f32, f64, f128, i32, compile
from pythoc.libc.stdio import printf

def gen_fadd(T):
    @compile(suffix=T)
    def add(a: T, b: T) -> T:
        return a + b
    return add

def gen_test_fx(T):
    fadd = gen_fadd(T)
    @compile(suffix=T)
    def test_fx() -> i32:
        x: T = T(1.0)
        y: T = T(2.0)
        z: T = fadd(x, y)
        printf("%f + %f = %f\n", x, y, z)
        return 0
    return test_fx

# f16/bf16/f128 requires higher version of gcc/clang
# test_f16 = gen_test_fx(f16)
# test_bf16 = gen_test_fx(bf16)
test_f32 = gen_test_fx(f32)
test_f64 = gen_test_fx(f64)
# test_f128 = gen_test_fx(f128)

@compile
def main() -> i32:
    printf("=== Float Types Test ===\n\n")
    
    # printf("--- f16 test ---\n")
    # test_f16()
    # printf("\n")
    
    # printf("--- bf16 test ---\n")
    # test_bf16()
    # printf("\n")
    
    printf("--- f32 test ---\n")
    test_f32()
    printf("\n")
    
    printf("--- f64 test ---\n")
    test_f64()
    printf("\n")
    
    # printf("--- f128 test ---\n")
    # test_f128()
    # printf("\n")
    
    printf("=== All Tests Complete ===\n")
    return 0

if __name__ == "__main__":
    main()
    print("Float types test passed!")
