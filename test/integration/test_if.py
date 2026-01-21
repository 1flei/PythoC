
from pythoc import i32, bool, compile, inline

@inline
def test_if_else(x):
    """Test if-else statements"""
    if x < 0:
        return 0
    elif x < 10:
        return x
    elif 10 <= x < 100:
        return x * 10
    else:
        return x * 100

@compile
def test_if_else_func(x: i32) -> i32:
    """Test if-else statements"""
    if x < 0:
        return 0
    elif x < 10:
        return x
    elif 10 <= x < 100:
        return x * 10
    else:
        return x * 100

if __name__ == "__main__":
    xs = [-5, 1, 10, 20, 100]
    expected = [0, 1, 100, 200, 10000]
    fs = []
    for i, x in enumerate(xs):
        @compile(suffix=f"const_folding_{i}")
        def const_folding() -> i32:
            if x < 0:
                return 0
            elif x < 10:
                return x
            elif 10 <= x < 100:
                return x * 10
            else:
                return x * 100
        fs.append(const_folding)

    # Test inline function
    for x, exp in zip(xs, expected):
        result = test_if_else(x)
        assert result == exp, f"test_if_else({x}) expected {exp}, got {result}"
        print(f"test_if_else({x}) = {result}")

    # Test compiled functions with different suffixes
    for f, x, exp in zip(fs, xs, expected):
        result = f(x)
        assert result == exp, f"const_folding({x}) expected {exp}, got {result}"
        print(f"const_folding({x}) = {result}")
    
    print("All if tests passed!")