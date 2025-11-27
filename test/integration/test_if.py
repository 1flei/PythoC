
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

xs = [-5, 1, 10, 20, 100]
fs = []
for x in xs:
    @compile(anonymous=True)
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

print(test_if_else(-5))
print(test_if_else(1))
print(test_if_else(10))
print(test_if_else(20))
print(test_if_else(100))

for f, x in zip(fs, xs):
    print(f(x))