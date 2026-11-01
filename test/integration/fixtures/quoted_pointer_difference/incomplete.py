from pythoc import compile, i64, ptr


@compile
def invalid_difference() -> i64:
    begin: ptr['UnresolvedPointerDifference'] = ptr['UnresolvedPointerDifference'](0)
    end: ptr['UnresolvedPointerDifference'] = ptr['UnresolvedPointerDifference'](16)
    return end - begin


invalid_difference()
