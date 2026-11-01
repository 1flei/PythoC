from pythoc import compile, i8, i64


@compile
class PointerDiffRecord:
    tag: i8
    value: i64
