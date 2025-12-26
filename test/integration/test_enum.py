from __future__ import annotations

from pythoc import *
from pythoc.libc.stdio import printf

# Test 1: Simple enum with only bare tags (no payload) -> should be just inttype
# Use None annotation for no-payload variants
@enum
class Color:
    Red: None
    Green = 5
    Blue: None

# Test 2: Enum with only annotated tags (with payload) -> inttype * union[type...]
@enum(i32)
class Result:
    Ok: i32
    Err: ptr[i8]

# Test 3: Mixed enum (bare tags + annotated tags)
@enum(i16)
class Mixed:
    None_Variant: None
    Int_Variant: i32 = 10
    Float_Variant: f64
    Struct_Variant: struct[i32, f64]
    Another_None: None = 20

# Test 4: Enum with explicit tag values
@enum(i64)
class Explicit:
    First = 100
    Second: i32 = 200
    Third: f64

@compile
def test_bare_tags_only() -> i32:
    """Test enum with only bare tags (no payload types)"""
    printf("Testing bare tags only enum (Color):\n")
    printf("Red=%d, Green=%d, Blue=%d\n", Color.Red, Color.Green, Color.Blue)
    
    # Verify auto-tag and explicit tag
    # Red=0 (auto), Green=5 (explicit), Blue=6 (auto after 5)
    if Color.Red != 0:
        printf("ERROR: Color.Red should be 0\n")
        return 1
    if Color.Green != 5:
        printf("ERROR: Color.Green should be 5\n")
        return 1
    if Color.Blue != 6:
        printf("ERROR: Color.Blue should be 6\n")
        return 1
    
    # For no-payload enum, memory layout is still struct{i8, union[void]}
    # But the union is size 0, so essentially just the tag
    # Constructor still needs 2 args, but payload is ignored for void variants
    c: Color = Color(Color.Red)  # payload is dummy for void
    
    printf("PASS: bare tags only\n")
    return 0

@compile
def test_init_enum() -> i32:
    """Test initialization from tuple"""
    ok_val: Result = (Result.Ok, 42)
    mixed_val: Mixed = (Mixed.Int_Variant, 123)
    mixed_val2: Mixed = (Mixed.Struct_Variant, (1, 2.0))
    c: Color = Color.Red            # Should allow init from tag for void variant
    return 0

@compile
def test_annotated_tags_only() -> i32:
    """Test enum with only annotated tags (has payload types)"""
    printf("Testing annotated tags only enum (Result):\n")
    printf("Ok=%d, Err=%d\n", Result.Ok, Result.Err)
    
    # Create enum instances
    ok_val: Result = Result(Result.Ok, 42)
    err_val: Result = Result(Result.Err, "Error message")
    
    # Tags should be 0, 1
    if Result.Ok != 0:
        printf("ERROR: Result.Ok should be 0\n")
        return 1
    if Result.Err != 1:
        printf("ERROR: Result.Err should be 1\n")
        return 1
    
    printf("PASS: annotated tags only\n")
    return 0

@compile
def test_mixed_tags() -> i32:
    """Test enum with mixed bare and annotated tags"""
    printf("Testing mixed tags enum (Mixed):\n")
    printf("None_Variant=%d, Int_Variant=%d, Float_Variant=%d, Another_None=%d\n",
           Mixed.None_Variant, Mixed.Int_Variant, Mixed.Float_Variant, Mixed.Another_None)
    
    # None_Variant=0 (auto), Int_Variant=10 (explicit), Float_Variant=11 (auto), Another_None=20 (explicit)
    if Mixed.None_Variant != 0:
        printf("ERROR: Mixed.None_Variant should be 0\n")
        return 1
    if Mixed.Int_Variant != 10:
        printf("ERROR: Mixed.Int_Variant should be 10\n")
        return 1
    if Mixed.Float_Variant != 11:
        printf("ERROR: Mixed.Float_Variant should be 11\n")
        return 1
    if Mixed.Another_None != 20:
        printf("ERROR: Mixed.Another_None should be 20\n")
        return 1
    
    # Create instances for variants with payload
    int_val: Mixed = Mixed(Mixed.Int_Variant, 123)
    float_val: Mixed = Mixed(Mixed.Float_Variant, 3.14)
    
    printf("PASS: mixed tags\n")
    return 0

@compile
def test_explicit_tags() -> i32:
    """Test enum with explicit tag values"""
    printf("Testing explicit tags enum (Explicit):\n")
    printf("First=%d, Second=%d, Third=%d\n", Explicit.First, Explicit.Second, Explicit.Third)
    
    # First=100, Second=200, Third=201
    if Explicit.First != 100:
        printf("ERROR: Explicit.First should be 100\n")
        return 1
    if Explicit.Second != 200:
        printf("ERROR: Explicit.Second should be 200\n")
        return 1
    if Explicit.Third != 201:
        printf("ERROR: Explicit.Third should be 201\n")
        return 1
    
    # Create instances
    second_val: Explicit = Explicit(Explicit.Second, 999)
    third_val: Explicit = Explicit(Explicit.Third, 2.71)
    
    printf("PASS: explicit tags\n")
    return 0

@compile
def test_tag_type_sizes() -> i32:
    """Test that different tag types work correctly"""
    printf("Testing tag type sizes:\n")
    
    # Color uses i8, Result uses i32, Mixed uses i16, Explicit uses i64
    # Just verify we can create and use them
    # Dummy payload for void variant
    c: Color = Color(Color.Red)
    
    printf("PASS: tag type sizes\n")
    return 0

# Test all three syntax forms
@enum(i32)
class Status:
    Ok: i32           # Syntax 1: auto tag 0, has payload (i32)
    Warning: i32 = 10 # Syntax 2: explicit tag 10, has payload (i32)
    Error: None       # Syntax 3: auto tag 11, no payload (void)
    Fatal: None = 99  # Syntax 3 with explicit tag: tag 99, no payload (void)

@compile
def test_status_enum() -> i32:
    printf("Status tag values:\n")
    printf("Ok=%d, Warning=%d, Error=%d, Fatal=%d\n", 
           Status.Ok, Status.Warning, Status.Error, Status.Fatal)
    
    # Create instances with payload
    s1: Status = Status(Status.Ok, 0)
    s2: Status = Status(Status.Warning, 42)
    
    # Create instances without payload (void variants - single argument form)
    s3: Status = Status(Status.Error)
    s4: Status = Status(Status.Fatal)
    
    printf("All Status instances created successfully\n")
    return 0

@enum(i8)
class SimpleEnum:
    A: None
    B: None

@enum(i32)
class ComplexEnum:
    Int: i32
    Float: f64
    NoPayload: None

@compile
def test_enum_layout() -> i32:
    # Check SimpleEnum layout (should be struct{i8, union[void]})
    printf("Testing SimpleEnum layout:\n")
    s: SimpleEnum = SimpleEnum(SimpleEnum.A)
    printf("SimpleEnum instance created\n")
    
    # Check ComplexEnum layout (should be struct{i32, union[i32, f64, void]})
    printf("Testing ComplexEnum layout:\n")
    c1: ComplexEnum = ComplexEnum(ComplexEnum.Int, 42)
    c2: ComplexEnum = ComplexEnum(ComplexEnum.Float, 3.14)
    c3: ComplexEnum = ComplexEnum(ComplexEnum.NoPayload)
    printf("ComplexEnum instances created\n")
    
    printf("All enum layout tests passed\n")
    return 0

@compile
def main() -> i32:
    printf("=== Enum Unit Tests ===\n\n")
    
    if test_bare_tags_only() != 0:
        return 1
    printf("\n")
    
    if test_annotated_tags_only() != 0:
        return 1
    printf("\n")
    
    if test_mixed_tags() != 0:
        return 1
    printf("\n")
    
    if test_explicit_tags() != 0:
        return 1
    printf("\n")
    
    if test_tag_type_sizes() != 0:
        return 1
    printf("\n")
    
    printf("=== All Enum Tests Passed ===\n")

    test_status_enum()

    test_enum_layout()
    return 0

if __name__ == "__main__":
    result = main()
    assert result == 0, f"Enum tests failed with code {result}"
    print("All enum tests passed!")
