"""Test char builtin function"""

from pythoc import compile, char, i8


@compile
def char_string_first() -> i8:
    """Test char() with multi-character string"""
    return char("abc")


@compile
def char_string_single() -> i8:
    """Test char() with single character string"""
    return char("s")


@compile
def char_string_empty() -> i8:
    """Test char() with empty string"""
    return char("")


@compile
def char_int() -> i8:
    """Test char() with integer"""
    return char(48)


@compile
def char_int_negative() -> i8:
    """Test char() with negative integer"""
    return char(-1)


@compile
def char_newline() -> i8:
    """Test char() with newline"""
    return char("\n")


@compile
def char_tab() -> i8:
    """Test char() with tab"""
    return char("\t")


@compile
def char_star() -> i8:
    """Test char() with star"""
    return char("*")


@compile
def char_slash() -> i8:
    """Test char() with slash"""
    return char("/")


if __name__ == "__main__":
    print("Testing char builtin function...")
    print()
    
    result = char_string_first()
    assert result == ord('a'), f"Expected {ord('a')}, got {result}"
    print(f"char('abc') = {result} (expected {ord('a')})")
    
    result = char_string_single()
    assert result == ord('s'), f"Expected {ord('s')}, got {result}"
    print(f"char('s') = {result} (expected {ord('s')})")
    
    result = char_string_empty()
    assert result == 0, f"Expected 0, got {result}"
    print(f"char('') = {result} (expected 0)")
    
    result = char_int()
    assert result == 48, f"Expected 48, got {result}"
    print(f"char(48) = {result} (expected 48)")
    
    result = char_int_negative()
    assert result == -1, f"Expected -1, got {result}"
    print(f"char(-1) = {result} (expected -1)")
    
    result_newline = char_newline()
    expected_newline = ord('\n')
    assert result_newline == expected_newline, f"Expected {expected_newline}, got {result_newline}"
    print(f"char('\\n') = {result_newline} (expected {expected_newline})")
    
    result_tab = char_tab()
    expected_tab = ord('\t')
    assert result_tab == expected_tab, f"Expected {expected_tab}, got {result_tab}"
    print(f"char('\\t') = {result_tab} (expected {expected_tab})")
    
    star = char_star()
    assert star == 42, f"Expected 42 (ord('*')), got {star}"
    print(f"char('*') = {star} (expected 42)")
    
    slash = char_slash()
    assert slash == 47, f"Expected 47 (ord('/')), got {slash}"
    print(f"char('/') = {slash} (expected 47)")
    
    print()
    print("All tests passed!")

