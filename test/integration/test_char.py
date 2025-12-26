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


import unittest


class TestChar(unittest.TestCase):
    """Test char builtin function"""

    def test_char_string_first(self):
        """Test char() with multi-character string"""
        self.assertEqual(char_string_first(), ord('a'))

    def test_char_string_single(self):
        """Test char() with single character string"""
        self.assertEqual(char_string_single(), ord('s'))

    def test_char_string_empty(self):
        """Test char() with empty string"""
        self.assertEqual(char_string_empty(), 0)

    def test_char_int(self):
        """Test char() with integer"""
        self.assertEqual(char_int(), 48)

    def test_char_int_negative(self):
        """Test char() with negative integer"""
        self.assertEqual(char_int_negative(), -1)

    def test_char_newline(self):
        """Test char() with newline"""
        self.assertEqual(char_newline(), ord('\n'))

    def test_char_tab(self):
        """Test char() with tab"""
        self.assertEqual(char_tab(), ord('\t'))

    def test_char_star(self):
        """Test char() with star"""
        self.assertEqual(char_star(), 42)

    def test_char_slash(self):
        """Test char() with slash"""
        self.assertEqual(char_slash(), 47)


if __name__ == "__main__":
    unittest.main()

