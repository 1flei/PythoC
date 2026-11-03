"""Execute generated control flow after CFG indexing and ownership analysis."""

import ast
import unittest

from pythoc import i64, meta


BRANCHES = 1200
source_lines = ['def cfg_large_dispatch(x):']
for branch in range(BRANCHES):
    source_lines.extend((f'    if x == {branch}:',
                         f'        return {branch * branch + 7}'))
source_lines.append('    return -1')
source = '\n'.join(source_lines) + '\n'
dispatch = meta.compile_ast(
    ast.parse(source).body[0],
    param_types={'x': i64},
    return_type=i64,
    source_file=__file__,
    source_code=source,
    user_globals={'i64': i64},
)


class TestLargeCFGDispatch(unittest.TestCase):
    def test_all_branches_and_misses(self):
        for value in range(-2, BRANCHES + 2):
            expected = value * value + 7 if 0 <= value < BRANCHES else -1
            self.assertEqual(dispatch(value), expected)


if __name__ == '__main__':
    unittest.main()
