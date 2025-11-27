"""
Tests for scope analysis

Tests variable classification (captured/local/param)
"""

import ast
import unittest
from pythoc.inline.scope_analyzer import ScopeAnalyzer, ScopeContext


class TestScopeContext(unittest.TestCase):
    """Test ScopeContext creation and queries"""
    
    def test_empty_context(self):
        """Empty context has no variables"""
        ctx = ScopeContext.empty()
        self.assertFalse(ctx.has_variable('x'))
        self.assertFalse(ctx.has_variable('y'))
    
    def test_from_var_list(self):
        """Create context from variable list"""
        ctx = ScopeContext.from_var_list(['x', 'y', 'z'])
        self.assertTrue(ctx.has_variable('x'))
        self.assertTrue(ctx.has_variable('y'))
        self.assertTrue(ctx.has_variable('z'))
        self.assertFalse(ctx.has_variable('a'))
    
    def test_has_variable(self):
        """Check variable availability"""
        ctx = ScopeContext(available_vars={'x', 'y'})
        self.assertTrue(ctx.has_variable('x'))
        self.assertFalse(ctx.has_variable('z'))


class TestScopeAnalyzer(unittest.TestCase):
    """Test scope analysis for inline operations"""
    
    def _parse_function(self, code: str) -> tuple:
        """Helper: Parse function and extract body/params"""
        tree = ast.parse(code)
        func = tree.body[0]
        return func.body, func.args.args
    
    def test_simple_function_no_captures(self):
        """
        def f(x):
            y = x + 1
            return y
            
        No captures (x is param, y is local)
        """
        code = """
def f(x):
    y = x + 1
    return y
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, set())
        self.assertEqual(local, {'y'})
        self.assertEqual(param, {'x'})
    
    def test_closure_with_capture(self):
        """
        # Outer scope has: base
        def inner(x):
            return x + base
            
        Captures: base
        """
        code = """
def inner(x):
    return x + base
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.from_var_list(['base'])
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, {'base'})
        self.assertEqual(local, set())
        self.assertEqual(param, {'x'})
    
    def test_multiple_locals(self):
        """
        def f(x):
            a = 1
            b = 2
            c = a + b
            return c
            
        All locals, no captures
        """
        code = """
def f(x):
    a = 1
    b = 2
    c = a + b
    return c
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, set())
        self.assertEqual(local, {'a', 'b', 'c'})
        self.assertEqual(param, {'x'})
    
    def test_multiple_captures(self):
        """
        # Outer scope: x, y, z
        def f(a):
            return x + y + z + a
            
        Captures: x, y, z
        """
        code = """
def f(a):
    return x + y + z + a
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.from_var_list(['x', 'y', 'z'])
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, {'x', 'y', 'z'})
        self.assertEqual(local, set())
        self.assertEqual(param, {'a'})
    
    def test_mixed_local_and_capture(self):
        """
        # Outer scope: base, multiplier
        def f(x):
            temp = x + base
            result = temp * multiplier
            return result
            
        Captures: base, multiplier
        Locals: temp, result
        """
        code = """
def f(x):
    temp = x + base
    result = temp * multiplier
    return result
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.from_var_list(['base', 'multiplier'])
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, {'base', 'multiplier'})
        self.assertEqual(local, {'temp', 'result'})
        self.assertEqual(param, {'x'})
    
    def test_for_loop_target_is_local(self):
        """
        def f():
            for i in range(10):
                pass
            return i
            
        i is local (for loop target)
        """
        code = """
def f():
    for i in range(10):
        pass
    return i
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, set())  # range is builtin, not captured
        self.assertEqual(local, {'i'})
        self.assertEqual(param, set())
    
    def test_annotated_assignment(self):
        """
        def f(x):
            y: int = x + 1
            return y
            
        y is local with annotation
        """
        code = """
def f(x):
    y: int = x + 1
    return y
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, set())
        self.assertEqual(local, {'y'})
        self.assertEqual(param, {'x'})
    
    def test_reference_not_in_outer_scope(self):
        """
        # Outer scope: x
        def f():
            return x + y  # y not in outer scope
            
        Captures only x (y is undefined, not our concern)
        """
        code = """
def f():
    return x + y
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.from_var_list(['x'])
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, {'x'})  # Only x is captured
        self.assertEqual(local, set())
        self.assertEqual(param, set())
    
    def test_nested_function_def_is_local(self):
        """
        def outer():
            def inner():
                pass
            return inner
            
        inner is local (function definition)
        """
        code = """
def outer():
    def inner():
        pass
    return inner
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, set())
        self.assertEqual(local, {'inner'})
        self.assertEqual(param, set())
    
    def test_tuple_unpacking_assignment(self):
        """
        def f(x):
            a, b = x, x + 1
            return a + b
            
        a, b are locals
        """
        code = """
def f(x):
    a, b = x, x + 1
    return a + b
"""
        body, params = self._parse_function(code)
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(body, params)
        
        self.assertEqual(captured, set())
        self.assertEqual(local, {'a', 'b'})
        self.assertEqual(param, {'x'})


class TestAssignmentCollector(unittest.TestCase):
    """Test assignment collection (helper)"""
    
    def test_augmented_assignment(self):
        """
        def f():
            x = 1
            x += 1
            
        x is local (both = and +=)
        """
        code = """
def f():
    x = 1
    x += 1
"""
        tree = ast.parse(code)
        func = tree.body[0]
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(func.body, func.args.args)
        
        self.assertEqual(local, {'x'})
    
    def test_with_statement_target(self):
        """
        def f():
            with open('file') as fp:
                pass
            
        fp is local
        """
        code = """
def f():
    with open('file') as fp:
        pass
"""
        tree = ast.parse(code)
        func = tree.body[0]
        ctx = ScopeContext.from_var_list(['open'])
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(func.body, func.args.args)
        
        self.assertEqual(local, {'fp'})
        self.assertEqual(captured, {'open'})


class TestReferenceCollector(unittest.TestCase):
    """Test reference collection (helper)"""
    
    def test_nested_function_references_not_counted(self):
        """
        def outer():
            x = 1
            def inner():
                return x  # This reference doesn't count
            return inner
            
        Only 'inner' and 'x' are in outer scope
        """
        code = """
def outer():
    x = 1
    def inner():
        return x
    return inner
"""
        tree = ast.parse(code)
        func = tree.body[0]
        ctx = ScopeContext.empty()
        analyzer = ScopeAnalyzer(ctx)
        
        captured, local, param = analyzer.analyze(func.body, func.args.args)
        
        # x is local to outer (assigned)
        # inner is local to outer (function def)
        # The 'x' reference inside inner() is NOT counted
        self.assertEqual(local, {'x', 'inner'})
        self.assertEqual(captured, set())


if __name__ == '__main__':
    unittest.main()
