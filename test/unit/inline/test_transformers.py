"""
Tests for AST transformers

Tests body transformation with variable renaming and exit point handling
"""

import ast
import unittest
from pythoc.inline.transformers import InlineBodyTransformer
from pythoc.inline.exit_rules import ReturnExitRule, YieldExitRule


class TestInlineBodyTransformer(unittest.TestCase):
    """Test body transformation"""
    
    def _parse_body(self, code: str):
        """Helper: Parse function and extract body"""
        tree = ast.parse(code)
        return tree.body[0].body
    
    def test_simple_renaming(self):
        """
        Rename local variables:
        x = 1  -->  x_inline_1 = 1
        return x  -->  result = move(x_inline_1)
        """
        code = """
def f():
    x = 1
    return x
"""
        body = self._parse_body(code)
        rename_map = {'x': 'x_inline_1'}
        rule = ReturnExitRule(result_var='result')
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # First statement: x_inline_1 = 1
        assign = new_body[0]
        self.assertIsInstance(assign, ast.Assign)
        self.assertEqual(assign.targets[0].id, 'x_inline_1')
        
        # Second statement: result = move(x_inline_1) (return transformed with move wrapper)
        result_assign = new_body[1]
        self.assertIsInstance(result_assign, ast.Assign)
        self.assertEqual(result_assign.targets[0].id, 'result')
        # Value is wrapped in move() for linear type support
        self.assertIsInstance(result_assign.value, ast.Call)
        self.assertEqual(result_assign.value.func.id, 'move')
        self.assertEqual(result_assign.value.args[0].id, 'x_inline_1')
    
    def test_no_renaming_for_params(self):
        """
        Parameters not in rename_map stay unchanged:
        return a + b  -->  result = move(a + b)
        """
        code = """
def f():
    return a + b
"""
        body = self._parse_body(code)
        rename_map = {}  # No renaming
        rule = ReturnExitRule(result_var='result')
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # result = move(a + b)
        assign = new_body[0]
        # Value is wrapped in move() for linear type support
        self.assertIsInstance(assign.value, ast.Call)
        self.assertEqual(assign.value.func.id, 'move')
        # The argument to move() is the BinOp
        binop = assign.value.args[0]
        self.assertIsInstance(binop, ast.BinOp)
        self.assertEqual(binop.left.id, 'a')
        self.assertEqual(binop.right.id, 'b')
    
    def test_partial_renaming(self):
        """
        Rename only locals, not params/captures:
        return a + local_var
        
        With rename_map = {'local_var': 'local_var_inline_1'}
        -->  result = move(a + local_var_inline_1)
        """
        code = """
def f():
    return a + local_var
"""
        body = self._parse_body(code)
        rename_map = {'local_var': 'local_var_inline_1'}
        rule = ReturnExitRule(result_var='result')
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # result = move(a + local_var_inline_1)
        assign = new_body[0]
        # Value is wrapped in move() for linear type support
        self.assertIsInstance(assign.value, ast.Call)
        self.assertEqual(assign.value.func.id, 'move')
        # The argument to move() is the BinOp
        binop = assign.value.args[0]
        self.assertEqual(binop.left.id, 'a')  # Not renamed
        self.assertEqual(binop.right.id, 'local_var_inline_1')  # Renamed
    
    def test_while_loop_transformation(self):
        """
        Transform while loop body:
        while x < 10:
            x = x + 1
            
        With rename_map = {'x': 'x_inline_1'}
        """
        code = """
def f():
    while x < 10:
        x = x + 1
"""
        body = self._parse_body(code)
        rename_map = {'x': 'x_inline_1'}
        rule = ReturnExitRule()
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # while x_inline_1 < 10:
        while_stmt = new_body[0]
        self.assertIsInstance(while_stmt, ast.While)
        self.assertEqual(while_stmt.test.left.id, 'x_inline_1')
        
        # Body: x_inline_1 = x_inline_1 + 1
        assign = while_stmt.body[0]
        self.assertEqual(assign.targets[0].id, 'x_inline_1')
        self.assertEqual(assign.value.left.id, 'x_inline_1')
    
    def test_if_statement_transformation(self):
        """
        Transform if statement:
        if x > 0:
            y = x
        else:
            y = 0
            
        With rename_map = {'x': 'x_i1', 'y': 'y_i1'}
        """
        code = """
def f():
    if x > 0:
        y = x
    else:
        y = 0
"""
        body = self._parse_body(code)
        rename_map = {'x': 'x_i1', 'y': 'y_i1'}
        rule = ReturnExitRule()
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # if x_i1 > 0:
        if_stmt = new_body[0]
        self.assertIsInstance(if_stmt, ast.If)
        self.assertEqual(if_stmt.test.left.id, 'x_i1')
        
        # Then: y_i1 = x_i1
        then_assign = if_stmt.body[0]
        self.assertEqual(then_assign.targets[0].id, 'y_i1')
        self.assertEqual(then_assign.value.id, 'x_i1')
        
        # Else: y_i1 = 0
        else_assign = if_stmt.orelse[0]
        self.assertEqual(else_assign.targets[0].id, 'y_i1')
    
    def test_for_loop_transformation(self):
        """
        Transform for loop:
        for i in items:
            process(i)
            
        With rename_map = {'i': 'i_inline_1'}
        """
        code = """
def f():
    for i in items:
        process(i)
"""
        body = self._parse_body(code)
        rename_map = {'i': 'i_inline_1'}
        rule = ReturnExitRule()
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # for i_inline_1 in items:
        for_stmt = new_body[0]
        self.assertIsInstance(for_stmt, ast.For)
        self.assertEqual(for_stmt.target.id, 'i_inline_1')
        
        # Body: process(i_inline_1)
        call = for_stmt.body[0]
        self.assertEqual(call.value.args[0].id, 'i_inline_1')
    
    def test_return_exit_transformation(self):
        """
        Transform return with ReturnExitRule:
        return x + 1  -->  result = x + 1
        """
        code = """
def f():
    return x + 1
"""
        body = self._parse_body(code)
        rename_map = {}
        rule = ReturnExitRule(result_var='result')
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # result = x + 1; break
        self.assertEqual(len(new_body), 2)
        assign = new_body[0]
        self.assertIsInstance(assign, ast.Assign)
        self.assertEqual(assign.targets[0].id, 'result')
        # Second statement is break
        self.assertIsInstance(new_body[1], ast.Break)
    
    def test_yield_exit_transformation(self):
        """
        Transform yield with YieldExitRule:
        yield i  -->  loop_var = i; loop_body
        """
        code = """
def f():
    yield i
"""
        body = self._parse_body(code)
        rename_map = {}
        
        # Create loop body: x = x + 1
        loop_body_code = "x = x + 1"
        loop_body_tree = ast.parse(loop_body_code)
        loop_body = loop_body_tree.body
        
        rule = YieldExitRule(loop_var='loop_var', loop_body=loop_body)
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # Should have: loop_var = move(i); x = x + 1
        self.assertGreaterEqual(len(new_body), 2)
        
        # First: loop_var = move(i)
        assign = new_body[0]
        self.assertEqual(assign.targets[0].id, 'loop_var')
        # Value is now wrapped in move() for linear type support
        self.assertIsInstance(assign.value, ast.Call)
        self.assertEqual(assign.value.func.id, 'move')
        self.assertEqual(assign.value.args[0].id, 'i')
        
        # Second: x = x + 1 (from loop body)
        loop_assign = new_body[1]
        self.assertIsInstance(loop_assign, ast.Assign)
    
    def test_multiple_returns(self):
        """
        Transform multiple returns:
        if cond:
            return a
        else:
            return b
            
        Both returns should be transformed with move() wrapper
        """
        code = """
def f():
    if cond:
        return a
    else:
        return b
"""
        body = self._parse_body(code)
        rename_map = {}
        rule = ReturnExitRule(result_var='result')
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # if cond:
        if_stmt = new_body[0]
        
        # Then: result = move(a)
        then_assign = if_stmt.body[0]
        self.assertEqual(then_assign.targets[0].id, 'result')
        # Value is wrapped in move() for linear type support
        self.assertIsInstance(then_assign.value, ast.Call)
        self.assertEqual(then_assign.value.func.id, 'move')
        self.assertEqual(then_assign.value.args[0].id, 'a')
        
        # Else: result = move(b)
        else_assign = if_stmt.orelse[0]
        self.assertEqual(else_assign.targets[0].id, 'result')
        # Value is wrapped in move() for linear type support
        self.assertIsInstance(else_assign.value, ast.Call)
        self.assertEqual(else_assign.value.func.id, 'move')
        self.assertEqual(else_assign.value.args[0].id, 'b')
    
    def test_nested_structures(self):
        """
        Transform nested control structures:
        while cond:
            if test:
                x = 1
            else:
                x = 2
                
        With rename_map = {'x': 'x_i1'}
        """
        code = """
def f():
    while cond:
        if test:
            x = 1
        else:
            x = 2
"""
        body = self._parse_body(code)
        rename_map = {'x': 'x_i1'}
        rule = ReturnExitRule()
        
        transformer = InlineBodyTransformer(rule, rename_map)
        new_body = transformer.transform(body)
        
        # while cond:
        while_stmt = new_body[0]
        
        # if test:
        if_stmt = while_stmt.body[0]
        
        # Then: x_i1 = 1
        then_assign = if_stmt.body[0]
        self.assertEqual(then_assign.targets[0].id, 'x_i1')
        
        # Else: x_i1 = 2
        else_assign = if_stmt.orelse[0]
        self.assertEqual(else_assign.targets[0].id, 'x_i1')


if __name__ == '__main__':
    unittest.main()
