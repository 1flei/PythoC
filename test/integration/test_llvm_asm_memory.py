"""Asm memory operands read and write ordinary PythoC variables."""

import platform
import unittest

from pythoc import array, bool, compile, i32, llvm_asm, ptr, u16


IS_X86 = platform.machine().lower() in ('x86_64', 'amd64')


@compile
class Address:
    target: ptr[u16]


@compile
def array_operand() -> i32:
    values: array[i32, 2] = [3, 7]
    llvm_asm('', '=*m', inputs=[ptr(values)])
    return values[1]


if IS_X86:
    @compile
    def memory_store(value: i32) -> i32:
        target: i32 = -1
        llvm_asm('movl $1, $0', '=*m,r', inputs=[ptr(target), value])
        return target

    @compile
    def memory_load(value: i32) -> i32:
        source: i32 = value
        return llvm_asm('movl $1, $0', '=r,*m', ret_type=i32,
                        inputs=[ptr(source)])

    @compile
    def mixed_outputs(value: i32) -> bool:
        target: i32 = -1
        result: i32 = llvm_asm('movl $2, $1; movl $2, $0', '=r,=*m,r',
                              ret_type=i32, inputs=[ptr(target), value])
        return result == value and target == value

    @compile
    def memory_copy(value: i32) -> i32:
        source: i32 = value
        target: i32 = -1
        llvm_asm('movl $1, %eax; movl %eax, $0', '=*m,*m,~{eax}',
                 inputs=[ptr(target), ptr(source)])
        return target

    @compile
    def increment_memory(initial: i32) -> i32:
        target: i32 = initial
        llvm_asm('addl $$1, $0', '=*m,*m,~{flags}',
                 inputs=[ptr(target), ptr(target)])
        return target

    @compile
    def store_through_field() -> u16:
        value: u16 = 7
        address: Address = (ptr(value),)
        llvm_asm('movw $$17185, $0', '=*m', inputs=[address.target])
        return value

    @compile
    def control_word() -> u16:
        result: u16 = 0
        llvm_asm('fnstcw $0', '=*m', inputs=[ptr(result)])
        return result


class AsmMemoryTest(unittest.TestCase):
    def test_array_pointee_has_aggregate_element_type(self):
        self.assertEqual(array_operand(), 7)

    @unittest.skipUnless(IS_X86, 'x86 assembly')
    def test_register_and_memory_outputs_use_different_operand_indices(self):
        for value in (-21, 0, 42):
            with self.subTest(value=value):
                self.assertEqual(memory_store(value), value)
                self.assertEqual(memory_load(value), value)
                self.assertTrue(mixed_outputs(value))

    @unittest.skipUnless(IS_X86, 'x86 assembly')
    def test_memory_inputs_clobbers_and_aliases(self):
        self.assertEqual(memory_copy(41), 41)
        self.assertEqual(increment_memory(41), 42)

    @unittest.skipUnless(IS_X86, 'x86 assembly')
    def test_pointer_field_uses_semantic_pointee_type(self):
        self.assertEqual(store_through_field(), 17185)

    @unittest.skipUnless(IS_X86, 'x86 assembly')
    def test_x87_control_word_is_materialized(self):
        self.assertEqual(int(control_word()) & 0x3F, 0x3F)


if __name__ == '__main__':
    unittest.main()
