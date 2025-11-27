#!/usr/bin/env python3
"""
PC Binary Tree Benchmark - Direct translation from base_binary_tree_test.c

This demonstrates PC's equivalence to C:
- PC struct classes map to C structs
- PC @compile decorator compiles functions to LLVM IR (like C compilation)
- PC type annotations (i32, ptr[T]) map directly to C types (int, T*)
- PC uses Python syntax but generates native code equivalent to C
"""

from __future__ import annotations
from pythoc import i8, i32, i64, u32, u64, f64, bool, ptr, compile, nullptr, seq, sizeof
from pythoc.libc.stdlib import malloc, free, atol
from pythoc.libc.stdio import printf
from pythoc.libc.math import pow


# C equivalent: typedef struct tn { struct tn* left; struct tn* right; } treeNode;
# PC uses @compile class with type annotations instead of typedef struct
@compile
class TreeNode:
    left: ptr[TreeNode]   # C: struct tn* left;
    right: ptr[TreeNode]  # C: struct tn* right;


# C: treeNode* NewTreeNode(treeNode* left, treeNode* right)
# PC: Same signature with Python syntax and type annotations
@compile
def NewTreeNode(left: ptr[TreeNode], right: ptr[TreeNode]) -> ptr[TreeNode]:
    # C: new = (treeNode*)malloc(sizeof(treeNode));
    # PC: Explicit type cast using ptr[TreeNode](malloc(...))
    new: ptr[TreeNode] = ptr[TreeNode](malloc(sizeof(TreeNode)))
    
    # C: new->left = left; new->right = right;
    # PC: Similar pointer member access syntax
    new.left = left
    new.right = right
    
    return new


# C: long ItemCheck(treeNode* tree)
# PC: Return type i32 (32-bit int) instead of long for consistency
@compile  
def ItemCheck(tree: ptr[TreeNode]) -> i32:
    # C: if (tree->left == NULL)
    # PC: if tree.left == nullptr (nullptr is PC's NULL)
    if tree.left == nullptr:
        return 1
    else:
        # C: return 1 + ItemCheck(tree->left) + ItemCheck(tree->right);
        # PC: Identical recursive call syntax
        return 1 + ItemCheck(tree.left) + ItemCheck(tree.right)


# C: treeNode* BottomUpTree(unsigned depth)
# PC: depth parameter uses i32 (signed) instead of unsigned for simplicity
@compile
def BottomUpTree(depth: i32) -> ptr[TreeNode]:
    # C: if (depth > 0) return NewTreeNode(BottomUpTree(depth-1), BottomUpTree(depth-1));
    # PC: Identical control flow and recursion
    if depth > 0:
        return NewTreeNode(
            BottomUpTree(depth - 1),
            BottomUpTree(depth - 1)
        )
    else:
        # C: return NewTreeNode(NULL, NULL);
        # PC: nullptr instead of NULL
        return NewTreeNode(nullptr, nullptr)


# C: void DeleteTree(treeNode* tree)
# PC: Return type annotation can be omitted, defaults to void (equivalent to C's void)
@compile
def DeleteTree(tree: ptr[TreeNode]):
    # C: if (tree->left != NULL) { DeleteTree(tree->left); DeleteTree(tree->right); }
    # PC: Same pointer comparison and recursive deletion
    if tree.left != nullptr:
        DeleteTree(tree.left)
        DeleteTree(tree.right)
    
    # C: free(tree);
    # PC: Direct call to libc free function
    free(tree)


# C: int main(int argc, char* argv[])
# PC: ptr[ptr[i8]] is equivalent to char** (array of string pointers)
@compile
def main(argc: i32, argv: ptr[ptr[i8]]) -> i32:
    # C: unsigned N, depth, minDepth, maxDepth, stretchDepth;
    # PC: Variables declared with type annotations (type inference also works)
    N: i32 = atol(argv[1])  # C: N = atol(argv[1]);
    
    minDepth: i32 = 4
    
    # C: if ((minDepth + 2) > N) maxDepth = minDepth + 2; else maxDepth = N;
    # PC: Same conditional logic with Python if/else syntax
    maxDepth: i32
    if (minDepth + 2) > N:
        maxDepth = minDepth + 2
    else:
        maxDepth = N
    
    stretchDepth: i32 = maxDepth + 1
    
    # C: stretchTree = BottomUpTree(stretchDepth);
    # PC: Same function call, type inference for local variables
    stretchTree: ptr[TreeNode] = BottomUpTree(stretchDepth)
    
    # C: printf("stretch tree of depth %u\t check: %li\n", stretchDepth, ItemCheck(stretchTree));
    # PC: Direct call to libc printf with same format string
    printf("stretch tree of depth %u\t check: %li\n", stretchDepth, ItemCheck(stretchTree))
    
    DeleteTree(stretchTree)
    
    # C: longLivedTree = BottomUpTree(maxDepth);
    longLivedTree: ptr[TreeNode] = BottomUpTree(maxDepth)
    
    # C: for (depth = minDepth; depth <= maxDepth; depth += 2)
    # PC: range(start, stop, step) is more Pythonic but compiles to same loop
    for depth in seq(minDepth, maxDepth + 1, 2):
        # C: long i, iterations, check;
        check: i32 = 0
        
        # C: iterations = pow(2, maxDepth - depth + minDepth);
        # PC: Explicit type conversions using type constructors
        d: f64 = f64(maxDepth - depth + minDepth)
        iterations: i32 = i32(pow(2.0, d))
        
        # C: for (i = 1; i <= iterations; i++)
        # PC: range(1, iterations + 1) generates same loop
        for i in seq(1, iterations + 1):
            # C: tempTree = BottomUpTree(depth);
            tempTree: ptr[TreeNode] = BottomUpTree(depth)
            # C: check += ItemCheck(tempTree);
            check = check + ItemCheck(tempTree)
            # C: DeleteTree(tempTree);
            DeleteTree(tempTree)
        
        # C: printf("%li\t trees of depth %u\t check: %li\n", iterations, depth, check);
        printf("%li\t trees of depth %u\t check: %li\n", iterations, depth, check)
    
    # C: printf("long lived tree of depth %u\t check: %li\n", maxDepth, ItemCheck(longLivedTree));
    printf("long lived tree of depth %u\t check: %li\n", maxDepth, ItemCheck(longLivedTree))
    
    # C: return 0;
    return 0

if __name__ == "__main__":
    from pythoc import compile_to_executable
    compile_to_executable()
