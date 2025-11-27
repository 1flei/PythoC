#!/usr/bin/env python3
"""
Run all example tests and verify their outputs

This script runs all examples in test/example/ directory and checks
if they produce expected outputs or run without errors.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class ExampleTest:
    """Represents a single example test"""
    def __init__(self, name: str, command: str, expected_output: Optional[List[str]] = None, 
                 should_succeed: bool = True, timeout: int = 30):
        self.name = name
        self.command = command
        self.expected_output = expected_output or []
        self.should_succeed = should_succeed
        self.timeout = timeout
        self.passed = False
        self.output = ""
        self.error = ""

def run_command(cmd: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="."
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)

def check_output(output: str, expected_patterns: List[str]) -> bool:
    """Check if output contains all expected patterns"""
    if not expected_patterns:
        return True
    
    for pattern in expected_patterns:
        if pattern not in output:
            return False
    return True

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def print_test_result(test: ExampleTest):
    """Print test result"""
    status = f"{GREEN}PASS{RESET}" if test.passed else f"{RED}FAIL{RESET}"
    print(f"{status} {test.name}")
    
    if not test.passed and test.error:
        print(f"  {RED}Error: {test.error}{RESET}")
    
    if test.expected_output and not test.passed:
        print(f"  {YELLOW}Expected patterns not found in output{RESET}")

def main():
    """Main test runner"""
    print_header("PC Compiler - Example Test Suite")
    
    # Define all example tests
    tests = [
        ExampleTest(
            name="Binary Tree Test",
            command="PYTHONPATH=. python test/example/pc_binary_tree_test.py; ./build/test/example/pc_binary_tree_test 10",
            expected_output=["stretch tree of depth", "long lived tree of depth"],
            timeout=30
        ),
        ExampleTest(
            name="Multiple Files - Main",
            command="PYTHONPATH=. python test/example/multiple_files/main.py; ./build/test/example/multiple_files/main",
            expected_output=[
                "Multi-File Test Demo",
                "Testing Node Operations",
                "List count: 3",
                "List sum: 60",
                "Testing Stack Operations",
                "Stack size: 3",
                "Top value: 300",
                "Testing Cross-Module Operations",
                "All tests completed"
            ],
            timeout=30
        ),
        ExampleTest(
            name="Multiple Files - Simple Test",
            command="PYTHONPATH=. python -m test.example.multiple_files.simple_test",
            expected_output=[
                "Simple Multi-File Test",
                "Node value: 42",
                "Stack size: 0",
                "Top value: 100"
            ],
            timeout=30
        ),
    ]
    
    # Run all tests
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        print(f"\n{BOLD}[{i}/{len(tests)}]{RESET} Running: {test.name}")
        print(f"  Command: {YELLOW}{test.command}{RESET}")
        
        exit_code, stdout, stderr = run_command(test.command, test.timeout)
        test.output = stdout
        test.error = stderr
        
        # Check if test passed
        if test.should_succeed:
            if exit_code == 0:
                if check_output(stdout + stderr, test.expected_output):
                    test.passed = True
                    passed += 1
                else:
                    test.passed = False
                    failed += 1
                    test.error = "Expected output patterns not found"
            else:
                test.passed = False
                failed += 1
                if not test.error:
                    test.error = f"Exit code: {exit_code}"
        else:
            # Test should fail
            if exit_code != 0:
                test.passed = True
                passed += 1
            else:
                test.passed = False
                failed += 1
                test.error = "Expected to fail but succeeded"
        
        print_test_result(test)
        
        # Show output for failed tests
        if not test.passed and (stdout or stderr):
            print(f"\n  {YELLOW}Output:{RESET}")
            if stdout:
                for line in stdout.split('\n')[:10]:  # Show first 10 lines
                    print(f"    {line}")
            if stderr:
                print(f"  {YELLOW}Error output:{RESET}")
                for line in stderr.split('\n')[:10]:
                    print(f"    {line}")
    
    # Print summary
    print_header("Test Summary")
    total = len(tests)
    print(f"Total tests: {BOLD}{total}{RESET}")
    print(f"Passed: {GREEN}{BOLD}{passed}{RESET}")
    print(f"Failed: {RED}{BOLD}{failed}{RESET}")
    
    if failed == 0:
        print(f"\n{GREEN}{BOLD} All examples passed!{RESET}\n")
        return 0
    else:
        print(f"\n{RED}{BOLD} Some examples failed{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
