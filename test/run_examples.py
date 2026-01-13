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
from typing import List, Tuple, Optional

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def get_workspace() -> Path:
    """Get workspace root directory"""
    return Path(__file__).parent.parent


def get_env() -> dict:
    """Get environment with PYTHONPATH set"""
    env = os.environ.copy()
    workspace = str(get_workspace())
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = workspace + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = workspace
    return env


def run_python_script(script_path: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Run a Python script and return exit code, stdout, stderr"""
    workspace = get_workspace()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace),
            env=get_env(),
            stdin=subprocess.DEVNULL
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Script timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)


def run_python_module(module_name: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Run a Python module and return exit code, stdout, stderr"""
    workspace = get_workspace()
    try:
        result = subprocess.run(
            [sys.executable, '-m', module_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace),
            env=get_env(),
            stdin=subprocess.DEVNULL
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Module timed out after {timeout} seconds"
    except Exception as e:
        return -1, "", str(e)


def run_executable(exe_path: str, args: List[str] = None,
                   timeout: int = 30) -> Tuple[int, str, str]:
    """Run an executable and return exit code, stdout, stderr"""
    workspace = get_workspace()
    exe_full_path = workspace / exe_path
    
    # On Windows, executables have .exe extension
    if sys.platform == 'win32' and not exe_full_path.suffix:
        exe_full_path = exe_full_path.with_suffix('.exe')
    
    if not exe_full_path.exists():
        return -1, "", f"Executable not found: {exe_full_path}"
    
    cmd = [str(exe_full_path)] + (args or [])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace),
            env=get_env(),
            stdin=subprocess.DEVNULL
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Executable timed out after {timeout} seconds"
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


class ExampleTest:
    """Represents a single example test"""
    def __init__(self, name: str, expected_output: Optional[List[str]] = None,
                 timeout: int = 30):
        self.name = name
        self.expected_output = expected_output or []
        self.timeout = timeout
        self.passed = False
        self.output = ""
        self.error = ""
    
    def run(self) -> bool:
        """Run the test - to be implemented by subclasses"""
        raise NotImplementedError


class CompileAndRunTest(ExampleTest):
    """Test that compiles a Python script then runs the resulting executable"""
    def __init__(self, name: str, script_path: str, exe_path: str,
                 exe_args: List[str] = None,
                 expected_output: Optional[List[str]] = None, timeout: int = 30):
        super().__init__(name, expected_output, timeout)
        self.script_path = script_path
        self.exe_path = exe_path
        self.exe_args = exe_args or []
    
    def run(self) -> bool:
        # Step 1: Compile
        exit_code, stdout, stderr = run_python_script(self.script_path, self.timeout)
        if exit_code != 0:
            self.error = f"Compilation failed: {stderr}"
            self.output = stdout
            return False
        
        # Step 2: Run executable
        exit_code, stdout, stderr = run_executable(
            self.exe_path, self.exe_args, self.timeout
        )
        self.output = stdout
        
        if exit_code != 0:
            self.error = f"Execution failed (exit {exit_code}): {stderr}"
            return False
        
        # Step 3: Check output
        if not check_output(stdout + stderr, self.expected_output):
            self.error = "Expected output patterns not found"
            return False
        
        self.passed = True
        return True


class ModuleTest(ExampleTest):
    """Test that runs a Python module"""
    def __init__(self, name: str, module_name: str,
                 expected_output: Optional[List[str]] = None, timeout: int = 30):
        super().__init__(name, expected_output, timeout)
        self.module_name = module_name
    
    def run(self) -> bool:
        exit_code, stdout, stderr = run_python_module(self.module_name, self.timeout)
        self.output = stdout
        
        if exit_code != 0:
            self.error = f"Module failed (exit {exit_code}): {stderr}"
            return False
        
        if not check_output(stdout + stderr, self.expected_output):
            self.error = "Expected output patterns not found"
            return False
        
        self.passed = True
        return True


def print_test_result(test: ExampleTest):
    """Print test result"""
    status = f"{GREEN}PASS{RESET}" if test.passed else f"{RED}FAIL{RESET}"
    print(f"{status} {test.name}")
    
    if not test.passed and test.error:
        print(f"  {RED}Error: {test.error}{RESET}")
    
    # Show output for failed tests
    if not test.passed and test.output:
        print(f"\n  {YELLOW}Output:{RESET}")
        for line in test.output.split('\n')[:10]:
            if line.strip():
                print(f"    {line}")


def main():
    """Main test runner"""
    print_header("PC Compiler - Example Test Suite")

    example_test_timeout = 60
    
    # Define all example tests
    tests: List[ExampleTest] = [
        CompileAndRunTest(
            name="Binary Tree Test",
            script_path="test/example/pc_binary_tree_test.py",
            exe_path="build/test/example/pc_binary_tree_test",
            exe_args=["10"],
            expected_output=["stretch tree of depth", "long lived tree of depth"],
            timeout=example_test_timeout
        ),
        CompileAndRunTest(
            name="Multiple Files - Main",
            script_path="test/example/multiple_files/main.py",
            exe_path="build/test/example/multiple_files/main",
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
            timeout=example_test_timeout
        ),
        ModuleTest(
            name="Multiple Files - Simple Test",
            module_name="test.example.multiple_files.simple_test",
            expected_output=[
                "Simple Multi-File Test",
                "Node value: 42",
                "Stack size: 0",
                "Top value: 100"
            ],
            timeout=example_test_timeout
        ),
    ]
    
    # Run all tests
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        print(f"\n{BOLD}[{i}/{len(tests)}]{RESET} Running: {test.name}")
        
        if test.run():
            passed += 1
        else:
            failed += 1
        
        print_test_result(test)
    
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
