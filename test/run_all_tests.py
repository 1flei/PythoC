#!/usr/bin/env python3
"""
Run all tests for PC compiler sequentially

This script runs:
1. Unit tests (test/unit/)
2. Integration tests (test/integration/)
3. Example tests (test/example/)

Tests are run sequentially to avoid CPU / filesystem contention on CI.
Each suite's output is printed directly as if run individually.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def run_test_suite(name: str, command: list) -> tuple[bool, float]:
    """Run a test suite, streaming its output directly to stdout/stderr.

    Returns (passed, duration).
    """
    print_header(name)
    start_time = time.time()

    env = os.environ.copy()
    workspace = str(Path(__file__).parent.parent)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = workspace + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = workspace

    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            timeout=600,
            env=env,
            stdin=subprocess.DEVNULL
        )
        duration = time.time() - start_time
        passed = result.returncode == 0
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"\n{RED}Test suite timed out after 600 seconds{RESET}")
        passed = False
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n{RED}Test suite failed with exception: {e}{RESET}")
        passed = False

    status = f"{GREEN}OK{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"\n{status} {name} completed in {duration:.2f}s")
    return passed, duration

def main():
    """Main test runner"""
    print_header("PC Compiler - Full Test Suite")

    python_exe = sys.executable
    test_suites = [
        ("Unit Tests", [python_exe, "test/run_unit_tests.py"]),
        ("Integration Tests", [python_exe, "test/run_integration_tests.py"]),
        ("Example Tests", [python_exe, "test/run_examples.py"]),
    ]

    print(f"Running {len(test_suites)} test suites sequentially...\n")

    suite_results = []
    for name, command in test_suites:
        passed, duration = run_test_suite(name, command)
        suite_results.append((name, passed, duration))

    # Final summary
    print_header("Test Summary")

    total = len(suite_results)
    num_passed = sum(1 for _, p, _ in suite_results if p)
    num_failed = total - num_passed
    total_duration = sum(d for _, _, d in suite_results)

    for name, passed, duration in suite_results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"{status} {name} ({duration:.2f}s)")

    print(f"\n{BOLD}Total: {total}{RESET}")
    print(f"{GREEN}Passed: {num_passed}{RESET}")
    print(f"{RED}Failed: {num_failed}{RESET}")
    print(f"{BLUE}Total time: {total_duration:.2f}s{RESET}\n")

    if num_failed == 0:
        print(f"{GREEN}{BOLD} All tests passed!{RESET}\n")
        return 0
    else:
        print(f"{RED}{BOLD} Some tests failed{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
