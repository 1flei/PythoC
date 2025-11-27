#!/usr/bin/env python3
"""
Run all integration tests in parallel

This script runs all integration tests in test/integration/ directory.
Integration tests are Python scripts that compile and execute PC code.
Tests are run in parallel for faster execution.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_single_test(test_file: Path) -> Tuple[str, bool, str, str, float]:
    """Run a single test file and return results"""
    test_name = test_file.stem
    start_time = time.time()
    
    cmd = f"PYTHONPATH=. python {test_file}"
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="."
        )
        duration = time.time() - start_time
        success = result.returncode == 0
        return test_name, success, result.stdout, result.stderr, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return test_name, False, "", f"Test timed out after 30 seconds", duration
    except Exception as e:
        duration = time.time() - start_time
        return test_name, False, "", str(e), duration

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def extract_error_info(stdout: str, stderr: str) -> list:
    """Extract error information from test output"""
    errors = []
    
    # Combine stdout and stderr for analysis
    combined = stdout + '\n' + stderr
    lines = combined.split('\n')
    
    # Find traceback or assertion errors
    in_traceback = False
    traceback_lines = []
    
    for i, line in enumerate(lines):
        # Start of traceback
        if 'Traceback (most recent call last):' in line:
            in_traceback = True
            traceback_lines = [line.strip()]
            continue
        
        # Collect traceback lines
        if in_traceback:
            stripped = line.strip()
            if stripped:
                traceback_lines.append(stripped)
                # Stop at the actual error
                if 'Error:' in line or 'Error ' in line:
                    errors.extend(traceback_lines)
                    break
        
        # Direct assertion or error
        elif 'AssertionError' in line or 'FAILED' in line:
            # Get context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            errors.extend([l.strip() for l in lines[start:end] if l.strip()])
            break
    
    return errors[:12]

def main():
    """Main test runner"""
    print_header("PC Compiler - Integration Test Suite (Parallel)")
    
    # Find all test files in integration directory
    integration_dir = Path("./test/integration")
    test_files = sorted(integration_dir.glob("test_*.py"))
    
    if not test_files:
        print(f"{YELLOW}No integration test files found{RESET}")
        return 0
    
    print(f"Found {len(test_files)} integration test files")
    print(f"Running tests in parallel with 8 workers...\n")
    
    passed = 0
    failed = 0
    results = []
    
    # Run tests in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all tests
        future_to_test = {
            executor.submit(run_single_test, test_file): test_file.stem
            for test_file in test_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_test):
            test_name = future_to_test[future]
            try:
                name, success, stdout, stderr, duration = future.result()
                
                # Print immediate feedback
                if success:
                    print(f"{GREEN}OK{RESET} {name} ({duration:.2f}s)")
                    passed += 1
                else:
                    print(f"{RED}FAIL{RESET} {name} ({duration:.2f}s)")
                    failed += 1
                
                results.append((name, success, stdout, stderr, duration))
                
            except Exception as e:
                print(f"{RED}FAIL{RESET} {test_name} (exception: {e})")
                failed += 1
                results.append((test_name, False, "", str(e), 0))
    
    # Sort results by test name for consistent output
    results.sort(key=lambda x: x[0])
    
    # Print detailed summary
    print_header("Test Summary")
    
    for test_name, success, stdout, stderr, duration in results:
        status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
        print(f"{status} {test_name}")
        
        # Show error details for failed tests
        if not success:
            errors = extract_error_info(stdout, stderr)
            if errors:
                print(f"{YELLOW}  Error details:{RESET}")
                for error in errors:
                    print(f"    {error}")
    
    total = len(test_files)
    total_time = sum(r[4] for r in results)
    print(f"\n{BOLD}Total: {total}{RESET}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"{BLUE}Total time: {total_time:.2f}s{RESET}\n")
    
    if failed == 0:
        print(f"{GREEN}{BOLD} All integration tests passed!{RESET}\n")
        return 0
    else:
        print(f"{RED}{BOLD} Some integration tests failed{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
