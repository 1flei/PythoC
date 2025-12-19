#!/usr/bin/env python3
"""
Run all tests for PC compiler in parallel

This script runs:
1. Unit tests (test/unit/)
2. Integration tests (test/integration/)
3. Example tests (test/example/)

Tests are run in parallel for faster execution.
"""

import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def run_test_suite(name: str, command: str) -> tuple[bool, str, float]:
    """Run a test suite and return (passed, output, duration)"""
    start_time = time.time()
    
    # Ensure PYTHONPATH is set for subprocess
    env = os.environ.copy()
    workspace = str(Path(__file__).parent.parent)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = workspace + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = workspace
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=workspace,
            timeout=120,
            env=env
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return True, result.stdout, duration
        else:
            # Combine stdout and stderr for error analysis
            combined_output = result.stdout + "\n" + result.stderr
            return False, combined_output, duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return False, "Test timed out", duration
    except Exception as e:
        duration = time.time() - start_time
        return False, str(e), duration

def extract_error_summary(output: str) -> list[str]:
    """Extract error summary from test output with context"""
    errors = []
    lines = output.split('\n')
    
    # Look for error patterns and get context
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in ["ERROR:", "FAIL:", "FAILED", "AssertionError", "Error:", "Traceback"]):
            # Get context: 2 lines before and 3 lines after
            start = max(0, i - 2)
            end = min(len(lines), i + 4)
            context = lines[start:end]
            errors.extend([l.strip() for l in context if l.strip()])
            
            # Stop after finding first error block
            if len(errors) >= 10:
                break
    
    return errors[:15]  # Return first 15 lines of error context

def main():
    """Main test runner"""
    print_header("PC Compiler - Full Test Suite (Parallel)")
    
    # Define test suites - use sys.executable for cross-platform compatibility
    python_exe = sys.executable
    test_suites = [
        ("Unit Tests", f"{python_exe} test/run_unit_tests.py"),
        ("Integration Tests", f"{python_exe} test/run_integration_tests.py"),
        ("Example Tests", f"{python_exe} test/run_examples.py"),
    ]
    
    print(f"Running {len(test_suites)} test suites in parallel...\n")
    
    # Run tests in parallel
    results = {}
    outputs = {}
    durations = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all test suites
        future_to_suite = {
            executor.submit(run_test_suite, name, command): name
            for name, command in test_suites
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_suite):
            suite_name = future_to_suite[future]
            try:
                passed, output, duration = future.result()
                results[suite_name] = passed
                outputs[suite_name] = output
                durations[suite_name] = duration
                
                # Print immediate feedback
                status = f"{GREEN}OK{RESET}" if passed else f"{RED}FAIL{RESET}"
                print(f"{status} {suite_name} completed in {duration:.2f}s")
                
            except Exception as e:
                results[suite_name] = False
                outputs[suite_name] = str(e)
                durations[suite_name] = 0
                print(f"{RED}FAIL{RESET} {suite_name} failed with exception: {e}")
    
    # Print detailed results - only show failures
    failed_suites = [(name, cmd) for name, cmd in test_suites if not results.get(name, False)]
    
    if failed_suites:
        print_header("Failed Test Details")
        
        for name, command in failed_suites:
            output = outputs.get(name, "")
            duration = durations.get(name, 0)
            
            print(f"\n{RED}FAIL{RESET} {name} ({duration:.2f}s)")
            
            # Show error summary with context
            errors = extract_error_summary(output)
            if errors:
                print(f"{YELLOW}Error details:{RESET}")
                for error in errors:
                    print(f"  {error}")
            else:
                # Fallback: show last few lines
                output_lines = output.split('\n')
                relevant_lines = [l for l in output_lines[-10:] if l.strip()]
                if relevant_lines:
                    print(f"{YELLOW}Last output lines:{RESET}")
                    for line in relevant_lines[:8]:
                        print(f"  {line}")
    
    # Print summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    total_duration = sum(durations.values())
    
    for name in results:
        status = f"{GREEN}PASS{RESET}" if results[name] else f"{RED}FAIL{RESET}"
        print(f"{status} {name}")
    
    print(f"\n{BOLD}Total: {total}{RESET}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"{BLUE}Total time: {total_duration:.2f}s{RESET}\n")
    
    if failed == 0:
        print(f"{GREEN}{BOLD} All tests passed!{RESET}\n")
        return 0
    else:
        print(f"{RED}{BOLD} Some tests failed{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
