#!/usr/bin/env python3
"""
Run all integration tests

This script runs all integration tests in test/integration/ directory.
Integration tests are Python scripts that compile and execute PC code.

Usage:
    python run_integration_tests.py           # Run in parallel (default)
    python run_integration_tests.py --serial  # Run serially (for benchmarking)
    python run_integration_tests.py --json    # Output JSON (for CI)
"""

import subprocess
import sys
import os
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any
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
    """Run a single test file and return results."""
    test_name = test_file.stem
    start_time = time.time()
    
    # Use sys.executable for cross-platform compatibility
    # Set PYTHONPATH via env dict instead of shell syntax
    env = os.environ.copy()
    workspace = str(Path(__file__).parent.parent)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = workspace + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = workspace
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL
        )
        duration = time.time() - start_time
        success = result.returncode == 0
        return test_name, success, result.stdout, result.stderr, duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return test_name, False, "", f"Test timed out after 120 seconds", duration
    except Exception as e:
        duration = time.time() - start_time
        return test_name, False, "", str(e), duration

def print_header(text: str, quiet: bool = False):
    """Print a formatted header"""
    if quiet:
        return
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

def run_tests_serial(test_files: List[Path], quiet: bool = False) -> List[Tuple[str, bool, str, str, float]]:
    """Run tests serially (for accurate timing benchmarks)"""
    results = []
    for i, test_file in enumerate(test_files):
        test_name, success, stdout, stderr, duration = run_single_test(test_file)
        if not quiet:
            status = f"{GREEN}OK{RESET}" if success else f"{RED}FAIL{RESET}"
            print(f"[{i+1}/{len(test_files)}] {status} {test_name} ({duration:.2f}s)")
        results.append((test_name, success, stdout, stderr, duration))
    return results


def run_tests_parallel(test_files: List[Path], max_workers: int = None) -> List[Tuple[str, bool, str, str, float]]:
    """Run tests in parallel (for faster execution)"""
    if max_workers is None:
        # Limit default parallelism to avoid filesystem contention on Windows.
        # 32 workers all compiling the same shared bindings creates heavy lock
        # contention and TOCTOU races on .o / .dll files.
        max_workers = min(8, os.cpu_count() or 4)

    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(run_single_test, test_file): test_file.stem
            for test_file in test_files
        }
        
        for future in as_completed(future_to_test):
            test_name = future_to_test[future]
            try:
                name, success, stdout, stderr, duration = future.result()
                status = f"{GREEN}OK{RESET}" if success else f"{RED}FAIL{RESET}"
                print(f"{status} {name} ({duration:.2f}s)")
                if not success:
                    combined = (stdout.rstrip('\n') + '\n' + stderr.rstrip('\n')).strip()
                    if combined:
                        print(f"{YELLOW}  --- output ---{RESET}")
                        for line in combined.split('\n'):
                            print(f"    {line}")
                        print(f"{YELLOW}  --- end ---{RESET}")
                results.append((name, success, stdout, stderr, duration))
            except Exception as e:
                print(f"{RED}FAIL{RESET} {test_name} (exception: {e})")
                results.append((test_name, False, "", str(e), 0))
    
    return results


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Run PC integration tests')
    parser.add_argument('--serial', action='store_true',
                        help='Run tests serially (for benchmarking)')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON (for CI)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output (for benchmarking)')
    args = parser.parse_args()
    
    mode = "Serial" if args.serial else "Parallel"
    print_header(f"PC Compiler - Integration Test Suite ({mode})", args.quiet)
    
    # Find all test files in integration directory
    workspace = Path(__file__).parent.parent
    integration_dir = workspace / "test" / "integration"
    test_files = sorted(integration_dir.glob("test_*.py"))
    
    if not test_files:
        if not args.quiet:
            print(f"{YELLOW}No integration test files found{RESET}")
        return 0
    
    if not args.quiet:
        print(f"Found {len(test_files)} integration test files")
        if args.serial:
            print(f"Running tests serially...\n")
        else:
            workers = min(16, os.cpu_count() or 4)
            print(f"Running tests in parallel with {workers} workers...\n")
    
    # Record wall-clock time for the entire test run
    wall_start = time.time()
    
    # Run tests
    if args.serial:
        results = run_tests_serial(test_files, args.quiet)
    else:
        results = run_tests_parallel(test_files)
    
    wall_time = time.time() - wall_start
    
    # Sort results by test name
    results.sort(key=lambda x: x[0])
    
    # Count results
    passed = sum(1 for r in results if r[1])
    failed = len(results) - passed
    total_cpu_time = sum(r[4] for r in results)
    
    # Output JSON if requested
    if args.json:
        json_result = {
            'total': len(test_files),
            'passed': passed,
            'failed': failed,
            'wall_time_seconds': round(wall_time, 2),
            'total_cpu_time_seconds': round(total_cpu_time, 2),
            'tests': [
                {
                    'name': name,
                    'passed': success,
                    'duration_seconds': round(duration, 3)
                }
                for name, success, _, _, duration in results
            ]
        }
        print(json.dumps(json_result, indent=2))
        return 0 if failed == 0 else 1
    
    # Print detailed summary
    print_header("Test Summary", args.quiet)
    
    if not args.quiet:
        for test_name, success, stdout, stderr, duration in results:
            status = f"{GREEN}PASS{RESET}" if success else f"{RED}FAIL{RESET}"
            print(f"{status} {test_name}")
            
            # Show full output for failed tests
            if not success:
                combined = (stdout.rstrip('\n') + '\n' + stderr.rstrip('\n')).strip()
                if combined:
                    print(f"{YELLOW}  --- output ---{RESET}")
                    for line in combined.split('\n'):
                        print(f"    {line}")
                    print(f"{YELLOW}  --- end ---{RESET}")
    
    total = len(test_files)
    print(f"\n{BOLD}Total: {total}{RESET}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"{BLUE}Total CPU time: {total_cpu_time:.2f}s{RESET}")
    print(f"{BLUE}Wall time: {wall_time:.2f}s{RESET}\n")
    
    if failed == 0:
        print(f"{GREEN}{BOLD} All integration tests passed!{RESET}\n")
        return 0
    else:
        print(f"{RED}{BOLD} Some integration tests failed{RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
