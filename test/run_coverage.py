#!/usr/bin/env python3
"""
Run integration and example tests with coverage collection.
Each test runs in a subprocess for isolation, coverage data is merged at the end.

Usage:
    PYTHONPATH=. python test/run_coverage.py

Output:
    test_report/coverage_report.txt  - Text summary
    test_report/htmlcov/             - HTML coverage report
"""

import subprocess
import sys
import os
import time
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
REPORT_DIR = PROJECT_ROOT / 'test_report'
COVERAGE_DIR = REPORT_DIR / 'coverage_data'
HTML_DIR = REPORT_DIR / 'htmlcov'

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text: str, file=None):
    line = f"\n{'='*70}\n{text:^70}\n{'='*70}\n"
    print(f"{BOLD}{BLUE}{line}{RESET}")
    if file:
        file.write(line + '\n')


def run_test_with_coverage(test_file: Path, index: int) -> tuple[str, bool, float, str]:
    """Run a single test file with coverage in a subprocess."""
    test_name = test_file.stem
    start_time = time.time()
    
    # Each test gets its own coverage data file
    cov_file = COVERAGE_DIR / f'.coverage.{index}'
    
    cmd = [
        sys.executable, '-m', 'coverage', 'run',
        '--source=pythoc',
        '--branch',
        f'--data-file={cov_file}',
        str(test_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)},
            stdin=subprocess.DEVNULL
        )
        duration = time.time() - start_time
        success = result.returncode == 0
        
        error_msg = ""
        if not success:
            # Extract error from stderr or stdout
            output = result.stderr + result.stdout
            lines = output.strip().split('\n')
            # Find the actual error message
            for i, line in enumerate(lines):
                if 'Error' in line or 'FAIL' in line or 'assert' in line.lower():
                    error_msg = '; '.join(lines[max(0,i-1):i+2])
                    break
            if not error_msg and lines:
                error_msg = lines[-1][:200]
        
        return test_name, success, duration, error_msg
        
    except subprocess.TimeoutExpired:
        return test_name, False, 60.0, "Timeout after 60s"
    except Exception as e:
        return test_name, False, time.time() - start_time, str(e)


def merge_coverage_data():
    """Merge all coverage data files."""
    cov_files = list(COVERAGE_DIR.glob('.coverage.*'))
    if not cov_files:
        return False
    
    # Combine all coverage files
    cmd = [
        sys.executable, '-m', 'coverage', 'combine',
        f'--data-file={REPORT_DIR / ".coverage"}',
    ] + [str(f) for f in cov_files]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL
    )
    return result.returncode == 0


def generate_reports(report_file):
    """Generate coverage reports."""
    cov_data = REPORT_DIR / '.coverage'
    
    # Text report
    print_header("Coverage Summary", report_file)
    
    cmd = [
        sys.executable, '-m', 'coverage', 'report',
        f'--data-file={cov_data}',
        '--show-missing'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                            stdin=subprocess.DEVNULL)
    print(result.stdout)
    report_file.write(result.stdout + '\n')
    
    # JSON report for analysis
    json_file = REPORT_DIR / 'coverage.json'
    cmd = [
        sys.executable, '-m', 'coverage', 'json',
        f'--data-file={cov_data}',
        '-o', str(json_file)
    ]
    subprocess.run(cmd, capture_output=True, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL)
    
    # HTML report
    cmd = [
        sys.executable, '-m', 'coverage', 'html',
        f'--data-file={cov_data}',
        '-d', str(HTML_DIR)
    ]
    subprocess.run(cmd, capture_output=True, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL)
    
    # Analyze JSON data
    if json_file.exists():
        analyze_coverage(json_file, report_file)
    
    return json_file


def analyze_coverage(json_file: Path, report_file):
    """Analyze coverage data and print statistics."""
    with open(json_file) as f:
        data = json.load(f)
    
    totals = data['totals']
    
    # Module-level summary
    print_header("Coverage by Module", report_file)
    
    modules = {}
    for filepath, filedata in data['files'].items():
        parts = filepath.replace('pythoc/', '').split('/')
        module = parts[0] if len(parts) > 1 else 'root'
        if module not in modules:
            modules[module] = {'stmts': 0, 'covered': 0}
        modules[module]['stmts'] += filedata['summary']['num_statements']
        modules[module]['covered'] += filedata['summary']['covered_lines']
    
    output = []
    for mod in sorted(modules.keys(), key=lambda m: -modules[m]['covered']/max(modules[m]['stmts'],1)):
        m = modules[mod]
        pct = (m['covered'] / m['stmts'] * 100) if m['stmts'] > 0 else 0
        bar_len = int(pct / 5)
        bar = f"[{'#' * bar_len}{'-' * (20 - bar_len)}]"
        line = f"  {pct:5.1f}% {bar} {mod:25s} ({m['covered']:4d}/{m['stmts']:4d})"
        output.append(line)
    
    print('\n'.join(output))
    report_file.write('\n'.join(output) + '\n')
    
    # Files with 0% coverage
    print_header("Files with 0% Coverage", report_file)
    zero_files = [f for f, d in data['files'].items() if d['summary']['percent_covered'] == 0]
    output = []
    for f in sorted(zero_files):
        output.append(f"  {f}")
    output.append(f"\nTotal: {len(zero_files)} files")
    print('\n'.join(output))
    report_file.write('\n'.join(output) + '\n')
    
    # Lowest coverage files
    print_header("Lowest Coverage Files (>0%)", report_file)
    low_cov = [(f, d['summary']['percent_covered'], d['summary']['num_statements']) 
               for f, d in data['files'].items() if d['summary']['percent_covered'] > 0]
    low_cov.sort(key=lambda x: x[1])
    output = []
    for f, pct, stmts in low_cov[:10]:
        output.append(f"  {pct:5.1f}%  ({stmts:4d} stmts)  {f}")
    print('\n'.join(output))
    report_file.write('\n'.join(output) + '\n')
    
    # Most executed files
    print_header("Most Executed Files (by covered lines)", report_file)
    exec_counts = [(f, len(d.get('executed_lines', [])), d['summary']['num_statements']) 
                   for f, d in data['files'].items()]
    exec_counts.sort(key=lambda x: -x[1])
    output = []
    for f, exec_lines, total_lines in exec_counts[:10]:
        output.append(f"  {exec_lines:5d} lines executed  ({total_lines:4d} total)  {f}")
    print('\n'.join(output))
    report_file.write('\n'.join(output) + '\n')
    
    return totals['percent_covered']


def main():
    # Setup report directory
    REPORT_DIR.mkdir(exist_ok=True)
    COVERAGE_DIR.mkdir(exist_ok=True)
    
    # Clean old coverage data
    for f in COVERAGE_DIR.glob('.coverage.*'):
        f.unlink()
    for f in REPORT_DIR.glob('.coverage*'):
        f.unlink()
    
    # Open report file
    report_path = REPORT_DIR / 'coverage_report.txt'
    report_file = open(report_path, 'w')
    
    print_header("Coverage Test Runner", report_file)
    
    # Collect test files
    integration_dir = PROJECT_ROOT / 'test' / 'integration'
    example_dir = PROJECT_ROOT / 'test' / 'example'
    
    integration_tests = sorted(integration_dir.glob('test_*.py'))
    example_tests = [f for f in sorted(example_dir.glob('*.py')) 
                     if f.name != '__init__.py'] if example_dir.exists() else []
    
    all_tests = integration_tests + example_tests
    
    print(f"Found {len(integration_tests)} integration tests")
    print(f"Found {len(example_tests)} example tests")
    print(f"Running tests with coverage (isolated subprocesses)...\n")
    
    report_file.write(f"Integration tests: {len(integration_tests)}\n")
    report_file.write(f"Example tests: {len(example_tests)}\n\n")
    
    passed = 0
    failed = 0
    results = []
    
    # Run tests in parallel with isolation
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(run_test_with_coverage, test_file, i): test_file.stem
            for i, test_file in enumerate(all_tests)
        }
        
        for future in as_completed(futures):
            name, success, duration, error = future.result()
            results.append((name, success, duration, error))
            
            if success:
                print(f"  {GREEN}PASS{RESET} {name} ({duration:.2f}s)")
                passed += 1
            else:
                print(f"  {RED}FAIL{RESET} {name} ({duration:.2f}s)")
                failed += 1
    
    total_time = time.time() - start_time
    
    # Sort results by name for report
    results.sort(key=lambda x: x[0])
    
    # Write test results to report
    print_header("Test Results", report_file)
    for name, success, duration, error in results:
        status = "PASS" if success else "FAIL"
        report_file.write(f"{status} {name} ({duration:.2f}s)\n")
        if not success and error:
            report_file.write(f"  Error: {error[:200]}\n")
    
    # Print summary
    print_header("Test Summary", report_file)
    summary = f"""Total: {passed + failed}
Passed: {passed}
Failed: {failed}
Time: {total_time:.2f}s
"""
    print(summary)
    report_file.write(summary)
    
    # Show failed tests
    failed_tests = [(n, e) for n, s, d, e in results if not s]
    if failed_tests:
        print(f"\n{RED}Failed Tests:{RESET}")
        report_file.write("\nFailed Tests:\n")
        for name, error in failed_tests:
            print(f"  {name}")
            report_file.write(f"  {name}\n")
            if error:
                short_error = error[:150].replace('\n', ' ')
                print(f"    {short_error}")
                report_file.write(f"    {error[:200]}\n")
    
    # Merge and generate coverage reports
    print(f"\n{BLUE}Merging coverage data...{RESET}")
    if merge_coverage_data():
        coverage_pct = 0
        try:
            json_file = generate_reports(report_file)
            if json_file.exists():
                with open(json_file) as f:
                    coverage_pct = json.load(f)['totals']['percent_covered']
        except Exception as e:
            print(f"{RED}Error generating reports: {e}{RESET}")
    else:
        print(f"{RED}Failed to merge coverage data{RESET}")
        coverage_pct = 0
    
    # Final summary
    print_header("Final Results", report_file)
    final = f"""Tests: {passed}/{passed+failed} passed
Coverage: {coverage_pct:.1f}%

Reports:
  Text:  {report_path}
  HTML:  {HTML_DIR}/index.html
  JSON:  {REPORT_DIR}/coverage.json
"""
    print(final)
    report_file.write(final)
    
    report_file.close()
    
    # Cleanup coverage data files
    shutil.rmtree(COVERAGE_DIR, ignore_errors=True)
    
    if failed > 0:
        print(f"\n{RED}{BOLD}Some tests failed!{RESET}")
        return 1
    else:
        print(f"\n{GREEN}{BOLD}All tests passed!{RESET}")
        return 0


if __name__ == '__main__':
    sys.exit(main())
