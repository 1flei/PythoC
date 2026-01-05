#!/usr/bin/env python3
"""
Performance comparison script for PC vs C implementations
Compares binary tree and nsieve benchmarks with proper warmup and averaging

Also includes compile-time benchmarking via integration tests.

Flow:
1. Compile C with gcc -O3
2. Run PC python script to generate .o file (not timed)
3. Link PC .o with cc to create executable (not timed)
4. Benchmark both executables (timed)
5. Run integration tests serially to measure compile speed
"""

import subprocess
import time
import sys
import os
from pathlib import Path


def get_exe_suffix():
    """Get platform-specific executable suffix."""
    return '.exe' if sys.platform == 'win32' else ''


# Configuration
WARMUP_RUNS = 1
BENCHMARK_RUNS = 5

# Test parameters
BINARY_TREE_DEPTH = 20
NSIEVE_SIZE = 15


def run_command(cmd, capture=True, cwd=None):
    """Run a command and return result"""
    result = subprocess.run(
        cmd if isinstance(cmd, list) else cmd,
        shell=not isinstance(cmd, list),
        capture_output=capture,
        text=True,
        cwd=cwd,
        stdin=subprocess.DEVNULL
    )
    return result


def compile_c_program(c_file, output_exe):
    """Compile C program with gcc -O3"""
    print(f"  Compiling C: {c_file.name}...")
    
    if not c_file.exists():
        print(f"    ERROR: C file not found: {c_file}")
        return False
    
    result = run_command(["gcc", "-O3", "-o", str(output_exe), str(c_file), "-lm"])
    if result.returncode != 0:
        print(f"    ERROR (returncode={result.returncode}):")
        print(f"    stdout: {result.stdout}")
        print(f"    stderr: {result.stderr}")
        return False
    print(f"    -> {output_exe.name}")
    return True


def compile_pc_program(pc_file, output_exe):
    """Compile PC program: run Python to generate executable"""
    print(f"  Compiling PC: {pc_file.name}...")
    
    workspace = Path(__file__).parent.parent  # Go up from test/ to workspace root
    
    # Step 1: Run Python script to compile to executable
    env = os.environ.copy()
    env['PYTHONPATH'] = str(workspace)
    env['PC_OPT_LEVEL'] = '3'  # Enable maximum optimization
    
    print(f"    Compiling with PC_OPT_LEVEL=3...")
    result = subprocess.run(
        [sys.executable, str(pc_file)],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        env=env,
        stdin=subprocess.DEVNULL
    )
    
    if result.returncode != 0:
        print(f"    ERROR (returncode={result.returncode}):")
        print(f"    stdout: {result.stdout}")
        print(f"    stderr: {result.stderr}")
        return False
    
    # Check if executable was created
    if not output_exe.exists():
        print(f"    ERROR: Executable not generated at {output_exe}")
        print(f"    stdout: {result.stdout}")
        return False
    
    print(f"    -> {output_exe.name}")
    return True


def run_benchmark(exe_path, args, runs=1):
    """Run executable multiple times and measure time"""
    times = []
    
    for i in range(runs):
        start = time.perf_counter()
        result = subprocess.run(
            [str(exe_path)] + [str(a) for a in args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL
        )
        end = time.perf_counter()
        
        if result.returncode != 0:
            print(f"    ERROR running {exe_path.name}: {result.stderr}")
            return None
        
        elapsed = end - start
        times.append(elapsed)
        print(f"    Run {i+1}: {elapsed:.4f}s")
    
    return times


def benchmark_binary_tree():
    """Benchmark binary tree (C vs PC)"""
    print("\n" + "="*70)
    print("BINARY TREE BENCHMARK")
    print("="*70)
    
    workspace = Path(__file__).parent.parent  # Go up from test/ to workspace root
    example_dir = workspace / "test" / "example"
    build_dir = workspace / "build" / "test" / "example"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    exe_suffix = get_exe_suffix()
    c_file = example_dir / "base_binary_tree_test.c"
    pc_file = example_dir / "pc_binary_tree_test.py"
    c_exe = build_dir / f"binary_tree_c_bench{exe_suffix}"
    pc_exe = build_dir / f"pc_binary_tree_test{exe_suffix}"
    
    print(f"\n[1/2] Compilation (not timed)")
    if not compile_c_program(c_file, c_exe):
        return None
    if not compile_pc_program(pc_file, pc_exe):
        return None
    
    print(f"\n[2/2] Benchmarking (depth={BINARY_TREE_DEPTH})")
    
    # Benchmark C (with warmup)
    print(f"\n  C version:")
    print(f"    Warmup ({WARMUP_RUNS} run)...")
    run_benchmark(c_exe, [BINARY_TREE_DEPTH], WARMUP_RUNS)
    print(f"    Benchmark ({BENCHMARK_RUNS} runs):")
    c_times = run_benchmark(c_exe, [BINARY_TREE_DEPTH], BENCHMARK_RUNS)
    
    # Benchmark PC (with warmup)
    print(f"\n  PC version:")
    print(f"    Warmup ({WARMUP_RUNS} run)...")
    run_benchmark(pc_exe, [BINARY_TREE_DEPTH], WARMUP_RUNS)
    print(f"    Benchmark ({BENCHMARK_RUNS} runs):")
    pc_times = run_benchmark(pc_exe, [BINARY_TREE_DEPTH], BENCHMARK_RUNS)
    
    if c_times is None or pc_times is None:
        return None
    
    c_avg = sum(c_times) / len(c_times)
    pc_avg = sum(pc_times) / len(pc_times)
    ratio = pc_avg / c_avg
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  C:   {c_avg:.4f}s  (min: {min(c_times):.4f}s, max: {max(c_times):.4f}s)")
    print(f"  PC:  {pc_avg:.4f}s  (min: {min(pc_times):.4f}s, max: {max(pc_times):.4f}s)")
    print(f"  PC/C ratio: {ratio:.2f}x")
    print(f"{'='*70}")
    
    return {"name": "binary_tree", "c_avg": c_avg, "pc_avg": pc_avg, "ratio": ratio}


def benchmark_compile_speed():
    """Benchmark compile speed by running integration tests serially"""
    print("\n" + "="*70)
    print("COMPILE SPEED BENCHMARK")
    print("="*70)
    
    workspace = Path(__file__).parent.parent
    run_tests_script = workspace / "test" / "run_integration_tests.py"
    
    # Clean build directory to ensure fresh compilation
    build_dir = workspace / "build"
    if build_dir.exists():
        print(f"\n  Cleaning build directory...")
        import shutil
        # Only clean test-related builds, not everything
        test_build = build_dir / "test"
        if test_build.exists():
            shutil.rmtree(test_build)
    
    print(f"\n  Running integration tests serially (fresh compile)...")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(workspace)
    
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(run_tests_script), '--serial', '--quiet'],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        env=env,
        stdin=subprocess.DEVNULL
    )
    elapsed = time.perf_counter() - start
    
    # Parse output to get test count
    # Strip ANSI color codes for reliable parsing
    import re
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    lines = result.stdout.strip().split('\n')
    total_tests = 0
    passed_tests = 0
    for line in lines:
        clean_line = ansi_escape.sub('', line).strip()
        if clean_line.startswith('Total:'):
            total_tests = int(clean_line.split(':')[1].strip())
        elif clean_line.startswith('Passed:'):
            passed_tests = int(clean_line.split(':')[1].strip())
    
    if result.returncode != 0:
        print(f"    ERROR: Some tests failed")
        print(f"    stdout: {result.stdout[-500:]}")
        print(f"    stderr: {result.stderr[-500:]}")
        return None
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Tests: {passed_tests}/{total_tests}")
    print(f"  Total compile time: {elapsed:.2f}s")
    if total_tests > 0:
        print(f"  Average per test: {elapsed/total_tests:.3f}s")
    else:
        print(f"  Average per test: N/A (no tests found)")
    print(f"{'='*70}")
    
    return {
        "name": "compile_speed",
        "total_tests": total_tests,
        "total_time": elapsed,
        "avg_per_test": elapsed / total_tests if total_tests > 0 else 0
    }


def benchmark_nsieve():
    """Benchmark nsieve (C vs PC)"""
    print("\n" + "="*70)
    print("NSIEVE BENCHMARK")
    print("="*70)
    
    workspace = Path(__file__).parent.parent  # Go up from test/ to workspace root
    example_dir = workspace / "test" / "example"
    build_dir = workspace / "build" / "test" / "example"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    exe_suffix = get_exe_suffix()
    c_file = example_dir / "nsieve.c"
    pc_file = example_dir / "nsieve_pc.py"
    c_exe = build_dir / f"nsieve_c_bench{exe_suffix}"
    pc_exe = build_dir / f"nsieve_pc{exe_suffix}"
    
    print(f"\n[1/2] Compilation (not timed)")
    if not compile_c_program(c_file, c_exe):
        return None
    if not compile_pc_program(pc_file, pc_exe):
        return None
    
    print(f"\n[2/2] Benchmarking (size={NSIEVE_SIZE})")
    
    # Benchmark C (with warmup)
    print(f"\n  C version:")
    print(f"    Warmup ({WARMUP_RUNS} run)...")
    run_benchmark(c_exe, [NSIEVE_SIZE], WARMUP_RUNS)
    print(f"    Benchmark ({BENCHMARK_RUNS} runs):")
    c_times = run_benchmark(c_exe, [NSIEVE_SIZE], BENCHMARK_RUNS)
    
    # Benchmark PC (with warmup)
    print(f"\n  PC version:")
    print(f"    Warmup ({WARMUP_RUNS} run)...")
    run_benchmark(pc_exe, [NSIEVE_SIZE], WARMUP_RUNS)
    print(f"    Benchmark ({BENCHMARK_RUNS} runs):")
    pc_times = run_benchmark(pc_exe, [NSIEVE_SIZE], BENCHMARK_RUNS)
    
    if c_times is None or pc_times is None:
        return None
    
    c_avg = sum(c_times) / len(c_times)
    pc_avg = sum(pc_times) / len(pc_times)
    ratio = pc_avg / c_avg
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  C:   {c_avg:.4f}s  (min: {min(c_times):.4f}s, max: {max(c_times):.4f}s)")
    print(f"  PC:  {pc_avg:.4f}s  (min: {min(pc_times):.4f}s, max: {max(pc_times):.4f}s)")
    print(f"  PC/C ratio: {ratio:.2f}x")
    print(f"{'='*70}")
    
    return {"name": "nsieve", "c_avg": c_avg, "pc_avg": pc_avg, "ratio": ratio}


def main():
    """Run all benchmarks"""
    import argparse
    parser = argparse.ArgumentParser(description='PC vs C performance benchmarks')
    parser.add_argument('--compile-speed', action='store_true',
                        help='Also run compile speed benchmark (for CI)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PC vs C PERFORMANCE COMPARISON")
    print(f"Warmup: {WARMUP_RUNS} run(s), Benchmark: {BENCHMARK_RUNS} runs")
    print("="*70)
    
    results = []
    compile_result = None
    
    # Runtime benchmarks
    result = benchmark_binary_tree()
    if result:
        results.append(result)
    
    result = benchmark_nsieve()
    if result:
        results.append(result)
    
    # Compile speed benchmark (only with --compile-speed flag)
    if args.compile_speed:
        compile_result = benchmark_compile_speed()
    
    if results or compile_result:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Runtime results
        if results:
            print("\nRuntime Performance:")
            for r in results:
                print(f"  {r['name']:15s} | C: {r['c_avg']:.4f}s | PC: {r['pc_avg']:.4f}s | Ratio: {r['ratio']:.2f}x")
            
            avg_ratio = sum(r['ratio'] for r in results) / len(results)
            print(f"\n  Average PC/C ratio: {avg_ratio:.2f}x")
        
        # Compile speed results
        if compile_result:
            print("\nCompile Speed:")
            print(f"  Total: {compile_result['total_time']:.2f}s for {compile_result['total_tests']} tests")
            print(f"  Average: {compile_result['avg_per_test']:.3f}s per test")
        
        print("="*70)
    else:
        print("\nNo benchmarks completed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
