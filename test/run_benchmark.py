#!/usr/bin/env python3
"""
Performance comparison script for PC vs C implementations
Compares binary tree and nsieve benchmarks with proper warmup and averaging

Flow:
1. Compile C with gcc -O3
2. Run PC python script to generate .o file (not timed)
3. Link PC .o with cc to create executable (not timed)
4. Benchmark both executables (timed)
"""

import subprocess
import time
import sys
import os
from pathlib import Path

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
        cwd=cwd
    )
    return result


def compile_c_program(c_file, output_exe):
    """Compile C program with gcc -O3"""
    print(f"  Compiling C: {c_file.name}...")
    result = run_command(["gcc", "-O3", "-o", str(output_exe), str(c_file), "-lm"])
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        return False
    print(f"    -> {output_exe.name}")
    return True


def compile_pc_program(pc_file, output_exe):
    """Compile PC program: run Python to generate .o, then link with clang"""
    print(f"  Compiling PC: {pc_file.name}...")
    
    workspace = Path(__file__).parent.parent  # Go up from test/ to workspace root
    obj_file = workspace / "build" / pc_file.relative_to(workspace).with_suffix(".o")
    
    # Step 1: Run Python script to compile to .o
    env = os.environ.copy()
    env['PYTHONPATH'] = str(workspace)
    env['PC_OPT_LEVEL'] = '3'  # Enable maximum optimization
    
    print(f"    Step 1: Generating {obj_file.relative_to(workspace)} (PC_OPT_LEVEL=3)...")
    result = subprocess.run(
        ["python", str(pc_file)],
        capture_output=True,
        text=True,
        cwd=str(workspace),
        env=env
    )
    
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr}")
        return False
    
    if not obj_file.exists():
        print(f"    ERROR: Object file not generated at {obj_file}")
        return False
    return True


def run_benchmark(exe_path, args, runs=1):
    """Run executable multiple times and measure time"""
    times = []
    
    for i in range(runs):
        start = time.perf_counter()
        result = subprocess.run(
            [str(exe_path)] + [str(a) for a in args],
            capture_output=True,
            text=True
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
    
    c_file = example_dir / "base_binary_tree_test.c"
    pc_file = example_dir / "pc_binary_tree_test.py"
    c_exe = build_dir / "binary_tree_c_bench"
    pc_exe = build_dir / "pc_binary_tree_test"
    
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


def benchmark_nsieve():
    """Benchmark nsieve (C vs PC)"""
    print("\n" + "="*70)
    print("NSIEVE BENCHMARK")
    print("="*70)
    
    workspace = Path(__file__).parent.parent  # Go up from test/ to workspace root
    example_dir = workspace / "test" / "example"
    build_dir = workspace / "build" / "test" / "example"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    c_file = example_dir / "nsieve.c"
    pc_file = example_dir / "nsieve_pc.py"
    c_exe = build_dir / "nsieve_c_bench"
    pc_exe = build_dir / "nsieve_pc"
    
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
    print("\n" + "="*70)
    print("PC vs C PERFORMANCE COMPARISON")
    print(f"Warmup: {WARMUP_RUNS} run(s), Benchmark: {BENCHMARK_RUNS} runs")
    print("="*70)
    
    results = []
    
    result = benchmark_binary_tree()
    if result:
        results.append(result)
    
    result = benchmark_nsieve()
    if result:
        results.append(result)
    
    if results:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        for r in results:
            print(f"{r['name']:15s} | C: {r['c_avg']:.4f}s | PC: {r['pc_avg']:.4f}s | Ratio: {r['ratio']:.2f}x")
        
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        print(f"\nAverage PC/C ratio: {avg_ratio:.2f}x")
        print("="*70)
    else:
        print("\nNo benchmarks completed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
