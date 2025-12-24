#!/usr/bin/env python3
"""
Profiling tool for pythoc compilation.

Usage:
    # Profile a single test file
    python test/run_profile.py test/integration/test_basic_types.py

    # Profile all integration tests
    python test/run_profile.py --all

    # Generate flame graph (requires flameprof: pip install flameprof)
    python test/run_profile.py --all --flame

    # Generate interactive HTML (requires snakeviz: pip install snakeviz)
    python test/run_profile.py --all --snakeviz

    # Show top N functions by cumulative time
    python test/run_profile.py --all --top 30

Output:
    build/profile/profile.prof     - Raw profile data
    build/profile/profile.svg      - Flame graph (with --flame)
    build/profile/profile_stats.txt - Text summary
"""

import argparse
import cProfile
import pstats
import sys
import os
import io
import subprocess
from pathlib import Path


def get_integration_test_files():
    """Get all integration test files"""
    workspace = Path(__file__).parent.parent
    test_dir = workspace / "test" / "integration"
    return sorted(test_dir.glob("test_*.py"))


def run_single_test(test_file: Path):
    """Run a single test file (import and execute)"""
    workspace = Path(__file__).parent.parent
    sys.path.insert(0, str(workspace))
    
    # Import and run the test
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    module = importlib.util.module_from_spec(spec)
    
    # Suppress output during profiling
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass  # Some tests call sys.exit
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print(f"Error in {test_file.name}: {e}")
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def run_all_tests():
    """Run all integration tests"""
    test_files = get_integration_test_files()
    for test_file in test_files:
        run_single_test(test_file)


def profile_compilation(target: str, output_dir: Path):
    """Profile the compilation process"""
    output_dir.mkdir(parents=True, exist_ok=True)
    prof_file = output_dir / "profile.prof"
    stats_file = output_dir / "profile_stats.txt"
    
    profiler = cProfile.Profile()
    
    print(f"Profiling: {target}")
    print("This may take a while...")
    
    # Clean build to ensure fresh compilation
    workspace = Path(__file__).parent.parent
    build_test = workspace / "build" / "test"
    if build_test.exists():
        import shutil
        shutil.rmtree(build_test)
    
    profiler.enable()
    
    if target == "--all":
        run_all_tests()
    else:
        run_single_test(Path(target))
    
    profiler.disable()
    
    # Save raw profile
    profiler.dump_stats(str(prof_file))
    print(f"\nProfile saved to: {prof_file}")
    
    # Generate text stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(50)
    
    with open(stats_file, 'w') as f:
        f.write(stream.getvalue())
    print(f"Stats saved to: {stats_file}")
    
    return prof_file, stats


def print_top_functions(prof_file: Path, n: int = 20):
    """Print top N functions by cumulative time"""
    print(f"\n{'='*80}")
    print(f"TOP {n} FUNCTIONS BY CUMULATIVE TIME")
    print('='*80)
    
    stream = io.StringIO()
    stats = pstats.Stats(str(prof_file), stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(n)
    print(stream.getvalue())


def print_pythoc_hotspots(prof_file: Path):
    """Print pythoc-specific hotspots"""
    print(f"\n{'='*80}")
    print("PYTHOC MODULE HOTSPOTS")
    print('='*80)
    
    stats = pstats.Stats(str(prof_file))
    
    # Filter for pythoc modules
    pythoc_stats = {}
    for key, value in stats.stats.items():
        filename = key[0]
        if 'pythoc' in filename and 'site-packages' not in filename:
            pythoc_stats[key] = value
    
    if not pythoc_stats:
        print("No pythoc-specific stats found")
        return
    
    # Sort by cumulative time
    sorted_stats = sorted(
        pythoc_stats.items(),
        key=lambda x: x[1][3],  # cumtime
        reverse=True
    )
    
    print(f"\n{'ncalls':>10} {'tottime':>10} {'cumtime':>10}  function")
    print('-' * 80)
    
    for (filename, lineno, func), (cc, nc, tt, ct, callers) in sorted_stats[:30]:
        # Shorten filename
        short_file = filename.split('pythoc/')[-1] if 'pythoc/' in filename else filename
        print(f"{nc:>10} {tt:>10.4f} {ct:>10.4f}  {short_file}:{lineno}({func})")


def generate_flame_graph(prof_file: Path, output_dir: Path):
    """Generate flame graph using flameprof"""
    svg_file = output_dir / "profile_flame.svg"
    
    try:
        import flameprof
        print(f"\nGenerating flame graph...")
        subprocess.run(
            [sys.executable, '-m', 'flameprof', str(prof_file), '-o', str(svg_file)],
            check=True
        )
        print(f"Flame graph saved to: {svg_file}")
        print(f"Open in browser: file://{svg_file.absolute()}")
    except ImportError:
        print("\nflameprof not installed. Install with: pip install flameprof")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate flame graph: {e}")


def launch_snakeviz(prof_file: Path):
    """Launch snakeviz interactive viewer"""
    try:
        import snakeviz
        print(f"\nLaunching snakeviz...")
        subprocess.run([sys.executable, '-m', 'snakeviz', str(prof_file)])
    except ImportError:
        print("\nsnakeviz not installed. Install with: pip install snakeviz")


def print_summary_by_module(prof_file: Path):
    """Print time spent in each pythoc module"""
    print(f"\n{'='*80}")
    print("TIME BY PYTHOC MODULE")
    print('='*80)
    
    stats = pstats.Stats(str(prof_file))
    module_times = {}
    
    for (filename, lineno, func), (cc, nc, tt, ct, callers) in stats.stats.items():
        if 'pythoc' not in filename or 'site-packages' in filename:
            continue
        
        # Extract module path
        if 'pythoc/' in filename:
            module = filename.split('pythoc/')[-1]
            # Get directory/file
            parts = module.split('/')
            if len(parts) > 1:
                module_key = '/'.join(parts[:2])  # e.g., "codegen/expr.py"
            else:
                module_key = parts[0]
        else:
            module_key = filename
        
        if module_key not in module_times:
            module_times[module_key] = {'tottime': 0, 'cumtime': 0, 'calls': 0}
        
        module_times[module_key]['tottime'] += tt
        module_times[module_key]['calls'] += nc
    
    # Sort by tottime
    sorted_modules = sorted(
        module_times.items(),
        key=lambda x: x[1]['tottime'],
        reverse=True
    )
    
    total_time = sum(m['tottime'] for m in module_times.values())
    
    print(f"\n{'Module':<40} {'tottime':>10} {'%':>8} {'calls':>10}")
    print('-' * 70)
    
    for module, times in sorted_modules[:20]:
        pct = (times['tottime'] / total_time * 100) if total_time > 0 else 0
        print(f"{module:<40} {times['tottime']:>10.4f} {pct:>7.1f}% {times['calls']:>10}")
    
    print('-' * 70)
    print(f"{'TOTAL':<40} {total_time:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Profile pythoc compilation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('target', nargs='?', default='--all',
                        help='Test file to profile, or --all for all tests')
    parser.add_argument('--all', action='store_true',
                        help='Profile all integration tests')
    parser.add_argument('--flame', action='store_true',
                        help='Generate flame graph (requires flameprof)')
    parser.add_argument('--snakeviz', action='store_true',
                        help='Launch snakeviz interactive viewer')
    parser.add_argument('--top', type=int, default=20,
                        help='Show top N functions (default: 20)')
    parser.add_argument('--output', '-o', type=str, default='build/profile',
                        help='Output directory for profile data')
    
    args = parser.parse_args()
    
    workspace = Path(__file__).parent.parent
    output_dir = workspace / args.output
    
    # Determine target
    if args.all or args.target == '--all':
        target = '--all'
    else:
        target = args.target
        if not Path(target).exists():
            # Try relative to workspace
            target = str(workspace / target)
            if not Path(target).exists():
                print(f"Error: File not found: {args.target}")
                return 1
    
    # Run profiling
    prof_file, stats = profile_compilation(target, output_dir)
    
    # Print analysis
    print_top_functions(prof_file, args.top)
    print_summary_by_module(prof_file)
    print_pythoc_hotspots(prof_file)
    
    # Generate visualizations
    if args.flame:
        generate_flame_graph(prof_file, output_dir)
    
    if args.snakeviz:
        launch_snakeviz(prof_file)
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print('='*80)
    print(f"  View raw profile:    python -m pstats {prof_file}")
    print(f"  Interactive viewer:  pip install snakeviz && snakeviz {prof_file}")
    print(f"  Flame graph:         pip install flameprof && python test/run_profile.py --flame")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
