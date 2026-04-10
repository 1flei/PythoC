#!/usr/bin/env python
"""Run all unit tests."""

import argparse
import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_verbosity() -> int:
    parser = argparse.ArgumentParser(description="Run unit tests")
    parser.add_argument("--quiet", action="store_true", help="Only print summary and failures")
    parser.add_argument("--verbose", action="store_true", help="Print each test name")
    args = parser.parse_args()

    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose cannot be used together")

    if args.quiet:
        return 0
    if args.verbose:
        return 2
    return 1


if __name__ == '__main__':
    # Discover and run all tests in the unit directory
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'unit')
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=resolve_verbosity())
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
