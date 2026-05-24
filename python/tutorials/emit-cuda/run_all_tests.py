#!/usr/bin/env python3
"""
Run all CUDA emitter tutorial tests.

Usage:
    python run_all_tests.py           # Run all tests
    python run_all_tests.py 01 03     # Run only tutorials 01 and 03
"""

import sys
import importlib
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

TUTORIALS = {
    '01': '01-vector-add',
    '02': '02-fused-softmax',
    '03': '03-matrix-multiplication',
    '04': '04-low-memory-dropout',
    '05': '05-layer-norm',
    '06': '06-fused-attention',
    '07': '07-extern-functions',
    '08': '08-grouped-gemm',
    '09': '09-persistent-matmul',
    '10': '10-block-scaled-matmul',
    '11': '11-programmatic-dependent-launch',
}


def run_tutorial(num: str, name: str):
    print(f"\n{'#'*70}")
    print(f"# Tutorial {num}: {name}")
    print(f"{'#'*70}")
    module_name = name.replace('-', '_')
    try:
        mod = importlib.import_module(module_name)
        # Look for test functions
        for attr_name in sorted(dir(mod)):
            if attr_name.startswith('test_'):
                fn = getattr(mod, attr_name)
                if callable(fn):
                    print(f"\n--- {attr_name} ---")
                    try:
                        fn()
                    except Exception as e:
                        print(f"[ERROR] {attr_name}: {e}")
                        import traceback
                        traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] Failed to import {name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Parse which tutorials to run
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
    else:
        selected = list(TUTORIALS.keys())

    print("=" * 70)
    print("CUDA Emitter Tutorial Tests")
    print("=" * 70)

    for num in selected:
        if num in TUTORIALS:
            run_tutorial(num, TUTORIALS[num])
        else:
            print(f"Unknown tutorial: {num}")

    print(f"\n{'='*70}")
    print("All tests completed.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
