"""
Common utilities for emit-cuda tutorial tests.

Provides functions to compare Triton's default compilation path
with the CUDA emitter path for correctness and performance.
"""

import torch
import triton
import triton.language as tl
import time
import sys

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def compare_results(name: str, triton_output: torch.Tensor, cuda_output: torch.Tensor,
                    atol: float = 0.0, rtol: float = 0.0):
    """Compare two tensor outputs for correctness."""
    if atol == 0.0 and rtol == 0.0:
        # Bitwise comparison
        match = torch.equal(triton_output, cuda_output)
        if match:
            print(f"[PASS] {name}: bitwise match")
        else:
            max_diff = torch.max(torch.abs(triton_output.float() - cuda_output.float()))
            num_diff = torch.sum(triton_output != cuda_output).item()
            total = triton_output.numel()
            print(f"[FAIL] {name}: {num_diff}/{total} elements differ, max_diff={max_diff:.6e}")
    else:
        match = torch.allclose(triton_output, cuda_output, atol=atol, rtol=rtol)
        if match:
            print(f"[PASS] {name}: within tolerance (atol={atol}, rtol={rtol})")
        else:
            max_diff = torch.max(torch.abs(triton_output.float() - cuda_output.float()))
            print(f"[FAIL] {name}: max_diff={max_diff:.6e} exceeds tolerance")
    return match


def benchmark_comparison(name: str, triton_fn, cuda_fn, warmup: int = 25, rep: int = 100):
    """Compare performance of Triton default vs CUDA emitter paths."""
    triton_ms = triton.testing.do_bench(triton_fn, warmup=warmup, rep=rep, return_mode='median')
    cuda_ms = triton.testing.do_bench(cuda_fn, warmup=warmup, rep=rep, return_mode='median')
    ratio = cuda_ms / triton_ms if triton_ms > 0 else float('inf')
    status = "PASS" if ratio <= 1.05 else "WARN"  # Allow 5% tolerance
    print(f"[{status}] {name} perf: triton={triton_ms:.3f}ms, cuda_emit={cuda_ms:.3f}ms, ratio={ratio:.3f}")
    return ratio


def run_test(test_fn, name: str = ""):
    """Run a test function and catch exceptions."""
    name = name or test_fn.__name__
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    try:
        test_fn()
        print(f"[OK] {name} completed")
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        import traceback
        traceback.print_exc()
