"""
Extern Functions — CUDA Emitter Test
======================================
Tests the CUDA emitter with external function calls (libdevice).
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from test_utils import compare_results, run_test, DEVICE


@triton.jit
def asin_kernel(x_ptr, y_ptr, n_elements,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = libdevice.asin(x)
    tl.store(y_ptr + offsets, x, mask=mask)


def compute_asin(x, **kwargs):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    asin_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024, **kwargs)
    return output


def test_correctness():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE, dtype=torch.float32) * 2 - 1  # [-1, 1]

    # Default Triton path
    output_triton = compute_asin(x)

    # CUDA emitter path
    output_cuda = compute_asin(x, emit_cuda=True)

    # Compare
    compare_results("asin_f32", output_triton, output_cuda, atol=1e-6, rtol=1e-5)

    # Compare with PyTorch
    output_torch = torch.asin(x)
    compare_results("asin_vs_torch", output_triton, output_torch, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_test(test_correctness, "07-extern-functions correctness")
