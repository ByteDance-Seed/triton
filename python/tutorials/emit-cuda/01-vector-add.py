"""
Vector Addition — CUDA Emitter Test
====================================
Tests that the CUDA emitter produces bitwise-identical results
and comparable performance to the default Triton compilation path.
"""

import torch
import triton
import triton.language as tl
from test_utils import compare_results, benchmark_comparison, run_test, DEVICE


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_triton(x, y, **kwargs):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, **kwargs)
    return output


def test_correctness():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    # Default Triton path
    output_triton = add_triton(x, y)

    # CUDA emitter path
    output_cuda = add_triton(x, y, emit_cuda=True)

    # Compare
    compare_results("vector_add_f32", output_triton, output_cuda)

    # Also test with different dtypes
    for dtype in [torch.float16, torch.bfloat16]:
        x_t = x.to(dtype)
        y_t = y.to(dtype)
        out_triton = add_triton(x_t, y_t)
        out_cuda = add_triton(x_t, y_t, emit_cuda=True)
        compare_results(f"vector_add_{dtype}", out_triton, out_cuda)


def test_performance():
    size = 2**20
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    benchmark_comparison(
        "vector_add",
        lambda: add_triton(x, y),
        lambda: add_triton(x, y, emit_cuda=True),
    )


if __name__ == "__main__":
    run_test(test_correctness, "01-vector-add correctness")
    run_test(test_performance, "01-vector-add performance")
