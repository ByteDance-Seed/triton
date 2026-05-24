"""
Fused Softmax — CUDA Emitter Test
===================================
Tests the CUDA emitter with reduction operations (max, sum) and
elementwise math (exp, div).
"""

import torch
import triton
import triton.language as tl
from test_utils import compare_results, benchmark_comparison, run_test, DEVICE


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x, **kwargs):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_stages = 4 if BLOCK_SIZE <= 2048 else 2
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages,
        **kwargs,
    )
    return y


def test_correctness():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE, dtype=torch.float32)

    # Default Triton path
    output_triton = softmax(x)

    # CUDA emitter path
    output_cuda = softmax(x, emit_cuda=True)

    # Softmax involves floating point reductions, so allow small tolerance
    compare_results("fused_softmax_f32", output_triton, output_cuda, atol=1e-6, rtol=1e-5)

    # Verify against PyTorch
    output_torch = torch.softmax(x, axis=-1)
    compare_results("softmax_vs_torch", output_triton, output_torch, atol=1e-4, rtol=1e-4)


def test_performance():
    x = torch.randn(8192, 4096, device=DEVICE, dtype=torch.float32)

    benchmark_comparison(
        "fused_softmax",
        lambda: softmax(x),
        lambda: softmax(x, emit_cuda=True),
    )


if __name__ == "__main__":
    run_test(test_correctness, "02-fused-softmax correctness")
    run_test(test_performance, "02-fused-softmax performance")
