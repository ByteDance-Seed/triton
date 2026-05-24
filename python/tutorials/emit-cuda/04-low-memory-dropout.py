"""
Low-Memory Dropout — CUDA Emitter Test
========================================
Tests the CUDA emitter with random number generation (Philox)
and elementwise operations.
"""

import torch
import triton
import triton.language as tl
from test_utils import compare_results, run_test, DEVICE


@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed, **kwargs):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed,
                          BLOCK_SIZE=1024, **kwargs)
    return output


def test_correctness():
    torch.manual_seed(0)
    x = torch.randn(size=(10000,), device=DEVICE, dtype=torch.float32)
    p = 0.5
    seed = 123

    # Default Triton path
    output_triton = seeded_dropout(x, p, seed)

    # CUDA emitter path
    output_cuda = seeded_dropout(x, p, seed, emit_cuda=True)

    # Dropout with same seed should produce same results
    compare_results("dropout_same_seed", output_triton, output_cuda, atol=1e-6, rtol=1e-5)

    # Check dropout statistics
    triton_zeros = (output_triton == 0).sum().item()
    cuda_zeros = (output_cuda == 0).sum().item()
    print(f"  Triton zeros: {triton_zeros}/{x.numel()}, CUDA zeros: {cuda_zeros}/{x.numel()}")


if __name__ == "__main__":
    run_test(test_correctness, "04-low-memory-dropout correctness")
