"""
Layer Normalization — CUDA Emitter Test
========================================
Tests the CUDA emitter with reduction operations (mean, variance),
sqrt, and elementwise normalization.
"""

import torch
import triton
import triton.language as tl
from test_utils import compare_results, benchmark_comparison, run_test, DEVICE


@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd,
                          stride, N,
                          eps,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Store mean and rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def layer_norm(x, normalized_shape, weight, bias, eps=1e-5, **kwargs):
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _layer_norm_fwd_fused[(M,)](
        x_arg, y, weight, bias, mean, rstd,
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        **kwargs,
    )
    return y, mean, rstd


def test_correctness():
    torch.manual_seed(0)
    M, N = 512, 1024
    x = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    w = torch.randn(N, device=DEVICE, dtype=torch.float32)
    b = torch.randn(N, device=DEVICE, dtype=torch.float32)
    eps = 1e-5

    # Default Triton path
    y_triton, mean_triton, rstd_triton = layer_norm(x, (N,), w, b, eps)

    # CUDA emitter path
    y_cuda, mean_cuda, rstd_cuda = layer_norm(x, (N,), w, b, eps, emit_cuda=True)

    # Compare outputs
    compare_results("layer_norm_output", y_triton, y_cuda, atol=1e-5, rtol=1e-5)
    compare_results("layer_norm_mean", mean_triton, mean_cuda, atol=1e-5, rtol=1e-5)
    compare_results("layer_norm_rstd", rstd_triton, rstd_cuda, atol=1e-5, rtol=1e-5)

    # Compare with PyTorch
    y_torch = torch.nn.functional.layer_norm(x, (N,), w, b, eps)
    compare_results("layer_norm_vs_torch", y_triton, y_torch, atol=1e-4, rtol=1e-4)


def test_performance():
    M, N = 4096, 2048
    x = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    w = torch.randn(N, device=DEVICE, dtype=torch.float32)
    b = torch.randn(N, device=DEVICE, dtype=torch.float32)

    benchmark_comparison(
        "layer_norm",
        lambda: layer_norm(x, (N,), w, b),
        lambda: layer_norm(x, (N,), w, b, emit_cuda=True),
    )


if __name__ == "__main__":
    run_test(test_correctness, "05-layer-norm correctness")
    run_test(test_performance, "05-layer-norm performance")
