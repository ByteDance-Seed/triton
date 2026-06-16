import os
import torch
import triton
import triton.language as tl


@triton.jit
def add_mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = x * y + tl.exp(x) - tl.sqrt(tl.abs(y) + 1.0)
    tl.store(out_ptr + offs, z, mask=mask)


def run():
    n = 8192
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    add_mul_kernel[grid](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    ref = x * y + torch.exp(x) - torch.sqrt(torch.abs(y) + 1.0)
    ok = torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    maxerr = (out - ref).abs().max().item()
    print(f"mode={'PLUGIN' if os.environ.get('TRITON_CUDA_PLUGIN')=='1' else 'PYBIND'} "
          f"allclose={ok} maxerr={maxerr:.2e}")
    assert ok, "numeric mismatch"


if __name__ == "__main__":
    run()
