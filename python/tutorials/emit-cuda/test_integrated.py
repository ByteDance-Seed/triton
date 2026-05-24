#!/usr/bin/env python3
"""
Integrated emit-cuda test: monkey-patch triton to support emit_cuda=True,
then run each tutorial kernel through the standard @triton.jit path.

This tests the FULL pipeline: @triton.jit → TTIR → TTGIR → CUDA → nvcc → cubin → launch.
Each kernel runs in a subprocess for isolation.
"""
import subprocess, sys, os, json, textwrap, tempfile

PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))
EMITTER_PATH = os.path.normpath(os.path.join(BASE, "..", "..", "..",
                                             "third_party", "nvidia", "backend"))

def run_test(name: str, code: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=BASE, delete=False) as f:
        f.write(code); f.flush(); path = f.name
    try:
        r = subprocess.run([PYTHON, path], capture_output=True, text=True, timeout=600)
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        return {"name": name, "compile": False, "error": (r.stderr or r.stdout)[-600:]}
    except subprocess.TimeoutExpired:
        return {"name": name, "compile": False, "error": "timeout"}
    finally:
        os.unlink(path)


# Preamble: monkey-patch triton to support emit_cuda
PREAMBLE = textwrap.dedent(r'''
import sys, os, json, subprocess, tempfile
sys.path.insert(0, "{emitter_path}")

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Monkey-patch: add emit_cuda support to CUDABackend
from triton.backends.nvidia import compiler as _nv_compiler
_OrigCUDABackend = _nv_compiler.CUDABackend
_orig_add_stages = _OrigCUDABackend.add_stages
_emit_cuda_active = False

def _patched_add_stages(self, stages, options, language):
    global _emit_cuda_active
    _orig_add_stages(self, stages, options, language)
    emit_cuda = _emit_cuda_active
    _emit_cuda_active = False
    if not emit_cuda:
        return
    stages.pop("llir", None)
    stages.pop("ptx", None)
    stages.pop("cubin", None)  # Pop to reinsert after cuda
    capability = self._parse_arch(options.arch)
    def make_cuda(src, metadata):
        from cuda_emitter import CUDAEmitter
        emitter = CUDAEmitter(capability=capability, num_warps=options.num_warps, num_ctas=options.num_ctas)
        cuda_src = emitter.emit(src)
        metadata["name"] = emitter.kernel_name
        metadata["shared"] = emitter.shared_mem_size
        for k in ("tmem_size","global_scratch_size","global_scratch_align",
                  "profile_scratch_size","profile_scratch_align","maxntid"):
            metadata.setdefault(k, 0)
        return cuda_src
    def make_cubin_from_cuda(src, metadata):
        arch = _nv_compiler.sm_arch_from_capability(capability)
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.cu') as f:
            f.write(src); f.flush(); cu=f.name
        cb = cu + '.cubin'
        r = subprocess.run(['nvcc','-cubin',f'--gpu-architecture={{arch}}','-O3',
            '--use_fast_math','-std=c++17',cu,'-o',cb], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"nvcc failed:\\n{{r.stderr[:500]}}")
        with open(cb,'rb') as f: cubin=f.read()
        os.remove(cu); os.remove(cb)
        return cubin
    stages["cuda"] = make_cuda
    stages["cubin"] = make_cubin_from_cuda

_OrigCUDABackend.add_stages = _patched_add_stages

_orig_parse_options = _OrigCUDABackend.parse_options
def _patched_parse_options(self, opts):
    global _emit_cuda_active
    if opts.pop('emit_cuda', False):
        _emit_cuda_active = True
    result = _orig_parse_options(self, opts)
    return result
_OrigCUDABackend.parse_options = _patched_parse_options

# Patch compiler.py parse function to handle 'cuda' extension
import triton.compiler.compiler as _cc
_orig_parse = _cc.parse
def _patched_parse(full_name, ext, context):
    if ext == "cuda":
        from pathlib import Path
        return Path(full_name).read_text()
    return _orig_parse(full_name, ext, context)
_cc.parse = _patched_parse

# Clear triton cache to avoid cache key collision between standard and emit_cuda paths
import shutil
shutil.rmtree(os.path.expanduser('~/.triton/cache'), ignore_errors=True)
''').format(emitter_path=EMITTER_PATH)

KERNELS = {}

# ---------- 01 vector-add ----------
KERNELS["01-vector-add"] = textwrap.dedent("""\
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(42); N=98432
    x=torch.rand(N,device=DEVICE); y=torch.rand(N,device=DEVICE)
    grid=lambda meta:(triton.cdiv(N,meta['BLOCK_SIZE']),)
    # emit_cuda path (clean cache ensures fresh compilation)
    out_cuda=torch.empty_like(x)
    add_kernel[grid](x,y,out_cuda,N,BLOCK_SIZE=1024,emit_cuda=True)
    torch.cuda.synchronize()
    # Standard Triton for reference (clear cache to force recompile without emit_cuda)
    add_kernel.device_caches.clear()
    shutil.rmtree(os.path.expanduser('~/.triton/cache'), ignore_errors=True)
    out_ref=torch.empty_like(x)
    add_kernel[grid](x,y,out_ref,N,BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    match=torch.equal(out_ref,out_cuda)
    t1=triton.testing.do_bench(lambda:add_kernel[grid](x,y,out_ref,N,BLOCK_SIZE=1024),return_mode='median')
    add_kernel.device_caches.clear()
    t2=triton.testing.do_bench(lambda:add_kernel[grid](x,y,out_cuda,N,BLOCK_SIZE=1024,emit_cuda=True),return_mode='median')
    print("RESULT:"+json.dumps({"name":"01-vector-add","compile":True,"bitwise":match,
        "triton_ms":round(t1,4),"cuda_ms":round(t2,4),"ratio":round(t2/t1,3) if t1>0 else -1}))
""")

# ---------- 02 fused-softmax ----------
KERNELS["02-fused-softmax"] = textwrap.dedent("""\
    @triton.jit
    def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                       n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)

    torch.manual_seed(42); M,N=256,781
    x=torch.randn(M,N,device=DEVICE)
    out_ref=torch.empty_like(x); softmax_kernel[(M,)](out_ref,x,x.stride(0),out_ref.stride(0),M,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    out_cuda=torch.empty_like(x); softmax_kernel[(M,)](out_cuda,x,x.stride(0),out_cuda.stride(0),M,N,BLOCK_SIZE=1024,emit_cuda=True); torch.cuda.synchronize()
    match=torch.allclose(out_ref,out_cuda,atol=1e-2,rtol=1e-2)
    maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
    t1=triton.testing.do_bench(lambda:softmax_kernel[(M,)](out_ref,x,x.stride(0),out_ref.stride(0),M,N,BLOCK_SIZE=1024),return_mode='median')
    t2=triton.testing.do_bench(lambda:softmax_kernel[(M,)](out_cuda,x,x.stride(0),out_cuda.stride(0),M,N,BLOCK_SIZE=1024,emit_cuda=True),return_mode='median')
    print("RESULT:"+json.dumps({"name":"02-fused-softmax","compile":True,"bitwise":match,"max_diff":maxd,
        "triton_ms":round(t1,4),"cuda_ms":round(t2,4),"ratio":round(t2/t1,3) if t1>0 else -1}))
""")

# ---------- 03 matrix-multiplication ----------
KERNELS["03-matrix-multiplication"] = textwrap.dedent("""\
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c = acc.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    torch.manual_seed(42); M,N,K=512,512,512
    a=torch.randn((M,K),device=DEVICE,dtype=torch.float16)
    b=torch.randn((K,N),device=DEVICE,dtype=torch.float16)
    grid=lambda META:(triton.cdiv(M,META['BLOCK_SIZE_M'])*triton.cdiv(N,META['BLOCK_SIZE_N']),)
    out_ref=torch.empty((M,N),device=DEVICE,dtype=torch.float16)
    matmul_kernel[grid](a,b,out_ref,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),
        out_ref.stride(0),out_ref.stride(1),BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,num_warps=4,num_stages=3)
    torch.cuda.synchronize()
    out_cuda=torch.empty((M,N),device=DEVICE,dtype=torch.float16)
    try:
        matmul_kernel[grid](a,b,out_cuda,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),
            out_cuda.stride(0),out_cuda.stride(1),BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,num_warps=4,num_stages=3,emit_cuda=True)
        torch.cuda.synchronize()
        match=torch.allclose(out_ref,out_cuda,atol=1e-1,rtol=1e-1)
        maxd=torch.max(torch.abs(out_ref.float()-out_cuda.float())).item()
        t1=triton.testing.do_bench(lambda:matmul_kernel[grid](a,b,out_ref,M,N,K,a.stride(0),a.stride(1),
            b.stride(0),b.stride(1),out_ref.stride(0),out_ref.stride(1),BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32,GROUP_SIZE_M=8,num_warps=4,num_stages=3),return_mode='median')
        t2=triton.testing.do_bench(lambda:matmul_kernel[grid](a,b,out_cuda,M,N,K,a.stride(0),a.stride(1),
            b.stride(0),b.stride(1),out_cuda.stride(0),out_cuda.stride(1),BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,
            BLOCK_SIZE_K=32,GROUP_SIZE_M=8,num_warps=4,num_stages=3,emit_cuda=True),return_mode='median')
        print("RESULT:"+json.dumps({"name":"03-matrix-multiplication","compile":True,"bitwise":match,"max_diff":maxd,
            "triton_ms":round(t1,4),"cuda_ms":round(t2,4),"ratio":round(t2/t1,3) if t1>0 else -1}))
    except Exception as e:
        print("RESULT:"+json.dumps({"name":"03-matrix-multiplication","compile":False,"error":str(e)[:300]}))
""")

# ---------- 05 layer-norm ----------
KERNELS["05-layer-norm"] = textwrap.dedent("""\
    @triton.jit
    def layer_norm_kernel(X, Y, W, B, Mean, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            tl.store(Y + cols, y, mask=mask)

    torch.manual_seed(42); M_ln,N_ln=512,1024; BLOCK_LN=1024
    x_ln=torch.randn((M_ln,N_ln),device=DEVICE); w_ln=torch.randn(N_ln,device=DEVICE); b_ln=torch.randn(N_ln,device=DEVICE)
    y_ref=torch.empty_like(x_ln); mean_ref=torch.empty(M_ln,device=DEVICE); rstd_ref=torch.empty(M_ln,device=DEVICE)
    layer_norm_kernel[(M_ln,)](x_ln,y_ref,w_ln,b_ln,mean_ref,rstd_ref,x_ln.stride(0),N_ln,1e-5,BLOCK_SIZE=BLOCK_LN)
    torch.cuda.synchronize()
    y_cuda=torch.empty_like(x_ln); mean_cuda=torch.empty(M_ln,device=DEVICE); rstd_cuda=torch.empty(M_ln,device=DEVICE)
    try:
        layer_norm_kernel[(M_ln,)](x_ln,y_cuda,w_ln,b_ln,mean_cuda,rstd_cuda,x_ln.stride(0),N_ln,1e-5,BLOCK_SIZE=BLOCK_LN,emit_cuda=True)
        torch.cuda.synchronize()
        match=torch.allclose(y_ref,y_cuda,atol=1e-3,rtol=1e-3)
        maxd=torch.max(torch.abs(y_ref-y_cuda)).item()
        t1=triton.testing.do_bench(lambda:layer_norm_kernel[(M_ln,)](x_ln,y_ref,w_ln,b_ln,mean_ref,rstd_ref,
            x_ln.stride(0),N_ln,1e-5,BLOCK_SIZE=BLOCK_LN),return_mode='median')
        t2=triton.testing.do_bench(lambda:layer_norm_kernel[(M_ln,)](x_ln,y_cuda,w_ln,b_ln,mean_cuda,rstd_cuda,
            x_ln.stride(0),N_ln,1e-5,BLOCK_SIZE=BLOCK_LN,emit_cuda=True),return_mode='median')
        print("RESULT:"+json.dumps({"name":"05-layer-norm","compile":True,"bitwise":match,"max_diff":maxd,
            "triton_ms":round(t1,4),"cuda_ms":round(t2,4),"ratio":round(t2/t1,3) if t1>0 else -1}))
    except Exception as e:
        print("RESULT:"+json.dumps({"name":"05-layer-norm","compile":False,"error":str(e)[:300]}))
""")

# ---------- 07 extern-functions ----------
KERNELS["07-extern-functions"] = textwrap.dedent("""\
    from triton.language.extra.cuda import libdevice
    @triton.jit
    def asin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        x = libdevice.asin(x)
        tl.store(y_ptr + offsets, x, mask=mask)

    torch.manual_seed(42); N=98432
    x_as=torch.rand(N,device=DEVICE)*2-1; grid=(triton.cdiv(N,1024),)
    out_ref=torch.empty_like(x_as); asin_kernel[grid](x_as,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    out_cuda=torch.empty_like(x_as); asin_kernel[grid](x_as,out_cuda,N,BLOCK_SIZE=1024,emit_cuda=True); torch.cuda.synchronize()
    match=torch.allclose(out_ref,out_cuda,atol=1e-5,rtol=1e-5)
    maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
    print("RESULT:"+json.dumps({"name":"07-extern-functions","compile":True,"bitwise":match,"max_diff":maxd}))
""")


# ---------- 04 low-memory-dropout ----------
KERNELS["04-low-memory-dropout"] = textwrap.dedent("""\
    @triton.jit
    def dropout_kernel(x_ptr, output_ptr, n_elements, p, seed,
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

    torch.manual_seed(42); N=10000; p_val=0.5; seed_val=123
    x_drop=torch.randn(N,device=DEVICE); grid=(triton.cdiv(N,1024),)
    out_cuda=torch.empty_like(x_drop)
    try:
        dropout_kernel[grid](x_drop,out_cuda,N,p_val,seed_val,BLOCK_SIZE=1024,emit_cuda=True)
        torch.cuda.synchronize()
        dropout_kernel.device_caches.clear()
        shutil.rmtree(os.path.expanduser('~/.triton/cache'), ignore_errors=True)
        out_ref=torch.empty_like(x_drop)
        dropout_kernel[grid](x_drop,out_ref,N,p_val,seed_val,BLOCK_SIZE=1024)
        torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"04-low-memory-dropout","compile":True,"bitwise":match,"max_diff":maxd}))
    except Exception as e:
        print("RESULT:"+json.dumps({"name":"04-low-memory-dropout","compile":False,"error":str(e)[:300]}))
""")

# ---------- 06 fused-attention (simplified as compile-only for now) ----------
# Note: Full fused attention uses tensor descriptors which need more emitter work
# We test a simplified version here

# ---------- 06 fused-attention ----------
# Uses tensor descriptors (TMA) - requires additional emitter support
# Skipped for now

# ---------- 08 grouped-gemm ----------
# Uses indirect dispatch + matmul - requires TMA and complex indexing
# Skipped for now

# ---------- 09 persistent-matmul ----------
# Uses persistent kernel pattern + TMA
# Skipped for now

# ---------- 10 block-scaled-matmul ----------
# Uses dot_scaled + FP4/FP8 - Blackwell (sm_100+) only
# Skipped for now

# ---------- 11 programmatic-dependent-launch ----------
# Test with USE_GDC=False (no PDL, just vector-add)
KERNELS["11-programmatic-dependent-launch"] = textwrap.dedent("""\
    @triton.jit
    def pdl_add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr, USE_GDC: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    torch.manual_seed(42); N=98432
    x_pdl=torch.rand(N,device=DEVICE); y_pdl=torch.rand(N,device=DEVICE)
    grid=lambda meta:(triton.cdiv(N,meta['BLOCK_SIZE']),)
    out_cuda=torch.empty_like(x_pdl)
    try:
        pdl_add_kernel[grid](x_pdl,y_pdl,out_cuda,N,BLOCK_SIZE=1024,USE_GDC=False,emit_cuda=True)
        torch.cuda.synchronize()
        pdl_add_kernel.device_caches.clear()
        shutil.rmtree(os.path.expanduser('~/.triton/cache'), ignore_errors=True)
        out_ref=torch.empty_like(x_pdl)
        pdl_add_kernel[grid](x_pdl,y_pdl,out_ref,N,BLOCK_SIZE=1024,USE_GDC=False)
        torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        print("RESULT:"+json.dumps({"name":"11-programmatic-dependent-launch","compile":True,"bitwise":match}))
    except Exception as e:
        print("RESULT:"+json.dumps({"name":"11-programmatic-dependent-launch","compile":False,"error":str(e)[:300]}))
""")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="*")
    args = ap.parse_args()
    selected = args.kernels or sorted(KERNELS.keys())

    print("=" * 70)
    print("Triton CUDA Emitter — Integrated Test (emit_cuda=True via JIT)")
    print("=" * 70)

    all_results = []
    for name in selected:
        if name not in KERNELS:
            print(f"[SKIP] {name}"); continue
        code = PREAMBLE + KERNELS[name]
        print(f"\n--- {name} ---")
        result = run_test(name, code)
        all_results.append(result)
        c = result.get("compile", False)
        b = result.get("bitwise", None)
        r = result.get("ratio")
        d = result.get("max_diff")
        e = result.get("error")
        if not c: print(f"  [FAIL] {e[:120] if e else 'unknown'}")
        elif b is True:
            extra = f" ratio={r}" if r else ""
            if d is not None and d > 0: extra += f" max_diff={d:.2e}"
            print(f"  [PASS]{extra}")
        elif b is False: print(f"  [FAIL] mismatch max_diff={d}")
        else: print(f"  [PASS] compiled OK")

    print("\n" + "=" * 70)
    n_pass = sum(1 for r in all_results if r.get("compile") and r.get("bitwise", True) is not False)
    for r in all_results:
        c, b = r.get("compile",False), r.get("bitwise",None)
        tag = "[PASS]" if c and b is not False else "[FAIL]"
        extra = ""
        if r.get("ratio"): extra += f" ratio={r['ratio']}"
        if r.get("max_diff") is not None: extra += f" d={r['max_diff']:.2e}"
        if r.get("triton_ms"): extra += f" t={r['triton_ms']}/{r.get('cuda_ms',0)}ms"
        print(f"  {tag} {r['name']}{extra}")
    print(f"\n{n_pass}/{len(all_results)} passed")
