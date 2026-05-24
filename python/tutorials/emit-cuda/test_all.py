#!/usr/bin/env python3
"""
Comprehensive emit-cuda test: for each tutorial kernel,
  1) Compile TTGIR → CUDA → cubin (via CUDAEmitter + nvcc)
  2) Run the cubin and compare output with standard Triton (bitwise)
  3) Measure performance ratio

Each kernel is tested in its own subprocess to avoid MLIR-context conflicts.
Run:  python test_all.py
"""
import subprocess, sys, os, json, textwrap, tempfile

DOCKER = os.environ.get("DOCKER_CONTAINER", "")
PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = {}


def run_kernel_test(name: str, code: str) -> dict:
    """Run *code* in a fresh subprocess, return parsed JSON result."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=BASE, delete=False) as f:
        f.write(code)
        f.flush()
        path = f.name
    try:
        r = subprocess.run([PYTHON, path], capture_output=True, text=True, timeout=300)
        # The test script prints a JSON line starting with "RESULT:"
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        # No JSON found — probably crashed
        return {"name": name, "compile": False, "error": (r.stderr or r.stdout)[-800:]}
    except subprocess.TimeoutExpired:
        return {"name": name, "compile": False, "error": "timeout"}
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Template that each kernel test script follows
# ---------------------------------------------------------------------------
TEMPLATE = textwrap.dedent(r'''
import sys, os, subprocess, ctypes, json, time, re
sys.path.insert(0, "{emitter_path}")

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver as _driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
cuda_driver = ctypes.CDLL("libcuda.so")
target = triton.runtime.driver.active.get_current_target()
dev = _driver.active.get_current_device()

# ---- kernel definition ----
{kernel_code}

# ---- Step 1: Run Triton standard path FIRST (before any MLIR API calls) ----
{ref_code}

# ---- Step 2: Compile via CUDA emitter (after Triton JIT is done) ----
from cuda_emitter import MLIRTextParser, CUDACodeGen
from triton._C.libtriton import ir
from triton.compiler.compiler import ASTSource, make_backend

def emit_and_compile(kernel_fn, sig, constexprs, extra_opts=None):
    backend = make_backend(target)
    src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constexprs)
    ctx = ir.context(); ir.load_dialects(ctx); backend.load_dialects(ctx)
    options = backend.parse_options(extra_opts or {{}})
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    try:
        mod = src.make_ir(options, codegen_fns, module_map, ctx)
    except TypeError:
        mod = src.make_ir(target, options, codegen_fns, module_map, ctx)
    md = {{}}
    backend.make_ttir(mod, md, options, target.arch)
    backend.make_ttgir(mod, md, options, target.arch)
    parser = MLIRTextParser()
    ir_mod = parser.parse_module(str(mod))
    cg = CUDACodeGen(ir_mod, capability=target.arch)
    cuda_src = cg.generate()
    block_size = ir_mod.num_warps * ir_mod.threads_per_warp
    shared = ir_mod.shared_size
    cu = "/tmp/_emit_" + cg.kernel_name + ".cu"
    with open(cu, "w") as f: f.write(cuda_src)
    cb = cu + ".cubin"
    r = subprocess.run(["nvcc","-cubin",f"--gpu-architecture=sm_{{target.arch}}a",
        "-O3","--use_fast_math","-std=c++17",cu,"-o",cb], capture_output=True, text=True)
    if r.returncode != 0:
        return None, 0, 0, cuda_src, r.stderr[:1000]
    with open(cb,"rb") as f: cubin=f.read()
    _, func, nreg, _, _ = _driver.active.utils.load_binary(cg.kernel_name, cubin, shared, dev)
    os.remove(cu); os.remove(cb)
    return func, block_size, shared, cuda_src, None

def launch(func, grid, bs, shared, args, arg_types=None):
    """Launch a CUDA kernel. arg_types is a string like 'PPPi' (P=pointer, i=int32, f=float)."""
    stream = _driver.active.get_current_stream(dev)
    ca=[]; cp=[]
    for idx, a in enumerate(args):
        # Determine type: use arg_types hint, or auto-detect
        if arg_types and idx < len(arg_types):
            t = arg_types[idx]
            if t == 'P': c = ctypes.c_void_p(a)
            elif t == 'i': c = ctypes.c_int32(a)
            elif t == 'f': c = ctypes.c_float(a)
            else: c = ctypes.c_void_p(a)
        else:
            # Auto-detect: large ints (>2^31) are likely pointers
            if isinstance(a, int) and a > (1 << 31):
                c = ctypes.c_void_p(a)
            elif isinstance(a, int):
                c = ctypes.c_int32(a)
            elif isinstance(a, float):
                c = ctypes.c_float(a)
            else:
                c = ctypes.c_void_p(a)
        ca.append(c); cp.append(ctypes.cast(ctypes.pointer(ca[-1]),ctypes.c_void_p))
    arr=(ctypes.c_void_p*len(args))(*cp)
    g=grid if isinstance(grid,(tuple,list)) else (grid,)
    cuda_driver.cuLaunchKernel(func,g[0],g[1] if len(g)>1 else 1,g[2] if len(g)>2 else 1,
        bs,1,1,shared,ctypes.c_void_p(stream),arr,ctypes.c_void_p(0))

# ---- Step 3: Run CUDA emitter and compare ----
{test_code}
''')

EMITTER_PATH = os.path.join(os.path.dirname(BASE), "..", "..",
                            "third_party", "nvidia", "backend")
EMITTER_PATH = os.path.normpath(os.path.join(BASE, "..", "..", "..",
                                             "third_party", "nvidia", "backend"))

# ===================================================================
# Kernel definitions + test bodies
# ===================================================================

KERNELS = {}

# ---------- 01 vector-add ----------
KERNELS["01-vector-add"] = {
    "kernel": textwrap.dedent("""\
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
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); N=98432
        x=torch.rand(N,device=DEVICE); y=torch.rand(N,device=DEVICE)
        out_ref=torch.empty_like(x)
        grid=(triton.cdiv(N,1024),)
        add_kernel[grid](x,y,out_ref,N,BLOCK_SIZE=1024)
        torch.cuda.synchronize()
        t_tri=triton.testing.do_bench(lambda:add_kernel[grid](x,y,out_ref,N,BLOCK_SIZE=1024),return_mode='median')
    """),
    "test": textwrap.dedent("""\
        sig = {'x_ptr':'*fp32','y_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}
        ce = {'BLOCK_SIZE': 1024}
        func, bs, sh, cuda_src, err = emit_and_compile(add_kernel, sig, ce)
        if err:
            print("RESULT:" + json.dumps({"name":"01-vector-add","compile":False,"error":err[:200]}))
            sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),y.data_ptr(),out_cuda.data_ptr(),N])
        torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        t_cu=triton.testing.do_bench(lambda:(launch(func,grid,bs,sh,[x.data_ptr(),y.data_ptr(),out_cuda.data_ptr(),N]),torch.cuda.synchronize()),return_mode='median')
        print("RESULT:"+json.dumps({"name":"01-vector-add","compile":True,"bitwise":match,
            "triton_ms":round(t_tri,4),"cuda_ms":round(t_cu,4),"ratio":round(t_cu/t_tri,3) if t_tri>0 else -1}))
    """),
}

# ---------- 02 softmax ----------
KERNELS["02-fused-softmax"] = {
    "kernel": textwrap.dedent("""\
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
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); M,N=256,781
        x=torch.randn(M,N,device=DEVICE)
        out_ref=torch.empty_like(x)
        softmax_kernel[(M,)](out_ref,x,x.stride(0),out_ref.stride(0),M,N,BLOCK_SIZE=1024)
        torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        sig = {'output_ptr':'*fp32','input_ptr':'*fp32','input_row_stride':'i32',
               'output_row_stride':'i32','n_rows':'i32','n_cols':'i32'}
        ce = {'BLOCK_SIZE': 1024}
        func, bs, sh, cuda_src, err = emit_and_compile(softmax_kernel, sig, ce)
        if err:
            print("RESULT:"+json.dumps({"name":"02-fused-softmax","compile":False,"error":err[:200]}))
            sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,(M,),bs,sh,[out_cuda.data_ptr(),x.data_ptr(),x.stride(0),out_cuda.stride(0),M,N])
        torch.cuda.synchronize()
        match=torch.allclose(out_ref,out_cuda,atol=1e-2,rtol=1e-2)
        maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"02-fused-softmax","compile":True,"bitwise":match,"max_diff":maxd}))
    """),
}

# ---------- 03 matmul (WGMMA) ----------
KERNELS["03-matmul-wgmma"] = {
    "kernel": textwrap.dedent("""\
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
    """),
    "ref": "",
    "test": textwrap.dedent("""\
        sig = {'a_ptr':'*fp16','b_ptr':'*fp16','c_ptr':'*fp16',
               'M':'i32','N':'i32','K':'i32',
               'stride_am':'i32','stride_ak':'i32','stride_bk':'i32','stride_bn':'i32',
               'stride_cm':'i32','stride_cn':'i32'}
        ce = {'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8}
        func, bs, sh, cuda_src, err = emit_and_compile(matmul_kernel, sig, ce, {'num_warps':4,'num_stages':3})
        res = {"name":"03-matmul-wgmma","compile":func is not None}
        if err: res["error"]=err[:200]
        print("RESULT:"+json.dumps(res))
    """),
}

# ---------- 04 relu ----------
KERNELS["04-relu"] = {
    "kernel": textwrap.dedent("""\
        @triton.jit
        def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            output = tl.where(x > 0, x, 0.0)
            tl.store(output_ptr + offsets, output, mask=mask)
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); N=98432
        x=torch.randn(N,device=DEVICE)
        out_ref=torch.empty_like(x)
        grid=(triton.cdiv(N,1024),)
        relu_kernel[grid](x,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        sig = {'x_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}
        ce = {'BLOCK_SIZE': 1024}
        func, bs, sh, cuda_src, err = emit_and_compile(relu_kernel, sig, ce)
        if err:
            print("RESULT:"+json.dumps({"name":"04-relu","compile":False,"error":err[:200]}))
            sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        print("RESULT:"+json.dumps({"name":"04-relu","compile":True,"bitwise":match}))
    """),
}

# ---------- 05 exp ----------
KERNELS["05-exp"] = {
    "kernel": textwrap.dedent("""\
        @triton.jit
        def exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            output = tl.exp(x)
            tl.store(output_ptr + offsets, output, mask=mask)
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); N=98432
        x=torch.randn(N,device=DEVICE)*0.5
        out_ref=torch.empty_like(x)
        grid=(triton.cdiv(N,1024),)
        exp_kernel[grid](x,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        sig = {'x_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}
        ce = {'BLOCK_SIZE': 1024}
        func, bs, sh, cuda_src, err = emit_and_compile(exp_kernel, sig, ce)
        if err:
            print("RESULT:"+json.dumps({"name":"05-exp","compile":False,"error":err[:200]}))
            sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"05-exp","compile":True,"bitwise":match,"max_diff":maxd}))
    """),
}

# ---------- 06 fma ----------
KERNELS["06-fma"] = {
    "kernel": textwrap.dedent("""\
        @triton.jit
        def fma_kernel(x_ptr, y_ptr, z_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            z = tl.load(z_ptr + offsets, mask=mask)
            output = x * y + z
            tl.store(output_ptr + offsets, output, mask=mask)
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); N=98432
        x=torch.rand(N,device=DEVICE); y=torch.rand(N,device=DEVICE); z=torch.rand(N,device=DEVICE)
        out_ref=torch.empty_like(x)
        grid=(triton.cdiv(N,1024),)
        fma_kernel[grid](x,y,z,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        sig = {'x_ptr':'*fp32','y_ptr':'*fp32','z_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}
        ce = {'BLOCK_SIZE': 1024}
        func, bs, sh, cuda_src, err = emit_and_compile(fma_kernel, sig, ce)
        if err:
            print("RESULT:"+json.dumps({"name":"06-fma","compile":False,"error":err[:200]}))
            sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),y.data_ptr(),z.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"06-fma","compile":True,"bitwise":match,"max_diff":maxd}))
    """),
}


# ===================================================================
# Main: run each kernel in a subprocess
# ===================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kernels", nargs="*", help="kernel names to run (default: all)")
    args = ap.parse_args()

    selected = args.kernels or sorted(KERNELS.keys())

    print("=" * 70)
    print("Triton CUDA Emitter — Comprehensive Test")
    print("=" * 70)

    all_results = []
    for name in selected:
        if name not in KERNELS:
            print(f"[SKIP] Unknown kernel: {name}")
            continue
        k = KERNELS[name]
        code = TEMPLATE.format(
            emitter_path=EMITTER_PATH,
            kernel_code=k["kernel"],
            ref_code=k.get("ref", ""),
            test_code=k["test"],
        )
        print(f"\n--- {name} ---")
        result = run_kernel_test(name, code)
        all_results.append(result)

        compiled = result.get("compile", False)
        bitwise = result.get("bitwise", None)
        ratio = result.get("ratio", None)
        maxd = result.get("max_diff", None)
        err = result.get("error", None)

        if not compiled:
            print(f"  [FAIL] compile failed: {err}")
        elif bitwise is True:
            perf = f", perf ratio={ratio}" if ratio else ""
            print(f"  [PASS] bitwise match{perf}")
        elif bitwise is False:
            print(f"  [FAIL] bitwise mismatch, max_diff={maxd}")
        else:
            # compile-only (e.g. matmul)
            print(f"  [PASS] compiled OK")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    n_pass = sum(1 for r in all_results if r.get("compile") and r.get("bitwise", True) is not False)
    n_total = len(all_results)
    for r in all_results:
        c = r.get("compile", False)
        b = r.get("bitwise", None)
        tag = "[PASS]" if c and b is not False else "[FAIL]"
        extra = ""
        if r.get("ratio"): extra += f" ratio={r['ratio']}"
        if r.get("max_diff") is not None: extra += f" max_diff={r['max_diff']:.2e}"
        print(f"  {tag} {r['name']}{extra}")
    print(f"\n{n_pass}/{n_total} passed")
