#!/usr/bin/env python3
"""
Comprehensive emit-cuda test: for each tutorial kernel,
  1) Compile TTGIR → CUDA → cubin (via CUDAEmitter + nvcc)
  2) Run the cubin and compare output with standard Triton (bitwise)
  3) Measure performance ratio

Each kernel is tested in its own subprocess to avoid MLIR-context conflicts.
Run:  python test_all.py           # all tests
      python test_all.py 01 03     # specific tests
"""
import subprocess, sys, os, json, textwrap, tempfile

PYTHON = sys.executable
BASE = os.path.dirname(os.path.abspath(__file__))
EMITTER_PATH = os.path.normpath(os.path.join(BASE, "..", "..", "..",
                                             "third_party", "nvidia", "backend"))

def run_kernel_test(name: str, code: str) -> dict:
    """Run *code* in a fresh subprocess, return parsed JSON result."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=BASE, delete=False) as f:
        f.write(code)
        f.flush()
        path = f.name
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

# ---------------------------------------------------------------------------
# Template: ref runs FIRST (Triton JIT), then CUDA emitter compiles + runs
# ---------------------------------------------------------------------------
TEMPLATE = textwrap.dedent(r'''
import sys, os, subprocess, ctypes, json, time
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

# ---- Step 1: Triton standard path (before MLIR API calls) ----
{ref_code}

# ---- Step 2: Compile via CUDA emitter ----
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
    try: mod = src.make_ir(options, codegen_fns, module_map, ctx)
    except TypeError: mod = src.make_ir(target, options, codegen_fns, module_map, ctx)
    md = {{}}
    backend.make_ttir(mod, md, options, target.arch)
    backend.make_ttgir(mod, md, options, target.arch)
    parser = MLIRTextParser()
    ir_mod = parser.parse_module(str(mod))
    cg = CUDACodeGen(ir_mod, capability=target.arch)
    cuda_src = cg.generate()
    bs = ir_mod.num_warps * ir_mod.threads_per_warp
    sh = ir_mod.shared_size
    cu = "/tmp/_emit_" + cg.kernel_name + ".cu"
    with open(cu, "w") as f: f.write(cuda_src)
    cb = cu + ".cubin"
    r = subprocess.run(["nvcc","-cubin",f"--gpu-architecture=sm_{{target.arch}}a",
        "-O3","--use_fast_math","-std=c++17",cu,"-o",cb], capture_output=True, text=True)
    if r.returncode != 0:
        return None, 0, 0, cuda_src, r.stderr[:1500]
    with open(cb,"rb") as f: cubin=f.read()
    _, func, nreg, _, _ = _driver.active.utils.load_binary(cg.kernel_name, cubin, sh, dev)
    os.remove(cu); os.remove(cb)
    return func, bs, sh, cuda_src, None

def launch(func, grid, bs, shared, args):
    stream = _driver.active.get_current_stream(dev)
    ca=[]; cp=[]
    for a in args:
        if isinstance(a, int) and a > (1 << 31): c = ctypes.c_void_p(a)
        elif isinstance(a, int): c = ctypes.c_int32(a)
        elif isinstance(a, float): c = ctypes.c_float(a)
        else: c = ctypes.c_void_p(a)
        ca.append(c); cp.append(ctypes.cast(ctypes.pointer(ca[-1]),ctypes.c_void_p))
    arr=(ctypes.c_void_p*len(args))(*cp)
    g=grid if isinstance(grid,(tuple,list)) else (grid,)
    cuda_driver.cuLaunchKernel(func,g[0],g[1] if len(g)>1 else 1,g[2] if len(g)>2 else 1,
        bs,1,1,shared,ctypes.c_void_p(stream),arr,ctypes.c_void_p(0))

# ---- Step 3: Emit + compare ----
{test_code}
''')

# ===================================================================
# Kernel definitions
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
        out_ref=torch.empty_like(x); grid=(triton.cdiv(N,1024),)
        add_kernel[grid](x,y,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
        t_tri=triton.testing.do_bench(lambda:add_kernel[grid](x,y,out_ref,N,BLOCK_SIZE=1024),return_mode='median')
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(add_kernel,
            {'x_ptr':'*fp32','y_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}, {'BLOCK_SIZE':1024})
        if err: print("RESULT:"+json.dumps({"name":"01-vector-add","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),y.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        t_cu=triton.testing.do_bench(lambda:(launch(func,grid,bs,sh,[x.data_ptr(),y.data_ptr(),out_cuda.data_ptr(),N]),torch.cuda.synchronize()),return_mode='median')
        print("RESULT:"+json.dumps({"name":"01-vector-add","compile":True,"bitwise":match,
            "triton_ms":round(t_tri,4),"cuda_ms":round(t_cu,4),"ratio":round(t_cu/t_tri,3) if t_tri>0 else -1}))
    """),
}

# ---------- 02 fused-softmax ----------
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
        x=torch.randn(M,N,device=DEVICE); out_ref=torch.empty_like(x)
        softmax_kernel[(M,)](out_ref,x,x.stride(0),out_ref.stride(0),M,N,BLOCK_SIZE=1024)
        torch.cuda.synchronize()
        t_tri=triton.testing.do_bench(lambda:softmax_kernel[(M,)](out_ref,x,x.stride(0),out_ref.stride(0),M,N,BLOCK_SIZE=1024),return_mode='median')
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(softmax_kernel,
            {'output_ptr':'*fp32','input_ptr':'*fp32','input_row_stride':'i32',
             'output_row_stride':'i32','n_rows':'i32','n_cols':'i32'}, {'BLOCK_SIZE':1024})
        if err: print("RESULT:"+json.dumps({"name":"02-fused-softmax","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,(M,),bs,sh,[out_cuda.data_ptr(),x.data_ptr(),x.stride(0),out_cuda.stride(0),M,N])
        torch.cuda.synchronize()
        match=torch.allclose(out_ref,out_cuda,atol=1e-2,rtol=1e-2)
        maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        t_cu=triton.testing.do_bench(lambda:(launch(func,(M,),bs,sh,[out_cuda.data_ptr(),x.data_ptr(),x.stride(0),out_cuda.stride(0),M,N]),torch.cuda.synchronize()),return_mode='median')
        print("RESULT:"+json.dumps({"name":"02-fused-softmax","compile":True,"bitwise":match,"max_diff":maxd,
            "triton_ms":round(t_tri,4),"cuda_ms":round(t_cu,4),"ratio":round(t_cu/t_tri,3) if t_tri>0 else -1}))
    """),
}

# ---------- 03 matrix-multiplication ----------
KERNELS["03-matrix-multiplication"] = {
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
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); M,N,K=512,512,512
        a=torch.randn((M,K),device=DEVICE,dtype=torch.float16)
        b=torch.randn((K,N),device=DEVICE,dtype=torch.float16)
        out_ref=torch.empty((M,N),device=DEVICE,dtype=torch.float16)
        grid_mm=lambda META:(triton.cdiv(M,META['BLOCK_SIZE_M'])*triton.cdiv(N,META['BLOCK_SIZE_N']),)
        matmul_kernel[grid_mm](a,b,out_ref,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),
            out_ref.stride(0),out_ref.stride(1),BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,BLOCK_SIZE_K=32,GROUP_SIZE_M=8,
            num_warps=4,num_stages=3)
        torch.cuda.synchronize()
        t_tri=triton.testing.do_bench(lambda:matmul_kernel[grid_mm](a,b,out_ref,M,N,K,a.stride(0),a.stride(1),
            b.stride(0),b.stride(1),out_ref.stride(0),out_ref.stride(1),
            BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,BLOCK_SIZE_K=32,GROUP_SIZE_M=8,num_warps=4,num_stages=3),return_mode='median')
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(matmul_kernel,
            {'a_ptr':'*fp16','b_ptr':'*fp16','c_ptr':'*fp16','M':'i32','N':'i32','K':'i32',
             'stride_am':'i32','stride_ak':'i32','stride_bk':'i32','stride_bn':'i32',
             'stride_cm':'i32','stride_cn':'i32'},
            {'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},
            {'num_warps':4,'num_stages':3})
        if err: print("RESULT:"+json.dumps({"name":"03-matrix-multiplication","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty((M,N),device=DEVICE,dtype=torch.float16)
        n_grid=triton.cdiv(M,128)*triton.cdiv(N,128)
        launch(func,(n_grid,),bs,sh,[a.data_ptr(),b.data_ptr(),out_cuda.data_ptr(),M,N,K,
            a.stride(0),a.stride(1),b.stride(0),b.stride(1),out_cuda.stride(0),out_cuda.stride(1)])
        torch.cuda.synchronize()
        match=torch.allclose(out_ref,out_cuda,atol=1e-2,rtol=1e-2)
        maxd=torch.max(torch.abs(out_ref.float()-out_cuda.float())).item()
        t_cu=triton.testing.do_bench(lambda:(launch(func,(n_grid,),bs,sh,[a.data_ptr(),b.data_ptr(),out_cuda.data_ptr(),
            M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),out_cuda.stride(0),out_cuda.stride(1)]),
            torch.cuda.synchronize()),return_mode='median')
        print("RESULT:"+json.dumps({"name":"03-matrix-multiplication","compile":True,"bitwise":match,"max_diff":maxd,
            "triton_ms":round(t_tri,4),"cuda_ms":round(t_cu,4),"ratio":round(t_cu/t_tri,3) if t_tri>0 else -1}))
    """),
}

# ---------- 04 low-memory-dropout ----------
KERNELS["04-low-memory-dropout"] = {
    "kernel": textwrap.dedent("""\
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
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); N=10000; p=0.5; seed=123
        x=torch.randn(N,device=DEVICE); out_ref=torch.empty_like(x)
        grid=(triton.cdiv(N,1024),)
        dropout_kernel[grid](x,out_ref,N,p,seed,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(dropout_kernel,
            {'x_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32','p':'fp32','seed':'i32'},
            {'BLOCK_SIZE':1024})
        if err: print("RESULT:"+json.dumps({"name":"04-low-memory-dropout","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),out_cuda.data_ptr(),N,p,seed]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"04-low-memory-dropout","compile":True,"bitwise":match,"max_diff":maxd}))
    """),
}

# ---------- 05 layer-norm ----------
KERNELS["05-layer-norm"] = {
    "kernel": textwrap.dedent("""\
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
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); M,N_ln=512,1024
        x_ln=torch.randn((M,N_ln),device=DEVICE,dtype=torch.float32)
        w_ln=torch.randn(N_ln,device=DEVICE,dtype=torch.float32)
        b_ln=torch.randn(N_ln,device=DEVICE,dtype=torch.float32)
        y_ref=torch.empty_like(x_ln); mean_ref=torch.empty(M,device=DEVICE,dtype=torch.float32)
        rstd_ref=torch.empty(M,device=DEVICE,dtype=torch.float32)
        BLOCK_LN=triton.next_power_of_2(N_ln)
        layer_norm_kernel[(M,)](x_ln,y_ref,w_ln,b_ln,mean_ref,rstd_ref,x_ln.stride(0),N_ln,1e-5,BLOCK_SIZE=BLOCK_LN)
        torch.cuda.synchronize()
        t_tri=triton.testing.do_bench(lambda:layer_norm_kernel[(M,)](x_ln,y_ref,w_ln,b_ln,mean_ref,rstd_ref,
            x_ln.stride(0),N_ln,1e-5,BLOCK_SIZE=BLOCK_LN),return_mode='median')
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(layer_norm_kernel,
            {'X':'*fp32','Y':'*fp32','W':'*fp32','B':'*fp32','Mean':'*fp32','Rstd':'*fp32',
             'stride':'i32','N':'i32','eps':'fp32'}, {'BLOCK_SIZE':BLOCK_LN})
        if err: print("RESULT:"+json.dumps({"name":"05-layer-norm","compile":False,"error":err[:200]})); sys.exit()
        y_cuda=torch.empty_like(x_ln); mean_cuda=torch.empty(M,device=DEVICE,dtype=torch.float32)
        rstd_cuda=torch.empty(M,device=DEVICE,dtype=torch.float32)
        launch(func,(M,),bs,sh,[x_ln.data_ptr(),y_cuda.data_ptr(),w_ln.data_ptr(),b_ln.data_ptr(),
            mean_cuda.data_ptr(),rstd_cuda.data_ptr(),x_ln.stride(0),N_ln,1e-5]); torch.cuda.synchronize()
        match=torch.allclose(y_ref,y_cuda,atol=1e-4,rtol=1e-4)
        maxd=torch.max(torch.abs(y_ref-y_cuda)).item()
        t_cu=triton.testing.do_bench(lambda:(launch(func,(M,),bs,sh,[x_ln.data_ptr(),y_cuda.data_ptr(),
            w_ln.data_ptr(),b_ln.data_ptr(),mean_cuda.data_ptr(),rstd_cuda.data_ptr(),
            x_ln.stride(0),N_ln,1e-5]),torch.cuda.synchronize()),return_mode='median')
        print("RESULT:"+json.dumps({"name":"05-layer-norm","compile":True,"bitwise":match,"max_diff":maxd,
            "triton_ms":round(t_tri,4),"cuda_ms":round(t_cu,4),"ratio":round(t_cu/t_tri,3) if t_tri>0 else -1}))
    """),
}

# ---------- 07 extern-functions ----------
KERNELS["07-extern-functions"] = {
    "kernel": textwrap.dedent("""\
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
    """),
    "ref": textwrap.dedent("""\
        torch.manual_seed(42); N=98432
        x_as=torch.rand(N,device=DEVICE)*2-1; out_ref=torch.empty_like(x_as)
        grid=(triton.cdiv(N,1024),)
        asin_kernel[grid](x_as,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(asin_kernel,
            {'x_ptr':'*fp32','y_ptr':'*fp32','n_elements':'i32'}, {'BLOCK_SIZE':1024})
        if err: print("RESULT:"+json.dumps({"name":"07-extern-functions","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty_like(x_as)
        launch(func,grid,bs,sh,[x_as.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.allclose(out_ref,out_cuda,atol=1e-5,rtol=1e-5)
        maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"07-extern-functions","compile":True,"bitwise":match,"max_diff":maxd}))
    """),
}

# ---------- helper kernels for testing: relu, exp, fma ----------
KERNELS["t-relu"] = {
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
        x=torch.randn(N,device=DEVICE); out_ref=torch.empty_like(x)
        grid=(triton.cdiv(N,1024),)
        relu_kernel[grid](x,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(relu_kernel,
            {'x_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}, {'BLOCK_SIZE':1024})
        if err: print("RESULT:"+json.dumps({"name":"t-relu","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda)
        print("RESULT:"+json.dumps({"name":"t-relu","compile":True,"bitwise":match}))
    """),
}

KERNELS["t-exp"] = {
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
        x=torch.randn(N,device=DEVICE)*0.5; out_ref=torch.empty_like(x)
        grid=(triton.cdiv(N,1024),)
        exp_kernel[grid](x,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(exp_kernel,
            {'x_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}, {'BLOCK_SIZE':1024})
        if err: print("RESULT:"+json.dumps({"name":"t-exp","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda); maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"t-exp","compile":True,"bitwise":match,"max_diff":maxd}))
    """),
}

KERNELS["t-fma"] = {
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
        out_ref=torch.empty_like(x); grid=(triton.cdiv(N,1024),)
        fma_kernel[grid](x,y,z,out_ref,N,BLOCK_SIZE=1024); torch.cuda.synchronize()
    """),
    "test": textwrap.dedent("""\
        func,bs,sh,cuda_src,err = emit_and_compile(fma_kernel,
            {'x_ptr':'*fp32','y_ptr':'*fp32','z_ptr':'*fp32','output_ptr':'*fp32','n_elements':'i32'}, {'BLOCK_SIZE':1024})
        if err: print("RESULT:"+json.dumps({"name":"t-fma","compile":False,"error":err[:200]})); sys.exit()
        out_cuda=torch.empty_like(x)
        launch(func,grid,bs,sh,[x.data_ptr(),y.data_ptr(),z.data_ptr(),out_cuda.data_ptr(),N]); torch.cuda.synchronize()
        match=torch.equal(out_ref,out_cuda); maxd=torch.max(torch.abs(out_ref-out_cuda)).item()
        print("RESULT:"+json.dumps({"name":"t-fma","compile":True,"bitwise":match,"max_diff":maxd}))
    """),
}


# ===================================================================
# Main
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
            print(f"  [FAIL] compile: {err[:120] if err else 'unknown'}")
        elif bitwise is True:
            perf = f", ratio={ratio}" if ratio else ""
            print(f"  [PASS] bitwise match{perf}")
        elif bitwise is False:
            print(f"  [FAIL] mismatch, max_diff={maxd}")
        else:
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
        if r.get("triton_ms"): extra += f" tri={r['triton_ms']}ms cuda={r.get('cuda_ms',0)}ms"
        print(f"  {tag} {r['name']}{extra}")
    print(f"\n{n_pass}/{n_total} passed")
