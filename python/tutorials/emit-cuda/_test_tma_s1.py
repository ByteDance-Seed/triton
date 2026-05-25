import sys, os, subprocess, tempfile, shutil
sys.path.insert(0, '/data00/zheng.size/share/triton-cuda/third_party/nvidia/backend')
import torch, triton, triton.language as tl
DEVICE = triton.runtime.driver.active.get_active_torch_device()
from triton.backends.nvidia import compiler as _nv
_Orig = _nv.CUDABackend; _orig_as = _Orig.add_stages; _emit_cuda_active = False
def _patched_as(self, stages, options, language):
    global _emit_cuda_active; _orig_as(self, stages, options, language)
    ec = _emit_cuda_active; _emit_cuda_active = False
    if not ec: return
    stages.pop("llir",None); stages.pop("ptx",None); stages.pop("cubin",None)
    cap = self._parse_arch(options.arch)
    def mk_cuda(src, md):
        from cuda_emitter import CUDAEmitter
        e = CUDAEmitter(capability=cap, num_warps=options.num_warps, num_ctas=options.num_ctas)
        cs = e.emit(src); md["name"]=e.kernel_name; md["shared"]=e.shared_mem_size
        for k in ("tmem_size","global_scratch_size","global_scratch_align","profile_scratch_size","profile_scratch_align","maxntid"):
            md.setdefault(k,0)
        with open('/data00/zheng.size/share/triton-cuda/python/tutorials/emit-cuda/tma_matmul_generated.cu','w') as f: f.write(cs)
        return cs
    def mk_cubin(src, md):
        arch = _nv.sm_arch_from_capability(cap)
        with tempfile.NamedTemporaryFile(delete=False,mode='w',suffix='.cu') as f:
            f.write(src); f.flush(); cu=f.name
        cb=cu+'.cubin'
        r=subprocess.run(['nvcc','-cubin',f'--gpu-architecture={arch}','-O3','--use_fast_math','-std=c++17',cu,'-o',cb],capture_output=True,text=True)
        if r.returncode!=0: raise RuntimeError(f"nvcc:\n{r.stderr}")
        with open(cb,'rb') as f: cubin=f.read()
        os.remove(cu); os.remove(cb); return cubin
    stages["cuda"]=mk_cuda; stages["cubin"]=mk_cubin
_Orig.add_stages = _patched_as
_orig_po = _Orig.parse_options
def _patched_po(self, opts):
    global _emit_cuda_active
    if opts.pop('emit_cuda',False): _emit_cuda_active=True
    return _orig_po(self,opts)
_Orig.parse_options = _patched_po
import triton.compiler.compiler as _cc
_orig_p = _cc.parse
def _pp(fn,ext,ctx):
    if ext=="cuda": from pathlib import Path; return Path(fn).read_text()
    return _orig_p(fn,ext,ctx)
_cc.parse = _pp
shutil.rmtree(os.path.expanduser('~/.triton/cache'), ignore_errors=True)

@triton.jit
def matmul_kernel_tma(a_desc, b_desc, c_desc, M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([offs_am, k * BLOCK_SIZE_K])
        b = b_desc.load([k * BLOCK_SIZE_K, offs_bn])
        accumulator = tl.dot(a, b, accumulator)
    c = accumulator.to(tl.float16)
    c_desc.store([offs_am, offs_bn], c)

from triton.tools.tensor_descriptor import TensorDescriptor
M,N,K = 512,512,512
a = torch.randn((M,K),device=DEVICE,dtype=torch.float16)
b = torch.randn((K,N),device=DEVICE,dtype=torch.float16)
c_cuda = torch.empty((M,N),device=DEVICE,dtype=torch.float16)
a_desc = TensorDescriptor(a, [M,K], [K,1], [128,32])
b_desc = TensorDescriptor(b, [K,N], [N,1], [32,128])
c_desc = TensorDescriptor(c_cuda, [M,N], [N,1], [128,128])
grid = lambda META: (triton.cdiv(M,META['BLOCK_SIZE_M'])*triton.cdiv(N,META['BLOCK_SIZE_N']),)
try:
    # Use num_stages=1 to eliminate pipeline
    matmul_kernel_tma[grid](a_desc, b_desc, c_desc, M, N, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8,
        num_warps=4, num_stages=1, emit_cuda=True)
    torch.cuda.synchronize()
    print("[OK] kernel launched (num_stages=1)")
    ref = torch.matmul(a.float(), b.float()).half()
    maxd = torch.max(torch.abs(ref.float()-c_cuda.float())).item()
    print(f"vs torch: max_diff={maxd:.4e}")
except Exception as e:
    print(f"[FAIL] {e}")
