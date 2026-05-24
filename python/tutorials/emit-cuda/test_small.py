"""Debug test with small sizes."""
import sys, os, subprocess, ctypes
sys.path.insert(0, '/data00/zheng.size/share/triton-cuda/third_party/nvidia/backend')
from cuda_emitter import MLIRTextParser, CUDACodeGen
import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.compiler.compiler import ASTSource, make_backend
from triton.runtime.driver import driver as _driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
cuda_driver = ctypes.CDLL('libcuda.so')

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

# Compile to TTGIR and then CUDA
target = triton.runtime.driver.active.get_current_target()
backend = make_backend(target)
sig = {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'output_ptr': '*fp32', 'n_elements': 'i32'}
constexprs = {'BLOCK_SIZE': 1024}
src = ASTSource(fn=add_kernel, signature=sig, constexprs=constexprs)
context = ir.context()
ir.load_dialects(context)
backend.load_dialects(context)
options = backend.parse_options({})
codegen_fns = backend.get_codegen_implementation(options)
module_map = backend.get_module_map()
try:
    mod = src.make_ir(options, codegen_fns, module_map, context)
except TypeError:
    mod = src.make_ir(target, options, codegen_fns, module_map, context)
metadata = {}
backend.make_ttir(mod, metadata, options, target.arch)
backend.make_ttgir(mod, metadata, options, target.arch)

parser = MLIRTextParser()
ir_module = parser.parse_module(str(mod))
codegen = CUDACodeGen(ir_module, capability=target.arch)
cuda_src = codegen.generate()

cu_path = '/tmp/_test_add.cu'
with open(cu_path, 'w') as f:
    f.write(cuda_src)
cubin_path = cu_path + '.cubin'
result = subprocess.run(['nvcc', '-cubin', f'--gpu-architecture=sm_{target.arch}a', '-O3', '--use_fast_math', '-std=c++17', cu_path, '-o', cubin_path], capture_output=True, text=True)
assert result.returncode == 0, f'nvcc: {result.stderr}'
with open(cubin_path, 'rb') as f:
    cubin = f.read()
device = _driver.active.get_current_device()
mod_h, func, n_regs, _, _ = _driver.active.utils.load_binary(codegen.kernel_name, cubin, 0, device)
print(f'Compiled: {n_regs} regs')

# Test with different sizes
for size in [128, 1024, 2048, 98432]:
    torch.manual_seed(42)
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    # Standard triton
    out_triton = torch.empty_like(x)
    n_blocks = triton.cdiv(size, 1024)
    add_kernel[(n_blocks,)](x, y, out_triton, size, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    # CUDA emitter
    out_cuda = torch.empty_like(x)
    stream = _driver.active.get_current_stream(device)
    x_p = ctypes.c_void_p(x.data_ptr())
    y_p = ctypes.c_void_p(y.data_ptr())
    o_p = ctypes.c_void_p(out_cuda.data_ptr())
    n_p = ctypes.c_int32(size)
    args = (ctypes.c_void_p * 4)(
        ctypes.cast(ctypes.pointer(x_p), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(y_p), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(o_p), ctypes.c_void_p),
        ctypes.cast(ctypes.pointer(n_p), ctypes.c_void_p),
    )
    err = cuda_driver.cuLaunchKernel(func, n_blocks, 1, 1, 128, 1, 1, 0, ctypes.c_void_p(stream), args, ctypes.c_void_p(0))
    torch.cuda.synchronize()

    match = torch.equal(out_triton, out_cuda)
    if match:
        print(f'[PASS] size={size}: BITWISE MATCH')
    else:
        diff = torch.abs(out_triton - out_cuda)
        n_diff = torch.sum(diff > 0).item()
        print(f'[FAIL] size={size}: {n_diff}/{size} differ, max_diff={diff.max().item():.6e}')
        # Check which elements differ
        idx = torch.where(diff > 0)[0][:5]
        for i in idx:
            i = i.item()
            print(f'  [{i}] triton={out_triton[i].item():.10f} cuda={out_cuda[i].item():.10f}')

os.remove(cu_path)
os.remove(cubin_path)
