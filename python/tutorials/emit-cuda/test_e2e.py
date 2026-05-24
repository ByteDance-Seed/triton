"""End-to-end test: compile vector-add TTGIR to CUDA, compile to cubin, load and run."""
import sys, os, tempfile, subprocess
import ctypes
import struct

sys.path.insert(0, '/data00/zheng.size/share/triton-cuda/third_party/nvidia/backend')
from cuda_emitter import CUDAEmitter, MLIRTextParser, CUDACodeGen

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Step 1: Compile vector-add using standard Triton
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

# Step 2: Get the TTGIR from Triton compilation
from triton._C.libtriton import ir
from triton.compiler.compiler import ASTSource, make_backend

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
    ttir_mod = src.make_ir(options, codegen_fns, module_map, context)
except TypeError:
    ttir_mod = src.make_ir(target, options, codegen_fns, module_map, context)

capability = target.arch
metadata = {}
backend.make_ttir(ttir_mod, metadata, options, capability)
backend.make_ttgir(ttir_mod, metadata, options, capability)

ttgir_text = str(ttir_mod)
print("=== TTGIR generated ===")

# Step 3: Generate CUDA from TTGIR
parser = MLIRTextParser()
module = parser.parse_module(ttgir_text)
codegen = CUDACodeGen(module, capability=capability)
cuda_src = codegen.generate()
print(f"Kernel: {codegen.kernel_name}")
print(f"CUDA lines: {len(cuda_src.splitlines())}")

# Step 4: Compile with nvcc
cu_path = '/tmp/test_add_kernel.cu'
cubin_path = cu_path + '.cubin'
with open(cu_path, 'w') as f:
    f.write(cuda_src)

cmd = ['nvcc', '-cubin', f'--gpu-architecture=sm_{capability}a', '-O3', '--use_fast_math',
       '-std=c++17', cu_path, '-o', cubin_path]
result = subprocess.run(cmd, capture_output=True, text=True)
assert result.returncode == 0, f"nvcc failed:\n{result.stderr}"
print("nvcc compilation: OK")

# Step 5: Load cubin and run
with open(cubin_path, 'rb') as f:
    cubin = f.read()

# Use CUDA driver API to load and run the kernel
from triton.runtime.driver import driver as _driver
device = _driver.active.get_current_device()
mod, func, n_regs, n_spills, n_max_threads = _driver.active.utils.load_binary(
    codegen.kernel_name, cubin, 0, device)
print(f"Loaded: regs={n_regs}, spills={n_spills}")

# Step 6: Test correctness
torch.manual_seed(42)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)

# Triton standard output
output_triton = torch.empty_like(x)
grid = (triton.cdiv(size, 1024),)
add_kernel[grid](x, y, output_triton, size, BLOCK_SIZE=1024)

# CUDA emitter output
output_cuda = torch.empty_like(x)
stream = _driver.active.get_current_stream(device)

# Launch using CUDA driver API directly
import ctypes
cuda_driver = ctypes.CDLL('libcuda.so')

# Pack kernel args as pointers
x_ptr = ctypes.c_void_p(x.data_ptr())
y_ptr = ctypes.c_void_p(y.data_ptr())
out_ptr = ctypes.c_void_p(output_cuda.data_ptr())
n_elem = ctypes.c_int32(size)

# Create array of kernel arg pointers
args = (ctypes.c_void_p * 4)(
    ctypes.cast(ctypes.pointer(x_ptr), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(y_ptr), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(out_ptr), ctypes.c_void_p),
    ctypes.cast(ctypes.pointer(n_elem), ctypes.c_void_p),
)

err = cuda_driver.cuLaunchKernel(
    func,
    grid[0], 1, 1,  # grid dims
    128, 1, 1,      # block dims
    0,               # shared mem
    ctypes.c_void_p(stream),  # stream
    args,
    ctypes.c_void_p(0),  # extra
)
torch.cuda.synchronize()

# Compare
diff = torch.abs(output_triton - output_cuda)
max_diff = torch.max(diff).item()
n_diff = torch.sum(diff > 0).item()
print(f"\nMax diff: {max_diff}")
print(f"N diff: {n_diff}/{size}")

if torch.equal(output_triton, output_cuda):
    print("\n[PASS] BITWISE MATCH!")
else:
    print(f"\n[FAIL] {n_diff} elements differ")
    # Show first few differences
    idx = torch.where(diff > 0)[0][:5]
    for i in idx:
        i = i.item()
        print(f"  [{i}] triton={output_triton[i].item():.10f} cuda={output_cuda[i].item():.10f}")

# Performance comparison
print("\n=== Performance ===")
import time

def bench_triton():
    add_kernel[grid](x, y, output_triton, size, BLOCK_SIZE=1024)

def bench_cuda():
    err = cuda_driver.cuLaunchKernel(
        func, grid[0], 1, 1, 128, 1, 1, 0,
        ctypes.c_void_p(stream), args, ctypes.c_void_p(0))

# Warmup
for _ in range(10):
    bench_triton()
    bench_cuda()
torch.cuda.synchronize()

# Benchmark triton
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    bench_triton()
torch.cuda.synchronize()
triton_time = (time.perf_counter() - t0) / 1000

# Benchmark cuda emitter
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    bench_cuda()
torch.cuda.synchronize()
cuda_time = (time.perf_counter() - t0) / 1000

gbps_triton = 3 * size * 4 / triton_time / 1e9
gbps_cuda = 3 * size * 4 / cuda_time / 1e9
print(f"Triton: {triton_time*1e6:.1f}us ({gbps_triton:.1f} GB/s)")
print(f"CUDA emitter: {cuda_time*1e6:.1f}us ({gbps_cuda:.1f} GB/s)")
print(f"Ratio: {cuda_time/triton_time:.3f}x")

# Cleanup
os.remove(cu_path)
os.remove(cubin_path)
