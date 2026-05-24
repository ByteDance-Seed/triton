"""Test CUDA emission and nvcc compilation for multiple kernels."""
import sys, os, subprocess
sys.path.insert(0, '/data00/zheng.size/share/triton-cuda/third_party/nvidia/backend')
from cuda_emitter import MLIRTextParser, CUDACodeGen

import torch
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.compiler.compiler import ASTSource, make_backend

target = triton.runtime.driver.active.get_current_target()

def test_kernel(name, kernel_fn, sig, constexprs, extra_opts=None):
    """Compile kernel through TTGIR -> CUDA -> nvcc and check if it compiles."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    try:
        backend = make_backend(target)
        src = ASTSource(fn=kernel_fn, signature=sig, constexprs=constexprs)
        context = ir.context()
        ir.load_dialects(context)
        backend.load_dialects(context)
        options = backend.parse_options(extra_opts or {})
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

        cu_path = f'/tmp/{name}.cu'
        with open(cu_path, 'w') as f:
            f.write(cuda_src)
        cubin_path = cu_path + '.cubin'
        result = subprocess.run(
            ['nvcc', '-cubin', f'--gpu-architecture=sm_{target.arch}a', '-O3',
             '--use_fast_math', '-std=c++17', cu_path, '-o', cubin_path],
            capture_output=True, text=True)

        if result.returncode == 0:
            sz = os.path.getsize(cubin_path)
            print(f"[PASS] {name}: nvcc OK ({sz} bytes, {len(cuda_src.splitlines())} CUDA lines)")
            os.remove(cubin_path)
            return True
        else:
            errors = [l for l in result.stderr.split('\n') if 'error' in l.lower()]
            print(f"[FAIL] {name}: nvcc failed ({len(errors)} errors)")
            for e in errors[:3]:
                print(f"  {e}")
            return False
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# Test kernels
# ============================================================

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

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

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
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
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

results = {}

results['01-vector-add'] = test_kernel('01_vector_add', add_kernel,
    {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'output_ptr': '*fp32', 'n_elements': 'i32'},
    {'BLOCK_SIZE': 1024})

results['03-relu'] = test_kernel('03_relu', relu_kernel,
    {'x_ptr': '*fp32', 'output_ptr': '*fp32', 'n_elements': 'i32'},
    {'BLOCK_SIZE': 1024})

results['02-softmax'] = test_kernel('02_softmax', softmax_kernel,
    {'output_ptr': '*fp32', 'input_ptr': '*fp32',
     'input_row_stride': 'i32', 'output_row_stride': 'i32',
     'n_rows': 'i32', 'n_cols': 'i32'},
    {'BLOCK_SIZE': 1024})

results['03-matmul-wgmma'] = test_kernel('03_matmul', matmul_kernel,
    {'a_ptr': '*fp16', 'b_ptr': '*fp16', 'c_ptr': '*fp16',
     'M': 'i32', 'N': 'i32', 'K': 'i32',
     'stride_am': 'i32', 'stride_ak': 'i32',
     'stride_bk': 'i32', 'stride_bn': 'i32',
     'stride_cm': 'i32', 'stride_cn': 'i32'},
    {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
    {'num_warps': 4, 'num_stages': 3})

print(f"\n{'='*50}")
print("Summary")
print(f"{'='*50}")
passed = sum(1 for v in results.values() if v)
total = len(results)
for name, ok in results.items():
    print(f"  {'[PASS]' if ok else '[FAIL]'} {name}")
print(f"\n{passed}/{total} passed")
