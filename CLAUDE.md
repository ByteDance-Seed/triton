# Triton-CUDA: CUDA Backend for Triton

## Project Goal

给 Triton 增加一个 CUDA C++ 代码生成后端。所有 Triton 程序能直接编译到可读的 CUDA 代码，
再通过 nvcc 编译到 cubin。在 TTGIR 层保留高级控制流结构，在 NVGPUIR 层翻译 sm90a 特有
操作（WGMMA、TMA、mbarrier 等）。

**新路线**: `TTIR → TTGIR(含NVGPUIR ops) → CUDA C++ → cubin (via nvcc)`
**原路线**: `TTIR → TTGIR(含NVGPUIR ops) → LLVM-IR → PTX → cubin`

## Why TTGIR/NVGPUIR Level

- **TTGIR/NVGPUIR** 是最接近人类手写 CUDA 代码的 IR 层级
- 已完成所有必要的优化（tiling、coalescing、pipelining、WGMMA lowering、TMA lowering 等）
- 保留了 `for` 循环（`scf.for`）、`if` 判断（`scf.if`）等高级控制流结构
- TTGIR 缺少 sm90a 信息时，在 NVGPUIR (`ttng.*`) 层翻译
- 到 LLVM-IR 层会丢失所有结构，不再可读

## Architecture

```
@triton.jit(emit_cuda=True)
         │
    ┌─────────┐
    │  TTIR   │  make_ttir() — 不变
    └────┬────┘
         ▼
    ┌─────────┐
    │  TTGIR  │  make_ttgir() — 不变（包含所有 GPU 优化 pass）
    │+NVGPUIR │  ttng.warp_group_dot, ttng.fence_async_shared, ...
    └────┬────┘
         ▼
         还需要lower一下IR，到合适的语义之后进行emit cuda
    ┌─────────────────────────────────────────────────────┐
    │  CUDAEmitter (cuda_emitter.py, ~2700 行)            │
    │  ├─ MLIRTextParser: 解析 TTGIR+NVGPUIR MLIR text   │
    │  ├─ CUDACodeGen: 生成 CUDA C++ + PTX inline asm    │
    │  └─ LayoutComputer: blocked/mma/slice/shared layout │
    └────────────┬────────────────────────────────────────┘
                 ▼
    ┌─────────┐
    │  .cu    │  可读的 CUDA C++ 源码
    └────┬────┘
         │  nvcc -cubin --gpu-architecture=sm_90a
    ┌─────────┐
    │  cubin  │
    └─────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `third_party/nvidia/backend/compiler.py` | 编译管线 (`add_stages`, `make_cuda`, `make_cubin_from_cuda`) |
| `third_party/nvidia/lib/TritonGPUToCUDA/CUDACodeGen.cpp` | **核心**: C++ CUDA 代码生成器 |
| `python/triton/compiler/compiler.py` | 通用编译框架 (增加 `cuda` 格式支持) |
| `docs/emit-cuda/design.md` | 完整设计文档 (含 sm90a WGMMA/TMA 翻译细节) |
| `docs/emit-cuda/op-mapping.md` | Op 翻译对照表 (TTGIR + NVGPUIR → CUDA) |
| `python/tutorials/emit-cuda/` | 测试用例 |

## Usage

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

# emit_cuda=True 走 CUDA 路线
add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024, emit_cuda=True)
```

## Build & Test

```bash
# 在 Docker 容器 zhengsize-vibecuda 中运行
# (代码目录 mount 在容器内同一路径)
docker exec -w ~/share/triton-cuda zhengsize-vibecuda /bin/bash
cd ~/share/triton-cuda
pip3 install -e .
rm -rf ~/.triton/cache # 测试时候每次清理cache
TRITON_EMIT_CUDA=1 python3 python/tutorials/xxx.py
```

## Development

- Branch: `zsz/triton-cuda`
- Base: `dist`
- Working directory: `~/share/triton-cuda`
- Docker: `zhengsize-vibecuda` (H800, CUDA 12.2, triton 3.7.0)

注意事项：
1. 清理debug时候的临时文件
2. 清理测试性能输出的临时文件
3. 不要信任临时写的test和测试性能的脚本，始终使用 `python/tutorials/` 下的官方 tutorial 来验证正确性和性能
4. `asm volatile` 不会影响性能，不作为性能怀疑的原因
5. nvcc 编译 CUDA 和 Triton 直接生成 PTX 理论上性能应该一样，如果达不到是生成的 CUDA 代码质量问题，不是 nvcc 的问题
6. H800 性能常识：高度优化的 bf16 GEMM 性能应该在 800 TFLOPS，高度优化的 fp8 GEMM 性能应该在 1200 TFLOPS
7. 测试性能必须用空闲 GPU，避免其他任务干扰导致性能数据不准确

