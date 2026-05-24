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
| `third_party/nvidia/backend/cuda_emitter.py` | **核心**: CUDA 代码生成器 (~2700 行) |
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
export TRITON_BACKENDS_IN_TREE=1

# 运行所有 kernel 编译测试 (vector-add, relu, softmax, matmul-wgmma)
python python/tutorials/emit-cuda/_test_compile_all.py

# vector-add 端到端正确性 + 性能测试
python python/tutorials/emit-cuda/_test_small.py
python python/tutorials/emit-cuda/_test_e2e.py
```

## Results (H800, sm90, CUDA 12.2)

| Kernel | CUDA Lines | cubin | Correctness | Perf | sm90a |
|--------|-----------|-------|-------------|------|-------|
| vector-add (f32) | 110 | 4.4KB | **BITWISE MATCH** | **1.65x faster** | - |
| ReLU (f32) | 101 | 4.2KB | nvcc OK | - | - |
| softmax (f32) | 140 | 8.0KB | nvcc OK | - | reduce (warp shuffle) |
| matmul (fp16) | 443 | 14KB | nvcc OK | - | **WGMMA v3**, nvmma_shared |

## sm90a NVGPUIR 支持

### 已实现

| Feature | TTGIR/NVGPUIR Op | CUDA 翻译 |
|---------|-----------------|----------|
| WGMMA | `ttng.warp_group_dot` | PTX `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16` |
| WGMMA Wait | `ttng.warp_group_dot_wait` | PTX `wgmma.wait_group.sync.aligned N` |
| WGMMA Fence | (内置于 dot emitter) | PTX `wgmma.fence.sync.aligned` + `wgmma.commit_group.sync.aligned` |
| Async Fence | `ttng.fence_async_shared` | PTX `fence.proxy.async.shared::cta` |
| Shared Mem Desc | 64-bit descriptor | swizzle mode + base addr + stride/lead dim |
| mbarrier Init | `ttng.init_barrier` | PTX `mbarrier.init.shared::cta.b64` |
| mbarrier Wait | `ttng.wait_barrier` | PTX `mbarrier.try_wait.parity` (loop) |
| mbarrier Arrive | `ttng.arrive_barrier` | PTX `mbarrier.arrive.shared::cta.b64` |
| mbarrier Expect | `ttng.barrier_expect` | PTX `mbarrier.expect_tx.shared::cta.b64` |
| NVMMA Shared | `#ttg.nvmma_shared` layout | 128B-aligned alloc, swizzle descriptor |
| MMA Layout | `#ttg.nvidia_mma<v3>` | per-thread elem 计算 |
| Layout Conv | `ttg.convert_layout` #mma→#blocked | shared memory intermediary |

### TODO

| Feature | Op | 说明 |
|---------|-----|------|
| TMA Copy | `ttng.async_tma_copy_*` | 需要 host 端 tensor descriptor |
| Warp Specialize | `ttg.warp_specialize` | 不同 warp group 执行不同代码 |
| Tensor Memory | `ttng.tmem_*` | sm_100+ (Blackwell) |
| stmatrix | layout conv 优化 | 替代 shared memory 中转 |

## IR Dialects Reference

### TTGIR 核心 Op
- `tt.func` → kernel, `tt.get_program_id` → blockIdx
- `tt.load`/`tt.store` → global memory, `tt.addptr` → 指针算术
- `tt.dot` → 矩阵乘法, `tt.reduce` → reduction
- `tt.make_range`/`tt.splat`/`tt.broadcast`/`tt.expand_dims` → 索引
- `arith.*` (40+ ops), `math.*` (15+ ops) → 计算
- `scf.for`/`scf.if`/`scf.yield` → 控制流
- `ttg.local_alloc`/`ttg.local_load`/`ttg.local_store` → shared memory
- `ttg.convert_layout` → layout 转换

### NVGPUIR 核心 Op (sm90a)
- `ttng.warp_group_dot` / `ttng.warp_group_dot_wait` → WGMMA
- `ttng.fence_async_shared` → async fence
- `ttng.init_barrier` / `ttng.wait_barrier` / `ttng.arrive_barrier` → mbarrier
- `ttng.async_tma_copy_global_to_local` / `ttng.async_tma_copy_local_to_global` → TMA

### Layout Types
- `#ttg.blocked<{sizePerThread, threadsPerWarp, warpsPerCTA, order}>` → 基础分布
- `#ttg.nvidia_mma<{versionMajor=3, warpsPerCTA, instrShape}>` → WGMMA 输出
- `#ttg.nvmma_shared<{swizzlingByteWidth, transposed, elementBitWidth}>` → WGMMA 输入
- `#ttg.slice<{dim, parent}>` → 切片 (去掉一个维度)
- `#ttg.shared_memory` → 通用共享内存

## Development

- Branch: `zsz/triton-cuda`
- Base: `dist`
- Working directory: `/data00/zheng.size/share/triton-cuda`
- Docker: `zhengsize-vibecuda` (H800, CUDA 12.2, triton 3.7.0)
- 环境变量: `TRITON_BACKENDS_IN_TREE=1` (跳过 tilelang driver 冲突)
