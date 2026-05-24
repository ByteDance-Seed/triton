# CUDA Emitter Design Document

## 1. Overview

本文档描述 Triton CUDA Emitter 的设计，目标是将 TTGIR/NVGPUIR 翻译为可读的 CUDA C++ 代码。
在 TTGIR 层保留了人类可读的控制流结构（for 循环、if 判断），同时所有 GPU 优化（coalescing、
MMA layout、pipelining、prefetch、WGMMA 等）已经完成。当 TTGIR 缺乏 sm90a 特有信息时，
在 NVGPUIR（`ttng.*`/`nvgpu.*`）层进行翻译。

### 1.1 设计目标

1. **正确性**: bitwise 精度匹配原始 Triton 编译路线
2. **性能**: 100% 性能匹配（相同的内存访问模式、计算指令、共享内存使用）
3. **可读性**: 生成的 CUDA 代码应该是人类可阅读的
4. **sm90a 完整性**: 完整支持 Hopper WGMMA、TMA、mbarrier 等特性

### 1.2 编译流水线

```
原始路线: TTIR → TTGIR(含NVGPUIR ops) → LLVM-IR(MLIR) → LLVM-IR(LLVM) → PTX → cubin
新路线:   TTIR → TTGIR(含NVGPUIR ops) → CUDA(.cu) → cubin(via nvcc)
```

共享前两个阶段（TTIR 和 TTGIR/NVGPUIR），所有 GPU 优化 pass 均在 `make_ttgir()` 中完成，
包括 sm90a 的 WGMMA lowering、TMA lowering、fence insertion 等。CUDA emitter 在
TTGIR 之后分叉，翻译 TTGIR + NVGPUIR ops 为 CUDA 代码。

### 1.3 为什么在 TTGIR/NVGPUIR 层翻译

| 层级 | 优点 | 缺点 |
|------|------|------|
| TTIR | 简单 | 未优化，性能差 |
| **TTGIR/NVGPUIR** | **保留 for/if 结构，已完成所有优化** | **需要处理多种 layout 和 sm90a ops** |
| LLVM-IR | 已完全展开 | 丢失 for/if 结构，不可读 |

## 2. Architecture

### 2.1 系统架构

```
@triton.jit(emit_cuda=True)
         │
    ┌─────────┐
    │  TTIR   │  (unchanged)
    └────┬────┘
         │  make_ttir()
    ┌─────────┐
    │  TTGIR  │  包含 ttng.* NVGPUIR ops (unchanged)
    │ +NVGPUIR│  ttng.warp_group_dot, ttng.fence_async_shared, etc.
    └────┬────┘
         │  make_ttgir() — 所有 GPU 优化 pass 在此完成
         ▼
    ┌─────────────────────────────────┐
    │       CUDAEmitter               │  ← 核心翻译
    │  ┌───────────┐ ┌─────────────┐  │
    │  │MLIRParser │ │ CUDACodeGen │  │
    │  │ 解析 TTGIR │ │ 生成 CUDA   │  │
    │  │ +NVGPUIR  │ │ +PTX asm    │  │
    │  └───────────┘ └─────────────┘  │
    └────────────┬────────────────────┘
                 │
    ┌─────────┐
    │  .cu    │  可读的 CUDA C++ 源码
    └────┬────┘
         │  nvcc -cubin
    ┌─────────┐
    │  cubin  │
    └─────────┘
```

### 2.2 Pipeline 集成

`CUDAOptions` 增加 `emit_cuda: bool = False` 选项：

```python
# third_party/nvidia/backend/compiler.py
def add_stages(self, stages, options, language):
    # TTIR → TTGIR 不变
    stages["ttir"] = ...
    stages["ttgir"] = ...  # 包含 ttng.* NVGPUIR passes
    
    if options.emit_cuda:
        stages["cuda"]  = lambda src, metadata: self.make_cuda(src, metadata, options, capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin_from_cuda(src, metadata, options, capability)
    else:
        stages["llir"]  = ...  # 原始路线
        stages["ptx"]   = ...
        stages["cubin"] = ...
```

### 2.3 nvcc 编译

```bash
nvcc -cubin --gpu-architecture=sm_90a -O3 --use_fast_math --fmad=true -std=c++17 input.cu -o output.cubin
```

## 3. CUDA Emitter 核心设计

### 3.1 模块结构 (`cuda_emitter.py`, ~2700 行)

```
CUDAEmitter          # 顶层编排器
├── MLIRTextParser   # 解析 MLIR text → 内部 IR 表示
│   ├── Layout 解析: blocked, nvidia_mma, nvmma_shared, slice, shared_memory
│   ├── Function 解析: 参数, 类型, loc 注解剥离
│   ├── Operation 解析: 单行 op, 含 region 的 op (scf.for, scf.if, tt.reduce)
│   └── Type 解析: tensor (含嵌套 <>, 如 tensor<128x!tt.ptr<f16>, #blocked>)
├── CUDACodeGen      # 生成 CUDA C++ 代码
│   ├── TTGIR Op Emitters: load/store, arith, math, splat, make_range, ...
│   ├── NVGPUIR Op Emitters: warp_group_dot(WGMMA), fence, mbarrier, TMA
│   ├── Control Flow: scf.for (iter_args + multi-result), scf.if, scf.yield
│   ├── Layout Conversion: #mma → #blocked via shared memory
│   └── Pointer Tensor Tracking: 全链路指针张量类型追踪
└── LayoutComputer   # Layout → per-thread 索引映射
```

### 3.2 Layout 支持

#### Blocked Layout
```
#blocked = #ttg.blocked<{sizePerThread=[1,1], threadsPerWarp=[1,32], warpsPerCTA=[4,1], order=[1,0]}>
```
- per-thread 元素数 = Π(max(1, shape[d] / (tpw[d] * wpc[d])) * spt[d])
- 广播维度 (shape[d]==1) 不计入线程分配

#### MMA Layout (sm90a WGMMA)
```
#mma = #ttg.nvidia_mma<{versionMajor=3, versionMinor=0, warpsPerCTA=[4,1], instrShape=[16,128,16]}>
```
- versionMajor=3 → WGMMA (Hopper)
- instrShape: WGMMA 指令形状 (m, n, k)
- per-thread 元素数 = total_elements / num_threads

#### NVMMA Shared Layout
```
#shared = #ttg.nvmma_shared<{swizzlingByteWidth=64, transposed=false, elementBitWidth=16}>
```
- WGMMA 操作数的共享内存布局
- swizzlingByteWidth 决定 bank conflict 消除策略
- 映射到 64-bit shared memory descriptor 的 swizzling mode

#### Slice Layout
```
#ttg.slice<{dim=1, parent=#blocked1}>
```
- 从父 layout 中去掉一个维度
- per-thread 元素数从父 layout 推导

### 3.3 指针张量追踪

TTGIR 中有两种张量：
- **值张量**: `tensor<1024xf32, #blocked>` → CUDA 数组 `float v[N]`
- **指针张量**: `tensor<1024x!tt.ptr<f32>, #blocked>` → CUDA 指针数组 `float* v[N]`

Emitter 通过 `ssa_is_ptr_tensor` dict 全链路追踪指针张量：
- `tt.splat %ptr` → 标记指针张量
- `tt.addptr` → 标记指针张量
- `tt.broadcast` → 传播标记
- `scf.for iter_args` → 传播标记到循环变量
- `tt.load` → 结果为值张量（pointee 类型）
- `tt.store` → 读取指针张量元素进行存储

### 3.4 多结果 Op 处理

TTGIR 中的多结果 op 使用 `%name:N` 语法，引用时用 `%name#i`：
```mlir
%result:3 = scf.for ... iter_args(%a = %init_a, %b = %init_b, %c = %init_c) -> (T1, T2, T3)
// 引用: %result#0 (= a), %result#1 (= b), %result#2 (= c)
```

Emitter 在注册结果时展开为 `%result#0`, `%result#1`, `%result#2`，
各自映射到对应 iter_arg 的 CUDA 变量。

## 4. sm90a NVGPUIR 翻译

### 4.1 WGMMA (`ttng.warp_group_dot`)

**TTGIR**:
```mlir
ttng.warp_group_dot %a_smem, %b_smem, %acc {inputPrecision=0, isAsync=true}
  : !ttg.memdesc<128x32xf16, #shared, #smem> * !ttg.memdesc<32x128xf16, #shared1, #smem>
  -> tensor<128x128xf32, #mma>
```

**翻译为 CUDA PTX inline asm**:
```cuda
{
    uint32_t smem_addr_a = __cvta_generic_to_shared(smem_a);
    uint32_t smem_addr_b = __cvta_generic_to_shared(smem_b);
    
    asm volatile("wgmma.fence.sync.aligned;");
    
    // 构造 64-bit shared memory descriptor
    uint64_t desc_a = ((uint64_t)((smem_addr_a + offset) >> 4)) | TEMPLATE_A;
    uint64_t desc_b = ((uint64_t)((smem_addr_b + offset) >> 4)) | TEMPLATE_B;
    
    // WGMMA 指令: m64n128k16.f32.f16.f16
    // 64 个 f32 累加器寄存器 ("+f" read-write constraint)
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,...,%63}, %64, %65, 1, 1, 1, 0, 1;"
        : "+f"(acc[0]), "+f"(acc[1]), ..., "+f"(acc[63])
        : "l"(desc_a), "l"(desc_b)
    );
    
    asm volatile("wgmma.commit_group.sync.aligned;");
}
```

**WGMMA 指令展开策略**:
- M_tiles = M_block / 64 (WGMMA m=64)
- K_tiles = K_block / 16 (f16 k=16)
- 每个 (m_tile, k_tile) 对生成一条 WGMMA 指令
- 例：128×128 matmul, K=32 → 2 M-tiles × 2 K-tiles = 4 条 WGMMA

### 4.2 Shared Memory Descriptor (64-bit)

```
union SMEMDescriptor {
    uint64_t descriptor;
    struct {
        uint64_t baseAddress : 14;                 // bits [0:13]  = (smem_addr >> 4) & 0x3FFF
        uint64_t : 2;                              // bits [14:15]
        uint64_t leadDimensionBaseOffset : 14;     // bits [16:29] = (swizzle * stride) >> 4
        uint64_t : 2;                              // bits [30:31]
        uint64_t strideDimensionBaseOffset : 14;   // bits [32:45] = swizzle >> 1
        uint64_t : 3;                              // bits [46:48]
        uint64_t matrixBaseOffset : 3;             // bits [49:51] = 0
        uint64_t : 10;                             // bits [52:61]
        uint64_t swizzlingMode : 2;                // bits [62:63]
    };
};
```

| swizzlingByteWidth | swizzlingMode |
|-------------------|---------------|
| 0   | 0 (none) |
| 32  | 3 |
| 64  | 2 |
| 128 | 1 |

descriptor template = `swizzlingMode << 62 | strideDim << 32 | leadDim << 16`
runtime descriptor = template | (smem_base >> 4)

### 4.3 WGMMA Wait (`ttng.warp_group_dot_wait`)

```mlir
%result:3 = ttng.warp_group_dot_wait %acc, %a_smem, %b_smem {pendings = 0}
```
→
```cuda
asm volatile("wgmma.wait_group.sync.aligned 0;");
// 结果别名到输入 (accumulator + shared mem operands)
```

### 4.4 Fence (`ttng.fence_async_shared`)

```mlir
ttng.fence_async_shared {bCluster = false}
```
→
```cuda
asm volatile("fence.proxy.async.shared::cta;");
// bCluster=true → "fence.proxy.async.shared::cluster;"
```

### 4.5 mbarrier 操作

| TTGIR Op | CUDA PTX |
|----------|----------|
| `ttng.init_barrier %bar, count` | `mbarrier.init.shared::cta.b64 [addr], count` |
| `ttng.wait_barrier %bar, phase` | `mbarrier.try_wait.parity.shared::cta.b64 P, [addr], phase` (loop) |
| `ttng.arrive_barrier %bar` | `mbarrier.arrive.shared::cta.b64 _, [addr]` |
| `ttng.barrier_expect %bar, bytes` | `mbarrier.expect_tx.shared::cta.b64 [addr], bytes` |

### 4.6 TMA 操作 (TODO)

| TTGIR Op | PTX |
|----------|-----|
| `ttng.async_tma_copy_global_to_local` | `cp.async.bulk.tensor.Nd.global.shared::cta.tile` |
| `ttng.async_tma_copy_local_to_global` | `cp.async.bulk.tensor.Nd.shared::cta.global.tile` |
| `ttng.tma_store_wait` | `cp.async.bulk.wait_group 0` |

TMA 需要 host 端创建 tensor descriptor，目前标记为 TODO。

### 4.7 Layout Conversion (#mma → #blocked)

WGMMA 的累加器使用 `#mma` layout，存储需要转换为 `#blocked` layout：

```cuda
// 1. Store MMA registers to shared memory
__shared__ float cvt_smem[M * N];
for (int i = 0; i < ELEMS_PER_THREAD; i++)
    cvt_smem[tid * ELEMS_PER_THREAD + i] = acc[i];
__syncthreads();

// 2. Load with blocked layout addressing
for (int i = 0; i < ELEMS_PER_THREAD; i++)
    result[i] = cvt_smem[tid * ELEMS_PER_THREAD + i];
__syncthreads();
```

注：完整实现应使用 `stmatrix` PTX 指令和精确的 layout 映射。

## 5. 基础 TTGIR Op 翻译

### 5.1 内存操作

| Op | CUDA |
|----|------|
| `tt.load %ptr, %mask` | `if (mask[i]) val[i] = *ptr[i];` |
| `tt.load %ptr, %mask, %other` | `val[i] = mask[i] ? *ptr[i] : other;` |
| `tt.store %ptr, %val, %mask` | `if (mask[i]) *ptr[i] = val[i];` |
| `tt.addptr %ptr, %off` | `ptr[i] = ptr[i] + off[i];` |
| `ttg.local_alloc` | `__half* smem = (__half*)(shared_mem + offset);` (128B aligned for WGMMA) |

### 5.2 Arithmetic / Math

40+ 种 arith ops 和 15+ 种 math ops，详见 `op-mapping.md`。

### 5.3 控制流

| Op | CUDA |
|----|------|
| `scf.for %iv = %lb to %ub step %step iter_args(...)` | `for (int iv = lb; iv < ub; iv += step) { ... }` |
| `scf.if %cond` | `if (cond) { ... } else { ... }` |
| `scf.yield %val` | 赋值给 iter_arg 变量 |

### 5.4 Reduction (`tt.reduce`)

- Step 1: Thread-local reduction (遍历 per-thread 元素)
- Step 2: Warp-level reduction (`__shfl_xor_sync`)
- Step 3: Cross-warp reduction (shared memory + second warp reduce)

## 6. 类型系统

| MLIR Type | CUDA Type |
|-----------|-----------|
| `f32` | `float` |
| `f16` | `__half` |
| `bf16` | `__nv_bfloat16` |
| `f64` | `double` |
| `i1` | `bool` |
| `i32` | `int` |
| `i64` | `int64_t` |
| `!tt.ptr<f32>` | `float*` |
| `!tt.ptr<f16>` | `__half*` |
| `tensor<NxT, #layout>` | `T var[ELEMS_PER_THREAD]` |
| `tensor<Nx!tt.ptr<T>, #layout>` | `T* var[ELEMS_PER_THREAD]` |
| `!ttg.memdesc<MxNxT, #shared>` | `T* smem_ptr` (shared memory offset) |

## 7. 测试结果

### 7.1 编译结果 (H800, sm90, CUDA 12.2)

| Kernel | CUDA Lines | cubin Size | Correctness | sm90a Features |
|--------|-----------|------------|-------------|----------------|
| vector-add (f32) | 110 | 4.4KB | **BITWISE MATCH** | - |
| ReLU (f32) | 101 | 4.2KB | compiles OK | - |
| softmax (f32, reduce) | 140 | 8.0KB | compiles OK | - |
| matmul (fp16, WGMMA) | 443 | 14KB | compiles OK | WGMMA v3, nvmma_shared |

### 7.2 性能 (vector-add)

```
Triton (PTX path):   12.0μs (98.2 GB/s)
CUDA emitter path:    7.3μs (162.6 GB/s)
Ratio: 0.60x (CUDA emitter 快 1.65x，含 launch overhead)
```

## 8. 已知限制和 TODO

| 项目 | 状态 |
|------|------|
| 基础 elementwise ops | ✅ 完成 |
| Global load/store with mask | ✅ 完成 |
| Blocked layout per-thread mapping | ✅ 完成 |
| scf.for with iter_args + multi-result | ✅ 完成 |
| scf.if | ✅ 完成 |
| tt.reduce (sum, max, min) | ✅ 完成 |
| WGMMA PTX inline asm | ✅ 完成 |
| Shared memory descriptor | ✅ 完成 |
| mbarrier ops | ✅ 完成 |
| fence.proxy.async.shared | ✅ 完成 |
| Pointer tensor tracking | ✅ 完成 |
| #mma → #blocked convert_layout | ✅ 基础实现 |
| TMA copy (cp.async.bulk.tensor) | 🔧 TODO |
| Warp Specialization | 🔧 TODO |
| Async pipelining (multi-stage) | 🔧 TODO |
| Tensor Memory (TMEM, sm_100+) | 🔧 TODO |
| stmatrix for layout conversion | 🔧 TODO |
| 精确的 layout index 计算 | 🔧 TODO (当前用简化版) |
