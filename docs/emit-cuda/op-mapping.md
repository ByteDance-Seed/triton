# TTGIR/NVGPUIR → CUDA Op Mapping Reference

本文档详细列出每个 TTGIR/NVGPUIR operation 到 CUDA C++ 代码的翻译规则。
包含 sm90a 特有的 WGMMA、TMA、mbarrier 等操作。

## 1. Triton Dialect Ops (`tt.*`)

### 核心 Op

| Op | CUDA 翻译 | 说明 |
|----|----------|------|
| `tt.func public @name(args)` | `extern "C" __global__ void __launch_bounds__(N) name(args)` | N = warps × 32 |
| `tt.get_program_id x/y/z` | `blockIdx.x/y/z` | |
| `tt.get_num_programs x/y/z` | `gridDim.x/y/z` | |
| `tt.make_range {start, end}` | Per-thread index array | 根据 blocked layout 计算 |
| `tt.splat %scalar → tensor` | Broadcast 到 per-thread 数组 | 区分值 splat 和指针 splat |
| `tt.broadcast %tensor` | 沿广播维度复制元素 | `val[i % src_n]` |
| `tt.expand_dims %tensor {axis=N}` | 添加 size=1 维度 | 不改变数据，可能改变 per-thread 数量 |
| `tt.addptr %ptr, %offset` | `ptr[i] = ptr[i] + offset[i]` | 结果为指针张量 |
| `tt.load %ptr, %mask` | `if (mask[i]) val[i] = *ptr[i]; else val[i] = 0;` | |
| `tt.load %ptr, %mask, %other` | `val[i] = mask[i] ? *ptr[i] : other[i];` | |
| `tt.store %ptr, %val, %mask` | `if (mask[i]) *ptr[i] = val[i];` | |
| `tt.dot %a, %b, %acc` | FMA 循环 / MMA PTX asm | 见 §4 |
| `tt.reduce(%val) {axis=N}` | Thread + warp + cross-warp reduce | 见 §5 |
| `tt.trans %tensor` | 元数据变更（指针别名） | |
| `tt.return` | `return;` | |

### 指针张量跟踪

以下 op 产生指针张量，通过 `ssa_is_ptr_tensor` 全链路追踪：
```
tt.splat (当源为指针) → tt.addptr → tt.broadcast → scf.for iter_args
```

`tt.load` 的结果是 **值张量**（pointee 类型），不是指针张量。

## 2. Arithmetic Ops (`arith.*`)

### 二元算术

| Op | CUDA | Notes |
|----|------|-------|
| `arith.addi/addf` | `a + b` | |
| `arith.subi/subf` | `a - b` | |
| `arith.muli/mulf` | `a * b` | |
| `arith.divsi` | `a / b` | Signed |
| `arith.divui` | `(unsigned)a / (unsigned)b` | |
| `arith.divf` | `a / b` | |
| `arith.remsi/remui` | `a % b` | |
| `arith.andi` | `a & b` | |
| `arith.ori` | `a \| b` | |
| `arith.xori` | `a ^ b` | |
| `arith.shli` | `a << b` | |
| `arith.shrsi` | `a >> b` | Arithmetic |
| `arith.shrui` | `(unsigned)a >> b` | Logical |
| `arith.maxf/maximumf` | `fmaxf(a, b)` | |
| `arith.minf/minimumf` | `fminf(a, b)` | |
| `arith.maxsi/minsi` | `max(a, b)` / `min(a, b)` | |

### 比较 (predicate 在 op name 中: `arith.cmpi slt`)

| Predicate | CUDA |
|-----------|------|
| `eq`/`oeq` | `==` |
| `ne`/`one`/`une` | `!=` |
| `slt`/`olt` | `<` |
| `sle`/`ole` | `<=` |
| `sgt`/`ogt` | `>` |
| `sge`/`oge` | `>=` |
| `ult`/`ule`/`ugt`/`uge` | unsigned 比较 |

### 一元/转换

| Op | CUDA |
|----|------|
| `arith.negf` | `-a` |
| `arith.select` | `cond ? a : b` |
| `arith.extf` | `(double)(float)a` |
| `arith.truncf` | `(__half)(float)a` |
| `arith.extsi/extui` | `(int64_t)a` |
| `arith.trunci` | `(int32_t)a` |
| `arith.sitofp/uitofp` | `(float)a` |
| `arith.fptosi/fptoui` | `(int)a` |
| `arith.bitcast` | `__int_as_float(a)` / `__float_as_int(a)` |

## 3. Math Ops (`math.*`)

| Op | CUDA | Notes |
|----|------|-------|
| `math.exp` | `expf(x)` | |
| `math.exp2` | `exp2f(x)` | |
| `math.log` | `logf(x)` | |
| `math.log2` | `log2f(x)` | |
| `math.sqrt` | `sqrtf(x)` | |
| `math.rsqrt` | `rsqrtf(x)` | |
| `math.sin/cos` | `sinf/cosf(x)` | |
| `math.tanh` | `tanhf(x)` | |
| `math.abs/absf` | `fabsf(x)` | |
| `math.ceil/floor` | `ceilf/floorf(x)` | |
| `math.fma` | `fmaf(a, b, c)` | |
| `math.erf` | `erff(x)` | |
| `math.powf` | `powf(x, y)` | |

## 4. TritonGPU Ops (`ttg.*`)

| Op | CUDA | 说明 |
|----|------|------|
| `ttg.local_alloc` | `T* smem = (T*)(shared_mem + offset)` | 128B aligned for WGMMA |
| `ttg.local_alloc %src` | alloc + store to shared | 带源的分配=分配+写入 |
| `ttg.local_store` | `smem[tid * N + i] = val[i]` | |
| `ttg.local_load` | `val[i] = smem[tid * N + i]` | |
| `ttg.convert_layout` | shared memory 中转 | #mma→#blocked 需要 stmatrix |
| `gpu.barrier` | `__syncthreads()` | |

### Layout Conversion 详解

**#blocked → #blocked** (不同 layout):
```cuda
smem[src_idx(tid, i)] = val[i];
__syncthreads();
val[i] = smem[dst_idx(tid, i)];
__syncthreads();
```

**#mma → #blocked** (WGMMA 累加器输出):
```cuda
// MMA registers → shared memory → blocked registers
_cvt_smem[tid * N + i] = acc[i];
__syncthreads();
result[i] = _cvt_smem[tid * N + i];
__syncthreads();
```

## 5. SCF Ops (Control Flow)

### scf.for (含 iter_args 和多结果)

```mlir
%result:3 = scf.for %iv = %lb to %ub step %step
    iter_args(%a = %init_a, %b = %init_b, %c = %init_c)
    -> (tensor<...>, tensor<...>, tensor<...>) : i32 {
    ...
    scf.yield %new_a, %new_b, %new_c
}
// 后续引用: %result#0, %result#1, %result#2
```
→
```cuda
T_a iter_a[N_a];
copy(iter_a, init_a);
T_b* iter_b[N_b];     // 指针张量用 T*
copy(iter_b, init_b);
T_c iter_c[N_c];
copy(iter_c, init_c);

for (int iv = lb; iv < ub; iv += step) {
    // body...
    copy(iter_a, new_a);  // scf.yield
    copy(iter_b, new_b);
    copy(iter_c, new_c);
}
// result#0 = iter_a, result#1 = iter_b, result#2 = iter_c
```

### scf.if / scf.while

标准 C++ `if/else` 和 `while` 翻译。

## 6. NVGPUIR Ops — sm90a (`ttng.*`)

### 6.1 WGMMA (`ttng.warp_group_dot`)

```mlir
ttng.warp_group_dot %a_smem, %b_smem, %acc {inputPrecision=0, isAsync=true}
  : !ttg.memdesc<128x32xf16, #shared> * !ttg.memdesc<32x128xf16, #shared1>
  -> tensor<128x128xf32, #mma>
```
→ PTX inline asm:
```cuda
asm volatile("wgmma.fence.sync.aligned;");

// 对每个 (m_tile, k_tile):
asm volatile(
    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
    "{%0,...,%63}, %64, %65, 1, 1, 1, 0, 1;"
    : "+f"(acc[0]), ..., "+f"(acc[63])   // 64 个 f32 read-write 累加器
    : "l"(desc_a), "l"(desc_b)           // 64-bit shared mem descriptors
);

asm volatile("wgmma.commit_group.sync.aligned;");
```

**展开规则**:
- M_tiles = M_block / 64
- K_tiles = K_block / (16 for f16, 8 for tf32)
- 每个 m_tile 64 个输出寄存器
- 总 WGMMA 指令数 = M_tiles × K_tiles

### 6.2 WGMMA Wait (`ttng.warp_group_dot_wait`)

```mlir
%out:3 = ttng.warp_group_dot_wait %acc, %a, %b {pendings = 0}
```
→
```cuda
asm volatile("wgmma.wait_group.sync.aligned 0;");
// %out#0 = %acc, %out#1 = %a, %out#2 = %b (别名)
```

### 6.3 Fence (`ttng.fence_async_shared`)

```mlir
ttng.fence_async_shared {bCluster = false}
```
→
```cuda
asm volatile("fence.proxy.async.shared::cta;");     // bCluster=false
asm volatile("fence.proxy.async.shared::cluster;");  // bCluster=true
```

### 6.4 Barrier Ops

| Op | PTX |
|----|-----|
| `ttng.init_barrier %bar` | `mbarrier.init.shared::cta.b64 [addr], count;` |
| `ttng.wait_barrier %bar, %phase` | `mbarrier.try_wait.parity.shared::cta.b64 P, [addr], phase;` (循环等待) |
| `ttng.arrive_barrier %bar` | `mbarrier.arrive.shared::cta.b64 _, [addr];` |
| `ttng.barrier_expect %bar, %bytes` | `mbarrier.expect_tx.shared::cta.b64 [addr], bytes;` |
| `ttng.inval_barrier %bar` | (no-op) |

### 6.5 TMA Copy (TODO)

| Op | PTX |
|----|-----|
| `ttng.async_tma_copy_global_to_local` | `cp.async.bulk.tensor.Nd.global.shared::cta.tile [smem], [desc, coords], [mbar];` |
| `ttng.async_tma_copy_local_to_global` | `cp.async.bulk.tensor.Nd.shared::cta.global.tile [desc, coords], [smem];` |
| `ttng.tma_store_wait` | `cp.async.bulk.wait_group 0;` |

## 7. Shared Memory Descriptor (64-bit)

WGMMA 操作数使用 64-bit shared memory descriptor：

```
Bits     Field                     Computation
[0:13]   baseAddress               (smem_addr >> 4) & 0x3FFF
[16:29]  leadDimensionBaseOffset   (swizzle * stride_bytes) >> 4
[32:45]  strideDimensionBaseOffset swizzle >> 1
[62:63]  swizzlingMode             {0:none, 1:128B, 2:64B, 3:32B}
```

descriptor = `template | (runtime_base >> 4)`

其中 template 在编译时计算：
```python
template = (swizzle_mode << 62) | (stride_dim << 32) | (lead_dim << 16)
```

## 8. 生成的 CUDA 代码结构

```cuda
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

// Warp reduction helpers
__device__ __forceinline__ float warp_reduce_sum(float val) { ... }
__device__ __forceinline__ float warp_reduce_max(float val) { ... }

extern "C" __global__ void __launch_bounds__(128)
kernel_name(__half* arg0, __half* arg1, __half* arg2, int M, int N, int K, ...) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    extern __shared__ char shared_mem[];

    // ... index computation ...
    // ... pointer setup ...
    // ... main loop with WGMMA ...
    // ... store results ...
}
```
