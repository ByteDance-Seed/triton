"""
Tutorial 01 — Warp-Specialized GEMM (Gluon frontend → emit_cuda → CUDA)
=======================================================================

GOAL
----
Express a Hopper (sm90) **warp-specialized** GEMM in the *Gluon* frontend with
ONE producer warpgroup and TWO consumer warpgroups, then lower it through the
out-of-tree CUDA backend (``emit_cuda``) and show that the emitted CUDA C++ has
the *same structure* as the hand-written reference kernel
``XPUTutorial/nvgpu/sm90/bf16_dense_gemm/gemm_06_autotuned.cu``.

This is the canonical producer/consumer ("ping-pong" / CUTLASS-style) design:

    warpgroup 0  (warps  0.. 3)  -> PRODUCER : issues TMA loads, setmaxnreg.dec
    warpgroup 1  (warps  4.. 7)  -> CONSUMER0: WGMMA + epilogue, setmaxnreg.inc
    warpgroup 2  (warps  8..11)  -> CONSUMER1: WGMMA + epilogue, setmaxnreg.inc

Each consumer is a self-contained 128-thread WGMMA unit that computes its OWN
output row-tile. The producer feeds both consumers through a shared ring of
SMEM buffers, synchronised with mbarrier full/empty pairs. The producer gives up
registers (``setmaxnreg.dec 40``) so the two math warpgroups can each claim 232 —
the exact register split gemm_06 uses (``PRODUCER_REGS=40``, ``CONSUMER_REGS=232``).

HOW THE FRONTEND MAPS TO WARP SPECIALIZATION
--------------------------------------------
``tl.range(..., warp_specialize=True)`` is *automatic* WS (a bool; Blackwell-only
for matmul). To get *explicit* control over partitions, warp counts and register
budgets on Hopper, we drop to Gluon's ``gl.warp_specialize``:

    gl.warp_specialize(
        [ (producer_fn,  producer_args),     # functions_and_args[0] = DEFAULT partition
          (consumer_fn,  consumer0_args),    # functions_and_args[1:] = WORKER partitions
          (consumer_fn,  consumer1_args) ],
        [4, 4],            # worker_num_warps   : each consumer warpgroup = 4 warps
        [232, 232],        # worker_num_regs    : each consumer requests 232 registers
    )

  * The DEFAULT partition runs in the parent kernel's ``num_warps`` (=4 here) — it
    is our producer.
  * Each WORKER partition runs in *additional* warps. Two workers × 4 warps =
    our two consumer warpgroups. The block therefore launches 4+4+4 = 12 warps,
    i.e. ``__launch_bounds__(384)``.

KEY CONSTRAINT — ONE SHARED CAPTURE LIST
----------------------------------------
``ttg.warp_specialize`` captures a SINGLE operand list that binds positionally to
EVERY partition region's block args (the regions are isolated-from-above). So you
cannot hand consumer 0 and consumer 1 *different* runtime buffers — both see the
same captured SMEM. The two consumers diverge purely through a **constexpr**
``CONSUMER_ID`` baked into each traced body, which they use to index their own
slice of the shared ring (``base = CONSUMER_ID * num_buffers``) and their own
output row-tile (``off_m = (pid_m*NUM_CONSUMERS + CONSUMER_ID) * BLOCK_M``).

A second constraint: an ``NVMMASharedLayout`` is 2-D, so a SMEM allocation may be
at most 3-D (one leading ring dimension). We therefore *flatten*
``[NUM_CONSUMERS, num_buffers]`` into a single ring of ``NUM_CONSUMERS*num_buffers``
slots rather than nesting.

THE GLUON → CUDA LOWERING PATH (what emit_cuda does)
----------------------------------------------------
  1. Gluon ``@gluon.jit`` traces to TTGIR. ``gl.warp_specialize`` becomes a
     ``ttg.warp_specialize`` op with a ``default`` region (producer) and N
     ``partition<i>`` regions (consumers), carrying ``partitionNumWarps=[4,4]``
     and ``requestedRegisters=[232,232]``.
  2. ``emit_cuda`` (CUDACodeGen.cpp ``emitWarpSpecialize``) lowers that op to a
     warp-id dispatch — see the emitted CUDA:
         if (warp_id < 4)        { setmaxnreg.dec 40;  <producer body>   }   // default region
         else if (warp_id < 8)   { setmaxnreg.inc 232; <consumer body 0> }   // partition 0
         else if (warp_id < 12)  { setmaxnreg.inc 232; <consumer body 1> }   // partition 1
     Inside each consumer branch it shadows tid/warp_id/lane_id to be
     *warpgroup-local* so the WGMMA / TMA / mma->smem emitters all address that
     consumer's own 128-thread tile (``_wg_m == 0``).
  3. Per-op emitters then turn:
         tma.async_copy_global_to_shared  -> cp.async.bulk.tensor.2d...complete_tx
         mbarrier.expect / .wait / .arrive-> mbarrier.arrive.expect_tx / try_wait.parity / arrive
         warpgroup_mma / _wait            -> wgmma.fence + wgmma.mma_async.m64nNk16 + commit/wait
         c_smem.store + tma store         -> stmatrix (R2S) + cp.async.bulk.tensor (S2G)
     The producer's ``setmaxnreg.dec`` value is computed to exactly balance the
     consumers' ``.inc`` within the 65536-reg SM budget — the gemm_02 invariant
     (1·(168-40)·128 == 2·(232-168)·128), reproducing gemm_06's 40/232 split.

REGISTER-BUDGET ARITHMETIC (why 40 / 232)
-----------------------------------------
``__launch_bounds__(384)`` ⇒ ptxas hands every thread floor(65536/384)&~7 = 168
registers. Each consumer raises to 232 (gain 64/thread × 256 threads = 16384).
The producer lowers by the same total over its 128 threads: 16384/128 = 128, so
168-128 = 40. Hence ``setmaxnreg.dec 40`` / ``setmaxnreg.inc 232``.

RUN
---
    docker exec -w /root/share/triton-cuda/triton_cuda_backend/tutorials \
      -e PYTHONPATH=/root/share/triton-cuda/python \
      -e PATH=/usr/local/cuda/bin:$PATH \
      -e TRITON_EMIT_CUDA=1 \
      -e TRITON_CUDA_DUMP=/tmp/t01_cu \
      zhengsize-vibecuda python 01-warp-specialize-gemm.py

NOTE: gluon needs the *repo* triton (PYTHONPATH=.../python); the container's
dist-packages triton is stale and lacks newer gluon APIs.
"""
import os
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared, warpgroup_mma, warpgroup_mma_wait,
)


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


def _cdiv(a, b):
    return (a + b - 1) // b


# These kernels are TMA-based, so ARBITRARY problem shapes are handled with NO
# alignment to the tile (BLOCK_M/N/K): we round the grid and the K-loop UP and let
# the TMA engine zero-fill out-of-bounds loads and drop out-of-bounds stores
# against each descriptor's real tensor extent. The ONLY residual requirement is a
# hardware floor of the TMA unit itself: a global tensor's row stride must be a
# multiple of 16 bytes, i.e. for fp16 the contiguous (column) dim must be a
# multiple of 8. That makes K (cols of A) and N (cols of B and C) require %8==0;
# M (a row count) is fully arbitrary. gemm_06 has the exact same floor.
def _check_tma_floor(K, N, dtype=torch.float16):
    elem = torch.tensor([], dtype=dtype).element_size()
    align = 16 // elem  # =8 for fp16
    assert K % align == 0 and N % align == 0, (
        f"TMA fp16 floor: K and N must be multiples of {align} (16-byte global "
        f"row stride); got K={K}, N={N}. M is unconstrained.")


# ---------------------------------------------------------------------------
# WGMMA accumulator layout helpers (constexpr so the dep-finder doesn't try to
# hash them as JIT functions; they only manipulate compile-time values).
# ---------------------------------------------------------------------------
@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, num_warps):
    # Start from the WGMMA atom [4, 1] and grow along M until we use num_warps.
    wpc = [4, 1]
    while wpc[0] * wpc[1] != num_warps:
        if BLOCK_M > 16 * wpc[0]:
            wpc[0] *= 2
        else:
            wpc[1] *= 2
    return wpc


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=get_warps_per_cta(BLOCK_M, num_warps),
        instr_shape=[16, BLOCK_N, 256 // dtype.primitive_bitwidth],
    )


# ---------------------------------------------------------------------------
# PRODUCER partition (the `default` region): TMA-load A and B for BOTH consumers.
# ---------------------------------------------------------------------------
@gluon.jit
def producer_partition(a_desc, b_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                       BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                       num_buffers: gl.constexpr, NUM_CONSUMERS: gl.constexpr):
    K = a_desc.shape[1]
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_n = pid_n * BLOCK_N

    # Feed each consumer's own sub-ring in turn.
    for c in gl.static_range(NUM_CONSUMERS):
        off_m = (pid_m * NUM_CONSUMERS + c) * BLOCK_M
        base = c * num_buffers
        index = 0
        phase = 1  # empty barriers are "available" on the first pass
        for k in range(0, K, BLOCK_K):
            slot = base + index
            mbarrier.wait(empty_bars.index(slot), phase)         # buffer free?
            bar = ready_bars.index(slot)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(slot))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(slot))
            index += 1
            if index == num_buffers:
                index = 0
                phase ^= 1


# ---------------------------------------------------------------------------
# CONSUMER partition (each `partition<i>` region): WGMMA over K + register-acc
# epilogue store. CONSUMER_ID (constexpr) selects this warpgroup's sub-ring and
# output row-tile — the only thing that differs between the two consumers.
# ---------------------------------------------------------------------------
@gluon.jit
def consumer_partition(c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                       BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                       num_buffers: gl.constexpr, num_warps: gl.constexpr,
                       NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr,
                       NUM_K_ITERS: gl.constexpr):
    dtype: gl.constexpr = c_desc.dtype
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = (pid_m * NUM_CONSUMERS + CONSUMER_ID) * BLOCK_M
    off_n = pid_n * BLOCK_N

    base: gl.constexpr = CONSUMER_ID * num_buffers
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    index = 0
    phase = 0
    use_acc = False
    # The producer issued exactly NUM_K_ITERS (= K / BLOCK_K) loads into this sub-ring.
    for kk in range(0, NUM_K_ITERS):
        slot = base + index
        mbarrier.wait(ready_bars.index(slot), phase)             # operands ready?
        acc = warpgroup_mma(a_bufs.index(slot), b_bufs.index(slot), acc,
                            is_async=True, use_acc=use_acc)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        mbarrier.arrive(empty_bars.index(slot), count=1)         # buffer is free
        use_acc = True
        index += 1
        if index == num_buffers:
            index = 0
            phase ^= 1

    # Epilogue: WGMMA accumulator (registers) -> SMEM (stmatrix) -> global (TMA).
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def ws_gemm_kernel(a_desc, b_desc, c_desc, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                   BLOCK_K: gl.constexpr, num_buffers: gl.constexpr, num_warps: gl.constexpr,
                   NUM_CONSUMERS: gl.constexpr, NUM_K_ITERS: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype
    total_bufs: gl.constexpr = NUM_CONSUMERS * num_buffers

    # Shared ring (flattened: consumer c owns slots [c*num_buffers, (c+1)*num_buffers)).
    a_bufs = gl.allocate_shared_memory(dtype, [total_bufs] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [total_bufs] + b_desc.block_type.shape, b_desc.layout)
    empty_bars = gl.allocate_shared_memory(gl.int64, [total_bufs, 1], mbarrier.MBarrierLayout())
    ready_bars = gl.allocate_shared_memory(gl.int64, [total_bufs, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(total_bufs):
        mbarrier.init(empty_bars.index(i), count=1)
        mbarrier.init(ready_bars.index(i), count=1)

    # 1 producer (default) + 2 consumers (workers). worker_num_warps / _regs give
    # us the gemm_06 register split (producer .dec 40, consumers .inc 232).
    gl.warp_specialize(
        [
            (producer_partition, (a_desc, b_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                  BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, NUM_CONSUMERS)),
            (consumer_partition, (c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                  BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS, 0, NUM_K_ITERS)),
            (consumer_partition, (c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                  BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS, 1, NUM_K_ITERS)),
        ],
        [num_warps, num_warps],
        [232, 232],
    )


# ===========================================================================
# V2 — CLOSING THE GAP WITH gemm_06
# ===========================================================================
# Profiling v1 (above) shows ~382 TFLOP/s — roughly HALF of cuBLAS. The dump
# diff against gemm_06 pinned the cause: v1 is *producer-bandwidth bound*, not
# WGMMA-bound. v1 gives each consumer its OWN sub-ring, so the single producer
# loads the *identical* N-tile of B twice (once per consumer) and a separate A
# tile per consumer — ~1.5–1.7× the necessary TMA traffic. The lone producer
# warpgroup cannot keep two math warpgroups fed.
#
# gemm_06 avoids this: the producer loads ONE BM_TOTAL(=NUM_CONSUMERS*BLOCK_M)-row
# A tile and ONE B tile per K-step into a SINGLE shared ring; both math
# warpgroups read the same buffers, each WGMMA-ing its own 64-row slice of A
# (gemm_06's `my_sA = cur_sA + row_offset*BK`). That halves producer traffic and
# lets both consumers run flat out. We reproduce it with `smem.slice(...)`:
#
#     a_slice = a_bufs.index(index).slice(CONSUMER_ID*BLOCK_M, BLOCK_M, dim=0)
#
# The empty barrier is init'd with count=NUM_CONSUMERS so a slot is recycled only
# after BOTH consumers have finished reading it. With BLOCK_N raised to 256
# (m64n256k16, gemm_06's tile) this lifts the kernel to ~735 TFLOP/s through
# emit_cuda — ~1.9× v1, on par with cuBLAS and ~92% of gemm_06 — while the
# emitted CUDA stays bitwise-identical in structure to v1 (same 40/232 split,
# __launch_bounds__(384), 3-way dispatch), now with a sliced A operand descriptor.
# ---------------------------------------------------------------------------
@gluon.jit
def producer_partition_v2(a_desc, b_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                          BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                          num_buffers: gl.constexpr, NUM_CONSUMERS: gl.constexpr):
    K = a_desc.shape[1]
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * (BLOCK_M * NUM_CONSUMERS)   # one BM_TOTAL-row tile for BOTH consumers
    off_n = pid_n * BLOCK_N
    index = 0
    phase = 1
    for k in range(0, K, BLOCK_K):
        mbarrier.wait(empty_bars.index(index), phase)
        bar = ready_bars.index(index)
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))
        index += 1
        if index == num_buffers:
            index = 0
            phase ^= 1


@gluon.jit
def consumer_partition_v2(c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                          BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                          num_buffers: gl.constexpr, num_warps: gl.constexpr,
                          NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr,
                          NUM_K_ITERS: gl.constexpr):
    dtype: gl.constexpr = c_desc.dtype
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * (BLOCK_M * NUM_CONSUMERS) + CONSUMER_ID * BLOCK_M
    off_n = pid_n * BLOCK_N

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    index = 0
    phase = 0
    use_acc = False
    for kk in range(0, NUM_K_ITERS):
        mbarrier.wait(ready_bars.index(index), phase)
        # This consumer's 64-row slice of the SHARED 128-row A tile.
        a_slice = a_bufs.index(index).slice(CONSUMER_ID * BLOCK_M, BLOCK_M, dim=0)
        acc = warpgroup_mma(a_slice, b_bufs.index(index), acc, is_async=True, use_acc=use_acc)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        mbarrier.arrive(empty_bars.index(index), count=1)   # one of NUM_CONSUMERS arrivals
        use_acc = True
        index += 1
        if index == num_buffers:
            index = 0
            phase ^= 1

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def ws_gemm_kernel_v2(a_desc, b_desc, c_desc, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                      BLOCK_K: gl.constexpr, num_buffers: gl.constexpr, num_warps: gl.constexpr,
                      NUM_CONSUMERS: gl.constexpr, NUM_K_ITERS: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype
    # Single shared ring: one BM_TOTAL-row A tile + one B tile per slot.
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(empty_bars.index(i), count=NUM_CONSUMERS)  # recycle after BOTH consumers
        mbarrier.init(ready_bars.index(i), count=1)
    gl.warp_specialize(
        [
            (producer_partition_v2, (a_desc, b_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, NUM_CONSUMERS)),
            (consumer_partition_v2, (c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS, 0, NUM_K_ITERS)),
            (consumer_partition_v2, (c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS, 1, NUM_K_ITERS)),
        ],
        [num_warps, num_warps],
        [232, 232],
    )


def ws_gemm_v2(A, B, C, BLOCK_M=64, BLOCK_N=256, BLOCK_K=64, num_buffers=3,
               num_warps=4, NUM_CONSUMERS=2, emit=True):
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb
    _check_tma_floor(K, N)
    BM_TOTAL = BLOCK_M * NUM_CONSUMERS

    a_layout = gl.NVMMASharedLayout.get_default_for([BM_TOTAL, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BM_TOTAL, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (_cdiv(M, BM_TOTAL), _cdiv(N, BLOCK_N))
    kw = {"emit_cuda": True} if emit else {}
    ws_gemm_kernel_v2[grid](
        a_desc, b_desc, c_desc, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers,
        num_warps=num_warps, NUM_CONSUMERS=NUM_CONSUMERS, NUM_K_ITERS=_cdiv(K, BLOCK_K), **kw)


# ===========================================================================
# V3 — L2 TILE RASTERIZATION (the 8192 fix)
# ===========================================================================
# v2 matches cuBLAS at 4096 but LOSES to it at 8192. Measured on H800 (132 SMs,
# 50 MB L2) with the gemm_06 harness method (warm-L2 batched events / cold-L2 with
# a 50 MB flush per call): v2 @8192 = 678 warm / 637 cold TFLOP/s.
#
# Root cause is L2 locality, not compute. v2 launches a 2-D grid
# (M//BM_TOTAL, N//BLOCK_N) in naive row-major order, so one wave of 132 CTAs
# spans the ENTIRE N dimension — each CTA loads a disjoint B column block and
# almost nothing is reused from L2. This is invisible at 4096 (few waves) but
# costly at 8192 (2048 tiles / 132 ≈ 16 waves). cuBLAS and gemm_06 both swizzle
# threadblocks so a wave covers a compact ~square output region, maximizing
# A-row / B-column reuse in L2 (gemm_06_autotuned.cu's `get_tile_coords`).
#
# v3 keeps v2's math verbatim but launches a 1-D grid and applies the standard
# Triton group-M rasterization: consecutive program_ids walk GROUP_M tile-rows
# DOWN before stepping right, so a wave touches only ~GROUP_M columns. The remap
# is computed identically in BOTH producer and consumer from program_id(0). At
# GROUP_M=12 this lifts 8192 to 689 warm / 662 cold (+1.5% / +3.9% over v2) — now
# matching cuBLAS warm-L2 and ~100% of stock-PTX cold-L2 through emit_cuda.
#
# The residual gap to gemm_06 is its CLUSTER TMA MULTICAST (CGA=2 broadcasts each
# B tile to both CTAs, halving HBM traffic) layered on a PERSISTENT grid with
# cross-tile epilogue overlap. v4 below adds exactly those three things and closes
# most of the gap (~96% of cuBLAS warm-L2, ~92-95% of gemm_06). A deferred-by-one
# WGMMA wait (keeping 2 wgmma.mma_async in flight) was also tried and measured
# flat-to-worse here: with num_buffers=3 the wait-every-iter loop already overlaps,
# and holding a second buffer for the in-flight MMA just steals producer headroom.
# ---------------------------------------------------------------------------
@gluon.jit
def _swizzle_tile(pid, num_pid_m, num_pid_n, GROUP_M: gl.constexpr):
    # Standard Triton group-M rasterization: walk GROUP_M rows down, then right.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def producer_partition_v3(a_desc, b_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                          BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                          num_buffers: gl.constexpr, NUM_CONSUMERS: gl.constexpr,
                          num_pid_m, num_pid_n, GROUP_M: gl.constexpr):
    K = a_desc.shape[1]
    pid_m, pid_n = _swizzle_tile(gl.program_id(0), num_pid_m, num_pid_n, GROUP_M)
    off_m = pid_m * (BLOCK_M * NUM_CONSUMERS)
    off_n = pid_n * BLOCK_N
    index = 0
    phase = 1
    for k in range(0, K, BLOCK_K):
        mbarrier.wait(empty_bars.index(index), phase)
        bar = ready_bars.index(index)
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index))
        tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index))
        index += 1
        if index == num_buffers:
            index = 0
            phase ^= 1


@gluon.jit
def consumer_partition_v3(c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                          BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                          num_buffers: gl.constexpr, num_warps: gl.constexpr,
                          NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr,
                          NUM_K_ITERS: gl.constexpr,
                          num_pid_m, num_pid_n, GROUP_M: gl.constexpr):
    dtype: gl.constexpr = c_desc.dtype
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    pid_m, pid_n = _swizzle_tile(gl.program_id(0), num_pid_m, num_pid_n, GROUP_M)
    off_m = pid_m * (BLOCK_M * NUM_CONSUMERS) + CONSUMER_ID * BLOCK_M
    off_n = pid_n * BLOCK_N

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
    index = 0
    phase = 0
    use_acc = False
    for kk in range(0, NUM_K_ITERS):
        mbarrier.wait(ready_bars.index(index), phase)
        a_slice = a_bufs.index(index).slice(CONSUMER_ID * BLOCK_M, BLOCK_M, dim=0)
        acc = warpgroup_mma(a_slice, b_bufs.index(index), acc, is_async=True, use_acc=use_acc)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        mbarrier.arrive(empty_bars.index(index), count=1)
        use_acc = True
        index += 1
        if index == num_buffers:
            index = 0
            phase ^= 1

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def ws_gemm_kernel_v3(a_desc, b_desc, c_desc, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                      BLOCK_K: gl.constexpr, num_buffers: gl.constexpr, num_warps: gl.constexpr,
                      NUM_CONSUMERS: gl.constexpr, NUM_K_ITERS: gl.constexpr,
                      num_pid_m, num_pid_n, GROUP_M: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(empty_bars.index(i), count=NUM_CONSUMERS)
        mbarrier.init(ready_bars.index(i), count=1)
    gl.warp_specialize(
        [
            (producer_partition_v3, (a_desc, b_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, NUM_CONSUMERS,
                                     num_pid_m, num_pid_n, GROUP_M)),
            (consumer_partition_v3, (c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS, 0, NUM_K_ITERS,
                                     num_pid_m, num_pid_n, GROUP_M)),
            (consumer_partition_v3, (c_desc, a_bufs, b_bufs, empty_bars, ready_bars,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS, 1, NUM_K_ITERS,
                                     num_pid_m, num_pid_n, GROUP_M)),
        ],
        [num_warps, num_warps],
        [232, 232],
    )


def ws_gemm_v3(A, B, C, BLOCK_M=64, BLOCK_N=256, BLOCK_K=64, num_buffers=3,
               num_warps=4, NUM_CONSUMERS=2, GROUP_M=12, emit=True):
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb
    _check_tma_floor(K, N)
    BM_TOTAL = BLOCK_M * NUM_CONSUMERS

    a_layout = gl.NVMMASharedLayout.get_default_for([BM_TOTAL, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BM_TOTAL, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_pid_m = _cdiv(M, BM_TOTAL)
    num_pid_n = _cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)
    kw = {"emit_cuda": True} if emit else {}
    ws_gemm_kernel_v3[grid](
        a_desc, b_desc, c_desc, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers,
        num_warps=num_warps, NUM_CONSUMERS=NUM_CONSUMERS, NUM_K_ITERS=_cdiv(K, BLOCK_K),
        num_pid_m=num_pid_m, num_pid_n=num_pid_n, GROUP_M=GROUP_M, **kw)


# ===========================================================================
# v4 — the gemm_06-class kernel: PERSISTENT tiles + cross-tile EPILOGUE OVERLAP
#      + CLUSTER_SZ=2 B-multicast. This is what actually reaches ~92-95% of the
#      hand-tuned XPUTutorial gemm_06_autotuned.cu through emit_cuda.
#
# Three structural upgrades over v3, each closing a piece of the residual gap:
#
#   (1) PERSISTENT GRID. v3 launches one CTA per output tile (num_tiles CTAs).
#       v4 launches exactly num_SMs/CLUSTER_SZ clusters and each CTA grid-strides
#       over its tiles (`while tile < num_tiles: tile += NUM_CLUSTERS`). This
#       amortizes the prologue/epilogue and keeps the WGMMA pipe hot across tiles.
#
#   (2) CROSS-TILE EPILOGUE OVERLAP. The TMA store of tile T's C is launched and
#       NOT waited on; the wait is deferred to the TOP of tile T+1's epilogue
#       (`if not first_tile: tma.store_wait`). So tile T's S2G store overlaps
#       tile T+1's entire K-loop. A final drain after the tile loop flushes the
#       last store. (This is exactly the gemm_06 `if(!first_tile) tma_store_wait`
#       at lines 788-792 + the line-831 drain — see project memory P1/P2.)
#
#   (3) CLUSTER_SZ=2 B-MULTICAST. Two CTAs form a cluster and own ADJACENT M
#       blocks of the SAME N tile, so B is identical for both. With num_ctas=2 +
#       a cluster-replicated descriptor, ONE TMA load broadcasts B into both CTAs'
#       smem (`multicast=...`), halving B's HBM traffic — the dominant lever at
#       large N. A is loaded per-CTA. To keep A sliceable by the two consumers and
#       C a private per-consumer buffer (so the epilogue overlap above still
#       works), we read the real cluster CTA rank as a scalar via inline asm
#       (`%cluster_ctarank`) + a max-reduce and offset A/C MANUALLY with normal
#       (non-cga) buffers, avoiding the cga-split smem tax that would force NB=2.
#
# Measured warm-L2 (H800, GROUP_M=8, BM=64/consumer=128 total, BN=256, NB=3):
#   v4 @4096 ≈ 740 TF, @8192 ≈ 751 TF  vs  gemm_06 CL=2  786 / 811 TF.
# All structure here is hand-written Gluon; the same shape can instead be
# introduced by the emit_cuda P1/P2/P3 IR→IR passes (see project memory).
# ===========================================================================
@gluon.constexpr_function
def pick_wgmma_layout_v4(dtype, BLOCK_M, BLOCK_N, num_warps):
    # cga_layout=[[0,0]] -> the per-CTA acc tensor is replicated across the
    # cluster (identical thread map in every CTA; per-CTA data differs and is
    # addressed manually via cta_rank).
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=get_warps_per_cta(BLOCK_M, num_warps),
        instr_shape=[16, BLOCK_N, 256 // dtype.primitive_bitwidth],
        cga_layout=[[0, 0]],
    )


@gluon.jit
def _cta_rank_v4(num_warps: gl.constexpr):
    # %cluster_ctarank is uniform across the CTA. Read it into a cluster-replicated
    # blocked vector, then max-reduce to a uniform rank-0 scalar usable as a coord.
    n: gl.constexpr = 32 * num_warps
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0], [[0]])
    z = gl.zeros([n], dtype=gl.int32, layout=layout)
    r = gl.inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;", "=r,r", [z],
                                  dtype=gl.int32, is_pure=True, pack=1)
    return gl.max(r, axis=0)


@gluon.jit
def producer_partition_v4(a_desc, b_desc, a0_bufs, a1_bufs, b_bufs, empty_bars, ready_bars,
                          cta_rank, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                          BLOCK_K: gl.constexpr, num_buffers: gl.constexpr,
                          NUM_CONSUMERS: gl.constexpr, NUM_CTAS: gl.constexpr,
                          num_warps: gl.constexpr, num_pid_m, num_pid_n, num_tiles,
                          NUM_CLUSTERS: gl.constexpr, GROUP_M: gl.constexpr,
                          MULTICAST: gl.constexpr):
    K = a_desc.shape[1]
    CTA_M: gl.constexpr = BLOCK_M * NUM_CONSUMERS              # rows this CTA owns
    CLUSTER_M: gl.constexpr = CTA_M * NUM_CTAS                 # rows the cluster owns
    index = 0
    phase = 1
    tile = gl.program_id(0)
    while tile < num_tiles:
        pid_m, pid_n = _swizzle_tile(tile, num_pid_m, num_pid_n, GROUP_M)
        off_m = pid_m * CLUSTER_M + cta_rank * CTA_M           # per-CTA M base
        off_n = pid_n * BLOCK_N                                # shared N base
        for k in range(0, K, BLOCK_K):
            mbarrier.wait(empty_bars.index(index), phase)
            bar = ready_bars.index(index)
            # A: two per-consumer 64-row tiles (a cga buffer cannot be .slice()d,
            # so each consumer owns its own buffer). B: shared, optionally multicast.
            mbarrier.expect(bar, 2 * a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a0_bufs.index(index))
            tma.async_copy_global_to_shared(a_desc, [off_m + BLOCK_M, k], bar, a1_bufs.index(index))
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(index),
                                            multicast=MULTICAST)
            index += 1
            if index == num_buffers:
                index = 0
                phase ^= 1
        tile += NUM_CLUSTERS


@gluon.jit
def consumer_partition_v4(c_desc, a_bufs, b_bufs, empty_bars, ready_bars, cta_rank,
                          BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                          num_buffers: gl.constexpr, num_warps: gl.constexpr,
                          NUM_CONSUMERS: gl.constexpr, NUM_CTAS: gl.constexpr,
                          CONSUMER_ID: gl.constexpr, NUM_K_ITERS: gl.constexpr,
                          num_pid_m, num_pid_n, num_tiles, NUM_CLUSTERS: gl.constexpr,
                          GROUP_M: gl.constexpr, EPILOGUE_OVERLAP: gl.constexpr,
                          WGMMA_PIPE: gl.constexpr):
    dtype: gl.constexpr = c_desc.dtype
    mma_layout: gl.constexpr = pick_wgmma_layout_v4(dtype, BLOCK_M, BLOCK_N, num_warps)
    CTA_M: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    CLUSTER_M: gl.constexpr = CTA_M * NUM_CTAS

    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)

    index = 0
    phase = 0
    first_tile = True
    tile = gl.program_id(0)
    while tile < num_tiles:
        pid_m, pid_n = _swizzle_tile(tile, num_pid_m, num_pid_n, GROUP_M)
        off_m = pid_m * CLUSTER_M + cta_rank * CTA_M + CONSUMER_ID * BLOCK_M
        off_n = pid_n * BLOCK_N

        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        use_acc = False
        if WGMMA_PIPE:
            # Deferred-wait pipelining: issue mma(kk) but only drain the PREVIOUS
            # group (num_outstanding=1), releasing its buffer one iteration late so
            # the current mma overlaps the next iter's barrier-wait + issue. This is
            # what gemm_06 does to keep the WGMMA pipe full; ~+1% warm / +2% cold
            # vs full-drain (num_outstanding=0) on the cluster-2 8192 GEMM.
            prev_index = 0
            for kk in range(0, NUM_K_ITERS):
                mbarrier.wait(ready_bars.index(index), phase)
                acc = warpgroup_mma(a_bufs.index(index), b_bufs.index(index), acc,
                                    is_async=True, use_acc=use_acc)
                # Unconditional wait keeps acc a consistent fp32 tensor across the
                # runtime branch; only the buffer release is deferred.
                acc = warpgroup_mma_wait(num_outstanding=1, deps=(acc,))
                if kk > 0:
                    mbarrier.arrive(empty_bars.index(prev_index), count=1)
                use_acc = True
                prev_index = index
                index += 1
                if index == num_buffers:
                    index = 0
                    phase ^= 1
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
            mbarrier.arrive(empty_bars.index(prev_index), count=1)
        else:
            for kk in range(0, NUM_K_ITERS):
                mbarrier.wait(ready_bars.index(index), phase)
                acc = warpgroup_mma(a_bufs.index(index), b_bufs.index(index), acc,
                                    is_async=True, use_acc=use_acc)
                acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
                mbarrier.arrive(empty_bars.index(index), count=1)
                use_acc = True
                index += 1
                if index == num_buffers:
                    index = 0
                    phase ^= 1

        if EPILOGUE_OVERLAP:
            # Cross-tile overlap: drain the PREVIOUS tile's store here (tile 0 has
            # nothing outstanding), then launch this tile's store without waiting.
            if not first_tile:
                tma.store_wait(pendings=0)
            c_smem.store(acc.to(dtype))
            fence_async_shared()
            tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
            first_tile = False
        else:
            c_smem.store(acc.to(dtype))
            fence_async_shared()
            tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
            tma.store_wait(pendings=0)
        tile += NUM_CLUSTERS

    if EPILOGUE_OVERLAP:
        tma.store_wait(pendings=0)


@gluon.jit
def ws_gemm_kernel_v4(a_desc, b_desc, c_desc, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                      BLOCK_K: gl.constexpr, num_buffers: gl.constexpr, num_warps: gl.constexpr,
                      NUM_CONSUMERS: gl.constexpr, NUM_CTAS: gl.constexpr, NUM_K_ITERS: gl.constexpr,
                      num_pid_m, num_pid_n, num_tiles, NUM_CLUSTERS: gl.constexpr,
                      GROUP_M: gl.constexpr, EPILOGUE_OVERLAP: gl.constexpr, MULTICAST: gl.constexpr,
                      WGMMA_PIPE: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype
    a0_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    a1_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    # Under num_ctas>1 the mbarrier MUST be cga-replicated so the consumer's
    # empty-arrive broadcasts to every cluster CTA's copy; the init count is then
    # scaled by the number of arriving CTAs: NUM_CONSUMERS per CTA * NUM_CTAS.
    bar_layout: gl.constexpr = mbarrier.MBarrierLayout(cga_layout=[[0]])
    empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], bar_layout)
    ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], bar_layout)
    for i in gl.static_range(num_buffers):
        mbarrier.init(empty_bars.index(i), count=NUM_CONSUMERS * NUM_CTAS)
        mbarrier.init(ready_bars.index(i), count=1)
    if NUM_CTAS > 1:
        # Publish the mbarrier init cluster-wide before any CTA arrives/waits.
        mbarrier.sync_cluster_init()
    cta_rank = _cta_rank_v4(num_warps)
    gl.warp_specialize(
        [
            (producer_partition_v4, (a_desc, b_desc, a0_bufs, a1_bufs, b_bufs, empty_bars, ready_bars,
                                     cta_rank, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, NUM_CONSUMERS,
                                     NUM_CTAS, num_warps, num_pid_m, num_pid_n, num_tiles,
                                     NUM_CLUSTERS, GROUP_M, MULTICAST)),
            (consumer_partition_v4, (c_desc, a0_bufs, b_bufs, empty_bars, ready_bars, cta_rank,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS,
                                     NUM_CTAS, 0, NUM_K_ITERS, num_pid_m, num_pid_n, num_tiles,
                                     NUM_CLUSTERS, GROUP_M, EPILOGUE_OVERLAP, WGMMA_PIPE)),
            (consumer_partition_v4, (c_desc, a1_bufs, b_bufs, empty_bars, ready_bars, cta_rank,
                                     BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, NUM_CONSUMERS,
                                     NUM_CTAS, 1, NUM_K_ITERS, num_pid_m, num_pid_n, num_tiles,
                                     NUM_CLUSTERS, GROUP_M, EPILOGUE_OVERLAP, WGMMA_PIPE)),
        ],
        [num_warps, num_warps],
        [232, 232],
    )


def ws_gemm_v4(A, B, C, BLOCK_M=64, BLOCK_N=256, BLOCK_K=64, num_buffers=3,
               num_warps=4, NUM_CONSUMERS=2, NUM_CTAS=2, GROUP_M=8, NUM_SMS=132,
               EPILOGUE_OVERLAP=1, MULTICAST=1, WGMMA_PIPE=1, emit=True):
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb
    _check_tma_floor(K, N)
    CTA_M = BLOCK_M * NUM_CONSUMERS
    CLUSTER_M = CTA_M * NUM_CTAS
    # A/C are per-CTA NORMAL buffers (offset by cta_rank). B is cluster-replicated
    # so one multicast load broadcasts it to both CTAs' smem.
    rep2 = [[0, 0]]
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16, cga_layout=rep2)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16, cga_layout=rep2)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16, cga_layout=rep2)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)
    num_pid_m = _cdiv(M, CLUSTER_M)
    num_pid_n = _cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    n_clusters = min(num_tiles, NUM_SMS // NUM_CTAS)
    grid = (n_clusters,)
    kw = {"emit_cuda": True} if emit else {}
    ws_gemm_kernel_v4[grid](
        a_desc, b_desc, c_desc, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers,
        num_warps=num_warps, NUM_CONSUMERS=NUM_CONSUMERS, NUM_CTAS=NUM_CTAS,
        NUM_K_ITERS=_cdiv(K, BLOCK_K), num_pid_m=num_pid_m, num_pid_n=num_pid_n,
        num_tiles=num_tiles, NUM_CLUSTERS=n_clusters, GROUP_M=GROUP_M,
        EPILOGUE_OVERLAP=EPILOGUE_OVERLAP, MULTICAST=MULTICAST,
        WGMMA_PIPE=WGMMA_PIPE,
        num_ctas=NUM_CTAS, **kw)


def ws_gemm(A, B, C, BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_buffers=3,
            num_warps=4, NUM_CONSUMERS=2, emit=True):
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb
    _check_tma_floor(K, N)

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    # Round UP — partial edge tiles are covered by TMA OOB zero-fill / store-drop.
    grid = (_cdiv(M, BLOCK_M * NUM_CONSUMERS), _cdiv(N, BLOCK_N))
    kw = {"emit_cuda": True} if emit else {}
    ws_gemm_kernel[grid](
        a_desc, b_desc, c_desc, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers,
        num_warps=num_warps, NUM_CONSUMERS=NUM_CONSUMERS, NUM_K_ITERS=_cdiv(K, BLOCK_K), **kw)


import triton.profiler as proton
import triton.profiler.viewer as proton_viewer

# Proton (CUPTI) timing.  Unlike a host-side perf_counter wall-clock, Proton
# reads each kernel's on-GPU duration straight from CUPTI, so the number is the
# true device time with no Python launch / synchronize overhead folded in — the
# same kind of measurement gemm_06's own harness uses.
_PROTON_FILE = "tut01"


def _proton_run(fn, scope, reps, warmup):
    """Launch `fn` under a uniquely-named Proton scope `reps` times (after
    `warmup` un-profiled launches).  Time accumulates into that scope."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    proton.activate()
    for _ in range(reps):
        with proton.scope(scope):
            fn()
    proton.deactivate()


def _proton_times(filename):
    """Parse the Proton hatchet and return {scope_name: total_device_ms}."""
    gf, metrics = proton_viewer.parse(["time/ms"], filename)
    df = gf.dataframe
    col = metrics[0]  # "time/ms (inc)"
    return {name: float(t) for name, t in zip(df["name"], df[col])}


def _load_official_gluon():
    """Import Triton's own best Hopper Gluon GEMM as an external reference:
    `persistent_matmul` + the grouped persistent tile scheduler from
    python/tutorials/gluon/07-persistence.py (pure Gluon → stock PTX, no
    emit_cuda).  Returns run(A,B,C) or None if it can't be loaded."""
    import importlib
    import sys
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gdir = os.path.join(repo, "python", "tutorials", "gluon")
    if not os.path.isdir(gdir):
        return None
    if gdir not in sys.path:
        sys.path.insert(0, gdir)
    try:
        mod = importlib.import_module("07-persistence")
        sched = mod.GroupedPersistentTileScheduler(8)  # GROUP_SIZE_M=8
    except Exception as e:  # missing API / Blackwell-only import, etc.
        print(f"  (official Gluon GEMM unavailable: {e})")
        return None

    # Hopper config used by 07's grouped-scheduler benchmark.
    def run(A, B, C):
        mod.persistent_matmul(A, B, C, 128, 256, 64, 3, 8, sched)

    return run


def main():
    if not is_hopper():
        raise RuntimeError("This tutorial requires a Hopper (sm90) GPU")
    torch.manual_seed(0)
    emit = os.environ.get("EMIT", "1") == "1"
    print(f"backend = {'emit_cuda' if emit else 'stock PTX'}")

    # --- Correctness: ARBITRARY shapes, NO tile-alignment assumption ---
    # All four kernels are pure TMA, so any problem shape is handled by OOB
    # zero-fill (loads) / store-drop (stores) — we test tiny, huge, and shapes
    # that are deliberately NOT multiples of the tile (BLOCK_M=64/BLOCK_N=256/
    # BLOCK_K=64) or the cluster (CLUSTER_M=256). The only floor is the TMA fp16
    # hardware requirement K%8==0 and N%8==0 (16-byte global row stride); M is
    # fully arbitrary (we use odd / prime-ish M on purpose).
    shapes = [
        (8, 8, 8),         # minimal
        (1, 256, 256),     # single output row
        (17, 24, 40),      # tiny, M odd, none tile-aligned
        (200, 136, 72),    # small, all misaligned to tile
        (64, 256, 256),    # exactly one v1/v2/v3 tile, one v4 half-cluster
        (255, 257 - 1, 511 - 7),  # = (255, 256, 504): M arbitrary, mid misaligned
        (320, 768, 256),   # M not cluster-aligned (320 = 1.25*CLUSTER_M)
        (513, 1024, 1024),  # M odd, large-ish, N/K aligned
        (1000, 1024, 1000),  # M,K misaligned to tile, all %8
        (4096, 4096, 4096),  # large aligned
        (4097, 4104, 4096),  # large, M arbitrary + N misaligned to tile (4104%256!=0)
    ]
    # Each cell shows two checks:
    #   tol  — emit_cuda result vs an fp16 torch matmul reference (atol/rtol 1e-2)
    #   bit  — is emit_cuda BITWISE-identical to a trusted same-math reference?
    #          "=" identical, "~" close-but-not-exact.
    # (torch's own matmul is never bit-identical to ours — different accumulation
    #  order — so bitwise parity is measured against a same-kernel reference.)
    #
    # Bitwise reference choice per version:
    #   v1/v2/v3 are plain (num_ctas=1) kernels, so we lower the SAME kernel
    #   through the stock PTX backend (emit=False) and demand bit-identical output.
    #   v4 launches a num_ctas=2 CLUSTER kernel; the stock PTX backend deadlocks on
    #   that cluster launch under CUDA 12.9 (GPU spins at 100% util — a driver/ptxas
    #   issue, not an emit_cuda bug), so we do NOT run v4 through PTX.  Instead we
    #   compare emit_cuda v4 bitwise against emit_cuda v1: all four versions share
    #   the exact same per-element fp32 K-reduction (BLOCK_K=64 chunks), so a
    #   bit-identical v1 (itself PTX-verified) is a sound reference for v4.
    cases = [("v1", ws_gemm), ("v2", ws_gemm_v2), ("v3", ws_gemm_v3), ("v4", ws_gemm_v4)]
    print("correctness (arbitrary shapes — tiny / huge / misaligned)")
    print("  cell = <tol vs torch>/<bitwise>   ('=' bit-identical, '~' not)")
    print("  bitwise ref: v1/v2/v3 vs stock PTX backend; v4 vs emit_cuda v1")
    print(f"  {'shape':>16} | " + " | ".join(f"{n:>7}" for n, _ in cases))
    all_tol = True
    all_bit = True
    for M, N, K in shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ref = (A @ B).to(torch.float16)
        cells = []
        v1_ref = None  # emit_cuda v1 output, used as v4's bitwise reference
        for name, fn in cases:
            Ce = torch.empty(M, N, device="cuda", dtype=torch.float16)
            fn(A, B, Ce, emit=True)    # emit_cuda (CUDA C++ backend)
            if name == "v4":
                # stock PTX backend deadlocks on the num_ctas=2 cluster launch in
                # CUDA 12.9 — compare against emit_cuda v1 (same math, PTX-verified).
                torch.cuda.synchronize()
                bit = torch.equal(Ce, v1_ref)
            else:
                Cp = torch.empty(M, N, device="cuda", dtype=torch.float16)
                fn(A, B, Cp, emit=False)   # stock PTX backend (same kernel)
                torch.cuda.synchronize()
                bit = torch.equal(Ce, Cp)
            if name == "v1":
                v1_ref = Ce
            tol = torch.allclose(Ce.float(), ref.float(), atol=1e-2, rtol=1e-2)
            all_tol = all_tol and tol
            all_bit = all_bit and bit
            cells.append(f"{'OK' if tol else 'FAIL':>4}{'=' if bit else '~'}")
        print(f"  {f'{M}x{N}x{K}':>16} | " + " | ".join(f"{c:>7}" for c in cells))
    print(f"  => tolerance: {'ALL PASS' if all_tol else 'SOME FAILED'}"
          f"   |   bitwise: {'ALL IDENTICAL' if all_bit else 'SOME DIFFER'}")

    # --- Performance ---
    # Timed with Proton (CUPTI): each launch runs inside a named Proton scope and
    # we read the accumulated on-GPU duration back from the hatchet profile.
    # We compare, all on the SAME timer:
    #   * emit_cuda  v1-v4   (CUDA C++ backend)
    #   * stock PTX  v1-v4   (the SAME kernels lowered through Triton's PTX backend;
    #                         v4 is the num_ctas=2 cluster kernel, which deadlocks
    #                         on PTX under CUDA 12.9 — skipped, shown as "—")
    #   * cuBLAS  (torch.matmul)
    #   * Triton's own best Hopper Gluon GEMM (07 grouped-persistent) as an
    #     external reference of where the stock teaching kernels land.
    perf_shapes = [(4096, 4096, 4096), (8192, 8192, 8192)]
    perf_kernels = [("v1", ws_gemm), ("v2", ws_gemm_v2), ("v3", ws_gemm_v3),
                    ("v4", ws_gemm_v4)]
    # Warm each kernel before profiling so it is measured at a steady-state clock;
    # average over many reps to damp the residual power-throttle jitter that a
    # shared GPU shows on short bursty launch loops.
    REPS, WARMUP = 100, 30
    gluon_run = _load_official_gluon()

    proton.start(_PROTON_FILE, hook="triton")
    proton.deactivate()  # correctness above + warmups below stay un-profiled
    for M, N, K in perf_shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)
        for vname, fn in perf_kernels:
            # emit_cuda backend
            _proton_run(lambda fn=fn: fn(A, B, C, emit=True),
                        f"emit_{vname}_{M}x{N}x{K}", REPS, WARMUP)
            # stock PTX backend (skip v4: its cluster launch deadlocks on PTX/cu12.9)
            if vname != "v4":
                _proton_run(lambda fn=fn: fn(A, B, C, emit=False),
                            f"ptx_{vname}_{M}x{N}x{K}", REPS, WARMUP)
        _proton_run(lambda: torch.matmul(A, B), f"cublas_{M}x{N}x{K}", REPS, WARMUP)
        if gluon_run is not None:
            _proton_run(lambda: gluon_run(A, B, C), f"gluon_{M}x{N}x{K}", REPS, WARMUP)
    proton.finalize()

    times = _proton_times(f"{_PROTON_FILE}.hatchet")  # {scope: total_device_ms}

    def _tf(tag, M, N, K):
        key = f"{tag}_{M}x{N}x{K}"
        if key not in times:
            return None
        ms = times[key] / REPS  # per-launch device ms
        return (2.0 * M * N * K) / (ms * 1e-3) / 1e12

    def _cell(v):
        return "      —" if v is None else f"{v:7.1f}"

    print("performance (TFLOP/s, Proton/CUPTI device time):")
    print(f"  {'shape':>11} | {'backend':>9} | {'v1':>7} | {'v2':>7} | {'v3':>7} | {'v4':>7}"
          f" | {'cuBLAS':>7} | {'Gluon*':>7}")
    print("  (* = Triton's best Hopper Gluon GEMM, 07 grouped-persistent; reference)")
    for M, N, K in perf_shapes:
        shp = f"{M}x{N}x{K}"
        fb = _tf("cublas", M, N, K)
        fg = _tf("gluon", M, N, K)
        emit_row = [_tf(f"emit_{v}", M, N, K) for v, _ in perf_kernels]
        ptx_row = [_tf(f"ptx_{v}", M, N, K) for v, _ in perf_kernels]
        print(f"  {shp:>11} | {'emit_cuda':>9} | " + " | ".join(_cell(x) for x in emit_row)
              + f" | {_cell(fb)} | {_cell(fg)}")
        print(f"  {'':>11} | {'stock PTX':>9} | " + " | ".join(_cell(x) for x in ptx_row)
              + f" | {'      —'} | {'      —'}")


if __name__ == "__main__":
    main()
