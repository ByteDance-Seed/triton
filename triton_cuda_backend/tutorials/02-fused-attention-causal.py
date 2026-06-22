"""
Tutorial 02 — Causal FlashAttention forward (Gluon frontend → emit_cuda → CUDA)
===============================================================================

GOAL
----
Express a Hopper (sm90) **causal** FlashAttention-2/3 forward pass in the *Gluon*
frontend and lower it through the out-of-tree CUDA backend (``emit_cuda``), matching
the canonical Hopper FA3 design: TMA-loaded Q/K/V, WGMMA QK^T and P@V, register
online softmax, causal masking, and a TMA epilogue. Correctness and performance are
checked against Triton's own built-in FlashAttention (tutorial ``06-fused-attention``).

THREE STAGES (each a measured, prominent performance step over the previous one)
-------------------------------------------------------------------------------
  * **v1 — warp-specialized 1-producer / 2-consumer baseline.** ONE producer
    warpgroup TMA-loads Q once and runs a kStages=2 K/V pipeline; TWO consumer
    (math) warpgroups each own half the Q row-tile (the "m-split"). Establishes
    the full FA3 structure — TMA g2s, WGMMA QK^T via ``k.permute((1,0))``, register
    online softmax (exp2/row-max/row-sum/rescale), RS-WGMMA P@V with P as the
    register A operand, TMA s2g epilogue — and the numerics.

  * **v2 — mask-split + dynamic work-stealing scheduler.** Two improvements over
    v1: (1) the inner loop is split into a mask-free body (blocks strictly below
    the diagonal) and a masked body (the diagonal blocks), so the common case
    skips the per-element causal compare; (2) a GLOBAL atomicAdd work-stealing
    counter replaces static tile partition, fixing the causal load-imbalance
    (per-tile cost grows with row index).

  * **v3 — FFMA-fused softmax (final).** v2 serializes QK→softmax→P@V, so the
    softmax sits on the critical path between the two WGMMAs and idles the tensor
    core. An ncu opcode diff showed v2 emitted ~18M FMUL+FADD that a peak FA3
    kernel fuses into FFMA: v2 scaled qk and reused the scaled value for both the
    row-max and the `qk - m` subtract, blocking FMA contraction. Taking the max on
    raw qk and writing `qk*qk_scale - m` as one single-use expression lets ptxas
    contract to FFMA, shortening the serial softmax and raising tensor-pipe
    occupancy — the fastest of the three across all measured shapes.

THE TWO MATMULS AND THEIR OPERAND FORMS
---------------------------------------
  QK^T :  S[M,N] = Q[M,D] · K[N,D]^T      contracts over D (head_dim)
          Q and K both live in SMEM with the contiguous dim = D, so K must be
          *transposed* to act as the WGMMA B operand [D, N]. Gluon expresses this
          with ``k_smem.permute((1, 0))`` — the emitter sets the wgmma ``.trans``
          bit on operand B.

  P@V  :  O[M,D] = P[M,N] · V[N,D]        contracts over N (kv block)
          P (softmax result) lives in *registers* (the qk accumulator's NVMMA
          distributed layout). To feed it as the WGMMA A operand it is converted to
          ``DotOperandLayout(operand_index=0, parent=o_layout, k_width=2)``. V is a
          plain SMEM B operand.

RUN
---
    docker exec -w /root/share/triton-cuda/triton_cuda_backend/tutorials \
      -e PYTHONPATH=/root/share/triton-cuda/python \
      -e PATH=/usr/local/cuda/bin:$PATH \
      -e TRITON_EMIT_CUDA=1 -e TRITON_CUDA_DUMP=/tmp/t02_cu \
      <container> python 02-fused-attention-causal.py
"""

import math
import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor as TLTensorDescriptor
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma, mbarrier, fence_async_shared, warpgroup_mma, warpgroup_mma_wait,
)

LOG2E = 1.4426950408889634


def is_hopper():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


def _cdiv(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# WGMMA layout helpers (constexpr so the dep-finder treats them as compile-time).
# ---------------------------------------------------------------------------
@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, num_warps):
    wpc = [4, 1]
    while wpc[0] * wpc[1] != num_warps:
        if BLOCK_M > 16 * wpc[0]:
            wpc[0] *= 2
        else:
            wpc[1] *= 2
    return wpc


@gluon.constexpr_function
def mma_layout(dtype, BLOCK_M, N, num_warps):
    # NVMMA distributed (WGMMA) accumulator layout for an [BLOCK_M, N] output.
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=get_warps_per_cta(BLOCK_M, num_warps),
        instr_shape=[16, N, 256 // dtype.primitive_bitwidth],
    )


@gluon.constexpr_function
def dot_operand_layout(parent, dtype):
    # Register A operand layout for the P@V WGMMA (P held in registers).
    return gl.DotOperandLayout(operand_index=0, parent=parent,
                               k_width=32 // dtype.primitive_bitwidth)




# ===========================================================================
# V1 — WARP-SPECIALIZED 1-PRODUCER / 2-CONSUMER (canonical FA3 structure)
# ===========================================================================
# A peak Hopper FA3 forward runs ONE producer warpgroup that feeds TWO consumer
# (math) warpgroups, each owning half the Q row-tile (the "m-split": consumer 0 →
# rows [0,64), consumer 1 → [64,128)). This is the foundational structure every
# later version keeps. The producer TMA-loads a single BM_TOTAL(=128)-row Q tile
# once and runs a kStages=2 K/V pipeline; both consumers read the SAME K/V buffers
# (each slices its own 64-row Q), so producer HBM traffic is shared. It carries the
# online-softmax recurrence (exp2/row-max/row-sum/rescale) and a causal mask.
#
# CAUSAL + SHARED PIPELINE. The two consumers attend different Q rows, so their
# causal key ranges differ by one BLOCK_N at the diagonal. Because both consumers
# must `arrive` on every K/V buffer the producer fills (empty-bar count=2), they
# iterate the SAME n-range = [0, (pid+1)*BM_TOTAL); consumer 0 simply masks the
# extra diagonal block to all -inf (a no-op: p=0, alpha=1, acc unchanged). Block 0
# is never fully masked, so no 0/0 (exp2(m_i-m_ij) with m_i=m_ij=-inf) can arise.
# Every block is masked here for structural clarity; v2 adds the mask-split that
# skips the compare on below-diagonal blocks.
@gluon.jit
def _attn_producer_v1(desc_q, desc_k, desc_v, q_smem, k_bufs, v_bufs,
                   q_ready, kv_empty, kv_ready, N_CTX, H_Q, GROUP,
                   BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                   kStages: gl.constexpr, NUM_CONSUMERS: gl.constexpr):
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    pid_m = gl.program_id(0)
    off_hz = gl.program_id(1)              # = z * H_Q + h_q  (query head)
    seq_base = off_hz * N_CTX              # Q rows (H_Q query heads)
    # GQA: K/V live under FEWER heads. The query head h_q reads the kv head
    # h_kv = h_q // GROUP; for MHA (GROUP == 1) kv_seq_base == seq_base.
    H_KV = H_Q // GROUP
    kv_off_hz = (off_hz // H_Q) * H_KV + (off_hz % H_Q) // GROUP
    kv_seq_base = kv_off_hz * N_CTX
    off_m_total = pid_m * BM_TOTAL

    # Q: one BM_TOTAL-row tile for BOTH consumers, loaded once.
    mbarrier.expect(q_ready, desc_q.block_type.nbytes)
    tma.async_copy_global_to_shared(desc_q, [seq_base + off_m_total, 0], q_ready, q_smem)

    # K/V pipeline over causal key range [0, off_m_total + BM_TOTAL).
    hi = off_m_total + BM_TOTAL
    index = 0
    phase = 1
    for start_n in range(0, hi, BLOCK_N):
        mbarrier.wait(kv_empty.index(index), phase)
        bar = kv_ready.index(index)
        mbarrier.expect(bar, desc_k.block_type.nbytes + desc_v.block_type.nbytes)
        tma.async_copy_global_to_shared(desc_k, [kv_seq_base + start_n, 0], bar, k_bufs.index(index))
        tma.async_copy_global_to_shared(desc_v, [kv_seq_base + start_n, 0], bar, v_bufs.index(index))
        index += 1
        if index == kStages:
            index = 0
            phase ^= 1


@gluon.jit
def _attn_consumer_v1(desc_o, q_smem, k_bufs, v_bufs, q_ready, kv_empty, kv_ready,
                   N_CTX, sm_scale,
                   BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                   kStages: gl.constexpr, num_warps: gl.constexpr,
                   NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr):
    dtype: gl.constexpr = desc_o.dtype
    qk_scale = sm_scale * 1.4426950408889634   # log2(e)
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS

    # WGMMA k-dim is set by the bf16 *operand* bitwidth (256//16=16), not the f32 acc.
    qk_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    o_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, HEAD_DIM, num_warps)
    p_layout: gl.constexpr = dot_operand_layout(o_layout, dtype)
    row_qk: gl.constexpr = gl.SliceLayout(1, qk_layout)
    row_o: gl.constexpr = gl.SliceLayout(1, o_layout)
    col_qk: gl.constexpr = gl.SliceLayout(0, qk_layout)

    pid_m = gl.program_id(0)
    off_hz = gl.program_id(1)
    seq_base = off_hz * N_CTX                             # stacked (batch,head) row base
    off_m_total = pid_m * BM_TOTAL
    off_m = off_m_total + CONSUMER_ID * BLOCK_M          # this consumer's 64 rows (seq-relative)

    # Wait for Q, then take this consumer's 64-row slice.
    mbarrier.wait(q_ready, 0)
    q_slice = q_smem.slice(CONSUMER_ID * BLOCK_M, BLOCK_M, dim=0)

    m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=row_qk)
    l_i = gl.full([BLOCK_M], 1.0, gl.float32, layout=row_qk)
    acc = gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, layout=o_layout)
    offs_m = off_m + gl.arange(0, BLOCK_M, layout=row_qk)   # seq-relative, for causal mask

    hi = off_m_total + BM_TOTAL
    index = 0
    phase = 0
    for start_n in range(0, hi, BLOCK_N):
        mbarrier.wait(kv_ready.index(index), phase)

        qk = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
        qk = warpgroup_mma(q_slice, k_bufs.index(index).permute((1, 0)), qk,
                           use_acc=False, is_async=True)
        qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))

        offs_n = start_n + gl.arange(0, BLOCK_N, layout=col_qk)
        mask = offs_m[:, None] >= offs_n[None, :]
        qk = gl.where(mask, qk * qk_scale, float("-inf"))

        m_ij = gl.maximum(m_i, gl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = gl.exp2(qk)
        l_ij = gl.sum(p, axis=1)
        alpha = gl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        alpha_o = gl.convert_layout(alpha, row_o)
        acc = acc * alpha_o[:, None]
        p_dot = gl.convert_layout(p.to(dtype), p_layout)
        acc = warpgroup_mma(p_dot, v_bufs.index(index), acc, use_acc=True, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

        mbarrier.arrive(kv_empty.index(index), count=1)   # one of NUM_CONSUMERS arrivals
        m_i = m_ij
        index += 1
        if index == kStages:
            index = 0
            phase ^= 1

    l_o = gl.convert_layout(l_i, row_o)
    acc = acc * (1.0 / l_o)[:, None]
    o_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], desc_o.layout)
    o_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(desc_o, [seq_base + off_m, 0], o_smem)
    tma.store_wait(pendings=0)


@gluon.jit
def attn_fwd_v1(desc_q, desc_k, desc_v, desc_o, N_CTX, sm_scale, H_Q, GROUP,
                BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                kStages: gl.constexpr, num_warps: gl.constexpr, NUM_CONSUMERS: gl.constexpr):
    dtype: gl.constexpr = desc_q.dtype
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS

    q_smem = gl.allocate_shared_memory(dtype, [BM_TOTAL, HEAD_DIM], desc_q.layout)
    k_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_k.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_v.layout)
    q_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    kv_empty = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    kv_ready = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    mbarrier.init(q_ready, count=1)
    for i in gl.static_range(kStages):
        mbarrier.init(kv_empty.index(i), count=NUM_CONSUMERS)  # recycled after BOTH consumers
        mbarrier.init(kv_ready.index(i), count=1)

    gl.warp_specialize(
        [
            (_attn_producer_v1, (desc_q, desc_k, desc_v, q_smem, k_bufs, v_bufs,
                              q_ready, kv_empty, kv_ready, N_CTX, H_Q, GROUP,
                              BLOCK_M, BLOCK_N, HEAD_DIM, kStages, NUM_CONSUMERS)),
            (_attn_consumer_v1, (desc_o, q_smem, k_bufs, v_bufs, q_ready, kv_empty, kv_ready,
                              N_CTX, sm_scale, BLOCK_M, BLOCK_N, HEAD_DIM, kStages,
                              num_warps, NUM_CONSUMERS, 0)),
            (_attn_consumer_v1, (desc_o, q_smem, k_bufs, v_bufs, q_ready, kv_empty, kv_ready,
                              N_CTX, sm_scale, BLOCK_M, BLOCK_N, HEAD_DIM, kStages,
                              num_warps, NUM_CONSUMERS, 1)),
        ],
        [num_warps, num_warps],
        [240, 240],
    )


def attention_v1(q, k, v, sm_scale=None, BLOCK_M=64, BLOCK_N=128, kStages=2,
                 num_warps=4, NUM_CONSUMERS=2, emit=True):
    """q: [Z, H_Q, N_CTX, HEAD_DIM] bf16; k, v: [Z, H_KV, N_CTX, HEAD_DIM] bf16
    (GQA when H_KV < H_Q, MHA when H_KV == H_Q). Causal. 1 producer + 2 consumer WGs."""
    Z, H_Q, N_CTX, HEAD_DIM = q.shape
    H_KV = k.shape[1]
    assert k.shape == (Z, H_KV, N_CTX, HEAD_DIM) and v.shape == k.shape
    assert H_Q % H_KV == 0, "H_Q must be a multiple of H_KV (GQA)"
    GROUP = H_Q // H_KV
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    BM_TOTAL = BLOCK_M * NUM_CONSUMERS

    q2 = q.reshape(Z * H_Q * N_CTX, HEAD_DIM)
    k2 = k.reshape(Z * H_KV * N_CTX, HEAD_DIM)
    v2 = v.reshape(Z * H_KV * N_CTX, HEAD_DIM)
    o = torch.empty_like(q)
    o2 = o.reshape(Z * H_Q * N_CTX, HEAD_DIM)

    ql = gl.NVMMASharedLayout.get_default_for([BM_TOTAL, HEAD_DIM], gl.bfloat16)
    kl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    vl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    ol = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], gl.bfloat16)
    desc_q = TensorDescriptor.from_tensor(q2, [BM_TOTAL, HEAD_DIM], ql)
    desc_k = TensorDescriptor.from_tensor(k2, [BLOCK_N, HEAD_DIM], kl)
    desc_v = TensorDescriptor.from_tensor(v2, [BLOCK_N, HEAD_DIM], vl)
    desc_o = TensorDescriptor.from_tensor(o2, [BLOCK_M, HEAD_DIM], ol)

    grid = (_cdiv(N_CTX, BM_TOTAL), Z * H_Q)
    kw = {"emit_cuda": True} if emit else {}
    attn_fwd_v1[grid](
        desc_q, desc_k, desc_v, desc_o, N_CTX, sm_scale, H_Q, GROUP,
        BLOCK_M, BLOCK_N, HEAD_DIM, kStages, num_warps=num_warps,
        NUM_CONSUMERS=NUM_CONSUMERS, **kw)
    return o


# ===========================================================================
# V2 — MASK-SPLIT + DYNAMIC WORK-STEALING scheduler (atomicAdd structure)
# ===========================================================================
# Two improvements over v1:
#
# (1) MASK-SPLIT. v1 applies the causal compare on every K/V block. v2 splits the
#     inner loop into a mask-FREE body (blocks strictly below the diagonal — the
#     common case for large N) and a masked body (only the diagonal blocks), so the
#     per-element `where(mask, ...)` runs on a handful of blocks instead of all.
#
# (2) DYNAMIC WORK-STEALING. A static round-robin tile partition load-imbalances
#     because causal masking makes per-tile cost grow with row index: a CTA dealt
#     all-late (heavy) tiles straggles while one dealt all-early (light) tiles idle.
#     A peak FA3 kernel fixes this with a GLOBAL work-stealing counter: each CTA
# starts on `pid = blockIdx.x`, then steals the next tile via `atomicAdd(counter,1)+gridDim.x`.
# The steal order is non-deterministic, so the producer (which alone does the atomic)
# must BROADCAST each stolen pid to its consumer warpgroups through shared memory +
# an mbarrier handshake (a producer-done work_info publish). Consumers
# peel a prologue receive and re-receive at the end of every body; the producer
# publishes one SENTINEL pid (>= num_tiles) after its loop so the `while pid<num_tiles`
# test fails and consumers exit (Triton structured CF has no mid-body break).
@gluon.constexpr_function
def sched_bcast_layout(num_warps):
    return gl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32],
                            warps_per_cta=[num_warps], order=[0])


@gluon.jit
def _attn_producer_v2(desc_q, desc_k, desc_v, q_bufs, k_bufs, v_bufs,
                       q_ready, q_empty, kv_empty, kv_ready,
                       sched_smem, sched_ready, sched_empty, counter_ptr, N_CTX, H_Q, GROUP,
                       num_pid_m, num_tiles, NUM_SMS,
                       BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                       kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                       NUM_CONSUMERS: gl.constexpr):
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    BCAST: gl.constexpr = 32 * num_warps
    bl: gl.constexpr = sched_bcast_layout(num_warps)

    sm_id = gl.program_id(0)
    pid = sm_id
    qt = 0
    kidx = 0
    kph = 1   # empty-barrier wait starts at phase 1 (fresh barrier passes immediately)
    sph = 1   # sched_empty wait starts at phase 1 (mirror of kv_empty idiom)
    while pid < num_tiles:
        # --- broadcast this stolen tile id to the consumer warpgroups ---
        mbarrier.wait(sched_empty, sph)
        sched_smem.store(gl.full([BCAST], pid, gl.int32, layout=bl))
        gl.barrier()   # collective store completes before release-arrive
        mbarrier.arrive(sched_ready, count=1)
        sph ^= 1

        pid_m = pid % num_pid_m
        off_hz = pid // num_pid_m            # = z * H_Q + h_q  (query head)
        seq_base = off_hz * N_CTX            # Q rows (H_Q query heads)
        # GQA: query head h_q reads kv head h_kv = h_q // GROUP; for MHA
        # (GROUP == 1) kv_seq_base == seq_base.
        H_KV = H_Q // GROUP
        kv_off_hz = (off_hz // H_Q) * H_KV + (off_hz % H_Q) // GROUP
        kv_seq_base = kv_off_hz * N_CTX
        off_m_total = pid_m * BM_TOTAL

        # Q double-buffer: wait this slot is free, then TMA-load BM_TOTAL rows.
        qidx = qt % QSTAGES
        qeph = ((qt // QSTAGES) % 2) ^ 1
        mbarrier.wait(q_empty.index(qidx), qeph)
        mbarrier.expect(q_ready.index(qidx), desc_q.block_type.nbytes)
        tma.async_copy_global_to_shared(desc_q, [seq_base + off_m_total, 0],
                                        q_ready.index(qidx), q_bufs.index(qidx))
        qt += 1

        # K/V pipeline over causal key range, ring continuous across tiles.
        hi = off_m_total + BM_TOTAL
        for start_n in range(0, hi, BLOCK_N):
            mbarrier.wait(kv_empty.index(kidx), kph)
            bar = kv_ready.index(kidx)
            mbarrier.expect(bar, desc_k.block_type.nbytes + desc_v.block_type.nbytes)
            tma.async_copy_global_to_shared(desc_k, [kv_seq_base + start_n, 0], bar, k_bufs.index(kidx))
            tma.async_copy_global_to_shared(desc_v, [kv_seq_base + start_n, 0], bar, v_bufs.index(kidx))
            kidx += 1
            if kidx == kStages:
                kidx = 0
                kph ^= 1

        # steal the next tile (atomicAdd(counter,1) + gridDim.x).
        # NOTE: keep this at the END of the body — hoisting it before the K/V
        # loop makes the GPU-scoped atomic contend with K/V TMA issue on the
        # producer warp and REGRESSES small-N (b2h16s2048 0.84->0.76, measured).
        pid = gl.atomic_add(counter_ptr, 1, sem="relaxed", scope="gpu") + NUM_SMS

    # sentinel publish (pid >= num_tiles) so consumers terminate
    mbarrier.wait(sched_empty, sph)
    sched_smem.store(gl.full([BCAST], pid, gl.int32, layout=bl))
    gl.barrier()
    mbarrier.arrive(sched_ready, count=1)


@gluon.jit
def _attn_consumer_v2(desc_o, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                       sched_smem, sched_ready, sched_empty,
                       N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                       BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                       kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                       NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr):
    dtype: gl.constexpr = desc_o.dtype
    qk_scale = sm_scale * 1.4426950408889634
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    bl: gl.constexpr = sched_bcast_layout(num_warps)

    qk_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    o_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, HEAD_DIM, num_warps)
    p_layout: gl.constexpr = dot_operand_layout(o_layout, dtype)
    row_qk: gl.constexpr = gl.SliceLayout(1, qk_layout)
    row_o: gl.constexpr = gl.SliceLayout(1, o_layout)
    col_qk: gl.constexpr = gl.SliceLayout(0, qk_layout)

    o_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], desc_o.layout)

    qt = 0
    kidx = 0
    kph = 0
    sph = 0
    # prologue: receive the first tile id from the producer
    mbarrier.wait(sched_ready, sph)
    pid = gl.max(sched_smem.load(bl), axis=0)
    mbarrier.arrive(sched_empty, count=1)
    sph ^= 1

    while pid < num_tiles:
        pid_m = pid % num_pid_m
        off_hz = pid // num_pid_m
        seq_base = off_hz * N_CTX
        off_m_total = pid_m * BM_TOTAL
        off_m = off_m_total + CONSUMER_ID * BLOCK_M

        qidx = qt % QSTAGES
        qph = (qt // QSTAGES) % 2
        mbarrier.wait(q_ready.index(qidx), qph)
        q_slice = q_bufs.index(qidx).slice(CONSUMER_ID * BLOCK_M, BLOCK_M, dim=0)

        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=row_qk)
        l_i = gl.full([BLOCK_M], 1.0, gl.float32, layout=row_qk)
        acc = gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, layout=o_layout)
        offs_m = off_m + gl.arange(0, BLOCK_M, layout=row_qk)

        hi = off_m_total + BM_TOTAL
        mask_start = ((off_m + 1) // BLOCK_N) * BLOCK_N

        # --- mask-free loop: blocks strictly below the diagonal ---
        for start_n in range(0, mask_start, BLOCK_N):
            mbarrier.wait(kv_ready.index(kidx), kph)
            qk = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
            qk = warpgroup_mma(q_slice, k_bufs.index(kidx).permute((1, 0)), qk,
                               use_acc=False, is_async=True)
            qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))

            qk = qk * qk_scale
            m_ij = gl.maximum(m_i, gl.max(qk, axis=1))
            qk = qk - m_ij[:, None]
            p = gl.exp2(qk)
            l_ij = gl.sum(p, axis=1)
            alpha = gl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            alpha_o = gl.convert_layout(alpha, row_o)
            acc = acc * alpha_o[:, None]
            p_dot = gl.convert_layout(p.to(dtype), p_layout)
            acc = warpgroup_mma(p_dot, v_bufs.index(kidx), acc, use_acc=True, is_async=True)
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

            mbarrier.arrive(kv_empty.index(kidx), count=1)
            m_i = m_ij
            kidx += 1
            if kidx == kStages:
                kidx = 0
                kph ^= 1

        # --- masked loop: the diagonal blocks ---
        for start_n in range(mask_start, hi, BLOCK_N):
            mbarrier.wait(kv_ready.index(kidx), kph)
            qk = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
            qk = warpgroup_mma(q_slice, k_bufs.index(kidx).permute((1, 0)), qk,
                               use_acc=False, is_async=True)
            qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))

            offs_n = start_n + gl.arange(0, BLOCK_N, layout=col_qk)
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = gl.where(mask, qk * qk_scale, float("-inf"))
            m_ij = gl.maximum(m_i, gl.max(qk, axis=1))
            qk = qk - m_ij[:, None]
            p = gl.exp2(qk)
            l_ij = gl.sum(p, axis=1)
            alpha = gl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            alpha_o = gl.convert_layout(alpha, row_o)
            acc = acc * alpha_o[:, None]
            p_dot = gl.convert_layout(p.to(dtype), p_layout)
            acc = warpgroup_mma(p_dot, v_bufs.index(kidx), acc, use_acc=True, is_async=True)
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

            mbarrier.arrive(kv_empty.index(kidx), count=1)
            m_i = m_ij
            kidx += 1
            if kidx == kStages:
                kidx = 0
                kph ^= 1

        l_o = gl.convert_layout(l_i, row_o)
        acc = acc * (1.0 / l_o)[:, None]
        o_smem.store(acc.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(desc_o, [seq_base + off_m, 0], o_smem)
        tma.store_wait(pendings=0)
        mbarrier.arrive(q_empty.index(qidx), count=1)
        qt += 1

        # receive the next stolen tile id (sentinel >= num_tiles exits the loop)
        mbarrier.wait(sched_ready, sph)
        pid = gl.max(sched_smem.load(bl), axis=0)
        mbarrier.arrive(sched_empty, count=1)
        sph ^= 1


@gluon.jit
def attn_fwd_v2(desc_q, desc_k, desc_v, desc_o, counter_ptr, N_CTX, sm_scale, H_Q, GROUP,
                 num_pid_m, num_tiles, NUM_SMS,
                 BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                 kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                 NUM_CONSUMERS: gl.constexpr):
    dtype: gl.constexpr = desc_q.dtype
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    BCAST: gl.constexpr = 32 * num_warps

    q_bufs = gl.allocate_shared_memory(dtype, [QSTAGES, BM_TOTAL, HEAD_DIM], desc_q.layout)
    k_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_k.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_v.layout)
    sched_smem = gl.allocate_shared_memory(gl.int32, [BCAST],
                                           gl.SwizzledSharedLayout(1, 1, 1, order=[0]))
    q_ready = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    kv_empty = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    kv_ready = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    sched_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    sched_empty = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    for i in gl.static_range(QSTAGES):
        mbarrier.init(q_ready.index(i), count=1)
        mbarrier.init(q_empty.index(i), count=NUM_CONSUMERS)
    for i in gl.static_range(kStages):
        mbarrier.init(kv_empty.index(i), count=NUM_CONSUMERS)
        mbarrier.init(kv_ready.index(i), count=1)
    mbarrier.init(sched_ready, count=1)
    mbarrier.init(sched_empty, count=NUM_CONSUMERS)

    gl.warp_specialize(
        [
            (_attn_producer_v2, (desc_q, desc_k, desc_v, q_bufs, k_bufs, v_bufs,
                                  q_ready, q_empty, kv_empty, kv_ready,
                                  sched_smem, sched_ready, sched_empty, counter_ptr, N_CTX, H_Q, GROUP,
                                  num_pid_m, num_tiles, NUM_SMS,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps,
                                  NUM_CONSUMERS)),
            (_attn_consumer_v2, (desc_o, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                                  sched_smem, sched_ready, sched_empty,
                                  N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES,
                                  num_warps, NUM_CONSUMERS, 0)),
            (_attn_consumer_v2, (desc_o, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                                  sched_smem, sched_ready, sched_empty,
                                  N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES,
                                  num_warps, NUM_CONSUMERS, 1)),
        ],
        [num_warps, num_warps],
        [240, 240],
    )


def attention_v2(q, k, v, sm_scale=None, BLOCK_M=64, BLOCK_N=128, kStages=2,
                  num_warps=4, NUM_CONSUMERS=2, emit=True):
    """mask-split + DYNAMIC work-stealing scheduler (atomicAdd structure).
    GQA-capable: q has H_Q heads, k/v have H_KV (<= H_Q) heads."""
    Z, H_Q, N_CTX, HEAD_DIM = q.shape
    H_KV = k.shape[1]
    assert k.shape == (Z, H_KV, N_CTX, HEAD_DIM) and v.shape == k.shape
    assert H_Q % H_KV == 0, "H_Q must be a multiple of H_KV (GQA)"
    GROUP = H_Q // H_KV
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    BM_TOTAL = BLOCK_M * NUM_CONSUMERS
    QSTAGES = 2

    q2 = q.reshape(Z * H_Q * N_CTX, HEAD_DIM)
    k2 = k.reshape(Z * H_KV * N_CTX, HEAD_DIM)
    v2 = v.reshape(Z * H_KV * N_CTX, HEAD_DIM)
    o = torch.empty_like(q)
    o2 = o.reshape(Z * H_Q * N_CTX, HEAD_DIM)

    ql = gl.NVMMASharedLayout.get_default_for([BM_TOTAL, HEAD_DIM], gl.bfloat16)
    kl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    vl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    ol = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], gl.bfloat16)
    desc_q = TensorDescriptor.from_tensor(q2, [BM_TOTAL, HEAD_DIM], ql)
    desc_k = TensorDescriptor.from_tensor(k2, [BLOCK_N, HEAD_DIM], kl)
    desc_v = TensorDescriptor.from_tensor(v2, [BLOCK_N, HEAD_DIM], vl)
    desc_o = TensorDescriptor.from_tensor(o2, [BLOCK_M, HEAD_DIM], ol)

    num_pid_m = _cdiv(N_CTX, BM_TOTAL)
    num_tiles = num_pid_m * Z * H_Q
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (min(NUM_SMS, num_tiles),)
    # work-stealing counter: seeded at 0; first steal = 0 + gridDim.x.
    counter = torch.zeros((1,), dtype=torch.int32, device=q.device)
    kw = {"emit_cuda": True} if emit else {}
    attn_fwd_v2[grid](
        desc_q, desc_k, desc_v, desc_o, counter, N_CTX, sm_scale, H_Q, GROUP,
        num_pid_m, num_tiles, grid[0],
        BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps=num_warps,
        NUM_CONSUMERS=NUM_CONSUMERS, **kw)
    return o


# ===========================================================================
# V3 — v2 + FFMA-fused softmax (kill the FMUL+FADD->FFMA contraction loss).
# ncu opcode diff vs a peak FA3 kernel: v2 emits 37.9M FMUL + 37.3M FADD and only
# 0.56M FFMA, while the reference emits 18.3M FFMA (and 21M FMUL/18M FADD). Root
# cause: v2 computes `qk = qk*qk_scale` and REUSES the scaled qk for BOTH the rowmax
# and the `qk - m` subtract, so the per-element multiply has two consumers and nvcc
# cannot contract it into the subtract. Fix: take the max on RAW qk, scale only
# the per-row scalar, and write `qk*qk_scale - m_ij` as one expression -> the
# per-element multiply is single-use -> ptxas fuses to FFMA. Since v2 SERIALIZES
# QK->softmax->PV (wgmma_wait(0)), softmax sits on the critical path between MMAs;
# shortening it raises tensor-pipe occupancy (v2 58.7% -> ~68.9%).
# ===========================================================================
@gluon.jit
def _attn_consumer_v3(desc_o, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                       sched_smem, sched_ready, sched_empty,
                       N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                       BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                       kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                       NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr):
    dtype: gl.constexpr = desc_o.dtype
    qk_scale = sm_scale * 1.4426950408889634
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    bl: gl.constexpr = sched_bcast_layout(num_warps)

    qk_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    o_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, HEAD_DIM, num_warps)
    p_layout: gl.constexpr = dot_operand_layout(o_layout, dtype)
    row_qk: gl.constexpr = gl.SliceLayout(1, qk_layout)
    row_o: gl.constexpr = gl.SliceLayout(1, o_layout)
    col_qk: gl.constexpr = gl.SliceLayout(0, qk_layout)

    o_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], desc_o.layout)

    qt = 0
    kidx = 0
    kph = 0
    sph = 0
    mbarrier.wait(sched_ready, sph)
    pid = gl.max(sched_smem.load(bl), axis=0)
    mbarrier.arrive(sched_empty, count=1)
    sph ^= 1

    while pid < num_tiles:
        pid_m = pid % num_pid_m
        off_hz = pid // num_pid_m
        seq_base = off_hz * N_CTX
        off_m_total = pid_m * BM_TOTAL
        off_m = off_m_total + CONSUMER_ID * BLOCK_M

        qidx = qt % QSTAGES
        qph = (qt // QSTAGES) % 2
        mbarrier.wait(q_ready.index(qidx), qph)
        q_slice = q_bufs.index(qidx).slice(CONSUMER_ID * BLOCK_M, BLOCK_M, dim=0)

        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=row_qk)
        l_i = gl.full([BLOCK_M], 1.0, gl.float32, layout=row_qk)
        acc = gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, layout=o_layout)
        offs_m = off_m + gl.arange(0, BLOCK_M, layout=row_qk)

        hi = off_m_total + BM_TOTAL
        mask_start = ((off_m + 1) // BLOCK_N) * BLOCK_N

        # --- mask-free loop: blocks strictly below the diagonal ---
        for start_n in range(0, mask_start, BLOCK_N):
            mbarrier.wait(kv_ready.index(kidx), kph)
            qk = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
            qk = warpgroup_mma(q_slice, k_bufs.index(kidx).permute((1, 0)), qk,
                               use_acc=False, is_async=True)
            qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))

            # FFMA-fused: max on RAW qk, scale the scalar, fuse qk*scale - m.
            m_ij = gl.maximum(m_i, gl.max(qk, axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = gl.exp2(qk)
            l_ij = gl.sum(p, axis=1)
            alpha = gl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            alpha_o = gl.convert_layout(alpha, row_o)
            acc = acc * alpha_o[:, None]
            p_dot = gl.convert_layout(p.to(dtype), p_layout)
            acc = warpgroup_mma(p_dot, v_bufs.index(kidx), acc, use_acc=True, is_async=True)
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

            mbarrier.arrive(kv_empty.index(kidx), count=1)
            m_i = m_ij
            kidx += 1
            if kidx == kStages:
                kidx = 0
                kph ^= 1

        # --- masked loop: the diagonal blocks ---
        for start_n in range(mask_start, hi, BLOCK_N):
            mbarrier.wait(kv_ready.index(kidx), kph)
            qk = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
            qk = warpgroup_mma(q_slice, k_bufs.index(kidx).permute((1, 0)), qk,
                               use_acc=False, is_async=True)
            qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))

            offs_n = start_n + gl.arange(0, BLOCK_N, layout=col_qk)
            mask = offs_m[:, None] >= offs_n[None, :]
            # mask on RAW qk (-inf stays -inf after scaling), then FFMA-fuse.
            qk = gl.where(mask, qk, float("-inf"))
            m_ij = gl.maximum(m_i, gl.max(qk, axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = gl.exp2(qk)
            l_ij = gl.sum(p, axis=1)
            alpha = gl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            alpha_o = gl.convert_layout(alpha, row_o)
            acc = acc * alpha_o[:, None]
            p_dot = gl.convert_layout(p.to(dtype), p_layout)
            acc = warpgroup_mma(p_dot, v_bufs.index(kidx), acc, use_acc=True, is_async=True)
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

            mbarrier.arrive(kv_empty.index(kidx), count=1)
            m_i = m_ij
            kidx += 1
            if kidx == kStages:
                kidx = 0
                kph ^= 1

        l_o = gl.convert_layout(l_i, row_o)
        acc = acc * (1.0 / l_o)[:, None]
        o_smem.store(acc.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(desc_o, [seq_base + off_m, 0], o_smem)
        tma.store_wait(pendings=0)
        mbarrier.arrive(q_empty.index(qidx), count=1)
        qt += 1

        mbarrier.wait(sched_ready, sph)
        pid = gl.max(sched_smem.load(bl), axis=0)
        mbarrier.arrive(sched_empty, count=1)
        sph ^= 1


@gluon.jit
def attn_fwd_v3(desc_q, desc_k, desc_v, desc_o, counter_ptr, N_CTX, sm_scale, H_Q, GROUP,
                 num_pid_m, num_tiles, NUM_SMS,
                 BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                 kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                 NUM_CONSUMERS: gl.constexpr):
    dtype: gl.constexpr = desc_q.dtype
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    BCAST: gl.constexpr = 32 * num_warps

    q_bufs = gl.allocate_shared_memory(dtype, [QSTAGES, BM_TOTAL, HEAD_DIM], desc_q.layout)
    k_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_k.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_v.layout)
    sched_smem = gl.allocate_shared_memory(gl.int32, [BCAST],
                                           gl.SwizzledSharedLayout(1, 1, 1, order=[0]))
    q_ready = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    kv_empty = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    kv_ready = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    sched_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    sched_empty = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    for i in gl.static_range(QSTAGES):
        mbarrier.init(q_ready.index(i), count=1)
        mbarrier.init(q_empty.index(i), count=NUM_CONSUMERS)
    for i in gl.static_range(kStages):
        mbarrier.init(kv_empty.index(i), count=NUM_CONSUMERS)
        mbarrier.init(kv_ready.index(i), count=1)
    mbarrier.init(sched_ready, count=1)
    mbarrier.init(sched_empty, count=NUM_CONSUMERS)

    gl.warp_specialize(
        [
            (_attn_producer_v2, (desc_q, desc_k, desc_v, q_bufs, k_bufs, v_bufs,
                                  q_ready, q_empty, kv_empty, kv_ready,
                                  sched_smem, sched_ready, sched_empty, counter_ptr, N_CTX, H_Q, GROUP,
                                  num_pid_m, num_tiles, NUM_SMS,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps,
                                  NUM_CONSUMERS)),
            (_attn_consumer_v3, (desc_o, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                                  sched_smem, sched_ready, sched_empty,
                                  N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES,
                                  num_warps, NUM_CONSUMERS, 0)),
            (_attn_consumer_v3, (desc_o, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                                  sched_smem, sched_ready, sched_empty,
                                  N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                                  BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES,
                                  num_warps, NUM_CONSUMERS, 1)),
        ],
        [num_warps, num_warps],
        [240, 240],
    )


def attention_v3(q, k, v, sm_scale=None, BLOCK_M=64, BLOCK_N=128, kStages=2,
                  num_warps=4, NUM_CONSUMERS=2, emit=True):
    """v2 work-stealing + FFMA-fused softmax. GQA-capable (H_KV <= H_Q)."""
    Z, H_Q, N_CTX, HEAD_DIM = q.shape
    H_KV = k.shape[1]
    assert k.shape == (Z, H_KV, N_CTX, HEAD_DIM) and v.shape == k.shape
    assert H_Q % H_KV == 0, "H_Q must be a multiple of H_KV (GQA)"
    GROUP = H_Q // H_KV
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    BM_TOTAL = BLOCK_M * NUM_CONSUMERS
    QSTAGES = 2

    q2 = q.reshape(Z * H_Q * N_CTX, HEAD_DIM)
    k2 = k.reshape(Z * H_KV * N_CTX, HEAD_DIM)
    v2 = v.reshape(Z * H_KV * N_CTX, HEAD_DIM)
    o = torch.empty_like(q)
    o2 = o.reshape(Z * H_Q * N_CTX, HEAD_DIM)

    ql = gl.NVMMASharedLayout.get_default_for([BM_TOTAL, HEAD_DIM], gl.bfloat16)
    kl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    vl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    ol = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], gl.bfloat16)
    desc_q = TensorDescriptor.from_tensor(q2, [BM_TOTAL, HEAD_DIM], ql)
    desc_k = TensorDescriptor.from_tensor(k2, [BLOCK_N, HEAD_DIM], kl)
    desc_v = TensorDescriptor.from_tensor(v2, [BLOCK_N, HEAD_DIM], vl)
    desc_o = TensorDescriptor.from_tensor(o2, [BLOCK_M, HEAD_DIM], ol)

    num_pid_m = _cdiv(N_CTX, BM_TOTAL)
    num_tiles = num_pid_m * Z * H_Q
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (min(NUM_SMS, num_tiles),)
    counter = torch.zeros((1,), dtype=torch.int32, device=q.device)
    kw = {"emit_cuda": True} if emit else {}
    attn_fwd_v3[grid](
        desc_q, desc_k, desc_v, desc_o, counter, N_CTX, sm_scale, H_Q, GROUP,
        num_pid_m, num_tiles, grid[0],
        BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps=num_warps,
        NUM_CONSUMERS=NUM_CONSUMERS, **kw)
    return o


def _ref_causal(q, k, v, sm_scale, qblk=1024):
    """Tiled fp32 causal-attention reference. Processes the query rows in blocks
    so the full N x N score matrix is never materialized (works up to N=128K)."""
    Z, H, N, D = q.shape
    out = torch.empty_like(q)
    for z in range(Z):
        for h in range(H):
            qs, ks, vs = q[z, h].float(), k[z, h].float(), v[z, h].float()
            for i0 in range(0, N, qblk):
                i1 = min(i0 + qblk, N)
                s = (qs[i0:i1] @ ks.transpose(-1, -2)) * sm_scale
                rows = torch.arange(i0, i1, device=q.device)[:, None]
                cols = torch.arange(0, N, device=q.device)[None, :]
                s = s.masked_fill(cols > rows, float("-inf"))
                out[z, h, i0:i1] = (torch.softmax(s, dim=-1) @ vs).to(q.dtype)
    return out


import triton.profiler as proton
import triton.profiler.viewer as proton_viewer

# Proton (CUPTI) device-time benchmarking — NO CUDA graphs. Proton reads each
# kernel's on-GPU duration straight from CUPTI, so the number is the true device
# time of the launches inside a scope (host overhead and gaps excluded).
_PROTON_FILE = "tut02"


def _proton_run(launch, scope, reps, warmup):
    """Launch `launch` under a uniquely-named Proton scope `reps` times, after
    `warmup` un-profiled launches. Device time accumulates into that scope."""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    proton.activate()
    for _ in range(reps):
        with proton.scope(scope):
            launch()
    proton.deactivate()
    torch.cuda.synchronize()


def _proton_times(filename):
    """Parse the Proton hatchet and return {scope_name: total_device_ms}."""
    gf, metrics = proton_viewer.parse(["time/ms"], filename)
    df = gf.dataframe
    col = metrics[0]  # "time/ms (inc)"
    return {name: float(t) for name, t in zip(df["name"], df[col])}


# ===========================================================================
# Reference: Triton's own built-in FlashAttention (the stock ``06-fused-attention``
# tutorial), copied in here so the tutorial is self-contained (no module import).
# The ONLY change from the stock kernel is the output/compute dtype: the stock
# kernel hardcodes ``tl.float16``; here it is derived from ``q.dtype`` after the
# Q load, so the reference runs natively in the sweep's bf16 (matching v1/v2/v3).
# Everything else — autotune configs, the causal STAGE split, the TMA descriptor
# host path, the online-softmax inner loop — is the upstream Triton code verbatim.
# ===========================================================================


def _tri_is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def _tri_is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def _tri_supports_host_descriptor():
    return _tri_is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _tri_is_blackwell():
    return _tri_is_cuda() and torch.cuda.get_device_capability()[0] == 10


@triton.jit
def _tri_attn_fwd_inner(acc, l_i, m_i, q,  #
                        desc_k, desc_v,  #
                        offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                        N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _tri_host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TLTensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if _tri_is_hip():
    _TRI_NUM_STAGES_OPTIONS = [1]
else:
    _TRI_NUM_STAGES_OPTIONS = [2, 3, 4]

_tri_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w,
                  pre_hook=_tri_host_descriptor_pre_hook)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in _TRI_NUM_STAGES_OPTIONS
    for w in [4, 8]
]


def _tri_keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (_tri_is_cuda() and torch.cuda.get_device_capability()[0] == 9
                and BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8)


def _tri_prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]
    # Filter out configs where BLOCK_M > N_CTX, and where BLOCK_M < BLOCK_N when causal.
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
    ]


@triton.jit
def _tri_maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(_tri_keep, _tri_configs)),
                 key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': _tri_prune_invalid_configs})
@triton.jit
def _tri_attn_fwd(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  FP8_OUTPUT: tl.constexpr,  #
                  STAGE: tl.constexpr,  #
                  warp_specialize: tl.constexpr,  #
                  IS_HOPPER: tl.constexpr,  #
                  ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _tri_maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _tri_maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                             block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _tri_maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                             block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _tri_maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _tri_maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # ONLY change vs stock: derive the compute/output dtype from q (bf16-capable)
    # instead of the upstream hardcoded ``tl.float16``.
    dtype = tl.float8e5 if FP8_OUTPUT else q.dtype
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _tri_attn_fwd_inner(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX,  #
                                            warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _tri_attn_fwd_inner(acc, l_i, m_i, q,  #
                                            desc_k, desc_v,  #
                                            offset_y, dtype, start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX,  #
                                            warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


def _triton_fa(q, k, v, causal, sm_scale, warp_specialize=False):
    """Self-contained copy of the stock Triton FA forward (``06-fused-attention``).
    Used as the correctness + performance reference. bf16-capable (the output dtype
    is taken from ``q.dtype`` inside the kernel)."""
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1

    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    # Use device_descriptor for Hopper + warpspec.
    if _tri_supports_host_descriptor() and not (is_hopper() and warp_specialize):
        y_dim = q.shape[0] * q.shape[1] * q.shape[2]
        dummy_block = [1, 1]
        desc_q = TLTensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        if q.dtype == torch.float8_e5m2:
            desc_v = TLTensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
        else:
            desc_v = TLTensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_k = TLTensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        desc_o = TLTensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
    else:
        desc_q, desc_v, desc_k, desc_o = q, v, k, o

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

    _tri_attn_fwd[grid](
        sm_scale, M,  #
        q.shape[0], q.shape[1],  #
        desc_q, desc_k, desc_v, desc_o,  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
        STAGE=stage,  #
        warp_specialize=warp_specialize,  #
        IS_HOPPER=is_hopper())
    return o


def main():
    if not is_hopper():
        raise RuntimeError("Tutorial 02 targets Hopper (sm90).")
    # Token-balanced shapes: Z*H*N ~ 512K so every case fills the GPU with ~4096
    # row-tiles (saturated regime) at a uniform ~940MB/tensor, seqlen 1K -> 128K.
    SWEEP = [(1024, 16, 32), (2048, 8, 32), (4096, 4, 32), (8192, 2, 32),
             (16384, 2, 16), (32768, 1, 16), (65536, 1, 8), (131072, 1, 4)]
    # GQA / MQA cases: (D, N, Z, H_Q, H_KV) — query heads share fewer KV heads.
    # GROUP = H_Q // H_KV (8 = Llama-3-70B style, 4 = common GQA, H_KV=1 = MQA).
    GQA_CASES = [(128, 4096, 2, 32, 4),   # GROUP 8
                 (128, 8192, 1, 32, 8),   # GROUP 4
                 (128, 2048, 4, 16, 1),   # MQA (single KV head)
                 ( 64, 4096, 2, 32, 8),   # GROUP 4
                 ( 64, 2048, 4, 32, 4)]   # GROUP 8
    KERNS = (("v1", attention_v1), ("v2", attention_v2), ("v3", attention_v3))
    TOL = 5e-3
    tri_fa = _triton_fa

    print("=" * 96)
    print("Tutorial 02 — causal FlashAttention forward (emit_cuda) : correctness + performance")
    print("reference = Triton's built-in FlashAttention (tutorial 06, copied in)")
    print("perf = Proton/CUPTI device time (no CUDA graphs)")
    print("=" * 96)

    # ---- Proton-profiled perf pass + correctness check, all shapes -------------
    proton.start(_PROTON_FILE, hook="triton")
    proton.deactivate()   # correctness checks + warmups stay un-profiled
    rel = {}; flops = {}; failures = []
    for D in (64, 128):
        for N, Z, H in SWEEP:
            reps = 50 if N <= 8192 else (20 if N <= 32768 else 10)
            warmup = 10 if N <= 32768 else 5
            torch.manual_seed(N + D)
            # bf16 — the native dtype of v1/v2/v3; the copied-in Triton FA reference
            # is now bf16-capable (output dtype derived from q.dtype in the kernel).
            q = torch.randn(Z, H, N, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q); v = torch.randn_like(q)
            sm = 1.0 / math.sqrt(D)
            flops[(D, N)] = Z * H * 4 * D * (N * (N + 1) // 2)   # causal QK^T + P@V MACs
            ref = _ref_causal(q, k, v, sm)

            relmax = 0.0
            for nm, fn in KERNS:
                o = fn(q, k, v, sm_scale=sm, emit=True); torch.cuda.synchronize()
                relmax = max(relmax, (o.float() - ref.float()).norm().item() / ref.float().norm().item())
                _proton_run(lambda fn=fn: fn(q, k, v, sm_scale=sm, emit=True),
                            f"{nm}_D{D}_N{N}", reps, warmup)
            if tri_fa is not None:
                try:
                    # signature: attention(q, k, v, causal, sm_scale, warp_specialize)
                    _proton_run(lambda: tri_fa(q, k, v, True, sm, False),
                                f"tri_D{D}_N{N}", reps, warmup)
                except Exception as ex:
                    print(f"  D{D} N{N} triton-FA perf ERR {repr(ex)[:100]}")

            rel[(D, N)] = relmax
            if relmax >= TOL:
                failures.append((D, N, relmax))
            del q, k, v, ref; torch.cuda.empty_cache()

    # ---- GQA / MQA section: q has H_Q heads, k/v have H_KV (<= H_Q) heads. The
    # kernels load each query head's KV group (h_kv = h_q // GROUP); the reference
    # and the (MHA-only) Triton baseline get k/v expanded back to H_Q heads. -------
    rel_gqa = {}; flops_gqa = {}
    for i, (D, N, Z, H_Q, H_KV) in enumerate(GQA_CASES):
        reps = 50 if N <= 8192 else (20 if N <= 32768 else 10)
        warmup = 10 if N <= 32768 else 5
        GROUP = H_Q // H_KV
        torch.manual_seed(N + D + H_KV)
        q = torch.randn(Z, H_Q, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(Z, H_KV, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(Z, H_KV, N, D, device="cuda", dtype=torch.bfloat16)
        sm = 1.0 / math.sqrt(D)
        flops_gqa[i] = Z * H_Q * 4 * D * (N * (N + 1) // 2)   # scales with QUERY heads
        # expand KV to H_Q heads for the MHA reference / baseline (same math).
        k_rep = k.repeat_interleave(GROUP, dim=1)
        v_rep = v.repeat_interleave(GROUP, dim=1)
        ref = _ref_causal(q, k_rep, v_rep, sm)

        relmax = 0.0
        for nm, fn in KERNS:
            o = fn(q, k, v, sm_scale=sm, emit=True); torch.cuda.synchronize()
            relmax = max(relmax, (o.float() - ref.float()).norm().item() / ref.float().norm().item())
            _proton_run(lambda fn=fn: fn(q, k, v, sm_scale=sm, emit=True),
                        f"{nm}_gqa{i}", reps, warmup)
        try:
            _proton_run(lambda: tri_fa(q, k_rep, v_rep, True, sm, False),
                        f"tri_gqa{i}", reps, warmup)
        except Exception as ex:
            print(f"  GQA D{D} N{N} H_Q{H_Q} H_KV{H_KV} triton-FA perf ERR {repr(ex)[:100]}")

        rel_gqa[i] = relmax
        if relmax >= TOL:
            failures.append((f"GQA D{D} N{N} H_Q{H_Q} H_KV{H_KV}", relmax))
        del q, k, v, k_rep, v_rep, ref; torch.cuda.empty_cache()
    proton.finalize()
    times = _proton_times(f"{_PROTON_FILE}.hatchet")   # {scope: total_device_ms}

    # ---- report ---------------------------------------------------------------
    def tf(tag, D, N, reps):
        key = f"{tag}_D{D}_N{N}"
        if key not in times:
            return float("nan")
        return flops[(D, N)] / (times[key] / reps * 1e-3) / 1e12

    print(f"{'D':>3} {'N':>7} {'Z':>2} {'H':>3} | {'rel_l2':>8} ok | "
          f"{'v1':>5} {'v2':>5} {'v3':>5} {'tri':>6}  TFLOP/s | {'v3/tri':>6}")
    print("-" * 96)
    for D in (64, 128):
        for N, Z, H in SWEEP:
            reps = 50 if N <= 8192 else (20 if N <= 32768 else 10)
            t = {nm: tf(nm, D, N, reps) for nm, _ in KERNS}
            tritf = tf("tri", D, N, reps)
            relmax = rel[(D, N)]; ok = relmax < TOL
            tris = f"{tritf:6.0f}" if tritf == tritf else "   n/a"
            ratio = f"{t['v3'] / tritf:6.2f}" if tritf == tritf else "     -"
            print(f"{D:>3} {N:>7} {Z:>2} {H:>3} | {relmax:8.1e} {'OK' if ok else '!!':>2} | "
                  f"{t['v1']:5.0f} {t['v2']:5.0f} {t['v3']:5.0f} {tris}          | {ratio}")
    print("=" * 96)

    # ---- GQA / MQA table ------------------------------------------------------
    def tf_gqa(tag, i, reps):
        key = f"{tag}_gqa{i}"
        if key not in times:
            return float("nan")
        return flops_gqa[i] / (times[key] / reps * 1e-3) / 1e12

    print()
    print("GQA / MQA (q: H_Q heads, k/v: H_KV heads; reference = MHA with KV expanded)")
    print(f"{'D':>3} {'N':>7} {'Z':>2} {'H_Q':>4} {'H_KV':>5} {'grp':>4} | {'rel_l2':>8} ok | "
          f"{'v1':>5} {'v2':>5} {'v3':>5} {'tri':>6}  TFLOP/s | {'v3/tri':>6}")
    print("-" * 96)
    for i, (D, N, Z, H_Q, H_KV) in enumerate(GQA_CASES):
        reps = 50 if N <= 8192 else (20 if N <= 32768 else 10)
        t = {nm: tf_gqa(nm, i, reps) for nm, _ in KERNS}
        tritf = tf_gqa("tri", i, reps)
        relmax = rel_gqa[i]; ok = relmax < TOL
        tris = f"{tritf:6.0f}" if tritf == tritf else "   n/a"
        ratio = f"{t['v3'] / tritf:6.2f}" if tritf == tritf else "     -"
        print(f"{D:>3} {N:>7} {Z:>2} {H_Q:>4} {H_KV:>5} {H_Q // H_KV:>4} | "
              f"{relmax:8.1e} {'OK' if ok else '!!':>2} | "
              f"{t['v1']:5.0f} {t['v2']:5.0f} {t['v3']:5.0f} {tris}          | {ratio}")
    print("=" * 96)
    if failures:
        raise AssertionError(f"correctness FAILED: {failures}")
    print("tutorial 02 PASSED — all MHA + GQA/MQA shapes correct (rel_l2 < 5e-3); v3 is the fastest version")


if __name__ == "__main__":
    main()
