# ===========================================================================
# Tutorial 03 — NON-CAUSAL FlashAttention forward + backward (emit_cuda)
# ===========================================================================
# Follows the omni_kernel FA3 reference
# (/data00/zheng.size/share/omni_kernel, csrc/attn/{fa3_fwd,fa3_bwd}.cu and
#  course/02_hopper/03_attention/readme.md).  Two pieces:
#
#   FORWARD  — a Gluon warp-specialized (1-producer / 2-consumer) non-causal
#              FA3 forward.  Identical machinery to tutorial 02 minus the causal
#              mask, plus it ALSO writes the per-row log-sum-exp
#                   LSE_i = m_i + log2(l_i)                     (base-2)
#              which the backward needs to recompute P without re-reading it.
#              (omni readme §2.2:  LSE_i = m_i·log2(e) + log2(ℓ_i); here m_i is
#              already accumulated in the log2 domain, qk·sm_scale·log2(e).)
#
#   BACKWARD — omni readme §6: a preprocess kernel that forms
#                   Delta_i = sum_c dO_ic · O_ic                (= sum_k P dP)
#              followed by the 5-GEMM main kernel.  Expressed here exactly as the
#              upstream Triton ``06-fused-attention`` backward with CAUSAL=False,
#              which IS omni's non-causal 5-GEMM recurrence (readme §6.2 table):
#                   S^T  = K Q^T                 (wgmma_ss)
#                   dP^T = V dO^T                (wgmma_ss)
#                   P^T  = exp2(S^T·scale_log2 − LSE)   recompute
#                   dS^T = P^T·(dP^T − Delta)    softmax-bwd elementwise
#                   dV  += P^T dO                (wgmma_rs)   — no scale
#                   dK  += dS^T Q                (wgmma_rs)   — ×sm_scale at end
#                   dQ  += dS  K                 (wgmma_ss)
#              omni folds all 5 into one outer-KV kernel and reduces dQ across
#              CTAs; the upstream layout below instead splits dK/dV (outer-KV,
#              register-accumulated → one write, no atomic) from dQ (outer-Q,
#              register-accumulated → one write, no atomic).  Same math, the
#              cleaner deterministic decomposition (omni readme §8 option B/C).
#
# Run (emit_cuda) inside a CUDA container that bind-mounts /data00, e.g.:
#   CUDA_VISIBLE_DEVICES=6 TRITON_EMIT_CUDA=1 EMIT=1 TRITON_BACKENDS_IN_TREE=1 \
#   PYTHONPATH=<repo>/python:<repo>/triton_cuda_backend python3 -u -c \
#     "import triton_cuda_backend, runpy; \
#      runpy.run_path('03-fused-attention-bwd.py', run_name='__main__')"
# ===========================================================================
# PERFORMANCE — emit_cuda vs omni_kernel (Hopper H800, bf16, D=128, non-causal).
# Measured 2026-06-26.  bwd = version=3 (fast config below); fwd = attention_fwd
# (overlap=False).  % = emit_cuda TFLOP/s / omni TFLOP/s.
#
# Two columns, because they measure different things:
#   * KERNEL-ONLY (CUDA-graph replay; the fair measure of generated-code quality —
#     no Python/launch overhead, matching omni's lean C++ binding).
#   * EAGER (median-of-50 Python calls).  At small N the EAGER number is dominated
#     by Python dispatch (~110µs to rebuild 4-5 TMA descriptors + reshape + launch),
#     NOT the kernel: at N=1024 the GPU runs ~10µs of a ~120µs call (CPU-bound).
#     This is framework overhead common to ALL Triton kernels — wrap the step in a
#     CUDA graph (or reuse descriptors) and the eager column collapses onto kernel.
#
#    Z   H      N | fwd kernel | bwd kernel
#    1  16   1024 |    108%    |     88%
#    2  16   2048 |  103/ 97%  |     95%
#    2  16   4096 |    101%    |     99%
#    1  16   8192 |     95%    |     98%
#    1   8  16384 |     93%    |     97%
#    4   8   2048 |     97%    |     92%
#    1  32   4096 |    101%    |     98%
#
#   FORWARD is at parity at the kernel level (93-108% of omni, same structure as
#   omni: BLOCK_M=64, 2 consumers, BLOCK_N=128, 2 KV stages, persistent, FFMA-fused
#   softmax + ex2.approx.ftz).  BACKWARD is 95-99% for N>=2048; only the two SMALLEST
#   tiles trail — N=1024 (88%) and the Z4 N=2048 (92%).  Root cause (nsys-MEASURED, NOT
#   launch count): omni's fa3_backward launches FOUR kernels @N=1024 — preprocess 4.6µs +
#   main 43.9µs + postprocess 5.9µs + dq-zero fill 3.6µs — i.e. MORE aux launches than this
#   file's two.  The whole gap is that the WS *main* kernel is ~64µs vs omni's 43.9µs at
#   N=1024 (32% slower): per-q-tile warp-spec schedule overhead exposed when a CTA runs few
#   q-iters (16 @N=1024).  At large N the main kernel MATCHES omni (97-99%).  So the lever is
#   the main-kernel small-N schedule, NOT launches and NOT Δ-fusion (which is anyway blocked:
#   producer warpgroup has ~32 regs, consumers pinned at 240).  Three launches were
#   nonetheless removed, all validated: (1) WS kernel reduce-adds sm_scale-prescaled dQ
#   straight into output dq (consumer folds ×sm_scale at handoff) — no dq_acc, no finalize;
#   (2) _bwd_preprocess clears dq (ZERO_DQ) — no dq.zero_(); (3) K is passed RAW and the
#   consumer applies α=sm·log2e inside exp2(st·α−m) — no arg_k=k·α materialization
#   (also GQA-safe).  Backward rel-L2: dq~1e-2, dk/dv~2.3e-3.
#
#   NOTE on EAGER vs KERNEL: a plain Python `attention_bwd(...)` call at small N is
#   CPU-bound on Triton's per-launch dispatch + TMA-descriptor build (~110µs at N=1024,
#   vs ~65µs of GPU), so eager TFLOP/s reads far below the kernel numbers above.  That
#   overhead is common to all Triton kernels — wrap the step in a CUDA graph (how the
#   table above is measured) and eager collapses onto the kernel column.
#
# BACKWARD VERSIONS (3 kept, increasing perf — see the attention_bwd dispatch):
#   v1 — SPLIT baseline (Triton, non-warp-specialized): dK/dV pass + dQ pass.
#   v2 — WARP-SPECIALIZED fused: 1 producer + 2 KV-row-split consumers.   ~77%
#   v3 — v2 + OVERLAP=5 (dQ = dS·K uses dS directly as a register A-operand,
#        no dQ smem round-trip) + hoisted & ONE-ITERATION-AHEAD-prefetched
#        LSE/Delta global loads.  95-97% of omni.  v3 requires the emitter opts:
#          TRITON_BWD_INPLACE_RS_PACK=1   TRITON_EMIT_INPLACE_ELTWISE=1
# ===========================================================================

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
DQ_REDUCE = tl.constexpr(True)  # diagnostic toggle: False = skip dq reduce to measure ceiling


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
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=get_warps_per_cta(BLOCK_M, num_warps),
        instr_shape=[16, N, 256 // dtype.primitive_bitwidth])


@gluon.constexpr_function
def dot_operand_layout(parent, dtype):
    return gl.DotOperandLayout(operand_index=0, parent=parent,
                               k_width=32 // dtype.primitive_bitwidth)


# ===========================================================================
# FORWARD — warp-specialized 1-producer / 2-consumer, NON-CAUSAL, writes O + LSE
# ===========================================================================
# PERSISTENT non-causal FA3 forward (+LSE).  Because every Q tile attends the FULL
# key range, all tiles cost the same — so unlike tut02's causal v3 (which needed an
# atomic work-stealing scheduler for load balance) this kernel uses a STATIC
# persistent stride: tile id  pid = sm_id + iter*NUM_SMS.  Both the producer and the
# two consumers recompute the same pid from program_id(0) + a private iteration
# counter, so NO scheduler broadcast (smem/atomic) is needed.  A QSTAGES Q double
# buffer lets the producer prefetch the next tile's Q while the consumers drain the
# current one.  Softmax is FFMA-fused (max on RAW qk, scale the scalar, write
# `qk*qk_scale - m_ij` as one single-use expression) so ptxas contracts the per-
# element multiply into the subtract — the lever that took tut02 to ~95% of level6.
@gluon.jit
def _fwd_producer(desc_q, desc_k, desc_v, q_bufs, k_bufs, v_bufs,
                  q_ready, q_empty, kv_empty, kv_ready, N_CTX,
                  num_pid_m, num_tiles, NUM_SMS, H_q, H_kv, G,
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                  kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                  NUM_CONSUMERS: gl.constexpr):
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS
    pid = gl.program_id(0)
    qt = 0
    kidx = 0
    kph = 1   # empty-barrier wait starts at phase 1 (fresh barrier passes immediately)
    while pid < num_tiles:
        pid_m = pid % num_pid_m
        off_hz = pid // num_pid_m            # combined (z, query-head) index
        seq_base_q = off_hz * N_CTX
        # GQA: query head off_hz%H_q maps to KV head (off_hz%H_q)//G in batch off_hz//H_q.
        z = off_hz // H_q
        h_kv = (off_hz % H_q) // G
        seq_base_kv = (z * H_kv + h_kv) * N_CTX
        off_m_total = pid_m * BM_TOTAL

        # Q double-buffer: wait this slot is free, then TMA-load BM_TOTAL rows.
        qidx = qt % QSTAGES
        qeph = ((qt // QSTAGES) % 2) ^ 1
        mbarrier.wait(q_empty.index(qidx), qeph)
        mbarrier.expect(q_ready.index(qidx), desc_q.block_type.nbytes)
        tma.async_copy_global_to_shared(desc_q, [seq_base_q + off_m_total, 0],
                                        q_ready.index(qidx), q_bufs.index(qidx))
        qt += 1

        # NON-CAUSAL: every Q row attends the FULL key range; ring continuous across tiles.
        for start_n in range(0, N_CTX, BLOCK_N):
            mbarrier.wait(kv_empty.index(kidx), kph)
            bar = kv_ready.index(kidx)
            mbarrier.expect(bar, desc_k.block_type.nbytes + desc_v.block_type.nbytes)
            tma.async_copy_global_to_shared(desc_k, [seq_base_kv + start_n, 0], bar, k_bufs.index(kidx))
            tma.async_copy_global_to_shared(desc_v, [seq_base_kv + start_n, 0], bar, v_bufs.index(kidx))
            kidx += 1
            if kidx == kStages:
                kidx = 0
                kph ^= 1
        pid += NUM_SMS


@gluon.jit
def _fwd_consumer(desc_o, M, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                  N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                  BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                  kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                  NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr):
    dtype: gl.constexpr = desc_o.dtype
    qk_scale = sm_scale * 1.4426950408889634   # log2(e)
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS

    qk_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    o_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, HEAD_DIM, num_warps)
    p_layout: gl.constexpr = dot_operand_layout(o_layout, dtype)
    row_qk: gl.constexpr = gl.SliceLayout(1, qk_layout)
    row_o: gl.constexpr = gl.SliceLayout(1, o_layout)

    o_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], desc_o.layout)

    pid = gl.program_id(0)
    qt = 0
    kidx = 0
    kph = 0
    while pid < num_tiles:
        pid_m = pid % num_pid_m
        off_hz = pid // num_pid_m
        seq_base = off_hz * N_CTX
        off_m_total = pid_m * BM_TOTAL
        off_m = off_m_total + CONSUMER_ID * BLOCK_M           # this consumer's BLOCK_M rows

        qidx = qt % QSTAGES
        qph = (qt // QSTAGES) % 2
        mbarrier.wait(q_ready.index(qidx), qph)
        q_slice = q_bufs.index(qidx).slice(CONSUMER_ID * BLOCK_M, BLOCK_M, dim=0)

        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=row_qk)
        l_i = gl.full([BLOCK_M], 1.0, gl.float32, layout=row_qk)
        acc = gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, layout=o_layout)

        for start_n in range(0, N_CTX, BLOCK_N):
            mbarrier.wait(kv_ready.index(kidx), kph)
            qk = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
            qk = warpgroup_mma(q_slice, k_bufs.index(kidx).permute((1, 0)), qk,
                               use_acc=False, is_async=True)
            qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk,))

            # NON-CAUSAL, FFMA-fused: max on RAW qk, scale the scalar, fuse qk*scale - m.
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

        # LSE in base-2: m_i is already the running max of qk·sm_scale·log2(e),
        # so LSE_i = m_i + log2(l_i) matches omni's attention_lse_log2 reference.
        lse = m_i + gl.log2(l_i)
        offs_lse = (seq_base + off_m) + gl.arange(0, BLOCK_M, layout=row_qk)
        gl.store(M + offs_lse, lse)
        pid += NUM_SMS


# ---------------------------------------------------------------------------
# v2 consumer — FA3 2-stage software pipeline: softmax(j) overlaps PV(j-1).
# ---------------------------------------------------------------------------
# The deferred-PV trick.  The PV GEMM of block j-1 is issued at the TOP of
# iteration j, then while it runs on the tensor cores we compute the softmax of
# block j on the CUDA cores, and only AFTER waiting PV do we rescale `acc`.  So:
#   * softmax never reads the in-flight PV accumulator  -> no ptxas C7514;
#   * only RESOLVED tensors (acc, P, m_i, l_i) cross the loop boundary, never a
#     warpgroup_mma accumulator object -> sidesteps the emit_cuda loop-carry bug.
# Online-softmax bookkeeping (acc always at the CURRENT max scale):
#   enter iter j:  acc = Σ_{i≤j-2} exp2(s_i−m)·v_i ,  P = p_{j-1} ,  scale = m
#   PV:            acc += P · V_{j-1}                       (still scale m)
#   softmax(j):    m_new=max(m,rowmax s_j); α=exp2(m−m_new); p_j=exp2(s_j−m_new)
#   rescale:       acc *= α   (after PV wait)               -> scale m_new
# Needs kStages≥3: a consumer holds two live KV stages (QK(j) + PV(j-1)) so the
# producer still needs one slot of run-ahead.  BLOCK_N=64 keeps that in 227KB.
#
# MEASURED RESULT (H800, 2026-06-23): this is CORRECT (bit-matches v1) and truly
# C7514-free, but it is NOT FASTER and is the wrong kernel for Hopper.  Two facts
# kill it:  (1) the overlap needs kStages≥3 which only fits at BLOCK_N=64, and
# BLOCK_N=128 wgmma is ~8% more efficient than BLOCK_N=64 (v1 81% vs any-BN64
# 75% @N8192);  (2) even at equal BLOCK_N=64, the explicit pipeline gives ZERO
# net gain (overlap 75% == non-overlap 75% @N8192) because the two independent
# consumer warpgroups already overlap each other's softmax/GEMM at the warp-
# scheduler level — there is no idle tensor/CUDA time left for the software
# pipeline to reclaim.  Kept as an instructive v2; the default stays v1 (BN128).
@gluon.jit
def _fwd_consumer_ov(desc_o, M, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                     N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                     BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
                     kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                     NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr):
    dtype: gl.constexpr = desc_o.dtype
    qk_scale = sm_scale * 1.4426950408889634
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS

    qk_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    o_layout: gl.constexpr = mma_layout(dtype, BLOCK_M, HEAD_DIM, num_warps)
    p_layout: gl.constexpr = dot_operand_layout(o_layout, dtype)
    row_qk: gl.constexpr = gl.SliceLayout(1, qk_layout)
    row_o: gl.constexpr = gl.SliceLayout(1, o_layout)

    o_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, HEAD_DIM], desc_o.layout)

    pid = gl.program_id(0)
    qt = 0
    kctr = 0   # monotonic global KV-block counter (continuous across tiles); stage =
               # kctr % kStages, phase = (kctr // kStages) % 2.  A single counter with
               # NO conditional reassignment avoids the emit_cuda loop-carried-scalar bug.
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

        # ---- prologue: block 0 (QK(0) -> softmax(0)), no PV yet (acc stays 0) ----
        s0 = kctr % kStages
        ph0 = (kctr // kStages) % 2
        mbarrier.wait(kv_ready.index(s0), ph0)
        qk0 = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
        qk0 = warpgroup_mma(q_slice, k_bufs.index(s0).permute((1, 0)), qk0,
                            use_acc=False, is_async=True)
        qk = warpgroup_mma_wait(num_outstanding=0, deps=(qk0,))
        m_ij = gl.maximum(m_i, gl.max(qk, axis=1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = gl.exp2(qk)
        l_i = l_i * gl.exp2(m_i - m_ij) + gl.sum(p, axis=1)
        m_i = m_ij
        P = gl.convert_layout(p.to(dtype), p_layout)   # carried softmax probs
        kctr += 1

        # ---- steady state: softmax(j) overlaps PV(j-1) ----
        for _ in range(1, N_CTX // BLOCK_N):
            s_qk = kctr % kStages
            ph_qk = (kctr // kStages) % 2
            s_pv = (kctr - 1) % kStages          # previous block's stage (for PV(j-1))
            mbarrier.wait(kv_ready.index(s_qk), ph_qk)
            qkj = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout)
            qkj = warpgroup_mma(q_slice, k_bufs.index(s_qk).permute((1, 0)), qkj,
                                use_acc=False, is_async=True)       # QK(j) — group A
            acc = warpgroup_mma(P, v_bufs.index(s_pv), acc,
                                use_acc=True, is_async=True)        # PV(j-1) — group B
            qk = warpgroup_mma_wait(num_outstanding=1, deps=(qkj,)) # retire A, keep B
            # softmax(j) — runs on CUDA cores while PV(j-1) runs on tensor cores
            m_ij = gl.maximum(m_i, gl.max(qk, axis=1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = gl.exp2(qk)
            alpha = gl.exp2(m_i - m_ij)
            l_i = l_i * alpha + gl.sum(p, axis=1)
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))  # retire B (PV)
            mbarrier.arrive(kv_empty.index(s_pv), count=1)
            acc = acc * gl.convert_layout(alpha, row_o)[:, None]     # rescale AFTER PV
            m_i = m_ij
            P = gl.convert_layout(p.to(dtype), p_layout)
            kctr += 1

        # ---- epilogue: last PV (block N_ITERS-1) lives on stage (kctr-1) ----
        s_last = (kctr - 1) % kStages
        acc = warpgroup_mma(P, v_bufs.index(s_last), acc, use_acc=True, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        mbarrier.arrive(kv_empty.index(s_last), count=1)

        l_o = gl.convert_layout(l_i, row_o)
        acc = acc * (1.0 / l_o)[:, None]
        o_smem.store(acc.to(dtype))
        fence_async_shared()
        tma.async_copy_shared_to_global(desc_o, [seq_base + off_m, 0], o_smem)
        tma.store_wait(pendings=0)
        mbarrier.arrive(q_empty.index(qidx), count=1)
        qt += 1

        lse = m_i + gl.log2(l_i)
        offs_lse = (seq_base + off_m) + gl.arange(0, BLOCK_M, layout=row_qk)
        gl.store(M + offs_lse, lse)
        pid += NUM_SMS


@gluon.jit
def attn_fwd(desc_q, desc_k, desc_v, desc_o, M, N_CTX, sm_scale,
             num_pid_m, num_tiles, NUM_SMS, H_q, H_kv, G,
             BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, HEAD_DIM: gl.constexpr,
             kStages: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
             NUM_CONSUMERS: gl.constexpr, OVERLAP: gl.constexpr):
    dtype: gl.constexpr = desc_q.dtype
    BM_TOTAL: gl.constexpr = BLOCK_M * NUM_CONSUMERS

    q_bufs = gl.allocate_shared_memory(dtype, [QSTAGES, BM_TOTAL, HEAD_DIM], desc_q.layout)
    k_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_k.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [kStages, BLOCK_N, HEAD_DIM], desc_v.layout)
    q_ready = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    kv_empty = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    kv_ready = gl.allocate_shared_memory(gl.int64, [kStages, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(QSTAGES):
        mbarrier.init(q_ready.index(i), count=1)
        mbarrier.init(q_empty.index(i), count=NUM_CONSUMERS)
    for i in gl.static_range(kStages):
        mbarrier.init(kv_empty.index(i), count=NUM_CONSUMERS)
        mbarrier.init(kv_ready.index(i), count=1)

    if OVERLAP:
        gl.warp_specialize(
            [(_fwd_producer, (desc_q, desc_k, desc_v, q_bufs, k_bufs, v_bufs,
                              q_ready, q_empty, kv_empty, kv_ready, N_CTX,
                              num_pid_m, num_tiles, NUM_SMS, H_q, H_kv, G,
                              BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps, NUM_CONSUMERS)),
             (_fwd_consumer_ov, (desc_o, M, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                                 N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                                 BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps, NUM_CONSUMERS, 0)),
             (_fwd_consumer_ov, (desc_o, M, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                                 N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                                 BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps, NUM_CONSUMERS, 1))],
            [num_warps, num_warps], [240, 240])
    else:
        gl.warp_specialize(
            [(_fwd_producer, (desc_q, desc_k, desc_v, q_bufs, k_bufs, v_bufs,
                              q_ready, q_empty, kv_empty, kv_ready, N_CTX,
                              num_pid_m, num_tiles, NUM_SMS, H_q, H_kv, G,
                              BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps, NUM_CONSUMERS)),
             (_fwd_consumer, (desc_o, M, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                              N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                              BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps, NUM_CONSUMERS, 0)),
             (_fwd_consumer, (desc_o, M, q_bufs, k_bufs, v_bufs, q_ready, q_empty, kv_empty, kv_ready,
                              N_CTX, sm_scale, num_pid_m, num_tiles, NUM_SMS,
                              BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps, NUM_CONSUMERS, 1))],
            [num_warps, num_warps], [240, 240])


def attention_fwd(q, k, v, sm_scale=None, BLOCK_M=64, BLOCK_N=128, kStages=2,
                  num_warps=4, NUM_CONSUMERS=2, QSTAGES=2, emit=True, overlap=False):
    """Persistent non-causal FA forward. q: [Z, H_q, N, D]; k,v: [Z, H_kv, N, D] bf16.
    Supports GQA (H_q == G*H_kv; G query heads share one KV head) and head_dim 64/128.
    Returns (o, M) where M[z,h_q,i] = LSE_i (base-2) for the backward."""
    Z, H_q, N_CTX, HEAD_DIM = q.shape
    H_kv = k.shape[1]
    assert k.shape == (Z, H_kv, N_CTX, HEAD_DIM) and v.shape == k.shape
    assert H_q % H_kv == 0, "GQA needs H_q divisible by H_kv"
    G = H_q // H_kv
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    BM_TOTAL = BLOCK_M * NUM_CONSUMERS
    assert N_CTX % BLOCK_N == 0 and N_CTX % BM_TOTAL == 0, "tutorial 03 sweep uses N%128==0"

    q2 = q.reshape(Z * H_q * N_CTX, HEAD_DIM)
    k2 = k.reshape(Z * H_kv * N_CTX, HEAD_DIM)
    v2 = v.reshape(Z * H_kv * N_CTX, HEAD_DIM)
    o = torch.empty_like(q)
    o2 = o.reshape(Z * H_q * N_CTX, HEAD_DIM)
    M = torch.empty((Z * H_q * N_CTX,), dtype=torch.float32, device=q.device)

    ql = gl.NVMMASharedLayout.get_default_for([BM_TOTAL, HEAD_DIM], gl.bfloat16)
    kl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    vl = gl.NVMMASharedLayout.get_default_for([BLOCK_N, HEAD_DIM], gl.bfloat16)
    ol = gl.NVMMASharedLayout.get_default_for([BLOCK_M, HEAD_DIM], gl.bfloat16)
    desc_q = TensorDescriptor.from_tensor(q2, [BM_TOTAL, HEAD_DIM], ql)
    desc_k = TensorDescriptor.from_tensor(k2, [BLOCK_N, HEAD_DIM], kl)
    desc_v = TensorDescriptor.from_tensor(v2, [BLOCK_N, HEAD_DIM], vl)
    desc_o = TensorDescriptor.from_tensor(o2, [BLOCK_M, HEAD_DIM], ol)

    num_pid_m = _cdiv(N_CTX, BM_TOTAL)
    num_tiles = num_pid_m * Z * H_q
    NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (min(NUM_SMS, num_tiles),)
    # Pass emit_cuda EXPLICITLY (True=CUDA backend, False=PTX backend).  The per-launch
    # kwarg overrides the TRITON_EMIT_CUDA env var, so `emit=False` always means PTX —
    # the comparison no longer depends on any environment variable.
    kw = {"emit_cuda": bool(emit)}
    attn_fwd[grid](
        desc_q, desc_k, desc_v, desc_o, M, N_CTX, sm_scale,
        num_pid_m, num_tiles, grid[0], H_q, H_kv, G,
        BLOCK_M, BLOCK_N, HEAD_DIM, kStages, QSTAGES, num_warps=num_warps,
        NUM_CONSUMERS=NUM_CONSUMERS, OVERLAP=overlap, **kw)
    return o, M.reshape(Z, H_q, N_CTX)


# ===========================================================================
# BACKWARD — omni 5-GEMM recurrence (preprocess Delta + dK/dV + dQ)
# ===========================================================================
@triton.jit
def _bwd_preprocess(O, DO, Delta, DQ, N_CTX,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
                    ZERO_DQ: tl.constexpr):
    # Delta_i = sum_c O_ic · dO_ic  (= sum_k P_ik dP_ik); omni readme §6.1/§7.1.
    # ZERO_DQ (v2/v3): also clear the bf16 dQ buffer that the WS kernel reduce-adds
    # into — this folds the separate dq_acc.zero_() launch into a kernel we already
    # run, removing one launch (~7-11µs at small N, the small-shape bwd bottleneck).
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    idx = off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    o = tl.load(O + idx)
    do = tl.load(DO + idx).to(tl.float32)
    delta = tl.sum(o.to(tl.float32) * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)
    if ZERO_DQ:
        tl.store(DQ + idx, tl.zeros([BLOCK_M, HEAD_DIM], dtype=DQ.dtype.element_ty))


@triton.jit
def _bwd_dkdv_inner(dk, dv, Q, k, v, DO, M, D, stride_tok, stride_d,
                    N_CTX, dtype: tl.constexpr,
                    BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, HEAD_DIM: tl.constexpr,
                    start_n, start_m, num_steps):
    # outer-KV / inner-Q: compute S^T / P^T / dP^T / dS^T and feed dV, dK.  Both
    # accumulate in registers (one write each, NO atomics).  dQ is produced by the
    # separate outer-Q kernel below — keeping dQ out of here avoids the cross-CTA
    # global atomic that throttled the old fused kernel to ~40 TFLOP/s (the dk/dv-only
    # loop runs ~600 TFLOP/s, omni-class).
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    curr_m = start_m
    for _ in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)                       # S^T = (K·Q^T)·(sm_scale·log2e), k pre-scaled
        pT = tl.math.exp2(qkT - m[None, :])       # P^T (recompute, no re-read)
        do = tl.load(do_ptrs)
        ppT = pT.to(dtype)
        dv += tl.dot(ppT, do)                     # dV += P^T dO  (no scale)
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)   # dP^T = V dO^T
        dsT = (pT * (dpT - Di[None, :])).to(dtype)     # dS^T = P^T·(dP^T − Delta)
        dk += tl.dot(dsT, tl.trans(qT))           # dK += dS^T Q  (×sm_scale at end)
        curr_m += BLOCK_M1
        qT_ptrs += BLOCK_M1 * stride_tok
        do_ptrs += BLOCK_M1 * stride_tok
    return dk, dv


@triton.jit
def _bwd_dkdv_kernel(Q, K, V, sm_scale, DO, DK, DV, M, D,
                     stride_h, stride_tok, stride_d, N_CTX, H_q, H_kv, G,
                     BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
                     HEAD_DIM: tl.constexpr, dtype: tl.constexpr):
    # ONE program per (KV tile, KV head): dV, dK register-accumulated (single write,
    # no atomics).  GQA: the G query heads sharing this KV head are summed into the
    # same dk/dv registers.  dQ is computed by _bwd_dq_kernel.
    pid = tl.program_id(0)                        # KV tile index along N
    bhid = tl.program_id(1)                       # z*H_kv + h_kv  (KV-head space)
    z = bhid // H_kv
    h_kv = bhid % H_kv
    kv_off = (bhid * stride_h).to(tl.int64)
    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    k = tl.load(K + kv_off + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + kv_off + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    num_steps = N_CTX // BLOCK_M1                 # NON-CAUSAL: scan ALL Q tiles
    for g in range(G):
        h_q = h_kv * G + g
        q_off = ((z * H_q + h_q) * stride_h).to(tl.int64)
        m_off = ((z * H_q + h_q) * N_CTX).to(tl.int64)
        dk, dv = _bwd_dkdv_inner(dk, dv, Q + q_off, k, v, DO + q_off,
                                 M + m_off, D + m_off, stride_tok, stride_d, N_CTX, dtype,
                                 BLOCK_M1, BLOCK_N1, HEAD_DIM, start_n, 0, num_steps)
    tl.store(DV + kv_off + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, dv.to(dtype))
    dk *= sm_scale
    tl.store(DK + kv_off + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d, dk.to(dtype))


@triton.jit
def _bwd_dq_kernel(Q, K, V, sm_scale, ln2, DO, DQ, M, D,
                   stride_h, stride_tok, stride_d, N_CTX, H_q, H_kv, G,
                   BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
                   HEAD_DIM: tl.constexpr, dtype: tl.constexpr):
    # ONE program per (Q tile, Q head): dQ register-accumulated (single write, NO
    # atomics — this is what unblocks the ~40→omni-class perf).  Recomputes S/P/dP/dS
    # per KV tile (the only redundancy vs the fused kernel; far cheaper than the
    # cross-CTA dQ atomic it replaces).  K is loaded in BOTH orientations (kT for
    # S = q·kT, k for dQ += dS·k) so no tl.trans of a loaded tile is needed.
    pid = tl.program_id(0)                        # Q tile index along N
    bhid = tl.program_id(1)                       # z*H_q + h_q  (Q-head space)
    z = bhid // H_q
    h_q = bhid % H_q
    h_kv = h_q // G
    q_off = (bhid * stride_h).to(tl.int64)
    kv_off = ((z * H_kv + h_kv) * stride_h).to(tl.int64)
    m_off = (bhid * N_CTX).to(tl.int64)
    offs_k = tl.arange(0, HEAD_DIM)
    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + q_off + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    do = tl.load(DO + q_off + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + m_off + offs_m)
    Di = tl.load(D + m_off + offs_m)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N2)
    kT_ptrs = K + kv_off + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d   # [D,N]
    k_ptrs = K + kv_off + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d    # [N,D]
    vT_ptrs = V + kv_off + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d   # [D,N]
    num_steps = N_CTX // BLOCK_N2                  # NON-CAUSAL: scan ALL KV tiles
    for _ in range(num_steps):
        kT = tl.load(kT_ptrs)                      # [D, BLOCK_N2]  (K pre-scaled sm·log2e)
        qk = tl.dot(q, kT)                         # S = q·kT  [BLOCK_M2, BLOCK_N2]
        p = tl.math.exp2(qk - m[:, None])          # P (recompute)
        vT = tl.load(vT_ptrs)                      # [D, BLOCK_N2]
        dp = tl.dot(do, vT).to(tl.float32)         # dP = dO·V^T
        ds = (p * (dp - Di[:, None])).to(dtype)    # dS = P·(dP − Delta)
        k = tl.load(k_ptrs)                        # [BLOCK_N2, D]
        dq += tl.dot(ds, k)                        # dQ += dS·K  (K pre-scaled)
        kT_ptrs += BLOCK_N2 * stride_tok
        k_ptrs += BLOCK_N2 * stride_tok
        vT_ptrs += BLOCK_N2 * stride_tok
    dq *= ln2                                      # sm·log2e·ln2 = sm_scale
    tl.store(DQ + q_off + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d, dq.to(dtype))


# ===========================================================================
# _bwd_dq_finalize — scale (×ln2) and cast the cross-CTA dQ accumulator to bf16.
# ===========================================================================
# Backward dQ background (v2/v3): the WS kernel's producer reduces each KV-CTA's
# partial dQ into a global fp32 dq_accum via a TMA bulk reduce-add — instead of a
# per-element global atomic_add, which throttled the old fully-fused kernel to
# ~40 TFLOP/s.  ADD is commutative so concurrent CTA reduces sum correctly (fp32,
# ~1e-7 reorder — inside the bwd rel_l2 < 0.02 tol).  This tiny finalize kernel then
# applies ×ln2 (restoring ×sm_scale on the pre-scaled-K product) and casts to bf16.
@triton.jit
def _bwd_dq_finalize(DQ_ACC, DQ, ln2, N_CTX,
                     BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    # dq_accum (fp32, summed over KV-CTAs, carries factor sm·log2e) → dq (bf16, ×ln2
    # restores ×sm_scale: sm·log2e·ln2 = sm_scale).
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    idx = off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    acc = tl.load(DQ_ACC + idx)
    tl.store(DQ + idx, (acc * ln2).to(DQ.dtype.element_ty))


# ===========================================================================
# _bwd_v5 — WARP-SPECIALIZED backward (the kernel behind v2 and v3).
# ===========================================================================
# 1 producer warpgroup (TMA-loads Q/dO tiles into a pipeline and issues the dQ
# cross-CTA TMA bulk reduce-add) + 2 KV-row-split consumer warpgroups (compute the
# 5 GEMMs + softmax-bwd, accumulating dV/dK in registers).  The OVERLAP constexpr
# picks the consumer schedule:  OVERLAP=0 → v2 (plain WS, ~77% omni);  OVERLAP=5 →
# v3 (S/P/dP/dS kept M-major so dQ = dS·K uses dS as a register A-operand with NO
# smem round-trip, plus hoisted & one-iteration-ahead-prefetched LSE/Delta loads;
# 95-97% omni with the TRITON_BWD_INPLACE_RS_PACK / TRITON_EMIT_INPLACE_ELTWISE opts).
@gluon.jit
def _bwd_v5_producer(desc_q, desc_k, desc_v, desc_do, desc_dq,
                     k_smem, v_smem, q_bufs, do_bufs, q_ready, q_empty, kv_ready,
                     dq_handoff, dq_full, dq_empty,
                     N_CTX, H_q, H_kv, G: gl.constexpr,
                     BLOCK_M1: gl.constexpr, BLOCK_N1: gl.constexpr, HEAD_DIM: gl.constexpr,
                     QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                     NUM_CONSUMERS: gl.constexpr):
    pid = gl.program_id(0)
    bhid = gl.program_id(1)
    z = bhid // H_kv
    h_kv = bhid % H_kv
    start_n = pid * BLOCK_N1
    kv_row = bhid * N_CTX

    BN: gl.constexpr = BLOCK_N1 // NUM_CONSUMERS
    mbarrier.expect(kv_ready.index(0),
                    NUM_CONSUMERS * (desc_k.block_type.nbytes + desc_v.block_type.nbytes))
    for c in gl.static_range(NUM_CONSUMERS):
        tma.async_copy_global_to_shared(desc_k, [kv_row + start_n + c * BN, 0],
                                        kv_ready.index(0), k_smem.index(c))
        tma.async_copy_global_to_shared(desc_v, [kv_row + start_n + c * BN, 0],
                                        kv_ready.index(0), v_smem.index(c))

    num_steps = N_CTX // BLOCK_M1
    total = G * num_steps
    LAG: gl.constexpr = QSTAGES                  # reduce[rt] lags load[qt] so consumer has produced rt
    qt = 0
    while qt < total + LAG:
        # ---- REDUCE part (lagging index rt) FIRST so the consumer's dQ path is
        # always serviced before the producer can block on the next q_empty load.
        rt = qt - LAG
        if rt >= 0 and rt < total:
            if G == 1:
                g = 0
                step = rt
            else:
                g = rt // num_steps
                step = rt % num_steps
            h_q = h_kv * G + g
            m_off = (z * H_q + h_q) * N_CTX
            curr_m = step * BLOCK_M1
            buf = rt % 2
            rfp = (rt // 2) % 2
            for c in gl.static_range(NUM_CONSUMERS):
                bidx = c * 2 + buf
                mbarrier.wait(dq_full.index(bidx), rfp)
                tma.async_reduce_shared_to_global(desc_dq, [m_off + curr_m, 0],
                                                  dq_handoff.index(bidx), kind="add")
            tma.store_wait(pendings=NUM_CONSUMERS)   # keep rt's reduces in flight, drain rt-1's
            if rt >= 1:
                pbuf = (rt - 1) % 2
                for c in gl.static_range(NUM_CONSUMERS):
                    mbarrier.arrive(dq_empty.index(c * 2 + pbuf), count=1)
        # ---- LOAD part (index qt)
        if qt < total:
            if G == 1:
                g2 = 0
                step2 = qt
            else:
                g2 = qt // num_steps
                step2 = qt % num_steps
            h_q2 = h_kv * G + g2
            q_row = (z * H_q + h_q2) * N_CTX + step2 * BLOCK_M1
            qidx = qt % QSTAGES
            qeph = ((qt // QSTAGES) % 2) ^ 1
            mbarrier.wait(q_empty.index(qidx), qeph)
            bar = q_ready.index(qidx)
            mbarrier.expect(bar, desc_q.block_type.nbytes + desc_do.block_type.nbytes)
            tma.async_copy_global_to_shared(desc_q, [q_row, 0], bar, q_bufs.index(qidx))
            tma.async_copy_global_to_shared(desc_do, [q_row, 0], bar, do_bufs.index(qidx))
        qt += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _bwd_v5_consumer(desc_q, desc_k, DK, DV, M, Delta,
                     k_smem, v_smem, q_bufs, do_bufs, q_ready, q_empty, kv_ready,
                     dq_handoff, dq_full, dq_empty,
                     N_CTX, sm_scale, H_q, H_kv, G: gl.constexpr,
                     BLOCK_M1: gl.constexpr, BLOCK_N1: gl.constexpr, HEAD_DIM: gl.constexpr,
                     dsl: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
                     NUM_CONSUMERS: gl.constexpr, CONSUMER_ID: gl.constexpr,
                     OVERLAP: gl.constexpr = False):
    dtype: gl.constexpr = desc_k.dtype
    BN: gl.constexpr = BLOCK_N1 // NUM_CONSUMERS
    HD2: gl.constexpr = HEAD_DIM // NUM_CONSUMERS
    st_layout: gl.constexpr = mma_layout(dtype, BN, BLOCK_M1, num_warps)
    kd_layout: gl.constexpr = mma_layout(dtype, BN, HEAD_DIM, num_warps)
    qd_layout: gl.constexpr = mma_layout(dtype, BLOCK_M1, HEAD_DIM, num_warps)
    qh_layout: gl.constexpr = mma_layout(dtype, BLOCK_M1, HD2, num_warps)
    a_kd: gl.constexpr = dot_operand_layout(kd_layout, dtype)
    col_st: gl.constexpr = gl.SliceLayout(0, st_layout)
    # OVERLAP==5 (non-transposed orientation): S,P,dP,dS live in [M1,BN] (M-major)
    # so dQ=dS.K uses dS DIRECTLY as a register A-operand (NO smem round-trip — the
    # measured #1 stall).  dV/dK take the transpose round-trips instead, but they are
    # use_acc accumulators (overlappable), not the critical drained dQ.
    s_layout: gl.constexpr = mma_layout(dtype, BLOCK_M1, BN, num_warps)
    a_qd: gl.constexpr = dot_operand_layout(qd_layout, dtype)
    row_s: gl.constexpr = gl.SliceLayout(1, s_layout)

    pid = gl.program_id(0)
    bhid = gl.program_id(1)
    z = bhid // H_kv
    h_kv = bhid % H_kv
    start_n = pid * BLOCK_N1 + CONSUMER_ID * BN
    if OVERLAP == 5:
        # one-iteration-ahead prefetch prologue: load m5/Di5 for qt=0
        _moff0 = (z * H_q + h_kv * G) * N_CTX
        _om5p = gl.arange(0, BLOCK_M1, layout=row_s)
        m5 = gl.load(M + _moff0 + _om5p)
        Di5 = gl.load(Delta + _moff0 + _om5p)

    dsT_smem = gl.allocate_shared_memory(dtype, [BN, BLOCK_M1], dsl)

    mbarrier.wait(kv_ready.index(0), 0)
    k_c = k_smem.index(CONSUMER_ID)
    v_c = v_smem.index(CONSUMER_ID)

    dv = gl.zeros([BN, HEAD_DIM], gl.float32, layout=kd_layout)
    dk = gl.zeros([BN, HEAD_DIM], gl.float32, layout=kd_layout)
    num_steps = N_CTX // BLOCK_M1
    total = G * num_steps
    qt = 0
    while qt < total:
        if G == 1:
            g = 0
            step = qt
        else:
            g = qt // num_steps
            step = qt % num_steps
        h_q = h_kv * G + g
        m_off = (z * H_q + h_q) * N_CTX
        curr_m = step * BLOCK_M1
        qidx = qt % QSTAGES
        qph = (qt // QSTAGES) % 2
        # Issue the LSE(m)/Delta(Di) GLOBAL loads BEFORE the q_ready wait so their LDG
        # latency overlaps the barrier wait instead of stalling the consumer after it.
        # ncu: these per-iter global loads were the #1 stall (long_scoreboard 35.6%).
        if OVERLAP != 5:
            offs_m = curr_m + gl.arange(0, BLOCK_M1, layout=col_st)
            m = gl.load(M + m_off + offs_m)
            Di = gl.load(Delta + m_off + offs_m)
        # OVERLAP==5: m5/Di5 for THIS iter come from the prologue / one-iter-ahead carry.
        mbarrier.wait(q_ready.index(qidx), qph)
        q_s = q_bufs.index(qidx)
        do_s = do_bufs.index(qidx)

        qT = q_s.permute((1, 0))
        doT = do_s.permute((1, 0))
        if OVERLAP == 5:
            # prefetch NEXT iter's m5/Di5 NOW so the LDG overlaps this iter's long compute
            # (full hide of the #1 long_scoreboard stall). Clamp index on the last iter.
            _nqt = qt + 1
            if _nqt >= total:
                _nqt = qt
            if G == 1:
                _ng5 = 0
                _ns5 = _nqt
            else:
                _ng5 = _nqt // num_steps
                _ns5 = _nqt % num_steps
            _nmoff5 = (z * H_q + h_kv * G + _ng5) * N_CTX
            _nom5 = _ns5 * BLOCK_M1 + gl.arange(0, BLOCK_M1, layout=row_s)
            m5_next = gl.load(M + _nmoff5 + _nom5)
            Di5_next = gl.load(Delta + _nmoff5 + _nom5)
        if OVERLAP == 3:
            # EARLY dsT store (omni fa3_bwd.cu:786 stores ds before the dv/dk/dq wgmmas):
            # in the serial path dsT_smem.store+fence happens immediately before the dq
            # wgmma reads it back, so dq stalls on the store->visible latency (ncu: smem
            # scoreboard 3.1cyc, 36.7%).  Here we store dsT FIRST, then run dv and dk
            # (independent tensor-core work) whose execution hides the store latency, so
            # by the time dq reads dsT_smem the data is already visible -> no stall.
            st = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            st = warpgroup_mma(k_c, qT, st, use_acc=False, is_async=True)
            st = warpgroup_mma_wait(num_outstanding=0, deps=(st,))
            pT = gl.exp2(st * (sm_scale * 1.4426950408889634) - m[None, :])
            ppT = gl.convert_layout(pT.to(dtype), a_kd)
            dpT = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            dpT = warpgroup_mma(v_c, doT, dpT, use_acc=False, is_async=True)
            dpT = warpgroup_mma_wait(num_outstanding=0, deps=(dpT,))
            dsT = (pT * (dpT - Di[None, :])).to(dtype)
            dsT_smem.store(dsT)                 # store EARLY
            fence_async_shared()
            dsT_a = gl.convert_layout(dsT, a_kd)
            dq_part = gl.zeros([BLOCK_M1, HEAD_DIM], gl.float32, layout=qd_layout)
            dv = warpgroup_mma(ppT, do_s, dv, use_acc=True, is_async=True)
            dk = warpgroup_mma(dsT_a, q_s, dk, use_acc=True, is_async=True)
            # dv,dk run on the tensor core, hiding the dsT store->visible latency
            dq_part = warpgroup_mma(dsT_smem.permute((1, 0)), k_c, dq_part, use_acc=False, is_async=True)
            dv = warpgroup_mma_wait(num_outstanding=2, deps=(dv,))
            dk = warpgroup_mma_wait(num_outstanding=1, deps=(dk,))
            dq_part = warpgroup_mma_wait(num_outstanding=0, deps=(dq_part,))
        elif OVERLAP == 2:
            # TARGETED dq||dk fuse (omni fa3_bwd.cu:790-798): in the serial path dk is
            # fully RETIRED (wait 0) before dq is even issued, so dq's smem-operand read
            # (dsT_smem after the fence) has NOTHING to overlap -> the L1TEX shared
            # scoreboard stall (ncu: 3.1cyc, 36.7% of the issue gap) is fully exposed.
            # Here we store dsT, then issue dq (smem-source) AND dk (register-source,
            # independent) back-to-back async under ONE wait, so dk's matmul hides dq's
            # smem read latency.
            st = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            st = warpgroup_mma(k_c, qT, st, use_acc=False, is_async=True)
            st = warpgroup_mma_wait(num_outstanding=0, deps=(st,))
            pT = gl.exp2(st * (sm_scale * 1.4426950408889634) - m[None, :])
            ppT = gl.convert_layout(pT.to(dtype), a_kd)
            dv = warpgroup_mma(ppT, do_s, dv, use_acc=True, is_async=True)
            dv = warpgroup_mma_wait(num_outstanding=0, deps=(dv,))
            dpT = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            dpT = warpgroup_mma(v_c, doT, dpT, use_acc=False, is_async=True)
            dpT = warpgroup_mma_wait(num_outstanding=0, deps=(dpT,))
            dsT = (pT * (dpT - Di[None, :])).to(dtype)
            dsT_a = gl.convert_layout(dsT, a_kd)
            dsT_smem.store(dsT)
            fence_async_shared()
            dq_part = gl.zeros([BLOCK_M1, HEAD_DIM], gl.float32, layout=qd_layout)
            dq_part = warpgroup_mma(dsT_smem.permute((1, 0)), k_c, dq_part, use_acc=False, is_async=True)
            dk = warpgroup_mma(dsT_a, q_s, dk, use_acc=True, is_async=True)
            dq_part = warpgroup_mma_wait(num_outstanding=1, deps=(dq_part,))  # retire dq (oldest)
            dk = warpgroup_mma_wait(num_outstanding=0, deps=(dk,))            # retire dk
        elif OVERLAP == 4:
            # OMNI-ORDER ds-store hidden by dk (fa3_bwd.cu:777-790): the serial path
            # does dsT_smem.store -> fence -> dq back-to-back, so dq's smem-operand read
            # eats the full store->visible latency (ncu: L1TEX shared scoreboard 3.1cyc,
            # 36.7% of the issue gap).  Here we store dsT FIRST, then ISSUE dk (rs,
            # register-source, independent of dsT_smem) BEFORE dq, so dk occupies the
            # tensor-core pipe while dq is queued behind it; by the time dq actually
            # executes and reads dsT_smem the store is long visible -> stall hidden.
            st = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            st = warpgroup_mma(k_c, qT, st, use_acc=False, is_async=True)
            st = warpgroup_mma_wait(num_outstanding=0, deps=(st,))
            pT = gl.exp2(st * (sm_scale * 1.4426950408889634) - m[None, :])
            ppT = gl.convert_layout(pT.to(dtype), a_kd)
            dv = warpgroup_mma(ppT, do_s, dv, use_acc=True, is_async=True)
            dv = warpgroup_mma_wait(num_outstanding=0, deps=(dv,))
            dpT = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            dpT = warpgroup_mma(v_c, doT, dpT, use_acc=False, is_async=True)
            dpT = warpgroup_mma_wait(num_outstanding=0, deps=(dpT,))
            dsT = (pT * (dpT - Di[None, :])).to(dtype)
            dsT_a = gl.convert_layout(dsT, a_kd)
            dsT_smem.store(dsT)                  # store EARLY
            fence_async_shared()
            dq_part = gl.zeros([BLOCK_M1, HEAD_DIM], gl.float32, layout=qd_layout)
            dk = warpgroup_mma(dsT_a, q_s, dk, use_acc=True, is_async=True)        # issue dk FIRST
            dq_part = warpgroup_mma(dsT_smem.permute((1, 0)), k_c, dq_part, use_acc=False, is_async=True)  # dq behind dk
            dk = warpgroup_mma_wait(num_outstanding=1, deps=(dk,))                 # retire dk
            dq_part = warpgroup_mma_wait(num_outstanding=0, deps=(dq_part,))       # retire dq
        elif OVERLAP == 1:
            # PAIR the two independent input GEMMs (S^T=K.Q^T, dP^T=V.dO^T) so the
            # tensor core pipelines them, then OVERLAP the three independent output
            # GEMMs (dV+=P^T.dO, dK+=dS^T.Q, dQ=dS.K) under one wait.  This is FA3's
            # intra-tile wgmma overlap; it fills the tensor core that the serial
            # wgmma_wait(0)-per-GEMM path left idle (the 82%->100% compute gap).
            # omni's schedule (fa3_bwd.cu:708-760): issue BOTH input GEMMs, then
            # wgmma_wait(1) so dP^T=V.dO^T keeps running on the tensor core WHILE we
            # compute the exp2 softmax of S^T on the CUDA cores.  This is the actual
            # 82%->100% overlap; the prior wait(0)-both drained dP^T first (same reg
            # cost, ZERO overlap).  Output GEMMs stay serial to bound register pressure.
            st = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            dpT = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            st = warpgroup_mma(k_c, qT, st, use_acc=False, is_async=True)
            dpT = warpgroup_mma(v_c, doT, dpT, use_acc=False, is_async=True)
            st = warpgroup_mma_wait(num_outstanding=1, deps=(st,))  # retire S^T, dP^T runs
            pT = gl.exp2(st * (sm_scale * 1.4426950408889634) - m[None, :])                           # overlaps dP^T wgmma
            ppT = gl.convert_layout(pT.to(dtype), a_kd)
            dpT = warpgroup_mma_wait(num_outstanding=0, deps=(dpT,))
            dsT = (pT * (dpT - Di[None, :])).to(dtype)
            dsT_a = gl.convert_layout(dsT, a_kd)
            dv = warpgroup_mma(ppT, do_s, dv, use_acc=True, is_async=True)
            dv = warpgroup_mma_wait(num_outstanding=0, deps=(dv,))
            dk = warpgroup_mma(dsT_a, q_s, dk, use_acc=True, is_async=True)
            dk = warpgroup_mma_wait(num_outstanding=0, deps=(dk,))
            dsT_smem.store(dsT)
            fence_async_shared()
            dq_part = gl.zeros([BLOCK_M1, HEAD_DIM], gl.float32, layout=qd_layout)
            dq_part = warpgroup_mma(dsT_smem.permute((1, 0)), k_c, dq_part, use_acc=False, is_async=True)
            dq_part = warpgroup_mma_wait(num_outstanding=0, deps=(dq_part,))
        elif OVERLAP == 5:
            # NON-TRANSPOSED orientation (ncu-driven attempt).  *** TESTED NEGATIVE ***
            # Hypothesis: ncu proved the v5 gap is the dQ smem round-trip (dsT_smem.store
            # -> fence -> dq wgmma reads it back, L1TEX shared scoreboard 35.7% of the
            # issue stall).  Computing S/P/dP/dS in [M1,BN] (M-major) lets dQ=dS.K use dS
            # DIRECTLY as a REGISTER A-operand (ds_a), so dQ has NO smem dependency.  The
            # transpose round-trips move onto dV=P^T.dO and dK=dS^T.Q.
            # MEASURED (H800, _bench_bwd_95.py, same harness as OVERLAP=0):
            #   N=1024/2048/4096/8192 -> 46/48/49/53% of omni  vs  OVERLAP=0 53/60/66/74%.
            # REGRESSION.  Root cause: this does NOT remove a round-trip, it ADDS one --
            # v5 had ONE (dQ); here dV stores P AND dK stores dS, i.e. TWO round-trips,
            # both serialized through the single shared dsT_smem (dv read must retire
            # before dk overwrites).  Numerically correct (dq/dk/dv all PASS) but slower.
            # Kept only as a documented negative result; default stays OVERLAP=0.
            # m5/Di5 hoisted before the q_ready wait (see above) to hide LDG latency
            s = gl.zeros([BLOCK_M1, BN], gl.float32, layout=s_layout)
            s = warpgroup_mma(q_s, k_c.permute((1, 0)), s, use_acc=False, is_async=True)
            s = warpgroup_mma_wait(num_outstanding=0, deps=(s,))
            p = gl.exp2(s * (sm_scale * 1.4426950408889634) - m5[:, None])
            dp = gl.zeros([BLOCK_M1, BN], gl.float32, layout=s_layout)
            dp = warpgroup_mma(do_s, v_c.permute((1, 0)), dp, use_acc=False, is_async=True)
            dp = warpgroup_mma_wait(num_outstanding=0, deps=(dp,))
            ds = (p * (dp - Di5[:, None])).to(dtype)
            ds_a = gl.convert_layout(ds, a_qd)
            dq_part = gl.zeros([BLOCK_M1, HEAD_DIM], gl.float32, layout=qd_layout)
            # dQ = dS.K : register A-operand, NO round-trip -> issue immediately
            dq_part = warpgroup_mma(ds_a, k_c, dq_part, use_acc=False, is_async=True)
            # dV = P^T.dO : transpose via smem (store [M1,BN] into the [BN,M1] buffer's
            # permuted view; read it back as [BN,M1] = P^T) -- runs behind dq
            dsT_smem.permute((1, 0)).store(p.to(dtype))
            fence_async_shared()
            dv = warpgroup_mma(dsT_smem, do_s, dv, use_acc=True, is_async=True)
            dv = warpgroup_mma_wait(num_outstanding=0, deps=(dv,))
            # dK = dS^T.Q : reuse the same buffer (dv read is retired)
            dsT_smem.permute((1, 0)).store(ds)
            fence_async_shared()
            dk = warpgroup_mma(dsT_smem, q_s, dk, use_acc=True, is_async=True)
            dk = warpgroup_mma_wait(num_outstanding=0, deps=(dk,))
            dq_part = warpgroup_mma_wait(num_outstanding=0, deps=(dq_part,))
        elif OVERLAP == 6:
            # FAITHFUL omni output-trio overlap (fa3_bwd.cu:783-815): issue dV, dQ, dK
            # back-to-back, then wgmma_wait(1) retiring dV+dQ while dK stays on the
            # tensor core THROUGH the dq handoff (the ~12% reduce tax overlaps dK's
            # execution).  Input pair overlapped like OVERLAP==1 (dP^T || softmax).
            # dk is retired AFTER the handoff (guarded post-branch).
            st = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            dpT = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            st = warpgroup_mma(k_c, qT, st, use_acc=False, is_async=True)
            dpT = warpgroup_mma(v_c, doT, dpT, use_acc=False, is_async=True)
            st = warpgroup_mma_wait(num_outstanding=1, deps=(st,))
            pT = gl.exp2(st * (sm_scale * 1.4426950408889634) - m[None, :])
            ppT = gl.convert_layout(pT.to(dtype), a_kd)
            dpT = warpgroup_mma_wait(num_outstanding=0, deps=(dpT,))
            dsT = (pT * (dpT - Di[None, :])).to(dtype)
            dsT_a = gl.convert_layout(dsT, a_kd)
            dsT_smem.store(dsT)
            fence_async_shared()
            dq_part = gl.zeros([BLOCK_M1, HEAD_DIM], gl.float32, layout=qd_layout)
            dv = warpgroup_mma(ppT, do_s, dv, use_acc=True, is_async=True)
            dq_part = warpgroup_mma(dsT_smem.permute((1, 0)), k_c, dq_part, use_acc=False, is_async=True)
            dk = warpgroup_mma(dsT_a, q_s, dk, use_acc=True, is_async=True)
            dv, dq_part = warpgroup_mma_wait(num_outstanding=1, deps=(dv, dq_part))
        elif OVERLAP == 7:
            # omni REGISTER-ECONOMY dQ split (fa3_bwd.cu:58,789 dq_cols_per_wg=64,
            # float dq_acc[40]).  MEASURED root cause of the 70% plateau: the full
            # dq_part [BLOCK_M1,HEAD_DIM]=64 regs forces dv64+dk64+dq64=192 > the 168
            # __launch_bounds__(384,1) cap -> SPILL (STACK:152) under the output-trio
            # overlap, which is WHY OVERLAP==6 regresses to 60%.  Here dQ is computed as
            # TWO sequential HEAD_DIM column-halves, each a [BLOCK_M1,HD2]=32-reg
            # accumulator REUSED in place (the first is stored & dead before the second
            # allocates), so peak dq=32 -> dv64+dk64+dq32=160 <= 168, NO spill, WITH the
            # full omni overlap (dP^T||softmax input pair; dv,dk,dq0 trio pipelined).
            # Each half stores straight into its dq_handoff column slice; the per-tile
            # handoff is done in-branch (post-branch handoff guarded off for ==7).
            st = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            dpT = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            st = warpgroup_mma(k_c, qT, st, use_acc=False, is_async=True)
            dpT = warpgroup_mma(v_c, doT, dpT, use_acc=False, is_async=True)
            st = warpgroup_mma_wait(num_outstanding=1, deps=(st,))
            pT = gl.exp2(st * (sm_scale * 1.4426950408889634) - m[None, :])
            ppT = gl.convert_layout(pT.to(dtype), a_kd)
            dpT = warpgroup_mma_wait(num_outstanding=0, deps=(dpT,))
            dsT = (pT * (dpT - Di[None, :])).to(dtype)
            dsT_a = gl.convert_layout(dsT, a_kd)
            dsT_smem.store(dsT)
            fence_async_shared()
            dsT_t = dsT_smem.permute((1, 0))
            # dv, dk (persistent accumulators) + dq half-0 pipelined as a 3-wgmma trio
            dv = warpgroup_mma(ppT, do_s, dv, use_acc=True, is_async=True)
            dk = warpgroup_mma(dsT_a, q_s, dk, use_acc=True, is_async=True)
            buf = qt % 2
            bidx = CONSUMER_ID * 2 + buf
            eph = ((qt // 2) % 2) ^ 1
            mbarrier.wait(dq_empty.index(bidx), eph)
            dq0 = gl.zeros([BLOCK_M1, HD2], gl.float32, layout=qh_layout)
            dq0 = warpgroup_mma(dsT_t, k_c.slice(0, HD2, 1), dq0, use_acc=False, is_async=True)
            dv, dk, dq0 = warpgroup_mma_wait(num_outstanding=0, deps=(dv, dk, dq0))
            dq_handoff.index(bidx).slice(0, HD2, 1).store((dq0 * sm_scale).to(dtype))
            # dq half-1 reuses the now-dead 32-reg accumulator -> peak dq stays 32
            dq1 = gl.zeros([BLOCK_M1, HD2], gl.float32, layout=qh_layout)
            dq1 = warpgroup_mma(dsT_t, k_c.slice(HD2, HD2, 1), dq1, use_acc=False, is_async=True)
            dq1 = warpgroup_mma_wait(num_outstanding=0, deps=(dq1,))
            dq_handoff.index(bidx).slice(HD2, HD2, 1).store((dq1 * sm_scale).to(dtype))
            fence_async_shared()
            mbarrier.arrive(dq_full.index(bidx), count=1)
            mbarrier.arrive(q_empty.index(qidx), count=1)
        else:
            st = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            st = warpgroup_mma(k_c, qT, st, use_acc=False, is_async=True)
            st = warpgroup_mma_wait(num_outstanding=0, deps=(st,))
            pT = gl.exp2(st * (sm_scale * 1.4426950408889634) - m[None, :])
            ppT = gl.convert_layout(pT.to(dtype), a_kd)
            dv = warpgroup_mma(ppT, do_s, dv, use_acc=True, is_async=True)
            dv = warpgroup_mma_wait(num_outstanding=0, deps=(dv,))
            dpT = gl.zeros([BN, BLOCK_M1], gl.float32, layout=st_layout)
            dpT = warpgroup_mma(v_c, doT, dpT, use_acc=False, is_async=True)
            dpT = warpgroup_mma_wait(num_outstanding=0, deps=(dpT,))
            dsT = (pT * (dpT - Di[None, :])).to(dtype)
            dsT_a = gl.convert_layout(dsT, a_kd)
            dk = warpgroup_mma(dsT_a, q_s, dk, use_acc=True, is_async=True)
            dk = warpgroup_mma_wait(num_outstanding=0, deps=(dk,))
            dsT_smem.store(dsT)
            fence_async_shared()
            dq_part = gl.zeros([BLOCK_M1, HEAD_DIM], gl.float32, layout=qd_layout)
            dq_part = warpgroup_mma(dsT_smem.permute((1, 0)), k_c, dq_part, use_acc=False, is_async=True)
            dq_part = warpgroup_mma_wait(num_outstanding=0, deps=(dq_part,))
        # HAND OFF dq_part to the producer's reduce: store into this consumer's
        # parity buffer, signal dq_full.  No store_wait here -> compute path clean.
        # OVERLAP==7 does its handoff in-branch (column-half stores), so skip here.
        if OVERLAP != 7:
            buf = qt % 2
            bidx = CONSUMER_ID * 2 + buf
            eph = ((qt // 2) % 2) ^ 1
            mbarrier.wait(dq_empty.index(bidx), eph)
            dq_handoff.index(bidx).store((dq_part * sm_scale).to(dtype))
            fence_async_shared()
            mbarrier.arrive(dq_full.index(bidx), count=1)
            mbarrier.arrive(q_empty.index(qidx), count=1)
        if OVERLAP == 6:
            # dK was left in flight by the output-trio overlap; it ran on the tensor
            # core WHILE the dq handoff (mbarrier wait + smem store + arrive) executed.
            dk = warpgroup_mma_wait(num_outstanding=0, deps=(dk,))
        if OVERLAP == 5:
            m5 = m5_next
            Di5 = Di5_next
        qt += 1

    kv_row = bhid * N_CTX
    offs_n = start_n + gl.arange(0, BN, layout=gl.SliceLayout(1, kd_layout))
    offs_d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, kd_layout))
    dkv_idx = (kv_row + offs_n)[:, None] * HEAD_DIM + offs_d[None, :]
    gl.store(DV + dkv_idx, dv.to(dtype))
    gl.store(DK + dkv_idx, (dk * sm_scale).to(dtype))


@gluon.jit
def _bwd_v5(desc_q, desc_k, desc_v, desc_do, desc_dq, DK, DV, M, Delta,
            N_CTX, sm_scale, H_q, H_kv, G: gl.constexpr,
            BLOCK_M1: gl.constexpr, BLOCK_N1: gl.constexpr, HEAD_DIM: gl.constexpr,
            dsl: gl.constexpr, QSTAGES: gl.constexpr, num_warps: gl.constexpr,
            NUM_CONSUMERS: gl.constexpr, OVERLAP: gl.constexpr = False):
    dtype: gl.constexpr = desc_k.dtype
    BN: gl.constexpr = BLOCK_N1 // NUM_CONSUMERS
    k_smem = gl.allocate_shared_memory(dtype, [NUM_CONSUMERS, BN, HEAD_DIM], desc_k.layout)
    v_smem = gl.allocate_shared_memory(dtype, [NUM_CONSUMERS, BN, HEAD_DIM], desc_v.layout)
    q_bufs = gl.allocate_shared_memory(dtype, [QSTAGES, BLOCK_M1, HEAD_DIM], desc_q.layout)
    do_bufs = gl.allocate_shared_memory(dtype, [QSTAGES, BLOCK_M1, HEAD_DIM], desc_do.layout)
    # dQ handoff: one parity-pair per consumer -> [NUM_CONSUMERS*2, M1, D] bf16.
    dq_handoff = gl.allocate_shared_memory(dtype, [NUM_CONSUMERS * 2, BLOCK_M1, HEAD_DIM], desc_dq.layout)
    q_ready = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    q_empty = gl.allocate_shared_memory(gl.int64, [QSTAGES, 1], mbarrier.MBarrierLayout())
    kv_ready = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    dq_full = gl.allocate_shared_memory(gl.int64, [NUM_CONSUMERS * 2, 1], mbarrier.MBarrierLayout())
    dq_empty = gl.allocate_shared_memory(gl.int64, [NUM_CONSUMERS * 2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(QSTAGES):
        mbarrier.init(q_ready.index(i), count=1)
        mbarrier.init(q_empty.index(i), count=NUM_CONSUMERS)
    mbarrier.init(kv_ready.index(0), count=1)
    for i in gl.static_range(NUM_CONSUMERS * 2):
        mbarrier.init(dq_full.index(i), count=1)
        mbarrier.init(dq_empty.index(i), count=1)

    gl.warp_specialize(
        [(_bwd_v5_producer, (desc_q, desc_k, desc_v, desc_do, desc_dq,
                             k_smem, v_smem, q_bufs, do_bufs, q_ready, q_empty, kv_ready,
                             dq_handoff, dq_full, dq_empty,
                             N_CTX, H_q, H_kv, G,
                             BLOCK_M1, BLOCK_N1, HEAD_DIM, QSTAGES, num_warps, NUM_CONSUMERS)),
         (_bwd_v5_consumer, (desc_q, desc_k, DK, DV, M, Delta,
                             k_smem, v_smem, q_bufs, do_bufs, q_ready, q_empty, kv_ready,
                             dq_handoff, dq_full, dq_empty,
                             N_CTX, sm_scale, H_q, H_kv, G,
                             BLOCK_M1, BLOCK_N1, HEAD_DIM, dsl, QSTAGES, num_warps, NUM_CONSUMERS, 0, OVERLAP)),
         (_bwd_v5_consumer, (desc_q, desc_k, DK, DV, M, Delta,
                             k_smem, v_smem, q_bufs, do_bufs, q_ready, q_empty, kv_ready,
                             dq_handoff, dq_full, dq_empty,
                             N_CTX, sm_scale, H_q, H_kv, G,
                             BLOCK_M1, BLOCK_N1, HEAD_DIM, dsl, QSTAGES, num_warps, NUM_CONSUMERS, 1, OVERLAP))],
        [num_warps, num_warps], [240, 240])


# ===========================================================================
# attention_bwd — host dispatch over the 3 kept backward versions (v1/v2/v3).
# ===========================================================================
def attention_bwd(q, k, v, o, do, M, sm_scale=None, emit=True, version=1,
                  BLOCK_M1=64, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=64,
                  num_warps=8, num_stages=3):
    """Non-causal FA backward. q,o,do: [Z,H_q,N,D]; k,v: [Z,H_kv,N,D] bf16;
    M: [Z,H_q,N] LSE (base-2).  Returns (dq, dk, dv).  Split decomposition (omni
    readme §8 option B/C): preprocess Delta, then a dK/dV kernel (outer-KV, inner-Q,
    register-accumulated, no atomics) and a dQ kernel (outer-Q, inner-KV, register-
    accumulated, no atomics).  Splitting dQ out of the dK/dV loop removes the cross-CTA
    global atomic that throttled the old fused kernel ~15× (40 → omni-class TFLOP/s);
    the dQ recompute of S/P is far cheaper than that atomic.  GQA-aware: the dK/dV
    kernel sums the G query heads sharing each KV head into the dk/dv registers."""
    Z, H_q, N_CTX, HEAD_DIM = q.shape
    H_kv = k.shape[1]
    G = H_q // H_kv
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    assert q.stride() == o.stride() == do.stride()
    assert k.stride() == v.stride()
    # per-head flattened layout: a [*,H,N,D] tensor has stride (H*N*D, N*D, D, 1);
    # head h of batch z lives at (z*H + h) * (N*D).  stride_tok=D, stride_d=1.
    stride_h = N_CTX * HEAD_DIM
    stride_tok = HEAD_DIM
    stride_d = 1
    dtype = tl.bfloat16
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    PRE_BLOCK = 128
    assert N_CTX % PRE_BLOCK == 0 and N_CTX % BLOCK_N1 == 0 and N_CTX % BLOCK_M1 == 0
    assert N_CTX % BLOCK_M2 == 0 and N_CTX % BLOCK_N2 == 0
    # Pass emit_cuda EXPLICITLY (True=CUDA backend, False=PTX backend).  The per-launch
    # kwarg overrides the TRITON_EMIT_CUDA env var, so `emit=False` always means PTX —
    # the comparison no longer depends on any environment variable.
    kw = {"emit_cuda": bool(emit)}

    delta = torch.empty_like(M)
    pre_grid = (N_CTX // PRE_BLOCK, Z * H_q)
    # v2/v3 reduce-add the WS kernel's partial dQ straight into `dq`; preprocess clears
    # it here so we drop both the separate dq_acc.zero_() and the _bwd_dq_finalize launch
    # (the two auxiliary launches that dominated the small-N bwd, 13-29% at N≤2048).
    _zero_dq = version in (2, 3)
    _bwd_preprocess[pre_grid](o, do, delta, dq, N_CTX, BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM,
                              ZERO_DQ=_zero_dq, **kw)

    # K is pre-scaled by sm_scale·log2(e) so the kernels' exp2 needs no extra mul.
    if version == 1:
        # v1 pre-scales K by sm·log2e on the host (its Triton kernels expect αK).
        arg_k = k * (sm_scale * LOG2E)
        # v1 — SPLIT baseline (Triton, non-warp-specialized). Two passes: a dK/dV kernel
        # (outer-KV, inner-Q, register-accumulated, no atomics) and a dQ kernel (outer-Q,
        # inner-KV, register-accumulated, single store). Splitting dQ out of the dK/dV loop
        # removes the cross-CTA global atomic that throttled the naive fully-fused kernel
        # ~15× (40 → omni-class TFLOP/s). Clearest correct baseline. ~30-46% of omni.
        dkdv_grid = (N_CTX // BLOCK_N1, Z * H_kv)
        _bwd_dkdv_kernel[dkdv_grid](
            q, arg_k, v, sm_scale, do, dk, dv, M, delta,
            stride_h, stride_tok, stride_d, N_CTX, H_q, H_kv, G,
            BLOCK_M1, BLOCK_N1, HEAD_DIM, dtype,
            num_warps=num_warps, num_stages=num_stages, **kw)
        dq_grid = (N_CTX // BLOCK_M2, Z * H_q)
        _bwd_dq_kernel[dq_grid](
            q, arg_k, v, sm_scale, math.log(2.0), do, dq, M, delta,
            stride_h, stride_tok, stride_d, N_CTX, H_q, H_kv, G,
            BLOCK_M2, BLOCK_N2, HEAD_DIM, dtype,
            num_warps=num_warps, num_stages=num_stages, **kw)
        return dq, dk, dv

    if version in (2, 3):
        # v2/v3 — WARP-SPECIALIZED fused (kernel _bwd_v5): 1 producer warpgroup + 2
        # KV-row-split consumer warpgroups, with the producer issuing the dQ TMA bulk
        # reduce. v2 is the plain WS schedule (~77% of omni). v3 runs the SAME kernel with
        # the OVERLAP=5 schedule (S/P/dP/dS kept M-major so dQ = dS·K uses dS DIRECTLY as a
        # register A-operand — no dQ smem round-trip), plus hoisted global LSE/Delta
        # (m/Di and m5/Di5) loads and ONE-ITERATION-AHEAD m5/Di5 prefetch that fully hides
        # the per-iteration global-load latency. v3 ≈ 95-97% of omni @ D128 N≥8192.
        # NOTE: v3 requires the emitter opts  TRITON_BWD_INPLACE_RS_PACK=1  and
        # TRITON_EMIT_INPLACE_ELTWISE=1  (they free the registers that make OVERLAP=5 fit).
        os.environ["BWD_V5_OVERLAP"] = "5" if version == 3 else "0"
        # dq was already zeroed by _bwd_preprocess (ZERO_DQ); the WS kernel reduce-adds
        # sm_scale-prescaled partials straight into it (the consumer folds ×sm_scale at
        # handoff), so there is NO separate fp32 dq_acc and NO finalize pass.  K is passed
        # RAW — the consumer applies α=sm·log2e inside the softmax (exp2(st·α−m)), so there
        # is also NO arg_k = k·α materialization (one fewer launch + bf16 tensor; the
        # ~9-18µs that bounded the small-N bwd).  GQA-safe (no per-head scale tensor).
        qf = q.reshape(Z * H_q * N_CTX, HEAD_DIM)
        dof = do.reshape(Z * H_q * N_CTX, HEAD_DIM)
        kf = k.reshape(Z * H_kv * N_CTX, HEAD_DIM)
        vf = v.reshape(Z * H_kv * N_CTX, HEAD_DIM)
        dqf = dq.reshape(Z * H_q * N_CTX, HEAD_DIM)
        nw = 4
        QSTAGES = int(os.environ.get("BWD_V5_QS", "2"))
        NUM_CONSUMERS = 2
        BN = BLOCK_N1 // NUM_CONSUMERS
        _OVL = int(os.environ.get("BWD_V5_OVERLAP", "5"))
        _HD2 = HEAD_DIM // NUM_CONSUMERS
        _kdsw: gl.constexpr = (min(128, _HD2 * 2) if _OVL == 7 else 128)
        ql = gl.NVMMASharedLayout.get_default_for([BLOCK_M1, HEAD_DIM], gl.bfloat16)
        kl = gl.NVMMASharedLayout(swizzle_byte_width=_kdsw, element_bitwidth=16, rank=2)
        dql = gl.NVMMASharedLayout(swizzle_byte_width=_kdsw, element_bitwidth=16, rank=2)
        desc_q = TensorDescriptor.from_tensor(qf, [BLOCK_M1, HEAD_DIM], ql)
        desc_do = TensorDescriptor.from_tensor(dof, [BLOCK_M1, HEAD_DIM], ql)
        desc_k = TensorDescriptor.from_tensor(kf, [BN, HEAD_DIM], kl)
        desc_v = TensorDescriptor.from_tensor(vf, [BN, HEAD_DIM], kl)
        desc_dq = TensorDescriptor.from_tensor(dqf, [BLOCK_M1, HEAD_DIM], dql)
        dsl = gl.NVMMASharedLayout.get_default_for([BN, BLOCK_M1], gl.bfloat16)
        fused_grid = (N_CTX // BLOCK_N1, Z * H_kv)
        _bwd_v5[fused_grid](
            desc_q, desc_k, desc_v, desc_do, desc_dq, dk, dv, M, delta,
            N_CTX, sm_scale, H_q, H_kv, G,
            BLOCK_M1, BLOCK_N1, HEAD_DIM, dsl, QSTAGES, NUM_CONSUMERS=NUM_CONSUMERS,
            OVERLAP=_OVL, num_warps=nw, **kw)
        return dq, dk, dv

    raise ValueError(f"unknown backward version {version}; tutorial keeps v1, v2, v3")


# ===========================================================================
# References (PyTorch) — non-causal MHA, the omni_kernel reference definitions.
# ===========================================================================
def _gqa_expand(t, H_q):
    """Repeat each KV head G times along the head axis so it lines up with H_q query
    heads.  [Z,H_kv,N,D] -> [Z,H_q,N,D] with query head h reading KV head h//G."""
    Z, H_kv, N, D = t.shape
    G = H_q // H_kv
    return t.repeat_interleave(G, dim=1)


def _ref_fwd_lse(q, k, v, sm_scale):
    """Returns (o, lse_log2) with the omni convention lse = log2(sum 2^(scores·log2e)).
    GQA: k,v may have fewer heads than q; expand them first."""
    Z, H_q, N, D = q.shape
    qf = q.float()
    kf = _gqa_expand(k, H_q).float()
    vf = _gqa_expand(v, H_q).float()
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * sm_scale     # [Z,H_q,N,N]
    probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs, vf).to(q.dtype)
    scores_log2 = scores * LOG2E
    lse = torch.logsumexp(scores_log2 * math.log(2.0), dim=-1) / math.log(2.0)
    return o, lse


def _ref_bwd(q, k, v, do, sm_scale):
    """Autograd ground truth for non-causal GQA backward.  Differentiate w.r.t. the
    REAL (unexpanded) k,v so dk/dv come back summed over the G query heads."""
    qf = q.float().detach().requires_grad_(True)
    kf = k.float().detach().requires_grad_(True)
    vf = v.float().detach().requires_grad_(True)
    H_q = q.shape[1]
    ke = _gqa_expand(kf, H_q)
    ve = _gqa_expand(vf, H_q)
    scores = torch.matmul(qf, ke.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    o = torch.matmul(probs, ve)
    o.backward(do.float())
    return qf.grad, kf.grad, vf.grad


def main():
    if not is_hopper():
        raise RuntimeError("Tutorial 03 targets Hopper (sm90).")
    # The showcase backward is v3, whose schedule relies on two emit_cuda-INTERNAL
    # optimizations (in-place bf16 RS-pack + in-place elementwise reuse).  Enable them
    # here so the tutorial is self-contained.  NOTE: these are emitter feature flags that
    # ONLY affect the emit_cuda (emit=True) path — they have no effect on the PTX backend,
    # and they are NOT how the backend is selected (that is the per-launch emit= kwarg).
    os.environ.setdefault("TRITON_BWD_INPLACE_RS_PACK", "1")
    os.environ.setdefault("TRITON_EMIT_INPLACE_ELTWISE", "1")
    # Non-causal shapes.  (D, N, Z, H_q, H_kv).  H_kv<H_q => GQA (G=H_q//H_kv).
    # N must be divisible by the bwd tiles (BLOCK_N1=128, BLOCK_M2=128) and the fwd
    # BLOCK_M=64 -> all N below are multiples of 128 (incl. non-pow2 1536/3072/768).
    SWEEP = [
        # MHA (H_kv == H_q), power-of-2 seqlens, both head dims
        (64, 1024, 4, 16, 16), (64, 2048, 2, 16, 16), (64, 4096, 2, 8, 8),
        (128, 1024, 4, 16, 16), (128, 2048, 2, 16, 16), (128, 4096, 2, 8, 8),
        # non-power-of-2 seqlens
        (128, 1536, 2, 16, 16), (128, 3072, 2, 8, 8), (64, 768, 4, 16, 16),
        (64, 1536, 2, 16, 16),
        # GQA (G=4 and G=8), both head dims, incl. a non-pow2 seqlen
        (128, 2048, 2, 16, 4), (128, 1536, 2, 16, 2), (64, 2048, 2, 16, 4),
        (128, 4096, 2, 8, 1), (64, 1024, 4, 16, 4),
    ]
    TOL = 5e-3
    BTOL = 2e-2   # backward accumulates more bf16 rounding (autograd ref is fp32)

    # L2-flush timer: median of `iters` graph-free launches, each preceded by an
    # L2 wipe so every kernel runs from a cold cache (apples-to-apples CUDA vs PTX).
    _flush = torch.empty(int(50e6 // 4), dtype=torch.float32, device="cuda")

    def _bench(fn, warmup=8, iters=30):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            _flush.zero_(); e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort(); return ts[len(ts) // 2]   # ms

    print("=" * 116)
    print("Tutorial 03 — NON-CAUSAL FlashAttention forward + backward")
    print("compares the emit_cuda CUDA backend (emit=True) vs the stock PTX backend (emit=False)")
    print("on BOTH performance (TFLOP/s) AND correctness (rel_l2 vs PyTorch fp32 reference).")
    print("forward = attention_fwd (persistent WS);  backward = attention_bwd version=3 (WS, v3).")
    print("`speedup` = PTX_time / CUDA_time  (>1.0 => CUDA backend faster). GQA when H_kv<H_q.")
    print("emit=True/False is passed as a per-launch kwarg — NO env var (TRITON_EMIT_CUDA) needed.")
    print("=" * 116)
    hdr = (f"{'D':>3} {'N':>6} {'Z':>2} {'Hq':>3} {'Hkv':>2} {'G':>2} | "
           f"{'fwd CUDA':>8} {'fwd PTX':>8} {'spd':>5} {'ok':>3} | "
           f"{'bwd CUDA':>8} {'bwd PTX':>8} {'spd':>5} {'ok':>3}")
    print(hdr); print("-" * 116)

    failures = []
    fwd_spd, bwd_spd = [], []
    for D, N, Z, H_q, H_kv in SWEEP:
        G = H_q // H_kv
        torch.manual_seed(N + D + H_kv)
        q = torch.randn(Z, H_q, N, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(Z, H_kv, N, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        do = torch.randn_like(q) * 0.1
        sm = 1.0 / math.sqrt(D)
        Ffwd = 4.0 * Z * H_q * N * N * D
        Fbwd = 10.0 * Z * H_q * N * N * D

        o_ref, lse_ref = _ref_fwd_lse(q, k, v, sm)
        dq_ref, dk_ref, dv_ref = _ref_bwd(q, k, v, do, sm)

        def _rel(a, b):
            return (a.float() - b.float()).norm().item() / b.float().norm().clamp_min(1e-9).item()

        row_ok = True
        cells = []
        for tag, Fl, run, ref_ok in (
            ("fwd", Ffwd, lambda emit: attention_fwd(q, k, v, sm_scale=sm, emit=emit),
             lambda out: _rel(out[0], o_ref) < TOL and _rel(out[1], lse_ref) < TOL),
            ("bwd", Fbwd, lambda emit: attention_bwd(q, k, v, o_c, do, M_c, sm_scale=sm, emit=emit, version=3),
             lambda out: (_rel(out[0], dq_ref) < BTOL and _rel(out[1], dk_ref) < BTOL
                          and _rel(out[2], dv_ref) < BTOL)),
        ):
            out_c = run(True); torch.cuda.synchronize()
            if tag == "fwd":
                o_c, M_c = out_c                       # bwd consumes the CUDA forward outputs
            out_p = run(False); torch.cuda.synchronize()
            ok = ref_ok(out_c) and ref_ok(out_p)
            row_ok = row_ok and ok
            t_c = _bench(lambda: run(True))
            t_p = _bench(lambda: run(False))
            tf_c = Fl / (t_c * 1e-3) / 1e12
            tf_p = Fl / (t_p * 1e-3) / 1e12
            spd = t_p / t_c
            (fwd_spd if tag == "fwd" else bwd_spd).append(spd)
            cells.append((tf_c, tf_p, spd, ok))

        (fc, fp, fs, fok), (bc, bp, bs, bok) = cells
        if not row_ok:
            failures.append((D, N, H_q, H_kv))
        print(f"{D:>3} {N:>6} {Z:>2} {H_q:>3} {H_kv:>2} {G:>2} | "
              f"{fc:8.0f} {fp:8.0f} {fs:4.2f}x {'OK' if fok else '!!':>3} | "
              f"{bc:8.0f} {bp:8.0f} {bs:4.2f}x {'OK' if bok else '!!':>3}")
        del q, k, v, do, o_c, M_c; torch.cuda.empty_cache()

    print("-" * 116)
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0
    print(f"geomean speedup (CUDA vs PTX):  fwd {gm(fwd_spd):.2f}x   bwd {gm(bwd_spd):.2f}x")
    print("=" * 116)
    if failures:
        raise AssertionError(f"correctness FAILED: {failures}")
    print("tutorial 03 PASSED — CUDA & PTX backends both correct "
          f"(fwd rel_l2 < {TOL}, bwd rel_l2 < {BTOL}); TFLOP/s in bf16, TF/s = TFLOP/s.")


if __name__ == "__main__":
    main()
