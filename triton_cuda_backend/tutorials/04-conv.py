"""
Tutorial 04 — Dense Conv3d/2d on the Triton CUDA backend (fwd + bwd)
===================================================================

Port of omni_kernel's hand-written SM90 dense Conv3d (NDHWC implicit-GEMM) to the
out-of-tree ``emit_cuda`` CUDA-C++ backend. All changes are confined to
``triton_cuda_backend`` (zero Triton-source edits).

Layout / contract (mirrors omni_kernel/ops/conv/conv3d.py):
    input  : NDHWC          [N, D, H, W, Cin]       bf16, contiguous
    weight : KTHWC (C-last) [Cout, kT, kH, kW, Cin] bf16  == panel [Cout, K]
    output : NDHWC          [N, oD, oH, oW, Cout]    bf16  == [M, Cout]
where  M = N*oD*oH*oW,  K = kT*kH*kW*Cin,  oX = (X - kX)//sX + 1,
       K column order = [kt, kh, kw, cin] (cin innermost), tap = (kt*kH+kh)*kW+kw.
Constraints: groups=1, dilation=1, stride>=1, no padding, Cin/Cout %64==0.
Correctness oracle: fp32 ``F.conv3d`` (NDHWC<->NCDHW permute), rel-Frobenius < 2e-2.

DESIGN
------
Forward / dgrad / wgrad are implicit GEMMs over the affine im2col index map
M=N*oD*oH*oW, K=taps*Cin. Two forward backends, dispatched per-shape (timed once,
cached) to the faster:
  (A) MANUAL-GATHER: a ``tl.dot`` GEMM with a masked-load im2col gather (autotuned,
      split-K). Wins on huge-channel / small / strided shapes.
  (B) HW TMA-IM2COL (NEW): a warp-specialized Gluon kernel (1 producer + 2
      consumers) whose producer issues a real Hopper ``cp.async.bulk.tensor...
      im2col`` — the conv gather done by the TMA engine. Wins on large-spatial /
      compute-bound shapes (incl. H/W=1k/2k/4k: ~1.0-1.27x of omni production).
This TMA-im2col path is implemented ENTIRELY in triton_cuda_backend (zero Triton
edits): a backend MLIR pass (Im2colRewritePass) synthesizes the im2col op from a
tagged ``inline_asm_elementwise("im2col.marker")`` carrier + placeholder copy, the
emitter prints the ``.im2col`` PTX, and the launcher (``inject_im2col_launcher``)
builds the im2col CUtensorMap via Triton's own ``fill_tma_descriptor_im2col``.
dgrad/wgrad use (A).

  * fwd        : out[M,Cout] = A_im2col(input,pad=0)[M,K] @ W[Cout,K]^T  (one GEMM)
  * dgrad s==1 : "dgrad-as-fprop" — one fused GEMM reusing the fwd kernel with
                 grad_out im2col'd at full padding (k-1) and a flipped/transposed
                 weight panel Wflip[Cin, taps*Cout]. No scatter, no atomics.
  * dgrad s>1  : single-launch transpose-conv GATHER (each input voxel sums its
                 contributing output x tap; overwrite store, no scatter/atomics).
  * wgrad      : one fused GEMM dW[Cout,K] = grad_out[M,Cout]^T @ A_im2col(input)[M,K]
                 reducing over M, with split-K (fp32 scratch sum) to fill SMs when
                 the (Cout x K) tile count is small. Deterministic (fixed sum order).
  * gbias      : grad_out.sum(0).

KEY PERF LEVERS (each was necessary to clear 95% on every shape):
  * scalar-tap gather: Cin/Cout are 64-multiples and BLOCK_K<=CC, so each K-tile
    lies within ONE tap => (kt,kh,kw)+channel-start are scalars (not per-element).
  * hoisted voxel base + HAS_PAD-gated bounds (forward never goes OOB).
  * coalesced weight load ([BLOCK_N,BLOCK_K] then tl.trans).
  * split-K (fwd + wgrad) for small-tile, occupancy-bound shapes.
  * LOW-OVERHEAD LAUNCH: do NOT use triton.autotune — its per-call Python dispatch
    (~25us) silently dominated the 40us stride-2 forward (measured 58us vs a true
    30us kernel!). _tuned_launch tunes once per shape-key then launches the raw JIT
    fn directly. This single change turned the stride-2 forward from 0.75x -> 1.05x.

MEASURED forward TFLOP/s vs omni_kernel PRODUCTION conv (csrc/varlen_conv
standard_fused — the real HW-TMA-im2col kernel), H800/sm90 bf16. ours = per-shape
best of {gather, TMA-im2col}; omni numbers are offline-measured and EMBEDDED below
(users have no omni_kernel). TFLOP/s = 2*M*Cout*K / t; peak bf16 ~989.

  shape (Cin->Cout, k, s)         ours   omni   ours/omni   backend
  128->256  k333 s1               462    486     0.95x      gather (64x256 nw4)
   64->128  k333 s1               340    282     1.21x      gather
  256->256  k333 s1               519    534     0.97x      im2col (BN=128)
  128->256  k333 s2  (small)      182    187     0.98x      gather (BK=128,runner)
  320->640  k133 s1               316    313     1.01x      gather
  512->512  k333 s1               650    556     1.17x      im2col
  2048->2048 k333 s1             531    505     1.05x      gather split-K=2
  4096->4096 k333 s1 (tiny M)    413    423     0.98x      gather split-K=2
  128->256  k333 s1 (128^3 sp)   611    517     1.18x      im2col
  256->256  k133 s1 (H/W=1k)     618    523     1.18x      im2col
  128->256  k133 s1 (H/W=2k)     566    486     1.16x      im2col
   64->128  k133 s1 (H/W=4k)     364    298     1.22x      im2col
  ------------------------------------------------------------
  vs omni-production: geomean ~1.08x, **12/12 shapes >= 0.95x** (min 0.95). The
  forward dispatch picks, per shape (timed once, cached), the fastest of {gather x
  split-K(1/2/4), TMA-im2col x BN(128/256)}: TMA-im2col wins large-spatial (incl.
  all H/W=1k/2k/4k) and the medium compute-bound 256->256 (BN=128, NOT 256: the
  m64n256 accumulator starves occupancy and halves it to ~0.48x); gather wins
  huge-channel (split-K=2 lifts 4096-ch tiny-M from 0.78->1.02x), moderate-K
  (128->256, 64x256 nw4 tile) and strided. The tiny stride-2 conv (M~3.4k, ~12us
  kernel) was the hardest: its kernel is actually ~1.3x of omni, but the per-call
  Python of the dispatch/wrapper chain (~6us) buried it to ~0.84x -- a cached lean
  launch runner (one alloc + one kernel launch, like omni's single C++ op) restored
  it to ~0.98x. BLOCK_K=128 (halves the K-loop, valid when Cin%128==0) also helped.
  ours is also ~10-40x faster than cuDNN F.conv3d. Numbers are clean per-shape on
  idle H800s; the live `bench` dips under shared-GPU contention.

Run `python3 04-conv.py bench` to reproduce the live table (times OURS + cuDNN;
omni is the embedded offline reference, no omni_kernel needed). Each shape is timed
in a fresh subprocess (cool context, robust to shared-GPU contention) and WITHOUT
bias -- matching how the omni reference was captured (the gather path fuses bias for
free; the im2col path adds it as a separate ~4GB elementwise pass on big-spatial
shapes, which would be an unfair, unfused-epilogue artifact against no-bias omni).
"""
import os

import torch
import triton
import triton.language as tl

# Gluon (for the warp-specialized HW TMA-im2col forward path). Imported lazily-safe:
# if unavailable, the manual-gather path still works.
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as _GTD
    from triton.experimental.gluon.language.nvidia.hopper import (
        tma as _gtma, mbarrier as _gmbar, fence_async_shared as _gfence,
        warpgroup_mma as _gwgmma, warpgroup_mma_wait as _gwgmma_wait)
    _HAVE_GLUON = True
except Exception:
    _HAVE_GLUON = False


def _cdiv(a, b):
    return (a + b - 1) // b


# Config = (BLOCK dict, num_warps, num_stages). We deliberately do NOT use
# triton.autotune: its per-call Python dispatch (~25us) dominates small/strided
# convs (a ~40us kernel). Instead a tiny cache (_tuned_launch) tunes once per
# shape-key then launches the raw JIT fn directly, recovering the true speed.
_GEMM_CFGS = [
    ({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, 4, 3),
    ({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, 8, 3),
    ({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, 8, 3),
    ({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, 8, 3),
    ({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, 8, 4),
    ({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, 4, 3),
    # Medium compute-bound + moderate-K (e.g. 128->256, M~25k, K~3.5k): a wide
    # N tile with nw=4 + 4 stages reaches ~0.96x where the nw=8 variants stall at
    # ~0.80-0.86x.
    ({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, 4, 4),
    # BLOCK_K=128 (only used when Cin/CC is a 128-multiple -- see the CC%bk filter
    # in _run_implicit_gemm): halves the K-loop, which lifts the tiny stride-2 conv
    # (M~3k, K~3.5k) from ~0.94x to ~0.96x of omni -- the deciding config for the
    # last sub-0.95 shape. scalar-tap stays valid because BLOCK_K==CC (one tap).
    ({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, 4, 4),
]
# wgrad: BLOCK_K pinned 64 (scalar-tap needs BLOCK_K<=Cin, a 64-multiple); MMA
# width via larger BLOCK_CO.
_WGRAD_CFGS = [
    ({"BLOCK_CO": 64, "BLOCK_K": 64, "BLOCK_M": 64}, 4, 3),
    ({"BLOCK_CO": 128, "BLOCK_K": 64, "BLOCK_M": 64}, 8, 3),
    ({"BLOCK_CO": 256, "BLOCK_K": 64, "BLOCK_M": 64}, 8, 3),
    ({"BLOCK_CO": 128, "BLOCK_K": 64, "BLOCK_M": 128}, 8, 4),
    ({"BLOCK_CO": 64, "BLOCK_K": 64, "BLOCK_M": 128}, 4, 3),
    ({"BLOCK_CO": 256, "BLOCK_K": 64, "BLOCK_M": 128}, 8, 4),
]

_TUNE_CACHE = {}
_CFG_FILTER_CACHE = {}


def _cfgs_for_cc(cc):
    """_GEMM_CFGS with only the configs whose BLOCK_K divides the channel count cc
    (scalar-tap tap-alignment), cached so the tiny/strided shapes don't pay a
    per-call list comprehension on their ~16us critical path."""
    out = _CFG_FILTER_CACHE.get(cc)
    if out is None:
        out = [c for c in _GEMM_CFGS if cc % c[0]["BLOCK_K"] == 0]
        _CFG_FILTER_CACHE[cc] = out
    return out


def _tuned_launch(cache_key, cfgs, launch):
    """Pick the fastest config for cache_key once (timed), cache it, then call
    launch(cfg) on every invocation — avoiding triton.autotune's per-call dispatch
    overhead (decisive for tiny/strided convs). launch(cfg) does one kernel launch
    given a config dict {BLOCK_..., 'num_warps', 'num_stages'}. All three conv
    kernels are overwrite-store, so re-running configs during tuning is safe."""
    cfg = _TUNE_CACHE.get(cache_key)
    if cfg is None:
        # Spin up GPU clocks first so the first configs aren't penalised, then time
        # each config as best-of-3 rounds (min) to be robust to contention on a
        # shared GPU — a noisy single measurement can otherwise cache a bad config.
        for _ in range(10):
            try:
                launch(cfgs[0])
            except Exception:
                pass
        torch.cuda.synchronize()
        best, best_t = None, float("inf")
        for c in cfgs:
            try:
                # Per-config warmup (clocks + I-cache) so configs timed later in the
                # list aren't penalised; then best-of-3 x 40 iters. Tiny kernels
                # (e.g. stride-2, M~3k) need enough iters to rank reliably -- 10 was
                # too noisy and mis-picked a 156 TF config over the true 228 TF one.
                for _ in range(5):
                    launch(c)
                torch.cuda.synchronize()
                dt = float("inf")
                for _ in range(3):
                    st = torch.cuda.Event(enable_timing=True)
                    en = torch.cuda.Event(enable_timing=True)
                    st.record()
                    for _ in range(40):
                        launch(c)
                    en.record()
                    torch.cuda.synchronize()
                    dt = min(dt, st.elapsed_time(en))
            except Exception:
                continue
            if dt < best_t:
                best_t, best = dt, c
        cfg = best if best is not None else cfgs[0]
        _TUNE_CACHE[cache_key] = cfg
    launch(cfg)


# ===========================================================================
# General padded implicit-GEMM: out[M_out, NN] = A_im2col[M_out, K] @ B[NN, K]^T
#   A gathers SRC (NDHWC, CC channels) over the output grid (gD,gH,gW) with
#   per-tap window  src_coord = out*stride - pad + tap, OOB -> 0 (zero pad).
#   K order = [kt, kh, kw, cc] (cc innermost); B panel is [NN, K] (read as B^T).
# Forward       : SRC=input(Cin),  grid=(oD,oH,oW), pad=0,     B=W[Cout,K]   -> out[M,Cout]
# dgrad stride1 : SRC=grad_out(Cout), grid=(D,H,W), pad=k-1, stride=1,
#                 B=Wflip[Cin, taps*Cout] -> grad_input[N*DHW, Cin]  (dgrad-as-fprop)
# ===========================================================================
@triton.jit
def _conv3d_implicit_gemm_kernel(
    src_ptr, w_ptr, b_ptr, out_ptr,
    N, sD, sH, sW, CC, NN, gD, gH, gW,
    kT, kH, kW, stT, stH, stW, pT, pH, pW, M, K, NSPLIT,
    HAS_BIAS: tl.constexpr, HAS_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)   # split-K index over the contraction K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < NN

    gHW = gH * gW
    gTHW = gD * gHW
    sHsW = sH * sW
    n_o = offs_m // gTHW
    rem = offs_m % gTHW
    t_o = rem // gHW
    rt = rem % gHW
    h_o = rt // gW
    w_o = rt % gW
    d_base = t_o * stT - pT
    h_base = h_o * stH - pH
    w_base = w_o * stW - pW

    # This split's BLOCK_K-aligned slice of the K reduction.
    n_ktiles = tl.cdiv(K, BLOCK_K)
    ktiles_per_split = tl.cdiv(n_ktiles, NSPLIT)
    k_lo = pid_s * ktiles_per_split * BLOCK_K
    k_hi = min(k_lo + ktiles_per_split * BLOCK_K, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    khkw = kH * kW
    arange_k = tl.arange(0, BLOCK_K)
    # Cin/Cout are 64-multiples and BLOCK_K<=CC with CC%BLOCK_K==0, so each
    # 64-aligned K-tile lies within ONE tap => (kt,kh,kw)+channel-start are SCALARS,
    # making per-tap window offsets [BLOCK_M] (not [BLOCK_M,BLOCK_K]) — removes the
    # dominant integer-gather overhead. No-pad fwd: voxel = base_lin + scalar
    # (hoisted, no bounds check). Padded (dgrad-as-fprop) keeps per-tap window+mask.
    base_lin = ((n_o * sD + d_base) * sH + h_base) * sW + w_base
    for k0 in range(k_lo, k_hi, BLOCK_K):
        ktile = k0 // CC
        cc0 = k0 - ktile * CC
        kw = ktile % kW
        kh = (ktile // kW) % kH
        kt = ktile // khkw
        cc = cc0 + arange_k
        if HAS_PAD:
            d_i = d_base + kt
            h_i = h_base + kh
            w_i = w_base + kw
            in_bounds = ((d_i >= 0) & (d_i < sD) & (h_i >= 0) & (h_i < sH)
                         & (w_i >= 0) & (w_i < sW))
            voxel = ((n_o * sD + d_i) * sH + h_i) * sW + w_i
            a = tl.load(src_ptr + voxel[:, None] * CC + cc[None, :],
                        mask=m_mask[:, None] & in_bounds[:, None], other=0.0)
        else:
            voxel = base_lin + (kt * sHsW + kh * sW + kw)
            a = tl.load(src_ptr + voxel[:, None] * CC + cc[None, :],
                        mask=m_mask[:, None], other=0.0)
        offs_k = k0 + arange_k
        # B (=W panel [NN,K]) loaded coalesced [BLOCK_N,BLOCK_K] then transposed.
        bt = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                     mask=n_mask[:, None], other=0.0)
        acc += tl.dot(a, tl.trans(bt))

    if HAS_BIAS:
        if pid_s == 0:
            bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]
    out_off = (pid_s * M + offs_m[:, None]) * NN + offs_n[None, :]
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


# ===========================================================================
# Fused wgrad: dW[Cout, K] = grad_out[M, Cout]^T @ A_im2col(input)[M, K], one
# launch reducing over M (deterministic, fixed reduction order).
# ===========================================================================
@triton.jit
def _conv3d_wgrad_kernel(
    go_ptr, x_ptr, dw_ptr,
    N, D, H, W, Cin, Cout, oD, oH, oW,
    kT, kH, kW, sT, sH, sW, M, K, NSPLIT,
    BLOCK_CO: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_co = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_s = tl.program_id(2)   # split-K index over the M reduction
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    arange_k = tl.arange(0, BLOCK_K)
    offs_k = pid_k * BLOCK_K + arange_k   # over K = taps*Cin
    co_mask = offs_co < Cout
    # Scalar tap for the whole K-tile (BLOCK_K<=Cin, Cin%BLOCK_K==0 -> within one tap).
    ktile = (pid_k * BLOCK_K) // Cin
    cc0 = pid_k * BLOCK_K - ktile * Cin
    kw = ktile % kW
    kh = (ktile // kW) % kH
    kt = ktile // (kH * kW)
    cin = cc0 + arange_k                   # [BLOCK_K] channels within this tap

    # This split's contiguous slice of the M reduction (BLOCK_M-aligned).
    n_mtiles = tl.cdiv(M, BLOCK_M)
    tiles_per_split = tl.cdiv(n_mtiles, NSPLIT)
    m_lo = pid_s * tiles_per_split * BLOCK_M
    m_hi = min(m_lo + tiles_per_split * BLOCK_M, M)

    HW_o = oH * oW
    THW_o = oD * HW_o
    acc = tl.zeros((BLOCK_CO, BLOCK_K), dtype=tl.float32)
    for m0 in range(m_lo, m_hi, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        m_mask = offs_m < m_hi
        go = tl.load(go_ptr + offs_m[:, None] * Cout + offs_co[None, :],
                     mask=m_mask[:, None] & co_mask[None, :], other=0.0)  # [M, CO]
        n_o = offs_m // THW_o
        rem = offs_m % THW_o
        t_o = rem // HW_o
        rt = rem % HW_o
        h_o = rt // oW
        w_o = rt % oW
        d_i = t_o * sT + kt
        h_i = h_o * sH + kh
        w_i = w_o * sW + kw
        voxel = ((n_o * D + d_i) * H + h_i) * W + w_i   # [BLOCK_M]
        x_off = voxel[:, None] * Cin + cin[None, :]
        xt = tl.load(x_ptr + x_off, mask=m_mask[:, None], other=0.0)  # [M, K]
        acc += tl.dot(tl.trans(go), xt)  # [CO, K]

    # Write this split's partial into slab pid_s (rows [pid_s*Cout, ...)).
    dw_off = (pid_s * Cout + offs_co[:, None]) * K + offs_k[None, :]
    tl.store(dw_ptr + dw_off, acc.to(dw_ptr.dtype.element_ty),
             mask=co_mask[:, None])


# ===========================================================================
# dgrad (gather form, any stride): for each input voxel, sum over the (tap, co)
# that map to it. grad_input[m_in,Cin] = sum_tap sum_co grad_out[vox(m_in,tap),co]
# * W[co, tap*Cin+cin], with per-row validity masks. ONE launch, overwrite store
# (autotune-safe), no scatter / atomics / fp32 buffer; deterministic.
# ===========================================================================
@triton.jit
def _conv3d_dgrad_gather_kernel(
    go_ptr, w_ptr, gi_ptr,
    N, D, H, W, Cin, Cout, oD, oH, oW,
    kT, kH, kW, sT, sH, sW, M_in, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # input voxel
    offs_cin = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M_in
    cin_mask = offs_cin < Cin

    HW = H * W
    DHW = D * HW
    n_ = offs_m // DHW
    r = offs_m % DHW
    d_ = r // HW
    r2 = r % HW
    h_ = r2 // W
    w_ = r2 % W

    khkw = kH * kW
    taps = kT * khkw
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tap in range(0, taps):
        kt = tap // khkw
        rem = tap % khkw
        kh = rem // kW
        kw = rem % kW
        dn = d_ - kt
        hn = h_ - kh
        wn = w_ - kw
        od = dn // sT
        oh = hn // sH
        ow = wn // sW
        valid = ((dn >= 0) & (dn - od * sT == 0) & (od < oD)
                 & (hn >= 0) & (hn - oh * sH == 0) & (oh < oH)
                 & (wn >= 0) & (wn - ow * sW == 0) & (ow < oW))     # [BLOCK_M]
        src = ((n_ * oD + od) * oH + oh) * oW + ow                  # [BLOCK_M]
        for c0 in range(0, Cout, BLOCK_K):
            offs_co = c0 + tl.arange(0, BLOCK_K)
            co_mask = offs_co < Cout
            go = tl.load(go_ptr + src[:, None] * Cout + offs_co[None, :],
                         mask=m_mask[:, None] & valid[:, None] & co_mask[None, :], other=0.0)
            wt = tl.load(w_ptr + offs_co[:, None] * K + (tap * Cin + offs_cin)[None, :],
                         mask=co_mask[:, None] & cin_mask[None, :], other=0.0)
            acc += tl.dot(go, wt)

    gi_off = offs_m[:, None] * Cin + offs_cin[None, :]
    tl.store(gi_ptr + gi_off, acc.to(gi_ptr.dtype.element_ty),
             mask=m_mask[:, None] & cin_mask[None, :])


# ===========================================================================
# Host glue.
# ===========================================================================
def _emit_kw(emit):
    return {"emit_cuda": True} if emit else {}


_NUM_SM = 132


def _pick_ksplit(M, NN, K):
    """split-K factor for an implicit GEMM: split the K reduction to fill SMs when
    the (M x NN) output-tile count is small (occupancy-bound small/strided convs)."""
    base = _cdiv(M, 64) * _cdiv(NN, 64)
    k_iters = _cdiv(K, 64)
    if base > 0 and base * 2 <= _NUM_SM:
        return max(1, min(_NUM_SM // base, 8, max(1, k_iters // 2)))
    return 1


def _run_implicit_gemm(src, wpanel, bias, out2d, M, NN, K, dims, has_bias, emit,
                       force_nsplit=None):
    """Launch the implicit-GEMM with optional split-K; writes [M, NN] into out2d.
    dims = (N, sD, sH, sW, CC, gD, gH, gW, kT, kH, kW, stT, stH, stW, pT, pH, pW)."""
    (Nn, sD, sH, sW, CC, gD, gH, gW, kT, kH, kW, stT, stH, stW, pT, pH, pW) = dims
    nsplit = force_nsplit if force_nsplit is not None else _pick_ksplit(M, NN, K)
    has_pad = (pT != 0) or (pH != 0) or (pW != 0)
    scratch = torch.empty((nsplit * M, NN), device=src.device, dtype=torch.bfloat16) if nsplit > 1 else out2d
    b_arg = bias if has_bias else src

    def launch(cfg):
        bm, bn, bk = cfg[0]["BLOCK_M"], cfg[0]["BLOCK_N"], cfg[0]["BLOCK_K"]
        grid = (_cdiv(M, bm), _cdiv(NN, bn), nsplit)
        _conv3d_implicit_gemm_kernel[grid](
            src, wpanel, b_arg, scratch,
            Nn, sD, sH, sW, CC, NN, gD, gH, gW,
            kT, kH, kW, stT, stH, stW, pT, pH, pW, M, K, nsplit,
            HAS_BIAS=has_bias, HAS_PAD=has_pad,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=cfg[1], num_stages=cfg[2], **_emit_kw(emit))

    # scalar-tap correctness REQUIRES each BLOCK_K tile to stay within one tap, i.e.
    # BLOCK_K must divide the channel count CC (CC=Cin for fwd, Cout for dgrad-s1).
    # A BLOCK_K=128 config on a Cin=64 shape would straddle two taps -> wrong. Drop
    # configs whose BLOCK_K does not divide CC (cached, see _cfgs_for_cc).
    cfgs = _cfgs_for_cc(CC)
    if emit:
        _tuned_launch(("gemm", M, NN, K, CC, nsplit, has_pad), cfgs, launch)
    else:
        launch(cfgs[0])
    if nsplit > 1:
        out2d.copy_(scratch.reshape(nsplit, M, NN).float().sum(0).to(torch.bfloat16))


def _conv3d_forward_gather(x, w, bias=None, stride=(1, 1, 1), emit=True, nsplit=None):
    """Forward via the manual masked-load im2col gather + tl.dot (autotuned)."""
    N, D, H, W, Cin = x.shape
    Cout, kT, kH, kW, Cin_w = w.shape
    sT, sH, sW = stride
    oD = (D - kT) // sT + 1
    oH = (H - kH) // sH + 1
    oW = (W - kW) // sW + 1
    M = N * oD * oH * oW
    K = kT * kH * kW * Cin
    out = torch.empty((N, oD, oH, oW, Cout), device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return out
    x_c = x.contiguous()
    w_c = w.contiguous().reshape(Cout, K)
    has_bias = bias is not None
    b_c = bias.contiguous() if has_bias else x_c
    dims = (N, D, H, W, Cin, oD, oH, oW, kT, kH, kW, sT, sH, sW, 0, 0, 0)
    _run_implicit_gemm(x_c, w_c, b_c, out.view(M, Cout), M, Cout, K, dims, has_bias, emit,
                       force_nsplit=nsplit)
    return out


_FWD_IMPL_CACHE = {}
_GATHER_RUNNER = {}


def _make_gather_runner(N, D, H, W, Cin, Cout, kT, kH, kW, sT, sH, sW,
                        M, K, oD, oH, oW, cfg, has_bias):
    """Build a lean per-shape launch closure for the nsplit=1 gather forward.

    The kernel itself is ~1.3x of omni; on the tiny/strided shapes the dominant
    cost is the per-call Python of the conv3d_forward -> gather -> run_implicit_gemm
    -> tuned_launch chain (~6us on a ~12us kernel). Once the dispatch + config are
    chosen, we cache this closure so steady-state calls do ONE allocation + ONE
    kernel launch, skipping all those layers (a single C++ op like omni's)."""
    bm, bn, bk = cfg[0]["BLOCK_M"], cfg[0]["BLOCK_N"], cfg[0]["BLOCK_K"]
    nw, nstg = cfg[1], cfg[2]
    grid = (_cdiv(M, bm), _cdiv(Cout, bn), 1)
    kern = _conv3d_implicit_gemm_kernel
    out_shape = (N, oD, oH, oW, Cout)
    dev_dtype = torch.bfloat16

    def runner(x, w, bias):
        x = x if x.is_contiguous() else x.contiguous()
        wp = w.reshape(Cout, K)
        out = torch.empty(out_shape, device=x.device, dtype=dev_dtype)
        kern[grid](
            x, wp, bias if has_bias else x, out.view(M, Cout),
            N, D, H, W, Cin, Cout, oD, oH, oW,
            kT, kH, kW, sT, sH, sW, 0, 0, 0, M, K, 1,
            HAS_BIAS=has_bias, HAS_PAD=False,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=nw, num_stages=nstg, emit_cuda=True)
        return out

    return runner


def conv3d_forward(x, w, bias=None, stride=(1, 1, 1), emit=True):
    """NDHWC bf16 conv3d forward. Dispatches per-shape (timed once, cached) between
    the manual-gather GEMM and the HW TMA-im2col warp-specialized kernel, picking
    the faster. TMA im2col wins on large-spatial / compute-bound shapes; the
    gather wins on huge-channel / small / strided shapes."""
    assert x.is_cuda and w.is_cuda and x.dtype == torch.bfloat16
    N, D, H, W, Cin = x.shape
    Cout, kT, kH, kW, Cin_w = w.shape
    assert Cin == Cin_w
    sT, sH, sW = stride
    use_ws = (emit and _HAVE_GLUON and Cin % 64 == 0 and Cout % 64 == 0)
    if not use_ws:
        return _conv3d_forward_gather(x, w, bias, stride, emit)

    key = (N, D, H, W, Cin, Cout, kT, kH, kW, sT, sH, sW)
    # Steady-state fast path: a cached lean launch closure for the nsplit=1 gather
    # winner, skipping the dispatch/wrapper/tuner call chain (see _make_gather_runner).
    runner = _GATHER_RUNNER.get((key, bias is not None))
    if runner is not None:
        return runner(x, w, bias)

    oD = (D - kT) // sT + 1
    oH = (H - kH) // sH + 1
    oW = (W - kW) // sW + 1
    M = N * oD * oH * oW
    K = kT * kH * kW * Cin
    impl = _FWD_IMPL_CACHE.get(key)
    if impl is None:
        # Candidates: gather with a few split-K factors (huge-K/tiny-M shapes need
        # split-K=2-4; small/strided need 1) + the TMA-im2col kernel. Time each
        # (prewarmed, best-of-3) once per shape; cache the winner.
        # im2col BN candidates: 128 always (best for medium/compute-bound), plus
        # 256 for large Cout (wider tile amortizes there). The dispatch tag's 2nd
        # field is the split-K factor for gather, or the BN for im2col.
        # Offer split-K (timer decides) EXCEPT when the split-K scratch offset
        # (idx * M * Cout, idx up to nsplit-1=3) would overflow int32 -> illegal
        # memory access. This only bites the giant-spatial shapes (e.g. 2048x2048,
        # M~4.2M); the tiny-M / huge-channel shapes (4096->4096 M=392, 2048->2048
        # M=1800, stride-2 M=3375) all stay well under the limit and genuinely need
        # split-K for latency hiding (huge-K reduction), so they keep it.
        gather_ns = [1] + ([2, 4] if M * Cout * 4 < (1 << 31) else [])
        im2col_bns = [128, 256] if Cout >= 256 else [128]
        cands = [("gather", ns) for ns in gather_ns] + \
                [("im2col", bn) for bn in im2col_bns]
        fns = {
            ("gather", ns): (lambda ns=ns: _conv3d_forward_gather(x, w, None, stride, emit, nsplit=ns))
            for ns in gather_ns}
        for bn in im2col_bns:
            fns[("im2col", bn)] = (lambda bn=bn: _conv3d_forward_im2col(x, w, stride, BN=bn))
        best, best_t = ("gather", 1), float("inf")
        for c in cands:
            try:
                fn = fns[c]
                for _ in range(4):
                    fn()
                torch.cuda.synchronize()
                t = _time_call(fn)
                if t < best_t:
                    best_t, best = t, c
            except Exception:
                continue
        impl = best
        _FWD_IMPL_CACHE[key] = impl

    kind, ns = impl
    if kind == "im2col":
        out = _conv3d_forward_im2col(x, w, stride, BN=ns)
        if bias is not None:
            out = out + bias
        return out
    # gather: first call tunes the config; then build & cache the lean runner so
    # subsequent calls skip the wrapper layers (decisive for the tiny/strided shape
    # whose ~12us kernel was otherwise buried under ~6us of per-call Python).
    out = _conv3d_forward_gather(x, w, bias, stride, emit, nsplit=ns)
    if emit and ns == 1:
        cfg = _TUNE_CACHE.get(("gemm", M, Cout, K, Cin, 1, False))
        if cfg is not None:
            _GATHER_RUNNER[(key, bias is not None)] = _make_gather_runner(
                N, D, H, W, Cin, Cout, kT, kH, kW, sT, sH, sW,
                M, K, oD, oH, oW, cfg, bias is not None)
    return out


def _time_call(fn, iters=30, warmup=12, rounds=3):
    # Warmup absorbs first-call compile (nvcc) + the gather path's autotune; then
    # best-of-`rounds` min timing to be robust to shared-GPU contention so the
    # cached gather-vs-im2col dispatch decision is reliable.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en))
    return best


# ===========================================================================
# Warp-specialized HW TMA-im2col forward (1 producer + 2 consumers). The producer
# issues one HW im2col load (the conv gather, done by the TMA engine) per K-iter
# via the `_im2col_load` helper -> the backend Im2colRewritePass + emitter turn it
# into a real `cp.async.bulk.tensor...im2col`; the consumers WGMMA + TMA-store.
# Only defined when Gluon is available.
# ===========================================================================
if _HAVE_GLUON:

    @gluon.constexpr_function
    def _wpc(BM, nw):
        wpc = [4, 1]
        while wpc[0] * wpc[1] != nw:
            if BM > 16 * wpc[0]:
                wpc[0] *= 2
            else:
                wpc[1] *= 2
        return wpc

    @gluon.constexpr_function
    def _mma_layout(BM, BN, nw):
        return gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=_wpc(BM, nw),
                                         instr_shape=[16, BN, 16])

    @gluon.jit
    def _im2col_load(a_desc, c, w, h, d, n, kw, kh, kt, bar, smem, pred):
        # Carrier marker (consumed + erased by Im2colRewritePass) + the placeholder
        # tiled copy it rewrites into a real HW im2col copy. We use a TAGGED inline-
        # asm op rather than device_print: its whole purpose is to hand operands +
        # a verbatim string to the backend (no print-IO semantics to abuse, no
        # prefix mangling). is_pure=False so Triton's TTGIR DCE keeps it alive until
        # our backend pass runs; the asm body ("im2col.marker") is never emitted --
        # the pass deletes the op first. The 8 i32s are the pixel base (c,w,h,d,n)
        # and the kernel taps (kw,kh,kt); the dummy result is discarded.
        gl.inline_asm_elementwise(
            asm="im2col.marker", constraints="=r,r,r,r,r,r,r,r,r",
            args=[c, w, h, d, n, kw, kh, kt],
            dtype=gl.int32, is_pure=False, pack=1)
        _gtma.async_copy_global_to_shared(a_desc, [0, 0], bar, smem, pred)

    @gluon.jit
    def _swz(pid, npm, npn, GM: gl.constexpr):
        ng = GM * npn
        gid = pid // ng
        fm = gid * GM
        gsm = min(npm - fm, GM)
        return fm + ((pid % ng) % gsm), (pid % ng) // gsm

    @gluon.jit
    def _conv_prod(a_desc, b_desc, a_bufs, b_bufs, empty, ready, M, Cout, Cin,
                   oD, oH, oW, sT, sH, sW, kH, kW,
                   BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                   nbuf: gl.constexpr, NC: gl.constexpr, npm, npn,
                   GM: gl.constexpr, NKI: gl.constexpr):
        pid_m, pid_n = _swz(gl.program_id(0), npm, npn, GM)
        off_n = pid_n * BN
        HW = oH * oW
        THW = oD * HW
        m0 = pid_m * (BM * NC)
        n0 = m0 // THW
        rem = m0 % THW
        t0 = rem // HW
        rt = rem % HW
        h0 = rt // oW
        w0 = rt % oW
        cd0 = t0 * sT
        ch0 = h0 * sH
        cw0 = w0 * sW
        cch = Cin // BK
        khkw = kH * kW
        idx = 0
        ph = 1
        for ki in range(NKI):
            ks = ki // cch
            cc = (ki - ks * cch) * BK
            kw = ks % kW
            kh = (ks // kW) % kH
            kt = ks // khkw
            _gmbar.wait(empty.index(idx), ph)
            bar = ready.index(idx)
            _gmbar.expect(bar, (BM * NC) * BK * 2 + BK * BN * 2)
            _im2col_load(a_desc, cc, cw0, ch0, cd0, n0, kw, kh, kt, bar,
                         a_bufs.index(idx), True)
            _gtma.async_copy_global_to_shared(b_desc, [ki * BK, off_n], bar,
                                              b_bufs.index(idx))
            idx += 1
            if idx == nbuf:
                idx = 0
                ph ^= 1

    @gluon.jit
    def _conv_cons(c_desc, a_bufs, b_bufs, empty, ready, M, Cout, oD, oH, oW,
                   BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                   nbuf: gl.constexpr, nw: gl.constexpr, NC: gl.constexpr,
                   CID: gl.constexpr, npm, npn, GM: gl.constexpr, NKI: gl.constexpr):
        mma_l: gl.constexpr = _mma_layout(BM, BN, nw)
        pid_m, pid_n = _swz(gl.program_id(0), npm, npn, GM)
        m0 = pid_m * (BM * NC) + CID * BM
        off_n = pid_n * BN
        acc = gl.zeros((BM, BN), dtype=gl.float32, layout=mma_l)
        idx = 0
        ph = 0
        ua = False
        # Deferred-wait WGMMA pipe (tut01 v4): issue mma(ki) but only drain the
        # PREVIOUS group (num_outstanding=1), releasing its buffer one iter late so
        # the current mma overlaps the next iter's barrier-wait + issue.
        prev = 0
        for ki in range(NKI):
            _gmbar.wait(ready.index(idx), ph)
            a_sl = a_bufs.index(idx).slice(CID * BM, BM, dim=0)
            acc = _gwgmma(a_sl, b_bufs.index(idx), acc, is_async=True, use_acc=ua)
            acc = _gwgmma_wait(num_outstanding=1, deps=(acc,))
            if ki > 0:
                _gmbar.arrive(empty.index(prev), count=1)
            ua = True
            prev = idx
            idx += 1
            if idx == nbuf:
                idx = 0
                ph ^= 1
        acc = _gwgmma_wait(num_outstanding=0, deps=(acc,))
        _gmbar.arrive(empty.index(prev), count=1)
        do_store = m0 < M
        c_smem = gl.allocate_shared_memory(gl.bfloat16, c_desc.block_type.shape, c_desc.layout)
        c_smem.store(acc.to(gl.bfloat16))
        _gfence()
        if do_store:
            _gtma.async_copy_shared_to_global(c_desc, [m0, off_n], c_smem)
            _gtma.store_wait(pendings=0)

    @gluon.jit
    def _conv_ws_kernel(a_desc, b_desc, c_desc, M, Cout, Cin, oD, oH, oW,
                        sT, sH, sW, kH, kW,
                        BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr,
                        nbuf: gl.constexpr, nw: gl.constexpr, NC: gl.constexpr,
                        npm, npn, GM: gl.constexpr, NKI: gl.constexpr):
        a_bufs = gl.allocate_shared_memory(gl.bfloat16, [nbuf, BM * NC, BK], a_desc.layout)
        b_bufs = gl.allocate_shared_memory(gl.bfloat16, [nbuf, BK, BN], b_desc.layout)
        empty = gl.allocate_shared_memory(gl.int64, [nbuf, 1], _gmbar.MBarrierLayout())
        ready = gl.allocate_shared_memory(gl.int64, [nbuf, 1], _gmbar.MBarrierLayout())
        for i in gl.static_range(nbuf):
            _gmbar.init(empty.index(i), count=NC)
            _gmbar.init(ready.index(i), count=1)
        gl.warp_specialize(
            [(_conv_prod, (a_desc, b_desc, a_bufs, b_bufs, empty, ready, M, Cout, Cin,
                           oD, oH, oW, sT, sH, sW, kH, kW, BM, BN, BK, nbuf, NC,
                           npm, npn, GM, NKI)),
             (_conv_cons, (c_desc, a_bufs, b_bufs, empty, ready, M, Cout, oD, oH, oW,
                           BM, BN, BK, nbuf, nw, NC, 0, npm, npn, GM, NKI)),
             (_conv_cons, (c_desc, a_bufs, b_bufs, empty, ready, M, Cout, oD, oH, oW,
                           BM, BN, BK, nbuf, nw, NC, 1, npm, npn, GM, NKI))],
            [nw, nw], [232, 232])


def _conv3d_forward_im2col(x, w, stride=(1, 1, 1), BM=64, BK=64, nbuf=None, nw=4, NC=2,
                           GM=8, BN=None):
    """Forward via the warp-specialized HW TMA-im2col kernel. Returns NDHWC output
    (no bias). Requires Gluon + Cin/Cout %64==0.

    BN default is 128, NOT 256: the m64n256 WGMMA accumulator (64*256 fp32 regs per
    consumer) starves occupancy and forces nbuf=3, leaving the medium compute-bound
    256->256 shape at ~245 TF (~0.48x omni). BN=128 doubles the N-blocks (better
    wave balance) and halves accumulator pressure, allowing nbuf=4 -> ~463 TF
    (~0.91x). BN=256 is still offered as an explicit dispatch candidate for large
    Cout where the wider tile amortizes better, but it must keep nbuf=3 (nbuf>=4
    with BN=256 over-subscribes registers and deadlocks)."""
    N, D, H, W, Cin = x.shape
    Cout, kT, kH, kW, _ = w.shape
    sT, sH, sW = stride
    oD = (D - kT) // sT + 1
    oH = (H - kH) // sH + 1
    oW = (W - kW) // sW + 1
    M = N * oD * oH * oW
    K = kT * kH * kW * Cin
    if BN is None:
        BN = 128 if Cout >= 128 else Cout
    if nbuf is None:
        nbuf = 3 if BN >= 256 else 4
    x = x.contiguous()
    w_kn = w.reshape(Cout, K).t().contiguous()       # [K, Cout] (WGMMA B operand)
    out = torch.empty((M, Cout), device=x.device, dtype=torch.bfloat16)
    al = gl.NVMMASharedLayout.get_default_for([BM * NC, BK], gl.bfloat16)
    bl = gl.NVMMASharedLayout.get_default_for([BK, BN], gl.bfloat16)
    cl = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)
    a_desc = _GTD.from_tensor(x.reshape(N * D * H * W, Cin), [BM * NC, BK], al)
    a_desc._im2col_params = (N, D, H, W, Cin, kT, kH, kW, 0, 0, 0, sT, sH, sW, BM * NC, BK)
    b_desc = _GTD.from_tensor(w_kn, [BK, BN], bl)
    c_desc = _GTD.from_tensor(out, [BM, BN], cl)
    npm = _cdiv(M, BM * NC)
    npn = _cdiv(Cout, BN)
    grid = (npm * npn,)
    _conv_ws_kernel[grid](a_desc, b_desc, c_desc, M, Cout, Cin, oD, oH, oW,
                          sT, sH, sW, kH, kW, BM, BN, BK, nbuf, nw, NC, npm, npn, GM,
                          num_warps=nw, NKI=K // BK, emit_cuda=True)
    return out.reshape(N, oD, oH, oW, Cout)


def conv3d_backward(grad_out, x, w, stride=(1, 1, 1), output_mask=(True, True, True),
                    has_bias=True, emit=True):
    N, D, H, W, Cin = x.shape
    Cout, kT, kH, kW, Cin_w = w.shape
    sT, sH, sW = stride
    oD = (D - kT) // sT + 1
    oH = (H - kH) // sH + 1
    oW = (W - kW) // sW + 1
    M = N * oD * oH * oW
    K = kT * kH * kW * Cin
    go = grad_out.contiguous().reshape(M, Cout)
    x_c = x.contiguous()
    w_c = w.contiguous().reshape(Cout, K)
    grad_input = grad_weight = grad_bias = None
    unit_stride = (sT == 1 and sH == 1 and sW == 1)

    if output_mask[0]:
        if unit_stride:
            # dgrad-as-fprop: grad_input[N*DHW,Cin] = im2col(grad_out, pad=k-1) @ Wflip,
            # Wflip[Cin, taps*Cout], K order [kt,kh,kw,co]. One fused GEMM.
            K_dg = kT * kH * kW * Cout
            M_in = N * D * H * W
            wflip = (w.flip([1, 2, 3]).permute(4, 1, 2, 3, 0).contiguous().reshape(Cin, K_dg))
            out_gi = torch.empty((M_in, Cin), device=x.device, dtype=torch.bfloat16)
            dims = (N, oD, oH, oW, Cout, D, H, W, kT, kH, kW, 1, 1, 1, kT - 1, kH - 1, kW - 1)
            _run_implicit_gemm(go, wflip, go, out_gi, M_in, Cin, K_dg, dims, False, emit)
            grad_input = out_gi.reshape(N, D, H, W, Cin)
        else:
            # stride>1: single-launch gather (transpose-conv), no scatter/atomics.
            M_in = N * D * H * W
            out_gi = torch.empty((M_in, Cin), device=x.device, dtype=torch.bfloat16)

            def launch_dg(cfg):
                bm, bn, bk = cfg[0]["BLOCK_M"], cfg[0]["BLOCK_N"], cfg[0]["BLOCK_K"]
                grid = (_cdiv(M_in, bm), _cdiv(Cin, bn))
                _conv3d_dgrad_gather_kernel[grid](
                    go, w_c, out_gi, N, D, H, W, Cin, Cout, oD, oH, oW,
                    kT, kH, kW, sT, sH, sW, M_in, K,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                    num_warps=cfg[1], num_stages=cfg[2], **_emit_kw(emit))

            # dgrad-s>1 reduces over Cout per tap, so BLOCK_K must divide Cout for
            # scalar-tap correctness (same constraint as fwd's CC%bk filter).
            cfgs_dg = _cfgs_for_cc(Cout)
            if emit:
                _tuned_launch(("dgrad", M_in, Cin, Cout), cfgs_dg, launch_dg)
            else:
                launch_dg(cfgs_dg[0])
            grad_input = out_gi.reshape(N, D, H, W, Cin)

    if output_mask[1]:
        # split-K over the M reduction to fill SMs when (Cout x K) tiles are few.
        NUM_SM = 132
        base_tiles = _cdiv(Cout, 64) * _cdiv(K, 64)
        m_tiles = _cdiv(M, 64)
        nsplit = 1
        if base_tiles * 2 <= NUM_SM and base_tiles > 0:
            nsplit = max(1, min(NUM_SM // base_tiles, 8, max(1, m_tiles // 2)))
        if nsplit > 1:
            scratch = torch.empty((nsplit * Cout, K), device=x.device, dtype=torch.bfloat16)
        else:
            scratch = torch.empty((Cout, K), device=x.device, dtype=torch.bfloat16)
        def launch_wg(cfg):
            bco, bk, bm = cfg[0]["BLOCK_CO"], cfg[0]["BLOCK_K"], cfg[0]["BLOCK_M"]
            grid = (_cdiv(Cout, bco), _cdiv(K, bk), nsplit)
            _conv3d_wgrad_kernel[grid](
                go, x_c, scratch, N, D, H, W, Cin, Cout, oD, oH, oW,
                kT, kH, kW, sT, sH, sW, M, K, nsplit,
                BLOCK_CO=bco, BLOCK_K=bk, BLOCK_M=bm,
                num_warps=cfg[1], num_stages=cfg[2], **_emit_kw(emit))

        if emit:
            _tuned_launch(("wgrad", M, K, Cout, nsplit), _WGRAD_CFGS, launch_wg)
        else:
            launch_wg(_WGRAD_CFGS[0])
        if nsplit > 1:
            dw = scratch.reshape(nsplit, Cout, K).float().sum(0).to(torch.bfloat16)
        else:
            dw = scratch
        grad_weight = dw.reshape(Cout, kT, kH, kW, Cin)

    if output_mask[2] and has_bias:
        grad_bias = go.to(torch.float32).sum(0).to(torch.bfloat16)
    return grad_input, grad_weight, grad_bias


# ===========================================================================
# Autograd wrappers (mirror omni ops/conv/conv3d.py).
# ===========================================================================
class _Conv3dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, stride, emit):
        ctx.save_for_backward(x, w)
        ctx.stride = stride
        ctx.has_bias = bias is not None
        ctx.emit = emit
        return conv3d_forward(x, w, bias, stride=stride, emit=emit)

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        mask = (ctx.needs_input_grad[0], ctx.needs_input_grad[1],
                ctx.has_bias and ctx.needs_input_grad[2])
        gi, gw, gb = conv3d_backward(grad_out.contiguous(), x, w, stride=ctx.stride,
                                     output_mask=mask, has_bias=ctx.has_bias, emit=ctx.emit)
        return gi, gw, (gb if ctx.has_bias else None), None, None


def conv3d(x, w, bias=None, stride=1, emit=True):
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    return _Conv3dFn.apply(x, w, bias, tuple(stride), emit)


def conv2d(x, w, bias=None, stride=1, emit=True):
    if isinstance(stride, int):
        sH, sW = stride, stride
    else:
        sH, sW = int(stride[0]), int(stride[1])
    out = conv3d(x.unsqueeze(1), w.unsqueeze(1), bias, stride=(1, sH, sW), emit=emit)
    return out.squeeze(1)


# ===========================================================================
# Correctness harness vs fp32 F.conv3d (mirrors omni test/conv/test_conv3d.py).
# ===========================================================================
def _reference(x_ndhwc, w_kthwc, bias, stride):
    import torch.nn.functional as F
    x = x_ndhwc.float().permute(0, 4, 1, 2, 3).contiguous()
    w = w_kthwc.float().permute(0, 4, 1, 2, 3).contiguous()
    b = bias.float() if bias is not None else None
    y = F.conv3d(x, w, b, stride=stride, padding=0)
    return y.permute(0, 2, 3, 4, 1).contiguous()


def _rel_err(out, ref):
    return (out.float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)


_CASES = [
    (2, 8, 16, 16, 64, 128, 3, 3, 3, 1, 1, 1, True),
    (2, 6, 14, 14, 128, 192, 3, 3, 3, 1, 2, 2, True),
    (1, 4, 8, 8, 64, 128, 1, 3, 3, 1, 1, 1, True),
]
RTOL = 0.02


def _test_forward(emit):
    ok_all = True
    for c in _CASES:
        N, D, H, W, Cin, Cout, kT, kH, kW, sT, sH, sW, hb = c
        x = torch.randn(N, D, H, W, Cin, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(Cout, kT, kH, kW, Cin, device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn(Cout, device="cuda", dtype=torch.bfloat16) if hb else None
        out = conv3d_forward(x, w, b, stride=(sT, sH, sW), emit=emit)
        err = _rel_err(out, _reference(x, w, b, (sT, sH, sW)))
        ok = err < RTOL
        ok_all &= ok
        print(f"  fwd  {Cin}->{Cout} k{kT}{kH}{kW} s{sT}{sH}{sW}: rel_err={err:.3e} {'OK' if ok else 'FAIL'}")
    return ok_all


def _test_backward(emit):
    import torch.nn.functional as F
    ok_all = True
    for c in _CASES:
        N, D, H, W, Cin, Cout, kT, kH, kW, sT, sH, sW, hb = c
        x = torch.randn(N, D, H, W, Cin, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(Cout, kT, kH, kW, Cin, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(True)
        b = torch.randn(Cout, device="cuda", dtype=torch.bfloat16, requires_grad=True) if hb else None
        out = conv3d(x, w, b, stride=(sT, sH, sW), emit=emit)
        go = torch.randn_like(out)
        out.backward(go)
        xf = x.detach().float().permute(0, 4, 1, 2, 3).contiguous().requires_grad_(True)
        wf = w.detach().float().permute(0, 4, 1, 2, 3).contiguous().requires_grad_(True)
        bf = b.detach().float().requires_grad_(True) if hb else None
        yf = F.conv3d(xf, wf, bf, stride=(sT, sH, sW), padding=0)
        yf.backward(go.float().permute(0, 4, 1, 2, 3).contiguous())
        gx_ref = xf.grad.permute(0, 2, 3, 4, 1).contiguous()
        gw_ref = wf.grad.permute(0, 2, 3, 4, 1).contiguous()
        e_gi = _rel_err(x.grad, gx_ref)
        e_gw = _rel_err(w.grad, gw_ref)
        e_gb = _rel_err(b.grad, bf.grad) if hb else 0.0
        ok = e_gi < RTOL and e_gw < RTOL and e_gb < RTOL
        ok_all &= ok
        print(f"  bwd  {Cin}->{Cout} k{kT}{kH}{kW} s{sT}{sH}{sW}: "
              f"gi={e_gi:.3e} gw={e_gw:.3e} gb={e_gb:.3e} {'OK' if ok else 'FAIL'}")
    return ok_all


def _test_conv2d(emit):
    import torch.nn.functional as F
    N, H, W, Cin, Cout = 2, 16, 16, 64, 128
    x = torch.randn(N, H, W, Cin, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(Cout, 3, 3, Cin, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    b = torch.randn(Cout, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = conv2d(x, w, b, stride=1, emit=emit)
    go = torch.randn_like(out)
    out.backward(go)
    xf = x.detach().float().permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    wf = w.detach().float().permute(0, 3, 1, 2).contiguous().requires_grad_(True)
    bf = b.detach().float().requires_grad_(True)
    yf = F.conv2d(xf, wf, bf, stride=1, padding=0)
    yf.backward(go.float().permute(0, 3, 1, 2).contiguous())
    ok = (out.shape == (N, H - 2, W - 2, Cout)
          and _rel_err(out, yf.permute(0, 2, 3, 1)) < RTOL
          and _rel_err(x.grad, xf.grad.permute(0, 2, 3, 1)) < RTOL
          and _rel_err(w.grad, wf.grad.permute(0, 2, 3, 1)) < RTOL
          and _rel_err(b.grad, bf.grad) < RTOL)
    print(f"  conv2d via conv3d: {'OK' if ok else 'FAIL'}")
    return ok


def _test_determinism(emit):
    N, D, H, W, Cin, Cout = 2, 6, 12, 12, 64, 128
    x = torch.randn(N, D, H, W, Cin, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(Cout, 3, 3, 3, Cin, device="cuda", dtype=torch.bfloat16) * 0.1).requires_grad_(True)
    go = torch.randn(N, D - 2, H - 2, W - 2, Cout, device="cuda", dtype=torch.bfloat16)
    grads = []
    for _ in range(3):
        x.grad = w.grad = None
        out = conv3d(x, w, None, stride=1, emit=emit)
        out.backward(go)
        grads.append((x.grad.clone(), w.grad.clone()))
    ok = all(torch.equal(gx, grads[0][0]) and torch.equal(gw, grads[0][1]) for gx, gw in grads[1:])
    print(f"  determinism (bitwise grad_input/grad_weight): {'OK' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# Performance benchmark: times OURS (+ cuDNN, which everyone has via torch) live,
# and compares against omni_kernel forward TFLOP/s that were MEASURED OFFLINE and
# embedded below. External users do NOT have omni_kernel, so we never build/look
# for it here — the omni numbers are a hardcoded reference table.
# ===========================================================================
# omni_kernel PRODUCTION conv (csrc/varlen_conv standard_fused, the real optimized
# kernel with HW TMA-im2col) forward TFLOP/s, measured offline on an H800 (bf16).
# Keyed by (N,D,H,W,Cin,Cout,kT,kH,kW,sT,sH,sW).
_OMNI_PROD_FWD_TFLOPS = {
    (2, 16, 32, 32, 128, 256, 3, 3, 3, 1, 1, 1): 480,
    (1, 16, 64, 64, 64, 128, 3, 3, 3, 1, 1, 1): 281,
    (2, 16, 32, 32, 256, 256, 3, 3, 3, 1, 1, 1): 528,
    (1, 32, 32, 32, 128, 256, 3, 3, 3, 2, 2, 2): 186,
    (1, 8, 32, 32, 320, 640, 1, 3, 3, 1, 1, 1): 311,
    (2, 16, 32, 32, 512, 512, 3, 3, 3, 1, 1, 1): 556,
    (1, 4, 32, 32, 2048, 2048, 3, 3, 3, 1, 1, 1): 510,
    (1, 4, 16, 16, 4096, 4096, 3, 3, 3, 1, 1, 1): 415,
    (2, 8, 128, 128, 128, 256, 3, 3, 3, 1, 1, 1): 524,
    (1, 1, 1024, 1024, 256, 256, 1, 3, 3, 1, 1, 1): 521,   # H/W=1k (2D)
    (1, 1, 2048, 2048, 128, 256, 1, 3, 3, 1, 1, 1): 491,   # H/W=2k (2D)
    (1, 1, 4096, 4096, 64, 128, 1, 3, 3, 1, 1, 1): 298,    # H/W=4k (2D)
}
# (N,D,H,W,Cin,Cout,kT,kH,kW,sT,sH,sW): omni bench + larger (channels->4096) +
# H/W = 1k/2k/4k spatial. All keys appear in _OMNI_PROD_FWD_TFLOPS above.
_PERF_CASES = list(_OMNI_PROD_FWD_TFLOPS.keys())
_PEAK_BF16 = {"H100": 989.0, "H200": 989.0, "H800": 989.0, "H20": 148.0,
              "A100": 312.0, "B200": 2250.0, "GB200": 2250.0}


def _bench(fn, iters=50, warmup=15, rounds=4):
    # Best-of-`rounds` (min) -- a single timing window (esp. for the big-spatial
    # shapes, where 50 iters span ~250ms) almost always catches a contention blip
    # on a shared GPU; taking the min across rounds rejects those so the table
    # reflects achievable (uncontended) throughput, matching how omni was measured.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) / iters)
    return best  # ms


def _measure_one(c, with_cudnn=True):
    """Measure ours (and optionally cuDNN) TFLOP/s for one shape in the CURRENT
    process. Used both in-process and as the body of the per-shape subprocess.

    NO bias: the embedded omni reference numbers were taken without bias, so we
    compare apples-to-apples. (The gather path fuses bias in-kernel for free, but
    the im2col path adds it as a separate elementwise pass -- on the big-spatial
    shapes that ~4GB add alone would knock ~0.69x off an otherwise ~1.2x shape, an
    artifact of an unfused epilogue + a bias/no-bias mismatch, not the conv itself.)"""
    N, D, H, W, Cin, Cout, kT, kH, kW, sT, sH, sW = c
    oD, oH, oW = (D - kT) // sT + 1, (H - kH) // sH + 1, (W - kW) // sW + 1
    fl = 2.0 * N * oD * oH * oW * Cout * (kT * kH * kW * Cin)
    x = torch.randn(N, D, H, W, Cin, device="cuda", dtype=torch.bfloat16) * 0.5
    w = torch.randn(Cout, kT, kH, kW, Cin, device="cuda", dtype=torch.bfloat16) * 0.5
    of = _bench(lambda: conv3d_forward(x, w, None, (sT, sH, sW), emit=True))
    ours_tf = fl / of / 1e9
    cudnn_tf = float("nan")
    if with_cudnn:
        try:
            xc = x.permute(0, 4, 1, 2, 3).contiguous()
            wc = w.permute(0, 4, 1, 2, 3).contiguous()
            cf = _bench(lambda: torch.nn.functional.conv3d(xc, wc, None, stride=(sT, sH, sW)),
                        iters=20, warmup=8, rounds=2)
            cudnn_tf = fl / cf / 1e9
        except Exception:
            pass
    return ours_tf, cudnn_tf


def bench_perf(with_cudnn=True, isolate=True):
    import math
    import os as _os
    import subprocess
    import sys
    dev = torch.cuda.get_device_name(0)
    peak = next((v for k, v in _PEAK_BF16.items() if k in dev), None)
    print(f"\nforward performance vs omni_kernel (production) + cuDNN  (device={dev}, bf16)")
    print(f"TFLOP/s = 2*M*Cout*(kT*kH*kW*Cin)/t; omni = offline-measured reference"
          + (f"; peak bf16 ~{peak:.0f}" if peak else ""))
    h = (f"{'shape (N,DHW,Cin>Cout,k,s)':30} | {'ours':>6} {'omni':>6} {'cudnn':>6} TF | "
         f"{'o/omni':>6}" + ("  MFU" if peak else ""))
    # Measure each shape in a FRESH SUBPROCESS (isolate=True): a short-lived process
    # per shape starts from a clean/cool CUDA context (no allocator fragmentation, no
    # heat-soak) and limits each shape's wall-clock exposure to transient shared-GPU
    # contention -- matching the per-shape conditions under which the embedded omni
    # numbers were taken. Falls back to in-process if the subprocess can't be spawned.
    best_ours, best_cudnn = {}, {}
    for i, c in enumerate(_PERF_CASES):
        got = False
        if isolate:
            try:
                r = subprocess.run([sys.executable, _os.path.abspath(__file__),
                                    "benchone", str(i)],
                                   capture_output=True, text=True, timeout=600,
                                   env=_os.environ)
                for ln in r.stdout.splitlines():
                    if ln.startswith("RESULT "):
                        _, o_s, c_s = ln.split()
                        best_ours[c], best_cudnn[c] = float(o_s), float(c_s)
                        got = True
            except Exception:
                pass
        if not got:  # fallback: measure in-process
            best_ours[c], best_cudnn[c] = _measure_one(c, with_cudnn)
    print(h); print("-" * len(h))
    ratios = []
    for c in _PERF_CASES:
        N, D, H, W, Cin, Cout, kT, kH, kW, sT, sH, sW = c
        ours_tf = best_ours[c]
        cudnn_tf = best_cudnn.get(c, float("nan"))
        omni_tf = _OMNI_PROD_FWD_TFLOPS.get(c)
        r = ours_tf / omni_tf if omni_tf else float("nan")
        if omni_tf:
            ratios.append(r)
        tag = f"({N},{D}x{H}x{W},{Cin}>{Cout},{kT}{kH}{kW},s{sT})"
        cstr = f"{cudnn_tf:6.0f}" if cudnn_tf == cudnn_tf else "   n/a"
        ostr = f"{omni_tf:6.0f}" if omni_tf else "   n/a"
        rstr = f"{r:5.2f}x" if omni_tf else "  n/a"
        mfu = f"  {ours_tf/peak*100:3.0f}%" if peak else ""
        print(f"{tag:30} | {ours_tf:6.0f} {ostr} {cstr} TF | {rstr}{mfu}")
    if ratios:
        gm = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        print("-" * len(h))
        print(f"forward ours/omni(production): min {min(ratios):.2f}x  geomean {gm:.2f}x  "
              f"({sum(v >= 0.95 for v in ratios)}/{len(ratios)} shapes >= 0.95x)")


def main():
    import sys
    # Hidden per-shape worker for the isolated bench (see bench_perf): measures one
    # shape in this fresh process and prints a machine-readable result, no tests.
    if len(sys.argv) >= 3 and sys.argv[1] == "benchone":
        torch.manual_seed(0)
        ours_tf, cudnn_tf = _measure_one(_PERF_CASES[int(sys.argv[2])], with_cudnn=True)
        print(f"RESULT {ours_tf} {cudnn_tf}")
        return
    do_bench = "bench" in sys.argv[1:]
    emit = os.environ.get("EMIT", "1") == "1"
    print(f"backend = {'emit_cuda' if emit else 'stock PTX'}")
    torch.manual_seed(0)
    ok = True
    ok &= _test_forward(emit)
    ok &= _test_backward(emit)
    ok &= _test_conv2d(emit)
    ok &= _test_determinism(emit)
    print("tutorial 04 fwd+bwd:", "PASSED" if ok else "FAILED")
    if do_bench and emit:
        bench_perf()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
