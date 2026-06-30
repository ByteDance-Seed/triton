# conv-im2col-perf — dense Conv3d/2d on the Triton CUDA backend

How tutorial 04 (`tutorials/04-conv.py`) ports omni_kernel's dense Conv3d (NDHWC
implicit-GEMM) to the out-of-tree `emit_cuda` backend, with **zero Triton-source
edits**. TWO forward backends, dispatched per-shape: a manual-gather `tl.dot` GEMM
and a **HW TMA-im2col** warp-specialized kernel (added entirely in the backend —
see the last section). Forward is **geomean ~1.08× of omni PRODUCTION, 12/12 shapes
≥0.95× (min 0.95)** (clean per-shape, H800; TMA-im2col wins large-spatial incl.
H/W=1k/2k/4k *and* the medium 256→256, gather wins huge-channel/strided/moderate-K);
backward (gather, fused dgrad/wgrad + split-K) ~1.0–1.4×; all 10–40× over cuDNN.

### Things that moved the needle (2026-06, getting to 11/12)
- **im2col BN=128, NOT 256.** The default BN=256 gave the m64n256 WGMMA a 64×256
  fp32 accumulator per consumer → starved occupancy, forced nbuf=3, and left the
  medium 256→256 shape at ~245 TF (**0.48×**). BN=128 (2 N-blocks → better wave
  balance, half the accumulator) → ~520 TF (**0.98×**). nbuf must stay 3 for BN=256
  (nbuf≥4 there deadlocks); BN=128 wants nbuf=4. The dispatch tries both BN and the
  timer picks. This was the single biggest fix.
- **gather config `(BLOCK_M=64,BLOCK_N=256,BLOCK_K=64, num_warps=4, num_stages=4)`**
  for moderate-K medium shapes (128→256): nw=4+4-stage beats the nw=8 variants
  (0.80→0.96×). Recurring lesson: small/medium tiles want **fewer warps + deeper
  pipeline**, not more warps.
- **split-K candidate gating by int32 overflow, NOT occupancy.** The dispatch
  offers gather split-K {2,4} unless `M*Cout*4 ≥ 2³¹` (the split-K scratch offset
  `idx*M*Cout` overflows int32 → illegal memory access on the 2048² shape, M~4.2M).
  Do NOT gate on `_pick_ksplit`/tile-count — the tiny-M huge-channel shapes
  (4096→4096 M=392, 2048→2048 M=1800) have many N-tiles yet still NEED split-K for
  latency hiding; gating them out regressed 0.99→0.53×.
- **Per-call Python overhead is the whole ballgame on tiny shapes.** The stride-2
  conv (M~3.4k) sat at ~0.84×. Breakdown vs omni (a single C++ op): the **kernel
  alone is ~1.3× of omni** (246 vs 186 TF), but the `conv3d_forward → gather →
  run_implicit_gemm → tuned_launch` call chain adds ~6µs of Python to a ~12µs
  kernel → 0.85×. Fix: once the dispatch + config are chosen, cache a **lean launch
  runner** (`_make_gather_runner`) that does ONE alloc + ONE kernel launch and skips
  all wrapper layers → restored to ~0.98–1.0×. Lesson: for sub-20µs kernels, profile
  the *wrapper*, not just the kernel; match omni's near-zero host overhead.
- **BLOCK_K=128** (only valid when Cin%128==0 — see the `CC%bk` filter): halves the
  K-loop and lifted stride-2's kernel meaningfully. Generally try BK=128 whenever
  the channel count is a 128-multiple.
- **Bias must match the reference (no-bias here).** omni's embedded numbers were
  captured WITHOUT bias. The gather path fuses bias in-kernel (free), but the im2col
  path adds it as a separate elementwise pass — on the big-spatial shapes that ~4GB
  add knocked an otherwise ~1.2× shape down to ~0.69×, a pure bias/no-bias-mismatch
  + unfused-epilogue artifact. `bench_perf`/`_measure_one` time WITHOUT bias so it's
  apples-to-apples; bias correctness is covered by the correctness tests. (A future
  win: fuse bias into the im2col epilogue so with-bias is also free.)
- **Measurement robustness is part of the result.** On a shared GPU: (a) `_bench` is
  best-of-4 (min) — a single window for the big H/W=2k/4k shapes spans ~250ms and
  almost always catches a contention blip; (b) `_tuned_launch` does per-config
  warmup + best-of-3×40; (c) a long *sequential* 12-shape bench contaminates the
  late (largest) shapes — verify suspect shapes in a FRESH process on an idle GPU;
  (d) omni is load-STABLE (HW TMA), so compare ours-vs-omni *co-measured* in the
  same process — our live number dropping while the fixed embedded omni doesn't is
  a contention artifact, not a real regression.

NOTE: lever 1 and the "not needed here" section below are HISTORICAL (the first
pass beat the reference conv without TMA-im2col); the production conv needed it, so
it is now implemented — see "HW TMA-im2col (NEW)".

## Problem shape

Conv = implicit GEMM over an affine im2col index map:
`M = N·oD·oH·oW`, `K = kT·kH·kW·Cin`, `out[M,Cout] = A_im2col[M,K] @ W[Cout,K]^T`.
NDHWC bf16, Cin/Cout %64, groups=1, dilation=1, no pad, stride≥1. K column order
`[kt,kh,kw,cin]` (cin innermost), `tap = (kt·kH+kh)·kW+kw`.

## Key decisions (what actually mattered)

1. **Manual masked-load im2col gather + `tl.dot`, not HW TMA-im2col.** The gather
   address per (m,k) is pure affine arithmetic (decode m→n,od,oh,ow; k→kt,kh,kw,cin;
   form the flat NDHWC offset). `tl.load(..., mask=in_bounds, other=0)` gives free
   zero-padding for OOB taps. The Triton compiler hides this gather behind the
   WGMMA pipeline well enough to match omni's hardware TMA-im2col at these sizes.
   → A whole class of emitter work (TMA `.im2col` PTX, im2col CUtensorMap, a
   backend MLIR pass to synthesize the op) turned out **not to be necessary**.

2. **One general padded implicit-GEMM kernel serves fwd AND dgrad(stride==1).**
   dgrad-as-fprop: `grad_input = im2col(grad_out, pad=k-1, stride=1) @ Wflip`,
   `Wflip = weight.flip({1,2,3}).permute(4,1,2,3,0).reshape(Cin, taps·Cout)`.
   Same kernel, different host descriptors. No scatter / atomics / fp32 buffer.

3. **wgrad = one fused GEMM with split-K.** `dW[Cout,K] = grad_out^T @ A_im2col(input)`,
   reduce over M. Single-launch (vs 27 per-tap launches) was the big bwd win
   (~10×). Then **split-K over the M reduction** (partials → fp32 scratch → fixed-
   order sum) fills SMs when `(Cout/64)·(K/64)` tiles < #SM — lifted the
   small-channel shapes from 0.82× to 1.20× of omni. Deterministic.

4. **dgrad(stride>1) stays per-tap injective scatter** (omni does too: im2col would
   do ~stride³ redundant FLOP). Fixed tap → output→input is injective, so plain
   `+=` (no atomics); sequential per-tap launches give a deterministic fp32 accum.

5. **scalar-tap gather** (the big general win): Cin/Cout are 64-multiples and we
   keep `BLOCK_K ≤ CC`, so each 64-aligned K-tile lies within ONE tap ⇒ `(kt,kh,kw)`
   and the tap's channel-start are *scalars* per k-iter, not `[BLOCK_M,BLOCK_K]`
   vectors. Collapses the dominant integer-gather math. (+ hoist the voxel base out
   of the K-loop; skip bounds checks when pad=0; load B coalesced then `tl.trans`.)

6. **DO NOT use `triton.autotune`** — its per-call Python dispatch (~25µs) silently
   dominates a tiny (40µs) kernel: the stride-2 forward measured **58µs autotuned
   vs a true 30µs** with a direct launch. Use a tiny config cache (`_tuned_launch`)
   that times the candidate configs once per shape-key, caches the winner, then
   launches the **raw JIT fn directly**. This one change took the stride-2 forward
   from 0.75× → 1.05× of omni — the difference between missing and meeting the goal.
   (Configs: ~6 GEMM / 6 wgrad tiles; all three kernels are overwrite-store so
   re-running configs during tuning is safe.)

## Gotchas

- The biggest perf trap was **autotune dispatch overhead on small kernels** — see
  lever 6. Profile the *direct* launch before blaming the kernel.
- The scalar-tap trick REQUIRES `BLOCK_K ≤ CC` (CC=Cin for fwd, Cout for dgrad-s1,
  Cin for wgrad). A wider K-tile straddles two taps → wrong results. wgrad pins
  `BLOCK_K=64` for this reason and widens via `BLOCK_CO`.
- Each candidate config is a separate nvcc compile under emit_cuda → keep config
  lists tiny; `rm -rf ~/.triton/cache` after any emitter rebuild.
- The `_tuned_launch` cache re-runs configs during tuning, so every tuned kernel
  must be **overwrite-store** (no in-place RMW accumulate). All three conv kernels
  are; the old per-tap RMW dgrad was replaced by the overwrite gather kernel.
- Determinism (omni's `test_bitwise`): split-K sums partials in a FIXED order (fp32
  scratch `.sum(0)`); every kernel is itself deterministic.
- Plain `@triton.jit` kernels run through emit_cuda fine (no Gluon required) —
  activate with `emit_cuda=True` kwarg or `TRITON_EMIT_CUDA=1`.

## On HW TMA-im2col (not needed here)

omni's perf lever is hardware TMA-im2col. It turned out NOT to be required to beat
omni on these shapes. For reference, it is also hard to add under the strict
non-invasive constraint: libtriton HAS the `AsyncTMACopyGlobalToLocalOp` im2col
mode (+`offsets`, `TensorDescIm2ColType`, host `fill_tma_descriptor_im2col`), but
no pristine Python binding creates them, the plugin ABI cannot add pybind fns, and
the trace-time verifier ties im2col mode to the descriptor type (so a backend pass
has no valid tiled placeholder to rewrite). It would need a backend MLIR synthesis
pass + emitter `.im2col` emission + a launcher hook — large and unnecessary.

## Run

```
cd $TRITON_ROOT/triton_cuda_backend
python3 tutorials/04-conv.py          # correctness (fwd/dgrad/wgrad/conv2d/determinism) vs fp32 F.conv3d
```

## HW TMA-im2col (NEW) — implemented entirely in the backend

omni's production conv (`csrc/varlen_conv` standard_fused) uses the Hopper HW
TMA-im2col gather (`cp.async.bulk.tensor...im2col`). We added it to the backend
with **zero Triton-source edits** and use it via a per-shape dispatch in
`conv3d_forward` (timed once, cached: best of gather vs im2col). It wins the
large-spatial / compute-bound shapes (512->512 1.23x, H/W=1k/2k/4k 1.0-1.27x of
omni production); gather still wins huge-channel / strided / tiny-M.

The chicken-and-egg: the frontend cannot create the im2col op (no pristine binding
passes `offsets` or builds `tensordesc_im2col`; the trace-time verifier ties
im2col mode to the descriptor type). Solution, four backend-owned pieces:

1. **Frontend marker** (`_im2col_load` in tutorials/04-conv.py): emits
   `gl.device_print("__IM2COL__", c,w,h,d,n, kw,kh,kt)` (carries the 8 i32 coord/
   tap values — `tt.print` takes scalar varargs) immediately followed by a VALID
   placeholder `tma.async_copy_global_to_shared(a_desc, [0,0], bar, smem)`.
2. **Backend MLIR pass** (`csrc/Im2colRewritePass.cpp`, runs first in the emit_cuda
   pipeline): matches the "__IM2COL__" print (device_print mangles the prefix to
   " __IM2COL__: " -> match by `.contains`), retypes the descriptor block-arg to
   `ttng::TensorDescIm2ColType`, and replaces the placeholder with a real im2col
   `AsyncTMACopyGlobalToLocalOp` (coords [n,d,h,w,c]; offsets i16 [kw,kh,kt]),
   then erases the print + placeholder.
3. **Emitter** (`CUDACodeGen.cpp`): treats `TensorDescIm2ColType` as a
   `__grid_constant__ CUtensorMap` param, and for a copy with offsets emits
   omni's exact `cp.async.bulk.tensor.<rank>d.shared::cluster.global.im2col...
   [smem],[desc,{c,w,h,d,n}],[mbar],{kw,kh,kt}` (coords innermost-first).
4. **Launcher** (`inject_im2col_launcher` in _gluon_ext.py, installed by register):
   for a TensorDescriptor tagged `_im2col_params`, builds the im2col CUtensorMap
   via Triton's own `fill_tma_descriptor_im2col` (driver reverses inputs, so pass
   NDHWC-order shape [N,D,H,W,C], corners [-pT,-pH,-pW]/[pT-(kT-1),...],
   elemStrides [1,sT,sH,sW,1], channelsPerPixel=BK, pixelsPerColumn=BM).

Gotcha that cost time: TWO emit_cuda.so on TRITON_PLUGIN_PATHS (the build copy +
a stale dist-packages copy) — the stale one shadowed the rebuild. After any plugin
rebuild, also copy it over the dist-packages copy.
