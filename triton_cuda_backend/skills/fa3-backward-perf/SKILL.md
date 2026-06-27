---
name: fa3-backward-perf
description: How to push a Gluon WS TMA+WGMMA Flash-Attention-3 BACKWARD kernel to ≥95% of a hand-written CUTLASS reference (omni_kernel) on Hopper via the emit_cuda backend. The ncu-driven optimization method + the specific levers that worked, and the dead ends that did not.
---

# FA3 backward → 95% of omni_kernel (emit_cuda, Hopper)

This is the distilled playbook from taking tutorial-03 backward from **74% → 95%** of
omni_kernel (a hand-written CUTLASS FA3). It is equally a method (how to find the next
lever) and a catalog (which levers actually moved the number, and which were noise).

## The method that works: ncu-driven find-stall → fix → measure

Do **not** guess. Every real gain came from profiling the dominant warp-issue stall,
fixing exactly that, and re-measuring. Every guess (register pressure, barrier count,
in-place packing reasoning) was wrong or noise until ncu pointed at the truth.

```
NCU=/root/share/cuda-12.2/nsight-compute-2023.2.0/ncu   # see "ncu gotcha" below
CUDA_VISIBLE_DEVICES=<free> TRITON_EMIT_CUDA=1 TRITON_ALLOW_NON_POW2=1 \
  <opt env flags> $NCU --target-processes all --launch-count 1 \
  --kernel-name regex:"_bwd_v5" \
  --section SpeedOfLight --section WarpStateStats <kernel-harness>.py
```
Key metrics to read each iteration:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — compute SOL. <60% ⇒ latency-bound.
- `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` for
  `long_scoreboard` (global/L1TEX loads), `barrier` (mbarrier/named-bar waits),
  `wait` (wgmma waits), `short_scoreboard`, `mio_throttle`, `drain`.
- Source metrics to localize: `smsp__inst_executed_op_shared_ld.sum` (0 ⇒ wgmma reads
  smem via descriptor, NOT LDS, so a "shared scoreboard" stall is really **global**),
  `smsp__pcsamp_warps_issue_stalled_long_scoreboard`.

Then fix the #1 stall, re-profile, repeat. Each lever below was one turn of this loop.

## The levers that took bwd 74 → 95% (in order, each measured)

1. **Emitter in-place opts (74→78%).** `TRITON_BWD_INPLACE_RS_PACK=1` +
   `TRITON_EMIT_INPLACE_ELTWISE=1`. These pack the bf16 RS-operand convert directly
   and reuse same-type dead register arrays in place. On their own they look
   register-neutral, but they **free enough registers that the overlapped schedule
   below becomes viable** — they are a prerequisite, not a standalone win.

2. **Hoist per-iteration global loads before the barrier wait (78→82%).** The consumer
   loads LSE (`m`) and Delta (`Di`) from **global memory every q-iteration**. ncu
   showed this was the #1 stall (`long_scoreboard`, ~36% of the issue gap) — NOT smem,
   NOT wgmma. Moving `m = gl.load(M+...)`, `Di = gl.load(Delta+...)` to *before*
   `mbarrier.wait(q_ready...)` overlaps the LDG latency with the barrier wait.

3. **Switch to the overlapped schedule once registers are free (82→87%).** The
   `OVERLAP==5` consumer (S/P/dP/dS kept in [M1,BN] M-major so dQ=dS·K uses dS directly
   as a register A-operand, no dQ smem round-trip) **lost** before the opts existed
   (it spilled). With the opts freeing registers it becomes the winner. Lesson: a
   schedule that regresses under register pressure can win after you cut the pressure —
   re-test "negative" variants after every register win.

4. **Hoist that schedule's own loads too (87→89%).** OVERLAP==5 reloads its own
   row-major `m5/Di5`; hoist them before the wait as well.

5. **Guard dead loads — emit_cuda does NOT DCE unused `gl.load` (89→90%).** For the
   default schedule the other schedules' `m/Di` loads are dead but still issued every
   iteration. Wrap them in a `constexpr` guard (`if OVERLAP != 5:`) so they compile
   away. **General gotcha: the emitter does not dead-code-eliminate global loads —
   guard any load that a constexpr branch makes dead.**

6. **One-iteration-ahead prefetch — the finisher (90→95%).** The barrier wait is short,
   so hoisting only partly hides the LDG latency; `long_scoreboard` stayed #1. Prefetch
   the *next* iteration's `m5/Di5` during *this* iteration's long compute:
   - prologue before the loop loads `m5/Di5` for qt=0;
   - inside the loop, right after `q_ready`, compute the qt+1 index and issue
     `m5_next/Di5_next = gl.load(...)` (clamp the index on the last iter);
   - carry `m5 = m5_next; Di5 = Di5_next` at the bottom before `qt += 1`.
   This overlaps the global-load latency with the full per-iteration WGMMA compute,
   fully hiding it. Guard the whole thing with the schedule constexpr so other
   variants are unaffected.

Final fast config: `version=5`, `OVERLAP=5` (default), `TRITON_BWD_INPLACE_RS_PACK=1`,
`TRITON_EMIT_INPLACE_ELTWISE=1`, + load hoists + dead-load guard + one-iter-ahead
prefetch. Result: 588–601 TF/s = 95–97% of omni (cool/isolated GPU).

## Dead ends — do NOT re-chase (all measured)

- **Register-pressure reduction on the 80-row (M1=80) tile.** v13 80-row hits ptxas
  C7512 (wgmma serialized for insufficient registers). Three levers — RS-pack alone,
  in-place-eltwise alone, scf.while loop-carry aliasing — were flat or **regressed**.
  Carry-aliasing even removed the C7512 warning yet made it *slower*: the `witer/wcond`
  accumulator "doubling" is **functional double-buffering** that hides cross-iteration
  latency; killing it tightens the dependency chain. Register pressure was never the
  wall-clock bottleneck there.
- **Bigger tiles for amortization.** 64-row is the sweet spot. 80-row's better
  amortization is eaten by the overlap it must give up to fit registers + non-pow2
  codegen overhead (62%); 128-row overflows smem + spills 1152 B (fails).
- **Named barriers for register lifetime.** omni uses 16 named barriers vs our 4, but
  `bar.sync` synchronizes threads — it does **not** free a thread's own registers.
  Barrier count correlates with omni's structure but is not the register lever (red
  herring). Per-thread spill is per-thread register pressure only.
- **In-place elementwise / RS-pack as a *spill* fix.** ptxas already coalesces short
  same-size transients across the convert; eliminating the C-source "doubling" changes
  nothing for spill. (They still matter as the register-freeing prerequisite in lever 1.)
- **Deeper q-pipeline (QSTAGES=3).** Smem-blocked (~14 KB over the 232 KB cap at 64-row).

## Environment gotchas

- **ncu version.** The system `/usr/bin/ncu` (2022.3) is too old for this CUDA-12
  PyTorch — it breaks the driver-symbol lookup (`Can't find cuDeviceGetAttribute`).
  Use `/root/share/cuda-12.2/nsight-compute-2023.2.0/ncu` (2023.2). It profiles
  emit_cuda kernels fine (they are triton-cached, no rebuild under ncu). A torch JIT
  C++ extension (omni) fails to *build* under ncu (env sanitization drops ninja) —
  pre-build it outside ncu, or only profile the triton kernels.
- **Thermal-throttle measurement artifact.** Measuring a kernel *after* a long
  sustained run of other kernels (e.g. v5 timed after omni+v13 in the same process)
  reports ~20-25% low from clock throttling. Always measure on a cool/idle GPU
  (representative of standalone use) or warm uniformly; bench only on a verified-free
  GPU. The isolated number is the fair one.
- **emit_cuda does not DCE dead `gl.load`** (see lever 5).

## How to verify

- Correctness: rel-L2 dq≈1e-2, dk/dv≈3e-3 vs PyTorch fp32 ref (`_ref_bwd`).
- Perf: isolated harness (warm then time on a free GPU), compare TF/s to omni
  measured under the *same* conditions. omni API: `fa3_forward_lse(q,k,v,cu,N,SM)` /
  `fa3_backward(do,q,k,v,o,lse,cu,N,causal,bias,SM)` with q in [Z,N,H,D] layout,
  `cu` int64, `SM`=multiprocessor count; build sources include `binding.cpp`.
