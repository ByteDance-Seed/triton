// P4 WgPingpongPass
// =================
// Realizes level6's inter-warpgroup ping-pong scheduling for the 2-consumer
// FlashAttention mainloop. The CUDA emitter is a pure printer, so the named
// barriers that stagger the two consumer warpgroups must be introduced here as
// IR ops (emitcuda.named_barrier) rather than hand-written in the kernel.
//
// level6 (flash_attn_varlen.cu) staggers two consumer warpgroups (WG1/WG2) so
// one WG's softmax (CUDA cores) overlaps the other's WGMMA (tensor cores):
//
//   // once, before the mainloop:
//   if (wg == WG1) bar.arrive(SCHED_WG1, 256);   // pre-arrive so WG1 starts free
//   ... mainloop iteration ...
//     mbarrier_wait(K);
//     bar.sync(own_sched_bar, 256);              // gate QK on the OTHER WG's PV
//     COMPUTE_QK(); scale_acc; mbarrier_wait(V);
//     COMPUTE_PV();                              // async
//     bar.arrive(other_sched_bar, 256);          // release the other WG
//     wgmma_wait_group<1>(); ... softmax ...
//
//   bar SCHED_WG1 participants = WG1(sync) + WG2(arrive)  -> 256 threads
//   bar SCHED_WG2 participants = WG2(sync) + WG1(arrive)  -> 256 threads
//
// Because WG1 pre-arrives on its own barrier, WG1's first sync clears
// immediately and runs QK[0] while WG2 blocks; once WG1 issues PV[0] it arrives
// on WG2's barrier, releasing WG2 to run QK while WG1 does softmax. From then on
// the two WGs leapfrog. bar.arrive is non-blocking so the lone leftover arrival
// at loop exit hangs nothing.
//
// Transform per consumer partition c in {0,1} (own = SCHED[c], other = SCHED[1-c]):
//   * c==0 only: insert named_barrier{arrive, own, T} immediately before the loop.
//   * insert named_barrier{sync,   own,   T} at the top of the loop body.
//   * insert named_barrier{arrive, other, T} right after the 2nd WGMMA dot.
// where T = total consumer threads (sum of partition warps * 32).
//
// Guarded: only transforms a warp_specialize with exactly 2 partition regions
// whose mainloop (the scf.for whose body holds >=2 ttng.WarpGroupDotOp) is
// found in each. No-op otherwise, so non-FA kernels are untouched.

#include "Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// Private named-barrier IDs for the warp scheduler (match level6's 11/12).
// emit_cuda reserves 0..3 (syncthreads + producer + per-partition syncs), so
// 11/12 are well clear of any clash.
static constexpr int kSchedBar[2] = {11, 12};

// Build an unregistered emitcuda.named_barrier op at the builder's insertion pt.
static void emitNamedBarrier(OpBuilder &b, Location loc, int barrierId,
                             int numThreads, bool isArrive) {
  OperationState state(loc, "emitcuda.named_barrier");
  state.addAttribute("barrier_id", b.getI32IntegerAttr(barrierId));
  state.addAttribute("num_threads", b.getI32IntegerAttr(numThreads));
  state.addAttribute("is_arrive", b.getBoolAttr(isArrive));
  b.create(state);
}

// Find ALL FA mainloops in a consumer partition: every scf.for whose body
// contains >= 2 ttng.WarpGroupDotOp at top level (QK and PV). The causal kernel
// has TWO such loops (mask-free blocks + masked diagonal blocks); level6 runs
// its warp_scheduler_barrier over the entire n_block range, so BOTH must be
// instrumented or the WG0-only K/V wait races on the uninstrumented loop.
static SmallVector<scf::ForOp> findMainloops(Region &region) {
  SmallVector<scf::ForOp> loops;
  if (region.empty())
    return loops;
  region.front().walk([&](scf::ForOp f) {
    int dots = 0;
    for (Operation &op : f.getBody()->without_terminator())
      if (isa<ttng::WarpGroupDotOp>(&op))
        ++dots;
    if (dots >= 2)
      loops.push_back(f);
  });
  return loops;
}

struct WgPingpongPass
    : public PassWrapper<WgPingpongPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(WgPingpongPass)

  StringRef getArgument() const override { return "triton-cuda-wg-pingpong"; }
  StringRef getDescription() const override {
    return "Inter-warpgroup ping-pong scheduling for 2-consumer FlashAttention "
           "(level6 warp_scheduler_barrier overlap)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // NOTE: the emitcuda.named_barrier op is unregistered; the context must
    // already have allowUnregisteredDialects(true) set BEFORE the pass manager
    // runs (it cannot be toggled mid-run — multi-threaded execution context).
    // EmitCudaPlugin::addEmitCudaPass sets it when wg_pingpong is enabled.
    bool dbg = ::getenv("TRITON_WGPP_DEBUG") != nullptr;
    SmallVector<ttg::WarpSpecializeOp> wsOps;
    mod.walk([&](ttg::WarpSpecializeOp ws) { wsOps.push_back(ws); });
    if (dbg)
      llvm::errs() << "[wgpp] num warp_specialize ops = " << wsOps.size() << "\n";

    for (ttg::WarpSpecializeOp ws : wsOps) {
      auto parts = ws.getPartitionRegions();
      if (dbg)
        llvm::errs() << "[wgpp] ws partitions = " << parts.size() << "\n";
      if (parts.size() != 2)
        continue; // only the 2-consumer FA shape

      // Total consumer threads participating in each scheduler barrier.
      ArrayRef<int32_t> partWarps = ws.getPartitionNumWarps();
      int numThreads = 0;
      for (int32_t w : partWarps)
        numThreads += w * 32;

      // Both partitions must expose the FA mainloop(s); otherwise skip the ws.
      // The causal kernel has TWO qualifying loops (mask-free + masked); level6
      // runs warp_scheduler_barrier over the whole n_block range, so we
      // instrument every qualifying loop in each partition.
      SmallVector<scf::ForOp> loopsVec[2] = {findMainloops(*parts[0]),
                                             findMainloops(*parts[1])};
      if (dbg)
        llvm::errs() << "[wgpp] mainloops found: " << loopsVec[0].size() << " "
                     << loopsVec[1].size() << "  numThreads=" << numThreads << "\n";
      if (loopsVec[0].empty() || loopsVec[1].empty() ||
          loopsVec[0].size() != loopsVec[1].size())
        continue;

      for (int c = 0; c < 2; ++c) {
        int ownBar = kSchedBar[c];
        int otherBar = kSchedBar[1 - c];
        Location loc = loopsVec[c].front().getLoc();

        // (1) pre-arrive on own barrier — consumer 0 only — ONCE at the consumer
        // region entry (before ANY enclosing loop). In the persistent /
        // work-stealing kernel the mainloops are nested inside an outer per-tile
        // loop; a per-tile pre-arrive leaves an unmatched arrival that flips the
        // named-barrier phase early on the 2nd tile -> deadlock. A single
        // region-entry pre-arrive keeps the sync:arrive 1:1 balance across all
        // tiles and both inner loops (matches level6's once-before-persistent
        // arrive of WG1's scheduler barrier).
        if (c == 0) {
          Block &entry = parts[c]->front();
          OpBuilder pb(&entry, entry.begin());
          emitNamedBarrier(pb, loc, ownBar, numThreads, /*isArrive=*/true);
        }

        for (scf::ForOp loop : loopsVec[c]) {
          Block *body = loop.getBody();

          // Locate the first WGMMA dot (QK) up front; both the sync and arrive
          // placements key off it.
          ttng::WarpGroupDotOp firstDot = nullptr;
          for (Operation &op : body->without_terminator()) {
            if (auto d = dyn_cast<ttng::WarpGroupDotOp>(&op)) {
              firstDot = d;
              break;
            }
          }

          // (2) sync on own barrier. Default = very top of the loop body.
          // level6 (char-by-char, flash_attn_dense_full.cu fwd_step ~L1929)
          // puts the scheduler-barrier sync AFTER the K consumer_wait and right
          // BEFORE the QK gemm: `if(WG0) consumer_wait(K); sched_sync(); QK;`.
          // Our default sync-at-top has WG0 grab the scheduler token THEN stall
          // on the producer's K mbarrier, serializing the other WG's compute
          // turn behind a data dependency. TRITON_WGPP_SYNC_AT_QK=1 moves the
          // sync to immediately before the first WGMMA dot (QK) so a producer
          // stall no longer holds the scheduler token (faithful level6 order:
          // data-wait first, scheduler-sync second).
          bool syncAtQk = ::getenv("TRITON_WGPP_SYNC_AT_QK") != nullptr;
          if (syncAtQk && firstDot) {
            OpBuilder tb(firstDot->getContext());
            tb.setInsertionPoint(firstDot);
            emitNamedBarrier(tb, loop.getLoc(), ownBar, numThreads,
                             /*isArrive=*/false);
          } else {
            OpBuilder tb(body, body->begin());
            emitNamedBarrier(tb, loop.getLoc(), ownBar, numThreads,
                             /*isArrive=*/false);
          }

          // (3) arrive on the other barrier right after a WGMMA dot. Default =
          // after the 2nd dot (PV), matching level6's source order (COMPUTE_PV;
          // bar.arrive). But level6's SASS schedule effectively releases the
          // other WG right after the QK issue (BAR.ARV lands before the PV
          // HGMMA + softmax). TRITON_WGPP_ARRIVE_AFTER_QK=1 forces the arrive
          // after the 1st dot (QK) to replicate that earlier inter-WG release.
          int arriveAfterDot =
              (::getenv("TRITON_WGPP_ARRIVE_AFTER_QK") != nullptr) ? 1 : 2;
          ttng::WarpGroupDotOp secondDot = nullptr;
          int seen = 0;
          for (Operation &op : body->without_terminator()) {
            if (auto d = dyn_cast<ttng::WarpGroupDotOp>(&op)) {
              if (++seen == arriveAfterDot) {
                secondDot = d;
                break;
              }
            }
          }
          if (secondDot) {
            OpBuilder ab(secondDot->getContext());
            ab.setInsertionPointAfter(secondDot);
            emitNamedBarrier(ab, loop.getLoc(), otherBar, numThreads,
                             /*isArrive=*/true);
          }
          if (dbg)
            llvm::errs() << "[wgpp] consumer " << c << " loop instrumented (own="
                         << ownBar << " other=" << otherBar
                         << " secondDot=" << (secondDot != nullptr) << ")\n";
        }
      }
    }
  }
};
} // namespace

namespace triton_cuda {
std::unique_ptr<mlir::Pass> createWgPingpongPass() {
  return std::make_unique<WgPingpongPass>();
}
} // namespace triton_cuda
