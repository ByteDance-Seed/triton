// P2 EpilogueOverlapPass
// ======================
// Runs AFTER PersistentTilePass. In each consumer partition the epilogue now
// sits inside the persistent tile loop and ends with a blocking
//     ttng.async_tma_store_wait {pendings = 0}
// so each tile drains its own C store before the next tile's mainloop can begin
// — store(t) and mainloop(t+1) are fully serialized.
//
// gemm_06_autotuned.cu instead defers that wait to the TOP of the next tile
// iteration (lines 788-792: `if(!first_tile){ tma_store_wait(); __syncwarp(); }`)
// and adds a final drain after the loop (line 831). That overlaps store(t) with
// mainloop(t+1).
//
// Transform per consumer partition (post-P1):
//   for tile {                          for tile {
//     <swizzle>                           ttng.async_tma_store_wait 0   // NEW: drains store(t-1)
//     <K-loop>                            <swizzle>
//     <C store>               =====>      <K-loop>
//     ttng.async_tma_store_wait 0         <C store>                     // wait removed
//     scf.yield <ring>                    scf.yield <ring>
//   }                                   }
//                                       ttng.async_tma_store_wait 0     // NEW: final drain
//
// On tile 0 the hoisted wait has no outstanding store, so it returns immediately
// — equivalent to gemm_06's `!first_tile` guard without materializing the branch.
// The single C smem buffer (one local_alloc reused each tile) stays correct
// because the wait-at-top guarantees store(t-1) finished before this tile writes
// the buffer in its epilogue.

#include "Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// Defer the epilogue store-wait across persistent tiles in one region. Returns
// true if the region had an in-tile-loop store-wait that was deferred.
static bool overlapRegion(Region &region) {
  if (region.empty())
    return false;
  Block &body = region.front();

  // The outer tile loop is the first scf.for at region-body level (P1 wrapped
  // {swizzle + K-loop + epilogue} in it). Producer partitions also have one.
  scf::ForOp tileLoop = nullptr;
  for (Operation &op : body)
    if (auto f = dyn_cast<scf::ForOp>(&op)) {
      tileLoop = f;
      break;
    }
  if (!tileLoop)
    return false;

  Block *tbody = tileLoop.getBody();

  // Find the epilogue store-wait at the top level of the tile-loop body. Only
  // consumer partitions have one; producer partitions have none -> skip.
  ttng::TMAStoreWaitOp wait = nullptr;
  for (Operation &op : *tbody)
    if (auto w = dyn_cast<ttng::TMAStoreWaitOp>(&op)) {
      wait = w;
      break;
    }
  if (!wait)
    return false;

  int32_t pendings = wait.getPendings();
  Location loc = wait.getLoc();

  // 1) Drop the per-tile blocking wait from the end of the epilogue.
  wait.erase();

  // 2) Re-insert it at the TOP of the tile-loop body. Tile 0: no outstanding
  //    store -> immediate return. Tile t>0: waits for store(t-1), so the prior
  //    tile's C store overlaps this tile's mainloop.
  OpBuilder b(tbody, tbody->begin());
  ttng::TMAStoreWaitOp::create(b, loc, pendings);

  // 3) Final drain after the tile loop (before the region terminator) so the
  //    last tile's store is guaranteed complete on exit.
  OpBuilder ab(region.getContext());
  ab.setInsertionPointAfter(tileLoop);
  ttng::TMAStoreWaitOp::create(ab, loc, pendings);

  return true;
}

struct EpilogueOverlapPass
    : public PassWrapper<EpilogueOverlapPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EpilogueOverlapPass)

  StringRef getArgument() const override {
    return "triton-cuda-epilogue-overlap";
  }
  StringRef getDescription() const override {
    return "Defer consumer TMA-store wait across persistent tiles "
           "(gemm_06-class epilogue/mainloop overlap)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<ttg::WarpSpecializeOp> wsOps;
    mod.walk([&](ttg::WarpSpecializeOp ws) { wsOps.push_back(ws); });

    for (ttg::WarpSpecializeOp ws : wsOps) {
      overlapRegion(ws.getDefaultRegion());
      for (Region *part : ws.getPartitionRegions())
        overlapRegion(*part);
    }
  }
};
} // namespace

namespace triton_cuda {
std::unique_ptr<mlir::Pass> createEpilogueOverlapPass() {
  return std::make_unique<EpilogueOverlapPass>();
}
} // namespace triton_cuda
