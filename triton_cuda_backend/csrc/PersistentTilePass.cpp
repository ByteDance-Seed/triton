// P1 PersistentTilePass
// =====================
// Transforms a *non-persistent* warp-specialized GEMM (one output tile per CTA,
// grid = num_tiles) into a *persistent* grid-stride form (each CTA walks many
// tiles, grid = ~num_SMs), reverse-engineered from gemm_06_autotuned.cu's
// persistent tile loop (`for(tile=cluster_id; tile<num_tiles; tile+=num_clusters)`).
//
// Per warp_specialize region (producer default + consumer partitions) the input
// shape is:
//     %pid = tt.get_program_id x
//     <swizzle arith using %pid>            -> off_m, off_n
//     %r = scf.for %k ... iter_args(<ring index,phase>, [acc, use_acc]) { ... }
//     <epilogue (consumers only)>
//     ttg.warp_yield / ttg.warp_return
//
// We wrap {swizzle + K-loop + epilogue} in an outer tile loop:
//     %pid = tt.get_program_id x
//     %nprog = tt.get_num_programs x
//     %num_tiles = num_pid_m * num_pid_n
//     scf.for %tile = %pid to %num_tiles step %nprog
//             iter_args(<ring index,phase>) {
//        <swizzle arith using %tile>
//        %r = scf.for %k ... iter_args(<ring from tile args>, [acc, use_acc])
//        <epilogue>
//        scf.yield <K-loop ring results>     // ring continuous across tiles
//     }
//
// The pipeline ring (index, phase) is threaded as tile-loop iter_args so the
// mbarrier phase state stays continuous across tiles (resetting it would
// desync producer/consumer). acc / use_acc stay K-loop-local (reset per tile).
//
// Correctness-preserving for ANY launch grid: with the original grid=num_tiles
// each CTA still does exactly one tile (loop runs once). Shrinking the grid to
// num_SMs (Python launch side) turns on actual persistence + reuse.

#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

// Find the first top-level op of type T in a block.
template <typename T> static T findFirst(Block &blk) {
  for (Operation &op : blk)
    if (auto t = dyn_cast<T>(&op))
      return t;
  return nullptr;
}

// Persistent-ize a single warp_specialize region. `numPidM`/`numPidN` are the
// per-grid tile counts, expressed as SSA values valid inside `region`.
// Returns true if the region was transformed.
static bool persistentizeRegion(Region &region, Value numPidM, Value numPidN) {
  if (region.empty())
    return false;
  Block &body = region.front();

  auto pidOp = findFirst<tt::GetProgramIdOp>(body);
  auto kloop = findFirst<scf::ForOp>(body);
  if (!pidOp || !kloop)
    return false;

  // Ring iter-args of the K-loop = the i32-typed ones (index, phase). acc is a
  // tensor and use_acc is i1, so they are excluded and stay loop-local.
  SmallVector<unsigned> ringPos;
  for (auto en : llvm::enumerate(kloop.getInitArgs()))
    if (en.value().getType().isInteger(32))
      ringPos.push_back(en.index());
  if (ringPos.empty())
    return false;

  Operation *term = body.getTerminator();
  Location loc = kloop.getLoc();

  // Tile-loop init args = the K-loop's current ring init values (constants).
  SmallVector<Value> tileInits;
  for (unsigned p : ringPos)
    tileInits.push_back(kloop.getInitArgs()[p]);

  OpBuilder b(pidOp);
  b.setInsertionPointAfter(pidOp);
  Value pid = pidOp.getResult();
  Value nprog = tt::GetNumProgramsOp::create(b, loc, 0);
  Value numTiles = arith::MulIOp::create(b, loc, numPidM, numPidN);

  auto tileLoop = scf::ForOp::create(
      b, loc, pid, numTiles, nprog, tileInits,
      [&](OpBuilder &nb, Location nloc, Value iv, ValueRange args) {
        scf::YieldOp::create(nb, nloc, args);
      });

  // Move {swizzle + K-loop + epilogue} (everything from just after tileLoop up
  // to the region terminator) into the tile-loop body, before its yield.
  Operation *moveStart = tileLoop->getNextNode();
  Block *tbody = tileLoop.getBody();
  Operation *tyield = tbody->getTerminator();
  tbody->getOperations().splice(Block::iterator(tyield), body.getOperations(),
                                Block::iterator(moveStart),
                                Block::iterator(term));

  // Swizzle now indexes by the tile induction var instead of program_id. Keep
  // the tile-loop's own lower-bound operand pointing at the original pid.
  pid.replaceAllUsesExcept(tileLoop.getInductionVar(), tileLoop);

  // Thread the ring: K-loop ring inits come from the tile-loop iter args, and
  // the tile loop yields the K-loop's final ring values for the next tile.
  for (auto en : llvm::enumerate(ringPos))
    kloop.getInitArgsMutable()[en.value()].assign(
        tileLoop.getRegionIterArg(en.index()));

  SmallVector<Value> yieldVals;
  for (unsigned p : ringPos)
    yieldVals.push_back(kloop.getResult(p));
  tyield->setOperands(yieldVals);

  return true;
}

struct PersistentTilePass
    : public PassWrapper<PersistentTilePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PersistentTilePass)

  StringRef getArgument() const override { return "triton-cuda-persistent-tile"; }
  StringRef getDescription() const override {
    return "Wrap warp-specialized GEMM regions in a persistent grid-stride "
           "tile loop (gemm_06-class), threading the pipeline ring across tiles";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<ttg::WarpSpecializeOp> wsOps;
    mod.walk([&](ttg::WarpSpecializeOp ws) { wsOps.push_back(ws); });

    for (ttg::WarpSpecializeOp ws : wsOps) {
      // num_pid_m / num_pid_n are the last two explicit captures (and the last
      // two block args of every partition). Require at least two captures.
      auto caps = ws.getPartitionOp().getExplicitCaptures();
      unsigned n = caps.size();
      if (n < 2)
        continue;
      Value capM = caps[n - 2];
      Value capN = caps[n - 1];
      if (!capM.getType().isInteger(32) || !capN.getType().isInteger(32))
        continue;

      // Default region implicitly captures: the capture SSA values (function
      // args) are in scope directly.
      persistentizeRegion(ws.getDefaultRegion(), capM, capN);

      // Partition regions are isolated-from-above: use their block args.
      for (Region *part : ws.getPartitionRegions()) {
        if (part->empty())
          continue;
        Block &pblk = part->front();
        unsigned na = pblk.getNumArguments();
        if (na < 2)
          continue;
        persistentizeRegion(*part, pblk.getArgument(na - 2),
                             pblk.getArgument(na - 1));
      }
    }
  }
};

} // namespace

namespace triton_cuda {
std::unique_ptr<mlir::Pass> createPersistentTilePass() {
  return std::make_unique<PersistentTilePass>();
}
} // namespace triton_cuda
