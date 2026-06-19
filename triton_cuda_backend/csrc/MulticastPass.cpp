// P3 MulticastPass
// ================
// Auto-detects which TMA global->shared loads in a cluster (num_ctas>1) GEMM are
// safe to multicast across the CTAs of a CGA, and sets the `multicast` UnitAttr on
// `ttng.async_tma_copy_global_to_local` accordingly. This is the structural
// counterpart of gemm_06's hand-written `cp.async.bulk.tensor ... .multicast`:
// the frontend kernel issues a plain (non-multicast) cluster GEMM, and this pass
// introduces the multicast.
//
// Correctness rule (reverse-engineered from ws_v8.py / gemm_06_autotuned.cu):
//   A multicast TMA load is one issued by all CTAs of the cluster but whose data
//   lands in every CTA's shared memory. That is only correct when every CTA wants
//   the *same* tile, i.e. when the load coordinates are INDEPENDENT of the CTA's
//   rank within the cluster (`%cluster_ctarank`). In the canonical cluster GEMM:
//     off_m = pid_m*CLUSTER_M + cta_rank*CTA_M   (A: rank-DEPENDENT -> no mc)
//     off_n = pid_n*BLOCK_N                       (B: rank-independent -> mc OK)
//   so B (shared N tile) is multicast and A (per-CTA M rows) is not.
//
// Implementation: a conservative forward taint of the `%cluster_ctarank` value.
//   seed   = results of `tt.elementwise_inline_asm` ops whose asm reads
//            `%cluster_ctarank`.
//   spread = any op with a tainted operand taints all its results; region ops
//            (scf.for/if/while) also taint all their region block args; the
//            warp_specialize capture list taints the matching partition block
//            args (capture i -> partition block-arg i). Over-tainting only loses
//            a multicast opportunity, so erring large is safe.
//   apply  = for each async_tma_copy_global_to_local, set `multicast` iff none of
//            its coord operands is tainted (and clear it otherwise).
//
// If no `%cluster_ctarank` seed exists the kernel is not a cluster GEMM, so the
// pass is a no-op (keeps non-cluster codegen byte-identical even with the flag
// on). The emitter additionally guards every multicast with `cgaBroadcastMask!=0`
// (i.e. the buffer actually has a cluster-replicated layout), so a stray attr on
// a single-CTA load can never miscompile.

#include "Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

struct MulticastPass
    : public PassWrapper<MulticastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MulticastPass)

  StringRef getArgument() const override { return "triton-cuda-multicast"; }
  StringRef getDescription() const override {
    return "Set the multicast attr on cluster-rank-independent TMA loads "
           "(gemm_06-class CGA multicast)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // --- seed: results of inline-asm that read %cluster_ctarank --------------
    llvm::DenseSet<Value> tainted;
    SmallVector<Value> worklist;
    auto taint = [&](Value v) {
      if (tainted.insert(v).second)
        worklist.push_back(v);
    };

    mod.walk([&](tt::ElementwiseInlineAsmOp asmOp) {
      if (asmOp.getAsmString().contains("cluster_ctarank"))
        for (Value r : asmOp.getResults())
          taint(r);
    });

    // No cluster-rank read -> not a cluster GEMM -> leave the module untouched.
    if (worklist.empty())
      return;

    // --- forward taint to a fixed point -------------------------------------
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      for (OpOperand &use : v.getUses()) {
        Operation *user = use.getOwner();

        // warp_specialize capture i -> partition block-arg i (regions are
        // isolated-from-above, so taint must be threaded explicitly).
        if (auto parts = dyn_cast<ttg::WarpSpecializePartitionsOp>(user)) {
          unsigned idx = use.getOperandNumber();
          for (Region &part : parts.getPartitionRegions()) {
            if (part.empty())
              continue;
            Block &blk = part.front();
            if (idx < blk.getNumArguments())
              taint(blk.getArgument(idx));
          }
          continue;
        }

        // Region-carrying ops (scf.for/if/while): conservatively taint every
        // region block arg as well as every result.
        for (Region &reg : user->getRegions())
          for (Block &blk : reg)
            for (BlockArgument arg : blk.getArguments())
              taint(arg);

        for (Value r : user->getResults())
          taint(r);
      }
    }

    // --- apply --------------------------------------------------------------
    // Three cases, keyed on whether any coord depends on cta_rank and whether the
    // frontend already requested multicast on this copy:
    //
    //   rank-INDEPENDENT coords  -> plain broadcast multicast. One representative
    //       CTA issues the load and broadcasts the SAME tile to every CTA (gemm_06
    //       A-broadcast / ws_v8 B-broadcast). Set `multicast`.
    //
    //   rank-DEPENDENT coords, frontend-marked multicast -> COOPERATIVE SPLIT. The
    //       frontend has each CTA load a DIFFERENT slice (off_n + cta_rank*HALF_N)
    //       into its own half-buffer and broadcast it (gemm_06 lines 695-701). Each
    //       CTA must issue its OWN copy, so the emitter must NOT representative-gate
    //       it. Keep `multicast` and additionally tag `cooperative` so the emitter
    //       drops the `(_cta_rank & mask)==0` guard. Total B HBM traffic = 1x.
    //
    //   rank-DEPENDENT coords, NOT frontend-marked -> per-CTA PRIVATE operand (A:
    //       each CTA's own M rows). Must never multicast. Clear `multicast`.
    MLIRContext *ctx = mod.getContext();
    UnitAttr unit = UnitAttr::get(ctx);
    StringAttr coopName = StringAttr::get(ctx, "cooperative");
    mod.walk([&](ttng::AsyncTMACopyGlobalToLocalOp copy) {
      bool rankDependent = false;
      for (Value c : copy.getCoord())
        if (tainted.contains(c)) {
          rankDependent = true;
          break;
        }
      // Control-dependence on cta_rank: the cooperative split issues each half
      // from a DIFFERENT CTA via `if cta_rank == 0 / else`, so the B-half coords
      // themselves are rank-INDEPENDENT (plain off_n / off_n+HALF_N) but the copy
      // is nested in a region whose guard is cta_rank-tainted. Walk the parent
      // chain; if any enclosing op has a tainted operand (e.g. the scf.if
      // condition), the copy is effectively per-CTA -> treat as rank-dependent.
      if (!rankDependent) {
        for (Operation *p = copy->getParentOp();
             p && !isa<ModuleOp>(p) && !isa<tt::FuncOp>(p);
             p = p->getParentOp()) {
          bool anyTainted = false;
          for (Value o : p->getOperands())
            if (tainted.contains(o)) {
              anyTainted = true;
              break;
            }
          if (anyTainted) {
            rankDependent = true;
            break;
          }
        }
      }
      bool wasMulticast = copy->hasAttr(copy.getMulticastAttrName());
      if (!rankDependent) {
        copy->setAttr(copy.getMulticastAttrName(), unit);
      } else if (wasMulticast) {
        // cooperative split: keep multicast, mark cooperative (each CTA issues).
        copy->setAttr(copy.getMulticastAttrName(), unit);
        copy->setAttr(coopName, unit);
      } else {
        copy->removeAttr(copy.getMulticastAttrName());
      }
    });
  }
};

} // namespace

namespace triton_cuda {
std::unique_ptr<mlir::Pass> createMulticastPass() {
  return std::make_unique<MulticastPass>();
}
} // namespace triton_cuda
