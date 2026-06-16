/// Top-level entry point: translate TritonGPU module to CUDA C++ source.
#include "TritonGPUToCUDA/TritonGPUToCUDA.h"
#include "CUDACodeGen.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/CFGToSCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

// The frontend lowers early returns / nested returns to cf-dialect branches.
// The emitter can if-convert tree-shaped CFGs itself, but a merge block with
// multiple predecessors (e.g. `if/else` where both arms fall through) or a cf
// loop cannot be inlined soundly. Lift those back to structured scf.if /
// scf.while via MLIR's generic CFG-to-SCF transform; the emitter already
// handles scf ops. Values left undefined on some paths become ub.poison,
// which the emitter zero-initializes.
static void liftCFGToSCF(ModuleOp module) {
  bool hasMultiBlock = false;
  module.walk([&](triton::FuncOp f) {
    if (!f.getBody().empty() && !llvm::hasSingleElement(f.getBody()))
      hasMultiBlock = true;
  });
  if (!hasMultiBlock)
    return;
  // The transform materializes ub.poison for path-undefined values; make sure
  // the dialect is available in this context.
  DialectRegistry registry;
  registry.insert<mlir::ub::UBDialect>();
  module.getContext()->appendDialectRegistry(registry);
  module.getContext()->loadAllAvailableDialects();
  ControlFlowToSCFTransformation transformation;
  module.walk([&](triton::FuncOp f) {
    if (f.getBody().empty() || llvm::hasSingleElement(f.getBody()))
      return;
    DominanceInfo domInfo(f);
    // Best-effort: on failure the emitter's honest multi-block handling
    // (if-conversion or hard error) still applies.
    (void)transformCFGToSCF(f.getBody(), transformation, domInfo);
  });
}

CUDATranslationResult translateTritonGPUToCUDA(ModuleOp module,
                                                int32_t capability,
                                                int32_t numWarps,
                                                int32_t numCtas,
                                                int32_t ptxVersion) {
  liftCFGToSCF(module);
  CUDACodeGen codegen(module, capability, numWarps, numCtas, ptxVersion);
  return codegen.generate();
}

} // namespace mlir::triton
