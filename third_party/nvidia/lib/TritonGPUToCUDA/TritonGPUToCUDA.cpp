/// Top-level entry point: translate TritonGPU module to CUDA C++ source.
#include "TritonGPUToCUDA/TritonGPUToCUDA.h"
#include "CUDACodeGen.h"

namespace mlir::triton {

CUDATranslationResult translateTritonGPUToCUDA(ModuleOp module,
                                                int32_t capability,
                                                int32_t numWarps,
                                                int32_t numCtas) {
  CUDACodeGen codegen(module, capability, numWarps, numCtas);
  return codegen.generate();
}

} // namespace mlir::triton
