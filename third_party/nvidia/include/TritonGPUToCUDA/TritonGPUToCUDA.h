#ifndef TRITON_TRITONGPUTOCUDA_H
#define TRITON_TRITONGPUTOCUDA_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace mlir::triton {

struct CUDATranslationResult {
  std::string cudaSource;
  std::string kernelName;
  int64_t sharedMemSize = 0;
  int64_t globalScratchSize = 0;
  int64_t globalScratchAlign = 1;
  // Number of warps the kernel launches. For warp-specialized kernels this is
  // wider than the requested num_warps (base + partition warps); the Python
  // launcher uses it to size blockDim.x (= 32 * numWarps). 0 means unset.
  int32_t numWarps = 0;
  // Set to false when an unsupported op was encountered; errorMessage then
  // holds a human-readable description. The pybind caller raises on this.
  bool ok = true;
  std::string errorMessage;
};

/// Translate a TritonGPU/NVGPUIR module to CUDA C++ source code.
CUDATranslationResult translateTritonGPUToCUDA(ModuleOp module,
                                                int32_t capability,
                                                int32_t numWarps,
                                                int32_t numCtas,
                                                int32_t ptxVersion);

} // namespace mlir::triton

#endif // TRITON_TRITONGPUTOCUDA_H
