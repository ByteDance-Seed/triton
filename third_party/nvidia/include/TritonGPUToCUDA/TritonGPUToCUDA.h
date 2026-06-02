#ifndef TRITON_TRITONGPUTOCUDA_H
#define TRITON_TRITONGPUTOCUDA_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace mlir::triton {

struct CUDATranslationResult {
  std::string cudaSource;
  std::string kernelName;
  int64_t sharedMemSize = 0;
};

/// Translate a TritonGPU/NVGPUIR module to CUDA C++ source code.
CUDATranslationResult translateTritonGPUToCUDA(ModuleOp module,
                                                int32_t capability,
                                                int32_t numWarps,
                                                int32_t numCtas);

} // namespace mlir::triton

#endif // TRITON_TRITONGPUTOCUDA_H
