// Out-of-tree Triton plugin entry for the CUDA C++ emitter.
//
// This wraps the existing `translateTritonGPUToCUDA` emitter as a regular MLIR
// pass and exposes it through Triton's official plugin ABI
// (`tritonGetPluginInfo`, see include/triton/Tools/PluginUtils.h). The pass is
// loaded at runtime via `TRITON_PLUGIN_PATHS=.../emit_cuda.so` and requires a
// libtriton built with `TRITON_EXT_ENABLED=1` (default symbol visibility).
//
// The emitter returns a CUDA source string plus launch metadata. Because the
// plugin ABI only lets us register passes/dialects/ops (no new pybind function)
// and the Python `module` wrapper exposes no string-attribute getter, the pass
// hands its results back through a small temp file whose path is supplied as the
// 5th pass argument. The Python stages hook reads and parses that file.
//
// File format (text):
//   line 1: "OK" or "ERR"
//   on ERR: remainder = error message
//   on OK:  line 2 = kernel name
//           line 3 = "<shared> <scratchSize> <scratchAlign> <numWarps>"
//           remainder = CUDA C++ source (kept verbatim, may contain newlines)

#include "TritonGPUToCUDA/TritonGPUToCUDA.h"
#include "triton/Tools/PluginUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <system_error>
#include <vector>

using namespace mlir;

namespace {

struct EmitCudaPass
    : public PassWrapper<EmitCudaPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitCudaPass)

  int32_t capability;
  int32_t numWarps;
  int32_t numCtas;
  int32_t ptxVersion;
  std::string outPath;

  EmitCudaPass(int32_t cap, int32_t nw, int32_t nc, int32_t pv,
               std::string out)
      : capability(cap), numWarps(nw), numCtas(nc), ptxVersion(pv),
        outPath(std::move(out)) {}

  StringRef getArgument() const override { return "emit-cuda"; }
  StringRef getDescription() const override {
    return "Translate TritonGPU/NVGPUIR to CUDA C++ (result via temp file)";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto result = mlir::triton::translateTritonGPUToCUDA(
        mod, capability, numWarps, numCtas, ptxVersion);

    // The Python side supplies a temp-file path as the 5th pass arg and reads
    // the result back from it. If no path was given there is nowhere to put the
    // result, so skip silently (e.g. `triton-opt -emit-cuda` IR inspection).
    if (outPath.empty())
      return;

    std::error_code ec;
    llvm::raw_fd_ostream os(outPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      mod.emitError("emit-cuda: cannot open result file '")
          << outPath << "': " << ec.message();
      signalPassFailure();
      return;
    }

    if (!result.ok) {
      os << "ERR\n" << result.errorMessage;
      return;
    }
    // line 1: status; line 2: kernel name; line 3: sizes; rest: CUDA source.
    os << "OK\n";
    os << result.kernelName << "\n";
    os << result.sharedMemSize << " " << result.globalScratchSize << " "
       << result.globalScratchAlign << " " << result.numWarps << "\n";
    os << result.cudaSource;
  }
};

// Parse `[capability, numWarps, numCtas, ptxVersion, outPath]` from the plugin
// args. Built with -fno-exceptions, so use atoi (no throw) rather than stoi.
void addEmitCudaPass(PassManager *pm, const std::vector<std::string> &args) {
  int32_t cap = args.size() > 0 ? std::atoi(args[0].c_str()) : 90;
  int32_t nw = args.size() > 1 ? std::atoi(args[1].c_str()) : 4;
  int32_t nc = args.size() > 2 ? std::atoi(args[2].c_str()) : 1;
  int32_t pv = args.size() > 3 ? std::atoi(args[3].c_str()) : 0;
  std::string out = args.size() > 4 ? args[4] : std::string();
  pm->addPass(std::make_unique<EmitCudaPass>(cap, nw, nc, pv, std::move(out)));
}

// Lets `triton-opt -emit-cuda` find the pass for IR inspection.
void registerEmitCudaPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<EmitCudaPass>(90, 4, 1, 0, std::string());
  });
}

} // namespace

static const char *PLUGIN_NAME = "EmitCudaPlugin";
static const char *PASS_NAME = "emit_cuda";
static const char *VERSION = "0.1.0";

TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  static mlir::triton::plugin::PassInfo pass = {PASS_NAME, VERSION,
                                                addEmitCudaPass,
                                                registerEmitCudaPass};
  static mlir::triton::plugin::PassInfo passes[] = {pass};
  static mlir::triton::plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      PLUGIN_NAME,
      VERSION,
      passes,
      1,
      nullptr,
      0,
      nullptr,
      0,
      TRITON_VERSION};
  return &info;
}
