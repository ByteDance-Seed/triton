// Im2colRewritePass
// =================
// Turns a frontend "placeholder + marker" into a real Hopper TMA **im2col** copy.
//
// The Triton/Gluon frontend cannot create an im2col `ttng.async_tma_copy_global_
// _to_local` directly: no pristine Python binding passes `offsets` or builds a
// `ttng.tensordesc_im2col` value, and the trace-time verifier ties im2col mode to
// the descriptor type. So the backend (this pass, all in triton_cuda_backend)
// synthesizes it. The frontend builtin `tma.async_copy_im2col` (in _gluon_ext.py)
// emits, back to back:
//   tt.print "__IM2COL__" : c, w, h, d, n, kw, kh, kt   (carries the 8 i32 vals)
//   ttng.async_tma_copy_global_to_local <a_desc>[0,0] <bar> <smem> <pred>  (placeholder)
// where a_desc is a (tiled) tensordesc whose block type is the 2D smem tile.
//
// This pass, for each "__IM2COL__" print: takes the 8 carried values, retypes the
// descriptor block-arg to `ttng.tensordesc_im2col`, and replaces the placeholder
// copy with a real im2col copy
//   coords  = [n, d, h, w, c]   (IR order; the emitter reverses -> PTX {c,w,h,d,n})
//   offsets = [kw, kh, kt]      (i16, the kernel taps)
// then erases the print + placeholder. The host builds the matching im2col
// CUtensorMap (cuTensorMapEncodeIm2col); the emitter prints the `.im2col` PTX.

#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

struct Im2colRewritePass
    : public PassWrapper<Im2colRewritePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Im2colRewritePass)

  StringRef getArgument() const override { return "im2col-rewrite"; }
  StringRef getDescription() const override {
    return "Rewrite __IM2COL__ placeholder into a real TMA im2col copy";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    SmallVector<tt::PrintOp> markers;
    // device_print mangles the prefix (leading " ", trailing ": "), so match by
    // substring rather than prefix.
    mod.walk([&](tt::PrintOp p) {
      if (p.getPrefix().contains("__IM2COL__"))
        markers.push_back(p);
    });

    for (tt::PrintOp print : markers) {
      auto args = print.getArgs();
      if (args.size() != 8) {
        print->emitError("im2col marker expects 8 args (c,w,h,d,n,kw,kh,kt)");
        return signalPassFailure();
      }
      // Find the placeholder copy following the marker in the same block.
      ttng::AsyncTMACopyGlobalToLocalOp copy;
      for (Operation *cur = print->getNextNode(); cur;
           cur = cur->getNextNode()) {
        if (auto c = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(cur)) {
          copy = c;
          break;
        }
      }
      if (!copy) {
        print->emitError("no async_tma_copy after __IM2COL__ marker");
        return signalPassFailure();
      }

      OpBuilder b(copy);
      Location loc = copy.getLoc();
      Value c = args[0], w = args[1], h = args[2], d = args[3], n = args[4];
      Value kw = args[5], kh = args[6], kt = args[7];

      // Retype the descriptor (a function block-arg) to tensordesc_im2col, once.
      Value desc = copy.getDesc();
      RankedTensorType blockTy;
      if (auto t = dyn_cast<tt::TensorDescType>(desc.getType()))
        blockTy = t.getBlockType();
      else if (auto t = dyn_cast<ttng::TensorDescIm2ColType>(desc.getType()))
        blockTy = t.getBlockType();
      else {
        copy->emitError("im2col: unexpected descriptor type");
        return signalPassFailure();
      }
      if (!isa<ttng::TensorDescIm2ColType>(desc.getType())) {
        auto im2colTy = ttng::TensorDescIm2ColType::get(ctx, blockTy);
        auto barg = dyn_cast<BlockArgument>(desc);
        if (!barg) {
          copy->emitError("im2col: descriptor must be a kernel argument");
          return signalPassFailure();
        }
        barg.setType(im2colTy);
        if (auto func = dyn_cast<tt::FuncOp>(barg.getOwner()->getParentOp())) {
          SmallVector<Type> inTys(func.getArgumentTypes().begin(),
                                  func.getArgumentTypes().end());
          inTys[barg.getArgNumber()] = im2colTy;
          func.setType(
              FunctionType::get(ctx, inTys, func.getResultTypes()));
        }
      }

      // coords IR order [n,d,h,w,c]; offsets i16 [kw,kh,kt].
      SmallVector<Value> coords = {n, d, h, w, c};
      auto i16 = b.getIntegerType(16);
      SmallVector<Value> offs = {
          b.create<arith::TruncIOp>(loc, i16, kw),
          b.create<arith::TruncIOp>(loc, i16, kh),
          b.create<arith::TruncIOp>(loc, i16, kt)};

      b.create<ttng::AsyncTMACopyGlobalToLocalOp>(
          loc, desc, ValueRange(coords), ValueRange(offs), copy.getBarrier(),
          copy.getResult(), copy.getPred(), /*multicast=*/false,
          tt::CacheModifier::NONE, tt::EvictionPolicy::NORMAL,
          /*isVolatile=*/false);

      copy.erase();
      print.erase();
    }
  }
};

} // namespace

namespace triton_cuda {
std::unique_ptr<mlir::Pass> createIm2colRewritePass() {
  return std::make_unique<Im2colRewritePass>();
}
} // namespace triton_cuda
