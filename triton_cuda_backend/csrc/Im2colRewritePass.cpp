// Im2colRewritePass
// =================
// Turns a frontend "placeholder + marker" into a real Hopper TMA **im2col** copy.
//
// The Triton/Gluon frontend cannot create an im2col `ttng.async_tma_copy_global_
// _to_local` directly: the `create_async_tma_copy_global_to_local` binding hard-
// wires the no-`offsets` build variant, no binding builds a `ttng.tensordesc_im2col`
// value, and the trace-time verifier ties im2col mode to the descriptor type. So
// the backend (this pass, all in triton_cuda_backend) synthesizes it. The frontend
// (`_im2col_load` in the tutorial) emits, back to back:
//   tt.elementwise_inline_asm "im2col.marker" : c,w,h,d,n,kw,kh,kt   (8 i32 carrier)
//   ttng.async_tma_copy_global_to_local <a_desc>[0,0] <bar> <smem> <pred>  (placeholder)
// where a_desc is a (tiled) tensordesc whose block type is the 2D smem tile. A
// tagged inline-asm op is used as the carrier (not device_print): its purpose IS to
// pass operands + a verbatim string to the backend, it carries the 8 i32 values as
// real operands, and the tag is matched exactly (no print-prefix mangling). It is
// is_pure=False so Triton's TTGIR DCE keeps it until this pass; the asm body is
// never emitted because this pass erases the op first.
//
// This pass, for each "im2col.marker" carrier: takes the 8 carried values, retypes
// the descriptor block-arg to `ttng.tensordesc_im2col`, and replaces the placeholder
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
    return "Rewrite im2col.marker placeholder into a real TMA im2col copy";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();

    SmallVector<tt::ElementwiseInlineAsmOp> markers;
    // The carrier is a tagged inline-asm op; its asm_string is verbatim (no
    // mangling), so match it exactly.
    mod.walk([&](tt::ElementwiseInlineAsmOp a) {
      if (a.getAsmString() == "im2col.marker")
        markers.push_back(a);
    });

    for (tt::ElementwiseInlineAsmOp marker : markers) {
      auto args = marker.getArgs();
      if (args.size() != 8) {
        marker->emitError("im2col marker expects 8 args (c,w,h,d,n,kw,kh,kt)");
        return signalPassFailure();
      }
      // Find the placeholder copy following the marker in the same block.
      ttng::AsyncTMACopyGlobalToLocalOp copy;
      for (Operation *cur = marker->getNextNode(); cur;
           cur = cur->getNextNode()) {
        if (auto c = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(cur)) {
          copy = c;
          break;
        }
      }
      if (!copy) {
        marker->emitError("no async_tma_copy after im2col.marker carrier");
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
      marker.erase();
    }
  }
};

} // namespace

namespace triton_cuda {
std::unique_ptr<mlir::Pass> createIm2colRewritePass() {
  return std::make_unique<Im2colRewritePass>();
}
} // namespace triton_cuda
