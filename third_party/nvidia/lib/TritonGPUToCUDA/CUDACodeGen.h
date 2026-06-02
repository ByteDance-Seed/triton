#ifndef TRITON_TRITONGPUTOCUDA_CUDACODEGEN_H
#define TRITON_TRITONGPUTOCUDA_CUDACODEGEN_H

#include "TritonGPUToCUDA/TritonGPUToCUDA.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::triton {
namespace gpu {
class BlockedEncodingAttr;
class NvidiaMmaEncodingAttr;
class NVMMASharedEncodingAttr;
class SliceEncodingAttr;
} // namespace gpu

class CUDACodeGen {
public:
  CUDACodeGen(ModuleOp module, int32_t capability, int32_t numWarps,
              int32_t numCtas);
  CUDATranslationResult generate();

private:
  // ── Emit infrastructure ──────────────────────────────────────────────
  void emit(const llvm::Twine &line);
  void emitBlank();
  void indent();
  void dedent();
  std::string newVar(llvm::StringRef prefix = "v");

  // ── Value tracking ───────────────────────────────────────────────────
  std::string getVar(Value val);
  std::string getCUDAType(Type type);
  std::string getElemTypeStr(Value val);
  int getElemsPerThread(Value val);
  int getElemsPerThread(RankedTensorType ty);
  bool isTensorValue(Value val);
  llvm::ArrayRef<int64_t> getTensorShape(Value val);

  // ── Layout utilities ─────────────────────────────────────────────────
  // Compute (row, col) for element index _i in blocked layout
  void emitBlockedStoreToSmem(llvm::StringRef srcVar, llvm::StringRef smemVar,
                              gpu::BlockedEncodingAttr enc,
                              llvm::ArrayRef<int64_t> shape,
                              llvm::StringRef elemType, int nElems);
  void emitBlockedLoadFromSmem(llvm::StringRef dstVar, llvm::StringRef smemVar,
                               gpu::BlockedEncodingAttr enc,
                               llvm::ArrayRef<int64_t> shape,
                               llvm::StringRef elemType, int nElems);
  std::string getBroadcastIndexExpr(Value src, RankedTensorType dstTy,
                                    int srcN, int dstN);

  // Swizzle helpers for NVMMAShared layout
  void emitSwizzledStoreToSmem(llvm::StringRef srcVar, llvm::StringRef dstVar,
                               gpu::BlockedEncodingAttr srcEnc,
                               llvm::ArrayRef<int64_t> shape,
                               llvm::StringRef elemType,
                               int swizzleByteWidth);

  // ── Op dispatch ──────────────────────────────────────────────────────
  void emitOp(Operation *op);
  void emitBlock(Block &block);
  void emitRegion(Region &region);

  // ── Per-op emitters (arith + math) ─────────────────────────────────
  void emitArithConstant(arith::ConstantOp op);
  void emitArithBinary(Operation *op); // addi, addf, muli, mulf, etc.
  void emitArithCast(Operation *op);   // extsi, extui, truncf, sitofp, etc.
  void emitArithCmp(Operation *op);    // cmpi, cmpf
  void emitArithSelect(arith::SelectOp op);
  void emitMathOp(Operation *op); // exp2, sqrt, abs, etc.

  // ── Per-op emitters (triton core) ──────────────────────────────────
  void emitGetProgramId(triton::GetProgramIdOp op);
  void emitMakeRange(triton::MakeRangeOp op);
  void emitSplat(triton::SplatOp op);
  void emitBroadcast(triton::BroadcastOp op);
  void emitExpandDims(triton::ExpandDimsOp op);
  void emitAddPtr(triton::AddPtrOp op);
  void emitLoad(triton::LoadOp op);
  void emitStore(triton::StoreOp op);
  void emitDot(triton::DotOp op);
  void emitReduce(triton::ReduceOp op);
  void emitTrans(triton::TransOp op);
  void emitBitcast(triton::BitcastOp op);
  void emitIntToPtr(triton::IntToPtrOp op);
  void emitExternElementwise(triton::ExternElementwiseOp op);
  void emitAtomicRMW(triton::AtomicRMWOp op);
  void emitAtomicCAS(triton::AtomicCASOp op);
  void emitPrint(triton::PrintOp op);
  void emitMulhiUI(triton::MulhiUIOp op);
  void emitPtrToInt(triton::PtrToIntOp op);
  void emitFpToFp(triton::FpToFpOp op);

  // ── Per-op emitters (scf) ──────────────────────────────────────────
  void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitScfWhile(scf::WhileOp op);
  void emitScfYield(scf::YieldOp op, scf::ForOp parentFor);

  // ── Per-op emitters (triton GPU) ───────────────────────────────────
  void emitLocalAlloc(gpu::LocalAllocOp op);
  void emitLocalStore(gpu::LocalStoreOp op);
  void emitLocalLoad(gpu::LocalLoadOp op);
  void emitLocalDealloc(gpu::LocalDeallocOp op);
  void emitConvertLayout(gpu::ConvertLayoutOp op);
  void emitMemDescSubview(Operation *op); // memdesc_subview / memdesc_index
  void emitMemDescTrans(gpu::MemDescTransOp op);

  // ── Per-op emitters (NVGPUIR / sm90a) ─────────────────────────────
  void emitWarpGroupDot(nvidia_gpu::WarpGroupDotOp op);
  void emitWarpGroupDotWait(nvidia_gpu::WarpGroupDotWaitOp op);
  void emitFenceAsyncShared(nvidia_gpu::FenceAsyncSharedOp op);
  void emitInitBarrier(nvidia_gpu::InitBarrierOp op);
  void emitWaitBarrier(nvidia_gpu::WaitBarrierOp op);
  void emitArriveBarrier(nvidia_gpu::ArriveBarrierOp op);
  void emitBarrierExpect(nvidia_gpu::BarrierExpectOp op);
  void emitInvalBarrier(nvidia_gpu::InvalBarrierOp op);
  void emitAsyncTMACopyG2L(nvidia_gpu::AsyncTMACopyGlobalToLocalOp op);
  void emitAsyncTMACopyL2G(nvidia_gpu::AsyncTMACopyLocalToGlobalOp op);
  void emitTMAStoreWait(nvidia_gpu::TMAStoreWaitOp op);

  // ── Per-op emitters (TTG async copy / cp.async) ───────────────────
  void emitAsyncCopyG2L(gpu::AsyncCopyGlobalToLocalOp op);
  void emitAsyncCommitGroup(gpu::AsyncCommitGroupOp op);
  void emitAsyncWait(gpu::AsyncWaitOp op);

  // ── cp.async helper (shared by emitAsyncCopyG2L and fused load+alloc) ──
  void emitCpAsyncToShared(Value ptrTensor, const std::string &dstVar,
                           gpu::MemDescType dstMemDescType,
                           Value mask, Value other);

  // ── WGMMA helpers ─────────────────────────────────────────────────
  uint64_t getWGMMADescTemplate(int swizzleBytes, int strideDimSize);

  // ── State ──────────────────────────────────────────────────────────
  ModuleOp module;
  int32_t capability, numWarps, numCtas;

  // SSA value → CUDA var name
  llvm::DenseMap<Value, std::string> valueToVar;
  // SSA value → element type string (e.g., "f32", "f16")
  llvm::DenseMap<Value, std::string> valueToElemType;
  // Tensor values that are really scalars (from splat) — don't materialize arrays
  llvm::DenseSet<Value> scalarValues;
  // Deferred addptr: store (base_scalar, offset_array) instead of ptr array
  // Key: result Value of addptr. Value: (base_var_name, offset_var_name)
  llvm::DenseMap<Value, std::pair<std::string, std::string>> deferredAddPtr;
  // For loop iter_args that carry offsets: regionIterArg -> base pointer name
  llvm::DenseMap<Value, std::string> iterArgDeferredBase;
  // Helper: get element access expression (returns "var" for scalars, "var[idx]" for arrays)
  std::string getElemExpr(Value val, const std::string &idx);
  // Helper: get pointer element expression (handles deferred addptr)
  std::string getPtrElemExpr(Value val, const std::string &idx);
  bool isScalarValue(Value val) { return scalarValues.contains(val); }

  int varCounter = 0;
  int indentLevel = 0;
  int sharedMemOffset = 0;
  int peakSharedMem = 0;
  bool hasPendingCpAsync = false;
  // Loads deferred for cp.async fusion (tt.load → local_alloc)
  llvm::DenseSet<Value> deferredCpAsyncLoads;
  std::string kernelName;
  llvm::SmallVector<std::string> lines;

  // Layout alias map (from module attributes)
  llvm::DenseMap<Attribute, Attribute> layoutAliases;
};

} // namespace mlir::triton

#endif // TRITON_TRITONGPUTOCUDA_CUDACODEGEN_H
