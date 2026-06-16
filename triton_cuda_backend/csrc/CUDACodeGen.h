#ifndef TRITON_TRITONGPUTOCUDA_CUDACODEGEN_H
#define TRITON_TRITONGPUTOCUDA_CUDACODEGEN_H

#include "TritonGPUToCUDA/TritonGPUToCUDA.h"
#include "triton/Analysis/AxisInfo.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <string>

namespace mlir::triton {
namespace gpu {
class BlockedEncodingAttr;
class NvidiaMmaEncodingAttr;
class NVMMASharedEncodingAttr;
class SliceEncodingAttr;
class MemDescType;
class Fp4ToFpOp;
} // namespace gpu

// Per-message plan for a TMA global<->local transfer of a swizzled NVMMA tile.
// A tile whose fast dimension exceeds the swizzle width is delivered as several
// cp.async.bulk "messages". `smemRank` is the rank of the shared tile (which may
// be smaller than the descriptor coordinate rank for ragged/persistent TMA).
// Per copy: `shMemElemOffset[c]` is the shared-memory element offset and
// `coordOff[c]` holds the per-smem-dim global coordinate offsets.
struct TMACopyPlan {
  int numCopies;
  int smemRank;
  llvm::SmallVector<int> shMemElemOffset;
  llvm::SmallVector<llvm::SmallVector<int>> coordOff;
  // CGA: per cluster-CTA-rank bit, the per-smem-dim global coordinate delta
  // (CTASplitNum slicing, mirrors msgToOffset's kBlock input). Empty for 1 CTA.
  llvm::SmallVector<llvm::SmallVector<int>> blockCoordDelta;
};

class CUDACodeGen {
public:
  CUDACodeGen(ModuleOp module, int32_t capability, int32_t numWarps,
              int32_t numCtas, int32_t ptxVersion);
  CUDATranslationResult generate();

private:
  // ── Emit infrastructure ──────────────────────────────────────────────
  void emit(const llvm::Twine &line);
  // Block-wide barrier that is warp_specialize-aware: a plain __syncthreads()
  // outside any region, or a region-scoped named bar.sync inside one.
  void blockSync();
  void emitBlank();
  void indent();
  void dedent();
  std::string newVar(llvm::StringRef prefix = "v");

  // ── Value tracking ───────────────────────────────────────────────────
  std::string getVar(Value val);
  std::string getCUDAType(Type type);
  std::string getElemTypeStr(Value val);
  std::string descAddrExpr(Value val);
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
                               llvm::StringRef elemType, int nElems,
                               int rowLo = 0, int rowHi = -1);
  std::string getBroadcastIndexExpr(Value src, RankedTensorType dstTy,
                                    int srcN, int dstN);
  // Ground-truth dst-register -> src-register mapping for a broadcast,
  // derived from LinearLayout (handles any register ordering).
  llvm::SmallVector<int> broadcastRegMapping(RankedTensorType srcRtt,
                                             RankedTensorType dstRtt,
                                             int srcN, int nElems);

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
  // Structured emission of an unstructured (cf dialect) CFG via if-conversion:
  // inlines single-predecessor successor blocks. Used for early-return etc.
  void emitCFBlock(Block *block);
  void assignBlockArgs(Block *dest, mlir::ValueRange operands);
  llvm::DenseSet<Block *> cfVisited;

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
  void emitUnsplat(triton::UnsplatOp op);
  void emitBroadcast(triton::BroadcastOp op);
  void emitExpandDims(triton::ExpandDimsOp op);
  void emitAddPtr(triton::AddPtrOp op);
  void emitLoad(triton::LoadOp op);
  void emitStore(triton::StoreOp op);
  void emitDot(triton::DotOp op);
  void emitReduce(triton::ReduceOp op);
  void emitScan(triton::ScanOp op);
  void emitGather(triton::GatherOp op);
  // Generic scalar inliner for a reduce/scan combine region. Binds the region's
  // block arguments to the given scalar CUDA expressions, emits one temp var per
  // region op (so shared sub-expressions are not duplicated), and returns the
  // scalar expression(s) yielded by the region terminator. Supports arbitrary
  // combines (e.g. routing's keyed-add), not just add/min/max. Sets emitFailed
  // and returns false on an unsupported op so we hard-error instead of silently
  // miscompiling. `elemType` names the CUDA type for declaring temps.
  bool emitScalarCombine(Region &region, llvm::ArrayRef<std::string> argExprs,
                         llvm::SmallVectorImpl<std::string> &results);
  void emitTrans(triton::TransOp op);
  void emitBitcast(triton::BitcastOp op);
  void emitIntToPtr(triton::IntToPtrOp op);
  void emitExternElementwise(triton::ExternElementwiseOp op);
  void emitAtomicRMW(triton::AtomicRMWOp op);
  void emitAtomicCAS(triton::AtomicCASOp op);
  // lane/warp free-bit redundancy predicate ("" if no replication)
  std::string redundantThreadGuard(mlir::RankedTensorType ty);
  // smem round-trip broadcasting canonical threads' atomic old-values to
  // replica threads (mirrors finalizeTensorAtomicResults in the PTX backend)
  void emitAtomicResultBroadcast(mlir::RankedTensorType ty,
                                 const std::string &var,
                                 const std::string &cudaType,
                                 const std::string &redGuard);
  void emitPrint(triton::PrintOp op);
  void emitMulhiUI(triton::MulhiUIOp op);
  void emitPtrToInt(triton::PtrToIntOp op);
  void emitFpToFp(triton::FpToFpOp op);
  void emitFp4ToFp(triton::gpu::Fp4ToFpOp op);
  void emitElementwiseInlineAsm(triton::ElementwiseInlineAsmOp op);
  void emitMapElementwise(triton::MapElementwiseOp op);
  // tt.call (noinline functions): emit-time inlining. Looks up the callee
  // tt.func, aliases its entry-block args to the call operands, re-walks the
  // body through emitOp, and maps tt.return operands to the call results.
  void emitCall(triton::CallOp op);
  int callInlineDepth = 0;

  // ── Per-op emitters (scf) ──────────────────────────────────────────
  void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitScfIndexSwitch(scf::IndexSwitchOp op);
  void emitScfWhile(scf::WhileOp op);
  void emitScfYield(scf::YieldOp op, scf::ForOp parentFor);

  // ── Per-op emitters (triton GPU) ───────────────────────────────────
  void emitLocalAlloc(gpu::LocalAllocOp op);
  void emitLocalStore(gpu::LocalStoreOp op);
  // Layout-aware register→shared store via invertAndCompose. Works for any
  // element size (including fp8) and any src distributed / dst swizzled layout.
  // Does NOT emit a trailing __syncthreads (caller decides).
  void emitLayoutAwareSharedStore(mlir::Value val, gpu::MemDescType memDescType,
                                  llvm::StringRef dstVar);
  void emitLocalLoad(gpu::LocalLoadOp op);
  void emitLocalDealloc(gpu::LocalDeallocOp op);
  void emitConvertLayout(gpu::ConvertLayoutOp op);
  void emitMemDescSubview(Operation *op); // memdesc_subview / memdesc_index
  void emitMemDescTrans(gpu::MemDescTransOp op);

  // ── Warp specialization (custom Hopper producer/consumer codegen) ──────
  void emitWarpSpecialize(gpu::WarpSpecializeOp op);
  // Emit the ops of a single warp_specialize region (one block), skipping the
  // region terminator (warp_yield / warp_return).
  void emitWSRegionBody(Region &region);

  // ── Per-op emitters (NVGPUIR / sm90a) ─────────────────────────────
  void emitWarpGroupDot(nvidia_gpu::WarpGroupDotOp op);
  void emitWarpGroupDotWait(nvidia_gpu::WarpGroupDotWaitOp op);
  void emitFenceAsyncShared(nvidia_gpu::FenceAsyncSharedOp op);
  void emitInitBarrier(nvidia_gpu::InitBarrierOp op);
  void emitWaitBarrier(nvidia_gpu::WaitBarrierOp op);
  void emitArriveBarrier(nvidia_gpu::ArriveBarrierOp op);
  void emitBarrierExpect(nvidia_gpu::BarrierExpectOp op);
  void emitInvalBarrier(nvidia_gpu::InvalBarrierOp op);
  TMACopyPlan computeTMACopyPlan(gpu::MemDescType smemTy, int coordRank);
  void emitAsyncTMACopyG2L(nvidia_gpu::AsyncTMACopyGlobalToLocalOp op);
  void emitAsyncTMACopyL2G(nvidia_gpu::AsyncTMACopyLocalToGlobalOp op);
  void emitAsyncTMAReduce(nvidia_gpu::AsyncTMAReduceOp op);
  // Shared body for TMA store-like ops (plain bulk store / reduce-store):
  // `reduceKind` empty → cp.async.bulk.tensor; else cp.reduce.async.bulk.tensor
  // with the given reduction qualifier (add/min/max/and/or/xor/...).
  void emitTMAStoreLike(Value desc, Value src, mlir::OperandRange coords,
                        llvm::StringRef reduceKind);
  void emitTMAStoreWait(nvidia_gpu::TMAStoreWaitOp op);

  // ── Device-side TMA descriptor creation (make_tensor_descriptor) ──────
  void emitGlobalScratchAlloc(gpu::GlobalScratchAllocOp op);
  void emitTensormapCreate(nvidia_gpu::TensormapCreateOp op);
  void emitTensormapFenceproxyAcquire(nvidia_gpu::TensormapFenceproxyAcquireOp op);
  void emitReinterpretTensorDesc(nvidia_gpu::ReinterpretTensorDescOp op);

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
  int32_t capability, numWarps, numCtas, ptxVersion;

  // Max safe vectorization width (in elements) for a tensor-of-pointers value,
  // from AxisInfo (min of per-thread contiguity and pointer alignment). Used to
  // clamp load/store vectorization so we never emit an over-wide (misaligned)
  // vector access. Lazily builds ModuleAxisInfoAnalysis on first use.
  unsigned getPtrContiguity(Value ptr);
  // Max safe vector width (elements) bounding contiguity AND, when a mask is
  // present, the mask's alignment (so a vector group is never partially masked,
  // which would otherwise read/write out-of-bounds tail elements).
  unsigned getMaxVecWidth(Value ptr, Value mask);
  std::unique_ptr<ModuleAxisInfoAnalysis> axisInfoAnalysis;

  // SSA value → CUDA var name
  llvm::DenseMap<Value, std::string> valueToVar;
  // fp8 warp-shuffle convert results that were emitted as pre-packed
  // uint32_t[N] (one word per 4 fp8 elements, in wgmma A-operand order) instead
  // of a byte array. Maps the convert result → number of uint32 words. The
  // consuming WarpGroupDot (RS-mode A) reads these words directly, skipping the
  // byte-unpack/re-pack round-trip. See emitConvertLayout fused path.
  llvm::DenseMap<Value, int> packedU32Convert;
  // local_alloc result → its assigned shared-memory byte offset. Used by
  // emitLocalDealloc to free the buffer by popping the bump pointer down to
  // this offset (instead of resetting to 0, which would clobber still-live
  // lower buffers such as a resident Q tile in flash-attention).
  llvm::DenseMap<Value, int> valueToSmemOffset;
  // Liveness-based smem reclaim for SOURCED (immutable, `local_alloc %v`)
  // buffers, which have NO `local_dealloc` and would otherwise leak in the
  // bump allocator forever — overflowing shared memory in kernels that hold
  // several long-lived sourced tiles (e.g. flash-attention BACKWARD holds
  // k/v/q/do all sourced; without reclaim it needs 294912 B > 232448 limit).
  //
  // We pre-scan each function to assign a program-order index to every op and,
  // for each sourced local_alloc, the order index of its LAST use. Then in
  // emitLocalAlloc, before placing the new buffer, any sourced buffer whose
  // last use precedes the current alloc is retired (removed from the live set).
  // The new buffer is placed above all still-live buffers (floor = max top of
  // the live set), so a dead sourced tile's space is reclaimed exactly like a
  // deallocated mutable buffer — but never below a still-live tile.
  llvm::DenseMap<Operation *, int> smemOpOrder;     // op -> program-order index
  llvm::DenseMap<Value, int> sourcedLastUse;        // sourced buf -> last-use order
  llvm::SmallVector<Value> sourcedBuffers;          // all sourced local_alloc results
  // Currently-live smem buffers (mutable: alloc→dealloc; sourced: alloc→retire)
  // mapped to their top offset (offset + size). recomputeSmemFloor() returns the
  // max top across this set (or 0), i.e. the lowest offset a new buffer may take.
  llvm::DenseMap<Value, int> liveSmemTop;
  int recomputeSmemFloor();
  // SSA value → element type string (e.g., "f32", "f16")
  llvm::DenseMap<Value, std::string> valueToElemType;
  // Tensor values that are really scalars (from splat) — don't materialize arrays
  llvm::DenseSet<Value> scalarValues;
  // Deferred addptr: store (base_scalar, offset_array) instead of ptr array
  // Key: result Value of addptr. Value: (base_var_name, offset_var_name)
  llvm::DenseMap<Value, std::pair<std::string, std::string>> deferredAddPtr;
  // WGMMA accumulator-reset cycle: an arith.select(pred, zeros, dotAcc) that
  // feeds a loop-carried WGMMA accumulator. Naively lowering it as
  // `next[i] = pred ? 0 : acc[i]` reads the live WGMMA accumulator registers
  // mid-pipeline (between wgmma.wait_group 1 and 0), triggering ptxas C7514
  // serialization (~1.5x slowdown). Instead we alias the select result to the
  // accumulator and defer a guarded zero-reset to the loop tail (after
  // wgmma.wait_group 0). Key: select result Value. Value: reset condition var.
  llvm::DenseMap<Value, std::string> deferredAccReset;
  // For loop iter_args that carry offsets: regionIterArg -> base pointer name
  llvm::DenseMap<Value, std::string> iterArgDeferredBase;
  // Pointer-based deferred addptr: loop-carried variables stored as const char*
  // instead of int offsets. Key: Value; Value: (bytesPerElem, elementCUDAType)
  // This eliminates the 32-bit offset → 64-bit address conversion overhead,
  // producing direct 64-bit pointer increments (IADD3+IADD3.X vs VIADD+LEA+IMAD.X)
  llvm::DenseMap<Value, std::pair<int, std::string>> ptrBasedDeferred;
  // Scalar-base deferred: loop-carried pointer arrays where all elements advance
  // by the same stride. Instead of carrying N pointers (2N registers), carry one
  // scalar byte offset (1 register). Eliminates phi-node register copies (IMAD.MOV).
  // Key: Value; Value: {basePtrVar, staticOffsetsVar, bytesPerElem, scalarDeltaVar}
  struct ScalarBaseInfo {
    std::string precompAddr;   // unsigned long long array: base + offset[i]*bpe
    int bpe;                   // bytes per element
    std::string deltaVar;      // scalar int delta (loop-carried, in bytes)
    std::string cType;         // CUDA element type for the pointee (e.g. "int")
  };
  llvm::DenseMap<Value, ScalarBaseInfo> scalarBaseDeferred;
  // ── Elementwise loop fusion (lazy materialization) ────────────────────
  // A pure elementwise op whose result is a tensor with a SINGLE use in the
  // SAME block is not emitted as its own `float v[N]; for(_i)v[_i]=...` loop.
  // Instead its per-element RHS is recorded as a `build(idx)` closure and
  // inlined directly into the consumer's loop via getElemExpr — so the chain
  // collapses into one loop body and intermediates become loop-local temps
  // instead of N-wide function-scope arrays. This matches what Triton's LLVM
  // backend does and cuts register pressure dramatically on large per-thread
  // reduction/elementwise kernels (e.g. layer-norm backward: 162 regs -> ~40).
  //
  // Safety: deferral requires single-use + same-block so (a) the inlined
  // expression's operands are always in C++ scope at the (single) use point,
  // and (b) we never recompute a value inside an scf.for that iterates K times.
  // getVar() transparently materializes a deferred value if any consumer needs
  // the array form, so unknown/array-consuming ops remain correct. Toggle off
  // with TRITON_CUDA_NO_FUSE for bisection.
  struct DeferredElemDef {
    std::function<std::string(const std::string &)> build;
    int nElems;
    std::string cudaType;
  };
  llvm::DenseMap<Value, DeferredElemDef> deferredElem;
  bool fuseElementwise = true;
  bool shouldDeferElementwise(Value result, Operation *op);
  std::string materializeDeferred(Value val);
  // Record `result` as a deferred elementwise value, or (when deferral is
  // unsafe/disabled) emit the materialized array + unrolled fill loop now.
  void emitOrDeferElementwise(Value result, Operation *op, int nElems,
                              const std::string &cudaType,
                              std::function<std::string(const std::string &)> build);
  // Helper: get element access expression (returns "var" for scalars, "var[idx]" for arrays)
  std::string getElemExpr(Value val, const std::string &idx);
  // Helper: get pointer element expression (handles deferred addptr)
  std::string getPtrElemExpr(Value val, const std::string &idx);
  // Helper: get pointer ADDRESS expression (handles deferred addptr) — like
  // getPtrElemExpr but without dereferencing; used by atomic ops.
  std::string getPtrAddrExpr(Value val, const std::string &idx);
  bool isScalarValue(Value val) { return scalarValues.contains(val); }

  // Device-side global scratch (TMA descriptors): per-alloc byte offset within
  // the per-CTA scratch region, descriptors that are CUtensorMap* (vs host
  // grid_constant CUtensorMap), and the per-CTA total scratch size/alignment.
  // Per-op LIST of offsets: a GlobalScratchAllocOp inside a tt.func callee is
  // inlined (re-emitted) once per call site and each instance needs its own
  // scratch slot; emitGlobalScratchAlloc consumes the list via the cursor.
  llvm::DenseMap<Operation *, llvm::SmallVector<int, 2>> globalScratchOffsets;
  llvm::DenseMap<Operation *, int> globalScratchCursor;
  llvm::DenseSet<Value> pointerDescriptors;
  int64_t globalScratchSize = 0;
  int64_t globalScratchAlign = 1;

  // Set when an unsupported op is encountered. generate() propagates this into
  // the result so the pybind caller can raise a Python exception (this TU is
  // compiled with -fno-exceptions, so we cannot throw here directly).
  bool emitFailed = false;
  std::string emitErrorMsg;

  // Source-level line info: when enabled (TRITON_DISABLE_LINE_INFO != 1),
  // emitOp records the current op's source location and emit() interleaves
  // #line directives so nvcc -lineinfo maps SASS back to the .py file. Since
  // a #line directive only sets the number of the *next* physical line (and
  // it auto-increments after that), emit() re-emits the directive whenever
  // the effective line nvcc would assume diverges from the op's location —
  // in practice before every generated line of a multi-line op body.
  bool lineInfoEnabled = false;
  int curLocLine = 0;          // source line of the op being emitted (0=none)
  std::string curLocFile;
  int effectiveLine = -1;      // line nvcc currently assumes for the next line
  std::string effectiveFile;

  // Total warps the kernel launches. For warp-specialized kernels this is the
  // base num_warps plus the partition warps (so blockDim widens to cover the
  // producer + consumer warpgroups); 0 means "no widening, use numWarps".
  int totalNumWarps = 0;

  // While emitting the body of a warp_specialize region, block-wide
  // __syncthreads() must be replaced by a named barrier scoped to just that
  // region's warps — the other warp-groups take a different branch and never
  // reach the sync, so a block-wide bar.sync 0 would deadlock. 0 => emit a
  // plain __syncthreads(); >0 => emit `bar.sync <id>, <wsSyncThreadCount>`.
  int wsSyncBarrierId = 0;
  int wsSyncThreadCount = 0;

  int varCounter = 0;
  int indentLevel = 0;
  int sharedMemOffset = 0;
  int peakSharedMem = 0;
  bool hasPendingCpAsync = false;
  bool hasEmittedWgmShfl = false; // __shfl_sync for _wg_m hoisted out of loop
  // Loads deferred for cp.async fusion (tt.load → local_alloc)
  llvm::DenseSet<Value> deferredCpAsyncLoads;
  std::string kernelName;
  llvm::SmallVector<std::string> lines;

  // Layout alias map (from module attributes)
  llvm::DenseMap<Attribute, Attribute> layoutAliases;
};

} // namespace mlir::triton

#endif // TRITON_TRITONGPUTOCUDA_CUDACODEGEN_H
