/// CUDACodeGen: Core code generation from TTGIR/NVGPUIR to CUDA C++ source.
///
/// This translates MLIR operations to CUDA C++ code by walking the IR and
/// emitting equivalent C++ for each operation. Unlike the Python regex-based
/// emitter, this directly accesses MLIR's typed op/attribute/type API.

#include "CUDACodeGen.h"
#include <cmath>
#include <set>
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

namespace mlir::triton {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// ═══════════════════════════════════════════════════════════════════════
// Kernel-name sanitization
// ═══════════════════════════════════════════════════════════════════════
//
// We emit kernels as `extern "C" __global__ void <name>(...)`. If the Triton
// kernel name happens to equal a CUDA/cmath function declared with C linkage
// (e.g. a kernel literally named `exp2`, `sin`, `sqrt`), nvcc rejects the TU
// with "more than one instance of overloaded function ... has C linkage".
// Append a suffix so the entry symbol no longer collides. The launcher reads
// the kernel name back from the emitted PTX `.visible .entry`, so renaming
// here stays consistent end-to-end.
static std::string sanitizeKernelName(const std::string &name) {
  static const std::set<std::string> kReserved = {
      "abs",    "fabs",   "exp",     "exp2",   "exp10",  "expm1",  "log",
      "log2",   "log10",  "log1p",   "sin",    "cos",    "tan",    "sinpi",
      "cospi",  "asin",   "acos",    "atan",   "atan2",  "sinh",   "cosh",
      "tanh",   "asinh",  "acosh",   "atanh",  "sqrt",   "cbrt",   "rsqrt",
      "pow",    "hypot",  "ceil",    "floor",  "trunc",  "round",  "rint",
      "nearbyint", "fmod", "fmin",   "fmax",   "fdim",   "copysign", "erf",
      "erfc",   "lgamma", "tgamma",  "ldexp",  "frexp",  "modf",   "fma",
      "remainder", "remquo", "scalbn", "ilogb", "logb",  "j0", "j1", "y0",
      "y1",     "erfinv", "erfcinv", "normcdf", "isnan", "isinf", "isfinite"};
  if (kReserved.count(name))
    return name + "_triton_kernel";
  return name;
}

// ═══════════════════════════════════════════════════════════════════════
// Type mapping: MLIR types → CUDA C++ type strings
// ═══════════════════════════════════════════════════════════════════════

static std::string mlirTypeToCUDA(Type type) {
  if (type.isF32())
    return "float";
  if (type.isF16())
    return "__half";
  if (type.isBF16())
    return "__nv_bfloat16";
  if (type.isF64())
    return "double";
  if (isa<Float8E4M3FNType>(type) || isa<Float8E5M2Type>(type))
    return "__nv_fp8_storage_t";
  if (type.isInteger(1))
    return "bool";
  if (type.isInteger(8))
    return "int8_t";
  if (type.isInteger(16))
    return "int16_t";
  if (type.isInteger(32))
    return "int";
  if (type.isInteger(64))
    return "int64_t";
  if (auto ptrType = dyn_cast<tt::PointerType>(type))
    return mlirTypeToCUDA(ptrType.getPointeeType()) + "*";
  if (isa<ttg::MemDescType>(type)) {
    // Memory descriptor = pointer to shared memory
    auto memDesc = cast<ttg::MemDescType>(type);
    return mlirTypeToCUDA(memDesc.getElementType()) + "*";
  }
  if (isa<tt::TensorDescType>(type)) {
    return "const __grid_constant__ CUtensorMap";
  }
  return "int"; // fallback
}

static int getTypeSizeInBytes(Type type) {
  if (type.isF64() || type.isInteger(64))
    return 8;
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isF16() || type.isBF16() || type.isInteger(16))
    return 2;
  if (type.isInteger(8) || type.isInteger(1))
    return 1;
  // fp8 types (f8E5M2, f8E4M3FN, etc.)
  if (isa<Float8E4M3FNType>(type) || isa<Float8E5M2Type>(type))
    return 1;
  return 4;
}

static std::string getUnsignedType(Type type) {
  if (type.isInteger(8))
    return "uint8_t";
  if (type.isInteger(16))
    return "uint16_t";
  if (type.isInteger(32))
    return "unsigned";
  if (type.isInteger(64))
    return "uint64_t";
  return "unsigned";
}

// ═══════════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════════

CUDACodeGen::CUDACodeGen(ModuleOp module, int32_t capability, int32_t numWarps,
                         int32_t numCtas, int32_t ptxVersion)
    : module(module), capability(capability), numWarps(numWarps),
      numCtas(numCtas), ptxVersion(ptxVersion) {
  // Mirror knobs.compilation.disable_line_info (TRITON_DISABLE_LINE_INFO).
  const char *e = getenv("TRITON_DISABLE_LINE_INFO");
  std::string v = e ? e : "";
  lineInfoEnabled = !(v == "1" || v == "true" || v == "on" || v == "ON");
  // Elementwise loop fusion on by default; TRITON_CUDA_NO_FUSE=1 disables it
  // (for bisecting a regression).
  const char *nf = getenv("TRITON_CUDA_NO_FUSE");
  if (nf && std::string(nf) == "1")
    fuseElementwise = false;
}

// ═══════════════════════════════════════════════════════════════════════
// Emit infrastructure
// ═══════════════════════════════════════════════════════════════════════

void CUDACodeGen::emit(const llvm::Twine &line) {
  // A #line directive sets the number nvcc assumes for the next physical line
  // and auto-increments after that, so re-emit it whenever the assumed line
  // diverges from the current op's source location.
  if (lineInfoEnabled && curLocLine > 0 &&
      (effectiveLine != curLocLine || effectiveFile != curLocFile)) {
    std::string escaped;
    for (char c : curLocFile) {
      if (c == '\\' || c == '"')
        escaped += '\\';
      escaped += c;
    }
    lines.push_back("#line " + std::to_string(curLocLine) + " \"" + escaped +
                    "\"");
    effectiveLine = curLocLine;
    effectiveFile = curLocFile;
  }
  std::string s;
  for (int i = 0; i < indentLevel; i++)
    s += "    ";
  s += line.str();
  lines.push_back(std::move(s));
  if (effectiveLine >= 0)
    effectiveLine++;
}

void CUDACodeGen::blockSync() {
  if (wsSyncBarrierId > 0)
    emit("asm volatile(\"bar.sync " + std::to_string(wsSyncBarrierId) + ", " +
         std::to_string(wsSyncThreadCount) + ";\");");
  else
    emit(llvm::Twine("__syncthreads();"));
}

void CUDACodeGen::emitBlank() { lines.push_back(""); }

void CUDACodeGen::indent() { indentLevel++; }

void CUDACodeGen::dedent() {
  if (indentLevel > 0)
    indentLevel--;
}

std::string CUDACodeGen::newVar(llvm::StringRef prefix) {
  return (prefix + llvm::Twine(varCounter++)).str();
}

// ═══════════════════════════════════════════════════════════════════════
// Value tracking
// ═══════════════════════════════════════════════════════════════════════

std::string CUDACodeGen::getVar(Value val) {
  // A consumer that needs the array form of a deferred elementwise value
  // forces it to materialize at the current emission point (which dominates
  // this single same-block use).
  if (deferredElem.count(val))
    return materializeDeferred(val);
  auto it = valueToVar.find(val);
  if (it != valueToVar.end())
    return it->second;
  // Auto-create for block arguments
  auto var = newVar("arg");
  valueToVar[val] = var;
  return var;
}

std::string CUDACodeGen::getElemExpr(Value val, const std::string &idx) {
  if (scalarValues.contains(val))
    return getVar(val); // scalar — no indexing needed
  // Deferred elementwise: inline the per-element RHS expression here instead
  // of materializing an array (this is the fusion).
  auto dit = deferredElem.find(val);
  if (dit != deferredElem.end())
    return "(" + dit->second.build(idx) + ")";
  auto var = getVar(val);
  // Check deferred addptr — return base + offset[idx] expression
  auto it = deferredAddPtr.find(val);
  if (it != deferredAddPtr.end())
    return "(" + it->second.first + " + " + it->second.second + "[" + idx + "])";
  return var + "[" + idx + "]";
}

bool CUDACodeGen::shouldDeferElementwise(Value result, Operation *op) {
  if (!fuseElementwise)
    return false;
  if (!isa<RankedTensorType>(result.getType()))
    return false;
  if (scalarValues.contains(result))
    return false;
  // Single use only: avoids recomputation/expression blow-up and keeps the
  // inlined operands trivially in scope.
  if (!result.hasOneUse())
    return false;
  // The single user must be in the same block as the defining op. This both
  // keeps the inlined expression's operands in C++ scope and prevents hoisting
  // a pure computation into a loop body where it would re-execute K times.
  Operation *user = *result.user_begin();
  if (user->getBlock() != op->getBlock())
    return false;
  return true;
}

std::string CUDACodeGen::materializeDeferred(Value val) {
  auto it = deferredElem.find(val);
  DeferredElemDef def = it->second;
  // Erase before building so a (degenerate) self-reference can't recurse, and
  // so the build() expansion of operands resolves normally.
  deferredElem.erase(it);
  // Compute the body expression FIRST: build() may force a deferred operand to
  // materialize (emitting its own array + loop). Those lines must land BEFORE
  // this value's array/loop, otherwise they'd be injected into our loop body.
  std::string body = def.build("_i");
  auto var = newVar("a");
  valueToVar[val] = var;
  emit(def.cudaType + " " + var + "[" + std::to_string(def.nElems) + "];");
  emit("#pragma unroll");
  emit("for (int _i = 0; _i < " + std::to_string(def.nElems) + "; _i++)");
  emit("    " + var + "[_i] = " + body + ";");
  return var;
}

void CUDACodeGen::emitOrDeferElementwise(
    Value result, Operation *op, int nElems, const std::string &cudaType,
    std::function<std::string(const std::string &)> build) {
  if (shouldDeferElementwise(result, op)) {
    deferredElem[result] = {std::move(build), nElems, cudaType};
    return;
  }
  // Compute the body FIRST so any deferred operand materializes before this
  // value's own array/loop is opened (see materializeDeferred).
  std::string body = build("_i");
  // In-place reuse (opt-in via TRITON_EMIT_INPLACE_ELTWISE=1): if an operand is
  // a materialized register array that is DEAD after this op (its only SSA use
  // is here) and has EXACTLY the same tensor type (→ identical array length +
  // element type), write the result INTO that operand's array instead of
  // declaring a fresh one. This mirrors omni FA3-bwd's hand-written in-place
  // updates (s_acc→P via exp2, dp_acc→dS via P*(dP-D)) that hold the consumer
  // live-set at 168/0; the emitter's default one-array-per-SSA-value scheme
  // spills because ptxas cannot coalesce fp32 transient arrays whose live
  // ranges cross WGMMA async-wait boundaries. Element-wise safety: every output
  // slot _i depends only on input slot _i of the SAME array, so the in-place
  // overwrite is exact. hasOneUse() guarantees the array is referenced nowhere
  // else (loop-carried/multi-use operands keep a use in the yield → skipped).
  static const bool inplaceEltwise =
      (getenv("TRITON_EMIT_INPLACE_ELTWISE") != nullptr) &&
      (std::string(getenv("TRITON_EMIT_INPLACE_ELTWISE")) == "1");
  std::string var;
  if (inplaceEltwise) {
    // The reusable dead array is often NOT a direct operand: elementwise fusion
    // inlines single-use producers (e.g. P=exp2(S-lse) emits one loop reading
    // the materialized S array inside a deferred subtract). So walk operands AND
    // recurse THROUGH deferred (fused-away) operands to reach the materialized
    // arrays actually referenced in this loop body. A materialized array that is
    // single-use (its sole SSA use is consumed within this fused expression) is
    // dead after the loop → safe to overwrite element-for-element.
    llvm::SmallPtrSet<Value, 8> visited;
    std::function<bool(Value)> findReuse = [&](Value v) -> bool {
      if (!visited.insert(v).second)
        return false;
      // Deferred (fused) operand: descend into its producer's operands.
      if (deferredElem.count(v)) {
        if (Operation *d = v.getDefiningOp())
          for (Value o : d->getOperands())
            if (findReuse(o))
              return true;
        return false;
      }
      if (!isa<RankedTensorType>(v.getType()))
        return false;
      if (v.getType() != result.getType())
        return false; // different array length or element type
      if (v.getDefiningOp() == nullptr)
        return false; // block arg / loop-carried — never reuse
      if (!v.hasOneUse())
        return false; // still live after this op
      if (!valueToVar.count(v))
        return false; // not a materialized register array
      if (scalarValues.count(v) || deferredAddPtr.count(v) ||
          ptrBasedDeferred.count(v) || scalarBaseDeferred.count(v))
        return false; // not a plain dense register array
      var = valueToVar[v]; // dead same-type array → destination-pass into it
      return true;
    };
    for (Value o : op->getOperands())
      if (findReuse(o))
        break;
  }
  if (var.empty()) {
    var = newVar("a");
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
  }
  valueToVar[result] = var;
  emit("#pragma unroll");
  emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
  emit("    " + var + "[_i] = " + body + ";");
}

std::string CUDACodeGen::getPtrElemExpr(Value val, const std::string &idx) {
  // For deferred addptr, dereference: *(base + offset[idx])
  auto it = deferredAddPtr.find(val);
  if (it != deferredAddPtr.end())
    return "*(" + it->second.first + " + " + it->second.second + "[" + idx + "])";
  // For pointer-based deferred: cast from const char* and dereference
  auto ptrIt = ptrBasedDeferred.find(val);
  if (ptrIt != ptrBasedDeferred.end())
    return "*((" + ptrIt->second.second + "*)" + getVar(val) + "[" + idx + "])";
  // For scalar-base deferred: precomputed 64-bit address + delta
  auto sbiIt = scalarBaseDeferred.find(val);
  if (sbiIt != scalarBaseDeferred.end()) {
    auto &sbi = sbiIt->second;
    std::string cty = sbi.cType.empty() ? "char" : sbi.cType;
    // delta is a signed byte offset (loop pointer updates such as `p -= BLOCK`
    // produce negatives); sign-extend via (long long) so the modular add to the
    // 64-bit base address moves backward. A 32-bit (unsigned) cast would wrap a
    // negative delta to ~4 GiB forward → out-of-bounds.
    return "*(" + cty + "*)(" + sbi.precompAddr + "[" + idx +
           "] + (long long)" + sbi.deltaVar + ")";
  }
  if (scalarValues.contains(val))
    return "*" + getVar(val);
  return "*" + getVar(val) + "[" + idx + "]";
}

std::string CUDACodeGen::getPtrAddrExpr(Value val, const std::string &idx) {
  // For deferred addptr, return the address: (base + offset[idx])
  auto it = deferredAddPtr.find(val);
  if (it != deferredAddPtr.end())
    return "(" + it->second.first + " + " + it->second.second + "[" + idx + "])";
  // Pointer-based deferred: cast from const char* base
  auto ptrIt = ptrBasedDeferred.find(val);
  if (ptrIt != ptrBasedDeferred.end())
    return "((" + ptrIt->second.second + "*)" + getVar(val) + "[" + idx + "])";
  // Scalar-base deferred: precomputed 64-bit address + delta
  auto sbiIt = scalarBaseDeferred.find(val);
  if (sbiIt != scalarBaseDeferred.end()) {
    auto &sbi = sbiIt->second;
    std::string cty = sbi.cType.empty() ? "char" : sbi.cType;
    // See getPtrElemExpr: sign-extend the (possibly negative) delta.
    return "(" + cty + "*)(" + sbi.precompAddr + "[" + idx + "] + (long long)" +
           sbi.deltaVar + ")";
  }
  if (scalarValues.contains(val))
    return getVar(val);
  return getVar(val) + "[" + idx + "]";
}

unsigned CUDACodeGen::getPtrContiguity(Value ptr) {
  // Max safe vector width (in elements) for a tensor-of-pointers, from AxisInfo:
  // min(per-thread contiguity, pointer alignment from divisibility). Mirrors the
  // PTX backend's getVectorSize so we never emit an over-wide vector access that
  // could be misaligned at runtime.
  if (!isa<RankedTensorType>(ptr.getType()))
    return 1;
  if (!axisInfoAnalysis)
    axisInfoAnalysis = std::make_unique<ModuleAxisInfoAnalysis>(module);
  unsigned c = axisInfoAnalysis->getContiguity(ptr);
  return c == 0 ? 1 : c;
}

unsigned CUDACodeGen::getMaxVecWidth(Value ptr, Value mask) {
  unsigned w = getPtrContiguity(ptr);
  if (mask && isa<RankedTensorType>(mask.getType())) {
    if (!axisInfoAnalysis)
      axisInfoAnalysis = std::make_unique<ModuleAxisInfoAnalysis>(module);
    unsigned m = axisInfoAnalysis->getMaskAlignment(mask);
    if (m == 0)
      m = 1;
    w = std::min(w, m);
  }
  return w == 0 ? 1 : w;
}

std::string CUDACodeGen::getCUDAType(Type type) {
  return mlirTypeToCUDA(type);
}

std::string CUDACodeGen::getElemTypeStr(Value val) {
  Type type = val.getType();
  if (auto rtt = dyn_cast<RankedTensorType>(type))
    return mlirTypeToCUDA(rtt.getElementType());
  if (auto memDescType = dyn_cast<ttg::MemDescType>(type))
    return mlirTypeToCUDA(memDescType.getElementType());
  if (auto ptrType = dyn_cast<tt::PointerType>(type))
    return mlirTypeToCUDA(ptrType.getPointeeType());
  return mlirTypeToCUDA(type);
}

// Address expression for a tensor-descriptor value, usable as a `const char*`
// pointing at the 128-byte CUtensorMap. Device-built descriptors (in
// pointerDescriptors) already hold a pointer; host grid_constant CUtensorMap
// params need their address taken.
std::string CUDACodeGen::descAddrExpr(Value val) {
  auto var = getVar(val);
  // Pointer descriptors may be typed char*/int8_t*/etc — normalize the type.
  return pointerDescriptors.contains(val) ? ("(const char*)" + var)
                                          : ("(const char*)&" + var);
}

bool CUDACodeGen::isTensorValue(Value val) {
  return isa<RankedTensorType>(val.getType());
}

llvm::ArrayRef<int64_t> CUDACodeGen::getTensorShape(Value val) {
  if (auto rtt = dyn_cast<RankedTensorType>(val.getType()))
    return rtt.getShape();
  return {};
}

// ─── Backend-private split-layout for native non-pow2 WGMMA N (omni-style) ──
// Triton's LinearLayout is GF(2)-based: a non-pow2 out-dim (e.g. N=80) cannot
// be represented, so toLinearLayout aborts in strided1D. But wgmma.m64nNk16
// accepts ANY N that is a multiple of 8. We model an n80 mma tensor as the
// concatenation of two REAL pow2 sub-fragments (n64 (+) n16): the f32 D-fragment
// registers are written n-block-contiguous, so a thread's regs[0..31] follow the
// [M,64] mma layout and regs[32..39] follow the [M,16] mma layout (cols +64).
// This helper returns the pow2 sub-RankedTensorTypes (with the encoding's
// instrShape-N cloned to match each chunk) so all existing LinearLayout
// machinery applies to each legal piece. Returns {} for pow2 / non-mma tensors.
static llvm::SmallVector<mlir::RankedTensorType>
splitNonPow2MmaTensor(mlir::RankedTensorType ty) {
  using namespace mlir;
  auto shape = ty.getShape();
  int npDim = -1;
  for (unsigned d = 0; d < shape.size(); d++)
    if (!llvm::isPowerOf2_64(shape[d])) { npDim = (int)d; break; }

  Attribute enc = ty.getEncoding();
  ttg::NvidiaMmaEncodingAttr mma;
  auto slice = dyn_cast<ttg::SliceEncodingAttr>(enc);
  auto dop = dyn_cast<ttg::DotOperandEncodingAttr>(enc);
  if (slice)
    mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(slice.getParent());
  else if (dop)
    mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(dop.getParent());
  else
    mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(enc);
  if (!mma)
    return {};

  auto instrShape = mma.getInstrShape();
  bool mmaNNonPow2 =
      instrShape.size() >= 2 && !llvm::isPowerOf2_64(instrShape[1]);
  if (npDim < 0 && !mmaNNonPow2)
    return {}; // fully representable — nothing to split

  // For a DotOperand, the non-pow2 axis we split is the CONTRACTION (K) dim of
  // the operand tensor itself (e.g. the [M=64, K=80] A operand of dV=P^T·dO),
  // NOT the parent mma's N.  The parent mma (its instrShape[1]=N) must stay
  // intact — only the operand's K shape splits into pow2 chunks (64⊕16).  So a
  // dop is cloned with its parent UNCHANGED; the caller resizes subShape[npDim].
  // For an accumulator (mma) / slice, the non-pow2 axis IS wgmma-N, so we clone
  // the parent with instrShape[1] replaced by the chunk size `n`.
  auto cloneEncWithN = [&](int64_t n) -> Attribute {
    if (dop)
      return (Attribute)dop; // keep parent mma + opIdx + kWidth; split shape only
    SmallVector<unsigned> instr(instrShape.begin(), instrShape.end());
    if (instr.size() >= 2)
      instr[1] = (unsigned)n;
    auto subMma = ttg::NvidiaMmaEncodingAttr::get(
        mma.getContext(), mma.getVersionMajor(), mma.getVersionMinor(),
        mma.getWarpsPerCTA(), mma.getCGALayout(), instr);
    if (slice)
      return ttg::SliceEncodingAttr::get(ty.getContext(), slice.getDim(),
                                         subMma);
    return (Attribute)subMma;
  };

  if (npDim < 0) {
    // The non-pow2 N is squeezed out of this tensor's shape (e.g. a [64]
    // row-slice of an N=80 mma): the per-thread distribution is N-independent,
    // so a single representative piece with N rounded down to a pow2 suffices.
    int64_t pn = 1;
    while (pn * 2 <= instrShape[1])
      pn *= 2;
    return {RankedTensorType::get(shape, ty.getElementType(), cloneEncWithN(pn))};
  }

  // Descending pow2 decomposition of the non-pow2 size (80 -> 64, 16).
  llvm::SmallVector<int64_t> chunks;
  for (int64_t rem = shape[npDim]; rem > 0;) {
    int64_t c = 1;
    while (c * 2 <= rem)
      c *= 2;
    chunks.push_back(c);
    rem -= c;
  }

  llvm::SmallVector<RankedTensorType> pieces;
  for (int64_t c : chunks) {
    Attribute subEnc = cloneEncWithN(c);
    SmallVector<int64_t> subShape(shape.begin(), shape.end());
    subShape[npDim] = c;
    pieces.push_back(
        RankedTensorType::get(subShape, ty.getElementType(), subEnc));
  }
  return pieces;
}

CUDACodeGen::RegCoordTable
CUDACodeGen::getRegCoordTable(RankedTensorType ty) {
  RegCoordTable t;
  auto ctx = ty.getContext();
  auto kReg = mlir::StringAttr::get(ctx, "register");
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  auto kWarp = mlir::StringAttr::get(ctx, "warp");
  auto kBlock = mlir::StringAttr::get(ctx, "block");

  auto fillBases = [&](const triton::LinearLayout &ll) {
    const auto &bases = ll.getBases();
    auto grab = [&](mlir::StringAttr k,
                    llvm::SmallVector<llvm::SmallVector<int>> &out) {
      auto it = bases.find(k);
      if (it == bases.end())
        return;
      for (auto &b : it->second) {
        llvm::SmallVector<int> v;
        for (auto x : b)
          v.push_back((int)x);
        out.push_back(v);
      }
    };
    grab(kLane, t.laneBases);
    grab(kWarp, t.warpBases);
    grab(kBlock, t.blockBases);
  };

  auto pieces = splitNonPow2MmaTensor(ty);
  if (pieces.empty()) {
    auto ll = ttg::toLinearLayout(ty);
    int n = getElemsPerThread(ty);
    for (int i = 0; i < n; i++) {
      auto coords = ll.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      llvm::SmallVector<int> v;
      for (auto &c : coords)
        v.push_back((int)c.second);
      t.regCoords.push_back(v);
    }
    fillBases(ll);
    return t;
  }

  // Native non-pow2 WGMMA N: concatenate the pow2 split-fragments. Each
  // fragment's lane0 coords get the cumulative N-dim offset baked in; the
  // lane/warp/block deltas (N-independent) come from the first fragment.
  auto shape = ty.getShape();
  int npDim = -1;
  for (unsigned d = 0; d < shape.size(); d++)
    if (!llvm::isPowerOf2_64(shape[d])) {
      npDim = (int)d;
      break;
    }
  // Collect each pow2 N-piece's lane0 register coords separately (in
  // toLinearLayout order, with the cumulative N offset baked in).
  std::vector<std::vector<llvm::SmallVector<int>>> pieceCoords;
  int off = 0;
  bool first = true;
  for (auto p : pieces) {
    auto llp = ttg::toLinearLayout(p);
    int np = getElemsPerThread(p);
    std::vector<llvm::SmallVector<int>> cur;
    for (int i = 0; i < np; i++) {
      auto coords = llp.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      llvm::SmallVector<int> v;
      for (auto &c : coords)
        v.push_back((int)c.second);
      if (npDim >= 0 && npDim < (int)v.size())
        v[npDim] += off;
      cur.push_back(v);
    }
    pieceCoords.push_back(std::move(cur));
    if (first) {
      fillBases(llp);
      first = false;
    }
    // When the non-pow2 N is squeezed out of this tensor's shape (npDim < 0),
    // splitNonPow2MmaTensor returns a SINGLE representative piece, so there is
    // no cumulative N offset to accumulate (and p.getShape()[npDim] would be an
    // out-of-bounds read). Only advance the offset for a real split dim.
    if (npDim >= 0)
      off += (int)p.getShape()[npDim];
  }

  // emitWarpGroupDot writes the accumulator registers m64-TILE-MAJOR: for each
  // m64 row-tile it emits one native wgmma whose n_out_regs outputs land
  // contiguously, ordered [N-piece0 .. N-pieceK]. So for M>64 (multiple m64
  // tiles) the producer layout is [m0: piece0,piece1][m1: piece0,piece1], NOT
  // the piece-major [piece0: m0,m1][piece1: m0,m1] that a naive piece-by-piece
  // concatenation would give. Regroup by m64 tile (the non-np / "M" dim of a 2D
  // acc, split at wgmma_m=64) so the table matches what the wgmma actually
  // produces. For M<=64 (single tile) this is a no-op. npDim<0 (N squeezed out)
  // has a single piece and is unaffected.
  if (npDim >= 0 && shape.size() == 2) {
    int mDim = (npDim == 0) ? 1 : 0;
    const int kWgmmaM = 64;
    std::set<int> mtiles;
    for (auto &pc : pieceCoords)
      for (auto &v : pc)
        mtiles.insert(v[mDim] / kWgmmaM);
    for (int mt : mtiles)
      for (auto &pc : pieceCoords)
        for (auto &v : pc)
          if (v[mDim] / kWgmmaM == mt)
            t.regCoords.push_back(v);
  } else {
    for (auto &pc : pieceCoords)
      for (auto &v : pc)
        t.regCoords.push_back(v);
  }
  return t;
}

int CUDACodeGen::getElemsPerThread(Value val) {
  if (auto rtt = dyn_cast<RankedTensorType>(val.getType()))
    return getElemsPerThread(rtt);
  return 1;
}

int CUDACodeGen::getElemsPerThread(RankedTensorType ty) {
  auto encoding = ty.getEncoding();
  auto shape = ty.getShape();

  // Native non-pow2 WGMMA N (omni-style n80): sum the per-thread counts of the
  // pow2 split-layout sub-fragments instead of calling toLinearLayout on the
  // GF(2)-non-representable shape.
  {
    auto pieces = splitNonPow2MmaTensor(ty);
    if (!pieces.empty()) {
      int total = 0;
      for (auto p : pieces)
        total += getElemsPerThread(p);
      return std::max(total, 1);
    }
  }

  // For 3D+ tensors or double-slice encodings, use LinearLayout for correct element count
  bool needsLL = false;
  if (shape.size() >= 3) needsLL = true;
  if (auto slice = dyn_cast<ttg::SliceEncodingAttr>(encoding)) {
    if (isa<ttg::SliceEncodingAttr>(slice.getParent())) needsLL = true; // double-slice from 3D
    // Single slice with 3D parent
    if (auto parentBlocked = dyn_cast<ttg::BlockedEncodingAttr>(slice.getParent())) {
      if (parentBlocked.getSizePerThread().size() >= 3) needsLL = true;
    }
  }
  if (needsLL && (isa<ttg::BlockedEncodingAttr>(encoding) || isa<ttg::SliceEncodingAttr>(encoding))) {
    auto ll = ttg::toLinearLayout(ty);
    auto kReg = mlir::StringAttr::get(ty.getContext(), "register");
    return std::max(ll.getInDimSize(kReg), 1);
  }

  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
    // Register counts are PER-CTA (PTX-backend convention): with CTASplitNum>1
    // each CTA only materializes its slice, so divide split dims first.
    // Mixing full-shape counts here with the per-CTA LL-based counts used
    // elsewhere double-enumerates the block bit as broadcast registers.
    auto shapePerCTA = ttg::getShapePerCTA(encoding, shape);
    int result = 1;
    for (unsigned d = 0; d < shape.size(); d++) {
      int spt = blocked.getSizePerThread()[d];
      int tpw = blocked.getThreadsPerWarp()[d];
      int wpc = blocked.getWarpsPerCTA()[d];
      int threadsInDim = tpw * wpc;
      if (shapePerCTA[d] == 1) {
        result *= 1;
      } else {
        int elems =
            std::max<int64_t>(1, shapePerCTA[d] / (threadsInDim * spt)) * spt;
        result *= elems;
      }
    }
    return std::max(result, 1);
  }

  if (auto slice = dyn_cast<ttg::SliceEncodingAttr>(encoding)) {
    // For double-slice (parent is also slice), use LL
    if (isa<ttg::SliceEncodingAttr>(slice.getParent())) {
      auto ll = ttg::toLinearLayout(ty);
      auto kReg = mlir::StringAttr::get(ty.getContext(), "register");
      return std::max(ll.getInDimSize(kReg), 1);
    }
    if (auto parent = dyn_cast<ttg::BlockedEncodingAttr>(slice.getParent())) {
      llvm::SmallVector<int64_t> parentShape(shape.begin(), shape.end());
      parentShape.insert(parentShape.begin() + slice.getDim(), 1);
      auto parentTy = RankedTensorType::get(parentShape, ty.getElementType(),
                                            parent);
      return std::max(getElemsPerThread(parentTy), 1);
    }
    if (auto mma =
            dyn_cast<ttg::NvidiaMmaEncodingAttr>(slice.getParent())) {
      // A thread owns 2 rows per M instr-tile; when the M dimension spans
      // multiple instr-tiles per warp (e.g. BLOCK_M=128 with warpsPerCTA[0]=4
      // and instrShape M=16 → 2 m-tiles → 4 rows/thread) the count is >2.
      // Use the LinearLayout register count for the exact value.
      auto ll = ttg::toLinearLayout(ty);
      auto kReg = mlir::StringAttr::get(ty.getContext(), "register");
      return std::max(ll.getInDimSize(kReg), 1);
    }
    // Any other parent (notably LinearEncoding produced by tl.sort / tl.topk):
    // derive the register count from the slice's own LinearLayout. Falling
    // through to the total/(numWarps*32) default below undercounts whenever the
    // sliced dim is lane- or warp-distributed — removing such a dim leaves the
    // per-thread register (row) count unchanged, but the naive total/threads
    // division collapses it (e.g. slice<dim1,#linear1> over 4 warps gave
    // 32/128 -> 1 instead of 8, making softmax/reduce results broadcast a single
    // row's value across all rows).
    {
      auto ll = ttg::toLinearLayout(ty);
      auto kReg = mlir::StringAttr::get(ty.getContext(), "register");
      return std::max(ll.getInDimSize(kReg), 1);
    }
  }

  if (auto mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(encoding)) {
    // Use LinearLayout for correct per-thread element count across all warp
    // group configurations (warpsPerCTA=[4,1] vs [8,1] etc.)
    auto ll = ttg::toLinearLayout(ty);
    auto kReg = mlir::StringAttr::get(ty.getContext(), "register");
    return std::max(ll.getInDimSize(kReg), 1);
  }

  // DotOperandEncoding: use LinearLayout for correct element count
  if (isa<ttg::DotOperandEncodingAttr>(encoding)) {
    auto ll = ttg::toLinearLayout(ty);
    auto kReg = mlir::StringAttr::get(ty.getContext(), "register");
    return std::max(ll.getInDimSize(kReg), 1);
  }

  // LinearEncoding: use the layout's full register in-dim size. This includes
  // degenerate (all-zero / broadcast) register bases, so the per-thread array
  // size is consistent with ll.apply({register, i}) raw indexing used by
  // make_range / reshape / broadcast emitters. Compacting away broadcast slots
  // would desync the array index from the layout's register index.
  if (isa<ttg::LinearEncodingAttr>(encoding)) {
    auto ll = ttg::toLinearLayout(ty);
    auto kReg = mlir::StringAttr::get(ty.getContext(), "register");
    return std::max(ll.getInDimSize(kReg), 1);
  }

  // Default: total / num_threads
  int64_t total = 1;
  for (auto s : shape)
    total *= s;
  return std::max(total / (numWarps * 32), (int64_t)1);
}

// ═══════════════════════════════════════════════════════════════════════
// Top-level generation
// ═══════════════════════════════════════════════════════════════════════

CUDATranslationResult CUDACodeGen::generate() {
  // Emit headers
  emit("#include <cuda.h>");
  emit("#include <cuda_fp16.h>");
  emit("#include <cuda_bf16.h>");
  emit("#include <cuda_fp8.h>");
  emit("#include <stdint.h>");
  emit("#include <float.h>");
  emit("#include <math.h>");
  emit("#include <assert.h>");
  emit("#include <cstdio>");
  emitBlank();
  emit("#ifndef __CUDA_ARCH__");
  emit("#define __CUDA_ARCH__ 90");
  emit("#endif");
  emitBlank();

  // Emit helper functions
  emit("__device__ __forceinline__ float warp_reduce_sum(float val) {");
  indent();
  emit("#pragma unroll");
  emit("for (int offset = 16; offset > 0; offset /= 2)");
  emit("    val += __shfl_xor_sync(0xffffffff, val, offset);");
  emit("return val;");
  dedent();
  emit("}");
  emitBlank();
  emit("__device__ __forceinline__ float warp_reduce_max(float val) {");
  indent();
  emit("#pragma unroll");
  emit("for (int offset = 16; offset > 0; offset /= 2)");
  emit("    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));");
  emit("return val;");
  dedent();
  emit("}");
  emitBlank();
  // f32 exp2: emit `ex2.approx.ftz.f32` directly, matching the PTX backend's
  // exp2 lowering. The library `exp2f` lowers to non-FTZ `ex2.approx.f32`,
  // which ptxas (under --ftz=false, required to keep general FP32 non-FTZ)
  // wraps in a denormal-range guard (FSETP x,-126 / FMUL x,0.5 / square)
  // around every EX2 — ~3x the FMUL and a flood of FSETP in softmax-style
  // inner loops. The FTZ instruction needs no guard; flushing exp2 denormals
  // (results < 2^-126) to zero is harmless for softmax and bit-matches Triton.
  emit("__device__ __forceinline__ float __triton_ex2f(float x) {");
  indent();
  emit("float r;");
  emit("asm(\"ex2.approx.ftz.f32 %0, %1;\" : \"=f\"(r) : \"f\"(x));");
  emit("return r;");
  dedent();
  emit("}");
  emitBlank();
  // RTZ float->e5m2: ptxas has no cvt.rz for fp8, so match the PTX
  // backend: f32 -> f16 with rz, then truncate to the high byte (e5m2
  // shares f16's exponent layout; dropping the low mantissa byte is rz).
  emit("__device__ __forceinline__ unsigned char cvt_f32_to_e5m2_rz(float x) {");
  indent();
  emit("__half h = __float2half_rz(x);");
  emit("return (unsigned char)(*(unsigned short*)&h >> 8);");
  dedent();
  emit("}");
  emitBlank();

  // Walk functions in the module
  module.walk([&](tt::FuncOp func) {
    // Skip non-kernel (private) functions: they are `noinline` callees whose
    // bodies are inlined at each tt.call site by emitCall.
    if (mlir::SymbolTable::getSymbolVisibility(func) ==
        mlir::SymbolTable::Visibility::Private)
      return;
    kernelName = sanitizeKernelName(func.getName().str());

    // Warp-specialization widening: a ttg.warp_specialize op spawns additional
    // warpgroups (the consumer "partitions") that run concurrently with the base
    // (producer/default) warps. The block must therefore launch base + Σpartition
    // warps. Multiple sequential warp_specialize ops reuse the same partition
    // warps, so take the max partition-warp count across ops, not the sum.
    int extraWarps = 0;
    func.walk([&](ttg::WarpSpecializeOp ws) {
      int s = 0;
      for (int32_t w : ws.getPartitionNumWarps())
        s += w;
      extraWarps = std::max(extraWarps, s);
    });
    totalNumWarps = numWarps + extraWarps;
    int numThreads = totalNumWarps * 32;

    // Pre-scan for device-side global scratch allocations (TMA descriptors).
    // Assign each alloc a byte offset within the per-CTA scratch region using
    // the same packing as the reference TritonGPUGlobalScratchAllocation pass.
    int scratchOff = 0;
    int scratchAlign = 1;
    bool funcHasTensormap = false;
    // Recurse into tt.call callees: their bodies are inlined at emit time
    // (emitCall), so their scratch allocs / tensormap creates live in THIS
    // kernel and must be counted here, or the kernel body references
    // global_scratch/_gscratch_cta without the parameter/definition.
    // NOTE: a GlobalScratchAllocOp in a callee called multiple times keeps
    // its first assigned offset (scratch reused across call sites).
    {
      llvm::SmallPtrSet<mlir::Operation *, 4> scanChain;
      std::function<void(mlir::Operation *)> scanScratch =
          [&](mlir::Operation *root) {
            root->walk([&](mlir::Operation *o) {
              if (auto a = dyn_cast<ttg::GlobalScratchAllocOp>(o)) {
                // Append (not overwrite): an alloc inside a callee is visited
                // once per call site and each inlined instance gets its own
                // slot, consumed in order by emitGlobalScratchAlloc.
                int al = (int)a.getAlignment();
                int nb = (int)a.getNbytes();
                int t = scratchOff + al - 1;
                scratchOff = t - (t % al); // roundUp(scratchOff, al)
                globalScratchOffsets[a.getOperation()].push_back(scratchOff);
                scratchOff += nb;
                scratchAlign = std::max(scratchAlign, al);
              } else if (isa<ttng::TensormapCreateOp>(o)) {
                funcHasTensormap = true;
              } else if (auto call = dyn_cast<tt::CallOp>(o)) {
                if (auto callee = module.lookupSymbol<tt::FuncOp>(
                        call.getCallee())) {
                  if (scanChain.insert(callee.getOperation()).second) {
                    scanScratch(callee.getOperation());
                    scanChain.erase(callee.getOperation());
                  }
                }
              }
            });
          };
      scanScratch(func.getOperation());
      globalScratchCursor.clear();
    }
    int totalScratch =
        scratchOff > 0 ? ((scratchOff + scratchAlign - 1) -
                          ((scratchOff + scratchAlign - 1) % scratchAlign))
                       : 0;
    bool funcHasScratch = totalScratch > 0;
    globalScratchSize = std::max<int64_t>(globalScratchSize, totalScratch);
    globalScratchAlign = std::max<int64_t>(globalScratchAlign, scratchAlign);

    // Build argument list
    std::string argList;
    for (unsigned i = 0; i < func.getNumArguments(); i++) {
      auto arg = func.getArgument(i);
      auto argType = arg.getType();
      std::string cudaType = getCUDAType(argType);
      auto var = newVar("v");
      valueToVar[arg] = var;
      if (i > 0)
        argList += ", ";
      argList += cudaType + " " + var;
    }
    // The launcher always passes a per-grid global scratch pointer right after
    // the user arguments (see driver.py). Declare it when the kernel uses it.
    if (funcHasScratch) {
      if (func.getNumArguments() > 0)
        argList += ", ";
      argList += "char* global_scratch";
    }

    // Per-kernel register cap (tl.<kernel>[grid](..., maxnreg=N)): lower the
    // ttg.maxnreg module attr to CUDA's __maxnreg__ qualifier, which nvcc
    // turns into the PTX `.maxnreg N` directive on the .entry (exactly what
    // the PTX backend emits via nvvm.maxnreg).
    // nvcc rejects __launch_bounds__ + __maxnreg__ on the same kernel, so the
    // explicit register cap replaces the launch-bounds-derived one.
    std::string boundsQual =
        " __launch_bounds__(" + std::to_string(numThreads) + ", 1)";
    if (auto mn = module->getAttrOfType<IntegerAttr>("ttg.maxnreg"))
      boundsQual = " __maxnreg__(" + std::to_string(mn.getInt()) + ")";
    emit("extern \"C\" __global__ void" + boundsQual);
    emit(kernelName + "(" + argList + ") {");
    indent();

    emit("// Thread indexing");
    emit("const int tid = threadIdx.x;");
    emit("const int warp_id = tid / 32;");
    emit("const int lane_id = tid % 32;");
    if (numCtas > 1) {
      // Linear CTA rank within the CGA cluster (x-fastest), matching the
      // LinearLayout "block" input dimension and the reference
      // TargetInfo::getClusterCTAId linearization.
      emit("unsigned _cta_rank;");
      emit("asm(\"mov.u32 %0, %%cluster_ctarank;\" : \"=r\"(_cta_rank));");
    }
    // Warp-group index for multi-warpgroup WGMMA — hoisted out of loops.
    // __shfl_sync makes this provably warp-uniform for ptxas to use UR.
    if (numWarps >= 4) {
      emit("const int _wg_m = __shfl_sync(0xFFFFFFFF, warp_id / 4, 0);");
    }
    emitBlank();
    // Align the dynamic shared base to the NVMMA swizzle period (1024 B). TMA
    // computes its shared swizzle phase from the ABSOLUTE shared address, so the
    // dynamic base must sit at phase 0. _tma_build is padded to 1024, but static
    // reduce buffers (_wb_red[]) are emitted AFTER it and would otherwise push
    // the dynamic base off the swizzle period (e.g. 2048+48=2096, not 128-aligned)
    // → UTMALDG misaligned-address crash. The attribute makes ptxas pad the whole
    // static region up to a 1024 boundary, restoring phase 0.
    emit("extern __shared__ __align__(1024) char shared_mem[];");
    if (funcHasScratch) {
      // Per-CTA base offset into the global scratch buffer (matches the
      // reference getGlobalScratchPtr: linearId * per_cta_size + alloc_off).
      emit("const long long _gscratch_cta = ((long long)blockIdx.x + "
           "(long long)gridDim.x * ((long long)blockIdx.y + "
           "(long long)gridDim.y * (long long)blockIdx.z)) * " +
           std::to_string(totalScratch) + "LL;");
    }
    if (funcHasTensormap) {
      // Shared scratch used to build CUtensorMap descriptors before copying
      // them to global memory (reused sequentially across descriptors).
      // The descriptor itself only needs 128 bytes, but this is a STATIC shared
      // allocation that the dynamic `extern shared_mem[]` is placed *after*.
      // NVMMA-swizzled buffers in shared_mem require their absolute byte address
      // to be aligned to the swizzle period (swizzleByteWidth * maxPhase =
      // 128 * 8 = 1024 bytes); otherwise the TMA hardware swizzle (computed from
      // the absolute address) disagrees with the register<->shared swizzle
      // (computed buffer-relative), corrupting TMA stores. Padding this static
      // block to a full swizzle period keeps the dynamic shared base at phase 0.
      // Under warp specialization the producer and each consumer warpgroup
      // build their TMA descriptors CONCURRENTLY. A single shared scratch would
      // be raced (interleaved tensormap.replace writes corrupt the descriptor →
      // garbage global_address → stores land in the wrong buffer, e.g. an input
      // tensor). Give every warpgroup its own slot, indexed by threadIdx.x/128.
      // A tensormap is 128 BYTES (PTX .b1024 = 1024 bits), so 128-byte slots
      // suffice; the extern shared_mem __align__(1024) below makes ptxas pad
      // the static region so the dynamic base stays phase-0 either way. Keeping
      // the slots small matters: WS kernels with per-partition epilogue buffers
      // (tut-09 descriptor_persistent_ws BM128/BK128/s3) sit ~1 KB under the
      // 232448 smem limit, and 1024-byte slots pushed them over.
      // Only under warp specialization does this need to be a STATIC shared
      // allocation (concurrent per-warpgroup slots, see above). For non-WS
      // kernels mirror PTX: the descriptor scratch is transient (built, copied
      // to global scratch via tensormap.cp_fenceproxy, then __syncthreads'd
      // before any other smem write), so emitTensormapCreate carves it out of
      // the DYNAMIC region at the current allocation floor instead. Static
      // smem stacks on top of the dynamic peak and pushed tight persistent
      // kernels (task #44) past the 232448 per-block limit.
      if (totalNumWarps > numWarps) {
        int numWgScratch = std::max(1, totalNumWarps / 4);
        emit("__shared__ __align__(128) char _tma_build[" +
             std::to_string(128 * numWgScratch) + "];");
      }
    }
    emitBlank();

    // Pre-scan for liveness-based reclaim of SOURCED (deallocless) smem buffers.
    // Assign a program-order index to every op, then for each `local_alloc %src`
    // record the order of its last user. emitLocalAlloc consults these to retire
    // dead sourced tiles and reclaim their shared memory (see header note).
    smemOpOrder.clear();
    sourcedLastUse.clear();
    sourcedBuffers.clear();
    liveSmemTop.clear();
    {
      int order = 0;
      // Assign indices in inlined program order: a tt.call expands its callee
      // body at the call site, so walk into callees there (a callee invoked
      // twice keeps the later index — conservative for last-use liveness).
      llvm::SmallPtrSet<Operation *, 4> orderChain; // guard vs recursive calls
      std::function<void(Operation *)> assignOrder = [&](Operation *o) {
        smemOpOrder[o] = order++;
        if (auto call = dyn_cast<tt::CallOp>(o))
          if (auto callee = module.lookupSymbol<tt::FuncOp>(call.getCallee()))
            if (callee != func &&
                orderChain.insert(callee.getOperation()).second) {
              callee.walk([&](Operation *inner) {
                if (inner != callee.getOperation())
                  assignOrder(inner);
              });
              orderChain.erase(callee.getOperation());
            }
      };
      func.walk([&](Operation *o) { assignOrder(o); });
      // Sourced allocs may live inside noinline callees too — scan the module.
      module.walk([&](ttg::LocalAllocOp a) {
        if (!a.getSrc())
          return; // only sourced (immutable, deallocless) buffers
        Value res = a.getResult();
        int last = smemOpOrder.lookup(a.getOperation());
        for (Operation *user : res.getUsers())
          last = std::max(last, smemOpOrder.lookup(user));
        sourcedBuffers.push_back(res);
        sourcedLastUse[res] = last;
      });
    }

    // Emit function body. A single block is the common case; multiple blocks
    // means the frontend emitted unstructured control flow (cf dialect, e.g.
    // an early `return`). For that we structurally re-emit via if-conversion.
    auto &body = func.getBody();
    if (llvm::hasSingleElement(body)) {
      emitBlock(body.front());
      emit("return;");
    } else {
      // Reject irreducible CFG: every non-entry block must have exactly one
      // predecessor so if-conversion (inlining) is sound and never duplicates a
      // shared block. Honest refusal otherwise.
      bool reducible = true;
      Block *entry = &body.front();
      for (Block &b : body) {
        if (&b == entry)
          continue;
        if (!b.getSinglePredecessor()) {
          reducible = false;
          break;
        }
      }
      if (!reducible) {
        emitFailed = true;
        emitErrorMsg = "CUDA emitter: unsupported irreducible control flow "
                       "(cf.cond_br with a shared-predecessor merge block). "
                       "Only tree-shaped CFG (e.g. early return) is supported.";
        emit("return;");
      } else {
        emitCFBlock(entry);
      }
    }
    dedent();
    emit("}");
  });

  // Build result
  CUDATranslationResult result;
  for (auto &line : lines) {
    result.cudaSource += line;
    result.cudaSource += "\n";
  }
  result.kernelName = kernelName;
  result.sharedMemSize =
      std::max({(int64_t)sharedMemOffset, (int64_t)peakSharedMem, (int64_t)0});

  // Also check module attribute for shared size
  if (auto attr = module->getAttrOfType<IntegerAttr>("ttg.shared")) {
    result.sharedMemSize =
        std::max(result.sharedMemSize, (int64_t)attr.getInt());
  }

  result.globalScratchSize = globalScratchSize;
  result.globalScratchAlign = globalScratchAlign;
  result.numWarps = totalNumWarps;
  result.ok = !emitFailed;
  result.errorMessage = emitErrorMsg;

  return result;
}

void CUDACodeGen::emitBlock(Block &block) {
  for (auto &op : block) {
    emitOp(&op);
  }
}

// Assign branch operands to the destination block's argument variables. For
// the early-return CFG this is a no-op (no block args), but handle the general
// case (scalar + tensor) so structured branches with arguments also work.
void CUDACodeGen::assignBlockArgs(Block *dest, mlir::ValueRange operands) {
  for (unsigned i = 0; i < dest->getNumArguments(); i++) {
    BlockArgument arg = dest->getArgument(i);
    Value src = operands[i];
    std::string srcVar = getVar(src);
    if (auto rtt = dyn_cast<RankedTensorType>(arg.getType())) {
      int nElems = getElemsPerThread(arg);
      auto cudaType = getCUDAType(rtt.getElementType());
      auto var = newVar("barg");
      valueToVar[arg] = var;
      emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      emit("    " + var + "[_i] = " + srcVar + "[_i];");
    } else {
      auto cudaType = getCUDAType(arg.getType());
      auto var = newVar("barg");
      valueToVar[arg] = var;
      emit("const " + cudaType + " " + var + " = " + srcVar + ";");
    }
  }
}

// Structurally emit a cf-dialect block by if-conversion: emit the block's body
// ops, then inline its successor(s). Caller guarantees each non-entry block has
// a single predecessor, so inlining never duplicates a block.
void CUDACodeGen::emitCFBlock(Block *block) {
  if (emitFailed)
    return;
  // Guard against back-edges (loops): if-conversion cannot express a cf loop.
  // Refuse honestly rather than recurse forever.
  if (!cfVisited.insert(block).second) {
    emitFailed = true;
    emitErrorMsg = "CUDA emitter: unsupported cf-dialect loop (back-edge); "
                   "only tree-shaped CFG (e.g. early return) is supported.";
    return;
  }
  // Emit all ops except the terminator.
  for (auto &op : block->without_terminator())
    emitOp(&op);

  Operation *term = block->getTerminator();
  if (isa<tt::ReturnOp>(term)) {
    emit("return;");
    return;
  }
  if (auto br = dyn_cast<mlir::cf::BranchOp>(term)) {
    assignBlockArgs(br.getDest(), br.getDestOperands());
    emitCFBlock(br.getDest());
    return;
  }
  if (auto cbr = dyn_cast<mlir::cf::CondBranchOp>(term)) {
    std::string cond = getVar(cbr.getCondition());
    emit("if (" + cond + ") {");
    indent();
    assignBlockArgs(cbr.getTrueDest(), cbr.getTrueDestOperands());
    emitCFBlock(cbr.getTrueDest());
    dedent();
    emit("} else {");
    indent();
    assignBlockArgs(cbr.getFalseDest(), cbr.getFalseDestOperands());
    emitCFBlock(cbr.getFalseDest());
    dedent();
    emit("}");
    return;
  }
  // Unknown terminator — fall back to dispatch (will hard-error if unsupported).
  emitOp(term);
}

// tt.call (noinline tt.func): emit-time inlining. `noinline` is a codegen
// strategy hint, not a semantic requirement — re-emitting the callee body at
// each call site is always correct. We alias the callee's entry-block args to
// the call operands (propagating every value-tracking map, since operands may
// be deferred pointers / scalars), walk the body through emitOp, then alias
// the call results to the tt.return operands.
void CUDACodeGen::emitCall(tt::CallOp op) {
  if (callInlineDepth >= 16) {
    emitFailed = true;
    emitErrorMsg = "CUDA emitter: tt.call inlining depth > 16 (recursive "
                   "noinline function?)";
    return;
  }
  auto callee = module.lookupSymbol<tt::FuncOp>(op.getCallee());
  if (!callee || callee.isExternal()) {
    emitFailed = true;
    emitErrorMsg = "CUDA emitter: tt.call to unknown/external function '" +
                   op.getCallee().str() + "'";
    return;
  }
  Region &body = callee.getBody();
  if (!llvm::hasSingleElement(body)) {
    emitFailed = true;
    emitErrorMsg = "CUDA emitter: tt.call callee '" + op.getCallee().str() +
                   "' has multi-block body (unstructured control flow) — "
                   "unsupported for inlining";
    return;
  }
  Block &entry = body.front();
  // Alias `to` to `from` across every value-tracking map.
  auto aliasTo = [&](Value to, Value from) {
    // A deferred elementwise value has no array yet; force it to materialize
    // at the call site so the alias resolves to a real variable.
    if (deferredElem.count(from))
      (void)materializeDeferred(from);
    auto it = valueToVar.find(from);
    if (it != valueToVar.end())
      valueToVar[to] = it->second;
    if (scalarValues.contains(from))
      scalarValues.insert(to);
    auto et = valueToElemType.find(from);
    if (et != valueToElemType.end())
      valueToElemType[to] = et->second;
    auto da = deferredAddPtr.find(from);
    if (da != deferredAddPtr.end())
      deferredAddPtr[to] = da->second;
    auto pb = ptrBasedDeferred.find(from);
    if (pb != ptrBasedDeferred.end())
      ptrBasedDeferred[to] = pb->second;
    auto sb = scalarBaseDeferred.find(from);
    if (sb != scalarBaseDeferred.end())
      scalarBaseDeferred[to] = sb->second;
    auto pu = packedU32Convert.find(from);
    if (pu != packedU32Convert.end())
      packedU32Convert[to] = pu->second;
    auto so = valueToSmemOffset.find(from);
    if (so != valueToSmemOffset.end())
      valueToSmemOffset[to] = so->second;
    if (pointerDescriptors.contains(from))
      pointerDescriptors.insert(to);
  };
  for (auto [arg, opnd] : llvm::zip(entry.getArguments(), op.getOperands()))
    aliasTo(arg, opnd);
  // NOTE: no `{}` scope — call results alias vars declared in the body, which
  // must stay visible after the call site.
  emit("// ── inlined tt.call @" + op.getCallee().str() + " ──");
  callInlineDepth++;
  for (auto &inner : entry.without_terminator()) {
    emitOp(&inner);
    if (emitFailed)
      break;
  }
  callInlineDepth--;
  if (emitFailed)
    return;
  auto ret = dyn_cast<tt::ReturnOp>(entry.getTerminator());
  if (!ret) {
    emitFailed = true;
    emitErrorMsg = "CUDA emitter: tt.call callee '" + op.getCallee().str() +
                   "' terminator is not tt.return";
    return;
  }
  for (auto [res, opnd] : llvm::zip(op.getResults(), ret.getOperands()))
    aliasTo(res, opnd);
  emit("// ── end inlined call @" + op.getCallee().str() + " ──");
}

// ═══════════════════════════════════════════════════════════════════════
// Op dispatch — TypeSwitch over all supported ops
// ═══════════════════════════════════════════════════════════════════════

void CUDACodeGen::emitOp(Operation *op) {
  // Once an unsupported op has been hit, stop emitting: downstream emitters may
  // crash on the resulting incomplete state. The failure is reported by caller.
  if (emitFailed)
    return;

  // Record this op's source location; emit() interleaves #line directives so
  // nvcc -lineinfo maps the generated CUDA back to the Triton .py source.
  if (lineInfoEnabled) {
    Location loc = op->getLoc();
    while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callLoc.getCallee();
    while (auto nameLoc = dyn_cast<NameLoc>(loc))
      loc = nameLoc.getChildLoc();
    if (auto flc = dyn_cast<FileLineColLoc>(loc)) {
      std::string file = flc.getFilename().str();
      int line = flc.getLine();
      if (line > 0 && !file.empty()) {
        curLocLine = line;
        curLocFile = std::move(file);
      }
    }
  }

  // Return ops
  if (isa<tt::ReturnOp>(op))
    return;

  // Arithmetic constants
  if (auto constOp = dyn_cast<arith::ConstantOp>(op))
    return emitArithConstant(constOp);

  // Arithmetic binary ops
  if (isa<arith::AddIOp, arith::AddFOp, arith::SubIOp, arith::SubFOp,
          arith::MulIOp, arith::MulFOp, arith::DivSIOp, arith::DivUIOp,
          arith::DivFOp, arith::RemSIOp, arith::RemUIOp, arith::RemFOp,
          arith::CeilDivSIOp, arith::CeilDivUIOp, arith::FloorDivSIOp,
          arith::AndIOp,
          arith::OrIOp, arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp,
          arith::ShRUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
          arith::MinUIOp, arith::MaxNumFOp, arith::MinNumFOp,
          arith::MaximumFOp, arith::MinimumFOp>(op))
    return emitArithBinary(op);

  // Arithmetic casts
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::ExtFOp,
          arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
          arith::FPToUIOp, arith::IndexCastOp, arith::IndexCastUIOp,
          arith::BitcastOp>(op))
    return emitArithCast(op);

  // Arithmetic comparisons
  if (isa<arith::CmpIOp, arith::CmpFOp>(op))
    return emitArithCmp(op);

  // Select
  if (auto selectOp = dyn_cast<arith::SelectOp>(op))
    return emitArithSelect(selectOp);

  // Math ops. Guard getDialect(): it is null for unregistered ops (e.g. the
  // emitcuda.named_barrier ops inserted by WgPingpongPass), so an unguarded
  // deref here would fault before those ops reach their own dispatch below.
  if (Dialect *d = op->getDialect())
    if (d->getNamespace() == "math")
      return emitMathOp(op);

  // Precise (round-to-nearest) float ops. tl.math.sqrt_rn / div_rn lower to
  // tt.precise_sqrt / tt.precise_divf. These need the IEEE round-to-nearest
  // intrinsics __fsqrt_rn / __fdiv_rn — plain sqrtf / division are not
  // correctly-rounded.
  if (isa<tt::PreciseSqrtOp, tt::PreciseDivFOp>(op)) {
    auto result = op->getResult(0);
    bool isTensor = isa<RankedTensorType>(result.getType());
    auto var = newVar("pm");
    valueToVar[result] = var;
    Type elemType = isTensor
                        ? cast<RankedTensorType>(result.getType()).getElementType()
                        : result.getType();
    auto cudaType = getCUDAType(elemType);
    bool isF32 = elemType.isF32();
    bool isDiv = isa<tt::PreciseDivFOp>(op);
    // Round-to-nearest intrinsics only exist for f32. For other float widths
    // (f16/bf16/f64) fall back to the ordinary operator on the value, which is
    // already correctly rounded by the hardware for those types.
    auto elemFn = [&](const std::string &a, const std::string &b) -> std::string {
      if (isDiv) {
        if (isF32)
          return "__fdiv_rn(" + a + ", " + b + ")";
        return "((" + a + ") / (" + b + "))";
      }
      if (isF32)
        return "__fsqrt_rn(" + a + ")";
      return "sqrtf(" + a + ")";
    };
    if (isTensor) {
      int nElems = getElemsPerThread(result);
      emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      std::string a = getElemExpr(op->getOperand(0), "_i");
      std::string b = isDiv ? getElemExpr(op->getOperand(1), "_i") : "";
      emit("    " + var + "[_i] = " + elemFn(a, b) + ";");
    } else {
      std::string a = getVar(op->getOperand(0));
      std::string b = isDiv ? getVar(op->getOperand(1)) : "";
      emit("const " + cudaType + " " + var + " = " + elemFn(a, b) + ";");
    }
    return;
  }

  // tt.clampf: clamp(x, lo, hi). With propagateNan=ALL the result must be NaN
  // when x is NaN (PTX backend uses min.NaN/max.NaN); with NONE, fminf/fmaxf
  // already give the clamp-to-range behavior for a NaN input clamped against
  // non-NaN bounds.
  if (auto clampOp = dyn_cast<tt::ClampFOp>(op)) {
    auto result = clampOp.getResult();
    bool isTensor = isa<RankedTensorType>(result.getType());
    auto var = newVar("clamp");
    valueToVar[result] = var;
    Type elemType =
        isTensor ? cast<RankedTensorType>(result.getType()).getElementType()
                 : result.getType();
    auto cudaType = getCUDAType(elemType);
    bool halfLike = (cudaType == "__half" || cudaType == "__nv_bfloat16");
    bool isF64 = cudaType == "double";
    bool propNan = clampOp.getPropagateNan() == tt::PropagateNan::ALL;
    auto elemFn = [&](std::string x, std::string lo,
                      std::string hi) -> std::string {
      if (halfLike) {
        x = "(float)(" + x + ")";
        lo = "(float)(" + lo + ")";
        hi = "(float)(" + hi + ")";
      }
      std::string fmin = isF64 ? "fmin" : "fminf";
      std::string fmax = isF64 ? "fmax" : "fmaxf";
      std::string call =
          fmin + "(" + fmax + "(" + x + ", " + lo + "), " + hi + ")";
      if (propNan) {
        std::string nanLit = isF64 ? "nan(\"\")" : "nanf(\"\")";
        call = "(isnan(" + x + ") ? " + nanLit + " : " + call + ")";
      }
      if (halfLike)
        call = "(" + cudaType + ")(" + call + ")";
      return call;
    };
    if (isTensor && !(scalarValues.contains(clampOp.getX()) &&
                      scalarValues.contains(clampOp.getMin()) &&
                      scalarValues.contains(clampOp.getMax()))) {
      int nElems = getElemsPerThread(result);
      emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      emit("    " + var + "[_i] = " +
           elemFn(getElemExpr(clampOp.getX(), "_i"),
                  getElemExpr(clampOp.getMin(), "_i"),
                  getElemExpr(clampOp.getMax(), "_i")) +
           ";");
    } else {
      if (isTensor)
        scalarValues.insert(result);
      std::string x = isTensor ? getElemExpr(clampOp.getX(), "0")
                               : getVar(clampOp.getX());
      std::string lo = isTensor ? getElemExpr(clampOp.getMin(), "0")
                                : getVar(clampOp.getMin());
      std::string hi = isTensor ? getElemExpr(clampOp.getMax(), "0")
                                : getVar(clampOp.getMax());
      emit("const " + cudaType + " " + var + " = " + elemFn(x, lo, hi) + ";");
    }
    return;
  }

  // Triton core ops
  if (auto getPidOp = dyn_cast<tt::GetProgramIdOp>(op))
    return emitGetProgramId(getPidOp);
  if (auto getNumOp = dyn_cast<tt::GetNumProgramsOp>(op)) {
    auto var = newVar("np");
    valueToVar[getNumOp.getResult()] = var;
    auto axis = getNumOp.getAxisAsInt();
    std::string dim = axis == 0 ? "x" : (axis == 1 ? "y" : "z");
    if (numCtas > 1) {
      // Number of programs = number of clusters (reference: %nclusterid).
      emit("int " + var + ";");
      emit("asm(\"mov.u32 %0, %%nclusterid." + dim + ";\" : \"=r\"(" + var +
           "));");
      return;
    }
    emit("const int " + var + " = gridDim." + dim + ";");
    return;
  }
  if (isa<mlir::ub::PoisonOp>(op)) {
    // ub.poison: produced by lift-cfg-to-scf for values left undefined on
    // some control-flow paths. Any value is valid; zero-init so garbage can
    // never propagate.
    Value res = op->getResult(0);
    auto var = newVar("poison");
    valueToVar[res] = var;
    if (auto rtt = dyn_cast<RankedTensorType>(res.getType())) {
      int n = getElemsPerThread(rtt);
      auto t = getCUDAType(rtt.getElementType());
      emit(t + " " + var + "[" + std::to_string(n) + "] = {};");
    } else {
      auto t = getCUDAType(res.getType());
      emit(t + " " + var + " = (" + t + ")0;");
    }
    return;
  }
  if (auto makeRangeOp = dyn_cast<tt::MakeRangeOp>(op))
    return emitMakeRange(makeRangeOp);
  if (auto splatOp = dyn_cast<tt::SplatOp>(op))
    return emitSplat(splatOp);
  if (auto unsplatOp = dyn_cast<tt::UnsplatOp>(op))
    return emitUnsplat(unsplatOp);
  if (auto broadcastOp = dyn_cast<tt::BroadcastOp>(op))
    return emitBroadcast(broadcastOp);
  if (auto expandDimsOp = dyn_cast<tt::ExpandDimsOp>(op))
    return emitExpandDims(expandDimsOp);
  if (auto addptrOp = dyn_cast<tt::AddPtrOp>(op))
    return emitAddPtr(addptrOp);
  if (auto loadOp = dyn_cast<tt::LoadOp>(op))
    return emitLoad(loadOp);
  if (auto storeOp = dyn_cast<tt::StoreOp>(op))
    return emitStore(storeOp);
  if (auto dotOp = dyn_cast<tt::DotOp>(op))
    return emitDot(dotOp);
  if (auto reduceOp = dyn_cast<tt::ReduceOp>(op))
    return emitReduce(reduceOp);
  if (auto scanOp = dyn_cast<tt::ScanOp>(op))
    return emitScan(scanOp);
  if (auto gatherOp = dyn_cast<tt::GatherOp>(op))
    return emitGather(gatherOp);
  if (auto transOp = dyn_cast<tt::TransOp>(op))
    return emitTrans(transOp);
  if (auto bitcastOp = dyn_cast<tt::BitcastOp>(op))
    return emitBitcast(bitcastOp);
  if (auto intToPtrOp = dyn_cast<tt::IntToPtrOp>(op))
    return emitIntToPtr(intToPtrOp);
  if (auto ptrToIntOp = dyn_cast<tt::PtrToIntOp>(op))
    return emitPtrToInt(ptrToIntOp);
  if (auto fpToFpOp = dyn_cast<tt::FpToFpOp>(op))
    return emitFpToFp(fpToFpOp);
  if (auto fp4ToFpOp = dyn_cast<ttg::Fp4ToFpOp>(op))
    return emitFp4ToFp(fp4ToFpOp);
  if (auto externOp = dyn_cast<tt::ExternElementwiseOp>(op))
    return emitExternElementwise(externOp);
  if (auto atomicOp = dyn_cast<tt::AtomicRMWOp>(op))
    return emitAtomicRMW(atomicOp);
  if (auto casOp = dyn_cast<tt::AtomicCASOp>(op))
    return emitAtomicCAS(casOp);
  if (auto assertOp = dyn_cast<tt::AssertOp>(op)) {
    // Mirror the PTX backend (AssertOpToLLVM): any zero element triggers
    // __assert_fail(message, file, line, func); barrier afterwards for tensor
    // conditions so no thread races ahead into a potentially-trapping op.
    auto cond = assertOp.getCondition();
    auto escape = [](llvm::StringRef s) {
      std::string out;
      for (char c : s) {
        if (c == '"' || c == '\\')
          out += '\\';
        if (c == '\n') {
          out += "\\n";
          continue;
        }
        out += c;
      }
      return out;
    };
    std::string file = "unknown", func = "unknown";
    int line = 0;
    Location loc = assertOp.getLoc();
    while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callLoc.getCallee();
    while (auto nameLoc = dyn_cast<NameLoc>(loc))
      loc = nameLoc.getChildLoc();
    if (auto flc = dyn_cast<FileLineColLoc>(loc)) {
      file = flc.getFilename().str();
      line = flc.getLine();
    }
    std::string failCall = "__assert_fail(\"" + escape(assertOp.getMessage()) +
                           "\", \"" + escape(file) + "\", " +
                           std::to_string(line) + ", \"" + escape(func) +
                           "\");";
    bool isTensor = isa<RankedTensorType>(cond.getType());
    if (isTensor) {
      int nElems = getElemsPerThread(cond);
      for (int i = 0; i < nElems; i++)
        emit("if (!(" + getElemExpr(cond, std::to_string(i)) + ")) " +
             failCall);
      blockSync();
    } else {
      emit("if (!(" + getVar(cond) + ")) " + failCall);
    }
    return;
  }
  if (auto reshapeOp = dyn_cast<tt::ReshapeOp>(op)) {
    // Reshape is a logical view change — same register data, different shape
    auto result = reshapeOp.getResult();
    auto src = reshapeOp.getSrc();
    int nDst = getElemsPerThread(result);
    int nSrc = getElemsPerThread(src);
    auto srcVar = getVar(src);
    auto var = newVar("rsh");
    valueToVar[result] = var;
    auto cudaType = getCUDAType(cast<RankedTensorType>(result.getType()).getElementType());
    // A bare pointer alias is only valid when src and dst layouts place each
    // register element on the SAME (lane,warp) AND at the same flattened
    // logical offset. When nDst==nSrc but the (reg,lane,warp)->offset mapping
    // differs (e.g. #linear -> #blocked from vpopc), the alias silently
    // corrupts data and we must fall through to the shared-memory roundtrip.
    bool aliasOK = (nDst == nSrc);
    // allow_reorder reshape: the PTX backend (ViewOpToLLVM.cpp
    // ReshapeOpConversion) lowers ANY non-expensive reshape as a direct
    // register forward (unpackLLElements -> packLLElements, identity order) —
    // zero shared memory, zero barriers. With allow_reorder the op explicitly
    // permits an arbitrary element-to-thread reassignment, so consumers are
    // order-insensitive (e.g. flexpoint absmax / integer checksum global
    // reductions in matmul_ogs). Mirror PTX exactly: plain alias.
    if (reshapeOp.getAllowReorder() && aliasOK) {
      emit(cudaType + "* " + var + " = " + srcVar +
           "; // reshape (allow_reorder register forward)");
      return;
    }
    // Register-permutation fast path: tt.reshape is contractually a per-thread
    // nop (the output encoding is inferred so no data moves across lanes or
    // warps; see TritonOps.td). When the flattened lane/warp deltas match but
    // the per-register offsets are merely permuted, emit register copies
    // instead of the shared-memory roundtrip (PTX backend never uses smem for
    // reshape). This is both a smem-footprint fix (task #44: two 128x256 f32
    // reshape scratches pushed persistent fp8 kernels over the smem limit) and
    // a perf win (no barriers).
    SmallVector<int> reshapePerm;
    if (aliasOK) {
      auto rtt0 = cast<RankedTensorType>(result.getType());
      auto srcRtt0 = cast<RankedTensorType>(src.getType());
      auto dstLL0 = ttg::toLinearLayout(rtt0);
      auto srcLL0 = ttg::toLinearLayout(srcRtt0);
      auto kReg0 = mlir::StringAttr::get(rtt0.getContext(), "register");
      auto kLane0 = mlir::StringAttr::get(rtt0.getContext(), "lane");
      auto kWarp0 = mlir::StringAttr::get(rtt0.getContext(), "warp");
      auto kBlock0 = mlir::StringAttr::get(rtt0.getContext(), "block");
      auto descOf = [&](LinearLayout &ll, ArrayRef<int64_t> shape, int n,
                        std::vector<int64_t> &regOff,
                        std::vector<int64_t> &laneD,
                        std::vector<int64_t> &warpD) {
        SmallVector<int64_t> strides(shape.size());
        strides.back() = 1;
        for (int d = (int)shape.size() - 2; d >= 0; d--)
          strides[d] = strides[d + 1] * shape[d + 1];
        const auto &bases = ll.getBases();
        const auto &laneBases = bases.find(kLane0)->second;
        const auto &warpBases = bases.find(kWarp0)->second;
        regOff.resize(n);
        for (int i = 0; i < n; i++) {
          auto coords = ll.apply({{kReg0, i}, {kLane0, 0}, {kWarp0, 0}, {kBlock0, 0}});
          int64_t off = 0;
          for (size_t d = 0; d < coords.size(); d++)
            off += coords[d].second * strides[d];
          regOff[i] = off;
        }
        laneD.resize(laneBases.size());
        for (size_t lb = 0; lb < laneBases.size(); lb++) {
          int64_t delta = 0;
          for (size_t d = 0; d < laneBases[lb].size(); d++)
            delta += laneBases[lb][d] * strides[d];
          laneD[lb] = delta;
        }
        warpD.resize(warpBases.size());
        for (size_t wb = 0; wb < warpBases.size(); wb++) {
          int64_t delta = 0;
          for (size_t d = 0; d < warpBases[wb].size(); d++)
            delta += warpBases[wb][d] * strides[d];
          warpD[wb] = delta;
        }
      };
      std::vector<int64_t> sReg, sLane, sWarp, dReg, dLane, dWarp;
      descOf(srcLL0, srcRtt0.getShape(), nSrc, sReg, sLane, sWarp);
      descOf(dstLL0, rtt0.getShape(), nDst, dReg, dLane, dWarp);
      aliasOK = (sReg == dReg) && (sLane == dLane) && (sWarp == dWarp);
      if (!aliasOK && sLane == dLane && sWarp == dWarp) {
        // Same lane/warp mapping — try a pure register permutation. Require
        // all source register offsets distinct (no broadcast duplicates) so
        // the offset->register map is a bijection.
        llvm::DenseMap<int64_t, int> srcOffToReg;
        for (int i = 0; i < nSrc; i++)
          srcOffToReg.try_emplace(sReg[i], i);
        if ((int)srcOffToReg.size() == nSrc) {
          reshapePerm.assign(nDst, -1);
          bool ok = true;
          for (int i = 0; i < nDst && ok; i++) {
            auto it = srcOffToReg.find(dReg[i]);
            if (it == srcOffToReg.end())
              ok = false;
            else
              reshapePerm[i] = it->second;
          }
          if (!ok)
            reshapePerm.clear();
        }
      }
    }
    if (aliasOK) {
      emit(cudaType + "* " + var + " = " + srcVar + "; // reshape (alias)");
    } else if (!reshapePerm.empty()) {
      emit(cudaType + " " + var + "[" + std::to_string(nDst) + "]; // reshape (register permutation)");
      for (int i = 0; i < nDst; i++)
        emit(var + "[" + std::to_string(i) + "] = " + srcVar + "[" +
             std::to_string(reshapePerm[i]) + "];");
    } else {
      // Element count changed — need convert_layout via shared memory
      emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
      auto rtt = cast<RankedTensorType>(result.getType());
      auto srcRtt = cast<RankedTensorType>(src.getType());
      auto dstLL = ttg::toLinearLayout(rtt);
      auto srcLL = ttg::toLinearLayout(srcRtt);
      auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
      auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
      auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
      auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");
      // Shared memory convert
      auto dstShape = rtt.getShape();
      auto srcShape = srcRtt.getShape();
      int64_t totalElems = 1;
      for (auto s : srcShape) totalElems *= s;
      int64_t smemElems = std::max(totalElems, (int64_t)(numWarps * 32 * std::max(nSrc, nDst)));
      int eb = getTypeSizeInBytes(rtt.getElementType());
      // Pure-temporary reshape scratch: stored then read back into `var` before
      // the block closes. Save/restore the bump pointer so this space is
      // reusable (see matching note in the convert_layout paths below).
      int savedSmemOffsetReshape = sharedMemOffset;
      int smemOff = (sharedMemOffset + 127) & ~127;
      sharedMemOffset = smemOff + smemElems * eb;
      if (sharedMemOffset > peakSharedMem) peakSharedMem = sharedMemOffset;
      emit("{");
      indent();
      emit(cudaType + "* _rsh = (" + cudaType + "*)(shared_mem + " + std::to_string(smemOff) + ");");
      // Store with source layout using LL
      {
        const auto &srcBases = srcLL.getBases();
        const auto &srcLaneBases = srcBases.find(kLane)->second;
        const auto &srcWarpBases = srcBases.find(kWarp)->second;
        SmallVector<int64_t> srcStrides(srcShape.size());
        srcStrides.back() = 1;
        for (int d = srcShape.size() - 2; d >= 0; d--)
          srcStrides[d] = srcStrides[d + 1] * srcShape[d + 1];
        for (int i = 0; i < nSrc; i++) {
          auto coords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
          int64_t regBase = 0;
          for (size_t d = 0; d < coords.size(); d++)
            regBase += coords[d].second * srcStrides[d];
          std::string offsetExpr = std::to_string(regBase);
          for (size_t lb = 0; lb < srcLaneBases.size(); lb++) {
            int64_t delta = 0;
            for (size_t d = 0; d < srcLaneBases[lb].size(); d++)
              delta += srcLaneBases[lb][d] * srcStrides[d];
            if (delta != 0)
              offsetExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
          }
          for (size_t wb = 0; wb < srcWarpBases.size(); wb++) {
            int64_t delta = 0;
            for (size_t d = 0; d < srcWarpBases[wb].size(); d++)
              delta += srcWarpBases[wb][d] * srcStrides[d];
            if (delta != 0)
              offsetExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
          }
          emit("_rsh[" + offsetExpr + "] = " + srcVar + "[" + std::to_string(i) + "];");
        }
      }
      blockSync();
      // Load with dest layout using LL
      {
        const auto &dstBases = dstLL.getBases();
        const auto &dstLaneBases = dstBases.find(kLane)->second;
        const auto &dstWarpBases = dstBases.find(kWarp)->second;
        SmallVector<int64_t> dstStrides(dstShape.size());
        dstStrides.back() = 1;
        for (int d = dstShape.size() - 2; d >= 0; d--)
          dstStrides[d] = dstStrides[d + 1] * dstShape[d + 1];
        for (int i = 0; i < nDst; i++) {
          auto coords = dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
          int64_t regBase = 0;
          for (size_t d = 0; d < coords.size(); d++)
            regBase += coords[d].second * dstStrides[d];
          std::string offsetExpr = std::to_string(regBase);
          for (size_t lb = 0; lb < dstLaneBases.size(); lb++) {
            int64_t delta = 0;
            for (size_t d = 0; d < dstLaneBases[lb].size(); d++)
              delta += dstLaneBases[lb][d] * dstStrides[d];
            if (delta != 0)
              offsetExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
          }
          for (size_t wb = 0; wb < dstWarpBases.size(); wb++) {
            int64_t delta = 0;
            for (size_t d = 0; d < dstWarpBases[wb].size(); d++)
              delta += dstWarpBases[wb][d] * dstStrides[d];
            if (delta != 0)
              offsetExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
          }
          emit(var + "[" + std::to_string(i) + "] = _rsh[" + offsetExpr + "];");
        }
      }
      blockSync();
      // Scratch consumed — restore the bump pointer so this space is reusable.
      sharedMemOffset = savedSmemOffsetReshape;
      dedent();
      emit("}");
    }
    return;
  }
  if (auto splitOp = dyn_cast<tt::SplitOp>(op)) {
    // Split a tensor [..., 2] into two tensors [...] = src[...,0], src[...,1].
    // Invariant (per verifier): the split (last) dim is distributed along
    // registers with sizePerThread=2, threadPerWarp=1, warpPerBlock=1.
    // So this is a pure register deinterleave (no data movement), matching the
    // reference SplitOpConversion in ViewOpToLLVM.cpp.
    auto src = splitOp.getSrc();
    auto res0 = splitOp.getOutLHS();
    auto res1 = splitOp.getOutRHS();
    int nSrc = getElemsPerThread(src);
    int nOut = nSrc / 2;
    auto srcRtt = cast<RankedTensorType>(src.getType());
    auto srcLL = ttg::toLinearLayout(srcRtt);
    int splitDim = srcRtt.getRank() - 1;
    auto kReg = mlir::StringAttr::get(srcRtt.getContext(), "register");
    const auto &regs = srcLL.getBases().find(kReg)->second;
    int numContiguousValues = 1;
    for (const auto &reg : regs) {
      if (reg[splitDim] == 1)
        break;
      numContiguousValues *= 2;
    }
    auto cudaType = getCUDAType(
        cast<RankedTensorType>(res0.getType()).getElementType());
    auto var0 = newVar("splL");
    auto var1 = newVar("splR");
    valueToVar[res0] = var0;
    valueToVar[res1] = var1;
    emit(cudaType + " " + var0 + "[" + std::to_string(nOut) + "];");
    emit(cudaType + " " + var1 + "[" + std::to_string(nOut) + "];");
    int k = 0;
    for (int i = 0; i < nSrc; i += 2 * numContiguousValues) {
      for (int j = 0; j < numContiguousValues; j++) {
        emit(var0 + "[" + std::to_string(k) + "] = " +
             getElemExpr(src, std::to_string(i + j)) + ";");
        emit(var1 + "[" + std::to_string(k) + "] = " +
             getElemExpr(src, std::to_string(i + numContiguousValues + j)) +
             ";");
        k++;
      }
    }
    return;
  }
  if (auto joinOp = dyn_cast<tt::JoinOp>(op)) {
    // Join two tensors [...] into [..., 2] — the inverse of SplitOp. Per the
    // verifier, the join (last) dim is distributed along registers with
    // sizePerThread=2; so this is a pure register interleave (no data movement).
    // result[2i+j] = lhs[i+j], result[2i+ncv+j] = rhs[i+j] for contiguous chunks.
    auto lhs = joinOp.getLhs();
    auto rhs = joinOp.getRhs();
    auto result = joinOp.getResult();
    int nLhs = getElemsPerThread(lhs);
    int nDst = nLhs * 2;
    auto dstRtt = cast<RankedTensorType>(result.getType());
    auto dstLL = ttg::toLinearLayout(dstRtt);
    int splitDim = dstRtt.getRank() - 1;
    auto kReg = mlir::StringAttr::get(dstRtt.getContext(), "register");
    const auto &regs = dstLL.getBases().find(kReg)->second;
    int numContiguousValues = 1;
    for (const auto &reg : regs) {
      if (reg[splitDim] == 1)
        break;
      numContiguousValues *= 2;
    }
    auto cudaType = getCUDAType(dstRtt.getElementType());
    auto var = newVar("join");
    valueToVar[result] = var;
    emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
    for (int i = 0; i < nLhs; i += numContiguousValues) {
      for (int j = 0; j < numContiguousValues; j++) {
        emit(var + "[" + std::to_string(2 * i + j) + "] = " +
             getElemExpr(lhs, std::to_string(i + j)) + ";");
        emit(var + "[" + std::to_string(2 * i + numContiguousValues + j) +
             "] = " + getElemExpr(rhs, std::to_string(i + j)) + ";");
      }
    }
    return;
  }
  if (auto catOp = dyn_cast<tt::CatOp>(op)) {
    // tt.cat: per the reference CatOpConversion (ViewOpToLLVM.cpp), the result
    // layout is defined so concatenation is a pure per-thread register append:
    // result regs = lhs regs ++ rhs regs. No data movement.
    auto lhs = catOp.getLhs();
    auto rhs = catOp.getRhs();
    auto result = catOp.getResult();
    int nLhs = getElemsPerThread(lhs);
    int nRhs = getElemsPerThread(rhs);
    int nDst = getElemsPerThread(result);
    if (nDst != nLhs + nRhs) {
      emitFailed = true;
      emitErrorMsg = "CUDA emitter: tt.cat register count mismatch (lhs " +
                     std::to_string(nLhs) + " + rhs " + std::to_string(nRhs) +
                     " != dst " + std::to_string(nDst) + ")";
      return;
    }
    auto cudaType = getCUDAType(
        cast<RankedTensorType>(result.getType()).getElementType());
    auto var = newVar("cat");
    valueToVar[result] = var;
    emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
    for (int i = 0; i < nLhs; i++)
      emit(var + "[" + std::to_string(i) + "] = " +
           getElemExpr(lhs, std::to_string(i)) + ";");
    for (int i = 0; i < nRhs; i++)
      emit(var + "[" + std::to_string(nLhs + i) + "] = " +
           getElemExpr(rhs, std::to_string(i)) + ";");
    return;
  }
  if (auto histOp = dyn_cast<tt::HistogramOp>(op)) {
    // Port of HistogramOpToLLVM.cpp (warp-ballot algorithm by @apgoucher):
    // one __ballot_sync per bin-index bit; each lane owns numBins/32 bins and
    // popcounts indicator masks built from the ballots. Cross-warp combine via
    // atomicAdd into a transient smem int[numBins] scratch.
    auto src = histOp.getSrc();
    auto maskVal = histOp.getMask();
    auto result = histOp.getResult();
    auto dstRtt = cast<RankedTensorType>(result.getType());
    auto srcRtt = cast<RankedTensorType>(src.getType());
    int64_t N = dstRtt.getDimSize(0);
    // Pad so every lane owns >= 1 bin (mirrors reference).
    int numBins = std::max<int>((int)N, 32);
    if ((numBins & (numBins - 1)) != 0) {
      emitFailed = true;
      emitErrorMsg = "CUDA emitter: tt.histogram with non-power-of-2 bins (" +
                     std::to_string(numBins) + ")";
      return;
    }
    int numBits = __builtin_ctz((unsigned)numBins);
    const int numBitsLaneId = 5; // 32 threads per warp
    int binsPerThread = numBins / 32;
    int nSrc = getElemsPerThread(src);
    int nDst = getElemsPerThread(result);

    auto var = newVar("hist");
    valueToVar[result] = var;
    emit("int " + var + "[" + std::to_string(nDst) + "];");

    // Transient smem scratch (save/restore bump pointer, reshape idiom).
    int savedSmemOffsetHist = sharedMemOffset;
    int smemOff = (sharedMemOffset + 127) & ~127;
    sharedMemOffset = smemOff + numBins * 4;
    if (sharedMemOffset > peakSharedMem)
      peakSharedMem = sharedMemOffset;
    emit("{");
    indent();
    emit("int* _hsm = (int*)(shared_mem + " + std::to_string(smemOff) + ");");
    // Warp-level histogram: binsPerThread bins owned by each lane.
    emit("int _hwl[" + std::to_string(binsPerThread) + "];");
    for (int k = 0; k < binsPerThread; k++)
      emit("_hwl[" + std::to_string(k) + "] = 0;");
    for (int i = 0; i < nSrc; i++) {
      emit("{");
      indent();
      std::string x = getElemExpr(src, std::to_string(i));
      emit("unsigned _hb[" + std::to_string(numBits) + "];");
      for (int j = 0; j < numBits; j++)
        emit("_hb[" + std::to_string(j) + "] = __ballot_sync(0xffffffffu, (" +
             x + " & " + std::to_string(1 << j) + ") != 0);");
      // Lane-owned mask: elements whose top numBitsLaneId index bits == lane.
      emit("unsigned _hm = 0xffffffffu;");
      for (int b = 0; b < numBitsLaneId; b++)
        emit("_hm &= _hb[" + std::to_string(b + numBits - numBitsLaneId) +
             "] ^ (((lane_id >> " + std::to_string(b) +
             ") & 1) ? 0u : 0xffffffffu);");
      if (maskVal)
        emit("_hm &= __ballot_sync(0xffffffffu, " +
             getElemExpr(maskVal, std::to_string(i)) + ");");
      for (int k = 0; k < binsPerThread; k++) {
        std::string bm = "_hbm" + std::to_string(k);
        emit("unsigned " + bm + " = _hm;");
        for (int j = 0; j < numBits - numBitsLaneId; j++)
          emit(bm + " &= _hb[" + std::to_string(j) + "] ^ " +
               ((k & (1 << j)) ? std::string("0u")
                               : std::string("0xffffffffu")) +
               ";");
        emit("_hwl[" + std::to_string(k) + "] += __popc(" + bm + ");");
      }
      dedent();
      emit("}");
    }
    // Cross-warp: zero smem, atomicAdd warp-level bins, read back.
    int zeroIters = (numBins + numWarps * 32 - 1) / (numWarps * 32);
    for (int i = 0; i < zeroIters; i++)
      emit("_hsm[(threadIdx.x + " + std::to_string(i * numWarps * 32) +
           ") % " + std::to_string(numBins) + "] = 0;");
    blockSync();
    for (int k = 0; k < binsPerThread; k++)
      emit("atomicAdd(&_hsm[lane_id * " + std::to_string(binsPerThread) +
           " + " + std::to_string(k) + "], _hwl[" + std::to_string(k) + "]);");
    blockSync();
    // Replication factor: lanes/warps holding duplicate copies of the source
    // each contributed a full warp-level histogram (reference divides by
    // numWarps*32 / (Πtpw × Πwpc); LL free bits are the equivalent).
    auto srcLL = ttg::toLinearLayout(srcRtt);
    auto freeMasks = srcLL.getFreeVariableMasks();
    auto kLaneF = mlir::StringAttr::get(srcRtt.getContext(), "lane");
    auto kWarpF = mlir::StringAttr::get(srcRtt.getContext(), "warp");
    int laneFree = freeMasks.count(kLaneF) ? freeMasks[kLaneF] : 0;
    int warpFree = freeMasks.count(kWarpF) ? freeMasks[kWarpF] : 0;
    int repl = 1 << (__builtin_popcount((unsigned)laneFree) +
                     __builtin_popcount((unsigned)warpFree));
    // Read back with the dst layout (1D).
    auto dstLL = ttg::toLinearLayout(dstRtt);
    auto kReg = mlir::StringAttr::get(dstRtt.getContext(), "register");
    auto kLane = kLaneF;
    auto kWarp = kWarpF;
    auto kBlock = mlir::StringAttr::get(dstRtt.getContext(), "block");
    const auto &dstBases = dstLL.getBases();
    const auto &dstLaneBases = dstBases.find(kLane)->second;
    const auto &dstWarpBases = dstBases.find(kWarp)->second;
    for (int i = 0; i < nDst; i++) {
      auto coords =
          dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      std::string offsetExpr = std::to_string(coords[0].second);
      for (size_t lb = 0; lb < dstLaneBases.size(); lb++)
        if (dstLaneBases[lb][0] != 0)
          offsetExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " +
                        std::to_string(dstLaneBases[lb][0]);
      for (size_t wb = 0; wb < dstWarpBases.size(); wb++)
        if (dstWarpBases[wb][0] != 0)
          offsetExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " +
                        std::to_string(dstWarpBases[wb][0]);
      emit(var + "[" + std::to_string(i) + "] = _hsm[" + offsetExpr + "]" +
           (repl > 1 ? " / " + std::to_string(repl) : "") + ";");
    }
    // Scratch consumed — guard against next smem reuse racing slow readers.
    blockSync();
    sharedMemOffset = savedSmemOffsetHist;
    dedent();
    emit("}");
    return;
  }
  if (auto callOp = dyn_cast<tt::CallOp>(op))
    return emitCall(callOp);
  if (auto printOp = dyn_cast<tt::PrintOp>(op))
    return emitPrint(printOp);
  if (auto mulhiOp = dyn_cast<tt::MulhiUIOp>(op))
    return emitMulhiUI(mulhiOp);
  if (auto asmOp = dyn_cast<tt::ElementwiseInlineAsmOp>(op))
    return emitElementwiseInlineAsm(asmOp);
  if (auto mapOp = dyn_cast<tt::MapElementwiseOp>(op))
    return emitMapElementwise(mapOp);

  // SCF control flow
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return emitScfFor(forOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return emitScfIf(ifOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return emitScfWhile(whileOp);
  if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op))
    return emitScfIndexSwitch(switchOp);
  if (isa<scf::YieldOp, scf::ConditionOp>(op))
    return; // Handled by parent op

  // GPU barrier. v3.7.0 lowers CTA-wide barriers to ttg.barrier (TTG_BarrierOp);
  // earlier Triton used gpu.barrier. Both are full-CTA __syncthreads().
  if (isa<mlir::gpu::BarrierOp, ttg::BarrierOp>(op)) {
    blockSync();
    return;
  }

  // tl.assume -> llvm.intr.assume: scalar i1 optimizer hint. Matched by name
  // so we don't need to link the LLVM dialect headers just for this op.
  if (op->getName().getStringRef() == "llvm.intr.assume") {
    Value cond = op->getOperand(0);
    if (!isa<RankedTensorType>(cond.getType())) {
      emit("__builtin_assume(" + getVar(cond) + ");");
      return;
    }
  }

  // TritonGPU ops
  if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op))
    return emitLocalAlloc(allocOp);
  if (auto localStoreOp = dyn_cast<ttg::LocalStoreOp>(op))
    return emitLocalStore(localStoreOp);
  if (auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(op))
    return emitLocalLoad(localLoadOp);
  if (auto deallocOp = dyn_cast<ttg::LocalDeallocOp>(op))
    return emitLocalDealloc(deallocOp);
  if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(op))
    return emitConvertLayout(cvtOp);
  if (isa<ttg::MemDescSubsliceOp, ttg::MemDescIndexOp>(op) ||
      op->getName().getStringRef() == "ttg.memdesc_index" ||
      op->getName().getStringRef() == "ttg.memdesc_subview")
    return emitMemDescSubview(op);
  if (auto transOp = dyn_cast<ttg::MemDescTransOp>(op))
    return emitMemDescTrans(transOp);
  if (auto reshapeOp = dyn_cast<ttg::MemDescReshapeOp>(op)) {
    // Layout-preserving view (verifier guarantees the underlying linear layout
    // is unchanged) — alias the shared-memory base pointer, no data movement.
    valueToVar[reshapeOp.getResult()] = getVar(reshapeOp.getSrc());
    return;
  }

  // Warp specialization (ttg.warp_specialize): our custom Hopper warp-group
  // dispatch codegen (producer/consumer split). See emitWarpSpecialize.
  if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op))
    return emitWarpSpecialize(wsOp);
  // Terminators of the warp_specialize regions — consumed by emitWarpSpecialize.
  if (isa<ttg::WarpYieldOp, ttg::WarpReturnOp>(op))
    return;
  // The partitions-holder op is walked explicitly by emitWarpSpecialize; if we
  // ever reach it via normal dispatch it carries no standalone semantics.
  if (isa<ttg::WarpSpecializePartitionsOp>(op))
    return;

  // emitcuda.named_barrier — unregistered op inserted by WgPingpongPass to
  // realize level6's inter-warpgroup ping-pong (bar.sync/bar.arrive %id,%n).
  if (op->getName().getStringRef() == "emitcuda.named_barrier")
    return emitNamedBarrier(op);

  // NVGPUIR ops (sm90a)
  if (auto wgDotOp = dyn_cast<ttng::WarpGroupDotOp>(op))
    return emitWarpGroupDot(wgDotOp);
  if (auto wgDotWaitOp = dyn_cast<ttng::WarpGroupDotWaitOp>(op))
    return emitWarpGroupDotWait(wgDotWaitOp);
  if (auto fenceOp = dyn_cast<ttng::FenceAsyncSharedOp>(op))
    return emitFenceAsyncShared(fenceOp);
  if (auto initBarOp = dyn_cast<ttng::InitBarrierOp>(op))
    return emitInitBarrier(initBarOp);
  if (auto waitBarOp = dyn_cast<ttng::WaitBarrierOp>(op))
    return emitWaitBarrier(waitBarOp);
  if (auto arriveBarOp = dyn_cast<ttng::ArriveBarrierOp>(op))
    return emitArriveBarrier(arriveBarOp);
  if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op))
    return emitBarrierExpect(expectOp);
  if (auto invalOp = dyn_cast<ttng::InvalBarrierOp>(op))
    return emitInvalBarrier(invalOp);
  if (auto fenceClOp =
          dyn_cast<ttng::FenceMBarrierInitReleaseClusterOp>(op))
    return emitFenceMBarrierInitReleaseCluster(fenceClOp);
  if (auto clArriveOp = dyn_cast<ttng::ClusterArriveOp>(op))
    return emitClusterArrive(clArriveOp);
  if (auto clWaitOp = dyn_cast<ttng::ClusterWaitOp>(op))
    return emitClusterWait(clWaitOp);
  if (auto tmaCopyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return emitAsyncTMACopyG2L(tmaCopyOp);
  if (auto tmaCopyOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return emitAsyncTMACopyL2G(tmaCopyOp);
  if (auto tmaReduceOp = dyn_cast<ttng::AsyncTMAReduceOp>(op))
    return emitAsyncTMAReduce(tmaReduceOp);
  if (auto tmaWaitOp = dyn_cast<ttng::TMAStoreWaitOp>(op))
    return emitTMAStoreWait(tmaWaitOp);

  // TTG async copy ops (cp.async based pipelining)
  if (auto asyncCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
    return emitAsyncCopyG2L(asyncCopyOp);
  if (auto commitOp = dyn_cast<ttg::AsyncCommitGroupOp>(op))
    return emitAsyncCommitGroup(commitOp);
  if (auto waitOp = dyn_cast<ttg::AsyncWaitOp>(op))
    return emitAsyncWait(waitOp);

  // tt.reduce.return — handled by reduce emitter
  if (isa<tt::ReduceReturnOp>(op))
    return;

  // Device-side TMA descriptor creation (tl.make_tensor_descriptor)
  if (auto allocOp = dyn_cast<ttg::GlobalScratchAllocOp>(op))
    return emitGlobalScratchAlloc(allocOp);
  if (auto tcOp = dyn_cast<ttng::TensormapCreateOp>(op))
    return emitTensormapCreate(tcOp);
  if (auto tfOp = dyn_cast<ttng::TensormapFenceproxyAcquireOp>(op))
    return emitTensormapFenceproxyAcquire(tfOp);
  if (auto rtOp = dyn_cast<ttng::ReinterpretTensorDescOp>(op))
    return emitReinterpretTensorDesc(rtOp);

  // ── Explicitly-skipped ops (true no-ops for CUDA codegen) ─────────────
  // These carry no runtime semantics we must emit; they are pure hints/markers.
  // We skip them EXPLICITLY (not via the silent catch-all) so the decision is
  // visible and auditable. Matched by name string to avoid extra dialect deps.
  {
    llvm::StringRef opName = op->getName().getStringRef();
    // llvm.intr.assume (from tl.assume): an optimizer hint only. Dropping it is
    // semantically correct; it only forgoes a ptxas range hint (perf, not
    // correctness). TODO(perf): could emit __builtin_assume(cond).
    if (opName == "llvm.intr.assume")
      return;
  }

  // Unhandled op — HARD ERROR. The emitter must never silently skip an op:
  // a silently-dropped op produces a subtly-broken kernel (e.g. an empty
  // warp-specialized matmul that the autotuner then false-passes). Every op
  // must be explicitly handled, explicitly skipped above, or fail loudly here.
  // The TU is compiled with -fno-exceptions, so we record the failure and let
  // generate()'s caller (the pybind layer) raise a Python exception.
  {
    std::string locStr;
    llvm::raw_string_ostream los(locStr);
    op->getLoc().print(los);
    std::string msg = "[emit_cuda] unsupported op '" +
                      op->getName().getStringRef().str() + "' (dialect '" +
                      op->getName().getDialectNamespace().str() +
                      "') has no CUDA emitter. Refusing to emit a silently "
                      "broken kernel. Location: " +
                      locStr;
    op->emitError(msg);
    if (!emitFailed) {
      emitFailed = true;
      emitErrorMsg = msg;
    }
    return;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Placeholder implementations for all op emitters.
// These will be filled in with the full logic ported from the Python emitter.
// ═══════════════════════════════════════════════════════════════════════

// For now, emit stub implementations that produce the same output format
// as the Python emitter. The full implementation will be done incrementally.

void CUDACodeGen::emitArithConstant(arith::ConstantOp op) {
  auto result = op.getResult();
  auto type = result.getType();

  if (auto rtt = dyn_cast<RankedTensorType>(type)) {
    // Dense tensor constant
    auto elemType = rtt.getElementType();
    int nElems = getElemsPerThread(rtt);
    auto var = newVar("c");
    valueToVar[result] = var;
    auto cudaType = getCUDAType(elemType);

    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      if (denseAttr.isSplat()) {
        std::string val;
        if (elemType.isF64()) {
          double v = denseAttr.getSplatValue<double>();
          if (std::isinf(v)) val = v > 0 ? "INFINITY" : "(-INFINITY)";
          else if (std::isnan(v)) val = "NAN";
          else { char buf[32]; snprintf(buf, sizeof(buf), "%.17e", v); val = buf; }
        } else if (elemType.isF32()) {
          float v = denseAttr.getSplatValue<float>();
          if (std::isinf(v)) val = v > 0 ? "INFINITY" : "(-INFINITY)";
          else if (std::isnan(v)) val = "NAN";
          else { char buf[32]; snprintf(buf, sizeof(buf), "%.10ef", (double)v); val = buf; }
        } else if (elemType.isF16() || elemType.isBF16()) {
          float v = denseAttr.getSplatValue<llvm::APFloat>().convertToFloat();
          if (std::isinf(v)) val = v > 0 ? "INFINITY" : "(-INFINITY)";
          else if (std::isnan(v)) val = "NAN";
          else { char buf[32]; snprintf(buf, sizeof(buf), "%.10e", (double)v); val = buf; }
        }
        else if (elemType.isInteger(64))
          val = std::to_string(denseAttr.getSplatValue<int64_t>());
        else if (elemType.isInteger(32))
          val = std::to_string(denseAttr.getSplatValue<int32_t>());
        else if (elemType.isInteger(16))
          val = std::to_string(denseAttr.getSplatValue<int16_t>());
        else if (elemType.isInteger(8))
          val = std::to_string(denseAttr.getSplatValue<int8_t>());
        else if (elemType.isInteger(1))
          val = denseAttr.getSplatValue<bool>() ? "1" : "0";
        else
          val = "0";

        // Splat constant: emit as scalar (avoids array overhead)
        scalarValues.insert(result);
        emit("const " + cudaType + " " + var + " = " + val + ";");
      }
    }
  } else {
    // Scalar constant
    auto var = newVar("c");
    valueToVar[result] = var;
    auto cudaType = getCUDAType(type);

    std::string val;
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      val = std::to_string(intAttr.getInt());
    } else if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
      double v = floatAttr.getValueAsDouble();
      if (std::isinf(v)) val = v > 0 ? "INFINITY" : "(-INFINITY)";
      else if (std::isnan(v)) val = "NAN";
      else {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.10e", v);
        val = buf;
        if (type.isF32()) val += "f";
      }
    } else {
      val = "0";
    }
    emit("const " + cudaType + " " + var + " = " + val + ";");
  }
}

void CUDACodeGen::emitGetProgramId(tt::GetProgramIdOp op) {
  auto var = newVar("pid");
  valueToVar[op.getResult()] = var;
  auto axis = op.getAxisAsInt();
  std::string dim = axis == 0 ? "x" : (axis == 1 ? "y" : "z");
  if (numCtas > 1) {
    // With CGA clusters the launcher scales the grid by the cluster dims and
    // the program id is the CLUSTER id (reference SPMDOpToLLVM / %clusterid).
    emit("int " + var + ";");
    emit("asm(\"mov.u32 %0, %%clusterid." + dim + ";\" : \"=r\"(" + var +
         "));");
    return;
  }
  emit("const int " + var + " = blockIdx." + dim + ";");
}

// ── Stub implementations for remaining ops ─────────────────────────
// These produce a TODO comment. Full port will replace each one.

#define STUB_EMIT(OpClass, opName)                                             \
  void CUDACodeGen::emit##OpClass(opName op) {                                \
    emit("// TODO: " #opName " not yet ported to C++");                        \
    for (auto result : op->getResults()) {                                     \
      auto var = newVar("stub");                                               \
      valueToVar[result] = var;                                                \
    }                                                                          \
  }

// Arith
void CUDACodeGen::emitArithBinary(Operation *op) {
  // TODO: full port
  auto result = op->getResult(0);
  bool isTensor = isa<RankedTensorType>(result.getType());

  // Determine the C++ operator
  std::string cppOp = "+";
  bool isCeilDiv = isa<arith::CeilDivSIOp, arith::CeilDivUIOp>(op);
  bool isFloorDiv = isa<arith::FloorDivSIOp>(op);
  llvm::StringRef opName = op->getName().getStringRef();
  if (isCeilDiv)
    cppOp = "ceildiv";
  else if (isFloorDiv)
    cppOp = "floordiv";
  else if (opName.contains("add"))
    cppOp = "+";
  else if (opName.contains("sub"))
    cppOp = "-";
  else if (opName.contains("mul"))
    cppOp = "*";
  else if (opName.contains("div"))
    cppOp = "/";
  else if (isa<arith::RemFOp>(op))
    cppOp = "fmod";
  else if (opName.contains("rem"))
    cppOp = "%";
  else if (opName.contains("and"))
    cppOp = "&";
  else if (opName.contains("or") && !opName.contains("xor"))
    cppOp = "|";
  else if (opName.contains("xor"))
    cppOp = "^";
  else if (opName.contains("shl"))
    cppOp = "<<";
  else if (opName.contains("shr"))
    cppOp = ">>";
  else if (opName.contains("maxnumf") || opName.contains("maximumf") ||
           opName.contains("maxsi") || opName.contains("maxui"))
    cppOp = "max";
  else if (opName.contains("minnumf") || opName.contains("minimumf") ||
           opName.contains("minsi") || opName.contains("minui"))
    cppOp = "min";

  auto cudaType = getCUDAType(
      isTensor
          ? cast<RankedTensorType>(result.getType()).getElementType()
          : result.getType());

  // i1 arithmetic is mod-2 (C++ bool saturates instead): add/sub == xor,
  // mul == and.
  {
    Type rt = isTensor
                  ? cast<RankedTensorType>(result.getType()).getElementType()
                  : result.getType();
    if (rt.isInteger(1)) {
      if (cppOp == "+" || cppOp == "-")
        cppOp = "^";
      else if (cppOp == "*")
        cppOp = "&";
    }
  }

  // Builds a min/max/fmod call expression with the correct overload and NaN
  // semantics. arith.minnumf/maxnumf return the non-NaN operand when one
  // input is NaN — exactly fminf/fmaxf's behavior. arith.minimumf/maximumf
  // PROPAGATE NaN (result is NaN if either input is), which fminf/fmaxf do
  // NOT do, so wrap them in an isnan ternary. Plain min()/max() must never be
  // used for floats: `(b<a)?b:a` returns the NaN operand under NaN compares.
  auto buildMinMaxFmod = [=](const std::string &a,
                             const std::string &b) -> std::string {
    std::string fn = cppOp;
    bool halfLike = (cudaType == "__half" || cudaType == "__nv_bfloat16");
    if (cudaType == "float" || halfLike) {
      if (cppOp == "max") fn = "fmaxf";
      else if (cppOp == "min") fn = "fminf";
      else fn = "fmodf";
    } else if (cudaType == "double") {
      if (cppOp == "max") fn = "fmax";
      else if (cppOp == "min") fn = "fmin";
      else fn = "fmod";
    }
    if (isa<arith::MaxUIOp, arith::MinUIOp>(op)) {
      Type elemTy = isTensor
                        ? cast<RankedTensorType>(result.getType()).getElementType()
                        : result.getType();
      auto utype = getUnsignedType(elemTy);
      return "(" + cudaType + ")" + fn + "((" + utype + ")(" + a + "), (" +
             utype + ")(" + b + "))";
    }
    std::string ae = a, be = b;
    if (halfLike) {
      // No min/max(__half,__half) overload; go through f32 (exact: f16/bf16
      // -> f32 is lossless and the result is one of the inputs).
      ae = "(float)(" + a + ")";
      be = "(float)(" + b + ")";
    }
    std::string call = fn + "(" + ae + ", " + be + ")";
    if (isa<arith::MaximumFOp, arith::MinimumFOp>(op)) {
      std::string nanLit =
          (cudaType == "double") ? "nan(\"\")" : "nanf(\"\")";
      call = "((isnan(" + ae + ") || isnan(" + be + ")) ? " + nanLit + " : " +
             call + ")";
    }
    if (halfLike)
      call = "(" + cudaType + ")" + call;
    return call;
  };

  if (isTensor) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    // If both inputs are scalar (splat), result is also scalar
    bool lhsScalar = scalarValues.contains(lhs);
    bool rhsScalar = scalarValues.contains(rhs);
    bool resultScalar = lhsScalar && rhsScalar;

    int nElems = getElemsPerThread(result);

    if (resultScalar) {
      scalarValues.insert(result);
      auto var = newVar("a");
      valueToVar[result] = var;
      std::string lhsE = getElemExpr(lhs, "_i");
      std::string rhsE = getElemExpr(rhs, "_i");
      // Scalar op — no array needed
      if (cppOp == "ceildiv") {
        emit("const " + cudaType + " " + var + " = ((" + lhsE +
             ") + (" + rhsE + ") - 1) / (" + rhsE + ");");
      } else if (cppOp == "floordiv") {
        emit("const " + cudaType + " " + var + " = (" + lhsE + " / " + rhsE +
             ") - ((" + lhsE + " % " + rhsE + " != 0) && ((" + lhsE +
             " ^ " + rhsE + ") < 0));");
      } else if (cppOp == "max" || cppOp == "min" || cppOp == "fmod") {
        emit("const " + cudaType + " " + var + " = " +
             buildMinMaxFmod(lhsE, rhsE) + ";");
      } else {
        emit("const " + cudaType + " " + var + " = (" + lhsE + " " + cppOp + " " + rhsE + ");");
      }
    } else {
      // For unsigned integer ops, cast operands to unsigned type
      bool isUnsignedOp = isa<arith::DivUIOp, arith::RemUIOp, arith::ShRUIOp,
                              arith::MaxUIOp, arith::MinUIOp>(op);
      std::string utype;
      if (isUnsignedOp)
        utype = getUnsignedType(
            cast<RankedTensorType>(result.getType()).getElementType());
      // Per-element RHS builder, evaluated lazily (deferred fusion) or at the
      // materialization point. Captures by value so it is safe to store.
      auto build = [=](const std::string &idx) -> std::string {
        std::string lhsE = getElemExpr(lhs, idx);
        std::string rhsE = getElemExpr(rhs, idx);
        if (cppOp == "ceildiv") {
          return "((" + lhsE + ") + (" + rhsE + ") - 1) / (" + rhsE + ")";
        } else if (cppOp == "floordiv") {
          return "(" + lhsE + " / " + rhsE + ") - ((" + lhsE + " % " + rhsE +
                 " != 0) && ((" + lhsE + " ^ " + rhsE + ") < 0))";
        } else if (cppOp == "max" || cppOp == "min" || cppOp == "fmod") {
          return buildMinMaxFmod(lhsE, rhsE);
        } else if (isUnsignedOp) {
          return "(" + cudaType + ")((" + utype + ")" + lhsE + " " + cppOp +
                 " (" + utype + ")" + rhsE + ")";
        } else {
          return "(" + lhsE + " " + cppOp + " " + rhsE + ")";
        }
      };
      emitOrDeferElementwise(result, op, nElems, cudaType, build);
    }
  } else {
    auto var = newVar("a");
    valueToVar[result] = var;
    auto lhsVar = getVar(op->getOperand(0));
    auto rhsVar = getVar(op->getOperand(1));
    if (cppOp == "ceildiv") {
      emit("const " + cudaType + " " + var + " = ((" + lhsVar +
           ") + (" + rhsVar + ") - 1) / (" + rhsVar + ");");
    } else if (cppOp == "floordiv") {
      // floordiv for signed: a/b rounded towards -inf
      emit("const " + cudaType + " " + var + " = (" + lhsVar + " / " + rhsVar +
           ") - ((" + lhsVar + " % " + rhsVar + " != 0) && ((" + lhsVar +
           " ^ " + rhsVar + ") < 0));");
    } else if (cppOp == "max" || cppOp == "min" || cppOp == "fmod") {
      emit("const " + cudaType + " " + var + " = " +
           buildMinMaxFmod(lhsVar, rhsVar) + ";");
    } else {
      bool isUnsignedOp = isa<arith::DivUIOp, arith::RemUIOp, arith::ShRUIOp,
                              arith::MaxUIOp, arith::MinUIOp>(op);
      if (isUnsignedOp) {
        auto utype = getUnsignedType(result.getType());
        emit("const " + cudaType + " " + var + " = (" + cudaType + ")((" +
             utype + ")" + lhsVar + " " + cppOp + " (" + utype + ")" + rhsVar + ");");
      } else {
        emit("const " + cudaType + " " + var + " = (" + lhsVar + " " + cppOp +
             " " + rhsVar + ");");
      }
    }
  }
}

void CUDACodeGen::emitArithCast(Operation *op) {
  auto result = op->getResult(0);
  auto src = op->getOperand(0);
  bool isTensor = isa<RankedTensorType>(result.getType());
  Type resultElemType =
      isTensor ? cast<RankedTensorType>(result.getType()).getElementType()
               : result.getType();
  auto cudaType = getCUDAType(resultElemType);
  bool isUnsignedCast =
      isa<arith::ExtUIOp, arith::UIToFPOp, arith::IndexCastUIOp>(op);

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    bool srcScalar = scalarValues.contains(src);

    if (srcScalar) {
      scalarValues.insert(result);
      auto var = newVar("a");
      valueToVar[result] = var;
      std::string srcE = getElemExpr(src, "_i");
      if (isUnsignedCast) {
        Type srcElemType =
            cast<RankedTensorType>(src.getType()).getElementType();
        auto unsignedSrc = getUnsignedType(srcElemType);
        emit("const " + cudaType + " " + var + " = ((" + cudaType + ")(" + unsignedSrc + ")" + srcE + ");");
      } else {
        emit("const " + cudaType + " " + var + " = ((" + cudaType + ")" + srcE + ");");
      }
    } else {
      std::string unsignedSrc;
      if (isUnsignedCast)
        unsignedSrc = getUnsignedType(
            cast<RankedTensorType>(src.getType()).getElementType());
      auto build = [=](const std::string &idx) -> std::string {
        std::string srcE = getElemExpr(src, idx);
        if (isUnsignedCast)
          return "((" + cudaType + ")(" + unsignedSrc + ")" + srcE + ")";
        return "((" + cudaType + ")" + srcE + ")";
      };
      emitOrDeferElementwise(result, op, nElems, cudaType, build);
    }
  } else {
    auto var = newVar("a");
    valueToVar[result] = var;
    auto srcVar = getVar(src);
    if (isUnsignedCast) {
      auto unsignedSrc = getUnsignedType(src.getType());
      emit("const " + cudaType + " " + var + " = ((" + cudaType + ")(" +
           unsignedSrc + ")" + srcVar + ");");
    } else {
      emit("const " + cudaType + " " + var + " = ((" + cudaType + ")" +
           srcVar + ");");
    }
  }
}

void CUDACodeGen::emitArithCmp(Operation *op) {
  auto result = op->getResult(0);
  bool isTensor = isa<RankedTensorType>(result.getType());

  std::string cmpOp = "==";
  bool isUnsignedCmp = false;
  if (auto cmpI = dyn_cast<arith::CmpIOp>(op)) {
    switch (cmpI.getPredicate()) {
    case arith::CmpIPredicate::eq:
      cmpOp = "==";
      break;
    case arith::CmpIPredicate::ne:
      cmpOp = "!=";
      break;
    case arith::CmpIPredicate::slt:
      cmpOp = "<";
      break;
    case arith::CmpIPredicate::ult:
      cmpOp = "<";
      isUnsignedCmp = true;
      break;
    case arith::CmpIPredicate::sle:
      cmpOp = "<=";
      break;
    case arith::CmpIPredicate::ule:
      cmpOp = "<=";
      isUnsignedCmp = true;
      break;
    case arith::CmpIPredicate::sgt:
      cmpOp = ">";
      break;
    case arith::CmpIPredicate::ugt:
      cmpOp = ">";
      isUnsignedCmp = true;
      break;
    case arith::CmpIPredicate::sge:
      cmpOp = ">=";
      break;
    case arith::CmpIPredicate::uge:
      cmpOp = ">=";
      isUnsignedCmp = true;
      break;
    }
  } else if (auto cmpF = dyn_cast<arith::CmpFOp>(op)) {
    switch (cmpF.getPredicate()) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      cmpOp = "==";
      break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      cmpOp = "!=";
      break;
    case arith::CmpFPredicate::OLT:
      cmpOp = "<";
      break;
    case arith::CmpFPredicate::OLE:
      cmpOp = "<=";
      break;
    case arith::CmpFPredicate::OGT:
      cmpOp = ">";
      break;
    case arith::CmpFPredicate::OGE:
      cmpOp = ">=";
      break;
    default:
      cmpOp = "==";
    }
  }

  if (isTensor) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    bool lhsScalar = scalarValues.contains(lhs);
    bool rhsScalar = scalarValues.contains(rhs);
    bool resultScalar = lhsScalar && rhsScalar;

    std::string utype;
    if (isUnsignedCmp)
      utype = getUnsignedType(
          cast<RankedTensorType>(lhs.getType()).getElementType());

    int nElems = getElemsPerThread(result);
    if (resultScalar) {
      scalarValues.insert(result);
      auto var = newVar("a");
      valueToVar[result] = var;
      std::string lhsE = getElemExpr(lhs, "_i");
      std::string rhsE = getElemExpr(rhs, "_i");
      if (isUnsignedCmp) {
        lhsE = "(" + utype + ")" + lhsE;
        rhsE = "(" + utype + ")" + rhsE;
      }
      emit("const bool " + var + " = (" + lhsE + " " + cmpOp + " " + rhsE + ");");
    } else {
      auto build = [=](const std::string &idx) -> std::string {
        std::string lhsE = getElemExpr(lhs, idx);
        std::string rhsE = getElemExpr(rhs, idx);
        if (isUnsignedCmp) {
          lhsE = "(" + utype + ")" + lhsE;
          rhsE = "(" + utype + ")" + rhsE;
        }
        return "(" + lhsE + " " + cmpOp + " " + rhsE + ")";
      };
      emitOrDeferElementwise(result, op, nElems, "bool", build);
    }
  } else {
    auto var = newVar("a");
    valueToVar[result] = var;
    auto lhsVar = getVar(op->getOperand(0));
    auto rhsVar = getVar(op->getOperand(1));
    // For unsigned integer comparisons, cast operands to unsigned type
    std::string lhsExpr, rhsExpr;
    if (isUnsignedCmp) {
      auto utype = getUnsignedType(op->getOperand(0).getType());
      lhsExpr = "(" + utype + ")" + lhsVar;
      rhsExpr = "(" + utype + ")" + rhsVar;
    } else {
      lhsExpr = lhsVar;
      rhsExpr = rhsVar;
    }
    emit("const bool " + var + " = (" + lhsExpr + " " + cmpOp + " " +
         rhsExpr + ");");
  }
}

// True iff v is a splat constant equal to floating-point zero.
static bool isZeroSplatConst(Value v) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst) return false;
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return false;
  auto et = dense.getElementType();
  if (et.isF16() || et.isBF16() || et.isF32() || et.isF64())
    return dense.getSplatValue<llvm::APFloat>().isZero();
  if (et.isIntOrIndex())
    return dense.getSplatValue<llvm::APInt>().isZero();
  return false;
}

void CUDACodeGen::emitArithSelect(arith::SelectOp op) {
  auto result = op.getResult();
  bool isTensor = isa<RankedTensorType>(result.getType());

  // Detect the WGMMA accumulator-reset cycle:
  //   %acc:f32 = scf.for iter_arg(%c = 0)
  //     %d  = ttng.warp_group_dot %a, %b, %c      (reuses %c in-place)
  //     %dw = ttng.warp_group_dot_wait %d
  //     %nx = arith.select %pred, %zeros, %dw
  //     scf.yield %nx                              (feeds back to %c)
  // The naive lowering `nx[i] = pred ? 0 : acc[i]` READS the live WGMMA
  // accumulator registers between wgmma.wait_group 1 and wgmma.wait_group 0,
  // forcing ptxas to serialize the async WGMMAs (C7514, ~1.5x slower). Alias
  // the select result to the accumulator and defer a guarded zero-reset to the
  // loop tail (after wait_group 0) — equivalent to the software-pipelined
  // register renaming the LLVM/PTX backend produces.
  if (isTensor &&
      cast<RankedTensorType>(result.getType()).getElementType().isF32() &&
      isZeroSplatConst(op.getTrueValue()) && result.hasOneUse()) {
    Value fv = op.getFalseValue();
    Value dotVal = fv;
    if (auto waitOp = fv.getDefiningOp<ttng::WarpGroupDotWaitOp>()) {
      for (unsigned wi = 0; wi < waitOp.getNumResults(); ++wi)
        if (waitOp.getResult(wi) == fv) { dotVal = waitOp.getOperand(wi); break; }
    }
    if (auto dotOp = dotVal.getDefiningOp<ttng::WarpGroupDotOp>()) {
      if (auto barg = dyn_cast<BlockArgument>(dotOp.getC())) {
        if (auto forOp = dyn_cast<scf::ForOp>(barg.getOwner()->getParentOp())) {
          unsigned argIdx = barg.getArgNumber();
          // arg 0 is the induction var; iter_args start at index 1.
          if (argIdx >= 1) {
            unsigned iterIdx = argIdx - 1;
            auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
            if (iterIdx < yieldOp.getNumOperands() &&
                yieldOp.getOperand(iterIdx) == result) {
              std::string accVar = getVar(dotOp.getC());
              valueToVar[result] = accVar;          // alias, no copy
              deferredAccReset[result] = getVar(op.getCondition());
              return;
            }
          }
        }
      }
    }
  }

  Type elemType = isTensor
                      ? cast<RankedTensorType>(result.getType()).getElementType()
                      : result.getType();
  auto cudaType = getCUDAType(elemType);

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    Value cond = op.getCondition();
    Value tv = op.getTrueValue();
    Value fv = op.getFalseValue();
    auto build = [=](const std::string &idx) -> std::string {
      // Per-operand: a non-tensor type or a tensor splat (scalarValues) reads
      // as a scalar; a real tensor reads element idx (deferred-aware).
      auto operandExpr = [&](Value v) -> std::string {
        if (!isa<RankedTensorType>(v.getType()))
          return getVar(v);
        return getElemExpr(v, idx);
      };
      return "(" + operandExpr(cond) + " ? " + operandExpr(tv) + " : " +
             operandExpr(fv) + ")";
    };
    emitOrDeferElementwise(result, op, nElems, cudaType, build);
  } else if (isa<tt::TensorDescType>(result.getType())) {
    // Selecting between tensor descriptors: hold a pointer to the 128-byte
    // CUtensorMap instead of copying it (CUtensorMap is not copyable in C++).
    auto var = newVar("a");
    valueToVar[result] = var;
    auto condVar = getVar(op.getCondition());
    pointerDescriptors.insert(result);
    emit("const char* " + var + " = (" + condVar + " ? " +
         descAddrExpr(op.getTrueValue()) + " : " +
         descAddrExpr(op.getFalseValue()) + ");");
  } else {
    auto var = newVar("a");
    valueToVar[result] = var;
    auto condVar = getVar(op.getCondition());
    auto trueVar = getVar(op.getTrueValue());
    auto falseVar = getVar(op.getFalseValue());
    // Don't use 'const' for pointer types — downstream splat/addptr need non-const
    bool isPtr = isa<tt::PointerType>(result.getType());
    std::string prefix = isPtr ? "" : "const ";
    emit(prefix + cudaType + " " + var + " = (" + condVar + " ? " +
         trueVar + " : " + falseVar + ");");
  }
}

void CUDACodeGen::emitMathOp(Operation *op) {
  auto result = op->getResult(0);
  auto src = op->getOperand(0);
  bool isTensor = isa<RankedTensorType>(result.getType());
  llvm::StringRef opName = op->getName().getStringRef();

  // Map math ops to CUDA functions
  std::string fn = "unknown";
  if (opName.contains("exp2"))
    fn = "exp2f"; // f32 routed to __triton_ex2f below (FTZ ex2.approx)
  else if (opName.contains("exp"))
    fn = "expf";
  else if (opName.contains("log2"))
    fn = "log2f";
  else if (opName.contains("log"))
    fn = "logf";
  else if (opName.contains("rsqrt")) // must precede "sqrt" (substring)
    fn = "rsqrtf";
  else if (opName.contains("sqrt"))
    fn = "sqrtf";
  else if (opName.contains("ceil"))
    fn = "ceilf";
  else if (opName.contains("floor"))
    fn = "floorf";
  else if (opName.contains("sin"))
    fn = "sinf";
  else if (opName.contains("cos"))
    fn = "cosf";
  else if (opName.contains("tanh"))
    fn = "tanhf";
  else if (opName.contains("erf"))
    fn = "erff";
  else if (opName.contains("round"))
    fn = "roundf";

  Type elemType = isTensor
                      ? cast<RankedTensorType>(result.getType()).getElementType()
                      : result.getType();
  auto cudaType = getCUDAType(elemType);

  // The `fn` names above are the f32 (single-precision) variants (sqrtf, expf,
  // ...). For f64 operands use the double-precision variants (sqrt, exp, ...)
  // otherwise we silently truncate to float precision. All our names end in
  // 'f', so stripping it yields the double variant.
  if (elemType.isF64() && fn != "unknown" && !fn.empty() && fn.back() == 'f')
    fn.pop_back();

  // f32 exp2 -> FTZ ex2.approx helper (see __triton_ex2f above). f64 stays
  // "exp2" (library); only the single-precision path hits ptxas's denormal
  // guard, so only it needs rerouting.
  if (fn == "exp2f")
    fn = "__triton_ex2f";

  // abs is type-polymorphic: math.absi (integer) and math.absf (float, incl
  // fp8). Mapping every abs to fabsf is wrong for integers (loses precision for
  // |v|>2^24) and for fp8 (packed byte). Build a per-element expression keyed
  // on the element type instead of a single function name.
  bool isAbs = opName.contains("abs");
  // math.fma is ternary (a*b+c) — handled separately below since the unary
  // dispatch only reads operand 0. Use fmaf for f32, fma for f64.
  bool isFma = opName.contains("fma");
  if (isFma) {
    std::string fmaFn = elemType.isF64() ? "fma" : "fmaf";
    if (isTensor) {
      int nElems = getElemsPerThread(result);
      Value o0 = op->getOperand(0), o1 = op->getOperand(1),
            o2 = op->getOperand(2);
      auto build = [=](const std::string &idx) -> std::string {
        std::string a = getElemExpr(o0, idx);
        std::string b = getElemExpr(o1, idx);
        std::string c = getElemExpr(o2, idx);
        return fmaFn + "(" + a + ", " + b + ", " + c + ")";
      };
      emitOrDeferElementwise(result, op, nElems, cudaType, build);
    } else {
      auto var = newVar("m");
      valueToVar[result] = var;
      std::string a = getVar(op->getOperand(0));
      std::string b = getVar(op->getOperand(1));
      std::string c = getVar(op->getOperand(2));
      emit("const " + cudaType + " " + var + " = " + fmaFn + "(" + a + ", " +
           b + ", " + c + ");");
    }
    return;
  }
  bool isFp8 =
      isa<Float8E4M3FNType>(elemType) || isa<Float8E5M2Type>(elemType);
  bool isInt = elemType.isIntOrIndex();
  auto absExpr = [=](const std::string &e) -> std::string {
    if (isFp8)
      // fp8 (e4m3/e5m2): sign bit is the MSB of the storage byte; clear it.
      return "(" + cudaType + ")(((unsigned char)(" + e + ")) & 0x7f)";
    if (isInt)
      // integer abs: avoid float round-trip through fabsf.
      return "((" + e + ") < 0 ? -(" + e + ") : (" + e + "))";
    // floating point: use double-precision fabs for f64.
    if (elemType.isF64())
      return "fabs(" + e + ")";
    return "fabsf(" + e + ")";
  };

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    Value o0 = op->getOperand(0);
    auto build = [=](const std::string &idx) -> std::string {
      std::string elemE = getElemExpr(o0, idx);
      if (isAbs)
        return absExpr(elemE);
      return fn + "(" + elemE + ")";
    };
    emitOrDeferElementwise(result, op, nElems, cudaType, build);
  } else {
    auto var = newVar("m");
    valueToVar[result] = var;
    auto srcVar = getVar(src);
    if (isAbs)
      emit("const " + cudaType + " " + var + " = " + absExpr(srcVar) + ";");
    else
      emit("const " + cudaType + " " + var + " = " + fn + "(" + srcVar + ");");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// tt.elementwise_inline_asm — run a user PTX asm block over packed elements.
// Mirrors the PTX backend's ElementwiseInlineAsmOpConversion: the asm sees
// `packed_element` elements at a time; sub-32-bit elements are packed into
// 32/64-bit registers; outputs come first in the `$N` placeholders, inputs
// after. We emit an `asm volatile(...)` block per chunk and pack/unpack the
// register operands. Placeholders are rewritten `$N` -> `%N` for nvcc.
// ═══════════════════════════════════════════════════════════════════════
void CUDACodeGen::emitElementwiseInlineAsm(tt::ElementwiseInlineAsmOp op) {
  int pack = op.getPackedElement();
  if (pack < 1)
    pack = 1;

  // Parse the comma-separated constraint string into trimmed tokens.
  SmallVector<std::string> tokens;
  {
    std::string cons = op.getConstraints().str();
    std::string cur;
    for (char c : cons) {
      if (c == ',') {
        tokens.push_back(cur);
        cur.clear();
      } else if (c != ' ' && c != '\t' && c != '\n') {
        cur.push_back(c);
      }
    }
    tokens.push_back(cur);
  }
  // Drop a trailing empty token (constraints like "=r,=r," produce one).
  while (!tokens.empty() && tokens.back().empty())
    tokens.pop_back();

  auto isOutputTok = [](const std::string &t) {
    return !t.empty() && (t[0] == '=' || t[0] == '+');
  };
  auto letterOf = [](const std::string &t) -> char {
    // last alphabetic char (skip leading '=','+').
    for (int i = (int)t.size() - 1; i >= 0; i--)
      if (isalpha((unsigned char)t[i]))
        return t[i];
    return 'r';
  };
  auto regBytesOf = [](char l) -> int {
    switch (l) {
    case 'l':
    case 'd':
      return 8;
    case 'h':
      return 2;
    default: // 'r', 'f'
      return 4;
    }
  };
  auto regCType = [](char l) -> std::string {
    switch (l) {
    case 'f':
      return "float";
    case 'd':
      return "double";
    case 'l':
      return "unsigned long long";
    case 'h':
      return "unsigned short";
    default:
      return "unsigned int";
    }
  };

  // Helpers to classify an element type.
  auto elemBytesOf = [&](Type t) -> int {
    if (isa<tt::PointerType>(t))
      return 8;
    return getTypeSizeInBytes(t);
  };

  // Translate the asm string: `$N` -> `%N`. Literal `$$` (rare) -> `$`.
  // Build the asm string for embedding in a C++ string literal. Placeholders
  // `$N` -> `%N` (nvcc syntax); literal `%` -> `%%`. The asm may span multiple
  // lines and contain `//` comments and quotes, so escape newlines/quotes/
  // backslashes/tabs for the C++ string literal (otherwise the literal breaks).
  std::string asmStr = op.getAsmString().str();
  std::string asmT;
  for (size_t i = 0; i < asmStr.size(); i++) {
    char c = asmStr[i];
    if (c == '$') {
      if (i + 1 < asmStr.size() && asmStr[i + 1] == '$') {
        asmT.push_back('$'); // literal $
        i++;
      } else {
        asmT.push_back('%');
      }
    } else if (c == '%') {
      asmT += "%%"; // escape literal percent for nvcc inline asm
    } else if (c == '\\') {
      asmT += "\\\\";
    } else if (c == '"') {
      asmT += "\\\"";
    } else if (c == '\n') {
      asmT += "\\n";
    } else if (c == '\t') {
      asmT += "\\t";
    } else if (c == '\r') {
      // drop carriage returns
    } else {
      asmT.push_back(c);
    }
  }

  // Number of valid elements per thread (from the first result, or the first
  // operand if there are no results).
  int nElems = 1;
  if (op->getNumResults() > 0)
    nElems = getElemsPerThread(op->getResult(0));
  else if (op->getNumOperands() > 0)
    nElems = getElemsPerThread(op->getOperand(0));
  if (nElems < 1)
    nElems = 1;

  // Per-result element type + output var arrays.
  struct PerVal {
    Type elemTy;
    int elemBytes;
    bool isTensor;
    bool isFloat;
    bool isF32;
    bool isF64;
    bool isPtr;
    std::string var;
  };
  auto describe = [&](Value v) -> PerVal {
    PerVal p;
    p.isTensor = isa<RankedTensorType>(v.getType());
    p.elemTy = p.isTensor
                   ? cast<RankedTensorType>(v.getType()).getElementType()
                   : v.getType();
    p.isPtr = isa<tt::PointerType>(p.elemTy);
    p.elemBytes = elemBytesOf(p.elemTy);
    p.isF32 = p.elemTy.isF32();
    p.isF64 = p.elemTy.isF64();
    p.isFloat = isa<FloatType>(p.elemTy);
    return p;
  };

  SmallVector<PerVal> results;
  for (auto res : op->getResults()) {
    PerVal p = describe(res);
    p.var = newVar("ia");
    valueToVar[res] = p.var;
    emit(getCUDAType(p.elemTy) + " " + p.var + "[" + std::to_string(nElems) +
         "];");
    results.push_back(p);
  }
  SmallVector<PerVal> operands;
  for (auto arg : op->getOperands())
    operands.push_back(describe(arg));

  // Build the register plan: number of registers for each result/operand, and
  // their constraint tokens (in $-placeholder order: outputs then inputs).
  auto regsForVal = [&](const PerVal &p, char letter) -> int {
    int rb = regBytesOf(letter);
    int epr = std::max(rb / std::max(p.elemBytes, 1), 1);
    epr = std::min(epr, pack);
    return std::max(pack / std::max(epr, 1), 1);
  };

  // Sanity: separate output and input constraint tokens.
  SmallVector<std::string> outToks, inToks;
  for (auto &t : tokens)
    (isOutputTok(t) ? outToks : inToks).push_back(t);

  // Emit one asm block per pack-chunk.
  for (int chunk = 0; chunk < nElems; chunk += pack) {
    emit("{");
    indent();

    // --- input registers ---
    SmallVector<std::string> inBindings; // "\"r\"(var)"
    int inTokIdx = 0;
    for (auto &p : operands) {
      char letter = inTokIdx < (int)inToks.size() ? letterOf(inToks[inTokIdx])
                                                  : 'r';
      int rb = regBytesOf(letter);
      int epr = std::max(rb / std::max(p.elemBytes, 1), 1);
      epr = std::min(epr, pack);
      int numRegs = std::max(pack / std::max(epr, 1), 1);
      Value v = op->getOperand(&p - &operands[0]);
      for (int rIdx = 0; rIdx < numRegs; rIdx++) {
        std::string rtype = regCType(letter);
        std::string rv = newVar("iar");
        // Assemble the register value from `epr` consecutive elements.
        std::string valExpr;
        if (letter == 'f' || letter == 'd') {
          int ai = std::min(chunk + rIdx * epr, nElems - 1);
          valExpr = getElemExpr(v, std::to_string(ai));
        } else if (p.isPtr) {
          int ai = std::min(chunk + rIdx * epr, nElems - 1);
          valExpr = "(unsigned long long)(" +
                    getElemExpr(v, std::to_string(ai)) + ")";
        } else if (p.isFloat && (p.isF32 || p.isF64)) {
          int ai = std::min(chunk + rIdx * epr, nElems - 1);
          std::string e = getElemExpr(v, std::to_string(ai));
          if (p.isF32)
            valExpr = "__float_as_uint(" + e + ")";
          else
            valExpr = "__double_as_longlong(" + e + ")";
        } else if (p.isFloat && p.elemBytes == 2) {
          // f16/bf16: pack `epr` BIT PATTERNS into the register, low-to-high.
          std::string asBits = p.elemTy.isBF16() ? "__bfloat16_as_ushort"
                                                 : "__half_as_ushort";
          for (int k = 0; k < epr; k++) {
            int ai = std::min(chunk + rIdx * epr + k, nElems - 1);
            std::string e = "((" + rtype + ")" + asBits + "(" +
                            getElemExpr(v, std::to_string(ai)) + "))";
            if (k > 0)
              e = "(" + e + " << " + std::to_string(k * 16) + ")";
            valExpr = valExpr.empty() ? e : (valExpr + " | " + e);
          }
        } else if (p.isFloat) {
          llvm::report_fatal_error(
              "CUDA emitter: inline asm packing of fp8 operands not supported");
        } else {
          // integer element: pack `epr` of them into the register, low-to-high.
          std::string mask =
              p.elemBytes >= rb
                  ? ""
                  : (" & 0x" +
                     [&]() {
                       std::string h;
                       unsigned bits = p.elemBytes * 8;
                       // build (1<<bits)-1 as hex
                       unsigned long long m =
                           (bits >= 64) ? ~0ull : ((1ull << bits) - 1);
                       char buf[32];
                       snprintf(buf, sizeof(buf), "%llx", m);
                       h = buf;
                       return h;
                     }());
          for (int k = 0; k < epr; k++) {
            int ai = std::min(chunk + rIdx * epr + k, nElems - 1);
            std::string e = "((" + rtype + ")(" +
                            getElemExpr(v, std::to_string(ai)) + ")" + mask +
                            ")";
            if (k > 0)
              e = "(" + e + " << " + std::to_string(k * p.elemBytes * 8) + ")";
            valExpr = valExpr.empty() ? e : (valExpr + " | " + e);
          }
        }
        emit(rtype + " " + rv + " = " + valExpr + ";");
        std::string tok = inTokIdx < (int)inToks.size() ? inToks[inTokIdx]
                                                        : std::string("r");
        inBindings.push_back("\"" + tok + "\"(" + rv + ")");
        inTokIdx++;
      }
    }

    // --- output registers ---
    SmallVector<std::string> outBindings;
    SmallVector<std::tuple<std::string, int, int, char>> outUnpack;
    // (regVar, resultIdx, firstElemAbsIdx, letter)
    int outTokIdx = 0;
    for (int ri = 0; ri < (int)results.size(); ri++) {
      auto &p = results[ri];
      char letter = outTokIdx < (int)outToks.size() ? letterOf(outToks[outTokIdx])
                                                    : 'r';
      int rb = regBytesOf(letter);
      int epr = std::max(rb / std::max(p.elemBytes, 1), 1);
      epr = std::min(epr, pack);
      int numRegs = std::max(pack / std::max(epr, 1), 1);
      for (int rIdx = 0; rIdx < numRegs; rIdx++) {
        std::string rtype = regCType(letter);
        std::string rv = newVar("iao");
        emit(rtype + " " + rv + ";");
        std::string tok = outTokIdx < (int)outToks.size()
                              ? outToks[outTokIdx]
                              : std::string("=r");
        outBindings.push_back("\"" + tok + "\"(" + rv + ")");
        outUnpack.push_back({rv, ri, chunk + rIdx * epr, letter});
        outTokIdx++;
      }
    }

    // --- the asm block itself ---
    std::string line = "asm volatile(\"" + asmT + "\"";
    line += " : ";
    for (size_t i = 0; i < outBindings.size(); i++)
      line += (i ? ", " : "") + outBindings[i];
    line += " : ";
    for (size_t i = 0; i < inBindings.size(); i++)
      line += (i ? ", " : "") + inBindings[i];
    line += ");";
    emit(line);

    // --- unpack outputs into result arrays ---
    for (auto &u : outUnpack) {
      std::string rv = std::get<0>(u);
      int ri = std::get<1>(u);
      int firstIdx = std::get<2>(u);
      char letter = std::get<3>(u);
      auto &p = results[ri];
      int rb = regBytesOf(letter);
      int epr = std::max(rb / std::max(p.elemBytes, 1), 1);
      epr = std::min(epr, pack);
      for (int k = 0; k < epr; k++) {
        int ai = firstIdx + k;
        if (ai >= nElems)
          break;
        std::string ctype = getCUDAType(p.elemTy);
        std::string val;
        if (letter == 'f' || letter == 'd') {
          val = rv;
        } else if (p.isFloat && p.isF32) {
          val = "__uint_as_float(" + rv + ")";
        } else if (p.isFloat && p.isF64) {
          val = "__longlong_as_double(" + rv + ")";
        } else if (p.isFloat && p.elemBytes == 2) {
          // f16/bf16: bit-reinterpret the packed 16-bit half — a numeric
          // (ctype)(reg) cast would CONVERT the integer value instead.
          std::string shifted =
              k == 0 ? rv : ("(" + rv + " >> " + std::to_string(k * 16) + ")");
          std::string fromBits = p.elemTy.isBF16() ? "__ushort_as_bfloat16"
                                                   : "__ushort_as_half";
          val = fromBits + "((unsigned short)(" + shifted + "))";
        } else if (p.isFloat) {
          llvm::report_fatal_error(
              "CUDA emitter: inline asm unpacking of fp8 results not supported");
        } else {
          // integer element
          std::string shifted =
              k == 0 ? rv
                     : ("(" + rv + " >> " + std::to_string(k * p.elemBytes * 8) +
                        ")");
          val = "(" + ctype + ")(" + shifted + ")";
        }
        emit(p.var + "[" + std::to_string(ai) + "] = (" + ctype + ")(" + val +
             ");");
      }
    }

    dedent();
    emit("}");
  }

  // Scalar (non-tensor) results were staged through a 1-element array for the
  // asm plumbing above; rebind them as true scalars so downstream consumers
  // (e.g. scalar tt.store emitting `*ptr = var;`) get a value, not an array.
  for (auto [idx, p] : llvm::enumerate(results)) {
    Value res = op->getResult(idx);
    if (p.isTensor)
      continue;
    auto sv = newVar("ias");
    emit("const " + getCUDAType(p.elemTy) + " " + sv + " = " + p.var + "[0];");
    valueToVar[res] = sv;
    scalarValues.insert(res);
  }
}

// tt.map_elementwise: apply a scalar region per group of `pack` consecutive
// per-thread elements. Entry block args are ordered src-major
// (src0[0..pack), src1[0..pack), ...); return operands are result-major
// (res0[0..pack), res1[0..pack), ...). The region may contain an acyclic
// cf-dialect CFG (e.g. early-return compare lowers to cond_br with a
// multi-predecessor merge block) — emit it by recursive if-conversion,
// duplicating merge blocks at each branch site (correct for tiny scalar
// regions; only true back-edges are rejected).
void CUDACodeGen::emitMapElementwise(tt::MapElementwiseOp op) {
  int pack = op.getPack();
  if (pack < 1)
    pack = 1;
  int nElems = getElemsPerThread(op->getResult(0));
  int nGroups = nElems / pack;

  // Declare result arrays.
  SmallVector<std::string> resVars;
  for (Value res : op->getResults()) {
    auto rtt = cast<RankedTensorType>(res.getType());
    auto cudaType = getCUDAType(rtt.getElementType());
    auto var = newVar("map");
    valueToVar[res] = var;
    resVars.push_back(var);
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
  }

  Block &entry = op.getScalarOp().front();
  unsigned nSrcs = op.getSrcs().size();

  // Recursive if-conversion over the region CFG. `path` detects back-edges
  // (true loops) on the current DFS path only, so diamond merges are allowed
  // (the merge block is duplicated per predecessor).
  llvm::SmallPtrSet<Block *, 8> path;
  std::function<void(Block *, int)> walk = [&](Block *block, int g) {
    if (emitFailed)
      return;
    if (!path.insert(block).second) {
      emitFailed = true;
      emitErrorMsg = "CUDA emitter: tt.map_elementwise region contains a "
                     "cf-dialect loop (back-edge) — unsupported";
      return;
    }
    for (auto &inner : block->without_terminator()) {
      emitOp(&inner);
      if (emitFailed) {
        path.erase(block);
        return;
      }
    }
    Operation *term = block->getTerminator();
    if (auto ret = dyn_cast<tt::MapElementwiseReturnOp>(term)) {
      for (unsigned r = 0; r < op->getNumResults(); r++)
        for (int j = 0; j < pack; j++)
          emit(resVars[r] + "[" + std::to_string(g * pack + j) +
               "] = " + getVar(ret->getOperand(r * pack + j)) + ";");
    } else if (auto br = dyn_cast<mlir::cf::BranchOp>(term)) {
      assignBlockArgs(br.getDest(), br.getDestOperands());
      walk(br.getDest(), g);
    } else if (auto cbr = dyn_cast<mlir::cf::CondBranchOp>(term)) {
      std::string cond = getVar(cbr.getCondition());
      emit("if (" + cond + ") {");
      indent();
      assignBlockArgs(cbr.getTrueDest(), cbr.getTrueDestOperands());
      walk(cbr.getTrueDest(), g);
      dedent();
      emit("} else {");
      indent();
      assignBlockArgs(cbr.getFalseDest(), cbr.getFalseDestOperands());
      walk(cbr.getFalseDest(), g);
      dedent();
      emit("}");
    } else {
      emitFailed = true;
      emitErrorMsg = "CUDA emitter: unsupported terminator '" +
                     term->getName().getStringRef().str() +
                     "' in tt.map_elementwise region";
    }
    path.erase(block);
  };

  // Unroll groups: bind entry args to source elements, emit the region body.
  for (int g = 0; g < nGroups && !emitFailed; g++) {
    emit("{ // map_elementwise group " + std::to_string(g));
    indent();
    for (unsigned s = 0; s < nSrcs; s++) {
      Value src = op.getSrcs()[s];
      auto srcRtt = cast<RankedTensorType>(src.getType());
      auto srcType = getCUDAType(srcRtt.getElementType());
      for (int j = 0; j < pack; j++) {
        BlockArgument arg = entry.getArgument(s * pack + j);
        auto var = newVar("marg");
        valueToVar[arg] = var;
        emit("const " + srcType + " " + var + " = " +
             getElemExpr(src, std::to_string(g * pack + j)) + ";");
      }
    }
    walk(&entry, g);
    dedent();
    emit("}");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Stubs for remaining ops — will be ported incrementally
// ═══════════════════════════════════════════════════════════════════════

void CUDACodeGen::emitMakeRange(tt::MakeRangeOp op) {
  auto result = op.getResult();
  auto rtt = cast<RankedTensorType>(result.getType());
  int start = op.getStart();
  int nElems = getElemsPerThread(rtt);
  auto var = newVar("rng");
  valueToVar[result] = var;

  // Per-register values as f(register, lane, warp). Uses the split-layout-aware
  // coordinate table so native non-pow2 WGMMA N (omni-style n80) ranges work.
  auto tbl = getRegCoordTable(rtt);
  const auto &laneBases = tbl.laneBases;
  const auto &warpBases = tbl.warpBases;
  const auto &blockBases = tbl.blockBases;

  emit("int " + var + "[" + llvm::Twine(nElems).str() + "];");
  emit("{");
  indent();

  for (int i = 0; i < nElems; i++) {
    int regBase = tbl.regCoords[i][0]; // dim0 coordinate at lane=0, warp=0

    // Build runtime expression: regBase + sum(lane_bit * lane_delta) + sum(warp_bit * warp_delta) + start
    std::string expr = std::to_string(regBase + start);
    for (size_t lb = 0; lb < laneBases.size(); lb++) {
      int delta = laneBases[lb][0]; // dim0 contribution from this lane bit
      if (delta != 0)
        expr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
    }
    for (size_t wb = 0; wb < warpBases.size(); wb++) {
      int delta = warpBases[wb][0]; // dim0 contribution from this warp bit
      if (delta != 0)
        expr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
    }
    // CGA: each CTA in the cluster owns a slice of the range (CTASplitNum>1);
    // the block bases carry the per-CTA global offset.
    for (size_t bb = 0; bb < blockBases.size(); bb++) {
      int delta = blockBases[bb][0];
      if (delta != 0)
        expr += " + ((_cta_rank >> " + std::to_string(bb) + ") & 1) * " +
                std::to_string(delta);
    }
    emit(var + "[" + std::to_string(i) + "] = " + expr + ";");
  }

  dedent();
  emit("}");
}

// The remaining op implementations are stubs — to be ported from Python.
// Each stub registers the result value and emits a TODO comment.

void CUDACodeGen::emitSplat(tt::SplatOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  auto srcVar = getVar(src);

  // Don't materialize array — just alias the scalar source.
  // Downstream ops use getElemExpr() which returns the scalar directly.
  valueToVar[result] = srcVar;
  scalarValues.insert(result);
}

void CUDACodeGen::emitUnsplat(tt::UnsplatOp op) {
  // unsplat converts a single-element tensor to a scalar. The source tensor is
  // replicated across all lanes (1 element per thread), so element [0] on this
  // thread is the value (mirrors the LLVM lowering: replaceOp with scrVals[0]).
  auto result = op.getResult();
  auto src = op.getSrc();
  if (scalarValues.contains(src)) {
    valueToVar[result] = getVar(src);
    scalarValues.insert(result);
    return;
  }
  auto var = newVar("unspl");
  auto cudaType = getCUDAType(result.getType());
  emit("const " + cudaType + " " + var + " = " + getElemExpr(src, "0") + ";");
  valueToVar[result] = var;
  scalarValues.insert(result);
}

llvm::SmallVector<int> CUDACodeGen::broadcastRegMapping(RankedTensorType srcRtt,
                                                        RankedTensorType dstRtt,
                                                        int srcN, int nElems) {
  // Native non-pow2 WGMMA N (omni-style n80): LinearLayout is non-representable,
  // so match coordinates via the split-layout-aware register-coordinate table.
  // Array index == register index on both sides (no LL-compaction subtlety,
  // since the mma D-fragment has no broadcast registers).
  if (llvm::any_of(srcRtt.getShape(),
                   [](int64_t d) { return !llvm::isPowerOf2_64(d); }) ||
      llvm::any_of(dstRtt.getShape(),
                   [](int64_t d) { return !llvm::isPowerOf2_64(d); })) {
    auto srcTbl = getRegCoordTable(srcRtt);
    auto dstTbl = getRegCoordTable(dstRtt);
    auto srcShape = srcRtt.getShape();
    std::map<llvm::SmallVector<int>, int> srcCoordToReg;
    for (int i = 0; i < srcN && i < (int)srcTbl.regCoords.size(); i++)
      srcCoordToReg.insert({srcTbl.regCoords[i], i}); // keep first occurrence
    llvm::SmallVector<int> dstToSrc(nElems, 0);
    for (int i = 0; i < nElems && i < (int)dstTbl.regCoords.size(); i++) {
      llvm::SmallVector<int> srcKey;
      for (size_t d = 0; d < dstTbl.regCoords[i].size(); d++)
        srcKey.push_back(dstTbl.regCoords[i][d] %
                         std::max((int64_t)1, srcShape[d]));
      auto it = srcCoordToReg.find(srcKey);
      dstToSrc[i] = (it != srcCoordToReg.end()) ? it->second : 0;
    }
    return dstToSrc;
  }

  // For each dst register, find which src register has the same coordinates
  // on the non-broadcast dimensions. Broadcast dims have src.shape[d] == 1,
  // so the dst coord is taken modulo src.shape[d].
  auto dstLL = ttg::toLinearLayout(dstRtt);
  auto srcLL = ttg::toLinearLayout(srcRtt);
  auto kReg = mlir::StringAttr::get(dstRtt.getContext(), "register");
  auto kLane = mlir::StringAttr::get(dstRtt.getContext(), "lane");
  auto kWarp = mlir::StringAttr::get(dstRtt.getContext(), "warp");
  auto kBlock = mlir::StringAttr::get(dstRtt.getContext(), "block");
  auto srcShape = srcRtt.getShape();

  // The emitter's per-thread arrays come in two conventions, distinguished by
  // their size relative to the raw LinearLayout register count:
  //  - COMPACT (size < raw regs): broadcast (zero-basis) registers map to
  //    duplicate coordinates and the producer deduped them in first-occurrence
  //    order. Array index == compact (dedup) index.
  //  - RAW (size == raw regs): the producer kept one entry per LL register,
  //    with broadcast duplicates filled in. Array index == LL register index.
  // Replicate the matching enumeration on both sides.
  std::map<SmallVector<int>, int> srcCoordToReg;
  int srcLLRegs = srcLL.getInDimSize(kReg);
  bool srcCompacted = srcN < srcLLRegs;
  int srcCompact = 0;
  for (int i = 0; i < srcLLRegs && srcCompact < srcN; i++) {
    auto coords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<int> key;
    for (auto &c : coords)
      key.push_back(c.second);
    if (srcCoordToReg.insert({key, srcCompacted ? srcCompact : i}).second)
      srcCompact++;
  }

  SmallVector<int> dstToSrc(nElems, 0);
  int dstLLRegs = dstLL.getInDimSize(kReg);
  bool dstCompacted = nElems < dstLLRegs;
  std::map<SmallVector<int>, int> seenDst;
  int dstCompact = 0;
  for (int i = 0; i < dstLLRegs && dstCompact < nElems; i++) {
    auto dstCoords =
        dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<int> dstKey, srcKey;
    for (size_t d = 0; d < dstCoords.size(); d++) {
      dstKey.push_back(dstCoords[d].second);
      srcKey.push_back(dstCoords[d].second %
                       std::max((int64_t)1, srcShape[d]));
    }
    if (dstCompacted && !seenDst.insert({dstKey, dstCompact}).second)
      continue; // duplicate dst register (broadcast) — already mapped
    auto it = srcCoordToReg.find(srcKey);
    dstToSrc[dstCompacted ? dstCompact : i] =
        (it != srcCoordToReg.end()) ? it->second : 0;
    dstCompact++;
  }
  return dstToSrc;
}

void CUDACodeGen::emitBroadcast(tt::BroadcastOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  auto rtt = cast<RankedTensorType>(result.getType());
  auto srcRtt = cast<RankedTensorType>(src.getType());
  int nElems = getElemsPerThread(rtt);
  int srcN = getElemsPerThread(src);
  srcN = std::max(srcN, 1);

  // If source is scalar, broadcast is also scalar
  if (scalarValues.contains(src)) {
    valueToVar[result] = getVar(src);
    scalarValues.insert(result);
    return;
  }

  // For deferred addptr source: broadcast the offsets and keep deferred
  {
    auto deferIt = deferredAddPtr.find(src);
    if (deferIt != deferredAddPtr.end()) {
      std::string base = deferIt->second.first;
      std::string srcOff = deferIt->second.second;
      // Broadcast the offset array (int, not __half*)
      auto bcastOff = newVar("boff");
      auto var2 = newVar("bcast");
      valueToVar[result] = var2;

      // Compute broadcast index expression (heuristic). The simple per-dim
      // modulo/divide formula assumes the default register order (last dim
      // fastest). Validate it against the LinearLayout ground truth below and
      // fall back to a per-element table when it disagrees (e.g. blocked
      // layouts with order=[0,1] where dim0 registers are fastest).
      bool hDiv = false;
      int hK = srcN;
      auto dstEnc2 = rtt.getEncoding();
      if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(dstEnc2)) {
        if (rtt.getRank() >= 2) {
          auto spt1 = blocked.getSizePerThread()[1];
          auto tpw1 = blocked.getThreadsPerWarp()[1];
          auto wpc1 = blocked.getWarpsPerCTA()[1];
          int totalT1 = tpw1 * wpc1;
          int reps1 = std::max((int64_t)1, rtt.getShape()[1] / (totalT1 * spt1));
          int elemsInner = spt1 * reps1;
          if (srcRtt.getShape()[1] == 1 && rtt.getShape()[1] > 1) {
            hDiv = true;
            hK = elemsInner;
          } else if (srcRtt.getShape()[0] == 1 && rtt.getShape()[0] > 1) {
            hDiv = false;
            hK = elemsInner;
          }
        }
      }
      auto dstToSrc = broadcastRegMapping(srcRtt, rtt, srcN, nElems);
      bool heuristicOk = true;
      for (int i = 0; i < nElems && heuristicOk; i++) {
        int idx = hDiv ? (hK ? i / hK : 0) : (hK ? i % hK : 0);
        if (idx >= srcN || idx != dstToSrc[i])
          heuristicOk = false;
      }

      emit("int " + bcastOff + "[" + std::to_string(nElems) + "];");
      if (heuristicOk) {
        std::string idxExpr = std::string("_i ") + (hDiv ? "/ " : "% ") +
                              std::to_string(hK);
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
             bcastOff + "[_i] = " + srcOff + "[" + idxExpr + "];");
      } else {
        // Per-element mapping from LinearLayout (non-default register order)
        for (int i = 0; i < nElems; i++)
          emit(bcastOff + "[" + std::to_string(i) + "] = " + srcOff + "[" +
               std::to_string(dstToSrc[i]) + "];");
      }
      deferredAddPtr[result] = {base, bcastOff};
      return;
    }
  }

  auto var = newVar("bcast");
  valueToVar[result] = var;

  auto srcVar = getVar(src);
  auto cudaType = getCUDAType(rtt.getElementType());

  emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");

  // Ground-truth dst-register -> src-register mapping from LinearLayout.
  SmallVector<int> dstToSrc = broadcastRegMapping(srcRtt, rtt, srcN, nElems);

  // For non-3D cases, prefer the compact heuristic expression (well-tested for
  // 2D row-major layouts), but only after VALIDATING it against the
  // LinearLayout ground truth: MMA/Linear encodings and blocked layouts with
  // non-default order (e.g. order=[0,1], dim0 registers fastest) have
  // register->coord mappings that are not a simple modulo of the register
  // index, and must use the exact per-element mapping instead.
  auto dstEnc = rtt.getEncoding();
  if (rtt.getRank() <= 2 &&
      !isa<ttg::NvidiaMmaEncodingAttr, ttg::LinearEncodingAttr>(dstEnc)) {
    bool hDiv = false;
    int hK = srcN;
    if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(dstEnc)) {
      if (rtt.getRank() >= 2) {
        auto spt1 = blocked.getSizePerThread()[1];
        auto tpw1 = blocked.getThreadsPerWarp()[1];
        auto wpc1 = blocked.getWarpsPerCTA()[1];
        int totalT1 = tpw1 * wpc1;
        int reps1 = std::max((int64_t)1, rtt.getShape()[1] / (totalT1 * spt1));
        int elemsInner = spt1 * reps1;
        if (srcRtt.getShape()[1] == 1 && rtt.getShape()[1] > 1) {
          hDiv = true;
          hK = elemsInner;
        } else if (srcRtt.getShape()[0] == 1 && rtt.getShape()[0] > 1) {
          hDiv = false;
          hK = elemsInner;
        }
      }
    }
    bool heuristicOk = true;
    for (int i = 0; i < nElems && heuristicOk; i++) {
      int idx = hDiv ? (hK ? i / hK : 0) : (hK ? i % hK : 0);
      if (idx >= srcN || idx != dstToSrc[i])
        heuristicOk = false;
    }
    if (heuristicOk) {
      std::string idxExpr =
          std::string("_i ") + (hDiv ? "/ " : "% ") + std::to_string(hK);
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
           var + "[_i] = " + srcVar + "[" + idxExpr + "];");
      return;
    }
    // Heuristic disagrees with LinearLayout — fall through to exact mapping.
  }

  // Exact mapping path (3D+, MMA/Linear encodings, or non-default reg order)
  emit("#pragma unroll");
  bool allSame = true;
  for (int i = 0; i < nElems; i++)
    if (dstToSrc[i] != dstToSrc[0]) allSame = false;

  if (allSame) {
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
         var + "[_i] = " + srcVar + "[" + std::to_string(dstToSrc[0]) + "];");
  } else {
    // Check if it's a simple modular pattern
    bool isSimpleMod = true;
    for (int i = 0; i < nElems && isSimpleMod; i++) {
      if (dstToSrc[i] != i % srcN) isSimpleMod = false;
    }
    bool isSimpleDiv = true;
    int divFactor = nElems / srcN;
    for (int i = 0; i < nElems && isSimpleDiv; i++) {
      if (dstToSrc[i] != i / divFactor) isSimpleDiv = false;
    }

    if (isSimpleMod) {
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
           var + "[_i] = " + srcVar + "[_i % " + std::to_string(srcN) + "];");
    } else if (isSimpleDiv && divFactor > 0) {
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
           var + "[_i] = " + srcVar + "[_i / " + std::to_string(divFactor) + "];");
    } else {
      // General case: emit per-element assignments
      for (int i = 0; i < nElems; i++) {
        emit(var + "[" + std::to_string(i) + "] = " + srcVar + "[" + std::to_string(dstToSrc[i]) + "];");
      }
    }
  }

}

void CUDACodeGen::emitExpandDims(tt::ExpandDimsOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  auto rtt = cast<RankedTensorType>(result.getType());
  auto srcRtt = cast<RankedTensorType>(src.getType());
  int nElems = getElemsPerThread(rtt);
  int srcN = getElemsPerThread(src);
  nElems = std::max(nElems, srcN);
  auto var = newVar("exp");
  valueToVar[result] = var;

  // Deferred pointer source: the source was never materialized as a real
  // array (only a (base, offset[]) / scalar-base record). Expand_dims is a
  // per-thread no-op, so propagate the deferred record instead of emitting an
  // alias to an undeclared placeholder variable.
  if (nElems == srcN) {
    if (auto it = deferredAddPtr.find(src); it != deferredAddPtr.end()) {
      deferredAddPtr[result] = it->second;
      return;
    }
    if (auto it = scalarBaseDeferred.find(src); it != scalarBaseDeferred.end()) {
      scalarBaseDeferred[result] = it->second;
      valueToVar[result] = getVar(src);
      return;
    }
    if (auto it = ptrBasedDeferred.find(src); it != ptrBasedDeferred.end()) {
      ptrBasedDeferred[result] = it->second;
      valueToVar[result] = getVar(src);
      return;
    }
  } else if (deferredAddPtr.count(src) || scalarBaseDeferred.count(src) ||
             ptrBasedDeferred.count(src)) {
    std::string msg = "[emit_cuda] expand_dims of deferred pointer with "
                      "element-count change is unsupported";
    op->emitError(msg);
    if (!emitFailed) {
      emitFailed = true;
      emitErrorMsg = msg;
    }
    return;
  }

  auto srcVar = getVar(src);
  auto cudaType = getCUDAType(rtt.getElementType());

  if (nElems == srcN) {
    emit(cudaType + "* " + var + " = " + srcVar + "; // expand_dims (alias)");
  } else {
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");

    // Use per-register coords to map dst register -> src register. Go through
    // getRegCoordTable (not raw toLinearLayout) so native non-pow2 WGMMA-N
    // tensors (omni-style n80) are handled via the pow2 split-layout instead of
    // aborting in the GF(2) LinearLayout machinery.
    int axis = op.getAxis();
    auto dstTbl = getRegCoordTable(rtt);
    auto srcTbl = getRegCoordTable(srcRtt);

    // Build src register → coord mapping (without the expanded dim)
    std::map<SmallVector<int>, int> srcCoordToReg;
    for (int i = 0; i < srcN && i < (int)srcTbl.regCoords.size(); i++) {
      SmallVector<int> key(srcTbl.regCoords[i].begin(),
                           srcTbl.regCoords[i].end());
      srcCoordToReg[key] = i;
    }

    // For each dst register, strip the expanded dim and find src register
    SmallVector<int> dstToSrc(nElems, 0);
    for (int i = 0; i < nElems && i < (int)dstTbl.regCoords.size(); i++) {
      auto &dstCoords = dstTbl.regCoords[i];
      SmallVector<int> srcKey;
      for (int d = 0; d < (int)dstCoords.size(); d++) {
        if (d != axis) srcKey.push_back(dstCoords[d]);
      }
      auto it = srcCoordToReg.find(srcKey);
      dstToSrc[i] = (it != srcCoordToReg.end()) ? it->second : 0;
    }

    // Emit mapping
    bool allSame = true;
    for (int i = 1; i < nElems; i++)
      if (dstToSrc[i] != dstToSrc[0]) { allSame = false; break; }

    if (allSame) {
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
           var + "[_i] = " + srcVar + "[" + std::to_string(dstToSrc[0]) + "];");
    } else {
      // Check simple patterns
      bool isSimpleMod = true;
      for (int i = 0; i < nElems; i++)
        if (dstToSrc[i] != i % srcN) { isSimpleMod = false; break; }

      if (isSimpleMod) {
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
             var + "[_i] = " + srcVar + "[_i % " + std::to_string(srcN) + "];");
      } else {
        for (int i = 0; i < nElems; i++)
          emit(var + "[" + std::to_string(i) + "] = " + srcVar + "[" + std::to_string(dstToSrc[i]) + "];");
      }
    }
  }
}

void CUDACodeGen::emitAddPtr(tt::AddPtrOp op) {
  auto result = op.getResult();
  auto ptr = op.getPtr();
  auto offset = op.getOffset();
  bool ptrIsTensor = isTensorValue(ptr);
  bool offIsTensor = isTensorValue(offset);
  auto var = newVar("ptr");
  valueToVar[result] = var;

  if (ptrIsTensor || offIsTensor) {
    bool ptrScalar = !ptrIsTensor || scalarValues.contains(ptr);
    bool offScalar = !offIsTensor || scalarValues.contains(offset);

    int nElems =
        std::max(ptrIsTensor ? getElemsPerThread(ptr) : 1,
                 offIsTensor ? getElemsPerThread(offset) : 1);

    // Get pointer element type
    Type resultElemType;
    if (auto rtt = dyn_cast<RankedTensorType>(result.getType()))
      resultElemType = rtt.getElementType();
    else
      resultElemType = result.getType();
    auto cudaType = getCUDAType(resultElemType);

    std::string ptrExpr = getElemExpr(ptr, "_i");
    std::string offExpr = getElemExpr(offset, "_i");

    if (ptrScalar && offScalar) {
      // Both scalar → result is scalar. Cast the RHS to the declared pointer
      // type: the base may be a differently-typed pointer (e.g. a char*
      // global-scratch base feeding an int8_t* TMA-descriptor pointer), and
      // C++ does not implicitly convert between distinct pointer types.
      scalarValues.insert(result);
      emit(cudaType + " " + var + " = (" + cudaType + ")(" + ptrExpr + " + " +
           offExpr + ");");
    } else if (ptrScalar && !offScalar) {
      // Scalar base + array offset → defer: don't materialize ptr array
      // Store (base, offset) and compute inline at load/store time
      // For chained addptr (ptr + offset where ptr is itself deferred),
      // we need to materialize the offset sum
      auto ptrDeferred = deferredAddPtr.find(ptr);
      if (ptrDeferred != deferredAddPtr.end()) {
        // Chained: (base + old_offset) + new_offset → base + (old_offset + new_offset)
        auto offVar = getVar(offset);
        auto combinedOff = newVar("off");
        emit("int " + combinedOff + "[" + std::to_string(nElems) + "];");
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
        emit("    " + combinedOff + "[_i] = " + ptrDeferred->second.second + "[_i] + " + offVar + "[_i];");
        deferredAddPtr[result] = {ptrDeferred->second.first, combinedOff};
        valueToVar[result] = var; // placeholder, not materialized
      } else {
        auto offVar = getVar(offset);
        deferredAddPtr[result] = {ptrExpr, offVar};
        valueToVar[result] = var; // placeholder, not materialized
      }
    } else if (scalarBaseDeferred.count(ptr)) {
      // ptr uses scalar-base (precomputed addresses + scalar byte-delta).
      auto sbi = scalarBaseDeferred[ptr];  // copy
      if (scalarValues.contains(offset)) {
        // Splat (uniform) offset. Derive result = base + off WITHOUT mutating
        // the base's delta in place: the base pointer may still be live (e.g. a
        // loop-invariant base reused to derive several temporary load/store
        // pointers inside the loop body — split_k reduction, gather epilogues).
        // The old `delta += off` corrupted every later use of that base. Emit a
        // FRESH delta instead; for genuine loop-carried advances (result yielded
        // back to the iter_arg) the scf.for yield handler copies this fresh
        // delta into the carried iterVar, so those stay correct too.
        auto offScalarVar = getVar(offset);
        std::string mulStr = (sbi.bpe == 1) ? "" : " * " + std::to_string(sbi.bpe);
        auto newDelta = newVar("iter");
        emit("int " + newDelta + " = " + sbi.deltaVar + " + (int)" +
             offScalarVar + mulStr + ";");
        sbi.deltaVar = newDelta;
        scalarBaseDeferred[result] = sbi;
        valueToVar[result] = newDelta; // alias
      } else {
        // Non-uniform offset: add to precomputed addresses in place. (Kept
        // in-place: loop-carried non-uniform advances rely on this — the yield
        // handler copies only the scalar delta, not the precompAddr array.)
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
        std::string mulStr = (sbi.bpe == 1) ? "" : " * " + std::to_string(sbi.bpe);
        emit("    " + sbi.precompAddr + "[_i] += (unsigned)(" + offExpr + mulStr + ");");
        scalarBaseDeferred[result] = sbi;
        valueToVar[result] = sbi.deltaVar; // alias
      }
    } else if (deferredAddPtr.count(ptr)) {
      // ptr is deferred (base, offset[N]) from loop iter_arg — chain offsets
      auto &[base, oldOff] = deferredAddPtr[ptr];
      auto combinedOff = newVar("off");
      emit("int " + combinedOff + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      emit("    " + combinedOff + "[_i] = " + oldOff + "[_i] + " + offExpr + ";");
      deferredAddPtr[result] = {base, combinedOff};
      valueToVar[result] = var; // placeholder
    } else {
      emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      emit("    " + var + "[_i] = " + ptrExpr + " + " + offExpr + ";");
    }
  } else {
    auto cudaType = getCUDAType(result.getType());
    auto ptrVar = getVar(ptr);
    auto offVar = getVar(offset);
    // Cast the RHS to the declared pointer type — the base pointer may have a
    // different C++ type (e.g. a char* global-scratch base producing an
    // int8_t* TMA-descriptor pointer), which C++ will not implicitly convert.
    emit(cudaType + " " + var + " = (" + cudaType + ")(" + ptrVar + " + " +
         offVar + ");");
  }
}

// Cache-modifier / eviction-policy PTX suffixes. Mirrors the PTX backend's
// LoadStoreOpToLLVM mapping: loads support .ca/.cg, stores support
// .wb/.cg/.cs/.wt, and both support .L1::evict_first/.L1::evict_last.
// When non-empty, loads/stores must go through inline PTX (a plain C
// dereference cannot carry a cache hint).
static std::string loadCacheSuffix(tt::CacheModifier cm,
                                   tt::EvictionPolicy ev) {
  std::string s;
  if (cm == tt::CacheModifier::CA)
    s += ".ca";
  else if (cm == tt::CacheModifier::CG)
    s += ".cg";
  if (ev == tt::EvictionPolicy::EVICT_FIRST)
    s += ".L1::evict_first";
  else if (ev == tt::EvictionPolicy::EVICT_LAST)
    s += ".L1::evict_last";
  return s;
}

static std::string storeCacheSuffix(tt::CacheModifier cm,
                                    tt::EvictionPolicy ev) {
  std::string s;
  if (cm == tt::CacheModifier::WB)
    s += ".wb";
  else if (cm == tt::CacheModifier::CG)
    s += ".cg";
  else if (cm == tt::CacheModifier::CS)
    s += ".cs";
  else if (cm == tt::CacheModifier::WT)
    s += ".wt";
  if (ev == tt::EvictionPolicy::EVICT_FIRST)
    s += ".L1::evict_first";
  else if (ev == tt::EvictionPolicy::EVICT_LAST)
    s += ".L1::evict_last";
  return s;
}

void CUDACodeGen::emitLoad(tt::LoadOp op) {
  auto result = op.getResult();
  auto ptr = op.getPtr();
  auto mask = op.getMask();
  auto other = op.getOther();

  bool isTensor = isa<RankedTensorType>(result.getType());

  // Check if this load can be fused with a subsequent local_alloc via cp.async.
  // Pattern: tt.load → local_alloc (with nvmma_shared encoding)
  if (isTensor && result.hasOneUse()) {
    auto *user = *result.user_begin();
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(user)) {
      auto memDescType = cast<ttg::MemDescType>(allocOp.getResult().getType());
      auto destEnc = memDescType.getEncoding();
      if (auto nvmmaEnc = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(destEnc)) {
        // Only fuse via cp.async if the destination is NOT transposed.
        // Transposed layouts (e.g. B tile with transposed=true) require
        // scattering elements across shared memory rows, which cp.async
        // cannot do — it only copies contiguous bytes.
        if (!nvmmaEnc.getTransposed()) {
          deferredCpAsyncLoads.insert(result);
          return;
        }
      }
    }
  }

  auto var = newVar("ld");
  valueToVar[result] = var;

  auto ptrVar = getVar(ptr);

  if (isTensor) {
    auto rtt = cast<RankedTensorType>(result.getType());
    int nElems = getElemsPerThread(rtt);
    auto cudaType = getCUDAType(rtt.getElementType());
    int elemBytes = getTypeSizeInBytes(rtt.getElementType());

    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");

    // Non-default cache modifier / eviction policy must go through inline
    // PTX: a plain C dereference cannot carry .ca/.cg/.L1::evict_* hints.
    std::string ldSfx = loadCacheSuffix(op.getCache(), op.getEvict());

    // Try vectorized load: for blocked encodings where the innermost dimension
    // has contiguous elements (sizePerThread[order[0]] > 1), generate 128-bit
    // vector loads instead of scalar loads. This dramatically reduces the
    // number of global load instructions (e.g., 4 LDG.128 vs 64 LDG.U8).
    auto enc = rtt.getEncoding();
    int vecElems = 0;
    if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(enc)) {
      auto order = blocked.getOrder();
      int innerDim = order[0];
      int sptInner = blocked.getSizePerThread()[innerDim];
      int vecBytes = sptInner * elemBytes;
      // Use vector loads for groups of 4+ bytes (int), up to 16 bytes (int4)
      if (vecBytes >= 4 && deferredAddPtr.count(ptr)) {
        // Clamp to 16 bytes (int4) max
        vecElems = sptInner;
        int loadBytes = vecElems * elemBytes;
        if (loadBytes > 16) {
          vecElems = 16 / elemBytes;
          loadBytes = 16;
        }
        // Clamp to the pointer's actual alignment/contiguity (and mask
        // alignment) from AxisInfo so we never emit an over-wide vector load
        // that is misaligned or that reads out-of-bounds tail elements when the
        // mask varies within the group. sizePerThread is a layout property, NOT
        // a guarantee of 16B base alignment or uniform masking.
        unsigned maxVec = getMaxVecWidth(ptr, mask);
        if (maxVec < (unsigned)vecElems) {
          // Round down to a power-of-2 element count that keeps vecBytes >= 4.
          int v = 1;
          while (v * 2 <= (int)maxVec && v * 2 * elemBytes <= 16)
            v *= 2;
          vecElems = (v * elemBytes >= 4) ? v : 0;
        }
      }
    }

    if (vecElems > 0 && deferredAddPtr.count(ptr)) {
      int vecBytes = vecElems * elemBytes;
      int nGroups = nElems / vecElems;
      std::string vecType = (vecBytes == 16) ? "int4" :
                            (vecBytes == 8)  ? "int2" : "int";
      auto &ptrInfo = deferredAddPtr[ptr];
      std::string basePtr = ptrInfo.first;
      std::string offArr = ptrInfo.second;
      emit("// Vectorized global load: " + std::to_string(nGroups) + " x " +
           std::to_string(vecBytes) + "B");
      emit("#pragma unroll");
      emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
      indent();
      std::string groupBase = "_g * " + std::to_string(vecElems);
      // Emits one vector-group load; uses inline PTX when a cache
      // modifier/eviction policy is present (vecBytes is 4/8/16 here).
      auto emitVecLd = [&](const std::string &gb) {
        std::string addr = basePtr + " + " + offArr + "[" + gb + "]";
        if (ldSfx.empty()) {
          emit("*(" + vecType + "*)(&" + var + "[" + gb + "]) = *(" + vecType +
               "*)(" + addr + ");");
          return;
        }
        std::string dst = "((uint32_t*)&" + var + "[" + gb + "])";
        int nU32 = vecBytes / 4;
        if (nU32 == 4)
          emit("asm(\"ld.global" + ldSfx +
               ".v4.b32 {%0, %1, %2, %3}, [%4];\" : \"=r\"(" + dst +
               "[0]), \"=r\"(" + dst + "[1]), \"=r\"(" + dst +
               "[2]), \"=r\"(" + dst + "[3]) : \"l\"(" + addr + "));");
        else if (nU32 == 2)
          emit("asm(\"ld.global" + ldSfx +
               ".v2.b32 {%0, %1}, [%2];\" : \"=r\"(" + dst +
               "[0]), \"=r\"(" + dst + "[1]) : \"l\"(" + addr + "));");
        else
          emit("asm(\"ld.global" + ldSfx + ".b32 %0, [%1];\" : \"=r\"(" + dst +
               "[0]) : \"l\"(" + addr + "));");
      };
      if (mask) {
        // Mask is uniform within each vector group for blocked encodings
        std::string maskE = getElemExpr(mask, groupBase);
        emit("if (" + maskE + ") {");
        indent();
        emitVecLd(groupBase);
        dedent();
        emit("} else {");
        indent();
        emit("#pragma unroll");
        emit("for (int _s = 0; _s < " + std::to_string(vecElems) + "; _s++)");
        if (other) {
          emit("    " + var + "[" + groupBase + " + _s] = " +
               getElemExpr(other, groupBase + " + _s") + ";");
        } else {
          // Zero-fill undef masked-off lanes (see scalar path comment).
          emit("    " + var + "[" + groupBase + " + _s] = (" + cudaType +
               ")0;");
        }
        dedent();
        emit("}");
      } else {
        emitVecLd(groupBase);
      }
      dedent();
      emit("}");
    } else {
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
      indent();
      std::string ptrDeref = getPtrElemExpr(ptr, "_i");
      // One scalar load; inline PTX when a cache modifier/eviction policy
      // is present (the C dereference cannot carry the hint).
      auto scalarLd = [&](const std::string &pre) {
        if (ldSfx.empty()) {
          emit(pre + var + "[_i] = " + ptrDeref + ";");
          return;
        }
        std::string addr = "&(" + ptrDeref + ")";
        std::string dst = "&" + var + "[_i]";
        if (elemBytes == 8)
          emit(pre + "asm(\"ld.global" + ldSfx +
               ".b64 %0, [%1];\" : \"=l\"(*(uint64_t*)" + dst + ") : \"l\"(" +
               addr + "));");
        else if (elemBytes == 4)
          emit(pre + "asm(\"ld.global" + ldSfx +
               ".b32 %0, [%1];\" : \"=r\"(*(uint32_t*)" + dst + ") : \"l\"(" +
               addr + "));");
        else if (elemBytes == 2)
          emit(pre + "asm(\"ld.global" + ldSfx +
               ".u16 %0, [%1];\" : \"=h\"(*(uint16_t*)" + dst + ") : \"l\"(" +
               addr + "));");
        else
          emit(pre + "{ uint16_t _lb; asm(\"ld.global" + ldSfx +
               ".u8 %0, [%1];\" : \"=h\"(_lb) : \"l\"(" + addr +
               ")); *(uint8_t*)" + dst + " = (uint8_t)_lb; }");
      };
      if (mask) {
        std::string maskE = getElemExpr(mask, "_i");
        if (other) {
          std::string otherE = getElemExpr(other, "_i");
          emit("if (" + maskE + ") {");
          scalarLd("    ");
          emit("} else {");
          emit("    " + var + "[_i] = " + otherE + ";");
          emit("}");
        } else {
          // No `other` operand: masked-off lanes are undef per tt.load
          // semantics. Zero-fill so garbage can never propagate (e.g. mx
          // scales 0xFF would upcast to NaN even when multiplied by 0).
          emit("if (" + maskE + ") {");
          scalarLd("    ");
          emit("} else {");
          emit("    " + var + "[_i] = (" + cudaType + ")0;");
          emit("}");
        }
      } else {
        scalarLd("");
      }
      dedent();
      emit("}");
    }
  } else {
    auto cudaType = getCUDAType(result.getType());
    emit(cudaType + " " + var + " = *" + ptrVar + ";");
  }
}

void CUDACodeGen::emitStore(tt::StoreOp op) {
  auto ptr = op.getPtr();
  auto val = op.getValue();
  auto mask = op.getMask();

  auto ptrVar = getVar(ptr);
  auto valVar = getVar(val);
  bool isTensor = isa<RankedTensorType>(val.getType());

  if (isTensor) {
    auto valRtt = cast<RankedTensorType>(val.getType());
    int nElems = getElemsPerThread(val);
    Type elemType = valRtt.getElementType();
    int elemBytes = getTypeSizeInBytes(elemType);
    bool valIsScalar = isScalarValue(val);

    // Non-default cache modifier / eviction policy (.wb/.cg/.cs/.wt /
    // .L1::evict_*) — appended to the PTX st.global opcode; the scalar
    // fallback must then also go through inline PTX.
    std::string stSfx = storeCacheSuffix(op.getCache(), op.getEvict());

    // Try vectorized store: blocked layout with contiguous inner dimension
    int vecWidth = 1;
    if (auto blk = dyn_cast<ttg::BlockedEncodingAttr>(valRtt.getEncoding())) {
      auto spt = blk.getSizePerThread();
      // 1D tensors (e.g. layer-norm's [BLOCK_SIZE] row) must vectorize too:
      // the load path already does, but gating the store on >= 2 dims left 1D
      // stores scalarized (32 STG.E instead of 4 STG.E.128), halving store
      // bandwidth on memory-bound elementwise/reduce kernels. order[0] and
      // spt[order[0]] are valid for rank-1 (order has size 1).
      if (spt.size() >= 1) {
        // Vectorize along the contiguous axis order[0] (fastest-varying in the
        // register linearization), NOT the last tensor axis. For the common
        // row-major case order[0] == last axis, so this matches the old
        // behavior; for transposed stores (order = [0, 1]) the contiguous axis
        // is the first axis and the last-axis spt is 1 (would scalarize).
        auto order = blk.getOrder();
        int innerSpt = spt[order[0]];
        // Maximum 16 bytes per vectorized store (uint4)
        int maxVec = 16 / elemBytes;
        vecWidth = std::min(innerSpt, maxVec);
        // Clamp to the pointer's actual alignment/contiguity (and mask
        // alignment) from AxisInfo so we never emit an over-wide vector store
        // that is misaligned or partially-masked (OOB tail elements).
        vecWidth = std::min<int>(vecWidth, (int)getMaxVecWidth(ptr, mask));
        // Must be power of 2 and >= 2
        while (vecWidth > 1 && (vecWidth & (vecWidth - 1))) vecWidth >>= 1;
        if (vecWidth < 2) vecWidth = 1;
      }
    }

    // Compute pointer address expression for element at _base index
    // For deferred addptr: (base + offset[_base]), for regular: ptrVar[_base]
    auto deferIt = deferredAddPtr.find(ptr);
    std::string ptrAddrBase; // expression for getting address at _base
    if (deferIt != deferredAddPtr.end()) {
      ptrAddrBase = "(void*)(" + deferIt->second.first + " + " + deferIt->second.second + "[_base])";
    } else {
      ptrAddrBase = "(void*)" + ptrVar + "[_base]";
    }

    // Helper to emit a predicated PTX store via setp + @p. Data registers are
    // _r0.._rN; 32-bit ("r") by default, 64-bit ("l") when wide is set.
    auto emitPtxStore = [&](const std::string &ptxOp, int nU32,
                            bool hasMask, bool wide = false) {
      const std::string dcon = wide ? "l" : "r";
      // Build reg list: {%N, %N+1, ...} for the data operands
      std::string regList, regConstraints;
      int argIdx = hasMask ? 2 : 1; // 0=pred(if masked), 1=addr, 2+=data (or 0=addr,1+=data)
      if (nU32 == 1) {
        regList = "%" + std::to_string(argIdx);
      } else {
        regList = "{";
        for (int i = 0; i < nU32; i++) {
          if (i) regList += ", ";
          regList += "%" + std::to_string(argIdx + i);
        }
        regList += "}";
      }
      // Build constraint string and args
      std::string args;
      if (hasMask) {
        // Use setp inside asm to create predicate from int32 input
        std::string addrIdx = "%1";
        std::string predIdx = "%0";
        emit("asm volatile(\"{.reg .pred p;\\n\\t"
             " setp.ne.b32 p, " + predIdx + ", 0;\\n\\t"
             " @p " + ptxOp + " [" + addrIdx + "], " + regList + ";}\\n\\t\"");
        // Operands: pred(r), addr(l), data regs(r)
        args = ":: \"r\"(_pred), \"l\"(" + ptrAddrBase + ")";
        for (int i = 0; i < nU32; i++)
          args += ", \"" + dcon + "\"(_r" + std::to_string(i) + ")";
        emit(args + ");");
      } else {
        std::string addrIdx = "%0";
        // Renumber reg list for non-predicated case
        regList = "";
        if (nU32 == 1) {
          regList = "%1";
        } else {
          regList = "{";
          for (int i = 0; i < nU32; i++) {
            if (i) regList += ", ";
            regList += "%" + std::to_string(1 + i);
          }
          regList += "}";
        }
        emit("asm volatile(\"" + ptxOp + " [" + addrIdx + "], " + regList + ";\"");
        args = ":: \"l\"(" + ptrAddrBase + ")";
        for (int i = 0; i < nU32; i++)
          args += ", \"" + dcon + "\"(_r" + std::to_string(i) + ")";
        emit(args + ");");
      }
    };

    if (vecWidth >= 2 && (elemType.isF16() || elemType.isBF16())) {
      // Vectorized store for f16/bf16 using inline PTX with predication
      int nGroups = nElems / vecWidth;
      int vecBytes = vecWidth * elemBytes;
      int nU32 = vecBytes / 4;
      if (nU32 < 1 || nU32 > 4) { vecWidth = 1; } // fallback

      if (vecWidth >= 2) {
        std::string ptxOp;
        if (nU32 == 1) ptxOp = "st.global" + stSfx + ".b32";
        else if (nU32 == 2) ptxOp = "st.global" + stSfx + ".v2.b32";
        else ptxOp = "st.global" + stSfx + ".v4.b32";

        emit("// Vectorized store via PTX: " + std::to_string(vecWidth) + " x " +
             getCUDAType(elemType) + " per store (" + std::to_string(nGroups) + " stores)");
        emit("#pragma unroll");
        emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
        indent();
        emit("int _base = _g * " + std::to_string(vecWidth) + ";");
        // Pack elements into uint32 registers
        for (int i = 0; i < nU32; i++) {
          int e0 = i * 2, e1 = i * 2 + 1;
          std::string ref0 = valIsScalar ? ("&" + valVar) : ("&" + valVar + "[_base+" + std::to_string(e0) + "]");
          std::string ref1 = valIsScalar ? ("&" + valVar) : ("&" + valVar + "[_base+" + std::to_string(e1) + "]");
          emit("uint32_t _r" + std::to_string(i) + " = "
               "*(uint16_t*)" + ref0 + " | "
               "((uint32_t)*(uint16_t*)" + ref1 + " << 16);");
        }
        if (mask) {
          auto maskVar = getVar(mask);
          std::string maskE = isScalarValue(mask) ? maskVar : (maskVar + "[_base + " + std::to_string(vecWidth - 1) + "]");
          emit("int _pred = (int)" + maskE + ";");
        }
        emitPtxStore(ptxOp, nU32, mask != nullptr);
        dedent();
        emit("}");
        return;
      }
    }
    if (vecWidth >= 4 && elemBytes == 1 && !valIsScalar &&
        (isa<Float8E4M3FNType>(elemType) || isa<Float8E5M2Type>(elemType) ||
         elemType.isInteger(8))) {
      // Vectorized store for fp8/int8: pack 4 bytes per uint32 register and
      // emit st.global.v{1,2,4}.b32. Matches the PTX backend's coalesced
      // 16-byte stores (8x STG.E.128) instead of per-byte scalar stores.
      int nGroups = nElems / vecWidth;
      int nU32 = vecWidth / 4; // vecWidth is power-of-2 >= 4 -> nU32 in {1,2,4}
      if (nU32 >= 1 && nU32 <= 4) {
        std::string ptxOp;
        if (nU32 == 1) ptxOp = "st.global" + stSfx + ".b32";
        else if (nU32 == 2) ptxOp = "st.global" + stSfx + ".v2.b32";
        else ptxOp = "st.global" + stSfx + ".v4.b32";

        emit("// Vectorized fp8 store via PTX: " + std::to_string(vecWidth) +
             " x " + getCUDAType(elemType) + " per store (" +
             std::to_string(nGroups) + " stores)");
        emit("#pragma unroll");
        emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
        indent();
        emit("int _base = _g * " + std::to_string(vecWidth) + ";");
        for (int i = 0; i < nU32; i++) {
          int b0 = i * 4, b1 = i * 4 + 1, b2 = i * 4 + 2, b3 = i * 4 + 3;
          emit("uint32_t _r" + std::to_string(i) + " = "
               "(uint32_t)*(uint8_t*)&" + valVar + "[_base+" + std::to_string(b0) + "] | "
               "((uint32_t)*(uint8_t*)&" + valVar + "[_base+" + std::to_string(b1) + "] << 8) | "
               "((uint32_t)*(uint8_t*)&" + valVar + "[_base+" + std::to_string(b2) + "] << 16) | "
               "((uint32_t)*(uint8_t*)&" + valVar + "[_base+" + std::to_string(b3) + "] << 24);");
        }
        if (mask) {
          auto maskVar = getVar(mask);
          std::string maskE = isScalarValue(mask) ? maskVar : (maskVar + "[_base + " + std::to_string(vecWidth - 1) + "]");
          emit("int _pred = (int)" + maskE + ";");
        }
        emitPtxStore(ptxOp, nU32, mask != nullptr);
        dedent();
        emit("}");
        return;
      }
    }
    if (vecWidth >= 2 && elemBytes == 4) {
      // Vectorized store for any 4-byte element (f32/i32) via inline PTX
      int nGroups = nElems / vecWidth;
      std::string ptxOp;
      if (vecWidth == 2) ptxOp = "st.global" + stSfx + ".v2.b32";
      else ptxOp = "st.global" + stSfx + ".v4.b32";

      emit("// Vectorized store via PTX: " + std::to_string(vecWidth) + " x " +
           getCUDAType(elemType) + " per store");
      emit("#pragma unroll");
      emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
      indent();
      emit("int _base = _g * " + std::to_string(vecWidth) + ";");
      // Bitcast element to uint32 for PTX
      for (int i = 0; i < vecWidth; i++) {
        std::string valE = valIsScalar ? valVar : (valVar + "[_base+" + std::to_string(i) + "]");
        if (elemType.isF32())
          emit("uint32_t _r" + std::to_string(i) + " = "
               "__float_as_uint(" + valE + ");");
        else
          emit("uint32_t _r" + std::to_string(i) + " = "
               "(uint32_t)(" + valE + ");");
      }
      if (mask) {
        auto maskVar = getVar(mask);
        std::string maskE = isScalarValue(mask) ? maskVar : (maskVar + "[_base + " + std::to_string(vecWidth - 1) + "]");
        emit("int _pred = (int)" + maskE + ";");
      }
      emitPtxStore(ptxOp, vecWidth, mask != nullptr);
      dedent();
      emit("}");
      return;
    }
    if (vecWidth >= 2 && elemBytes == 8) {
      // Vectorized store for 8-byte elements (f64/i64): st.global.v2.b64
      int groupW = 2; // 2 x 64-bit per store (16B)
      int nGroups = nElems / groupW;
      std::string ptxOp = "st.global" + stSfx + ".v2.b64";

      emit("// Vectorized store via PTX: 2 x " + getCUDAType(elemType) +
           " per store (" + std::to_string(nGroups) + " stores)");
      emit("#pragma unroll");
      emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
      indent();
      emit("int _base = _g * " + std::to_string(groupW) + ";");
      for (int i = 0; i < groupW; i++) {
        std::string valE = valIsScalar ? valVar : (valVar + "[_base+" + std::to_string(i) + "]");
        if (elemType.isF64())
          emit("uint64_t _r" + std::to_string(i) + " = "
               "(uint64_t)__double_as_longlong(" + valE + ");");
        else
          emit("uint64_t _r" + std::to_string(i) + " = "
               "(uint64_t)(" + valE + ");");
      }
      if (mask) {
        auto maskVar = getVar(mask);
        std::string maskE = isScalarValue(mask) ? maskVar : (maskVar + "[_base + " + std::to_string(groupW - 1) + "]");
        emit("int _pred = (int)" + maskE + ";");
      }
      emitPtxStore(ptxOp, groupW, mask != nullptr, /*wide=*/true);
      dedent();
      emit("}");
      return;
    }

    // Scalar fallback
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    {
      std::string ptrDeref = getPtrElemExpr(ptr, "_i");
      std::string valE = getElemExpr(val, "_i");
      // One scalar store; inline PTX when a cache modifier/eviction policy
      // is present (the C assignment cannot carry the hint).
      auto scalarSt = [&](const std::string &pre) {
        if (stSfx.empty()) {
          emit(pre + ptrDeref + " = " + valE + ";");
          return;
        }
        std::string addr = "&(" + ptrDeref + ")";
        // Copy into a typed temp so we can bitcast safely.
        std::string body = getCUDAType(elemType) + " _sv = " + valE + "; ";
        if (elemBytes == 8)
          body += "asm volatile(\"st.global" + stSfx +
                  ".b64 [%0], %1;\" :: \"l\"(" + addr +
                  "), \"l\"(*(uint64_t*)&_sv) : \"memory\");";
        else if (elemBytes == 4)
          body += "asm volatile(\"st.global" + stSfx +
                  ".b32 [%0], %1;\" :: \"l\"(" + addr +
                  "), \"r\"(*(uint32_t*)&_sv) : \"memory\");";
        else if (elemBytes == 2)
          body += "asm volatile(\"st.global" + stSfx +
                  ".b16 [%0], %1;\" :: \"l\"(" + addr +
                  "), \"h\"(*(uint16_t*)&_sv) : \"memory\");";
        else
          body += "uint16_t _sb = *(uint8_t*)&_sv; asm volatile(\"st.global" +
                  stSfx + ".b8 [%0], %1;\" :: \"l\"(" + addr +
                  "), \"h\"(_sb) : \"memory\");";
        emit(pre + "{ " + body + " }");
      };
      if (mask) {
        std::string maskE = getElemExpr(mask, "_i");
        emit("if (" + maskE + ")");
        scalarSt("    ");
      } else {
        scalarSt("");
      }
    }
    dedent();
    emit("}");
  } else {
    // Scalar store: predicate on thread 0 like the PTX backend (@%p st.global
    // with p = tid==0). Letting every thread store the "same" value races with
    // scalar LOADS of the same address elsewhere in the CTA: in
    // test_load_store_same_ptr a late warp re-loads the already-doubled value
    // and writes 4 instead of 2. Only one thread's store is semantically the
    // CTA's store.
    std::string guard = "tid == 0";
    if (mask)
      guard += " && (" + getVar(mask) + ")";
    emit("if (" + guard + ") *" + ptrVar + " = " + valVar + ";");
  }
}
void CUDACodeGen::emitDot(tt::DotOp op) {
  // tt.dot without WGMMA lowering — emit FMA via shared memory.
  // Store A and B to shared memory, then each thread computes its output
  // elements using direct multiply-accumulate.
  auto result = op.getResult();
  auto a = op.getA();
  auto b = op.getB();
  auto acc = op.getC();
  auto rtt = cast<RankedTensorType>(result.getType());
  auto aRtt = cast<RankedTensorType>(a.getType());
  auto bRtt = cast<RankedTensorType>(b.getType());
  int rank = rtt.getRank();
  // Work in per-CTA coordinates throughout: with num_ctas > 1 the layouts may
  // be CTA-split (CTASplitNum > 1), and applying the linear layout with
  // block=0 yields exactly the CTA-local coordinates of each CTA's tile.
  auto outShapePerCTA = ttg::getShapePerCTA(rtt);
  auto aShapePerCTA = ttg::getShapePerCTA(aRtt);
  auto bShapePerCTA = ttg::getShapePerCTA(bRtt);
  int M = outShapePerCTA[rank - 2], N = outShapePerCTA[rank - 1];
  int K = aShapePerCTA[rank - 1];
  // Leading dims (rank > 2) form a batch; A/B/out layouts all carry them.
  int64_t batch = 1;
  for (int d = 0; d < rank - 2; d++)
    batch *= outShapePerCTA[d];
  int nA = getElemsPerThread(a);
  int nB = getElemsPerThread(b);
  int nOut = getElemsPerThread(rtt);
  Type elemType = rtt.getElementType();
  Type aElemType = aRtt.getElementType();
  Type bElemType = bRtt.getElementType();
  auto outType = getCUDAType(elemType);
  auto aType = getCUDAType(aElemType);
  auto bType = getCUDAType(bElemType);
  auto aVar = getVar(a);
  auto bVar = getVar(b);
  auto accVar = getVar(acc);

  // ---- MMAv2 fast path: real mma.sync.m16n8k16 tensor-core instructions ----
  // Small-tile dots keep NvidiaMma versionMajor=2 encodings (instrShape
  // [16,8]); their dot_op(parent=#mma) register layouts ARE the PTX mma
  // fragment layouts. Register indices for every instruction operand are
  // resolved from the lane-0 LinearLayout enumeration: layouts are linear, so
  // the coord->register map is identical for every lane. Any lookup miss
  // falls back to the generic smem FMA path below.
  // Variant selection: 16-bit floats (m16n8k16), tf32 (m16n8k8, one f32 per
  // .b32 reg), int8 (m16n8k32, four s8 per .b32 reg).
  bool mma2F16 = ((aElemType.isF16() && bElemType.isF16()) ||
                  (aElemType.isBF16() && bElemType.isBF16())) &&
                 // f16 accumulate only exists for f16 inputs; bf16 needs f32.
                 (elemType.isF32() || (elemType.isF16() && aElemType.isF16()));
  bool mma2Tf32 = aElemType.isF32() && bElemType.isF32() &&
                  elemType.isF32() &&
                  op.getInputPrecision() == tt::InputPrecision::TF32;
  bool mma2S8 = aElemType.isInteger(8) && bElemType.isInteger(8) &&
                elemType.isInteger(32);
  int mma2K = mma2Tf32 ? 8 : mma2S8 ? 32 : 16;
  // Elements per 32-bit mma operand register.
  int mma2Epr = mma2Tf32 ? 1 : mma2S8 ? 4 : 2;
  if (auto mmaEnc = dyn_cast<ttg::NvidiaMmaEncodingAttr>(rtt.getEncoding());
      mmaEnc && mmaEnc.getVersionMajor() == 2 && rank == 2 &&
      (mma2F16 || mma2Tf32 || mma2S8) && (K % mma2K) == 0 && K >= mma2K &&
      isa<ttg::DotOperandEncodingAttr>(aRtt.getEncoding()) &&
      isa<ttg::DotOperandEncodingAttr>(bRtt.getEncoding()) &&
      !deferredAddPtr.count(a) && !deferredAddPtr.count(b) &&
      !scalarValues.contains(a) && !scalarValues.contains(b)) {
    auto *ctx = rtt.getContext();
    auto kReg = mlir::StringAttr::get(ctx, "register");
    auto kLane = mlir::StringAttr::get(ctx, "lane");
    auto kWarp = mlir::StringAttr::get(ctx, "warp");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    // coord -> per-thread array index at lane 0, honoring the COMPACT vs RAW
    // array conventions (see broadcastRegMapping).
    auto lane0Map = [&](RankedTensorType ty, int nArr,
                        std::map<std::pair<int, int>, int> &m) {
      auto ll = ttg::toLinearLayout(ty);
      int nRegs = ll.getInDimSize(kReg);
      bool compact = nArr < nRegs;
      int ci = 0;
      for (int r = 0; r < nRegs && ci < nArr; r++) {
        auto coords =
            ll.apply({{kReg, r}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        std::pair<int, int> key{coords[0].second, coords[1].second};
        if (m.insert({key, compact ? ci : r}).second)
          ci++;
      }
    };
    std::map<std::pair<int, int>, int> aMap, bMap, dMap;
    lane0Map(aRtt, nA, aMap);
    lane0Map(bRtt, nB, bMap);
    lane0Map(rtt, nOut, dMap);
    // Distinct D tiles: lane-0 in-fragment offsets are m in {0,8}, n in {0,1}.
    bool ok = true;
    std::map<std::pair<int, int>, int> tileIdx;
    SmallVector<std::pair<int, int>> tiles;
    for (auto &[c, r] : dMap) {
      std::pair<int, int> t{c.first & ~8, c.second & ~1};
      if ((t.first % 16) != 0 || (t.second % 8) != 0) {
        ok = false;
        break;
      }
      if (tileIdx.insert({t, (int)tiles.size()}).second)
        tiles.push_back(t);
    }
    std::string failInfo;
    auto look = [&ok, &failInfo](std::map<std::pair<int, int>, int> &m,
                                 const char *name, int x, int y) {
      auto it = m.find({x, y});
      if (it == m.end()) {
        if (ok)
          failInfo = std::string(name) + "(" + std::to_string(x) + "," +
                     std::to_string(y) + ")";
        ok = false;
        return 0;
      }
      return it->second;
    };
    // The mma contracts hardware k positions; correctness only requires that
    // A and B assign the SAME logical k to every hardware position (the dot
    // is invariant under a shared k permutation). Lane-0 lookups below fix
    // the per-quad slot values; across lanes consistency needs identical
    // quad-bit k-steps in A and B, and group/warp bits matching D.
    auto aLL = ttg::toLinearLayout(aRtt);
    auto bLL = ttg::toLinearLayout(bRtt);
    auto dLL = ttg::toLinearLayout(rtt);
    auto at = [&](decltype(aLL) &ll, int lane, int warp) {
      auto c = ll.apply({{kReg, 0}, {kLane, lane}, {kWarp, warp}, {kBlock, 0}});
      return std::pair<int, int>{c[0].second, c[1].second};
    };
    using P = std::pair<int, int>;
    // Degenerate (M<16 or N<8) tiles: the mma fragment broadcasts. The PTX
    // backend still issues a full m16n8k* instruction; canonical fragment row
    // offsets (1,2,4,8) that exceed the tensor's M wrap to (offset % M) in the
    // LinearLayout (likewise col offsets vs N). Wrapping the expected structure
    // coords and the register lookups lets these small tiles take the real mma
    // path — the hardware computes redundant out-of-range rows/cols whose
    // outputs duplicate valid ones (broadcast-consistent), so the aliased D
    // registers receive identical values. For M>=16, N>=8 the wraps are no-ops.
    auto wr = [&](int r) { return M > 0 ? ((r % M) + M) % M : r; };
    auto wc = [&](int c) { return N > 0 ? ((c % N) + N) % N : c; };
    // Quad bits (lanes 1,2): A=(0,kq), B=(kq,0) with equal kq; D=(0,2),(0,4).
    auto aq1 = at(aLL, 1, 0), aq2 = at(aLL, 2, 0);
    auto bq1 = at(bLL, 1, 0), bq2 = at(bLL, 2, 0);
    if (aq1.first != 0 || aq2.first != 0 || bq1.second != 0 ||
        bq2.second != 0 || aq1.second != bq1.first ||
        aq2.second != bq2.first || aq1.second == 0 || aq2.second == 0 ||
        at(dLL, 1, 0) != P{0, wc(2)} || at(dLL, 2, 0) != P{0, wc(4)}) {
      ok = false;
      failInfo = "lane-quad-structure";
    }
    // Group bits (lanes 4,8,16): A rows 1,2,4; B cols 1,2,4; D rows 1,2,4.
    if (at(aLL, 4, 0) != P{wr(1), 0} || at(aLL, 8, 0) != P{wr(2), 0} ||
        at(aLL, 16, 0) != P{wr(4), 0} || at(bLL, 4, 0) != P{0, wc(1)} ||
        at(bLL, 8, 0) != P{0, wc(2)} || at(bLL, 16, 0) != P{0, wc(4)} ||
        at(dLL, 4, 0) != P{wr(1), 0} || at(dLL, 8, 0) != P{wr(2), 0} ||
        at(dLL, 16, 0) != P{wr(4), 0}) {
      ok = false;
      failInfo = "lane-group-structure";
    }
    // Warp bits: D warp offset (mw,nw) requires A offset (mw,0), B (0,nw).
    for (int wb = 1; ok && wb < (int)dLL.getInDimSize(kWarp); wb <<= 1) {
      auto dw = at(dLL, 0, wb);
      if (at(aLL, 0, wb) != P{dw.first, 0} ||
          at(bLL, 0, wb) != P{0, dw.second}) {
        ok = false;
        failInfo = "warp-structure";
      }
    }
    // Lane-0 logical-k list: the k coset lane 0 holds (identical for every
    // row by linearity). Each group of 4 forms one mma k-step; together with
    // the quad k-steps it covers 16 logical k values. For the canonical
    // kWidth=2 layout this reproduces (k, k+1, k+8, k+9); for kWidth=4 it
    // yields (k, k+1, k+2, k+3) with quads striding by 4 — a valid shared
    // permutation of k.
    SmallVector<int> kList;
    if (ok && !tiles.empty()) {
      for (auto &[c, r] : aMap)
        if (c.first == tiles[0].first)
          kList.push_back(c.second);
      llvm::sort(kList);
      if ((int)kList.size() * 4 != K) {
        ok = false;
        failInfo = "klist=" + std::to_string(kList.size());
      }
    }
    // Resolve all operand register indices up front (pure lookups).
    // Per k-step, the e array holds a0,a1,a2,a3 then b0,b1 — epr element
    // indices per 32-bit register (lowest k first within a register).
    struct MMA2Instr {
      int rd[4];
      SmallVector<SmallVector<int, 12>> ks;
    };
    SmallVector<MMA2Instr> instrs;
    int grp = 2 * mma2Epr; // lane-0 logical k values per mma instruction
    for (auto [tm, tn] : tiles) {
      if (!ok)
        break;
      MMA2Instr ins;
      ins.rd[0] = look(dMap, "d", wr(tm), wc(tn));
      ins.rd[1] = look(dMap, "d", wr(tm), wc(tn + 1));
      ins.rd[2] = look(dMap, "d", wr(tm + 8), wc(tn));
      ins.rd[3] = look(dMap, "d", wr(tm + 8), wc(tn + 1));
      for (int g = 0; g + grp - 1 < (int)kList.size() && ok; g += grp) {
        SmallVector<int, 12> e;
        // a0 = (tm, ks half 0), a1 = (tm+8, half 0), a2 = (tm, half 1),
        // a3 = (tm+8, half 1); b0 = (half 0, tn), b1 = (half 1, tn).
        for (int half = 0; half < 2; half++)
          for (int row = 0; row < 2; row++)
            for (int j = 0; j < mma2Epr; j++)
              e.push_back(look(aMap, "a", wr(tm + row * 8),
                               kList[g + half * mma2Epr + j]));
        for (int half = 0; half < 2; half++)
          for (int j = 0; j < mma2Epr; j++)
            e.push_back(look(bMap, "b", kList[g + half * mma2Epr + j], wc(tn)));
        ins.ks.push_back(e);
      }
      instrs.push_back(ins);
    }
    if (!ok || (int)tiles.size() * 4 != nOut || instrs.empty())
      emit("// mma2 fastpath rejected: ok=" + std::to_string(ok) +
           " tiles=" + std::to_string(tiles.size()) +
           " nOut=" + std::to_string(nOut) + " nA=" + std::to_string(nA) +
           " nB=" + std::to_string(nB) + " aMap=" +
           std::to_string(aMap.size()) + " bMap=" + std::to_string(bMap.size()) +
           " dMap=" + std::to_string(dMap.size()) + " fail=" + failInfo);
    if (!ok) {
      auto dump = [&](const char *nm, std::map<std::pair<int, int>, int> &m) {
        std::string s = std::string("//   ") + nm + " lane0:";
        int cnt = 0;
        for (auto &[c, r] : m) {
          if (cnt++ >= 12)
            break;
          s += " (" + std::to_string(c.first) + "," +
               std::to_string(c.second) + ")=" + std::to_string(r);
        }
        emit(s);
      };
      dump("a", aMap);
      dump("b", bMap);
      dump("d", dMap);
    }
    if (ok && (int)tiles.size() * 4 == nOut && !instrs.empty()) {
      bool bf = aElemType.isBF16();
      bool accF16 = elemType.isF16();
      std::string asU16 = bf ? "__bfloat16_as_ushort" : "__half_as_ushort";
      std::string ity = mma2Tf32 ? "tf32" : mma2S8 ? "s8" : bf ? "bf16" : "f16";
      std::string aty = accF16 ? "f16" : mma2S8 ? "s32" : "f32";
      std::string mmaName =
          mma2Tf32 ? "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
          : mma2S8
              ? "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"
          : accF16 ? "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                   : "mma.sync.aligned.m16n8k16.row.col.f32." + ity + "." +
                         ity + ".f32";
      bool accScalar = scalarValues.contains(acc);
      auto mvar = newVar("mma2");
      valueToVar[result] = mvar;
      emit("// tt.dot via mma.sync.m16n8k" + std::to_string(mma2K) +
           " (MMAv2 " + ity + "->" + aty + ", " +
           std::to_string(instrs.size()) + " tiles x " +
           std::to_string(K / mma2K) + " k-steps)");
      emit(outType + " " + mvar + "[" + std::to_string(nOut) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nOut) + "; _i++) " +
           mvar + "[_i] = " + (accScalar ? accVar : accVar + "[_i]") + ";");
      // Pack one 32-bit mma operand register from mma2Epr elements (lowest k
      // in the least-significant position).
      auto pack = [&](Value v, ArrayRef<int> idx) -> std::string {
        if (mma2Tf32)
          return "__float_as_uint(" + getElemExpr(v, std::to_string(idx[0])) +
                 ")";
        if (mma2S8) {
          std::string s;
          for (int j = 3; j >= 1; j--)
            s += "((unsigned)(unsigned char)(" +
                 getElemExpr(v, std::to_string(idx[j])) + ") << " +
                 std::to_string(8 * j) + ") | ";
          return s + "(unsigned)(unsigned char)(" +
                 getElemExpr(v, std::to_string(idx[0])) + ")";
        }
        return "((unsigned)" + asU16 + "(" +
               getElemExpr(v, std::to_string(idx[1])) +
               ") << 16) | (unsigned)" + asU16 + "(" +
               getElemExpr(v, std::to_string(idx[0])) + ")";
      };
      for (auto &ins : instrs) {
        auto d = [&](int i) {
          return mvar + "[" + std::to_string(ins.rd[i]) + "]";
        };
        // Degenerate broadcast tiles (M<16 / N<8) alias D registers
        // (e.g. rd[2]==rd[0]); mma.sync requires 4 DISTINCT output registers,
        // so for the f32/s8/tf32 path accumulate into fresh temps and copy
        // back. Aliased slots hold identical broadcast values, so the unordered
        // copy-back is idempotent. (The f16-acc path already uses distinct
        // packed temps _mc0/_mc1, and its writeback is plain C++ assignment.)
        bool dAlias = !accF16 &&
                      (ins.rd[0] == ins.rd[1] || ins.rd[0] == ins.rd[2] ||
                       ins.rd[0] == ins.rd[3] || ins.rd[1] == ins.rd[2] ||
                       ins.rd[1] == ins.rd[3] || ins.rd[2] == ins.rd[3]);
        emit("{");
        indent();
        if (accF16) {
          // f16 accumulator: c0 packs (m,n),(m,n+1); c1 packs (m+8,*).
          emit("unsigned _mc0 = ((unsigned)__half_as_ushort(" + d(1) +
               ") << 16) | (unsigned)__half_as_ushort(" + d(0) + ");");
          emit("unsigned _mc1 = ((unsigned)__half_as_ushort(" + d(3) +
               ") << 16) | (unsigned)__half_as_ushort(" + d(2) + ");");
        }
        std::string dc = mma2S8 ? "+r" : "+f";
        if (dAlias)
          for (int i = 0; i < 4; i++)
            emit(outType + " _md" + std::to_string(i) + " = " + d(i) + ";");
        auto dt = [&](int i) {
          return dAlias ? ("_md" + std::to_string(i)) : d(i);
        };
        for (auto &e : ins.ks) {
          emit("{");
          indent();
          for (int i = 0; i < 4; i++)
            emit("const unsigned _ma" + std::to_string(i) + " = " +
                 pack(a, ArrayRef<int>(e).slice(i * mma2Epr, mma2Epr)) + ";");
          for (int i = 0; i < 2; i++)
            emit("const unsigned _mb" + std::to_string(i) + " = " +
                 pack(b, ArrayRef<int>(e).slice((4 + i) * mma2Epr, mma2Epr)) +
                 ";");
          if (accF16) {
            emit("asm(\"" + mmaName +
                 " {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\"");
            emit("    : \"+r\"(_mc0), \"+r\"(_mc1)");
            emit("    : \"r\"(_ma0), \"r\"(_ma1), \"r\"(_ma2), \"r\"(_ma3), "
                 "\"r\"(_mb0), \"r\"(_mb1));");
          } else {
            emit("asm(\"" + mmaName +
                 " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\"");
            emit("    : \"" + dc + "\"(" + dt(0) + "), \"" + dc + "\"(" + dt(1) +
                 "), \"" + dc + "\"(" + dt(2) + "), \"" + dc + "\"(" + dt(3) +
                 ")");
            emit("    : \"r\"(_ma0), \"r\"(_ma1), \"r\"(_ma2), \"r\"(_ma3), "
                 "\"r\"(_mb0), \"r\"(_mb1));");
          }
          dedent();
          emit("}");
        }
        if (accF16) {
          emit(d(0) + " = __ushort_as_half((unsigned short)(_mc0 & 0xffffu));");
          emit(d(1) + " = __ushort_as_half((unsigned short)(_mc0 >> 16));");
          emit(d(2) + " = __ushort_as_half((unsigned short)(_mc1 & 0xffffu));");
          emit(d(3) + " = __ushort_as_half((unsigned short)(_mc1 >> 16));");
        } else if (dAlias) {
          for (int i = 0; i < 4; i++)
            emit(d(i) + " = _md" + std::to_string(i) + ";");
        }
        dedent();
        emit("}");
      }
      // Broadcast-fill duplicate D registers. Degenerate tiles (M<16 / N<8)
      // map multiple result array slots to the same (m,n) coord; the mma only
      // wrote each coord's canonical slot (dMap's first occurrence), so the
      // store (which reads every register) would see stale acc values in the
      // duplicate slots. Copy each duplicate from its canonical slot. For
      // non-degenerate tiles every coord is unique → no copies emitted.
      {
        auto ll = ttg::toLinearLayout(rtt);
        int nRegsD = ll.getInDimSize(kReg);
        bool compact = nOut < nRegsD;
        std::map<std::pair<int, int>, int> firstSlot;
        int ci = 0;
        for (int r = 0; r < nRegsD && ci < nOut; r++) {
          auto coords =
              ll.apply({{kReg, r}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
          std::pair<int, int> key{coords[0].second, coords[1].second};
          int slot = compact ? ci : r;
          auto ins = firstSlot.insert({key, slot});
          if (ins.second)
            ci++;
          else if (slot != ins.first->second)
            emit(mvar + "[" + std::to_string(slot) + "] = " + mvar + "[" +
                 std::to_string(ins.first->second) + "];");
        }
      }
      return;
    }
  }

  auto var = newVar("dot");
  valueToVar[result] = var;

  // The accumulator (op.getC()) may be a splat constant emitted as a scalar
  // (e.g. fp8 dots use a fresh zero `%cst` as C and accumulate via a separate
  // addf because of maxNumImpreciseAcc). In that case accVar is a scalar, not
  // an array, so broadcast it rather than indexing accVar[_i].
  bool accIsScalar = scalarValues.contains(acc);
  emit("// tt.dot via shared memory FMA (" + std::to_string(M) + "x" +
       std::to_string(N) + "x" + std::to_string(K) + ")");
  emit(outType + " " + var + "[" + std::to_string(nOut) + "];");
  emit("#pragma unroll");
  emit("for (int _i = 0; _i < " + std::to_string(nOut) + "; _i++) " +
       var + "[_i] = " + (accIsScalar ? accVar : (accVar + "[_i]")) + ";");
  emit("{");
  indent();

  // If an operand was just local_load-ed from an unswizzled shared buffer,
  // read it in place instead of copying it to fresh scratch. Big-batch dot3d
  // cases otherwise double the operand footprint and exceed the smem limit.
  auto getSmemAlias = [&](Value operand, RankedTensorType ortt,
                          SmallVector<int64_t> &strides) -> std::string {
    auto loadOp = operand.getDefiningOp<ttg::LocalLoadOp>();
    if (!loadOp)
      return "";
    auto srcMem = loadOp.getSrc();
    // Direct local_alloc only: view shape == alloc shape, no subview offset.
    if (!srcMem.getDefiningOp<ttg::LocalAllocOp>())
      return "";
    auto mdt = cast<ttg::MemDescType>(srcMem.getType());
    // Note: this emitter addresses swizzled_shared buffers with plain
    // order-derived strides everywhere (store + load sides both ignore the
    // XOR swizzle), so any maxPhase is fine here as long as we mirror that.
    auto swiz = dyn_cast<ttg::SwizzledSharedEncodingAttr>(mdt.getEncoding());
    if (!swiz)
      return "";
    if (mdt.getShape() != ortt.getShape() || ortt.getRank() > 3)
      return "";
    // The buffer must still be live (not deallocated/reused) at the dot.
    for (auto *user : srcMem.getUsers())
      if (isa<ttg::LocalDeallocOp>(user))
        if (user->getBlock() != op->getBlock() || user->isBeforeInBlock(op))
          return "";
    auto it = valueToVar.find(srcMem);
    if (it == valueToVar.end())
      return "";
    auto order = swiz.getOrder();
    int r = ortt.getRank();
    // The smem buffer physically holds the per-CTA tile, so derive strides
    // from the per-CTA shape (matches the local_alloc store side).
    auto allocShapePerCTA = ttg::getShapePerCTA(mdt);
    strides.assign(r, 1);
    int64_t s = 1;
    for (int oi = 0; oi < r; oi++) {
      strides[order[oi]] = s;
      s *= allocShapePerCTA[order[oi]];
    }
    return it->second;
  };
  SmallVector<int64_t> aAliasStrides, bAliasStrides;
  std::string aAlias = getSmemAlias(a, aRtt, aAliasStrides);
  std::string bAlias = getSmemAlias(b, bRtt, bAliasStrides);

  // Allocate shared memory scratch only for non-aliased operands
  int aBytes = getTypeSizeInBytes(aElemType);
  int bBytes = getTypeSizeInBytes(bElemType);
  int smemOffA = (sharedMemOffset + 127) & ~127;
  int aScratch = aAlias.empty() ? (int)(batch * M * K) * aBytes : 0;
  int smemOffB = (smemOffA + aScratch + 127) & ~127;
  int bScratch = bAlias.empty() ? (int)(batch * K * N) * bBytes : 0;
  sharedMemOffset = smemOffB + bScratch;
  if (sharedMemOffset > peakSharedMem)
    peakSharedMem = sharedMemOffset;

  if (aAlias.empty())
    emit(aType + "* _dotA = (" + aType + "*)(shared_mem + " +
         std::to_string(smemOffA) + ");");
  else
    emit("const " + aType + "* _dotA = " + aAlias + "; // aliased operand");
  if (bAlias.empty())
    emit(bType + "* _dotB = (" + bType + "*)(shared_mem + " +
         std::to_string(smemOffB) + ");");
  else
    emit("const " + bType + "* _dotB = " + bAlias + "; // aliased operand");

  // Store A to shared memory using LinearLayout
  if (aAlias.empty()) {
    auto aLL = ttg::toLinearLayout(aRtt);
    auto kReg = mlir::StringAttr::get(aRtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(aRtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(aRtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(aRtt.getContext(), "block");
    auto &aShape = aShapePerCTA;
    SmallVector<int64_t> aStrides(aShape.size());
    aStrides.back() = 1;
    for (int d = aShape.size() - 2; d >= 0; d--)
      aStrides[d] = aStrides[d + 1] * aShape[d + 1];
    const auto &aBases = aLL.getBases();
    const auto &aLaneBases = aBases.find(kLane)->second;
    const auto &aWarpBases = aBases.find(kWarp)->second;
    emit("// Store A to shared memory");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nA) + "; _i++) {");
    indent();
    // Build offset as switch on _i for register component, + lane/warp runtime
    // For efficiency, emit per-register constants
    emit("int _aoff;");
    emit("switch (_i) {");
    for (int i = 0; i < nA; i++) {
      auto coords = aLL.apply(
          {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      int64_t regBase = 0;
      for (size_t d = 0; d < coords.size(); d++)
        regBase += coords[d].second * aStrides[d];
      emit("case " + std::to_string(i) + ": _aoff = " +
           std::to_string(regBase) + "; break;");
    }
    emit("default: _aoff = 0; break;");
    emit("}");
    // Add lane and warp contributions
    for (size_t lb = 0; lb < aLaneBases.size(); lb++) {
      int64_t delta = 0;
      for (size_t d = 0; d < aLaneBases[lb].size(); d++)
        delta += aLaneBases[lb][d] * aStrides[d];
      if (delta != 0) {
        emit("_aoff += ((lane_id >> " + std::to_string(lb) +
             ") & 1) * " + std::to_string(delta) + ";");
      }
    }
    for (size_t wb = 0; wb < aWarpBases.size(); wb++) {
      int64_t delta = 0;
      for (size_t d = 0; d < aWarpBases[wb].size(); d++)
        delta += aWarpBases[wb][d] * aStrides[d];
      if (delta != 0) {
        emit("_aoff += ((warp_id >> " + std::to_string(wb) +
             ") & 1) * " + std::to_string(delta) + ";");
      }
    }
    // Splat-constant operands are emitted as scalars (no [_i]).
    emit("_dotA[_aoff] = " + aVar + (isScalarValue(a) ? ";" : "[_i];"));
    dedent();
    emit("}");
  }

  // Store B to shared memory using LinearLayout
  if (bAlias.empty()) {
    auto bLL = ttg::toLinearLayout(bRtt);
    auto kReg = mlir::StringAttr::get(bRtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(bRtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(bRtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(bRtt.getContext(), "block");
    auto &bShape = bShapePerCTA;
    SmallVector<int64_t> bStrides(bShape.size());
    bStrides.back() = 1;
    for (int d = bShape.size() - 2; d >= 0; d--)
      bStrides[d] = bStrides[d + 1] * bShape[d + 1];
    const auto &bBases = bLL.getBases();
    const auto &bLaneBases = bBases.find(kLane)->second;
    const auto &bWarpBases = bBases.find(kWarp)->second;
    emit("// Store B to shared memory");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nB) + "; _i++) {");
    indent();
    emit("int _boff;");
    emit("switch (_i) {");
    for (int i = 0; i < nB; i++) {
      auto coords = bLL.apply(
          {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      int64_t regBase = 0;
      for (size_t d = 0; d < coords.size(); d++)
        regBase += coords[d].second * bStrides[d];
      emit("case " + std::to_string(i) + ": _boff = " +
           std::to_string(regBase) + "; break;");
    }
    emit("default: _boff = 0; break;");
    emit("}");
    for (size_t lb = 0; lb < bLaneBases.size(); lb++) {
      int64_t delta = 0;
      for (size_t d = 0; d < bLaneBases[lb].size(); d++)
        delta += bLaneBases[lb][d] * bStrides[d];
      if (delta != 0) {
        emit("_boff += ((lane_id >> " + std::to_string(lb) +
             ") & 1) * " + std::to_string(delta) + ";");
      }
    }
    for (size_t wb = 0; wb < bWarpBases.size(); wb++) {
      int64_t delta = 0;
      for (size_t d = 0; d < bWarpBases[wb].size(); d++)
        delta += bWarpBases[wb][d] * bStrides[d];
      if (delta != 0) {
        emit("_boff += ((warp_id >> " + std::to_string(wb) +
             ") & 1) * " + std::to_string(delta) + ";");
      }
    }
    emit("_dotB[_boff] = " + bVar + (isScalarValue(b) ? ";" : "[_i];"));
    dedent();
    emit("}");
  }

  blockSync();

  // Compute output elements: each thread reads A and B from shared memory
  {
    auto outLL = ttg::toLinearLayout(rtt);
    auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");
    auto &outShape = outShapePerCTA;
    const auto &outBases = outLL.getBases();
    const auto &outLaneBases = outBases.find(kLane)->second;
    const auto &outWarpBases = outBases.find(kWarp)->second;

    emit("// FMA: each thread accumulates its output elements");
    emit("#pragma unroll");
    emit("for (int _o = 0; _o < " + std::to_string(nOut) + "; _o++) {");
    indent();

    // Linearization strides for the leading (batch) dims of the output.
    SmallVector<int64_t> batStrides(rank - 2, 1);
    for (int d = rank - 4; d >= 0; d--)
      batStrides[d] = batStrides[d + 1] * outShape[d + 1];
    bool hasBatch = rank > 2;
    // Compute (batch, row, col) for this output element
    emit(std::string("int _row, _col") + (hasBatch ? ", _bat" : "") + ";");
    emit("switch (_o) {");
    for (int i = 0; i < nOut; i++) {
      auto coords =
          outLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      // coords is [(dim0, val0), ..., (dimRank-1, valRank-1)]
      int64_t bat = 0;
      for (int d = 0; d < rank - 2; d++)
        bat += coords[d].second * batStrides[d];
      int64_t row = coords[rank - 2].second;
      int64_t col = coords[rank - 1].second;
      emit("case " + std::to_string(i) + ": _row = " +
           std::to_string(row) + "; _col = " + std::to_string(col) +
           (hasBatch ? "; _bat = " + std::to_string(bat) : "") +
           "; break;");
    }
    emit(std::string("default: _row = 0; _col = 0;") +
         (hasBatch ? " _bat = 0;" : "") + " break;");
    emit("}");

    // Add lane and warp contributions to batch/row/col
    auto emitBaseContrib = [&](const std::vector<std::vector<int32_t>> &bases,
                               const std::string &idVar) {
      for (size_t b = 0; b < bases.size(); b++) {
        int64_t batDelta = 0;
        for (int d = 0; d < rank - 2; d++)
          batDelta += bases[b][d] * batStrides[d];
        int64_t rowDelta = bases[b][rank - 2];
        int64_t colDelta = bases[b][rank - 1];
        std::string bit =
            "((" + idVar + " >> " + std::to_string(b) + ") & 1) * ";
        if (batDelta != 0)
          emit("_bat += " + bit + std::to_string(batDelta) + ";");
        if (rowDelta != 0)
          emit("_row += " + bit + std::to_string(rowDelta) + ";");
        if (colDelta != 0)
          emit("_col += " + bit + std::to_string(colDelta) + ";");
      }
    };
    emitBaseContrib(outLaneBases, "lane_id");
    emitBaseContrib(outWarpBases, "warp_id");

    // Inner product: sum A[bat][row][k] * B[bat][k][col] for k in [0, K)
    // Index strides: scratch is row-major; aliased operands use the shared
    // encoding's order-derived strides (matching the local_alloc store side).
    int64_t aSb = (int64_t)M * K, aSr = K, aSk = 1;
    if (!aAlias.empty()) {
      int r = aRtt.getRank();
      aSk = aAliasStrides[r - 1];
      aSr = aAliasStrides[r - 2];
      aSb = (r > 2) ? aAliasStrides[0] : 0;
    }
    int64_t bSb = (int64_t)K * N, bSk = N, bSc = 1;
    if (!bAlias.empty()) {
      int r = bRtt.getRank();
      bSc = bAliasStrides[r - 1];
      bSk = bAliasStrides[r - 2];
      bSb = (r > 2) ? bAliasStrides[0] : 0;
    }
    std::string aIdx =
        "_row * " + std::to_string(aSr) + " + _k * " + std::to_string(aSk);
    std::string bIdx =
        "_k * " + std::to_string(bSk) + " + _col * " + std::to_string(bSc);
    if (hasBatch) {
      aIdx = "_bat * " + std::to_string(aSb) + " + " + aIdx;
      bIdx = "_bat * " + std::to_string(bSb) + " + " + bIdx;
    }
    // f16/bf16 outputs: accumulate in float (HMMA computes products in full
    // precision; raw __half FMA chains drift past test tolerances).
    bool f32Acc = elemType.isF16() || elemType.isBF16();
    std::string accType = f32Acc ? "float" : outType;
    emit(accType + " _sum = " + (f32Acc ? "(float)" : "") + var + "[_o];");
    emit("#pragma unroll");
    emit("for (int _k = 0; _k < " + std::to_string(K) + "; _k++) {");
    indent();
    // Cast to accumulation type
    if (f32Acc || aElemType.isF16() || aElemType.isBF16()) {
      emit("_sum += (" + accType + ")_dotA[" + aIdx + "] * (" + accType +
           ")_dotB[" + bIdx + "];");
    } else {
      emit("_sum += _dotA[" + aIdx + "] * _dotB[" + bIdx + "];");
    }
    dedent();
    emit("}");
    emit(var + "[_o] = " + (f32Acc ? "(" + outType + ")" : "") + "_sum;");

    dedent();
    emit("}");
  }
  blockSync();

  dedent();
  emit("}");
}

void CUDACodeGen::emitReduce(tt::ReduceOp op) {
  auto result = op->getResult(0);
  auto src = op.getOperands()[0];
  int axis = op.getAxis();
  int srcElems = getElemsPerThread(src);
  // CGA: reduction across CTAs in a cluster (CTASplitNum > 1 along the reduce
  // axis) needs DSmem exchange, which is not implemented yet. Hard-error
  // instead of silently producing per-CTA partial results.
  if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
    auto splitNum = ttg::getCTASplitNum(srcTy.getEncoding());
    if (axis < (int)splitNum.size() && splitNum[axis] > 1) {
      emitFailed = true;
      emitErrorMsg = "CUDA emitter: cross-CTA reduce (CTASplitNum > 1 on "
                     "reduce axis) is not implemented";
      return;
    }
  }
  int numResults = op->getNumResults();
  auto var = newVar("red");
  valueToVar[result] = var;

  // Effective warp count for this reduce. Inside a ttg.warp_specialize
  // partition only `partWarps[i]` warps physically execute (the producer/
  // consumer split), so the cross-warp combine must size its scratch and read
  // back exactly that many warp slots — using the global `numWarps` (e.g. 4)
  // in a 2-warp partition writes slots [0,1] but reads [0..3], pulling in
  // uninitialized shared memory (intermittent wrong sums, manifests when a
  // prior kernel left non-zero garbage in that smem). wsSyncThreadCount is the
  // current region's thread count (set per WS region; 0 outside WS).
  int effNumWarps = wsSyncThreadCount > 0 ? (wsSyncThreadCount / 32) : numWarps;

  // For multi-result reduces (e.g., max-with-indices), map all results
  std::string var1;
  if (numResults > 1) {
    var1 = newVar("red");
    valueToVar[op->getResult(1)] = var1;
  }

  // Determine reduce op from combiner region
  std::string reduceOp = "add";
  bool hasSelect = false;
  bool isUnsignedReduce = false;
  for (auto &combOp : op.getCombineOp().front()) {
    llvm::StringRef name = combOp.getName().getStringRef();
    if (name.contains("addf") || name.contains("addi"))
      reduceOp = "add";
    else if (name.contains("xori"))
      reduceOp = "xor";
    else if (name.contains("maxnumf") || name.contains("maxf") ||
             name.contains("maximumf") || name.contains("maxsi") ||
             name.contains("maxui"))
      reduceOp = "max";
    else if (name.contains("minnumf") || name.contains("minf") ||
             name.contains("minimumf") || name.contains("minsi") ||
             name.contains("minui"))
      reduceOp = "min";
    else if (isa<arith::SelectOp>(&combOp))
      // Only an actual compare+SELECT region is a min/max combinator. A bare
      // cmpi/cmpf may belong to something else entirely (e.g. the overflow
      // device_assert in test_side_effectful_reduction's sanitize_add) and
      // must not override the real combine op.
      hasSelect = true;
    else if (auto asmOp = dyn_cast<tt::ElementwiseInlineAsmOp>(&combOp)) {
      // Custom PTX-asm reduce combinator. The flexpoint absmax path uses
      // `max.NaN.xorsign.abs.f32` (max-of-magnitude, sign = xor of input
      // signs which the caller clears afterward). Treat it as an abs-max
      // reduction; without this the loop below leaves reduceOp at its "add"
      // default and emits a SUM, silently corrupting the output flex scale.
      StringRef asmStr = asmOp.getAsmString();
      if (asmStr.contains("max") && asmStr.contains("abs"))
        reduceOp = "absmax";
    }
    // Unsigned max/min reductions must use unsigned comparisons; the packed
    // sort/topk keys (uint32) have the high bit set for positive fp inputs, so
    // a signed max picks the wrong element.
    if (name.contains("maxui") || name.contains("minui"))
      isUnsignedReduce = true;
  }
  // For compare+select combiners detect the comparison direction. This also
  // covers SINGLE-result select-based min/max (e.g. test_chain_reduce's
  // `cmpi sgt + select` region) — without it reduceOp stays at the "add"
  // default and the kernel silently emits a SUM instead of a MAX.
  if (hasSelect) {
    // Only inspect compares on the VALUE block-arg pair (accumulator vs
    // current, i.e. arg0 vs arg(numOperands)). Other compares say nothing
    // about the combine direction: argmin/argmax tie-breaks compare the
    // INDEX args (cmpi slt would flip max→min), and device_assert range
    // checks (sanitize_add) compare derived values against constants.
    Block &combEntry = op.getCombineOp().front();
    int nIn = (int)op.getNumOperands();
    auto isValueArgCmp = [&](Operation *cmp) {
      if ((int)combEntry.getNumArguments() < 2 * nIn)
        return false;
      Value a0 = combEntry.getArgument(0);
      Value b0 = combEntry.getArgument(nIn);
      Value l = cmp->getOperand(0), r = cmp->getOperand(1);
      return (l == a0 && r == b0) || (l == b0 && r == a0);
    };
    for (auto &combOp : op.getCombineOp().front()) {
      if (!isa<arith::CmpFOp, arith::CmpIOp>(&combOp) ||
          !isValueArgCmp(&combOp))
        continue;
      if (auto cmpf = dyn_cast<arith::CmpFOp>(&combOp)) {
        auto pred = cmpf.getPredicate();
        if (pred == arith::CmpFPredicate::OGT || pred == arith::CmpFPredicate::OGE)
          reduceOp = "max";
        else if (pred == arith::CmpFPredicate::OLT || pred == arith::CmpFPredicate::OLE)
          reduceOp = "min";
      }
      if (auto cmpi = dyn_cast<arith::CmpIOp>(&combOp)) {
        auto pred = cmpi.getPredicate();
        if (pred == arith::CmpIPredicate::sgt || pred == arith::CmpIPredicate::sge ||
            pred == arith::CmpIPredicate::ugt || pred == arith::CmpIPredicate::uge)
          reduceOp = "max";
        else if (pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::sle ||
                 pred == arith::CmpIPredicate::ult || pred == arith::CmpIPredicate::ule)
          reduceOp = "min";
        if (pred == arith::CmpIPredicate::ugt || pred == arith::CmpIPredicate::uge ||
            pred == arith::CmpIPredicate::ult || pred == arith::CmpIPredicate::ule)
          isUnsignedReduce = true;
      }
    }
  }

  Type srcElemType;
  if (auto rtt = dyn_cast<RankedTensorType>(src.getType()))
    srcElemType = rtt.getElementType();
  else
    srcElemType = src.getType();
  auto cudaType = getCUDAType(srcElemType);
  auto srcVar = getVar(src);

  // For multi-result reduce, get the second operand (indices)
  std::string src1Var;
  std::string cuda1Type;
  if (numResults > 1 && op.getNumOperands() > 1) {
    auto src1 = op.getOperands()[1];
    src1Var = getVar(src1);
    Type src1ElemType;
    if (auto rtt = dyn_cast<RankedTensorType>(src1.getType()))
      src1ElemType = rtt.getElementType();
    else
      src1ElemType = src1.getType();
    cuda1Type = getCUDAType(src1ElemType);
  }

  // Identity value (must match the actual data type to avoid overflow)
  std::string identity;
  if (reduceOp == "add" || reduceOp == "xor") {
    identity = srcElemType.isIntOrIndex() ? "0" : "0.0f";
  } else if (reduceOp == "absmax") {
    // |x| >= 0, so 0 is the identity for max-of-magnitude.
    identity = "0.0f";
  } else if (reduceOp == "max") {
    if (srcElemType.isIntOrIndex()) {
      // Identity for max-reduce is the smallest representable value.
      if (isUnsignedReduce) {
        identity = "0";
      } else {
        int bits = srcElemType.isIndex() ? 64 : srcElemType.getIntOrFloatBitWidth();
        if (bits == 8) identity = "(-128)";
        else if (bits == 16) identity = "(-32768)";
        else if (bits == 64) identity = "LLONG_MIN";
        else identity = "INT_MIN";
      }
    } else {
      identity = "(-INFINITY)";
    }
  } else if (reduceOp == "min") {
    if (srcElemType.isIntOrIndex()) {
      // Identity for min-reduce is the largest representable value.
      int bits = srcElemType.isIndex() ? 64 : srcElemType.getIntOrFloatBitWidth();
      if (isUnsignedReduce) {
        if (bits == 8) identity = "0xFFU";
        else if (bits == 16) identity = "0xFFFFU";
        else if (bits == 64) identity = "ULLONG_MAX";
        else identity = "UINT_MAX";
      } else {
        if (bits == 8) identity = "127";
        else if (bits == 16) identity = "32767";
        else if (bits == 64) identity = "LLONG_MAX";
        else identity = "INT_MAX";
      }
    } else {
      identity = "INFINITY";
    }
  }

  // Reduce expression
  auto redExpr = [&](const std::string &a, const std::string &b) -> std::string {
    if (reduceOp == "add") {
      // i1 add is mod-2 (PTX semantics) — C++ bool `+` saturates to 1, so a
      // 1024-element sum of ones would wrongly yield 1 instead of 0. XOR is
      // exactly add-mod-2.
      if (srcElemType.isInteger(1))
        return "(" + a + " ^ " + b + ")";
      return "(" + a + " + " + b + ")";
    }
    if (reduceOp == "absmax")
      // max of magnitudes; fabsf is idempotent so re-applying it across the
      // per-element / cross-lane / cross-warp stages is harmless.
      return "fmaxf(fabsf(" + a + "), fabsf(" + b + "))";
    if (reduceOp == "max") {
      if (srcElemType.isF32() || srcElemType.isF16() || srcElemType.isBF16())
        return "fmaxf(" + a + ", " + b + ")";
      if (srcElemType.isF64())
        return "fmax(" + a + ", " + b + ")";
      if (isUnsignedReduce) {
        auto utype = getUnsignedType(srcElemType);
        return "(" + cudaType + ")max((" + utype + ")" + a + ", (" + utype +
               ")" + b + ")";
      }
      return "max(" + a + ", " + b + ")";
    }
    if (reduceOp == "min") {
      if (srcElemType.isF32() || srcElemType.isF16() || srcElemType.isBF16())
        return "fminf(" + a + ", " + b + ")";
      if (srcElemType.isF64())
        return "fmin(" + a + ", " + b + ")";
      if (isUnsignedReduce) {
        auto utype = getUnsignedType(srcElemType);
        return "(" + cudaType + ")min((" + utype + ")" + a + ", (" + utype +
               ")" + b + ")";
      }
      return "min(" + a + ", " + b + ")";
    }
    if (reduceOp == "xor")
      return "(" + a + " ^ " + b + ")";
    return a;
  };

  // For multi-result (argmax/argmin) compare-and-select sites, the value
  // comparison must use unsigned semantics when the reduce is unsigned.
  auto uCast = [&](const std::string &e) -> std::string {
    if (isUnsignedReduce)
      return "(" + getUnsignedType(srcElemType) + ")" + e;
    return e;
  };

  // Check if result is tensor (partial reduce) or scalar (full reduce)
  bool resultIsTensor = isa<RankedTensorType>(result.getType());
  auto srcEnc =
      isa<RankedTensorType>(src.getType())
          ? cast<RankedTensorType>(src.getType()).getEncoding()
          : Attribute();

  // Check for MMA layout reduce along axis=1
  bool isMMAReduce = false;
  if (resultIsTensor) {
    auto resultTy = cast<RankedTensorType>(result.getType());
    auto resultEnc = resultTy.getEncoding();
    if (auto slice = dyn_cast_or_null<ttg::SliceEncodingAttr>(resultEnc)) {
      if (isa<ttg::NvidiaMmaEncodingAttr>(slice.getParent()) && axis == 1)
        isMMAReduce = true;
    }
    if (!isMMAReduce && isa_and_nonnull<ttg::NvidiaMmaEncodingAttr>(srcEnc) &&
        axis == 1)
      isMMAReduce = true;
    // The hardcoded MMA fast-path assumes exactly 2 row-groups per thread
    // (one M instr-tile). When a thread owns >2 rows (multi m-tile, e.g.
    // BLOCK_M=128), it would collapse distinct rows together — fall back to
    // the general LinearLayout grouping which handles any row count.
    if (isMMAReduce && getElemsPerThread(result) != 2)
      isMMAReduce = false;
    // The fast path only combines across lane bits 0-1. If any *warp* base
    // moves along the reduce axis (warps split along N, e.g. 16x16 mma(v2)
    // with warpsPerCTA=[1,2]), the cross-warp combine would be silently
    // dropped (sum/max over half the row) — use the general path instead.
    if (isMMAReduce) {
      auto srcRtt2 = cast<RankedTensorType>(src.getType());
      auto ll2 = ttg::toLinearLayout(srcRtt2);
      auto ctx2 = srcRtt2.getContext();
      auto kReg2 = mlir::StringAttr::get(ctx2, "register");
      auto kLane2 = mlir::StringAttr::get(ctx2, "lane");
      auto kWarp2 = mlir::StringAttr::get(ctx2, "warp");
      auto kBlock2 = mlir::StringAttr::get(ctx2, "block");
      for (int b = 0; b < ll2.getInDimSizeLog2(kWarp2); b++) {
        auto idxs = ll2.apply(
            {{kReg2, 0}, {kLane2, 0}, {kWarp2, 1 << b}, {kBlock2, 0}});
        if (idxs[axis].second != 0) {
          isMMAReduce = false;
          break;
        }
      }
    }
  }

  if (resultIsTensor && isMMAReduce) {
    // MMA layout dim-1 reduce: 2 row groups per thread
    emit(cudaType + " " + var + "[2];");
    emit(var + "[0] = " + identity + "; " + var + "[1] = " + identity + ";");
    emit("#pragma unroll");
    emit("for (int _r = 0; _r < " + std::to_string(srcElems) + "; _r++) {");
    indent();
    emit("int _g = (_r >> 1) & 1;");
    emit(var + "[_g] = " + redExpr(var + "[_g]", srcVar + "[_r]") + ";");
    dedent();
    emit("}");
    // Cross-lane reduction (4 lanes per row)
    emit("// Cross-lane reduction (4 lanes per row)");
    for (int offset : {1, 2}) {
      emit("#pragma unroll");
      emit("for (int _g = 0; _g < 2; _g++)");
      emit("    " + var + "[_g] = " +
           redExpr(var + "[_g]",
                   "__shfl_xor_sync(0xffffffff, " + var + "[_g], " +
                       std::to_string(offset) + ")") +
           ";");
    }
  } else if (resultIsTensor) {
    // 2D/3D partial reduce using LinearLayout for correct element grouping
    int resultElems = getElemsPerThread(result);
    bool isMultiResult2D = numResults > 1 && !src1Var.empty();
    std::string accType = cudaType; // no promotion needed for tensor reduce (results stay in original type)
    emit(cudaType + " " + var + "[" + std::to_string(resultElems) + "];");
    if (isMultiResult2D)
      emit(cuda1Type + " " + var1 + "[" + std::to_string(resultElems) + "];");
    emit("// Reduction along axis=" + std::to_string(axis));

    // Use LinearLayout to compute per-register tensor coordinates
    auto srcRtt = cast<RankedTensorType>(src.getType());
    auto ll = ttg::toLinearLayout(srcRtt);
    auto kRegister = mlir::StringAttr::get(srcRtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(srcRtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(srcRtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(srcRtt.getContext(), "block");
    int rank = srcRtt.getRank();

    // Compute offset for each register (at lane=0, warp=0)
    SmallVector<SmallVector<int>> regOffsets;
    for (int i = 0; i < srcElems; i++) {
      auto idxs = ll.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      SmallVector<int> coords;
      for (auto &idx : idxs) coords.push_back(idx.second);
      regOffsets.push_back(coords);
    }

    // Map each RESULT register to its tensor coordinate (at lane=0,warp=0) using
    // the result's own LinearLayout. The result is a slice (reduce drops `axis`),
    // so its coords live in the source dim space with `axis` removed. We then feed
    // each source register into the result register(s) whose non-axis coordinate
    // matches. Driving the grouping from the result LL — instead of assuming
    // (#distinct source groups == resultElems) — correctly handles register-
    // broadcast result layouts, where several result registers share the same
    // coordinate and must ALL receive the reduced value (e.g. the rank-10
    // bitonic-sort slice layouts produced by tl.sort / topk over many experts).
    auto resultRtt = cast<RankedTensorType>(result.getType());
    auto resultLL = ttg::toLinearLayout(resultRtt);
    SmallVector<SmallVector<int>> resOffsets;
    for (int o = 0; o < resultElems; o++) {
      auto idxs =
          resultLL.apply({{kRegister, o}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      SmallVector<int> coords;
      for (auto &idx : idxs) coords.push_back(idx.second);
      resOffsets.push_back(coords);
    }
    // Non-axis coordinate of a source register (drops the reduced dim, leaving a
    // vector aligned with the result's dim numbering).
    auto srcKeyOf = [&](int s) {
      SmallVector<int> key;
      for (int d = 0; d < (int)regOffsets[s].size(); d++)
        if (d != axis) key.push_back(regOffsets[s][d]);
      return key;
    };

    // Emit per-thread reduction.
    emit("// Initialize reduction accumulators");
    emit("#pragma unroll");
    emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++) {");
    emit("    " + var + "[_o] = " + identity + ";");
    if (isMultiResult2D) emit("    " + var1 + "[_o] = 0;");
    emit("}");

    for (int oi = 0; oi < resultElems; oi++) {
      for (int si = 0; si < srcElems; si++) {
        if (srcKeyOf(si) != resOffsets[oi]) continue;
        if (isMultiResult2D) {
          std::string cmpOp = (reduceOp == "max") ? ">" : "<";
          emit("if (" + uCast(srcVar + "[" + std::to_string(si) + "]") + " " + cmpOp + " " +
               uCast(var + "[" + std::to_string(oi) + "]") + ") { " +
               var + "[" + std::to_string(oi) + "] = " + srcVar + "[" + std::to_string(si) + "]; " +
               var1 + "[" + std::to_string(oi) + "] = " + src1Var + "[" + std::to_string(si) + "]; }");
        } else {
          emit(var + "[" + std::to_string(oi) + "] = " +
               redExpr(var + "[" + std::to_string(oi) + "]", srcVar + "[" + std::to_string(si) + "]") + ";");
        }
      }
    }

    // Cross-lane reduction using LinearLayout lane bases
    {
      const auto &bases = ll.getBases();
      auto laneIt = bases.find(kLane);
      if (laneIt != bases.end()) {
        const auto &laneBases = laneIt->second;
        // Find the shuffle offset: first lane base with non-zero reduce axis
        int threadOffset = 1;
        for (const auto &base : laneBases) {
          if (base[axis] != 0) break;
          threadOffset *= 2;
        }
        int lanesAlongAxis = ttg::getThreadsPerWarp(srcRtt.getEncoding(), srcRtt.getShape())[axis];
        if (lanesAlongAxis > 1) {
          emit("// Cross-lane reduction (" + std::to_string(lanesAlongAxis) + " lanes along axis " + std::to_string(axis) + ")");
          for (int off = threadOffset; off < 32; off *= 2) {
            // Only shuffle if this offset contributes to the reduce axis
            int laneIdx = 0;
            int temp = off;
            while (temp > 1 && laneIdx < (int)laneBases.size()) { laneIdx++; temp >>= 1; }
            if (laneIdx < (int)laneBases.size() && laneBases[laneIdx][axis] != 0) {
              if (isMultiResult2D) {
                std::string cmpOp = (reduceOp == "max") ? ">" : "<";
                emit("{ // shuffle offset " + std::to_string(off));
                emit("    #pragma unroll");
                emit("    for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++) {");
                emit("        " + accType + " _ov = __shfl_xor_sync(0xffffffff, " + var + "[_o], " + std::to_string(off) + ");");
                emit("        " + cuda1Type + " _oi = __shfl_xor_sync(0xffffffff, " + var1 + "[_o], " + std::to_string(off) + ");");
                emit("        if (" + uCast("_ov") + " " + cmpOp + " " + uCast(var + "[_o]") + ") { " + var + "[_o] = _ov; " + var1 + "[_o] = _oi; }");
                emit("    }");
                emit("}");
              } else {
                emit("#pragma unroll");
                emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++)");
                emit("    " + var + "[_o] = " + redExpr(var + "[_o]",
                     "__shfl_xor_sync(0xffffffff, " + var + "[_o], " + std::to_string(off) + ")") + ";");
              }
            }
          }
        }
      }
    }

    // Cross-warp reduction (LinearLayout-aware keepdims reduce)
    // Each thread combines ONLY the partner warps that differ in the reduce
    // axis's warp-id bits, keeping its own non-axis warp coordinate. This is a
    // keepdims reduce: warp w and its partners all receive the combined result,
    // but warps differing in non-axis bits keep independent results.
    {
      const auto &bases = ll.getBases();
      auto warpIt = bases.find(kWarp);
      SmallVector<int> axisWarpBits;
      if (warpIt != bases.end()) {
        const auto &warpBases = warpIt->second;
        for (int b = 0; b < (int)warpBases.size(); b++) {
          if (axis < (int)warpBases[b].size() && warpBases[b][axis] != 0)
            axisWarpBits.push_back(b);
        }
      }
      if (!axisWarpBits.empty()) {
        int totalThreads = effNumWarps * 32;
        std::string bitsStr;
        for (int b : axisWarpBits)
          bitsStr += (bitsStr.empty() ? "" : ",") + std::to_string(b);
        emit("// Cross-warp reduction along axis " + std::to_string(axis) +
             " (keepdims; warp-id bits along axis: " + bitsStr + ")");
        emit("{");
        indent();
        auto bufName = "_xw_" + var;
        int bufSize = totalThreads * resultElems;
        // Static __shared__ is capped at 48 KB (0xc000) by ptxas and stacks on
        // top of the dynamic region (task #44). Large keepdims scratch (e.g.
        // chain_reduce 256x256 → 64 KB) must live in the DYNAMIC region at the
        // current allocation floor — it is transient (written and consumed
        // between the surrounding barriers). Keep static under warp
        // specialization where concurrent partitions would race the floor.
        bool xwDynamic = (totalNumWarps == numWarps);
        auto elemBytes = [](Type t) -> int {
          if (!t.isIntOrFloat())
            return 8;
          return std::max(1u, t.getIntOrFloatBitWidth() / 8);
        };
        std::string bufName1;
        if (xwDynamic) {
          int accBytes = elemBytes(srcElemType);
          int xwOff = (sharedMemOffset + 15) & ~15;
          emit(accType + "* " + bufName + " = (" + accType +
               "*)(shared_mem + " + std::to_string(xwOff) +
               "); // transient keepdims cross-warp scratch");
          int xwEnd = xwOff + bufSize * accBytes;
          if (isMultiResult2D) {
            bufName1 = "_xw_" + var1;
            int xwOff1 = (xwEnd + 15) & ~15;
            int idxBytes = elemBytes(
                cast<RankedTensorType>(op.getOperands()[1].getType())
                    .getElementType());
            emit(cuda1Type + "* " + bufName1 + " = (" + cuda1Type +
                 "*)(shared_mem + " + std::to_string(xwOff1) +
                 "); // transient keepdims cross-warp scratch");
            xwEnd = xwOff1 + bufSize * idxBytes;
          }
          peakSharedMem = std::max(peakSharedMem, xwEnd);
        } else {
          emit("__shared__ " + accType + " " + bufName + "[" +
               std::to_string(bufSize) + "];");
          if (isMultiResult2D) {
            bufName1 = "_xw_" + var1;
            emit("__shared__ " + cuda1Type + " " + bufName1 + "[" +
                 std::to_string(bufSize) + "];");
          }
        }
        // Each thread stores its partial result
        emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++)");
        emit("    " + bufName + "[tid * " + std::to_string(resultElems) + " + _o] = " + var + "[_o];");
        if (isMultiResult2D) {
          emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++)");
          emit("    " + bufName1 + "[tid * " + std::to_string(resultElems) + " + _o] = " + var1 + "[_o];");
        }
        blockSync();
        // Combine every non-empty subset of axis warp bits. Base = own warp's
        // value (already in var[_o]); partner warp = warp_id ^ warpXor.
        int nSub = 1 << axisWarpBits.size();
        emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++) {");
        indent();
        for (int sub = 1; sub < nSub; sub++) {
          int warpXor = 0;
          for (int b = 0; b < (int)axisWarpBits.size(); b++)
            if (sub & (1 << b))
              warpXor |= (1 << axisWarpBits[b]);
          emit("{");
          indent();
          emit("int _src = ((warp_id ^ " + std::to_string(warpXor) +
               ") * 32 + lane_id) * " + std::to_string(resultElems) + " + _o;");
          if (isMultiResult2D) {
            std::string cmpOp = (reduceOp == "max") ? ">" : "<";
            emit("if (" + uCast(bufName + "[_src]") + " " + cmpOp + " " + uCast(var + "[_o]") + ") {");
            emit("    " + var + "[_o] = " + bufName + "[_src];");
            emit("    " + var1 + "[_o] = " + bufName1 + "[_src]; }");
          } else {
            emit(var + "[_o] = " + redExpr(var + "[_o]", bufName + "[_src]") + ";");
          }
          dedent();
          emit("}");
        }
        dedent();
        emit("}");
        // Trailing barrier: the scratch lives at the shared-mem floor and is
        // reused by the very next emission (a second reduce or convert_layout
        // also at offset 0). Without it, fast warps' stores to the reused
        // scratch race slow warps' reads here (tut-05 dwdb db corruption).
        blockSync();
        dedent();
        emit("}");
      }
    }
  } else {
    // Scalar result — full reduction
    bool isMultiResult = numResults > 1 && !src1Var.empty();

    // Broadcast-aware free-variable masks: a "free" input bit of the source
    // layout does not change the tensor element it maps to — registers/lanes/
    // warps differing only in free bits hold DUPLICATE data. Blindly combining
    // duplicates overcounts non-idempotent reduces (sum): e.g. a keepdims
    // cross-warp reduce leaves every warp with the full row-sum, and a
    // following scalar sum must NOT add the warps again (test_chain_reduce
    // returned exactly numWarps× the answer). Restrict each combine stage to
    // representatives whose free bits are zero.
    int regFreeMask = 0, laneFreeMask = 0, warpFreeMask = 0;
    if (auto srcFreeRtt = dyn_cast<RankedTensorType>(src.getType())) {
      auto freeLL = ttg::toLinearLayout(srcFreeRtt);
      auto freeMasks = freeLL.getFreeVariableMasks();
      auto *ctx = srcFreeRtt.getContext();
      regFreeMask = freeMasks.lookup(mlir::StringAttr::get(ctx, "register"));
      laneFreeMask = freeMasks.lookup(mlir::StringAttr::get(ctx, "lane"));
      warpFreeMask = freeMasks.lookup(mlir::StringAttr::get(ctx, "warp"));
    }

    // Generic multi-result scalar reduction (e.g. Welford mean/m2/weight in
    // test_generic_reduction). The argmin/argmax path below only understands
    // the 2-result compare+select pattern; any other multi-result combine
    // must inline the region via emitScalarCombine. No identity value exists
    // for an arbitrary combine, so accumulators are seeded from register 0
    // and every combine stage walks representatives explicitly.
    bool argMinMax = numResults == 2 && hasSelect;
    if (numResults > 1 && !argMinMax) {
      if ((int)op.getNumOperands() != numResults) {
        emitFailed = true;
        emitErrorMsg = "[emit_cuda] generic multi-result reduce: operand/"
                       "result count mismatch";
        op->emitError(emitErrorMsg);
        return;
      }
      SmallVector<std::string> accVars(numResults), accTypes(numResults);
      for (int k = 0; k < numResults; k++) {
        auto opnd = op.getOperands()[k];
        Type et = opnd.getType();
        if (auto rtt = dyn_cast<RankedTensorType>(et))
          et = rtt.getElementType();
        accTypes[k] = getCUDAType(et);
        accVars[k] = (k == 0) ? var : (k == 1 ? var1 : newVar("red"));
        valueToVar[op->getResult(k)] = accVars[k];
      }
      // Thread-local: seed from register 0, fold remaining distinct registers.
      for (int k = 0; k < numResults; k++)
        emit(accTypes[k] + " " + accVars[k] + " = " +
             getElemExpr(op.getOperands()[k], "0") + ";");
      auto combineInto = [&](const SmallVector<std::string> &others) {
        SmallVector<std::string> args;
        for (auto &v : accVars)
          args.push_back(v);
        for (auto &v : others)
          args.push_back(v);
        SmallVector<std::string> res;
        if (!emitScalarCombine(op.getCombineOp(), args, res) ||
            (int)res.size() < numResults)
          return false;
        for (int k = 0; k < numResults; k++)
          emit(accVars[k] + " = " + res[k] + ";");
        return true;
      };
      for (int i = 1; i < srcElems; i++) {
        if (i & regFreeMask)
          continue; // register-broadcast duplicate
        SmallVector<std::string> others;
        for (int k = 0; k < numResults; k++)
          others.push_back(getElemExpr(op.getOperands()[k], std::to_string(i)));
        if (!combineInto(others))
          return;
      }
      // Warp shuffle over data-distinct lane bits.
      emit("// Warp-level reduction (generic combine)");
      for (int b = 4; b >= 0; b--) {
        if (laneFreeMask & (1 << b))
          continue;
        SmallVector<std::string> others;
        for (int k = 0; k < numResults; k++) {
          auto ov = newVar("ov");
          emit(accTypes[k] + " " + ov + " = __shfl_xor_sync(0xffffffff, " +
               accVars[k] + ", " + std::to_string(1 << b) + ");");
          others.push_back(ov);
        }
        if (!combineInto(others))
          return;
      }
      // Cross-warp: representatives combined sequentially on warp 0 lane 0
      // (no identity needed), result broadcast through shared memory.
      int gWarpDistinct = (effNumWarps - 1) & ~warpFreeMask;
      if (effNumWarps > 1 && gWarpDistinct != 0) {
        emit("// Cross-warp reduction (generic combine)");
        emit("{");
        indent();
        bool wbDyn = (totalNumWarps == numWarps);
        SmallVector<std::string> bufs(numResults);
        int gBase = (sharedMemOffset + 15) & ~15;
        for (int k = 0; k < numResults; k++) {
          bufs[k] = "_wbg_" + accVars[k];
          if (wbDyn) {
            int o = gBase + 8 * numWarps * k;
            emit(accTypes[k] + "* " + bufs[k] + " = (" + accTypes[k] +
                 "*)(shared_mem + " + std::to_string(o) +
                 "); // transient cross-warp scratch");
            peakSharedMem = std::max(peakSharedMem, o + numWarps * 8);
          } else {
            emit("__shared__ " + accTypes[k] + " " + bufs[k] + "[" +
                 std::to_string(effNumWarps) + "];");
          }
        }
        std::string stStmt = "if (lane_id == 0) {";
        for (int k = 0; k < numResults; k++)
          stStmt += " " + bufs[k] + "[warp_id] = " + accVars[k] + ";";
        stStmt += " }";
        emit(stStmt);
        blockSync();
        emit("if (warp_id == 0 && lane_id == 0) {");
        indent();
        for (int w = 1; w < effNumWarps; w++) {
          if (w & warpFreeMask)
            continue; // duplicate of a representative warp
          SmallVector<std::string> others;
          for (int k = 0; k < numResults; k++)
            others.push_back(bufs[k] + "[" + std::to_string(w) + "]");
          if (!combineInto(others))
            return;
        }
        for (int k = 0; k < numResults; k++)
          emit(bufs[k] + "[0] = " + accVars[k] + ";");
        dedent();
        emit("}");
        blockSync();
        for (int k = 0; k < numResults; k++)
          emit(accVars[k] + " = " + bufs[k] + "[0];");
        dedent();
        emit("}");
      }
      return;
    }

    // Determine how many threads actually hold valid data
    // For a 1D tensor with blocked layout, only totalT0 threads have data
    int totalActiveThreads = effNumWarps * 32; // default: all threads active
    if (auto srcRtt = dyn_cast<RankedTensorType>(src.getType())) {
      if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(srcRtt.getEncoding())) {
        auto shape = srcRtt.getShape();
        auto spt = blocked.getSizePerThread();
        auto tpw = blocked.getThreadsPerWarp();
        auto wpc = blocked.getWarpsPerCTA();
        if (shape.size() == 1) {
          totalActiveThreads = (tpw[0] * wpc[0]) * spt[0];
          if (totalActiveThreads > (int)shape[0])
            totalActiveThreads = shape[0];
        }
      }
    }
    bool needsThreadGuard = totalActiveThreads < effNumWarps * 32;
    // The thread guard de-duplicates differently: threads outside the active
    // range never accumulate and hold the IDENTITY (not duplicates), so the
    // blind combine over all lanes/warps is already correct. Applying the
    // free-bit skip on top would skip the cross-warp combine and leave the
    // guarded warps stuck at the identity value (broke test_reduce1d).
    // Disable the masks when the guard is active — the two mechanisms are
    // mutually exclusive.
    if (needsThreadGuard)
      regFreeMask = laneFreeMask = warpFreeMask = 0;

    // Promote half/bfloat16 to float for reduction accumulator
    // __shfl_xor_sync doesn't handle __half correctly
    bool promoteToFloat = (srcElemType.isF16() || srcElemType.isBF16());
    std::string accType = promoteToFloat ? "float" : cudaType;
    std::string accIdentity = identity;
    if (promoteToFloat) {
      if (reduceOp == "add") accIdentity = "0.0f";
      else if (reduceOp == "max") accIdentity = "(-INFINITY)";
      else if (reduceOp == "min") accIdentity = "INFINITY";
    }

    emit(accType + " " + var + " = " + accIdentity + ";");
    if (isMultiResult)
      emit(cuda1Type + " " + var1 + " = 0;");

    if (needsThreadGuard)
      emit("if (tid < " + std::to_string(totalActiveThreads) + ") {");
    std::string srcIdx = getElemExpr(op.getOperands()[0], "_i");
    std::string srcElem = promoteToFloat ? ("(float)" + srcIdx) : srcIdx;
    emit("#pragma unroll");
    if (isMultiResult) {
      // Multi-result: track both value and index
      emit("for (int _i = 0; _i < " + std::to_string(srcElems) + "; _i++) {");
      indent();
      std::string cmpOp = (reduceOp == "max") ? ">" : "<";
      emit("if (" + uCast(srcElem) + " " + cmpOp + " " + uCast(var) + ") {");
      indent();
      emit(var + " = " + srcElem + ";");
      emit(var1 + " = " + src1Var + "[_i];");
      dedent();
      emit("}");
      dedent();
      emit("}");
    } else if (regFreeMask != 0) {
      // Skip duplicate (register-broadcast) elements: only registers whose
      // free bits are zero carry distinct data.
      emit("for (int _i = 0; _i < " + std::to_string(srcElems) + "; _i++)");
      emit("    if ((_i & " + std::to_string(regFreeMask) + ") == 0) " + var +
           " = " + redExpr(var, srcElem) + ";");
    } else {
      emit("for (int _i = 0; _i < " + std::to_string(srcElems) + "; _i++)");
      emit("    " + var + " = " + redExpr(var, srcElem) + ";");
    }
    if (needsThreadGuard)
      emit("}");

    // Warp reduction
    emit("// Warp-level reduction");
    // Only use warp_reduce_sum/max helpers for float32 (they take float args)
    bool useFloatHelper = srcElemType.isF32() ||
                          (promoteToFloat && (srcElemType.isF16() || srcElemType.isBF16()));
    if (isMultiResult) {
      // For multi-result: shuffle both value and index
      emit("for (int _off = 16; _off > 0; _off /= 2) {");
      indent();
      emit(accType + " _other_val = __shfl_xor_sync(0xffffffff, " + var + ", _off);");
      emit(cuda1Type + " _other_idx = __shfl_xor_sync(0xffffffff, " + var1 + ", _off);");
      std::string cmpOp = (reduceOp == "max") ? ">" : "<";
      emit("if (" + uCast("_other_val") + " " + cmpOp + " " + uCast(var) + ") { " +
           var + " = _other_val; " + var1 + " = _other_idx; }");
      dedent();
      emit("}");
    } else if (laneFreeMask != 0) {
      // Lane-broadcast source: lanes differing only in free bits hold the
      // same element — shuffle ONLY across data-distinct lane bits, otherwise
      // a sum combines duplicates (overcount).
      for (int b = 4; b >= 0; b--) {
        if (laneFreeMask & (1 << b))
          continue;
        emit(var + " = " +
             redExpr(var, "__shfl_xor_sync(0xffffffff, " + var + ", " +
                              std::to_string(1 << b) + ")") +
             ";");
      }
    } else if (reduceOp == "add" && useFloatHelper)
      emit(var + " = warp_reduce_sum(" + var + ");");
    else if (reduceOp == "max" && useFloatHelper)
      emit(var + " = warp_reduce_max(" + var + ");");
    else {
      // Generic warp reduce via shuffle
      emit("for (int _off = 16; _off > 0; _off /= 2)");
      emit("    " + var + " = " +
           redExpr(var, "__shfl_xor_sync(0xffffffff, " + var + ", _off)") +
           ";");
    }

    // Cross-warp reduction. Warps differing only in FREE warp bits hold
    // identical data — if no warp bit is data-distinct, every warp already
    // holds the full result and the cross-warp combine must be skipped
    // entirely (summing the duplicates gave numWarps× the answer).
    int warpDistinctMask = (effNumWarps - 1) & ~warpFreeMask;
    if (effNumWarps > 1 && warpDistinctMask == 0 && !isMultiResult) {
      emit("// Cross-warp reduction skipped: all warps hold duplicate data");
    } else if (effNumWarps > 1) {
      emit("// Cross-warp reduction");
      emit("{");
      indent();
      auto bufName = "_wb_" + var;
      // Mirror PTX: cross-warp reduce scratch lives in the DYNAMIC shared
      // region at the current allocation floor (it is transient — written and
      // fully consumed between the surrounding barriers), NOT in static
      // __shared__. Static smem stacks on top of the dynamic peak and pushed
      // tight persistent kernels (task #44: 231424 dynamic + 4096 static)
      // past the 232448 per-block limit → cuLaunchKernel "invalid argument".
      // Under warp specialization keep per-site static buffers: concurrent
      // partitions at different reduce sites would race one shared floor slot.
      // The buffer is indexed by the GLOBAL warp_id (`_wb[warp_id]`), which
      // ranges over ALL totalNumWarps physical warps when the reduction runs in
      // the warp-specialized region (e.g. a uniform-scalar read in the kernel
      // body before the partition dispatch — all 12 warps execute it). Sizing it
      // by effNumWarps (the layout's warpsPerCTA) lets warps beyond that index
      // write OUT OF BOUNDS, corrupting adjacent shared memory (mbarriers / TMA
      // buffers) → "unspecified launch failure". Size by totalNumWarps so every
      // physical warp has its own slot; the read still consumes only the first
      // effNumWarps slots, which is correct when those warps hold the data.
      bool wbDynamic = (totalNumWarps == numWarps);
      int wbStaticWarps = std::max(effNumWarps, totalNumWarps);
      int wbOff = 0;
      if (wbDynamic) {
        wbOff = (sharedMemOffset + 15) & ~15;
        emit(accType + "* " + bufName + " = (" + accType + "*)(shared_mem + " +
             std::to_string(wbOff) + "); // transient cross-warp scratch");
        peakSharedMem = std::max(peakSharedMem, wbOff + numWarps * 8);
      } else {
        emit("__shared__ " + accType + " " + bufName + "[" +
             std::to_string(wbStaticWarps) + "];");
      }
      if (isMultiResult) {
        auto bufName1 = "_wb_" + var1;
        if (wbDynamic) {
          int wbOff1 = wbOff + numWarps * 8;
          emit(cuda1Type + "* " + bufName1 + " = (" + cuda1Type +
               "*)(shared_mem + " + std::to_string(wbOff1) +
               "); // transient cross-warp scratch");
          peakSharedMem = std::max(peakSharedMem, wbOff1 + numWarps * 8);
        } else {
          emit("__shared__ " + cuda1Type + " " + bufName1 + "[" +
               std::to_string(wbStaticWarps) + "];");
        }
        emit("if (lane_id == 0) { " + bufName + "[warp_id] = " + var + "; " +
             bufName1 + "[warp_id] = " + var1 + "; }");
        blockSync();
        emit("if (warp_id == 0) {");
        indent();
        emit(var + " = (lane_id < " + std::to_string(effNumWarps) +
             ") ? " + bufName + "[lane_id] : " + identity + ";");
        emit(var1 + " = (lane_id < " + std::to_string(effNumWarps) +
             ") ? " + bufName1 + "[lane_id] : 0;");
        emit("for (int _off = 16; _off > 0; _off /= 2) {");
        indent();
        emit(accType + " _other_val = __shfl_xor_sync(0xffffffff, " + var + ", _off);");
        emit(cuda1Type + " _other_idx = __shfl_xor_sync(0xffffffff, " + var1 + ", _off);");
        std::string cmpOp = (reduceOp == "max") ? ">" : "<";
        emit("if (" + uCast("_other_val") + " " + cmpOp + " " + uCast(var) + ") { " +
             var + " = _other_val; " + var1 + " = _other_idx; }");
        dedent();
        emit("}");
        dedent();
        emit("}");
        emit("// Broadcast");
        emit("if (warp_id == 0 && lane_id == 0) { " + bufName + "[0] = " + var + "; " +
             bufName1 + "[0] = " + var1 + "; }");
        blockSync();
        emit(var + " = " + bufName + "[0];");
        emit(var1 + " = " + bufName1 + "[0];");
      } else {
        emit("if (lane_id == 0) " + bufName + "[warp_id] = " + var + ";");
        blockSync();
        emit("if (warp_id == 0) {");
        indent();
        // Only representative warps (free warp bits == 0) contribute; warps
        // differing in free bits hold duplicates of a representative.
        std::string repCond = "lane_id < " + std::to_string(effNumWarps);
        if (warpFreeMask != 0)
          repCond += " && (lane_id & " + std::to_string(warpFreeMask) +
                     ") == 0";
        emit(var + " = (" + repCond + ") ? " + bufName + "[lane_id] : " +
             accIdentity + ";");
        if (reduceOp == "add" && useFloatHelper)
          emit(var + " = warp_reduce_sum(" + var + ");");
        else if (reduceOp == "max" && useFloatHelper)
          emit(var + " = warp_reduce_max(" + var + ");");
        else {
          emit("for (int _off = 16; _off > 0; _off /= 2)");
          emit("    " + var + " = " +
               redExpr(var, "__shfl_xor_sync(0xffffffff, " + var + ", _off)") +
               ";");
        }
        dedent();
        emit("}");
        emit("// Broadcast result from warp 0 to all warps");
        emit("if (warp_id == 0 && lane_id == 0) " + bufName + "[0] = " + var + ";");
        blockSync();
        emit(var + " = " + bufName + "[0];");
      }
      dedent();
      emit("}");
    }
  }
}
bool CUDACodeGen::emitScalarCombine(Region &region,
                                    llvm::ArrayRef<std::string> argExprs,
                                    llvm::SmallVectorImpl<std::string> &results) {
  Block &body = region.front();
  // Local SSA value -> scalar CUDA expression (a temp var name or a literal).
  llvm::DenseMap<Value, std::string> local;
  if (body.getNumArguments() != argExprs.size()) {
    emitFailed = true;
    emitErrorMsg = "[emit_cuda] combine region arity mismatch";
    return false;
  }
  for (auto [i, arg] : llvm::enumerate(body.getArguments()))
    local[arg] = argExprs[i];

  auto getE = [&](Value v) -> std::string {
    auto it = local.find(v);
    if (it != local.end()) return it->second;
    // Value captured from the enclosing scope (e.g. a key-mask constant in
    // routing's keyed-add). Inline constants directly; otherwise fall back to
    // the global SSA->var map for scalar captures.
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      Attribute a = cst.getValue();
      if (auto ia = dyn_cast<IntegerAttr>(a))
        return "(" + std::to_string(ia.getValue().getSExtValue()) + ")";
      if (auto fa = dyn_cast<FloatAttr>(a))
        return "(" + std::to_string(fa.getValueAsDouble()) + ")";
    }
    if (valueToVar.count(v)) return valueToVar[v];
    emitFailed = true;
    emitErrorMsg = "[emit_cuda] scan/reduce combine references an "
                   "unresolved captured value";
    return "/*?*/0";
  };
  // Bind op result to a freshly-emitted temp holding `expr`.
  auto bind = [&](Value res, const std::string &expr) {
    std::string t = newVar("_ct");
    emit(getCUDAType(res.getType()) + " " + t + " = " + expr + ";");
    local[res] = t;
  };

  for (Operation &o : body) {
    // Terminator: collect the yielded scalar expressions.
    if (isa<tt::ScanReturnOp>(o) || isa<tt::ReduceReturnOp>(o)) {
      for (Value v : o.getOperands()) results.push_back(getE(v));
      return true;
    }
    if (auto c = dyn_cast<arith::ConstantOp>(&o)) {
      Attribute a = c.getValue();
      std::string lit;
      if (auto ia = dyn_cast<IntegerAttr>(a))
        lit = std::to_string(ia.getValue().getSExtValue());
      else if (auto fa = dyn_cast<FloatAttr>(a))
        lit = std::to_string(fa.getValueAsDouble());
      else { emitFailed = true; emitErrorMsg = "[emit_cuda] combine: bad const"; return false; }
      local[c.getResult()] = "(" + lit + ")";
      continue;
    }
    std::string a = o.getNumOperands() > 0 ? getE(o.getOperand(0)) : "";
    std::string b = o.getNumOperands() > 1 ? getE(o.getOperand(1)) : "";
    Value r = o.getNumResults() == 1 ? o.getResult(0) : Value();
    // Integer / float binary arithmetic.
    if (isa<arith::AddIOp, arith::AddFOp>(o)) { bind(r, "(" + a + " + " + b + ")"); continue; }
    if (isa<arith::SubIOp, arith::SubFOp>(o)) { bind(r, "(" + a + " - " + b + ")"); continue; }
    if (isa<arith::MulIOp, arith::MulFOp>(o)) { bind(r, "(" + a + " * " + b + ")"); continue; }
    if (isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(o)) { bind(r, "(" + a + " / " + b + ")"); continue; }
    if (isa<arith::RemSIOp, arith::RemUIOp>(o)) { bind(r, "(" + a + " % " + b + ")"); continue; }
    if (isa<arith::AndIOp>(o)) { bind(r, "(" + a + " & " + b + ")"); continue; }
    if (isa<arith::OrIOp>(o))  { bind(r, "(" + a + " | " + b + ")"); continue; }
    if (isa<arith::XOrIOp>(o)) { bind(r, "(" + a + " ^ " + b + ")"); continue; }
    if (isa<arith::ShLIOp>(o)) { bind(r, "(" + a + " << " + b + ")"); continue; }
    if (isa<arith::ShRUIOp, arith::ShRSIOp>(o)) { bind(r, "(" + a + " >> " + b + ")"); continue; }
    if (isa<arith::MaxSIOp, arith::MaxUIOp>(o)) { bind(r, "max(" + a + ", " + b + ")"); continue; }
    if (isa<arith::MinSIOp, arith::MinUIOp>(o)) { bind(r, "min(" + a + ", " + b + ")"); continue; }
    if (isa<arith::MaxNumFOp, arith::MaximumFOp>(o)) { bind(r, "fmaxf(" + a + ", " + b + ")"); continue; }
    if (isa<arith::MinNumFOp, arith::MinimumFOp>(o)) { bind(r, "fminf(" + a + ", " + b + ")"); continue; }
    if (auto sel = dyn_cast<arith::SelectOp>(&o)) {
      std::string c = getE(sel.getCondition());
      std::string tv = getE(sel.getTrueValue());
      std::string fv = getE(sel.getFalseValue());
      // Cast both branches to the result type: a bf16 operand against a
      // double literal makes the ternary ambiguous for nvcc.
      std::string T = getCUDAType(r.getType());
      bind(r, "(" + c + " ? (" + T + ")(" + tv + ") : (" + T + ")(" + fv +
              "))");
      continue;
    }
    if (auto ci = dyn_cast<arith::CmpIOp>(&o)) {
      const char *opv = nullptr; bool u = false;
      switch (ci.getPredicate()) {
        case arith::CmpIPredicate::eq: opv = "=="; break;
        case arith::CmpIPredicate::ne: opv = "!="; break;
        case arith::CmpIPredicate::slt: opv = "<"; break;
        case arith::CmpIPredicate::sle: opv = "<="; break;
        case arith::CmpIPredicate::sgt: opv = ">"; break;
        case arith::CmpIPredicate::sge: opv = ">="; break;
        case arith::CmpIPredicate::ult: opv = "<"; u = true; break;
        case arith::CmpIPredicate::ule: opv = "<="; u = true; break;
        case arith::CmpIPredicate::ugt: opv = ">"; u = true; break;
        case arith::CmpIPredicate::uge: opv = ">="; u = true; break;
      }
      std::string lhs = a, rhs = b;
      if (u) { lhs = "(unsigned)(" + a + ")"; rhs = "(unsigned)(" + b + ")"; }
      bind(r, "(" + lhs + " " + opv + " " + rhs + ")");
      continue;
    }
    if (auto cf = dyn_cast<arith::CmpFOp>(&o)) {
      const char *opv = nullptr;
      switch (cf.getPredicate()) {
        case arith::CmpFPredicate::OEQ: case arith::CmpFPredicate::UEQ: opv = "=="; break;
        case arith::CmpFPredicate::ONE: case arith::CmpFPredicate::UNE: opv = "!="; break;
        case arith::CmpFPredicate::OLT: case arith::CmpFPredicate::ULT: opv = "<"; break;
        case arith::CmpFPredicate::OLE: case arith::CmpFPredicate::ULE: opv = "<="; break;
        case arith::CmpFPredicate::OGT: case arith::CmpFPredicate::UGT: opv = ">"; break;
        case arith::CmpFPredicate::OGE: case arith::CmpFPredicate::UGE: opv = ">="; break;
        default: opv = "=="; break;
      }
      bind(r, "(" + a + " " + opv + " " + b + ")");
      continue;
    }
    // Casts: just reinterpret through the C type.
    if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::TruncFOp,
            arith::ExtFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
            arith::FPToUIOp, arith::BitcastOp>(o)) {
      bind(r, "((" + getCUDAType(r.getType()) + ")" + a + ")");
      continue;
    }
    if (auto assertOp = dyn_cast<tt::AssertOp>(&o)) {
      // tl.device_assert inside a combine region (e.g. sanitize_add). The
      // condition is a scalar here, so emit a plain guarded __assert_fail.
      auto escape = [](llvm::StringRef s) {
        std::string out;
        for (char c : s) {
          if (c == '"' || c == '\\')
            out += '\\';
          if (c == '\n') {
            out += "\\n";
            continue;
          }
          out += c;
        }
        return out;
      };
      std::string file = "unknown", func = "unknown";
      int line = 0;
      Location loc = assertOp.getLoc();
      while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
        loc = callLoc.getCallee();
      while (auto nameLoc = dyn_cast<NameLoc>(loc))
        loc = nameLoc.getChildLoc();
      if (auto flc = dyn_cast<FileLineColLoc>(loc)) {
        file = flc.getFilename().str();
        line = flc.getLine();
      }
      emit("if (!(" + a + ")) __assert_fail(\"" +
           escape(assertOp.getMessage()) + "\", \"" + escape(file) + "\", " +
           std::to_string(line) + ", \"" + escape(func) + "\");");
      continue;
    }
    emitFailed = true;
    emitErrorMsg = "[emit_cuda] unsupported op in scan/reduce combine: " +
                   o.getName().getStringRef().str();
    o.emitError(emitErrorMsg);
    return false;
  }
  emitFailed = true;
  emitErrorMsg = "[emit_cuda] combine region has no terminator";
  return false;
}

void CUDACodeGen::emitScan(tt::ScanOp op) {
  // N-operand inclusive scan (tl.cumsum / tl.associative_scan with tuples).
  // Mirrors the PTX backend's ScanOpToLLVM fast-scan: (1) intra-thread
  // contiguous scan, (2) intra-warp shuffle scan of each chunk's last element,
  // (3) cross-warp combine via shared memory (or a register-only path when
  // only one warp owns the axis).
  // Only blocked layouts are supported (same restriction as the PTX backend).
  int N = op.getNumOperands();
  if (op->getNumResults() != N || N < 1) {
    emitFailed = true;
    emitErrorMsg = "[emit_cuda] tt.scan operand/result count mismatch";
    op->emitError(emitErrorMsg);
    return;
  }
  auto src = op.getOperands()[0];
  int axis = op.getAxis();
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  auto blocked =
      srcTy ? dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding()) : nullptr;
  if (!blocked) {
    emitFailed = true;
    emitErrorMsg = "[emit_cuda] tt.scan requires a blocked layout";
    op->emitError(emitErrorMsg);
    return;
  }

  // Per-operand element types (a tuple scan may mix types, e.g. cummax's
  // float value + int64 index). All operands share the same shape/encoding.
  SmallVector<std::string> cudaTypes(N);
  for (int k = 0; k < N; k++) {
    auto ty = cast<RankedTensorType>(op.getOperands()[k].getType());
    cudaTypes[k] = getCUDAType(ty.getElementType());
  }

  // The combine is inlined generically (see emitScalarCombine) so arbitrary
  // associative_scan combines are handled correctly, not just add/min/max.
  // Region argument order is (acc tuple..., current tuple...); for a
  // non-commutative combine the accumulated/earlier tuple is FIRST. Results
  // are copied into fresh temps so callers can assign them back to the very
  // values that fed the combine without aliasing hazards.
  auto combV = [&](const SmallVector<std::string> &a,
                   const SmallVector<std::string> &b) -> SmallVector<std::string> {
    SmallVector<std::string> args(a.begin(), a.end());
    args.append(b.begin(), b.end());
    SmallVector<std::string> res;
    if (!emitScalarCombine(op.getCombineOp(), args, res) ||
        (int)res.size() < N) {
      // emitScalarCombine already set emitFailed / error message.
      return a;
    }
    SmallVector<std::string> out(N);
    for (int k = 0; k < N; k++) {
      out[k] = newVar("cmb");
      emit(cudaTypes[k] + " " + out[k] + " = " + res[k] + ";");
    }
    return out;
  };
  // Scalar convenience for single-operand combines used in ternaries.
  auto comb = [&](const std::string &a, const std::string &b) -> std::string {
    return combV({a}, {b})[0];
  };
  (void)comb;

  // Layout-derived constants (mirror ScanLoweringHelper).
  auto spt = blocked.getSizePerThread();
  auto tpw = blocked.getThreadsPerWarp();
  auto wpc = blocked.getWarpsPerCTA();
  auto order = blocked.getOrder();
  auto prod = [](ArrayRef<unsigned> v) { int p = 1; for (auto x : v) p *= x; return p; };
  int Sa = spt[axis];
  int elementStride = 1, threadStride = 1, warpAxisStride = 1;
  for (unsigned d : order) { if ((int)d == axis) break; elementStride *= spt[d]; }
  for (unsigned d : order) { if ((int)d == axis) break; threadStride *= tpw[d]; }
  for (unsigned d : order) { if ((int)d == axis) break; warpAxisStride *= wpc[d]; }
  int scanDim = tpw[axis];
  int axisNumWarps = wpc[axis];
  int nThreadsPerWarp = prod(tpw);
  int nonAxisThreadsPerWarp = nThreadsPerWarp / scanDim;
  int nWarps = prod(wpc);
  int numParallelLane = (nWarps / axisNumWarps) * nonAxisThreadsPerWarp;
  int axisShape = srcTy.getShape()[axis];
  // Threads along the axis that hold UNIQUE data. When threadsPerWarp[axis]
  // exceeds what the axis length needs (e.g. a 16-element tensor with
  // threadsPerWarp[axis]=32), the surplus lanes are broadcast replicas holding
  // identical input. The warp shuffle scan must only span the unique lanes;
  // otherwise replica lanes accumulate across the duplicated sequence and end
  // up inconsistent with their canonical counterpart. Mirrors the PTX backend's
  // getAxisNumThreadsPerWarpWithUniqueData.
  int scanDimUnique = std::min(scanDim, (axisShape + Sa - 1) / Sa);
  int perCTA = Sa * scanDim * axisNumWarps;
  int numScanBlocks = (axisShape + perCTA - 1) / perCTA;
  // Multi-block constants (mirror ScanLoweringHelper). When the axis is longer
  // than one CTA pass (perCTA elements), each thread's axis registers span
  // `numScanBlocks` blocks that are prefix-combined sequentially via a
  // persistent per-(parallel-block,parallel-element) accumulator.
  auto rank = srcTy.getShape().size();
  auto ceilDivI = [](int a, int b) { return (a + b - 1) / b; };
  auto coverOf = [&](int d) { return (int)(spt[d] * tpw[d] * wpc[d]); };
  int Pe = 1;
  for (size_t d = 0; d < rank; d++)
    if ((int)d != axis) Pe *= spt[d];
  int numParallelBlocks = 1;
  for (size_t d = 0; d < rank; d++)
    if ((int)d != axis)
      numParallelBlocks *= ceilDivI(srcTy.getShape()[d], coverOf(d));
  int blockStride = 1;
  for (unsigned d : order) {
    if ((int)d == axis) break;
    blockStride *= ceilDivI(srcTy.getShape()[d], coverOf(d));
  }
  // Warps holding UNIQUE data along the axis. When warpsPerCTA[axis] exceeds
  // what the axis length needs, the extra warps replicate the data. A replica
  // warp maps to its unique counterpart via `warpAxis % uniqueWarps`, so it
  // stores/reads the same shared-memory slot and computes the same result.
  int warpsNeeded = (axisShape + Sa * scanDim - 1) / (Sa * scanDim);
  int axisNumWarpsUnique = std::min(axisNumWarps, warpsNeeded);

  int srcElems = getElemsPerThread(src);
  int numChunks = srcElems / Sa;

  // Axis lane/warp id + flat parallel-thread id expressions.
  std::string laneAxisE = "((lane_id/" + std::to_string(threadStride) + ")%" +
                          std::to_string(scanDim) + ")";
  std::string warpAxisE = "(((warp_id/" + std::to_string(warpAxisStride) +
                          ")%" + std::to_string(axisNumWarps) + ")%" +
                          std::to_string(axisNumWarpsUnique) + ")";
  std::string lanePar = "0";
  { int pstride = 1, lstride = 1;
    for (unsigned d : order) {
      if ((int)d != axis) {
        lanePar += " + ((lane_id/" + std::to_string(lstride) + ")%" +
                   std::to_string(tpw[d]) + ")*" + std::to_string(pstride);
        pstride *= tpw[d];
      }
      lstride *= tpw[d];
    } }
  std::string warpPar = "0";
  { int pstride = 1, lstride = 1;
    for (unsigned d : order) {
      if ((int)d != axis) {
        warpPar += " + ((warp_id/" + std::to_string(lstride) + ")%" +
                   std::to_string(wpc[d]) + ")*" + std::to_string(pstride);
        pstride *= wpc[d];
      }
      lstride *= wpc[d];
    } }
  std::string flatPar = "((" + lanePar + ") + (" + warpPar + ")*" +
                        std::to_string(nonAxisThreadsPerWarp) + ")";

  SmallVector<std::string> vars(N);
  for (int k = 0; k < N; k++) {
    vars[k] = newVar("scan");
    valueToVar[op->getResult(k)] = vars[k];
    emit(cudaTypes[k] + " " + vars[k] + "[" + std::to_string(srcElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(srcElems) + "; _i++)");
    emit("    " + vars[k] + "[_i] = " +
         getElemExpr(op.getOperands()[k], "_i") + ";");
  }
  const std::string &var = vars[0];
  emit("// tt.scan (inclusive) axis=" + std::to_string(axis) +
       " operands=" + std::to_string(N));

  std::string vLaneAxis = "_scLaneAx_" + var;
  std::string vWarpAxis = "_scWarpAx_" + var;
  std::string vFlatPar = "_scFlat_" + var;
  emit("int " + vLaneAxis + " = " + laneAxisE + ";");
  emit("int " + vWarpAxis + " = " + warpAxisE + ";");
  emit("int " + vFlatPar + " = " + flatPar + ";");

  // reverse=true is implemented as flip(scan(flip(x))) — mirrors the PTX
  // backend (ScanOpToLLVM flipSrcValues): reverse the per-thread register
  // order (mixed-radix digit complement) and butterfly lane-flip (xor 31,
  // equivalent to the upstream shuffleXor 16/8/4/2/1 chain), plus complement
  // the axis warp id. Non-axis coordinates are complemented too, but that
  // permutation is a bijection among parallel slices and is undone by the
  // second flip after the scan.
  bool reverse = op.getReverse();
  auto emitFlip = [&]() {
    for (int k = 0; k < N; k++) {
      if (srcElems > 1) {
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(srcElems / 2) +
             "; _i++) {");
        emit("    " + cudaTypes[k] + " _t = " + vars[k] + "[_i];");
        emit("    " + vars[k] + "[_i] = " + vars[k] + "[" +
             std::to_string(srcElems - 1) + " - _i];");
        emit("    " + vars[k] + "[" + std::to_string(srcElems - 1) +
             " - _i] = _t;");
        emit("}");
      }
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(srcElems) + "; _i++)");
      emit("    " + vars[k] + "[_i] = __shfl_xor_sync(0xffffffff, " + vars[k] +
           "[_i], 31);");
    }
  };
  if (reverse) {
    emit("// reverse scan: flip data so the axis runs forward");
    emitFlip();
    if (axisNumWarpsUnique > 1)
      emit(vWarpAxis + " = " + std::to_string(axisNumWarpsUnique - 1) + " - " +
           vWarpAxis + ";");
  }

  auto elemK = [&](int k, int i) {
    return vars[k] + "[" + std::to_string(i) + "]";
  };
  auto elems = [&](int i) {
    SmallVector<std::string> v(N);
    for (int k = 0; k < N; k++) v[k] = elemK(k, i);
    return v;
  };
  auto elem = [&](int i) { return elemK(0, i); };
  (void)elem;
  // Assign a combined tuple back, optionally guarded per element by `cond`
  // (combV results are fresh temps, so unconditional evaluation is safe).
  auto assignElems = [&](int i, const SmallVector<std::string> &v,
                         const std::string &cond = "") {
    for (int k = 0; k < N; k++)
      emit(elemK(k, i) + " = " +
           (cond.empty() ? v[k]
                         : "(" + cond + ") ? " + v[k] + " : " + elemK(k, i)) +
           ";");
  };

  // Step 1: intra-thread contiguous scan.
  {
    std::map<int, int> accLast;
    for (int si = 0; si < srcElems; si++) {
      int accIndex = (si % elementStride) +
                     ((si / elementStride) / Sa) * elementStride;
      auto it = accLast.find(accIndex);
      if (it != accLast.end())
        assignElems(si, combV(elems(it->second), elems(si)));
      accLast[accIndex] = si;
    }
  }

  // Step 2: intra-warp shuffle scan of each chunk's last element.
  if (scanDim > 1) {
    for (int si = 0; si < srcElems; si++) {
      int elementIdx = (si / elementStride) % Sa;
      if (elementIdx != Sa - 1) continue;
      emit("{");
      indent();
      SmallVector<std::string> a(N), s(N);
      for (int k = 0; k < N; k++) {
        a[k] = "_a" + std::to_string(k);
        s[k] = "_s" + std::to_string(k);
        emit(cudaTypes[k] + " " + a[k] + " = " + elemK(k, si) + ";");
        emit(cudaTypes[k] + " " + s[k] + ";");
      }
      for (int i = 1; i <= scanDimUnique / 2; i <<= 1) {
        for (int k = 0; k < N; k++)
          emit(s[k] + " = __shfl_up_sync(0xffffffff, " + a[k] + ", " +
               std::to_string(i * threadStride) + ");");
        auto r = combV(s, a);
        for (int k = 0; k < N; k++)
          emit("if (" + vLaneAxis + " >= " + std::to_string(i) + ") " + a[k] +
               " = " + r[k] + ";");
      }
      for (int k = 0; k < N; k++)
        emit(elemK(k, si) + " = " + a[k] + ";");
      dedent();
      emit("}");
    }
  }

  if (axisNumWarpsUnique > 1) {
    // Step 3 (slow path): cross-warp combine through shared memory (one
    // buffer per operand — element types may differ).
    int smemN = numParallelLane * axisNumWarpsUnique * numChunks;
    SmallVector<std::string> smem(N);
    for (int k = 0; k < N; k++) {
      smem[k] = "_scSmem" + std::to_string(k) + "_" + var;
      emit("__shared__ " + cudaTypes[k] + " " + smem[k] + "[" +
           std::to_string(smemN) + "];");
    }
    int chunkId = 0;
    for (int si = 0; si < srcElems; si++) {
      int elementIdx = (si / elementStride) % Sa;
      if (elementIdx != Sa - 1) continue;
      std::string idx =
          vFlatPar + " + " + vWarpAxis + "*" + std::to_string(numParallelLane) +
          " + " +
          std::to_string(chunkId * numParallelLane * axisNumWarpsUnique);
      for (int k = 0; k < N; k++)
        emit("if (" + vLaneAxis + " == " + std::to_string(scanDimUnique - 1) +
             ") " + smem[k] + "[" + idx + "] = " + elemK(k, si) + ";");
      chunkId++;
    }
    blockSync();
    // Persistent per-(parallel-block, parallel-element) accumulators carry the
    // running prefix across axis blocks (numScanBlocks>1). _sacc holds the
    // full running total; _smacc the warp-masked prefix used for this chunk.
    // Mirrors ScanOpToLLVM::AddPartialReduce.
    int nAcc = numParallelBlocks * Pe;
    SmallVector<std::string> sacc(N), smacc(N), pvar(N);
    for (int k = 0; k < N; k++) {
      sacc[k] = "_sacc" + std::to_string(k) + "_" + var;
      smacc[k] = "_smacc" + std::to_string(k) + "_" + var;
      emit(cudaTypes[k] + " " + sacc[k] + "[" + std::to_string(nAcc) + "];");
      emit(cudaTypes[k] + " " + smacc[k] + "[" + std::to_string(nAcc) + "];");
    }
    std::vector<bool> accInited(nAcc, false);
    chunkId = 0;
    for (int si = 0; si < srcElems; si++) {
      int elementIdx = (si / elementStride) % Sa;
      if (elementIdx != Sa - 1) continue;
      int blockId = chunkId / Pe;
      int parallelBlockId = blockId % blockStride +
                            ((blockId / blockStride) / numScanBlocks) *
                                blockStride;
      int accIdx = chunkId % Pe + parallelBlockId * Pe;
      int axisBlockId = (blockId / blockStride) % numScanBlocks;
      SmallVector<std::string> A(N), M(N);
      for (int k = 0; k < N; k++) {
        A[k] = sacc[k] + "[" + std::to_string(accIdx) + "]";
        M[k] = smacc[k] + "[" + std::to_string(accIdx) + "]";
      }
      emit("{");
      indent();
      for (int k = 0; k < N; k++) {
        pvar[k] = "_p" + std::to_string(k);
        emit(cudaTypes[k] + " " + pvar[k] + ";");
      }
      for (int i = 0; i < axisNumWarpsUnique; i++) {
        std::string idx =
            vFlatPar + " + " +
            std::to_string(numParallelLane * (i + chunkId * axisNumWarpsUnique));
        for (int k = 0; k < N; k++)
          emit(pvar[k] + " = " + smem[k] + "[" + idx + "];");
        if (!accInited[accIdx] && i == 0) {
          for (int k = 0; k < N; k++)
            emit(A[k] + " = " + pvar[k] + "; " + M[k] + " = " + pvar[k] + ";");
          accInited[accIdx] = true;
        } else {
          auto r = combV(A, pvar);
          for (int k = 0; k < N; k++) {
            emit(A[k] + " = " + r[k] + ";");
            emit("if (" + vWarpAxis + " >= " + std::to_string(i + 1) + ") " +
                 M[k] + " = " + A[k] + ";");
          }
        }
      }
      {
        auto r = combV(M, elems(si));
        assignElems(si, r,
                    axisBlockId == 0 ? vWarpAxis + " != 0" : std::string());
      }
      if (Sa > 1) {
        SmallVector<std::string> last(N);
        for (int k = 0; k < N; k++) {
          last[k] = "_last" + std::to_string(k);
          emit(cudaTypes[k] + " " + last[k] +
               " = __shfl_up_sync(0xffffffff, " + elemK(k, si) + ", " +
               std::to_string(threadStride) + ");");
          emit(last[k] + " = (" + vLaneAxis + " != 0) ? " + last[k] + " : " +
               M[k] + ";");
        }
        for (int k2 = 1; k2 < Sa; k2++) {
          int ei = si - k2 * elementStride;
          auto r = combV(last, elems(ei));
          assignElems(ei, r,
                      axisBlockId == 0 ? "(" + vWarpAxis + " != 0) || (" +
                                             vLaneAxis + " != 0)"
                                       : std::string());
        }
      }
      // Carry the full running total into the next axis block.
      for (int k = 0; k < N; k++)
        emit(M[k] + " = " + A[k] + ";");
      dedent();
      emit("}");
      chunkId++;
    }
  } else if (numScanBlocks > 1) {
    // Single warp owns the axis but it spans multiple axis blocks: carry the
    // running prefix across blocks in per-thread registers (mirrors the PTX
    // backend's AddPartialReduceOneWarp). No shared memory needed — all axis
    // data lives within one warp. This branch also performs the within-chunk
    // lane propagation, so the register-only fast path below must not run.
    int nAcc = numParallelBlocks * Pe;
    SmallVector<std::string> sacc(N);
    for (int k = 0; k < N; k++) {
      sacc[k] = "_sacc" + std::to_string(k) + "_" + var;
      emit(cudaTypes[k] + " " + sacc[k] + "[" + std::to_string(nAcc) + "];");
    }
    // Lane holding the last unique axis element of this chunk (the source of
    // the cross-block carry).
    std::string laneLast = "(lane_id + (" + std::to_string(scanDimUnique - 1) +
                           " - " + vLaneAxis + ")*" +
                           std::to_string(threadStride) + ")";
    int chunkId = 0;
    for (int si = 0; si < srcElems; si++) {
      int elementIdx = (si / elementStride) % Sa;
      if (elementIdx != Sa - 1) continue;
      int blockId = chunkId / Pe;
      int parallelBlockId = blockId % blockStride +
                            ((blockId / blockStride) / numScanBlocks) *
                                blockStride;
      int accIdx = chunkId % Pe + parallelBlockId * Pe;
      int axisBlockId = (blockId / blockStride) % numScanBlocks;
      SmallVector<std::string> A(N);
      for (int k = 0; k < N; k++)
        A[k] = sacc[k] + "[" + std::to_string(accIdx) + "]";
      emit("{");
      indent();
      // Fold the previous blocks' running total into this chunk's last
      // element (step 2 already gave it the inclusive cross-lane prefix
      // within this block).
      if (axisBlockId > 0)
        assignElems(si, combV(A, elems(si)));
      if (scanDim > 1) {
        // Previous lane's inclusive prefix (or the carried accumulator on
        // lane 0) propagates into this thread's earlier chunk elements.
        SmallVector<std::string> last(N);
        for (int k = 0; k < N; k++) {
          last[k] = "_last" + std::to_string(k);
          emit(cudaTypes[k] + " " + last[k] +
               " = __shfl_up_sync(0xffffffff, " + elemK(k, si) + ", " +
               std::to_string(threadStride) + ");");
          if (axisBlockId > 0)
            emit(last[k] + " = (" + vLaneAxis + " == 0) ? " + A[k] + " : " +
                 last[k] + ";");
        }
        // New running total: chunk total from the last unique axis lane.
        for (int k = 0; k < N; k++)
          emit(A[k] + " = __shfl_sync(0xffffffff, " + elemK(k, si) + ", " +
               laneLast + ");");
        for (int k2 = 1; k2 < Sa; k2++) {
          int ei = si - k2 * elementStride;
          auto r = combV(last, elems(ei));
          assignElems(ei, r,
                      axisBlockId == 0 ? vLaneAxis + " != 0" : std::string());
        }
      } else {
        // scanDim==1: a single thread owns the whole axis within the block;
        // the carry for earlier chunk elements is the (old) accumulator.
        if (axisBlockId > 0)
          for (int k2 = 1; k2 < Sa; k2++) {
            int ei = si - k2 * elementStride;
            assignElems(ei, combV(A, elems(ei)));
          }
        for (int k = 0; k < N; k++)
          emit(A[k] + " = " + elemK(k, si) + ";");
      }
      dedent();
      emit("}");
      chunkId++;
    }
  } else if (Sa > 1 && scanDim > 1) {
    // Step 3 (fast path): single warp owns the axis; propagate each thread's
    // last-element prefix (from the previous lane) to its earlier contiguous
    // elements. When scanDim==1 a single thread owns the whole axis and step 1
    // already produced the full inclusive scan, so there is nothing to do.
    for (int si = 0; si < srcElems; si++) {
      int elementIdx = (si / elementStride) % Sa;
      if (elementIdx != Sa - 1) continue;
      emit("{");
      indent();
      SmallVector<std::string> last(N);
      for (int k = 0; k < N; k++) {
        last[k] = "_last" + std::to_string(k);
        emit(cudaTypes[k] + " " + last[k] + " = __shfl_up_sync(0xffffffff, " +
             elemK(k, si) + ", " + std::to_string(threadStride) + ");");
      }
      for (int k2 = 1; k2 < Sa; k2++) {
        int ei = si - k2 * elementStride;
        auto r = combV(last, elems(ei));
        assignElems(ei, r, vLaneAxis + " != 0");
      }
      dedent();
      emit("}");
    }
  }

  // Broadcast-correct replica lanes. When threadsPerWarp[axis] exceeds the
  // unique axis lanes, lanes with axis-id >= scanDimUnique are duplicates whose
  // shuffle scan accumulated across the replicated sequence and are therefore
  // inconsistent with their canonical lane. Copy each canonical lane's final
  // result back to its replicas so downstream stores (which fire on every lane)
  // see identical data and don't race to the same address.
  if (scanDimUnique < scanDim) {
    std::string srcLane =
        "(lane_id - ((" + vLaneAxis + " - " + vLaneAxis + " % " +
        std::to_string(scanDimUnique) + ") * " + std::to_string(threadStride) +
        "))";
    for (int k = 0; k < N; k++) {
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(srcElems) + "; _i++)");
      emit("    " + vars[k] + "[_i] = __shfl_sync(0xffffffff, " + vars[k] +
           "[_i], " + srcLane + ");");
    }
  }

  if (reverse) {
    emit("// reverse scan: flip results back to the original orientation");
    emitFlip();
  }
}

// tt.gather: out[I] = src[I with axis replaced by idx[I]]. Port of
// GatherOpToLLVM.cpp's emitGatherInShared (the universally-correct path): store
// the full src tensor into a row-major shared scratch at each element's LOGICAL
// position, barrier, then each output thread reads from the scratch using its
// own non-axis coordinates and the data-dependent index value along the gather
// axis. The warp-shuffle specialization in the reference is a perf-only variant;
// this shared path is numerically identical for every layout.
void CUDACodeGen::emitGather(tt::GatherOp op) {
  auto src = op.getSrc();
  auto idx = op.getIndices();
  auto result = op.getResult();
  auto srcRtt = cast<RankedTensorType>(src.getType());
  auto dstRtt = cast<RankedTensorType>(result.getType());
  unsigned axis = op.getAxis();
  auto *ctx = srcRtt.getContext();
  auto kReg = mlir::StringAttr::get(ctx, "register");
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  auto kWarp = mlir::StringAttr::get(ctx, "warp");
  auto kBlock = mlir::StringAttr::get(ctx, "block");

  auto srcLL = ttg::toLinearLayout(srcRtt);
  auto dstLL = ttg::toLinearLayout(dstRtt);

  // Multi-CTA (CGA) gather would need cross-CTA data movement, which a per-CTA
  // shared scratch cannot provide — refuse rather than miscompile.
  if (srcLL.getInDimSize(kBlock) > 1 || dstLL.getInDimSize(kBlock) > 1) {
    emitFailed = true;
    emitErrorMsg = "[emit_cuda] tt.gather across CTAs (num_ctas>1) unsupported";
    return;
  }

  auto srcShape = srcRtt.getShape();
  int rank = srcShape.size();
  // Row-major strides over the src LOGICAL shape (scratch layout).
  SmallVector<int64_t> srcStrides(rank);
  srcStrides.back() = 1;
  for (int d = rank - 2; d >= 0; d--)
    srcStrides[d] = srcStrides[d + 1] * srcShape[d + 1];
  int64_t totalSrcElems = srcStrides[0] * srcShape[0];

  int nSrc = getElemsPerThread(src);
  int nDst = getElemsPerThread(result);
  auto cudaType = getCUDAType(srcRtt.getElementType());
  int eb = getTypeSizeInBytes(srcRtt.getElementType());

  auto var = newVar("gather");
  valueToVar[result] = var;
  emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");

  // ---- Warp-local fast path (PTX-parity with emitWarpLocalGather) ----
  // When the gather column is fully owned within a single lane's warp (no
  // cross-warp / cross-CTA data movement needed), lower to warp index-shuffles
  // instead of a shared-memory round-trip. This is the path the PTX backend
  // selects for `efficient_layout` gathers; mirroring it keeps test parity and
  // emits `__shfl_sync` directly. Algorithm (see GatherOpToLLVM.cpp):
  //   (src_lane, src_reg) = ll_src^-1( ll_idx(...)[otherDims], idxVal )
  // then for each candidate src register in the column, shuffle from src_lane
  // and select by src_reg.
  if (GatherLoweringHelper(op).isWarpLocal()) {
    auto idxRtt = cast<RankedTensorType>(idx.getType());
    auto idxLL = ttg::toLinearLayout(idxRtt);
    auto kGatherDim = mlir::StringAttr::get(ctx, "dim" + std::to_string(axis));
    SmallVector<mlir::StringAttr> allDims, otherDims;
    for (int d = 0; d < rank; d++) {
      auto dn = mlir::StringAttr::get(ctx, "dim" + std::to_string(d));
      allDims.push_back(dn);
      if (d != (int)axis)
        otherDims.push_back(dn);
    }
    // Invert src: tensor-dims -> {register, lane} (warp/block invariant here).
    auto invSrc = srcLL.pseudoinvert().sublayout(allDims, {kReg, kLane});
    auto idxCol = idxLL.sublayout({kBlock, kWarp, kLane, kReg}, otherDims);
    // Per-column src register enumeration (all compile-time).
    auto invRegMap =
        srcLL.pseudoinvert().sublayout(allDims, {kReg}).removeZeroBasesAlongDim(
            kGatherDim);
    unsigned numRegsPerColumn = invRegMap.getInDimSize(kGatherDim);
    auto invRegMapColPart = invRegMap.sublayout(otherDims, {kReg});
    auto invRegMapRest = invRegMap.sublayout({kGatherDim}, {kReg});
    auto idxRegToCol = idxLL.sublayout({kReg}, otherDims);

    // Evaluate a linear layout output as a runtime C++ int expr: XOR over every
    // input bit that is set, of that bit's output basis (GF(2)-linear, L(0)=0).
    auto applyLL =
        [&](const triton::LinearLayout &ll, mlir::StringAttr outName,
            ArrayRef<std::pair<mlir::StringAttr, std::string>> ins)
        -> std::string {
      int outPos = ll.getOutDimIndex(outName);
      const auto &bases = ll.getBases();
      std::string e = "0";
      for (auto &kv : ins) {
        auto it = bases.find(kv.first);
        if (it == bases.end())
          continue;
        const auto &inBases = it->second;
        for (size_t b = 0; b < inBases.size(); b++) {
          int32_t basis = inBases[b][outPos];
          if (basis != 0)
            e += " ^ ((((" + kv.second + ") >> " + std::to_string(b) +
                 ") & 1) * " + std::to_string(basis) + ")";
        }
      }
      return e;
    };

    emit("{");
    indent();
    for (int r = 0; r < nDst; r++) {
      emit("// gather reg " + std::to_string(r) + " (warp-local)");
      // Column coords: otherDims from idxCol(reg=r, lane, warp); axis from idx.
      SmallVector<std::pair<mlir::StringAttr, std::string>> colIns(rank);
      for (auto dn : otherDims) {
        std::string ce = applyLL(idxCol, dn,
                                 {{kBlock, "0"},
                                  {kWarp, "warp_id"},
                                  {kLane, "lane_id"},
                                  {kReg, std::to_string(r)}});
        int d = std::stoi(dn.str().substr(3));
        colIns[d] = {dn, "(" + ce + ")"};
      }
      colIns[axis] = {kGatherDim,
                      "(int)(" + getElemExpr(idx, std::to_string(r)) + ")"};
      // src lane and register as runtime exprs.
      std::string srcLaneE = applyLL(invSrc, kLane, colIns);
      std::string srcRegE = applyLL(invSrc, kReg, colIns);
      std::string lv = "_gl" + std::to_string(r);
      std::string rv = "_gr" + std::to_string(r);
      emit("int " + lv + " = " + srcLaneE + ";");
      emit("int " + rv + " = " + srcRegE + ";");
      // Compile-time base register for this idx reg's column.
      auto normCol = idxRegToCol.apply({{kReg, r}});
      int32_t srcBase = invRegMapColPart.apply(normCol).front().second;
      std::string acc = "_ga" + std::to_string(r);
      emit(cudaType + " " + acc + " = " + getElemExpr(src, "0") + ";");
      for (unsigned i = 0; i < numRegsPerColumn; i++) {
        int32_t rest = invRegMapRest.apply({{kGatherDim, (int)i}}).front().second;
        int32_t srcRegIdx = srcBase ^ rest;
        // The __shfl_sync must be executed unconditionally by ALL lanes in the
        // mask — divergent participation (e.g. hiding it inside the ?: branch)
        // is undefined behavior. Hoist the shuffle into a temp, then select.
        std::string sh = "_gs" + std::to_string(r) + "_" + std::to_string(i);
        emit(cudaType + " " + sh + " = __shfl_sync(0xffffffff, " +
             getElemExpr(src, std::to_string(srcRegIdx)) + ", " + lv + ");");
        emit(acc + " = (" + std::to_string(srcRegIdx) + " == " + rv + ") ? " +
             sh + " : " + acc + ";");
      }
      emit(var + "[" + std::to_string(r) + "] = " + acc + ";");
    }
    dedent();
    emit("}");
    return;
  }

  // Per-dim LOGICAL coordinate as a C++ int expression for register i of a
  // tensor with linear layout `ll`. The register part is a compile-time const;
  // lane/warp contributions are XOR-combined at runtime (the layout is GF(2)
  // linear, so XOR is exact even when bases share bit positions).
  auto perDimCoords = [&](const triton::LinearLayout &ll,
                          int i) -> SmallVector<std::string> {
    const auto &bases = ll.getBases();
    const auto &laneBases = bases.find(kLane)->second;
    const auto &warpBases = bases.find(kWarp)->second;
    auto regCoords =
        ll.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<std::string> exprs(rank);
    for (int d = 0; d < rank; d++) {
      std::string e = std::to_string((int64_t)regCoords[d].second);
      for (size_t lb = 0; lb < laneBases.size(); lb++) {
        int64_t b = laneBases[lb][d];
        if (b != 0)
          e += " ^ (((lane_id >> " + std::to_string(lb) + ") & 1) * " +
               std::to_string(b) + ")";
      }
      for (size_t wb = 0; wb < warpBases.size(); wb++) {
        int64_t b = warpBases[wb][d];
        if (b != 0)
          e += " ^ (((warp_id >> " + std::to_string(wb) + ") & 1) * " +
               std::to_string(b) + ")";
      }
      exprs[d] = e;
    }
    return exprs;
  };
  // Linearize per-dim coordinate expressions into the row-major src scratch.
  auto linearize = [&](const SmallVector<std::string> &coords) -> std::string {
    std::string off;
    for (int d = 0; d < rank; d++) {
      std::string term = "(" + coords[d] + ")";
      if (srcStrides[d] != 1)
        term += " * " + std::to_string(srcStrides[d]);
      off = off.empty() ? term : off + " + " + term;
    }
    return off;
  };

  // Transient row-major scratch holding the full src tensor (save/restore the
  // bump pointer so the space is reused, like the convert_layout scratch).
  int savedSmem = sharedMemOffset;
  int smemOff = (sharedMemOffset + 127) & ~127;
  sharedMemOffset = smemOff + totalSrcElems * eb;
  if (sharedMemOffset > peakSharedMem) peakSharedMem = sharedMemOffset;
  emit("{");
  indent();
  emit(cudaType + "* _gsh = (" + cudaType + "*)(shared_mem + " +
       std::to_string(smemOff) + "); // tt.gather src scratch");
  // Store every src element at its logical position (redundant for replicated
  // threads — identical value to the same address, harmless).
  for (int i = 0; i < nSrc; i++) {
    auto coords = perDimCoords(srcLL, i);
    emit("_gsh[" + linearize(coords) + "] = " +
         getElemExpr(src, std::to_string(i)) + ";");
  }
  emit("__syncthreads();");
  // Gather: replace the axis coordinate with the (data-dependent) index value.
  for (int i = 0; i < nDst; i++) {
    auto coords = perDimCoords(dstLL, i);
    coords[axis] = "(int)(" + getElemExpr(idx, std::to_string(i)) + ")";
    emit(var + "[" + std::to_string(i) + "] = _gsh[" + linearize(coords) + "];");
  }
  emit("__syncthreads();");
  dedent();
  emit("}");
  sharedMemOffset = savedSmem;
}

void CUDACodeGen::emitTrans(tt::TransOp op) {
  valueToVar[op.getResult()] = getVar(op.getSrc());
  emit("// tt.trans: alias");
}

void CUDACodeGen::emitBitcast(tt::BitcastOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("bc");
  valueToVar[result] = var;
  auto srcVar = getVar(src);
  Type rElemType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto cudaType = getCUDAType(rElemType);
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    Type sElemType = cast<RankedTensorType>(src.getType()).getElementType();
    bool srcDeferred = deferredAddPtr.count(src) || ptrBasedDeferred.count(src) ||
                       scalarBaseDeferred.count(src) || scalarValues.contains(src);
    if (isa<tt::PointerType>(sElemType) && srcDeferred) {
      // Source pointers may be deferred (base+offset, never materialized as an
      // array) or scalar; resolve each address via getPtrAddrExpr and cast.
      emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      emit("    " + var + "[_i] = (" + cudaType + ")" + getPtrAddrExpr(src, "_i") + ";");
    } else if (sElemType == rElemType) {
      emit(cudaType + "* " + var + " = " + srcVar + "; // bitcast alias");
    } else {
      emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      emit("    { auto _tmp = " + srcVar + "[_i]; memcpy(&" + var + "[_i], &_tmp, sizeof(_tmp)); }");
    }
  } else {
    emit(cudaType + " " + var + "; { auto _tmp = " + srcVar + "; memcpy(&" + var + ", &_tmp, sizeof(_tmp)); }");
  }
}

void CUDACodeGen::emitPtrToInt(tt::PtrToIntOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("pti");
  valueToVar[result] = var;
  auto srcVar = getVar(src);
  Type intType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto cudaType = getCUDAType(intType);
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
         var + "[_i] = (" + cudaType + ")(uint64_t)" + getPtrAddrExpr(src, "_i") + ";");
  } else {
    emit(cudaType + " " + var + " = (" + cudaType + ")(uint64_t)" + srcVar + ";");
  }
}

void CUDACodeGen::emitIntToPtr(tt::IntToPtrOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("itp");
  valueToVar[result] = var;
  auto srcVar = getVar(src);
  Type ptrType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto cudaType = getCUDAType(ptrType);
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
         var + "[_i] = (" + cudaType + ")(uint64_t)" + srcVar + "[_i];");
  } else {
    emit(cudaType + " " + var + " = (" + cudaType + ")(uint64_t)" + srcVar + ";");
  }
}

void CUDACodeGen::emitFpToFp(tt::FpToFpOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("cvt");
  valueToVar[result] = var;

  auto srcVar = getVar(src);
  Type srcElemType = isTensor ? cast<RankedTensorType>(src.getType()).getElementType() : src.getType();
  Type dstElemType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto dstCudaType = getCUDAType(dstElemType);

  // Determine conversion function
  bool srcIsFp8 = isa<Float8E4M3FNType>(srcElemType) || isa<Float8E5M2Type>(srcElemType);
  bool dstIsFp8 = isa<Float8E4M3FNType>(dstElemType) || isa<Float8E5M2Type>(dstElemType);
  // Round-toward-zero downcasts (tl.x.to(dst, fp_downcast_rounding='rtz')).
  // Default (absent attr or rtne) is round-to-nearest-even, which all the
  // intrinsics below implement; rtz needs dedicated _rz variants.
  bool rtz = op.getRounding() && *op.getRounding() == tt::RoundingMode::RTZ;

  std::string convertExpr;
  if (rtz) {
    if (srcElemType.isF32() && dstElemType.isF16()) {
      convertExpr = "__float2half_rz(SRC)";
    } else if (srcElemType.isF32() && dstElemType.isBF16()) {
      convertExpr = "__float2bfloat16_rz(SRC)";
    } else if (srcElemType.isF32() && isa<Float8E5M2Type>(dstElemType)) {
      convertExpr = "(" + dstCudaType + ")cvt_f32_to_e5m2_rz(SRC)";
    } else {
      emitFailed = true;
      emitErrorMsg = "fp_to_fp with rtz rounding unsupported for this "
                     "src/dst type combination";
      return;
    }
  } else if (srcIsFp8 && !dstIsFp8) {
    // fp8 → fp16/bf16/fp32: upcast via __half intermediate
    std::string fp8Type = isa<Float8E4M3FNType>(srcElemType) ? "__NV_E4M3" : "__NV_E5M2";
    if (dstElemType.isF16()) {
      convertExpr = "*((__half*)&(__nv_cvt_fp8_to_halfraw(SRC, " + fp8Type + ")))";
    } else if (dstElemType.isBF16()) {
      // fp8 → half → float → bf16
      convertExpr = "(__nv_bfloat16)(float)(*((__half*)&(__nv_cvt_fp8_to_halfraw(SRC, " + fp8Type + "))))";
    } else if (dstElemType.isF32()) {
      convertExpr = "(float)(*((__half*)&(__nv_cvt_fp8_to_halfraw(SRC, " + fp8Type + "))))";
    } else {
      convertExpr = "(double)(*((__half*)&(__nv_cvt_fp8_to_halfraw(SRC, " + fp8Type + "))))";
    }
  } else if (!srcIsFp8 && dstIsFp8) {
    // fp16/bf16/fp32 → fp8: downcast
    std::string fp8Type = isa<Float8E4M3FNType>(dstElemType) ? "__NV_E4M3" : "__NV_E5M2";
    if (srcElemType.isF16()) {
      convertExpr = "__nv_cvt_halfraw_to_fp8(*(__half_raw*)&(SRC), __NV_SATFINITE, " + fp8Type + ")";
    } else if (srcElemType.isBF16()) {
      convertExpr = "__nv_cvt_bfloat16raw_to_fp8(*(__nv_bfloat16_raw*)&(SRC), __NV_SATFINITE, " + fp8Type + ")";
    } else {
      // fp32 → fp8: direct single-step conversion (matches PTX F2FP.E4M3.F32,
      // avoids the f32→f16→fp8 double-rounding + extra F2F instructions).
      convertExpr = "__nv_cvt_float_to_fp8(SRC, __NV_SATFINITE, " + fp8Type + ")";
    }
  } else {
    // fp8 → fp8 or same type: use raw cast
    convertExpr = "(" + dstCudaType + ")SRC";
  }

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    // Vectorized fp32->fp8: convert element pairs with one packed F2FP
    // (float2 -> fp8x2), matching PTX's cvt.rn.satfinite.e5m2x2.f32 and halving
    // the conversion instruction count vs the scalar __nv_cvt_float_to_fp8 loop.
    if (!rtz && !srcIsFp8 && dstIsFp8 && srcElemType.isF32() &&
        (nElems % 2 == 0)) {
      std::string fp8Type =
          isa<Float8E4M3FNType>(dstElemType) ? "__NV_E4M3" : "__NV_E5M2";
      // __align__(16): a downstream warp-shuffle convert_layout reads this fp8
      // array as aligned 32-bit words (one __byte_perm gather per packed word)
      // instead of per-byte loads; the alignment makes those uint reads valid.
      emit("__align__(16) " + dstCudaType + " " + var + "[" +
           std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i += 2)");
      emit("    *(__nv_fp8x2_storage_t*)&" + var +
           "[_i] = __nv_cvt_float2_to_fp8x2(make_float2(" +
           getElemExpr(src, "_i") + ", " + getElemExpr(src, "(_i + 1)") +
           "), __NV_SATFINITE, " + fp8Type + ");");
      return;
    }
    emit(dstCudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    std::string expr = convertExpr;
    // Replace SRC with srcVar[_i] (or just srcVar for scalars)
    std::string srcExpr = getElemExpr(src, "_i");
    auto pos = expr.find("SRC");
    while (pos != std::string::npos) {
      expr.replace(pos, 3, srcExpr);
      pos = expr.find("SRC", pos + srcExpr.size());
    }
    emit("    " + var + "[_i] = " + expr + ";");
  } else {
    emit("const " + dstCudaType + " " + var + " = " +
         [&]{ auto e = convertExpr; auto p = e.find("SRC"); if (p != std::string::npos) e.replace(p, 3, srcVar); return e; }() + ";");
  }
}

void CUDACodeGen::emitFp4ToFp(ttg::Fp4ToFpOp op) {
  // ttg.fp4_to_fp unpacks packed fp4 (e2m1, 2 values per i8 byte) into bf16/f16.
  // Low nibble = first element, high nibble = second element. The result tensor
  // doubles the `axis` dimension; element at axis-coord c comes from the source
  // byte at axis-coord c/2, low nibble for even c, high nibble for odd c.
  //
  // For a simple (blocked) result layout the per-thread register order is exactly
  // interleaved (out[2i],out[2i+1] = nibbles of src byte i). For a dot-operand
  // result (kWidth=2) feeding WGMMA the register order is NOT a plain 2x of the
  // source (kWidth=1) order, so we resolve the dst-register -> (src-register,
  // nibble) mapping through the LinearLayouts.
  auto result = op.getResult();
  auto src = op.getSrc();
  auto var = newVar("fp4");
  valueToVar[result] = var;

  auto rtt = cast<RankedTensorType>(result.getType());
  auto srcRtt = cast<RankedTensorType>(src.getType());
  auto dstCudaType = getCUDAType(rtt.getElementType());
  int nDst = getElemsPerThread(result);
  int nSrc = getElemsPerThread(src);
  int axis = (int)op.getAxis();
  auto srcVar = getVar(src);

  auto dstLL = ttg::toLinearLayout(rtt);
  auto srcLL = ttg::toLinearLayout(srcRtt);
  auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
  auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
  auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
  auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");

  // src logical coord -> src register index
  std::map<SmallVector<int>, int> srcCoordToReg;
  for (int i = 0; i < nSrc; i++) {
    auto coords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<int> key;
    for (auto &c : coords) key.push_back(c.second);
    srcCoordToReg[key] = i;
  }

  // For each dst register, find (src register, nibble) and check if the mapping
  // is the trivial interleaved one (out[2i] low / out[2i+1] high of src[i]).
  SmallVector<int> dstToSrc(nDst, 0);
  SmallVector<int> dstNibble(nDst, 0);
  bool simpleInterleaved = (nDst == 2 * nSrc);
  for (int i = 0; i < nDst; i++) {
    auto dstCoords = dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<int> srcKey;
    int nib = 0;
    for (int d = 0; d < (int)dstCoords.size(); d++) {
      int c = dstCoords[d].second;
      if (d == axis) { nib = c & 1; c >>= 1; }
      srcKey.push_back(c);
    }
    auto it = srcCoordToReg.find(srcKey);
    dstToSrc[i] = (it != srcCoordToReg.end()) ? it->second : 0;
    dstNibble[i] = nib;
    if (dstToSrc[i] != i / 2 || dstNibble[i] != i % 2) simpleInterleaved = false;
  }

  emit(dstCudaType + " " + var + "[" + std::to_string(nDst) + "];");
  emit("{");
  indent();
  // e2m1 magnitudes keyed by the low 3 bits; bit3 is the sign.
  emit("const float _e2m1_lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};");
  if (simpleInterleaved) {
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nSrc) + "; _i++) {");
    indent();
    emit("unsigned char _b = (unsigned char)(" + getElemExpr(src, "_i") + ");");
    emit("float _lo = _e2m1_lut[_b & 7]; if (_b & 0x08) _lo = -_lo;");
    emit("float _hi = _e2m1_lut[(_b >> 4) & 7]; if (_b & 0x80) _hi = -_hi;");
    emit(var + "[2 * _i]     = (" + dstCudaType + ")_lo;");
    emit(var + "[2 * _i + 1] = (" + dstCudaType + ")_hi;");
    dedent();
    emit("}");
  } else {
    for (int i = 0; i < nDst; i++) {
      int sh = dstNibble[i] * 4;
      std::string b = "((unsigned char)(" + srcVar + "[" +
                      std::to_string(dstToSrc[i]) + "]))";
      std::string mag = "_e2m1_lut[(" + b + " >> " + std::to_string(sh) + ") & 7]";
      std::string sgn = "(" + b + " & " + std::to_string(0x08 << sh) + ")";
      emit("{ float _v = " + mag + "; if (" + sgn + ") _v = -_v; " + var + "[" +
           std::to_string(i) + "] = (" + dstCudaType + ")_v; }");
    }
  }
  dedent();
  emit("}");
}

void CUDACodeGen::emitExternElementwise(tt::ExternElementwiseOp op) {
  auto result = op.getResult();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("ext");
  valueToVar[result] = var;
  auto symbol = op.getSymbol().str();
  static const std::map<std::string, std::string> nvMap = {
      {"__nv_expf","expf"},{"__nv_exp2f","exp2f"},{"__nv_logf","logf"},
      {"__nv_log2f","log2f"},{"__nv_sqrtf","sqrtf"},{"__nv_rsqrtf","rsqrtf"},
      {"__nv_sinf","sinf"},{"__nv_cosf","cosf"},{"__nv_tanhf","tanhf"},
      {"__nv_erff","erff"},{"__nv_fabsf","fabsf"},
      {"__nv_asinf","asinf"},{"__nv_acosf","acosf"},{"__nv_atanf","atanf"},
      {"__nv_atan2f","atan2f"},{"__nv_sinhf","sinhf"},{"__nv_coshf","coshf"},
      {"__nv_floorf","floorf"},{"__nv_ceilf","ceilf"},{"__nv_truncf","truncf"},
      {"__nv_roundf","roundf"},{"__nv_fmodf","fmodf"},{"__nv_powf","powf"},
      {"__nv_fmaf","fmaf"},{"__nv_fminf","fminf"},{"__nv_fmaxf","fmaxf"},
      {"__nv_copysignf","copysignf"},{"__nv_nextafterf","nextafterf"},
      {"__nv_exp","exp"},{"__nv_exp2","exp2"},{"__nv_log","log"},
      {"__nv_log2","log2"},{"__nv_sqrt","sqrt"},{"__nv_rsqrt","rsqrt"},
      {"__nv_sin","sin"},{"__nv_cos","cos"},{"__nv_tanh","tanh"},
      {"__nv_erf","erf"},{"__nv_fabs","fabs"},
      {"__nv_asin","asin"},{"__nv_acos","acos"},{"__nv_atan","atan"},
      {"__nv_atan2","atan2"},{"__nv_sinh","sinh"},{"__nv_cosh","cosh"},
      {"__nv_floor","floor"},{"__nv_ceil","ceil"},{"__nv_trunc","trunc"},
      {"__nv_round","round"},{"__nv_fmod","fmod"},{"__nv_pow","pow"},
      {"__nv_fma","fma"},{"__nv_fmin","fmin"},{"__nv_fmax","fmax"},
      {"__nv_copysign","copysign"},{"__nv_nextafter","nextafter"},
  };
  auto it = nvMap.find(symbol);
  if (it != nvMap.end())
    symbol = it->second;
  else if (symbol.rfind("__nv_", 0) == 0)
    // libdevice naming convention: __nv_<name> is the CUDA math API <name>
    // (e.g. __nv_j0f → j0f, __nv_cyl_bessel_i0 → cyl_bessel_i0).
    symbol = symbol.substr(5);
  Type rElemType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto cudaType = getCUDAType(rElemType);
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    std::string args;
    for (unsigned i = 0; i < op.getNumOperands(); i++) {
      if (i) args += ", ";
      args += getElemExpr(op.getOperand(i), "_i");
    }
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + var + "[_i] = " + symbol + "(" + args + ");");
  } else {
    std::string args;
    for (unsigned i = 0; i < op.getNumOperands(); i++) {
      if (i) args += ", ";
      args += getVar(op.getOperand(i));
    }
    emit(cudaType + " " + var + " = " + symbol + "(" + args + ");");
  }
}

// Redundant-thread predicate: when the tensor's layout replicates the same
// logical element across multiple lanes/warps (free layout bits), only the
// canonical owner (free bits == 0) may perform an atomic; otherwise the side
// effect is multiplied by the replication factor. (Do NOT mask the `block`
// dim: distinct CTAs legitimately each contribute a partial.)
std::string CUDACodeGen::redundantThreadGuard(RankedTensorType ty) {
  // Native non-pow2 (n80) tensors abort ttg::toLinearLayout (GF(2) pow2). A
  // native non-pow2 WGMMA accumulator has NO thread replication — every
  // (reg,lane,warp) owns a distinct element — so the redundancy guard is empty.
  // (Only reachable for non-pow2 mma acc, e.g. the dQ=K^T·dS^T atomic-add drain;
  // pow2 layouts keep the exact prior behavior.)
  if (llvm::any_of(ty.getShape(),
                   [](int64_t d) { return !llvm::isPowerOf2_64(d); }) &&
      isa<ttg::NvidiaMmaEncodingAttr>(ty.getEncoding()))
    return "";
  auto ll = ttg::toLinearLayout(ty);
  auto *ctx = ty.getContext();
  auto freeMasks = ll.getFreeVariableMasks();
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  auto kWarp = mlir::StringAttr::get(ctx, "warp");
  int laneMask = freeMasks.count(kLane) ? freeMasks[kLane] : 0;
  int warpMask = freeMasks.count(kWarp) ? freeMasks[kWarp] : 0;
  std::string guard;
  if (laneMask)
    guard += "(lane_id & " + std::to_string(laneMask) + ") == 0";
  if (warpMask) {
    if (!guard.empty()) guard += " && ";
    guard += "(warp_id & " + std::to_string(warpMask) + ") == 0";
  }
  return guard;
}

// The redundancy guard leaves the atomic's old-value UNINITIALIZED in replica
// threads; anything consuming the result there (e.g. the unguarded tt.store
// in test_tensor_atomic_rmw check_return_val) reads garbage and races the
// canonical thread's good value. Mirror the PTX backend's
// finalizeTensorAtomicResults: canonical threads store their old-values to a
// transient smem scratch at the element's LOGICAL offset (free bits map to
// offset 0, so replicas compute the same address), barrier, everyone loads.
void CUDACodeGen::emitAtomicResultBroadcast(RankedTensorType ty,
                                            const std::string &var,
                                            const std::string &cudaType,
                                            const std::string &redGuard) {
  auto *ctx = ty.getContext();
  auto kReg = mlir::StringAttr::get(ctx, "register");
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  auto kWarp = mlir::StringAttr::get(ctx, "warp");
  auto kOffset = mlir::StringAttr::get(ctx, "offset");
  auto ll = ttg::toLinearLayout(ty);
  auto cvt =
      ll.sublayout({kReg, kLane, kWarp}, llvm::to_vector(ll.getOutDimNames()));
  cvt = cvt.reshapeOuts({{kOffset, cvt.getTotalOutDimSize()}});
  int total = cvt.getTotalOutDimSize();
  int nElems = cvt.getInDimSize(kReg);
  int eb = getTypeSizeInBytes(ty.getElementType());
  auto applyOff = [&](int reg, int lane, int warp) -> int64_t {
    auto out = cvt.apply({{kReg, reg}, {kLane, lane}, {kWarp, warp}});
    return out.front().second;
  };
  // Transient scratch: bump-allocate, then restore the offset after the
  // readback so the space is reused (same pattern as convert_layout scratch).
  int saved = sharedMemOffset;
  int smemOff = (sharedMemOffset + 127) & ~127;
  sharedMemOffset = smemOff + total * eb;
  if (sharedMemOffset > peakSharedMem) peakSharedMem = sharedMemOffset;
  emit("// broadcast atomic old-values to replicated lanes/warps via smem");
  emit("{");
  indent();
  emit(cudaType + "* _abc = (" + cudaType + "*)(shared_mem + " +
       std::to_string(smemOff) + "); // transient scratch");
  std::string lwExpr = "0";
  int nLaneBits = cvt.getInDimSizeLog2(kLane);
  int nWarpBits = cvt.getInDimSizeLog2(kWarp);
  for (int b = 0; b < nLaneBits; b++) {
    int64_t d = applyOff(0, 1 << b, 0);
    if (d)
      lwExpr += " ^ (((lane_id >> " + std::to_string(b) + ") & 1) * " +
                std::to_string(d) + ")";
  }
  for (int b = 0; b < nWarpBits; b++) {
    int64_t d = applyOff(0, 0, 1 << b);
    if (d)
      lwExpr += " ^ (((warp_id >> " + std::to_string(b) + ") & 1) * " +
                std::to_string(d) + ")";
  }
  emit("int _lw = " + lwExpr + ";");
  emit("if (" + redGuard + ") {");
  indent();
  for (int i = 0; i < nElems; i++)
    emit("_abc[_lw ^ " + std::to_string(applyOff(i, 0, 0)) + "] = " + var +
         "[" + std::to_string(i) + "];");
  dedent();
  emit("}");
  emit("__syncthreads();");
  for (int i = 0; i < nElems; i++)
    emit(var + "[" + std::to_string(i) + "] = _abc[_lw ^ " +
         std::to_string(applyOff(i, 0, 0)) + "];");
  emit("__syncthreads();");
  dedent();
  emit("}");
  sharedMemOffset = saved;
}

static std::string memSemStr(tt::MemSemantic sem) {
  switch (sem) {
  case tt::MemSemantic::RELAXED: return "relaxed";
  case tt::MemSemantic::ACQUIRE: return "acquire";
  case tt::MemSemantic::RELEASE: return "release";
  default: return "acq_rel";
  }
}

static std::string memScopeStr(tt::MemSyncScope scope) {
  switch (scope) {
  case tt::MemSyncScope::CTA: return "cta";
  case tt::MemSyncScope::SYSTEM: return "sys";
  default: return "gpu";
  }
}

// Unsigned int type of the element's bit width, for bit-casting operands of
// b-typed / 16-bit PTX atomics through plain registers.
static std::string getRawIntType(Type elemType) {
  int bits = elemType.getIntOrFloatBitWidth();
  return bits == 64 ? "unsigned long long"
                    : (bits == 16 ? "unsigned short" : "unsigned int");
}

// PTX atom `op.type` suffix + asm register constraint for an RMW op.
// `viaRaw` = operands must be bit-cast through an integer temp (16-bit floats
// use "h" registers; b-typed ops like exch on floats use "r"/"l").
// UMAX/UMIN come from the float-atomic-max-via-int lowering (positive value →
// signed `max`, negative → `umin` on the bit pattern); they map to the
// unsigned PTX flavors.
static std::tuple<std::string, std::string, bool> atomRMWPtx(tt::RMWOp rmw,
                                                             Type elemType) {
  int bits = elemType.getIntOrFloatBitWidth();
  bool isFloat = isa<FloatType>(elemType);
  bool b64 = bits == 64;
  std::string c = b64 ? "l" : "r";
  switch (rmw) {
  case tt::RMWOp::ADD:
  case tt::RMWOp::FADD:
    if (isFloat) {
      if (bits == 16)
        return {elemType.isBF16() ? "add.noftz.bf16" : "add.noftz.f16", "h", true};
      if (b64)
        return {"add.f64", "d", false};
      return {"add.f32", "f", false};
    }
    return {b64 ? "add.u64" : "add.u32", c, false};
  case tt::RMWOp::MAX:
    if (isFloat) // PTX has no float atom.max; should be lowered via-int before
      llvm::report_fatal_error("emitAtomicRMW: float MAX must be int-lowered");
    return {b64 ? "max.s64" : "max.s32", c, false};
  case tt::RMWOp::MIN:
    if (isFloat)
      llvm::report_fatal_error("emitAtomicRMW: float MIN must be int-lowered");
    return {b64 ? "min.s64" : "min.s32", c, false};
  case tt::RMWOp::UMAX:
    return {b64 ? "max.u64" : "max.u32", c, isFloat};
  case tt::RMWOp::UMIN:
    return {b64 ? "min.u64" : "min.u32", c, isFloat};
  case tt::RMWOp::AND:
    return {b64 ? "and.b64" : "and.b32", c, isFloat};
  case tt::RMWOp::OR:
    return {b64 ? "or.b64" : "or.b32", c, isFloat};
  case tt::RMWOp::XOR:
    return {b64 ? "xor.b64" : "xor.b32", c, isFloat};
  case tt::RMWOp::XCHG:
    return {b64 ? "exch.b64" : "exch.b32", c, isFloat};
  default:
    llvm::report_fatal_error("emitAtomicRMW: unsupported RMW op");
  }
}

void CUDACodeGen::emitAtomicRMW(tt::AtomicRMWOp op) {
  auto result = op.getResult();
  auto var = newVar("atom");
  valueToVar[result] = var;
  auto ptrVar = getVar(op.getPtr());
  auto valVar = getVar(op.getVal());
  bool isTensor = isa<RankedTensorType>(result.getType());
  Type elemType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto cudaType = getCUDAType(elemType);
  auto rmw = op.getAtomicRmwOp();
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    auto ty = cast<RankedTensorType>(result.getType());
    std::string redGuard = redundantThreadGuard(ty);
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    std::string guard;
    if (op.getMask()) guard = "(" + getElemExpr(op.getMask(), "_i") + ")";
    if (!redGuard.empty()) {
      if (!guard.empty()) guard += " && ";
      guard += "(" + redGuard + ")";
    }
    // Inline PTX instead of CUDA atomicXXX() builtins: the builtins have no
    // overloads for long long / double-max / 16-bit floats (nvcc "no instance
    // of overloaded function" on the 64-bit and f16/bf16 dtypes) and are
    // relaxed-only, dropping the op's sem/scope qualifiers.
    auto [suffix, cons, viaRaw] = atomRMWPtx(rmw, elemType);
    std::string ins = "atom.global." + memScopeStr(op.getScope()) + "." +
                      memSemStr(op.getSem()) + "." + suffix;
    std::string rawT = getRawIntType(elemType);
    if (!guard.empty()) emit("if (" + guard + ") {");
    else emit("{");
    indent();
    emit(cudaType + " _v = " + getElemExpr(op.getVal(), "_i") + ";");
    if (viaRaw) {
      emit(rawT + " _o, _r = *(" + rawT + "*)&_v;");
      emit("asm volatile(\"" + ins + " %0, [%1], %2;\" : \"=" + cons +
           "\"(_o) : \"l\"(" + getPtrAddrExpr(op.getPtr(), "_i") + "), \"" +
           cons + "\"(_r) : \"memory\");");
      emit(var + "[_i] = *(" + cudaType + "*)&_o;");
    } else {
      emit("asm volatile(\"" + ins + " %0, [%1], %2;\" : \"=" + cons + "\"(" +
           var + "[_i]) : \"l\"(" + getPtrAddrExpr(op.getPtr(), "_i") +
           "), \"" + cons + "\"(_v) : \"memory\");");
    }
    dedent();
    emit("}");
    dedent();
    emit("}");
    if (!result.use_empty() && !redGuard.empty())
      emitAtomicResultBroadcast(ty, var, cudaType, redGuard);
  } else {
    // Scalar atomic. TTGIR semantics: ONE atomic per CTA — the PTX backend
    // predicates it on thread 0 and broadcasts the old value through shared
    // memory. Emitting it for every thread multiplies the side effect by
    // blockDim (atomic_add accumulated 128x) and deadlocks spin-locks built
    // on atomic_cas/atomic_xchg (task #51: the winner thread holds the lock
    // at __syncthreads while its siblings spin forever). The per-thread mask
    // still applies (float-atomic-max-via-int emits two masked scalar atomics
    // on the same address). Memory-order qualifiers are honored via inline
    // PTX since CUDA's atomicXXX() builtins are relaxed-only.
    auto [suffix, cons, viaRaw] = atomRMWPtx(rmw, elemType);
    std::string ins = "atom.global." + memScopeStr(op.getScope()) + "." +
                      memSemStr(op.getSem()) + "." + suffix;
    bool used = !result.use_empty();
    emit(cudaType + " " + var + "{};");
    if (used)
      emit("__shared__ " + cudaType + " " + var + "_bc;");
    std::string guard = "tid == 0";
    if (op.getMask())
      guard += " && (" + getVar(op.getMask()) + ")";
    emit("if (" + guard + ") {");
    indent();
    std::string rawT = getRawIntType(elemType);
    if (viaRaw) {
      emit(rawT + " " + var + "_old, " + var + "_val = *(" + rawT + "*)&" +
           valVar + ";");
      emit("asm volatile(\"" + ins + " %0, [%1], %2;\" : \"=" + cons + "\"(" +
           var + "_old) : \"l\"(" + ptrVar + "), \"" + cons + "\"(" + var +
           "_val) : \"memory\");");
      emit(var + " = *(" + cudaType + "*)&" + var + "_old;");
    } else {
      emit("asm volatile(\"" + ins + " %0, [%1], %2;\" : \"=" + cons + "\"(" +
           var + ") : \"l\"(" + ptrVar + "), \"" + cons + "\"(" + valVar +
           ") : \"memory\");");
    }
    if (used)
      emit(var + "_bc = " + var + ";");
    dedent();
    emit("}");
    if (used) {
      // Broadcast thread 0's old value to the rest of the threads sharing this
      // atomic. The trailing barrier keeps the next emission's store to the
      // scratch (e.g. the following spin-loop iteration) from racing slow
      // readers of this one. Route through blockSync() instead of a hard
      // __syncthreads(): inside a ttg.warp_specialize partition the consumer
      // warps take a different branch and never reach a CTA-wide barrier, so a
      // raw __syncthreads() in the producer region deadlocks. blockSync()
      // lowers to a region-scoped `bar.sync <regionId>, <regionThreads>` when
      // emitting inside a WS region (matching level6's warp-local scheduler
      // broadcast) and to __syncthreads() in ordinary code.
      blockSync();
      emit(var + " = " + var + "_bc;");
      blockSync();
    }
  }
}

void CUDACodeGen::emitAtomicCAS(tt::AtomicCASOp op) {
  auto result = op.getResult();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("cas");
  valueToVar[result] = var;
  auto ptrVar = getVar(op.getPtr());
  auto cmpVar = getVar(op.getCmp());
  auto valVar = getVar(op.getVal());
  Type elemType = isTensor
      ? cast<RankedTensorType>(result.getType()).getElementType()
      : result.getType();
  auto cudaType = getCUDAType(elemType);

  // Inline PTX rather than atomicCAS(): the builtin only has int/uint/ull
  // overloads (nvcc "no instance of overloaded function" on float/16-bit
  // dtypes, test_tensor_atomic_cas) and is relaxed-only.
  int bits = elemType.getIntOrFloatBitWidth();
  bool isFloat = isa<FloatType>(elemType);
  std::string suffix = bits == 64 ? "cas.b64" : (bits == 16 ? "cas.b16" : "cas.b32");
  std::string cons = bits == 64 ? "l" : (bits == 16 ? "h" : "r");
  bool viaRaw = isFloat || bits == 16;
  std::string rawT = getRawIntType(elemType);
  std::string scopeS = memScopeStr(op.getScope());
  // scope `gpu` is PTX's default; omit it so the instruction reads
  // atom.global.<sem>.cas (the flavor test_core's asm["ptx"] asserts expect).
  std::string ins = "atom.global." + memSemStr(op.getSem()) +
                    (scopeS == "gpu" ? "" : "." + scopeS) + "." + suffix;

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    auto ty = cast<RankedTensorType>(result.getType());
    // Only the canonical owner of each replicated element may CAS (otherwise
    // replicas re-execute the swap), then broadcast old-values back.
    std::string redGuard = redundantThreadGuard(ty);
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    if (!redGuard.empty()) {
      emit("if (" + redGuard + ") {");
      indent();
    }
    emit(cudaType + " _c = " + getElemExpr(op.getCmp(), "_i") + ", _v = " +
         getElemExpr(op.getVal(), "_i") + ";");
    if (viaRaw) {
      emit(rawT + " _o, _rc = *(" + rawT + "*)&_c, _rv = *(" + rawT + "*)&_v;");
      emit("asm volatile(\"" + ins + " %0, [%1], %2, %3;\" : \"=" + cons +
           "\"(_o) : \"l\"(" + getPtrAddrExpr(op.getPtr(), "_i") + "), \"" +
           cons + "\"(_rc), \"" + cons + "\"(_rv) : \"memory\");");
      emit(var + "[_i] = *(" + cudaType + "*)&_o;");
    } else {
      emit("asm volatile(\"" + ins + " %0, [%1], %2, %3;\" : \"=" + cons +
           "\"(" + var + "[_i]) : \"l\"(" + getPtrAddrExpr(op.getPtr(), "_i") +
           "), \"" + cons + "\"(_c), \"" + cons + "\"(_v) : \"memory\");");
    }
    if (!redGuard.empty()) {
      dedent();
      emit("}");
    }
    dedent();
    emit("}");
    if (!result.use_empty() && !redGuard.empty())
      emitAtomicResultBroadcast(ty, var, cudaType, redGuard);
  } else {
    // Scalar CAS: ONE op per CTA, thread-0 predicated with the old value
    // broadcast through shared memory (mirrors the PTX backend). Unpredicated
    // emission deadlocked test_atomic_cas's spin lock (task #51): every
    // thread CASes, the winner waits at __syncthreads holding the lock while
    // sibling threads spin on it forever.
    bool used = !result.use_empty();
    emit(cudaType + " " + var + "{};");
    if (used)
      emit("__shared__ " + cudaType + " " + var + "_bc;");
    emit("if (tid == 0) {");
    indent();
    if (viaRaw) {
      emit(rawT + " " + var + "_old, " + var + "_cmp = *(" + rawT + "*)&" +
           cmpVar + ", " + var + "_val = *(" + rawT + "*)&" + valVar + ";");
      emit("asm volatile(\"" + ins + " %0, [%1], %2, %3;\" : \"=" + cons +
           "\"(" + var + "_old) : \"l\"(" + ptrVar + "), \"" + cons + "\"(" +
           var + "_cmp), \"" + cons + "\"(" + var + "_val) : \"memory\");");
      emit(var + " = *(" + cudaType + "*)&" + var + "_old;");
    } else {
      emit("asm volatile(\"" + ins + " %0, [%1], %2, %3;\" : \"=" + cons +
           "\"(" + var + ") : \"l\"(" + ptrVar + "), \"" + cons + "\"(" +
           cmpVar + "), \"" + cons + "\"(" + valVar + ") : \"memory\");");
    }
    if (used)
      emit(var + "_bc = " + var + ";");
    dedent();
    emit("}");
    if (used) {
      // Trailing barrier: keep the next iteration's scratch store (spin loop)
      // from racing slow readers of this broadcast.
      emit("__syncthreads();");
      emit(var + " = " + var + "_bc;");
      emit("__syncthreads();");
    }
  }
}

void CUDACodeGen::emitPrint(tt::PrintOp op) {
  // Mirror the PTX backend (PrintOpToLLVM): every thread printf's each of its
  // register elements with the element's logical tensor index:
  //   pid (<x>, <y>, <z>) idx (<i0>, <i1>, ...)<prefix>[(operand N) ]<elem>
  // The Python wrapper munges <prefix> so it starts with " " and ends ": ".
  auto escape = [](llvm::StringRef s) {
    std::string out;
    for (char c : s) {
      if (c == '"' || c == '\\')
        out += '\\';
      if (c == '\n') {
        out += "\\n";
        continue;
      }
      out += c;
    }
    return out;
  };
  std::string prefix = escape(op.getPrefix());
  std::string pidArgs = "blockIdx.x, blockIdx.y, blockIdx.z";

  if (op.getNumOperands() == 0) {
    emit("printf(\"pid (%u, %u, %u)" + prefix + "\\n\", " + pidArgs + ");");
    return;
  }

  for (size_t i = 0; i < op.getNumOperands(); i++) {
    bool isSigned = op.getIsSigned()[i] > 0;
    bool hex = op.getHex();
    Value operand = op.getOperand(i);
    auto rtt = dyn_cast<RankedTensorType>(operand.getType());
    Type elemType = rtt ? rtt.getElementType() : operand.getType();

    // Element format specifier + C cast (printf varargs need explicit
    // promotion for half types and width-correct unsigned reads).
    std::string fmt, cast;
    if (isa<tt::PointerType>(elemType)) {
      fmt = "%p";
      cast = "(void*)";
    } else if (hex) {
      unsigned bw = elemType.getIntOrFloatBitWidth();
      fmt = "0x%0" + std::to_string(bw / 4) + (bw > 32 ? "llx" : "x");
      if (bw > 32)
        cast = "(unsigned long long)";
      else if (bw == 16)
        cast = "(unsigned)(uint16_t)";
      else if (bw == 8)
        cast = "(unsigned)(uint8_t)";
      else
        cast = "(unsigned)";
    } else if (elemType.isF16() || elemType.isBF16()) {
      fmt = "%f";
      cast = "(float)";
    } else if (isa<FloatType>(elemType)) {
      fmt = "%f";
      cast = ""; // float/double promote to double automatically
    } else if (elemType.isInteger(64)) {
      fmt = isSigned ? "%lli" : "%llu";
      cast = isSigned ? "(long long)" : "(unsigned long long)";
    } else {
      unsigned bw = elemType.getIntOrFloatBitWidth();
      fmt = isSigned ? "%i" : "%u";
      if (isSigned)
        cast = "(int)";
      else if (bw == 16)
        cast = "(unsigned)(uint16_t)";
      else if (bw == 8)
        cast = "(unsigned)(uint8_t)";
      else
        cast = "(unsigned)";
    }
    std::string operandTag =
        op.getNumOperands() > 1 ? "(operand " + std::to_string(i) + ") " : "";

    if (!rtt) {
      emit("printf(\"pid (%u, %u, %u) idx ()" + prefix + operandTag + fmt +
           "\\n\", " + pidArgs + ", " + cast + getVar(operand) + ");");
      continue;
    }

    int rank = rtt.getRank();
    int nElems = getElemsPerThread(operand);
    auto ll = ttg::toLinearLayout(rtt);
    auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");
    const auto &bases = ll.getBases();
    const auto &laneBases = bases.find(kLane)->second;
    const auto &warpBases = bases.find(kWarp)->second;

    // Index format: each dim padded to the digit count of its extent.
    std::string idxFmt;
    for (int d = 0; d < rank; d++) {
      if (d)
        idxFmt += ", ";
      int64_t dim = rtt.getShape()[d];
      int w = dim > 0 ? (int)std::ceil(std::log10((double)dim)) : 0;
      idxFmt += "%" + std::to_string(w) + "u";
    }
    std::string fmtStr = "pid (%u, %u, %u) idx (" + idxFmt + ")" + prefix +
                         operandTag + fmt + "\\n";

    for (int r = 0; r < nElems; r++) {
      auto coords = ll.apply({{kReg, r}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      std::string idxArgs;
      for (int d = 0; d < rank; d++) {
        std::string expr = std::to_string(coords[d].second);
        for (size_t lb = 0; lb < laneBases.size(); lb++)
          if (laneBases[lb][d] != 0)
            expr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " +
                    std::to_string(laneBases[lb][d]);
        for (size_t wb = 0; wb < warpBases.size(); wb++)
          if (warpBases[wb][d] != 0)
            expr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " +
                    std::to_string(warpBases[wb][d]);
        idxArgs += ", (unsigned)(" + expr + ")";
      }
      emit("printf(\"" + fmtStr + "\", " + pidArgs + idxArgs + ", " + cast +
           getElemExpr(operand, std::to_string(r)) + ");");
    }
  }
}

void CUDACodeGen::emitMulhiUI(tt::MulhiUIOp op) {
  auto result = op.getResult();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("mulhi");
  valueToVar[result] = var;
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    auto aExpr = getElemExpr(op.getOperand(0), "_i");
    auto bExpr = getElemExpr(op.getOperand(1), "_i");
    emit("unsigned " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + var + "[_i] = __umulhi((unsigned)" + aExpr + ", (unsigned)" + bExpr + ");");
  } else {
    auto aVar = getVar(op.getOperand(0));
    auto bVar = getVar(op.getOperand(1));
    emit("unsigned " + var + " = __umulhi((unsigned)" + aVar + ", (unsigned)" + bVar + ");");
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Warp specialization — custom Hopper producer/consumer codegen
// ═══════════════════════════════════════════════════════════════════════
//
// `ttg.warp_specialize` splits a pipelined matmul loop into one PRODUCER
// warpgroup (the `default` region: issues the TMA loads) and N CONSUMER
// warpgroups (the `partition<i>` regions: run the WGMMAs and the epilogue
// store). They synchronise through the mbarrier full/empty pairs already
// allocated before the op. This is exactly the gemm_02/gemm_06 design:
//
//   warpgroup 0 (base warps)      -> producer, setmaxnreg.dec (few regs)
//   warpgroup 1.. (partitions)    -> consumers, setmaxnreg.inc (many regs)
//
// The block launches base + Σpartition warps (blockDim widened in generate()).
// We dispatch on the *global* warp_id, then inside each consumer branch shadow
// `tid`/`warp_id`/`lane_id`/`_wg_m` with warpgroup-LOCAL values so every
// existing per-op emitter (WGMMA, TMA, mma->smem store, …) works unchanged: a
// consumer warpgroup is a self-contained 128-thread WGMMA unit (_wg_m == 0).
//
// The `default` region is not isolated-from-above and references outer SSA
// values directly. The `partition<i>` regions ARE isolated; their block args
// bind positionally to the op's explicit captures, which we alias here.
void CUDACodeGen::emitWSRegionBody(Region &region) {
  for (Block &block : region)
    for (Operation &o : block)
      emitOp(&o); // warp_yield / warp_return terminators are no-ops in dispatch
}

void CUDACodeGen::emitWarpSpecialize(gpu::WarpSpecializeOp op) {
  auto partRegions = op.getPartitionRegions();
  ArrayRef<int32_t> partWarps = op.getPartitionNumWarps();
  std::optional<ArrayRef<int32_t>> reqRegs = op.getRequestedRegisters();
  auto caps = op.getPartitionOp().getExplicitCaptures();
  // Debug toggle: TRITON_CUDA_WS_NOREGS=1 disables setmaxnreg dec/inc so we can
  // isolate register-reallocation faults from the rest of the WS lowering.
  if (reqRegs && getenv("TRITON_CUDA_WS_NOREGS"))
    reqRegs = std::nullopt;

  // Register budget. With __launch_bounds__(totalThreads) ptxas hands every
  // thread `baseline` registers (65536 per SM / totalThreads, floored to a
  // multiple of 8). Consumers raise to their requested count; the producer
  // lowers by the exact amount the consumers take, keeping the CTA within
  // 65536 registers (the gemm_02 invariant: 1*(168-40)*128 == 2*(232-168)*128).
  int totalThreads = totalNumWarps * 32;
  int baseline = (65536 / std::max(totalThreads, 1)) & ~7;
  int producerRegs = baseline;
  if (reqRegs) {
    long gained = 0;
    for (unsigned i = 0; i < partRegions.size(); ++i)
      gained += (long)((*reqRegs)[i] - baseline) * (long)(partWarps[i] * 32);
    int producerThreads = numWarps * 32;
    producerRegs = baseline - (int)(gained / std::max(producerThreads, 1));
    producerRegs &= ~7;
    // setmaxnreg requires the operand in [24, 256]. The producer issues a .dec,
    // so it can never sensibly raise above `baseline` (the launch-bounds reg
    // count); if consumers request FEWER regs than baseline the formula would
    // overshoot (e.g. 280) and ptxas rejects it. Clamp to the legal range.
    if (producerRegs < 24)
      producerRegs = 24;
    if (producerRegs > baseline)
      producerRegs = baseline;
  }

  emit("// ─── warp-specialization dispatch (custom Hopper producer/consumer) ───");
  emit("{");
  indent();

  // Producer = default region, runs on the base warps [0, numWarps).
  // Block-wide __syncthreads() inside this branch would deadlock (consumers
  // take a different branch), so route blockSync() through a named barrier
  // scoped to just this region's threads. Barrier 0 is reserved for genuine
  // whole-block __syncthreads(); regions use ids 1, 2, 3, ...
  emit("if (warp_id < " + std::to_string(numWarps) + ") {");
  indent();
  if (reqRegs)
    emit("asm volatile(\"setmaxnreg.dec.sync.aligned.u32 " +
         std::to_string(producerRegs) + ";\");");
  wsSyncBarrierId = 1;
  wsSyncThreadCount = numWarps * 32;
  emitWSRegionBody(op.getDefaultRegion());
  wsSyncBarrierId = 0;
  wsSyncThreadCount = 0;
  dedent();
  emit("}");

  // Consumers = partition<i> regions, each a warpgroup-local 128-thread unit.
  int cum = numWarps;
  for (unsigned i = 0; i < partRegions.size(); ++i) {
    int lo = cum;
    int hi = cum + partWarps[i];
    emit("else if (warp_id < " + std::to_string(hi) + ") {");
    indent();
    // Shadow thread indices to be warpgroup-local so WGMMA/mma-store emitters
    // (which use warp_id/lane_id/_wg_m) address this consumer's own tile.
    emit("const int tid = (int)threadIdx.x - " + std::to_string(lo * 32) + ";");
    emit("const int warp_id = tid >> 5;");
    emit("const int lane_id = tid & 31;");
    emit("const int _wg_m = 0;");
    if (reqRegs)
      emit("asm volatile(\"setmaxnreg.inc.sync.aligned.u32 " +
           std::to_string((*reqRegs)[i]) + ";\");");
    // Bind isolated partition block args to the captured operand vars.
    Block &b = partRegions[i]->front();
    for (unsigned j = 0; j < b.getNumArguments() && j < caps.size(); ++j)
      valueToVar[b.getArgument(j)] = getVar(caps[j]);
    wsSyncBarrierId = 2 + (int)i;
    wsSyncThreadCount = partWarps[i] * 32;
    emitWSRegionBody(*partRegions[i]);
    wsSyncBarrierId = 0;
    wsSyncThreadCount = 0;
    dedent();
    emit("}");
    cum = hi;
  }

  dedent();
  emit("}");
}

void CUDACodeGen::emitScfFor(scf::ForOp op) {
  // Get loop bounds
  auto lbVar = getVar(op.getLowerBound());
  auto ubVar = getVar(op.getUpperBound());
  auto stepVar = getVar(op.getStep());
  auto ivVar = newVar("iv");
  valueToVar[op.getInductionVar()] = ivVar;

  // Handle iter_args
  auto iterArgs = op.getInitArgs();
  auto regionIterArgs = op.getRegionIterArgs();
  for (unsigned i = 0; i < iterArgs.size(); i++) {
    auto iterVar = newVar("iter");
    valueToVar[regionIterArgs[i]] = iterVar;

    bool isTensor = isa<RankedTensorType>(iterArgs[i].getType());
    if (isTensor) {
      int nElems = getElemsPerThread(iterArgs[i]);
      Type elemType = cast<RankedTensorType>(iterArgs[i].getType()).getElementType();
      auto cudaType = getCUDAType(elemType);

      // Check if init value is a deferred addptr — use scalar-base approach
      // to reduce loop-carried state from N pointers to 1 scalar offset.
      // This eliminates ~47 IMAD.MOV register copies per loop iteration.
      auto deferIt = deferredAddPtr.find(iterArgs[i]);
      if (deferIt != deferredAddPtr.end()) {
        std::string base = deferIt->second.first;
        std::string initOff = deferIt->second.second;
        bool useScalarBase = false;
        // Scalar-base is only valid when the loop advances this pointer tensor
        // in place: yield_i must chain back to iter_arg_i through addptr ops,
        // so the per-element address spread stays frozen in the precomputed
        // paddr array and only a scalar delta is carried. Pipelined stage-shift
        // chains (yield_i = iter_arg_j) or fresh addptr(base, varying_offsets)
        // yields have per-element offsets relative to a *different* address
        // array — those must carry the full offset array instead.
        bool inPlaceAdvance = false;
        if (auto yield = dyn_cast<scf::YieldOp>(op.getBody()->getTerminator())) {
          if (i < yield.getNumOperands()) {
            Value cur = yield.getOperand(i);
            while (auto ap = cur.getDefiningOp<tt::AddPtrOp>())
              cur = ap.getPtr();
            inPlaceAdvance = (cur == regionIterArgs[i]);
          }
        }
        if (auto ptrType = dyn_cast<tt::PointerType>(elemType);
            ptrType && inPlaceAdvance) {
          Type pointeeType = ptrType.getPointeeType();
          if (pointeeType.isIntOrFloat() && pointeeType.getIntOrFloatBitWidth() >= 8) {
            int bpe = pointeeType.getIntOrFloatBitWidth() / 8;
            // Precompute 64-bit addresses: base + offset[i]*bpe
            // This eliminates VIADD per address in the loop (only IADD3+IADD3.X needed)
            auto precompVar = newVar("paddr");
            emit("unsigned long long " + precompVar + "[" + std::to_string(nElems) + "];");
            std::string mulStr = (bpe == 1) ? "" : " * " + std::to_string(bpe);
            emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                 "; _t++) " + precompVar + "[_t] = (unsigned long long)(const char*)" +
                 base + " + (unsigned)(" + initOff + "[_t]" + mulStr + ");");
            // iter var is a scalar int (byte offset delta)
            emit("int " + iterVar + " = 0;");
            ScalarBaseInfo sbi{precompVar, bpe, iterVar, getCUDAType(pointeeType)};
            scalarBaseDeferred[regionIterArgs[i]] = sbi;
            if (i < op.getNumResults())
              scalarBaseDeferred[op.getResult(i)] = sbi;
            useScalarBase = true;
          }
        }
        if (!useScalarBase) {
          // Fallback: carry int offsets
          emit("int " + iterVar + "[" + std::to_string(nElems) + "];");
          emit("for (int _t = 0; _t < " + std::to_string(nElems) +
               "; _t++) " + iterVar + "[_t] = " + initOff + "[_t];");
          deferredAddPtr[regionIterArgs[i]] = {base, iterVar};
          iterArgDeferredBase[regionIterArgs[i]] = base;
          if (i < op.getNumResults())
            deferredAddPtr[op.getResult(i)] = {base, iterVar};
        }
      } else if (scalarValues.contains(iterArgs[i])) {
        auto initVar = getVar(iterArgs[i]);
        auto initCudaType = getCUDAType(iterArgs[i].getType());
        // Scalar init → still need array for loop mutation
        emit(cudaType + " " + iterVar + "[" + std::to_string(nElems) + "];");
        emit("for (int _t = 0; _t < " + std::to_string(nElems) +
             "; _t++) " + iterVar + "[_t] = " + initVar + ";");
      } else {
        auto initVar = getVar(iterArgs[i]);
        emit(cudaType + " " + iterVar + "[" + std::to_string(nElems) + "];");
        emit("for (int _t = 0; _t < " + std::to_string(nElems) +
             "; _t++) " + iterVar + "[_t] = " + initVar + "[_t];");
      }
    } else if (isa<tt::TensorDescType>(iterArgs[i].getType())) {
      // Loop-carried tensor descriptor: carry a pointer to the CUtensorMap
      // (CUtensorMap itself is not assignable).
      emit("const char* " + iterVar + " = " + descAddrExpr(iterArgs[i]) + ";");
      pointerDescriptors.insert(regionIterArgs[i]);
      if (i < op.getNumResults())
        pointerDescriptors.insert(op.getResult(i));
    } else {
      auto initVar = getVar(iterArgs[i]);
      auto cudaType = getCUDAType(iterArgs[i].getType());
      emit(cudaType + " " + iterVar + " = " + initVar + ";");
    }

    // Map result to iter var
    if (i < op.getNumResults()) {
      valueToVar[op.getResult(i)] = iterVar;
    }
  }

  // Emit for loop. Use the induction variable's actual type (i32/i64): a
  // hardcoded `int` truncates 64-bit bounds (e.g. 2**35 -> 0), turning a
  // finite or empty loop into an infinite one. Triton normalizes negative-step
  // ranges to positive-step scf.for with swapped bounds, so `<` is always the
  // correct comparison direction.
  std::string ivType = getCUDAType(op.getInductionVar().getType());
  emit("for (" + ivType + " " + ivVar + " = " + lbVar + "; " + ivVar + " < " +
       ubVar + "; " + ivVar + " += " + stepVar + ") {");
  indent();

  // Emit body
  auto &body = op.getBody()->getOperations();
  for (auto &bodyOp : body) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
      // Update iter_args with correct SSA semantics: all yields are
      // simultaneous, so we must save iter_args that are read by later
      // yields before any assignment overwrites them.

      // Phase 1: Detect cross-dependencies where yield[i] reads
      // iter_arg[j] with j < i (already overwritten by assignment j).
      std::map<std::string, std::string> savedIterVars; // iterVar -> tmpVar
      for (unsigned i = 0; i < yieldOp.getNumOperands() && i < regionIterArgs.size(); i++) {
        auto yieldVal = yieldOp.getOperand(i);
        std::string srcVar;
        auto deferIt = deferredAddPtr.find(yieldVal);
        if (isa<RankedTensorType>(yieldVal.getType()) && deferIt != deferredAddPtr.end()) {
          srcVar = deferIt->second.second;
        } else {
          srcVar = getVar(yieldVal);
        }
        auto iterVar = getVar(regionIterArgs[i]);
        if (srcVar == iterVar) continue; // self-assignment, no conflict

        // Check if srcVar matches any earlier iter_arg variable
        for (unsigned j = 0; j < i; j++) {
          if (srcVar == getVar(regionIterArgs[j]) && savedIterVars.find(srcVar) == savedIterVars.end()) {
            // This iter_arg will be overwritten before we read it — save it
            auto tmpVar = newVar("_sv");
            bool jIsTensor = isa<RankedTensorType>(regionIterArgs[j].getType());
            if (scalarBaseDeferred.count(regionIterArgs[j])) {
              // Scalar-base iter arg is a scalar int despite being a tensor SSA value
              emit("int " + tmpVar + " = " + srcVar + ";");
            } else if (jIsTensor) {
              int nElems = getElemsPerThread(regionIterArgs[j]);
              std::string savedType;
              if (ptrBasedDeferred.count(regionIterArgs[j]))
                savedType = "const char*";
              else {
                auto elemType = cast<RankedTensorType>(regionIterArgs[j].getType()).getElementType();
                savedType = getCUDAType(elemType);
              }
              emit(savedType + " " + tmpVar + "[" + std::to_string(nElems) + "];");
              emit("#pragma unroll");
              emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                   "; _t++) " + tmpVar + "[_t] = " + srcVar + "[_t];");
            } else if (isa<tt::TensorDescType>(regionIterArgs[j].getType())) {
              emit("const char* " + tmpVar + " = " + srcVar + ";");
            } else {
              emit(getCUDAType(regionIterArgs[j].getType()) + " " + tmpVar + " = " + srcVar + ";");
            }
            savedIterVars[srcVar] = tmpVar;
            break;
          }
        }
      }

      // Phase 2: Emit assignments, using saved temps where needed
      for (unsigned i = 0; i < yieldOp.getNumOperands(); i++) {
        if (i < regionIterArgs.size()) {
          auto iterVar = getVar(regionIterArgs[i]);
          auto yieldVal = yieldOp.getOperand(i);
          auto srcVar = getVar(yieldVal);
          bool isTensor =
              isa<RankedTensorType>(yieldVal.getType());

          // WGMMA accumulator-reset cycle (see emitArithSelect): emit a guarded
          // zero-reset at the loop tail — after wgmma.wait_group 0 — instead of
          // a live-accumulator copy that would serialize the async WGMMAs.
          if (auto arIt = deferredAccReset.find(yieldVal);
              arIt != deferredAccReset.end()) {
            int nElems = getElemsPerThread(yieldVal);
            emit("if (" + arIt->second + ") {");
            indent();
            emit("#pragma unroll");
            emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                 "; _t++) " + iterVar + "[_t] = 0.0f;");
            dedent();
            emit("}");
            continue;
          }

          // Replace srcVar with saved temp if it was saved
          auto savedIt = savedIterVars.find(srcVar);
          std::string effectiveSrc = (savedIt != savedIterVars.end()) ? savedIt->second : srcVar;

          // Skip self-assignments (e.g., when WGMMA reuses accumulator in-place)
          if (effectiveSrc == iterVar) continue;

          if (isTensor) {
            int nElems = getElemsPerThread(yieldVal);
            // Handle scalar-base deferred: iter var is a scalar int delta
            if (scalarBaseDeferred.count(yieldVal)) {
              // Delta was already updated in-place by addptr.
              // effectiveSrc == iterVar → self-assignment → skip.
              // If not self-assignment, do scalar copy:
              if (effectiveSrc != iterVar)
                emit(iterVar + " = " + effectiveSrc + ";");
            } else
            // Handle pointer-based deferred: just copy const char* array
            if (ptrBasedDeferred.count(yieldVal)) {
              emit("#pragma unroll");
              emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                   "; _t++) " + iterVar + "[_t] = " +
                   effectiveSrc + "[_t];");
            } else
            // Handle deferred addptr in yield
            if (auto deferIt = deferredAddPtr.find(yieldVal);
                deferIt != deferredAddPtr.end()) {
              std::string deferSrc = deferIt->second.second;
              auto deferSavedIt = savedIterVars.find(deferSrc);
              std::string effectiveDeferSrc = (deferSavedIt != savedIterVars.end()) ? deferSavedIt->second : deferSrc;

              auto baseIt = iterArgDeferredBase.find(regionIterArgs[i]);
              if (baseIt != iterArgDeferredBase.end() &&
                  baseIt->second == deferIt->second.first) {
                // Same base: just copy offsets (int, not __half*)
                emit("#pragma unroll");
                emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                     "; _t++) " + iterVar + "[_t] = " +
                     effectiveDeferSrc + "[_t];");
              } else {
                // Different base: materialize full pointer
                emit("#pragma unroll");
                emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                     "; _t++) " + iterVar + "[_t] = " +
                     deferIt->second.first + " + " + effectiveDeferSrc + "[_t];");
              }
            } else if (scalarValues.contains(yieldVal)) {
              emit("#pragma unroll");
              emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                   "; _t++) " + iterVar + "[_t] = " + effectiveSrc + ";");
            } else {
              emit("#pragma unroll");
              emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                   "; _t++) " + iterVar + "[_t] = " + effectiveSrc + "[_t];");
            }
          } else if (isa<tt::TensorDescType>(yieldVal.getType())) {
            // Loop-carried descriptor iter var is a const char* pointer.
            std::string srcExpr =
                (savedIt != savedIterVars.end()) ? savedIt->second
                                                 : descAddrExpr(yieldVal);
            if (iterVar != srcExpr)
              emit(iterVar + " = " + srcExpr + ";");
          } else {
            if (iterVar != effectiveSrc)
              emit(iterVar + " = " + effectiveSrc + ";");
          }
        }
      }
    } else {
      emitOp(&bodyOp);
    }
  }

  dedent();
  emit("}");
}
void CUDACodeGen::emitScfIf(scf::IfOp op) {
  auto condVar = getVar(op.getCondition());

  // If the scf.if has results, declare variables to hold them
  unsigned numResults = op.getNumResults();
  llvm::SmallVector<std::string> resultVars;
  for (unsigned i = 0; i < numResults; i++) {
    auto result = op.getResult(i);
    auto var = newVar("arg");
    valueToVar[result] = var;
    resultVars.push_back(var);
    bool isTensor = isa<RankedTensorType>(result.getType());
    if (isTensor) {
      int nElems = getElemsPerThread(result);
      Type elemType = cast<RankedTensorType>(result.getType()).getElementType();
      auto cudaType = getCUDAType(elemType);
      emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    } else if (isa<tt::TensorDescType>(result.getType())) {
      // Descriptor result: hold a pointer to the CUtensorMap.
      pointerDescriptors.insert(result);
      emit("const char* " + var + " = nullptr;");
    } else {
      auto cudaType = getCUDAType(result.getType());
      emit(cudaType + " " + var + ";");
    }
  }

  emit("if (" + condVar + ") {");
  indent();
  for (auto &bodyOp : op.getThenRegion().front()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
      // Map yielded values to the result variables
      for (unsigned i = 0; i < yieldOp.getNumOperands(); i++) {
        auto yieldedVar = getVar(yieldOp.getOperand(i));
        bool isTensor = isa<RankedTensorType>(yieldOp.getOperand(i).getType());
        if (isTensor) {
          int nElems = getElemsPerThread(yieldOp.getOperand(i));
          emit("#pragma unroll");
          emit("for (int _t = 0; _t < " + std::to_string(nElems) +
               "; _t++) " + resultVars[i] + "[_t] = " +
               getElemExpr(yieldOp.getOperand(i), "_t") + ";");
        } else if (isa<tt::TensorDescType>(yieldOp.getOperand(i).getType())) {
          emit(resultVars[i] + " = " + descAddrExpr(yieldOp.getOperand(i)) + ";");
        } else {
          emit(resultVars[i] + " = " + yieldedVar + ";");
        }
      }
    } else {
      emitOp(&bodyOp);
    }
  }
  dedent();
  emit("}");

  if (!op.getElseRegion().empty()) {
    emit("else {");
    indent();
    for (auto &bodyOp : op.getElseRegion().front()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); i++) {
          auto yieldedVar = getVar(yieldOp.getOperand(i));
          bool isTensor = isa<RankedTensorType>(yieldOp.getOperand(i).getType());
          if (isTensor) {
            int nElems = getElemsPerThread(yieldOp.getOperand(i));
            emit("#pragma unroll");
            emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                 "; _t++) " + resultVars[i] + "[_t] = " +
                 getElemExpr(yieldOp.getOperand(i), "_t") + ";");
          } else if (isa<tt::TensorDescType>(yieldOp.getOperand(i).getType())) {
            emit(resultVars[i] + " = " + descAddrExpr(yieldOp.getOperand(i)) + ";");
          } else {
            emit(resultVars[i] + " = " + yieldedVar + ";");
          }
        }
      } else {
        emitOp(&bodyOp);
      }
    }
    dedent();
    emit("}");
  }
}
// scf.index_switch: produced by lift-cfg-to-scf for multi-way merges
// (cf.switch). Emit a C switch; each case region yields into the shared
// result variables (same convention as emitScfIf).
void CUDACodeGen::emitScfIndexSwitch(scf::IndexSwitchOp op) {
  auto argVar = getVar(op.getArg());

  unsigned numResults = op.getNumResults();
  llvm::SmallVector<std::string> resultVars;
  for (unsigned i = 0; i < numResults; i++) {
    auto result = op.getResult(i);
    auto var = newVar("arg");
    valueToVar[result] = var;
    resultVars.push_back(var);
    if (auto rtt = dyn_cast<RankedTensorType>(result.getType())) {
      int nElems = getElemsPerThread(result);
      emit(getCUDAType(rtt.getElementType()) + " " + var + "[" +
           std::to_string(nElems) + "];");
    } else if (isa<tt::TensorDescType>(result.getType())) {
      pointerDescriptors.insert(result);
      emit("const char* " + var + " = nullptr;");
    } else {
      emit(getCUDAType(result.getType()) + " " + var + ";");
    }
  }

  auto emitCaseBody = [&](Region &region) {
    for (auto &bodyOp : region.front()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
        for (unsigned i = 0; i < yieldOp.getNumOperands(); i++) {
          Value y = yieldOp.getOperand(i);
          if (isa<RankedTensorType>(y.getType())) {
            int nElems = getElemsPerThread(y);
            emit("#pragma unroll");
            emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                 "; _t++) " + resultVars[i] + "[_t] = " +
                 getElemExpr(y, "_t") + ";");
          } else if (isa<tt::TensorDescType>(y.getType())) {
            emit(resultVars[i] + " = " + descAddrExpr(y) + ";");
          } else {
            emit(resultVars[i] + " = " + getVar(y) + ";");
          }
        }
      } else {
        emitOp(&bodyOp);
      }
    }
  };

  emit("switch ((long long)" + argVar + ") {");
  auto cases = op.getCases();
  for (unsigned c = 0; c < cases.size(); c++) {
    emit("case " + std::to_string(cases[c]) + ": {");
    indent();
    emitCaseBody(op.getCaseRegions()[c]);
    emit("break;");
    dedent();
    emit("}");
  }
  emit("default: {");
  indent();
  emitCaseBody(op.getDefaultRegion());
  emit("break;");
  dedent();
  emit("}");
  emit("}");
}

void CUDACodeGen::emitScfWhile(scf::WhileOp op) {
  // Initialize iteration variables
  auto inits = op.getInits();
  auto beforeArgs = op.getBeforeArguments();
  llvm::SmallVector<std::string> iterVars;

  for (unsigned i = 0; i < inits.size(); i++) {
    auto initVar = getVar(inits[i]);
    auto iterVar = newVar("witer");
    valueToVar[beforeArgs[i]] = iterVar;
    iterVars.push_back(iterVar);

    bool isTensor = isa<RankedTensorType>(inits[i].getType());
    if (isTensor) {
      int nElems = getElemsPerThread(inits[i]);
      Type elemType = cast<RankedTensorType>(inits[i].getType()).getElementType();
      auto cudaType = getCUDAType(elemType);
      emit(cudaType + " " + iterVar + "[" + std::to_string(nElems) + "];");
      emit("for (int _t = 0; _t < " + std::to_string(nElems) +
           "; _t++) " + iterVar + "[_t] = " + getElemExpr(inits[i], "_t") + ";");
    } else if (isa<tt::TensorDescType>(inits[i].getType())) {
      emit("const char* " + iterVar + " = " + descAddrExpr(inits[i]) + ";");
      pointerDescriptors.insert(beforeArgs[i]);
    } else {
      auto cudaType = getCUDAType(inits[i].getType());
      emit(cudaType + " " + iterVar + " = " + initVar + ";");
    }
  }

  // ── In-place loop-carry aliasing (opt-in via TRITON_EMIT_INPLACE_ELTWISE) ──
  // The default lowering gives each loop-carried tensor TWO register arrays:
  // `witer` (the iter-arg / yield target) and `wcond` (the before→after
  // forwarded copy the body accumulates into), with a per-iteration copy
  // `wcond[_t]=witer[_t]`. Both stay live across the whole body → the
  // persistent accumulators (dv/dk in FA3-bwd) are DOUBLED, which is the real
  // 316B-spill / C7512 driver (ptxas can't coalesce two [64] arrays live across
  // WGMMA async-waits). When the before-region simply FORWARDS the iter-arg
  // (condArgs[i] == beforeArgs[i], the ordinary for-loop case), `wcond` can
  // ALIAS `witer`: one array, no copy. Swap-safety guard below: never alias an
  // arg that a DIFFERENT yield reads (would clobber it mid yield-copy phase).
  static const bool inplaceCarry =
      (getenv("TRITON_EMIT_INPLACE_ELTWISE") != nullptr) &&
      (std::string(getenv("TRITON_EMIT_INPLACE_ELTWISE")) == "1");
  auto beforeArgsAll = op.getBeforeArguments();
  auto afterArgsAll = op.getAfterArguments();
  auto condTerm = dyn_cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
  auto yieldTerm = dyn_cast<scf::YieldOp>(op.getAfter().front().getTerminator());
  llvm::SmallVector<bool> aliasCarry(inits.size(), false);
  if (inplaceCarry && condTerm && yieldTerm) {
    auto condArgs = condTerm.getArgs();
    // Candidate: pure forward of the iter-arg, and a register tensor.
    for (unsigned i = 0; i < inits.size(); i++)
      aliasCarry[i] = isa<RankedTensorType>(inits[i].getType()) &&
                      i < condArgs.size() && i < beforeArgsAll.size() &&
                      condArgs[i] == beforeArgsAll[i];
    // Swap-safety: disable aliasing for any after-arg k that some yield operand
    // j != k reads (transitively). Aliasing k would overwrite witer[k] during
    // yield[j<k? — any order]; only self-referential accumulation is safe.
    for (unsigned k = 0; k < inits.size(); k++) {
      if (!aliasCarry[k] || k >= afterArgsAll.size())
        continue;
      Value selfAfter = afterArgsAll[k];
      for (unsigned j = 0; j < yieldTerm.getNumOperands(); j++) {
        if (j == k)
          continue;
        llvm::SmallPtrSet<Value, 16> seen;
        std::function<bool(Value)> reads = [&](Value v) -> bool {
          if (v == selfAfter)
            return true;
          if (!seen.insert(v).second)
            return false;
          if (Operation *d = v.getDefiningOp())
            for (Value o : d->getOperands())
              if (reads(o))
                return true;
          return false;
        };
        if (reads(yieldTerm.getOperand(j))) {
          aliasCarry[k] = false;
          break;
        }
      }
    }
  }

  // Create condition output variables + map results
  llvm::SmallVector<std::string> condOutVars;
  for (unsigned i = 0; i < inits.size(); i++) {
    // Aliased carry: reuse the witer array as the condition/after-arg storage —
    // no separate wcond array, no per-iteration copy.
    if (aliasCarry[i]) {
      condOutVars.push_back(iterVars[i]);
      if (i < op.getNumResults())
        valueToVar[op.getResult(i)] = iterVars[i];
      continue;
    }
    auto condVar = newVar("wcond");
    condOutVars.push_back(condVar);
    bool isTensor = isa<RankedTensorType>(inits[i].getType());
    if (isTensor) {
      int nElems = getElemsPerThread(inits[i]);
      Type elemType = cast<RankedTensorType>(inits[i].getType()).getElementType();
      emit(getCUDAType(elemType) + " " + condVar + "[" + std::to_string(nElems) + "];");
    } else if (isa<tt::TensorDescType>(inits[i].getType())) {
      emit("const char* " + condVar + " = nullptr;");
      if (i < op.getNumResults())
        pointerDescriptors.insert(op.getResult(i));
      if (i < op.getAfterArguments().size())
        pointerDescriptors.insert(op.getAfterArguments()[i]);
    } else {
      emit(getCUDAType(inits[i].getType()) + " " + condVar + ";");
    }
    // Map while op results
    if (i < op.getNumResults())
      valueToVar[op.getResult(i)] = condVar;
  }

  emit("while (true) {");
  indent();

  // Emit condition region (before block)
  auto &condBlock = op.getBefore().front();
  for (auto &condOp : condBlock) {
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(&condOp)) {
      auto condVar = getVar(conditionOp.getCondition());
      // Copy yielded values to condOutVars
      auto condArgs = conditionOp.getArgs();
      for (unsigned i = 0; i < condArgs.size() && i < condOutVars.size(); i++) {
        auto srcVar = getVar(condArgs[i]);
        // Aliased loop-carry: condOutVar IS the witer array → copy is a no-op.
        if (srcVar == condOutVars[i])
          continue;
        bool isTensor = isa<RankedTensorType>(condArgs[i].getType());
        if (isTensor) {
          int nElems = getElemsPerThread(condArgs[i]);
          emit("for (int _t = 0; _t < " + std::to_string(nElems) +
               "; _t++) " + condOutVars[i] + "[_t] = " +
               getElemExpr(condArgs[i], "_t") + ";");
        } else if (isa<tt::TensorDescType>(condArgs[i].getType())) {
          emit(condOutVars[i] + " = " + descAddrExpr(condArgs[i]) + ";");
        } else {
          emit(condOutVars[i] + " = " + srcVar + ";");
        }
      }
      emit("if (!" + condVar + ") break;");
    } else {
      emitOp(&condOp);
    }
  }

  // Emit body region (after block)
  auto &bodyBlock = op.getAfter().front();
  auto afterArgs = op.getAfterArguments();
  for (unsigned i = 0; i < afterArgs.size() && i < condOutVars.size(); i++)
    valueToVar[afterArgs[i]] = condOutVars[i];

  for (auto &bodyOp : bodyBlock) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
      for (unsigned i = 0; i < yieldOp.getNumOperands() && i < iterVars.size(); i++) {
        auto srcVar = getVar(yieldOp.getOperand(i));
        bool isTensor = isa<RankedTensorType>(yieldOp.getOperand(i).getType());
        if (isTensor) {
          // Aliased in-place accumulation: yield value already lives in the
          // witer array (body accumulated into it) → skip the self-copy.
          if (iterVars[i] != srcVar) {
            int nElems = getElemsPerThread(yieldOp.getOperand(i));
            emit("#pragma unroll");
            emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                 "; _t++) " + iterVars[i] + "[_t] = " +
                 getElemExpr(yieldOp.getOperand(i), "_t") + ";");
          }
        } else if (isa<tt::TensorDescType>(yieldOp.getOperand(i).getType())) {
          std::string srcExpr = descAddrExpr(yieldOp.getOperand(i));
          if (iterVars[i] != srcExpr)
            emit(iterVars[i] + " = " + srcExpr + ";");
        } else {
          if (iterVars[i] != srcVar)
            emit(iterVars[i] + " = " + srcVar + ";");
        }
      }
    } else {
      emitOp(&bodyOp);
    }
  }

  dedent();
  emit("}");
}
void CUDACodeGen::emitScfYield(scf::YieldOp op, scf::ForOp parentFor) {
  emit("// TODO: scf.yield not yet ported to C++");
}

void CUDACodeGen::emitLocalAlloc(ttg::LocalAllocOp op) {
  auto result = op.getResult();
  auto memDescType = cast<ttg::MemDescType>(result.getType());
  auto elemType = memDescType.getElementType();
  auto shape = memDescType.getShape();
  auto cudaType = getCUDAType(elemType);
  int elemBytes = getTypeSizeInBytes(elemType);

  // Compute allocation size
  int64_t totalElems = 1;
  for (auto s : shape) totalElems *= s;
  int byteSize = totalElems * elemBytes;

  // Align to 128 bytes for WGMMA by default.
  // For nvmma_shared (TMA-swizzled) buffers the TMA hardware computes the
  // swizzle phase from the ABSOLUTE shared address, while register<->shared
  // stores (emitLocalStore / cp.async) compute it from the buffer-relative row.
  // To keep the two consistent the buffer must start at swizzle-phase 0, i.e.
  // aligned to the full swizzle period = perPhase * maxPhase rows of
  // swizzleByteWidth bytes each = (128/W) * (W/16) * W = 8 * W bytes.
  // (For W=128 this is 1024; for W=64 it is 512 — NOT W*maxPhase=256, which
  // left fp8 64-byte-wide epilogue subtiles at offset%512=256 and corrupted
  // every TMA C-store: tut-09 fp8 BN128/EPILOGUE_SUBTILE, task #50.)
  // Otherwise an offset such as 65664 = 64*1024 + 128 introduces a constant
  // phase shift and corrupts data.
  int alignBytes = 128;
  if (auto nvmma =
          dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(memDescType.getEncoding())) {
    int swizzleBytes = nvmma.getSwizzlingByteWidth();
    if (swizzleBytes > 0) {
      int period = 8 * swizzleBytes; // perPhase * maxPhase * rowBytes
      if (period > alignBytes) alignBytes = period;
    }
  }
  // Retire any sourced (deallocless) buffer whose last use precedes this alloc,
  // then drop the bump pointer to the top of whatever is still live so this new
  // buffer can reuse the freed space (without aliasing a still-live tile).
  //
  // EXCEPTION: allocs inside ttg.warp_specialize regions. The textual op order
  // says partition1's epilogue alloc comes "after" partition0's last use, but
  // the partitions execute CONCURRENTLY (different warp groups) — reusing the
  // offset makes both partitions stmatrix into the same bytes while the other
  // partition's async TMA store is still reading them (tut-09 descriptor_
  // persistent_ws K=256: ~25% nondeterministically wrong C elements). Skip
  // retirement/reuse entirely so each partition gets a disjoint buffer.
  bool inWarpSpecialize =
      op->getParentOfType<ttg::WarpSpecializeOp>() != nullptr;
  if (!inWarpSpecialize) {
    int myOrder = smemOpOrder.lookup(op.getOperation());
    bool retiredAny = false;
    for (Value sb : sourcedBuffers) {
      if (sb == result || !liveSmemTop.count(sb))
        continue; // already retired or not yet allocated
      if (sourcedLastUse.lookup(sb) < myOrder) {
        liveSmemTop.erase(sb);
        retiredAny = true;
      }
    }
    if (retiredAny) {
      peakSharedMem = std::max(peakSharedMem, sharedMemOffset);
      sharedMemOffset = std::min(sharedMemOffset, recomputeSmemFloor());
    }
  }

  int offset = (sharedMemOffset + (alignBytes - 1)) & ~(alignBytes - 1);
  sharedMemOffset = offset + byteSize;
  if (sharedMemOffset > peakSharedMem) peakSharedMem = sharedMemOffset;

  auto var = newVar("smem");
  valueToVar[result] = var;
  valueToSmemOffset[result] = offset;
  liveSmemTop[result] = offset + byteSize;
  emit(cudaType + "* " + var + " = (" + cudaType + "*)(shared_mem + " +
       std::to_string(offset) + ");");

  // If op has a source operand, store data from registers to shared memory
  if (op.getSrc()) {
    auto src = op.getSrc();

    // Check if source is a deferred load — fuse into cp.async
    if (deferredCpAsyncLoads.contains(src)) {
      auto loadOp = cast<tt::LoadOp>(src.getDefiningOp());
      emitCpAsyncToShared(loadOp.getPtr(), var, memDescType,
                          loadOp.getMask(), loadOp.getOther());
      hasPendingCpAsync = true;
      return;
    }

    auto srcVar = getVar(src);
    bool srcIsTensor = isa<RankedTensorType>(src.getType());

    if (srcIsTensor) {
      auto srcRtt = cast<RankedTensorType>(src.getType());
      auto srcEnc = srcRtt.getEncoding();
      int nElems = getElemsPerThread(src);

      // Check if destination has nvmma_shared layout (needs swizzled store)
      auto destEnc = memDescType.getEncoding();
      auto nvmmaShared = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(destEnc);

      if (auto mmaEnc = dyn_cast_or_null<ttg::NvidiaMmaEncodingAttr>(srcEnc)) {
        auto wpcChk = mmaEnc.getWarpsPerCTA();
        bool nSplitWg = wpcChk.size() > 1 && wpcChk[1] > 1;
        // N-split warpgroups (warpsPerCTA=[4,2], BLOCK_M=64 num_warps=8) are
        // supported by the stmatrix path below iff each N-warpgroup owns a
        // whole number of 64-column N-blocks and warpgroups don't split both
        // dims at once.
        int numWgN = nSplitWg ? (int)wpcChk[1] : 1;
        // Per-CTA register convention (see emitBlockedStoreToSmem): with
        // CTASplitNum>1 each CTA's accumulator and smem tile cover only its
        // slice, so all rep/block math below must use shapePerCTA.
        auto shapePerCTAStm = ttg::getShapePerCTA(mmaEnc, shape);
        bool nSplitOk =
            !nSplitWg ||
            (shapePerCTAStm.size() > 1 &&
             ((int)shapePerCTAStm[1] / numWgN) % 64 == 0 && wpcChk[0] <= 4);
        if (elemBytes == 1 || !nSplitOk) {
          // fp8 (1-byte): stmatrix.b16 cannot express a 1-byte shared layout.
          // Unsupported N-split shapes use the general LinearLayout
          // invertAndCompose store (same path as ttg.local_store). No
          // __syncthreads here — the subsequent fence.proxy.async +
          // __syncthreads provides the barrier.
          emitLayoutAwareSharedStore(src, memDescType, var);
          return;
        }
        // MMA layout → shared memory via stmatrix (swizzle=128B).
        //
        // Generalized for arbitrary BLOCK_M / BLOCK_N. The C tile [BM, BN] is
        // stored as nBlocks = BN/64 swizzled N-blocks, each [BM,64] occupying
        // nblockBytes = BM*64*elemBytes contiguous bytes. A warpgroup computes
        // m64PerWg WGMMA m64nBN tiles; each tile owns a 64-row stripe
        // (stripeBytes = 64*64*elemBytes within an N-block) and BN/2 fp16
        // accumulators laid out as [N-block(blk)][nGrp(4)][4 regs].
        //
        // The per-thread accumulator srcVar has nElems = m64PerWg*(BN/2) fp16
        // values, ordered [m64 block (mb)][N-block (blk)][nGrp][4]. The shared
        // byte offset for (mb, blk, nGrp) is:
        //   blk*nblockBytes + (wg_m*m64PerWg + mb)*stripeBytes  (+ swizzle).
        // The stripe index (wg_m*m64PerWg + mb) is unambiguous because for the
        // supported BM=128, exactly one of numWgM / m64PerWg is >1.
        //
        // The prior hardcoded version (32 packed, 2 N-blocks, single mb) only
        // matched BN=128 / num_warps=8; it dropped 64-of-128 accumulators for
        // num_warps=4 and 2-of-4 N-blocks for BN=256.
        int BM = shapePerCTAStm[0];
        int BN = shapePerCTAStm[1];
        auto wpc = mmaEnc.getWarpsPerCTA();
        int numWgM = std::max<int>(1, (int)wpc[0] / 4);
        int BNwg = BN / numWgN;                // columns owned per N-warpgroup
        int m64PerWg = std::max(1, (BM / numWgM) / 64);
        int nBlocks = std::max(1, BNwg / 64);  // N-blocks per warpgroup
        int nblockBytes = BM * 64 * elemBytes;
        int stripeBytes = 64 * 64 * elemBytes;
        int packedPerMb = BNwg / 4;            // 16 packed per 64-col N-block
        int nPacked = m64PerWg * packedPerMb;  // = nElems/2
        // Warpgroup decomposition: wgLinear = warp_id/4 (== _wg_m in the
        // preamble). With 8 warps exactly one of numWgM/numWgN is >1, so
        // wgLinear is either the M index or the N index directly.
        std::string wgM = numWgM > 1 ? "_wg_m" : "0";
        std::string wgN = numWgN > 1 ? "_wg_m" : "0";

        emit("// MMA→shared via stmatrix (swizzle=128B, generalized)");
        emit("{");
        indent();
        emit("uint32_t _base = ((tid << 7) & 0x780) | ((tid << 4) & 0x70);");
        emit("_base = (_base ^ (tid & 0x10)) | ((tid << 6) & 0x1800);");
        emit("char* _smem_base = (char*)(shared_mem + " + std::to_string(offset) + ");");
        emit("uint32_t _packed[" + std::to_string(nPacked) + "];");
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(nPacked) + "; _i++) {");
        indent();
        // Reinterpret the raw 16 bits of each element directly. srcVar already
        // holds values in the output element type (fp16 or bf16); going through
        // (__half) would reinterpret bf16 values with the fp16 exponent layout
        // and corrupt the stmatrix payload (stmatrix.b16 just moves 16-bit
        // lanes — the bits must already be the destination dtype's bits).
        emit("uint32_t _lo = *(uint16_t*)&" + srcVar + "[2*_i];");
        emit("uint32_t _hi = *(uint16_t*)&" + srcVar + "[2*_i+1];");
        emit("_packed[_i] = _lo | (_hi << 16);");
        dedent();
        emit("}");
        for (int mb = 0; mb < m64PerWg; mb++) {
          for (int nGrp = 0; nGrp < 4; nGrp++) {
            for (int blk = 0; blk < nBlocks; blk++) {
              int localP = blk * 16 + nGrp * 4;
              int packedStart = mb * packedPerMb + localP;
              int xorVal = nGrp * 32;  // n-group column offset within swizzle
              // Global m64 stripe index. WGMMA compute interleaves warpgroups
              // along M (rep r covers rows r*64*numWgM + wg_m*64, see the A
              // descriptor offset _wg_m*64rows + mb*64*numWgM rows), so the
              // store must use mb*numWgM + wg_m — NOT wg_m*m64PerWg + mb,
              // which only coincides when one of numWgM/m64PerWg is 1
              // (e.g. BM=128; BM>=256 with nw=8 exposed the mismatch).
              std::string stripeExpr = "(" + std::to_string(mb) + " * " +
                                       std::to_string(numWgM) + " + " + wgM + ")";
              std::string smemOff = "(" + wgN + " * " + std::to_string(nBlocks) +
                                    " + " + std::to_string(blk) + ") * " +
                                    std::to_string(nblockBytes) + " + " +
                                    stripeExpr + " * " + std::to_string(stripeBytes);
              std::string addrExpr =
                  xorVal ? ("_base ^ " + std::to_string(xorVal)) : "_base";
              emit("asm volatile(\"stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\"");
              emit("    :: \"r\"((unsigned)__cvta_generic_to_shared(_smem_base + (" +
                   addrExpr + ") + (" + smemOff + "))),");
              emit("       \"r\"(_packed[" + std::to_string(packedStart) + "]), \"r\"(_packed[" +
                   std::to_string(packedStart+1) + "]),");
              emit("       \"r\"(_packed[" + std::to_string(packedStart+2) + "]), \"r\"(_packed[" +
                   std::to_string(packedStart+3) + "]));");
            }
          }
        }
        dedent();
        emit("}");
      } else if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(srcEnc)) {
        if (nvmmaShared && shape.size() == 2) {
          // Swizzled store: blocked → nvmma_shared
          int swizzleBytes = nvmmaShared.getSwizzlingByteWidth();
          int elemBits = nvmmaShared.getElementBitWidth();
          // Register convention is PER-CTA (see emitBlockedStoreToSmem): with
          // CTASplitNum>1 each CTA holds and stores only its slice, so rep
          // counts and the smem tile must use shapePerCTA, not the full shape.
          auto shapePerCTA = ttg::getShapePerCTA(blocked, shape);
          int rows = shapePerCTA[0], cols = shapePerCTA[1];
          int rowStrideBytes = cols * elemBytes;
          int vec = 16 / elemBytes;  // 16 bytes = one swizzle vector
          int perPhase = std::max(1, 128 / rowStrideBytes);
          int maxPhase = swizzleBytes / 16;

          auto spt = blocked.getSizePerThread();
          auto tpw = blocked.getThreadsPerWarp();
          auto wpc = blocked.getWarpsPerCTA();
          int spt0 = spt[0], spt1 = spt[1];
          int tpw0 = tpw[0], tpw1 = tpw[1];
          int wpc0 = wpc[0], wpc1 = wpc[1];
          int totalT0 = tpw0 * wpc0, totalT1 = tpw1 * wpc1;
          int reps0 = std::max(1, rows / (totalT0 * spt0));
          int reps1 = std::max(1, cols / (totalT1 * spt1));
          int strideRep0 = spt1 * reps1 * spt0;
          // Blocked layout order: order[0] is the fastest-varying dim for
          // register/lane/warp decomposition. order=[1,0] (row-major source)
          // is the historical default; order=[0,1] (e.g. trans_b=True int8 B
          // tile loaded column-major) flips all three decompositions.
          auto blkOrder = blocked.getOrder();
          bool dim0Fast = (blkOrder[0] == 0);
          auto emitPosDecomp = [&]() {
            if (dim0Fast) {
              emit("int _lane_d0 = lane_id % " + std::to_string(tpw0) + ";");
              emit("int _lane_d1 = lane_id / " + std::to_string(tpw0) + ";");
              emit("int _warp_d0 = warp_id % " + std::to_string(wpc0) + ";");
              emit("int _warp_d1 = warp_id / " + std::to_string(wpc0) + ";");
            } else {
              emit("int _lane_d0 = lane_id / " + std::to_string(tpw1) + ";");
              emit("int _lane_d1 = lane_id % " + std::to_string(tpw1) + ";");
              emit("int _warp_d0 = warp_id / " + std::to_string(wpc1) + ";");
              emit("int _warp_d1 = warp_id % " + std::to_string(wpc1) + ";");
            }
            emit("int _pos0 = _lane_d0 + _warp_d0 * " + std::to_string(tpw0) + ";");
            emit("int _pos1 = _lane_d1 + _warp_d1 * " + std::to_string(tpw1) + ";");
            // CTA tile larger than the tensor shape: threads wrap modulo the
            // shape (duplicate writers store identical values). Without this,
            // out-of-range _pos writes past the smem tile (IMA).
            if (totalT0 * spt0 > rows && rows % spt0 == 0)
              emit("_pos0 %= " + std::to_string(std::max(1, rows / spt0)) + ";");
            if (totalT1 * spt1 > cols && cols % spt1 == 0)
              emit("_pos1 %= " + std::to_string(std::max(1, cols / spt1)) + ";");
          };
          auto decomposeReg = [&](int i, int &s0, int &rep0, int &s1, int &rep1) {
            if (dim0Fast) {
              s0 = i % spt0;
              rep0 = (i / spt0) % reps0;
              s1 = (i / (spt0 * reps0)) % spt1;
              rep1 = i / (spt0 * reps0 * spt1);
            } else {
              rep0 = i / strideRep0;
              s0 = (i / (spt1 * reps1)) % spt0;
              rep1 = (i / spt1) % reps1;
              s1 = i % spt1;
            }
          };

          // Swizzled store: blocked → nvmma_shared with precomputed offsets.
          // Instead of runtime address computation (swizzle, transpose, phase),
          // precompute the shared memory byte offset for each register element
          // at code generation time. This eliminates ~132 integer ALU ops per
          // iteration and reduces register pressure, matching the PTX/LLVM
          // backend which uses fixed offsets in st.shared.b8 instructions.
          {
            bool isTransposed = nvmmaShared.getTransposed();
            int phys_rows, phys_cols;
            if (isTransposed) {
              phys_rows = cols;  // N
              phys_cols = rows;  // K
            } else {
              phys_rows = rows;
              phys_cols = cols;
            }
            int phys_rowStride = phys_cols; // in elements
            int phys_perPhase = std::max(1, 128 / (phys_cols * elemBytes));
            int phys_maxPhase = swizzleBytes / 16;
            // swizzleBytes == 0 means no swizzling: treat the whole physical
            // row as one "swizzle row" so the tiling/phase logic degenerates
            // to a plain row-major store (avoids div-by-zero below).
            int elemsPerSwizzlingRow =
                swizzleBytes > 0 ? swizzleBytes / elemBytes : phys_cols;

            // Precompute the byte offset from the thread's base address for
            // each register element. The offset decomposes as:
            //   base_offset(thread) + element_offset(i)
            // where base_offset depends on lane_id/warp_id (runtime) and
            // element_offset depends only on the register index i (compile-time).
            //
            // For each element i, compute: log_row, log_col from blocked layout,
            // then physical row/col (with transpose), then swizzled address.
            // Split into base (depends on _pos0, _pos1) + offset (constant per i).

            // Compute per-element offsets at code-gen time
            std::vector<int> elemOffsets(nElems);
            // The base address is: _pos0 * spt0 * (isTransposed ? 1 : phys_rowStride)
            //                    + _pos1 * spt1 * (isTransposed ? phys_rowStride : 1)
            // But with swizzle, it's easier to compute relative offsets.
            // For element i:
            //   _rep0 = i / strideRep0
            //   _s0 = (i / (spt1 * reps1)) % spt0
            //   _rep1 = (i / spt1) % reps1
            //   _s1 = i % spt1
            //   _log_row_offset = _s0 + _rep0 * totalT0 * spt0  (add _pos0 * spt0 at runtime)
            //   _log_col_offset = _s1 + _rep1 * totalT1 * spt1  (add _pos1 * spt1 at runtime)
            // After transpose:
            //   phys_row_offset, phys_col_offset
            // Address = phys_row * phys_rowStride + swizzled_col

            // Since _pos0 contributes to one dimension and _pos1 to the other,
            // decompose as: addr = _pos_row_factor * phys_rowStride + _pos_col_base + per_elem_offset
            // where per_elem_offset is constant.
            //
            // To get truly constant offsets, we need _pos0 and _pos1 to contribute
            // only as a single base address. This works when:
            //   base = _pos_row * phys_rowStride + _pos_col (in elements, no swizzle)
            // and the swizzle only depends on the element index, not the base.
            //
            // For transposed layout (fp8 B tile):
            //   _pos0 → K dimension (physical col)
            //   _pos1 → N dimension (physical row)
            //   base = _pos1 * spt1 * phys_rowStride + _pos0 * spt0
            //
            //   For element i:
            //     phys_row = _pos1 * spt1 + _s1 + _rep1 * totalT1 * spt1
            //     phys_col = _pos0 * spt0 + _s0 + _rep0 * totalT0 * spt0
            //     phase = (phys_row / phys_perPhase) % phys_maxPhase
            //     swizzled_col = phys_col ^ (phase * vec)  [in multi-tile: colInTile ^ ...]
            //
            //   phase depends on phys_row which depends on _pos1. BUT:
            //   - spt1 is always >= 16 in typical configs, so _pos1 * spt1 is always
            //     a multiple of phys_maxPhase * phys_perPhase (ensuring phase only
            //     depends on _s1/_rep1, which are constant per element).
            //   Actually, we need: (_pos1 * spt1) % (phys_maxPhase * phys_perPhase) == 0
            //   For our fp8 case: spt1=16, phys_perPhase=1, phys_maxPhase=8
            //   → _pos1*16 % 8 = 0 ✓

            // Check if we can use the fast path (constant offsets)
            bool canPrecompute = true;
            // The phase must be independent of _pos0/_pos1
            int posRowFactor = isTransposed ? spt1 : spt0;
            int phasePeriod = phys_maxPhase * phys_perPhase;
            if (phasePeriod > 1 && (posRowFactor % phasePeriod) != 0)
              canPrecompute = false;
            // Thread wrap (CTA tile > shape) only handled via _pos modulo,
            // which requires spt to divide the shape; otherwise use the
            // runtime fallback which wraps the full logical index.
            if ((totalT0 * spt0 > rows && rows % spt0 != 0) ||
                (totalT1 * spt1 > cols && cols % spt1 != 0))
              canPrecompute = false;

            // The precomputed offsets bake the swizzle XOR into the
            // per-element constant and ADD the runtime col-side base
            // (_pos * factor) afterwards. The true address XORs the FULL
            // column (base + elemOff). These differ when the base has bits
            // inside the swizzle mask range (e.g. int8 B tile 32xK:
            // _pos0*spt0 spans bytes 0..31, mask flips bit 4 → +16 vs ^16).
            // Verify exact equivalence over all pos/element combinations at
            // codegen time; fall back to the runtime path on any mismatch.
            if (canPrecompute && phys_maxPhase > 1) {
              int posColFactor = isTransposed ? spt0 : spt1;
              int posColCount = isTransposed ? totalT0 : totalT1;
              for (int p = 0; p < posColCount && canPrecompute; p++) {
                int baseCol = p * posColFactor;
                for (int i = 0; i < nElems && canPrecompute; i++) {
                  int s0, rep0, s1, rep1;
                  decomposeReg(i, s0, rep0, s1, rep1);
                  int logRowOff = s0 + rep0 * totalT0 * spt0;
                  int logColOff = s1 + rep1 * totalT1 * spt1;
                  int physRowOff = isTransposed ? logColOff : logRowOff;
                  int physColOff = isTransposed ? logRowOff : logColOff;
                  int phase = (physRowOff / phys_perPhase) % phys_maxPhase;
                  int fullCol = baseCol + physColOff;
                  int64_t exactAddr, precompAddr;
                  if (phys_cols > elemsPerSwizzlingRow) {
                    int tK = fullCol / elemsPerSwizzlingRow;
                    int cT = fullCol % elemsPerSwizzlingRow;
                    exactAddr = (int64_t)tK * phys_rows * elemsPerSwizzlingRow +
                                (int64_t)physRowOff * elemsPerSwizzlingRow +
                                (cT ^ (phase * vec));
                    int tKe = physColOff / elemsPerSwizzlingRow;
                    int cTe = physColOff % elemsPerSwizzlingRow;
                    precompAddr =
                        (int64_t)baseCol +
                        (int64_t)tKe * phys_rows * elemsPerSwizzlingRow +
                        (int64_t)physRowOff * elemsPerSwizzlingRow +
                        (cTe ^ (phase * vec));
                  } else {
                    exactAddr = (int64_t)physRowOff * phys_cols +
                                (fullCol ^ (phase * vec));
                    precompAddr = (int64_t)baseCol +
                                  (int64_t)physRowOff * phys_cols +
                                  (physColOff ^ (phase * vec));
                  }
                  if (exactAddr != precompAddr)
                    canPrecompute = false;
                }
              }
            }

            if (canPrecompute) {
              // Precompute offsets at code-gen time
              for (int i = 0; i < nElems; i++) {
                int s0, rep0, s1, rep1;
                decomposeReg(i, s0, rep0, s1, rep1);

                int logRowOff = s0 + rep0 * totalT0 * spt0;
                int logColOff = s1 + rep1 * totalT1 * spt1;

                int physRowOff, physColOff;
                if (isTransposed) {
                  physRowOff = logColOff;
                  physColOff = logRowOff;
                } else {
                  physRowOff = logRowOff;
                  physColOff = logColOff;
                }

                int phase = (phys_maxPhase > 1)
                    ? (physRowOff / phys_perPhase) % phys_maxPhase
                    : 0;

                int addr;
                if (phys_cols > elemsPerSwizzlingRow) {
                  int tileK = physColOff / elemsPerSwizzlingRow;
                  int colInTile = physColOff % elemsPerSwizzlingRow;
                  int swizzledCol = colInTile ^ (phase * vec);
                  addr = tileK * phys_rows * elemsPerSwizzlingRow +
                         physRowOff * elemsPerSwizzlingRow + swizzledCol;
                } else {
                  int swizzledCol = physColOff ^ (phase * vec);
                  addr = physRowOff * phys_cols + swizzledCol;
                }
                elemOffsets[i] = addr * elemBytes;
              }

              // Emit: compute base address from _pos0, _pos1, then use fixed offsets
              emit("// Swizzled store: blocked→nvmma_shared (" +
                   std::to_string(rows) + "x" + std::to_string(cols) + " " +
                   cudaType + ", swizzle=" + std::to_string(swizzleBytes) +
                   "B, precomputed offsets)");
              emit("{");
              indent();
              emitPosDecomp();

              // Base byte address: depends on which dimension _pos0/_pos1 map to
              if (isTransposed) {
                // _pos0 → K (physical col), _pos1 → N (physical row)
                // For multi-tile case, _pos0*spt0 might span multiple tiles
                if (phys_cols > elemsPerSwizzlingRow) {
                  // base = tile_base + _pos1*spt1*elemsPerSwizzlingRow + _pos0*spt0_within_tile
                  // This is complex; use the simpler approach with swizzle at runtime
                  // for the base, but offsets still constant
                  int baseTile = 0; // _pos0 * spt0 / elemsPerSwizzlingRow
                  int baseColInTile = 0; // _pos0 * spt0 % elemsPerSwizzlingRow
                  // For now, compute the base phase from _pos1*spt1 (which is 0 mod phasePeriod)
                  // The base swizzle is: baseColInTile ^ (basePhase * vec) = _pos0*spt0 ^ 0 = _pos0*spt0
                  emit("int _base = _pos0 * " + std::to_string(spt0 * elemBytes) +
                       " + _pos1 * " + std::to_string(spt1 * elemsPerSwizzlingRow * elemBytes) + ";");
                } else {
                  // Single tile: base = _pos1*spt1*phys_cols + _pos0*spt0
                  // Phase at _pos1*spt1 is 0 (guaranteed by canPrecompute check)
                  // So base swizzle = _pos0*spt0 ^ 0 = _pos0*spt0
                  emit("int _base = _pos0 * " + std::to_string(spt0 * elemBytes) +
                       " + _pos1 * " + std::to_string(spt1 * phys_cols * elemBytes) + ";");
                }
              } else {
                // _pos0 → M (physical row), _pos1 → K/N (physical col)
                if (phys_cols > elemsPerSwizzlingRow) {
                  emit("int _base = _pos0 * " + std::to_string(spt0 * elemsPerSwizzlingRow * elemBytes) +
                       " + _pos1 * " + std::to_string(spt1 * elemBytes) + ";");
                } else {
                  emit("int _base = _pos0 * " + std::to_string(spt0 * phys_cols * elemBytes) +
                       " + _pos1 * " + std::to_string(spt1 * elemBytes) + ";");
                }
              }
              emit("char* _smem = (char*)" + var + " + _base;");

              // Emit unrolled stores with constant byte offsets
              for (int i = 0; i < nElems; i++) {
                emit("*((" + cudaType + "*)(_smem + " + std::to_string(elemOffsets[i]) + ")) = " +
                     srcVar + "[" + std::to_string(i) + "];");
              }
              dedent();
              emit("}");
            } else {
              // Fallback: runtime address computation (original path)
              emit("// Swizzled store: blocked→nvmma_shared (" +
                   std::to_string(rows) + "x" + std::to_string(cols) + " " +
                   cudaType + ", swizzle=" + std::to_string(swizzleBytes) + "B)");
              emit("{");
              indent();
              emitPosDecomp();
              emit("#pragma unroll");
              emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
              indent();
              if (dim0Fast) {
                emit("int _s0 = _i % " + std::to_string(spt0) + ";");
                emit("int _rep0 = (_i / " + std::to_string(spt0) + ") % " + std::to_string(reps0) + ";");
                emit("int _s1 = (_i / " + std::to_string(spt0 * reps0) + ") % " + std::to_string(spt1) + ";");
                emit("int _rep1 = _i / " + std::to_string(spt0 * reps0 * spt1) + ";");
              } else {
                emit("int _rep0 = _i / " + std::to_string(strideRep0) + ";");
                emit("int _s0 = (_i / " + std::to_string(spt1 * reps1) + ") % " + std::to_string(spt0) + ";");
                emit("int _rep1 = (_i / " + std::to_string(spt1) + ") % " + std::to_string(reps1) + ";");
                emit("int _s1 = _i % " + std::to_string(spt1) + ";");
              }
              emit("int _log_row = _pos0 * " + std::to_string(spt0) + " + _s0 + _rep0 * " + std::to_string(totalT0 * spt0) + ";");
              emit("int _log_col = _pos1 * " + std::to_string(spt1) + " + _s1 + _rep1 * " + std::to_string(totalT1 * spt1) + ";");
              if (totalT0 * spt0 > rows && rows % spt0 != 0)
                emit("_log_row %= " + std::to_string(rows) + ";");
              if (totalT1 * spt1 > cols && cols % spt1 != 0)
                emit("_log_col %= " + std::to_string(cols) + ";");
              if (isTransposed) {
                emit("int _row = _log_col;");
                emit("int _col = _log_row;");
              } else {
                emit("int _row = _log_row;");
                emit("int _col = _log_col;");
              }
              if (phys_cols > elemsPerSwizzlingRow) {
                emit("int _tileK = _col / " + std::to_string(elemsPerSwizzlingRow) + ";");
                emit("int _colInTile = _col % " + std::to_string(elemsPerSwizzlingRow) + ";");
                if (phys_maxPhase > 1) {
                  emit("int _phase = (_row / " + std::to_string(phys_perPhase) + ") % " + std::to_string(phys_maxPhase) + ";");
                  emit("int _swizzled_col = _colInTile ^ (_phase * " + std::to_string(vec) + ");");
                  emit(var + "[_tileK * " + std::to_string(phys_rows * elemsPerSwizzlingRow) +
                       " + _row * " + std::to_string(elemsPerSwizzlingRow) + " + _swizzled_col] = " + srcVar + "[_i];");
                } else {
                  emit(var + "[_tileK * " + std::to_string(phys_rows * elemsPerSwizzlingRow) +
                       " + _row * " + std::to_string(elemsPerSwizzlingRow) + " + _colInTile] = " + srcVar + "[_i];");
                }
              } else {
                if (phys_maxPhase > 1) {
                  emit("int _phase = (_row / " + std::to_string(phys_perPhase) + ") % " + std::to_string(phys_maxPhase) + ";");
                  emit("int _swizzled_col = _col ^ (_phase * " + std::to_string(vec) + ");");
                  emit(var + "[_row * " + std::to_string(phys_cols) + " + _swizzled_col] = " + srcVar + "[_i];");
                } else {
                  emit(var + "[_row * " + std::to_string(phys_cols) + " + _col] = " + srcVar + "[_i];");
                }
              }
              dedent();
              emit("}");
              dedent();
              emit("}");
            }
          }
          // No __syncthreads() here: the subsequent fence_async_shared handler
          // emits fence.proxy.async + __syncthreads which provides the necessary
          // barrier for both regular stores (B tile) and async copies (A tile).
          // This matches the PTX/LLVM backend which has no bar.sync between
          // B stores and fence.proxy.async.
        } else if (nvmmaShared) {
          // Blocked src (any rank, e.g. 5D 1x1x1xBMxBN) into a swizzled
          // nvmma_shared dst that the 2D fast path above didn't cover. A linear
          // store would not match the swizzled TMA descriptor, so use the
          // dtype-agnostic layout-aware store.
          emitLayoutAwareSharedStore(src, memDescType, var);
          blockSync();
        } else {
          // Blocked src → plain / swizzled_shared (non-nvmma) shared store.
          //
          // The destination must be addressed by the element's LOGICAL (row,col)
          // tensor coordinate flattened row-major (flat = row*cols + col), NOT by
          // a naive tid*nElems+i layout: a blocked layout distributes rows across
          // lane/warp, so tid*nElems+i does not equal row*cols+col. The reader
          // (emitLocalLoad generic path) and cp.async (emitCpAsyncToShared, plain
          // case) both address swizzled_shared at the plain LinearLayout offset
          // row*cols+col (the swizzle XOR is applied by neither side for
          // non-nvmma shared), so the store must match that exact mapping or the
          // operands are scrambled — the non-pipelined block_m=16 HMMA-v2 matmul
          // (local_alloc instead of cp.async) read garbage otherwise.
          auto srcLL = ttg::toLinearLayout(srcRtt);
          auto kReg = mlir::StringAttr::get(srcRtt.getContext(), "register");
          auto kLane = mlir::StringAttr::get(srcRtt.getContext(), "lane");
          auto kWarp = mlir::StringAttr::get(srcRtt.getContext(), "warp");
          auto kBlock = mlir::StringAttr::get(srcRtt.getContext(), "block");
          int rank = srcRtt.getRank();
          auto sstShape = memDescType.getShape();
          // Honor the dst shared encoding's `order` so this store matches the
          // order-aware load (emitLocalLoad) and cp.async store. Identical to
          // row-major for the common order=[rank-1,...,0].
          llvm::SmallVector<int64_t> strides(rank, 1);
          if (auto swizEnc = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
                  memDescType.getEncoding())) {
            auto order = swizEnc.getOrder();
            int64_t s = 1;
            for (int oi = 0; oi < rank; oi++) {
              int d = order[oi];
              strides[d] = s;
              s *= sstShape[d];
            }
          } else {
            strides[rank - 1] = 1;
            for (int d = rank - 2; d >= 0; d--)
              strides[d] = strides[d + 1] * sstShape[d + 1];
          }
          auto regBases = srcLL.getBases().find(kReg);
          auto laneBases = srcLL.getBases().find(kLane);
          auto warpBases = srcLL.getBases().find(kWarp);
          int nLaneBits = (laneBases != srcLL.getBases().end()) ? laneBases->second.size() : 0;
          int nWarpBits = (warpBases != srcLL.getBases().end()) ? warpBases->second.size() : 0;
          (void)regBases;
          std::string baseExpr = "0";
          for (int lb = 0; lb < nLaneBits; lb++) {
            auto &coords = laneBases->second[lb];
            int flatDelta = 0;
            for (int d = 0; d < rank; d++) flatDelta += coords[d] * strides[d];
            if (flatDelta != 0)
              baseExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(flatDelta);
          }
          for (int wb = 0; wb < nWarpBits; wb++) {
            auto &coords = warpBases->second[wb];
            int flatDelta = 0;
            for (int d = 0; d < rank; d++) flatDelta += coords[d] * strides[d];
            if (flatDelta != 0)
              baseExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(flatDelta);
          }
          std::string offsets = "const int _sst_offsets[] = {";
          for (int i = 0; i < nElems; i++) {
            if (i > 0) offsets += ", ";
            auto coords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
            int flatOff = 0;
            for (size_t d = 0; d < coords.size(); d++)
              flatOff += coords[d].second * strides[d];
            offsets += std::to_string(flatOff);
          }
          offsets += "};";
          emit("// Store to shared memory (blocked LinearLayout offsets)");
          emit("{");
          indent();
          emit("int _sst_base = " + baseExpr + ";");
          emit(offsets);
          emit("#pragma unroll");
          emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
          emit("    " + var + "[_sst_base + _sst_offsets[_i]] = " + srcVar + "[_i];");
          dedent();
          emit("}");
          blockSync();
        }
      } else if (nvmmaShared) {
        // Catch-all src layout (e.g. an N-D reshape result) into a swizzled
        // nvmma_shared buffer — typically the persistent split_k fp32 TMA
        // scratch, whose source is a 5D 1x1x1xBMxBN blocked tensor. A linear
        // store would not match the swizzled TMA descriptor, so use the
        // dtype-agnostic layout-aware store: invertAndCompose maps the src
        // distributed layout onto the dst swizzle for any element size and any
        // rank, unlike the fp16-only stmatrix and the 2D-only blocked paths.
        emitLayoutAwareSharedStore(src, memDescType, var);
        blockSync();
      } else {
        // Distributed tensor src whose layout is neither Blocked nor NvidiaMma
        // (e.g. a #linear layout produced by tt.trans of an MMA result, or a
        // sliced layout) → plain / swizzled non-nvmma shared. The old naive
        // tid*nElems+i store is only correct when the per-thread register order
        // happens to equal smem row-major; for a #linear/MMA source it does NOT
        // (the registers carry (row,col) per the source LinearLayout, not
        // contiguously), so it scrambled the operand. This bit dq in the fused
        // FA3 backward: dq = dot(trans(dsT), k) lowers to local_alloc(#linear)
        // → local_load as the WGMMA A operand; the linear-fallback store placed
        // a thread's regs at tid*64+i while the reader expected the logical
        // (row,col) mapping, so the A operand was garbage and dq was wrong.
        // emitLayoutAwareSharedStore maps each register to its logical (row,col)
        // via the source LinearLayout and the dst order, exactly matching the
        // plain reader (emitLocalLoad generic path).
        emitLayoutAwareSharedStore(src, memDescType, var);
        blockSync();
      }
    }
  }
}

void CUDACodeGen::emitLocalStore(ttg::LocalStoreOp op) {
  auto val = op.getSrc();
  auto dst = op.getDst();
  auto valVar = getVar(val);
  auto dstVar = getVar(dst);

  if (!isa<RankedTensorType>(val.getType())) {
    emit(dstVar + "[tid] = " + valVar + ";");
    return;
  }

  // Layout-aware store: the register→shared offset map must respect both the
  // source distributed layout (e.g. MMA accumulator) and the destination
  // shared swizzle. emitLayoutAwareSharedStore mirrors MemoryOpToLLVM.cpp.
  auto memDescType = cast<ttg::MemDescType>(dst.getType());
  // Fast path: an MMA-accumulator register tensor stored into a 128B-swizzled
  // NVMMAShared buffer goes via stmatrix.x4 (same as the local_alloc-with-init
  // epilogue path), replacing the scalar per-element swizzled store (e.g. 32
  // st.shared.b16/thread for a [64,64] bf16 tile → 4 stmatrix). The PTX backend
  // lowers ttg.local_store this way too; emit_cuda previously only did it for
  // local_alloc, leaving FA-backward's dsT round-trip scalar.
  if (!emitStMatrixSharedStore(val, memDescType, "(char*)" + dstVar) &&
      !emitStMatrixTransStore(val, memDescType, "(char*)" + dstVar))
    emitLayoutAwareSharedStore(val, memDescType, dstVar);
  blockSync();
}

// stmatrix.x4 store of an MMA-accumulator src into a 128B-swizzled NVMMAShared
// dst. Mirrors the epilogue path in emitLocalAlloc (kept duplicated rather than
// refactored to avoid disturbing that proven GEMM path). Returns false (emitting
// nothing) when the layouts are not the supported MMA→128B-swizzle 16-bit case.
bool CUDACodeGen::emitStMatrixSharedStore(Value val,
                                          ttg::MemDescType memDescType,
                                          const std::string &smemBaseExpr) {
  auto srcRtt = cast<RankedTensorType>(val.getType());
  auto srcEnc = srcRtt.getEncoding();
  auto mmaEnc = dyn_cast_or_null<ttg::NvidiaMmaEncodingAttr>(srcEnc);
  if (!mmaEnc)
    return false;
  auto nvmmaShared =
      dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(memDescType.getEncoding());
  if (!nvmmaShared)
    return false;
  int elemBytes = getTypeSizeInBytes(srcRtt.getElementType());
  if (elemBytes != 2)
    return false; // stmatrix.b16 only; fp8/fp32 use scalar path
  if (nvmmaShared.getSwizzlingByteWidth() != 128)
    return false; // _base swizzle formula below assumes 128B
  if (nvmmaShared.getTransposed())
    return false; // transposed shared needs stmatrix.trans, not handled here

  auto shape = srcRtt.getShape();
  if (shape.size() != 2)
    return false;
  auto shapePerCTAStm = ttg::getShapePerCTA(mmaEnc, shape);
  auto wpcChk = mmaEnc.getWarpsPerCTA();
  bool nSplitWg = wpcChk.size() > 1 && wpcChk[1] > 1;
  int numWgN = nSplitWg ? (int)wpcChk[1] : 1;
  bool nSplitOk =
      !nSplitWg ||
      (shapePerCTAStm.size() > 1 &&
       ((int)shapePerCTAStm[1] / numWgN) % 64 == 0 && wpcChk[0] <= 4);
  if (!nSplitOk)
    return false;

  auto srcVar = getVar(val);
  int BM = shapePerCTAStm[0];
  int BN = shapePerCTAStm[1];
  if (BN % 64 != 0 || BM % 64 != 0)
    return false;
  auto wpc = mmaEnc.getWarpsPerCTA();
  int numWgM = std::max<int>(1, (int)wpc[0] / 4);
  int BNwg = BN / numWgN;
  int m64PerWg = std::max(1, (BM / numWgM) / 64);
  int nBlocks = std::max(1, BNwg / 64);
  int nblockBytes = BM * 64 * elemBytes;
  int stripeBytes = 64 * 64 * elemBytes;
  int packedPerMb = BNwg / 4;
  int nPacked = m64PerWg * packedPerMb;
  std::string wgM = numWgM > 1 ? "_wg_m" : "0";
  std::string wgN = numWgN > 1 ? "_wg_m" : "0";

  emit("// register→shared store via stmatrix (MMA acc → 128B-swizzle NVMMA)");
  emit("{");
  indent();
  emit("uint32_t _base = ((tid << 7) & 0x780) | ((tid << 4) & 0x70);");
  emit("_base = (_base ^ (tid & 0x10)) | ((tid << 6) & 0x1800);");
  emit("char* _smem_base = " + smemBaseExpr + ";");
  emit("uint32_t _packed[" + std::to_string(nPacked) + "];");
  emit("#pragma unroll");
  emit("for (int _i = 0; _i < " + std::to_string(nPacked) + "; _i++) {");
  indent();
  emit("uint32_t _lo = *(uint16_t*)&" + srcVar + "[2*_i];");
  emit("uint32_t _hi = *(uint16_t*)&" + srcVar + "[2*_i+1];");
  emit("_packed[_i] = _lo | (_hi << 16);");
  dedent();
  emit("}");
  for (int mb = 0; mb < m64PerWg; mb++) {
    for (int nGrp = 0; nGrp < 4; nGrp++) {
      for (int blk = 0; blk < nBlocks; blk++) {
        int localP = blk * 16 + nGrp * 4;
        int packedStart = mb * packedPerMb + localP;
        int xorVal = nGrp * 32;
        std::string stripeExpr = "(" + std::to_string(mb) + " * " +
                                 std::to_string(numWgM) + " + " + wgM + ")";
        std::string smemOff = "(" + wgN + " * " + std::to_string(nBlocks) +
                              " + " + std::to_string(blk) + ") * " +
                              std::to_string(nblockBytes) + " + " + stripeExpr +
                              " * " + std::to_string(stripeBytes);
        std::string addrExpr =
            xorVal ? ("_base ^ " + std::to_string(xorVal)) : "_base";
        emit("asm volatile(\"stmatrix.sync.aligned.m8n8.x4.shared.b16 "
             "[%0], {%1,%2,%3,%4};\"");
        emit("    :: \"r\"((unsigned)__cvta_generic_to_shared(_smem_base + (" +
             addrExpr + ") + (" + smemOff + "))),");
        emit("       \"r\"(_packed[" + std::to_string(packedStart) +
             "]), \"r\"(_packed[" + std::to_string(packedStart + 1) + "]),");
        emit("       \"r\"(_packed[" + std::to_string(packedStart + 2) +
             "]), \"r\"(_packed[" + std::to_string(packedStart + 3) + "]));");
      }
    }
  }
  dedent();
  emit("}");
  return true;
}

// Register→shared store via stmatrix.trans into a TRANSPOSED 128B-swizzled
// NVMMAShared dst (FA-backward dS^T / dq-handoff round-trip). CUDA-emission port
// of the PTX backend's LinearLayout-derived lowering (LLVM::NVIDIA::
// lowerLdStMatrix in third_party/nvidia/.../Utility.cpp, transpose=true/store).
// All addressing comes from cvt = regLayout.invertAndCompose(memLayout), so the
// bytes written match exactly what the dQ wgmma B-operand reads. Native non-pow2
// N (omni n80) is handled by padding both layouts to pow2 and emitting stmatrix
// instructions ONLY for tiles whose source registers all lie inside the real
// (unpadded) shape; phantom-padding tiles are skipped. Returns false (no output)
// for any layout it cannot prove stmatrix.trans-lowerable → scalar fallback.
bool CUDACodeGen::emitStMatrixTransStore(Value val,
                                         ttg::MemDescType memDescType,
                                         const std::string &smemBaseExpr) {
  auto srcRtt = cast<RankedTensorType>(val.getType());
  auto mmaEnc =
      dyn_cast_or_null<ttg::NvidiaMmaEncodingAttr>(srcRtt.getEncoding());
  if (!mmaEnc)
    return false;
  auto nvmmaShared =
      dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(memDescType.getEncoding());
  if (!nvmmaShared || !nvmmaShared.getTransposed())
    return false; // only the transposed case; non-trans → emitStMatrixSharedStore
  if (getTypeSizeInBytes(srcRtt.getElementType()) != 2)
    return false; // stmatrix.b16 only
  if (nvmmaShared.getSwizzlingByteWidth() != 128)
    return false;
  auto srcShape = srcRtt.getShape();
  auto memShape = memDescType.getShape();
  if (srcShape.size() != 2 || memShape.size() != 2)
    return false;
  // Under transposed=true the CONTIGUOUS/swizzle axis is dim0 (= swizzle width
  // / elemBytes); the strided axis is dim1 (its stride = dim0 size). Only dim1
  // may be non-pow2: padding it leaves every real offset unchanged because the
  // colgroup stride = dim0 size stays pow2. A non-pow2 dim0 (contiguous) would
  // need the scalar path's colstride rescale, so bail.
  if (!llvm::isPowerOf2_64(memShape[0]))
    return false;

  auto ctx = srcRtt.getContext();
  auto kReg = mlir::StringAttr::get(ctx, "register");
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  auto kWarp = mlir::StringAttr::get(ctx, "warp");
  auto kBlock = mlir::StringAttr::get(ctx, "block");
  auto kOffset = mlir::StringAttr::get(ctx, "offset");
  auto kAddr = mlir::StringAttr::get(ctx, "addr");
  auto kIdx = mlir::StringAttr::get(ctx, "idx");
  const int bitwidth = 16;
  const int byteScale = bitwidth / 8; // 2

  // ---- pad reg + mem layouts to pow2 (no-op when already pow2) -------------
  SmallVector<int64_t> padRegShape(srcShape.begin(), srcShape.end());
  int npReg = -1;
  for (int d = 0; d < (int)padRegShape.size(); d++)
    if (!llvm::isPowerOf2_64(padRegShape[d])) {
      int64_t p = 1;
      while (p < padRegShape[d])
        p *= 2;
      padRegShape[d] = p;
      npReg = d;
    }
  SmallVector<unsigned> instr(mmaEnc.getInstrShape().begin(),
                              mmaEnc.getInstrShape().end());
  if (npReg >= 0 && instr.size() >= 2)
    instr[1] = (unsigned)padRegShape[npReg];
  auto padMma = ttg::NvidiaMmaEncodingAttr::get(
      ctx, mmaEnc.getVersionMajor(), mmaEnc.getVersionMinor(),
      mmaEnc.getWarpsPerCTA(), mmaEnc.getCGALayout(), instr);
  auto padRegTy =
      RankedTensorType::get(padRegShape, srcRtt.getElementType(), padMma);

  SmallVector<int64_t> padMemShape(memShape.begin(), memShape.end());
  for (int d = 0; d < (int)padMemShape.size(); d++)
    if (!llvm::isPowerOf2_64(padMemShape[d])) {
      int64_t p = 1;
      while (p < padMemShape[d])
        p *= 2;
      padMemShape[d] = p;
    }
  auto realAlloc = memDescType.getAllocShape();
  SmallVector<int64_t> padAlloc(realAlloc.begin(), realAlloc.end());
  for (int d = 0; d < (int)padAlloc.size(); d++)
    if (!llvm::isPowerOf2_64(padAlloc[d])) {
      int64_t p = 1;
      while (p < padAlloc[d])
        p *= 2;
      padAlloc[d] = p;
    }
  auto padMemTy = ttg::MemDescType::get(
      ctx, padMemShape, memDescType.getElementType(),
      memDescType.getEncoding(), memDescType.getMemorySpace(),
      memDescType.getMutableMemory(), padAlloc);

  auto regLayoutPad = ttg::toLinearLayout(padRegTy);
  auto memLayoutPad = ttg::toLinearLayout(padMemTy);
  auto cvt = regLayoutPad.invertAndCompose(memLayoutPad);
  // Drop the (trivial, single-CTA) block dim → (reg,lane,warp)->offset.
  if (!cvt.sublayoutIsZero({kBlock}, {kOffset}))
    return false;
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

  // ---- transpose-store tile algebra (mirrors lowerLdStMatrix) -------------
  int contigRegs = 32 / bitwidth; // 2
  auto fullTile = LinearLayout::identity1D(contigRegs, kReg, kAddr) *
                  LinearLayout::identity1D(4, kLane, kAddr) *
                  LinearLayout::identity1D(8, kLane, kOffset) *
                  LinearLayout::identity1D(16 / bitwidth, kReg, kOffset);
  if (cvt.getInDimSize(kReg) < fullTile.getInDimSize(kReg))
    return false;
  std::vector<size_t> regBases, laneBases;
  for (const auto &basis : fullTile.invert().getBases().lookup(kOffset)) {
    if (basis.size() != 2)
      return false;
    if (basis[0] != 0)
      regBases.push_back(llvm::Log2_32(basis[0]));
    else
      laneBases.push_back(llvm::Log2_32(basis[1]));
  }
  for (int i = 0; i < cvt.getInDimSizeLog2(kReg); i++)
    if (!llvm::is_contained(regBases, (size_t)i))
      regBases.push_back(i);
  for (int i = 0; i < cvt.getInDimSizeLog2(kLane); i++)
    if (!llvm::is_contained(laneBases, (size_t)i))
      laneBases.push_back(i);
  if (laneBases != std::vector<size_t>({2, 3, 4, 0, 1}))
    return false;
  ColumnAction permReg(regBases, kReg, cvt.getInDimSizeLog2(kReg));
  ColumnAction permLanes(laneBases, kLane, cvt.getInDimSizeLog2(kLane));
  cvt = permReg.apply(cvt);
  cvt = permLanes.apply(cvt);
  // idLL tracks the register-index permutation applied to the value vector so
  // that final emission slot s → original padded-reg index = idLL.apply({s}).
  auto idLL = LinearLayout::identity1D(cvt.getInDimSize(kReg), kReg, kIdx);
  idLL = permReg.apply(idLL);

  auto tile = (LinearLayout::identity1D(8, kLane, kOffset) *
               LinearLayout::identity1D(16 / bitwidth, kReg, kOffset))
                  .transposeIns({kReg, kLane});
  auto maybePermDivide = regPermForDivide(cvt, tile, /*left=*/true);
  if (!maybePermDivide)
    return false;
  ColumnAction permDivide = maybePermDivide.value();
  cvt = permDivide.apply(cvt);
  idLL = permDivide.apply(idLL);
  auto maybeQuot = divideLeft(cvt, tile);
  if (!maybeQuot)
    return false;
  auto reps = zerosLike(tile) * maybeQuot.value();
  // revert lane/reg permutations
  reps = permLanes.inverse().apply(reps);
  reps = permReg.inverse().apply(reps);
  idLL = permReg.inverse().apply(idLL);

  // ---- vectorisation + per-lane address layout ----------------------------
  int regsPerCoreTile = fullTile.getInDimSize(kReg); // 2
  if (regsPerCoreTile * bitwidth != 32)
    return false;
  int vec = std::min<int>(128 / bitwidth, reps.getInDimSize(kReg)) /
            regsPerCoreTile;
  if (vec != 1 && vec != 2 && vec != 4)
    return false;
  auto fullTileVec = fullTile * LinearLayout::identity1D(vec, kReg, kAddr);
  fullTileVec = fullTileVec * LinearLayout::identity1D(1, kWarp, kAddr);
  auto addrToOffset = fullTileVec.invert().compose(reps);
  LinearLayout addrLayout(
      {{kLane, addrToOffset.getBases().lookup(kAddr)},
       {kWarp, reps.getBases().lookup(kWarp)}},
      {{kOffset, reps.getOutDimSize(kOffset)}}, false);
  auto addStrides = actionAdditiveStrides(reps, addrLayout, /*maskSpan=*/0);
  int64_t nAdditive = addStrides.first;
  ColumnAction permStrides = addStrides.second;
  reps = permStrides.apply(reps);
  idLL = permStrides.apply(idLL);
  if (nAdditive <= 0)
    return false;

  int elemsPerVec = 32 / bitwidth;                 // 2 (b16 per i32)
  int elemsPerInstr = fullTileVec.getInDimSize(kReg); // = contigRegs*vec
  int nInputs = elemsPerInstr / elemsPerVec;          // matrices per instr (.xN)
  if (nInputs != 1 && nInputs != 2 && nInputs != 4)
    return false;

  // real split-layout register coords → real emitted-register index
  auto realTbl = getRegCoordTable(srcRtt);
  int nRealElems = getElemsPerThread(srcRtt);
  std::map<SmallVector<int>, int> realCoordToReg;
  for (int i = 0; i < nRealElems && i < (int)realTbl.regCoords.size(); i++)
    realCoordToReg[realTbl.regCoords[i]] = i;
  // padded slot s → real register index (-1 if phantom/out-of-real-shape)
  auto mapReal = [&](int slot) -> int {
    int idx = (int)idLL.apply({{kReg, slot}}).front().second;
    auto c = regLayoutPad.apply(
        {{kReg, idx}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<int> key;
    for (int d = 0; d < (int)c.size(); d++) {
      int v = (int)c[d].second;
      if (d < (int)srcShape.size() && v >= (int)srcShape[d])
        return -1; // phantom padding element
      key.push_back(v);
    }
    auto it = realCoordToReg.find(key);
    return it == realCoordToReg.end() ? -1 : it->second;
  };

  auto cty = getCUDAType(srcRtt.getElementType());

  // ---- per-lane base address CUDA expression ------------------------------
  std::string raExpr = "0";
  for (int b = 0; b < addrLayout.getInDimSizeLog2(kLane); b++) {
    int64_t off =
        addrLayout.apply({{kLane, 1 << b}, {kWarp, 0}}).front().second *
        byteScale;
    if (off)
      raExpr += " ^ (((lane_id >> " + std::to_string(b) + ") & 1) * " +
                std::to_string(off) + ")";
  }
  for (int b = 0; b < addrLayout.getInDimSizeLog2(kWarp); b++) {
    int64_t off =
        addrLayout.apply({{kLane, 0}, {kWarp, 1 << b}}).front().second *
        byteScale;
    if (off)
      raExpr += " ^ (((warp_id >> " + std::to_string(b) + ") & 1) * " +
                std::to_string(off) + ")";
  }

  // qualifier order matches the existing non-trans path: shape.num.trans
  std::string xN = "x" + std::to_string(nInputs) + ".trans";
  std::string regList, regSpec;
  for (int k = 0; k < nInputs; k++) {
    regList += (k ? ",%" : "%") + std::to_string(k + 1);
    regSpec += std::string(k ? ", " : "") + "\"r\"(_pk" + std::to_string(k) + ")";
  }

  emit("// register→shared store via stmatrix.trans (MMA acc → transposed "
       "128B-swizzle NVMMA)");
  emit("{");
  indent();
  emit("char* _smem_base = " + smemBaseExpr + ";");
  emit("uint32_t _ra = (uint32_t)(" + raExpr + ");");
  for (int i = 0; i < cvt.getInDimSize(kReg); i += nAdditive) {
    int64_t regIdxOff =
        reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}}).front().second *
        byteScale;
    for (int i2 = 0; i2 < (int)nAdditive; i2 += elemsPerInstr) {
      int64_t regIdxAdd =
          reps.apply({{kReg, i2}, {kLane, 0}, {kWarp, 0}}).front().second *
          byteScale;
      // gather the nInputs packed register pairs for this instruction
      std::vector<std::pair<int, int>> packs;
      bool anyReal = false, allReal = true;
      for (int j = 0; j < elemsPerInstr; j += elemsPerVec) {
        int r0 = mapReal(i + i2 + j + 0);
        int r1 = mapReal(i + i2 + j + 1);
        if (r0 < 0 || r1 < 0)
          allReal = false;
        else
          anyReal = true;
        packs.push_back({r0, r1});
      }
      if (!anyReal)
        continue; // entirely phantom tile (padding) → skip
      if (!allReal)
        return false; // mixed real/phantom — unexpected → scalar fallback
      emit("{");
      indent();
      for (int k = 0; k < nInputs; k++) {
        std::string e0 = getElemExpr(val, std::to_string(packs[k].first));
        std::string e1 = getElemExpr(val, std::to_string(packs[k].second));
        emit(cty + " _t0_" + std::to_string(k) + " = (" + e0 + "); " + cty +
             " _t1_" + std::to_string(k) + " = (" + e1 + ");");
        emit("uint32_t _pk" + std::to_string(k) + " = (uint32_t)*(uint16_t*)&_t0_" +
             std::to_string(k) + " | ((uint32_t)*(uint16_t*)&_t1_" +
             std::to_string(k) + " << 16);");
      }
      std::string addrExpr = "(_ra ^ " + std::to_string(regIdxOff) + ") + " +
                             std::to_string(regIdxAdd);
      emit("asm volatile(\"stmatrix.sync.aligned.m8n8." + xN +
           ".shared.b16 [%0], {" + regList + "};\"");
      emit("    :: \"r\"((unsigned)__cvta_generic_to_shared(_smem_base) + (" +
           addrExpr + ")), " + regSpec + ");");
      dedent();
      emit("}");
    }
  }
  dedent();
  emit("}");
  return true;
}

// Register→shared store that respects an arbitrary src distributed layout and
// dst swizzled shared layout, computed via LinearLayout invertAndCompose:
//   cvt = regLayout.invertAndCompose(sharedLayout)  maps (reg,lane,warp)→off.
// Offsets combine over GF(2), so contributions are XOR-accumulated. This works
// for any element size — unlike stmatrix.b16 which is restricted to 16-bit
// units and cannot express an fp8 (1-byte) shared layout.
void CUDACodeGen::emitLayoutAwareSharedStore(Value val,
                                             ttg::MemDescType memDescType,
                                             llvm::StringRef dstVar) {
  auto srcRtt = cast<RankedTensorType>(val.getType());
  int nElems = getElemsPerThread(val);
  auto ctx = srcRtt.getContext();
  auto kReg = mlir::StringAttr::get(ctx, "register");
  auto kLane = mlir::StringAttr::get(ctx, "lane");
  auto kWarp = mlir::StringAttr::get(ctx, "warp");
  auto kOffset = mlir::StringAttr::get(ctx, "offset");
  auto kBlock0 = mlir::StringAttr::get(ctx, "block");

  // Native non-pow2 (omni-style n80) register→shared store. The GF(2)
  // LinearLayout machinery aborts on any non-pow2 shape dim, so PAD every
  // non-pow2 dim up to the next pow2 on BOTH the register tensor and the shared
  // destination, run the normal invertAndCompose addressing on the padded
  // (pow2) layouts, and emit stores ONLY for the registers whose padded coord
  // lies inside the REAL shape (mapped back to the real split-layout register
  // array via getRegCoordTable). Padding only an OUTER smem dim leaves the
  // inner-contiguous swizzle atom (and therefore every real element offset)
  // unchanged, so the wgmma B-operand reads back exactly what we wrote.
  {
    auto srcShape = srcRtt.getShape();
    auto memShape = memDescType.getShape();
    bool regNonPow2 = llvm::any_of(
        srcShape, [](int64_t d) { return !llvm::isPowerOf2_64(d); });
    bool memNonPow2 = llvm::any_of(
        memShape, [](int64_t d) { return !llvm::isPowerOf2_64(d); });
    auto srcMma =
        dyn_cast<ttg::NvidiaMmaEncodingAttr>(srcRtt.getEncoding());
    if ((regNonPow2 || memNonPow2) && srcMma) {
      // --- padded register type (bump instrShape-N to the padded N dim) ---
      SmallVector<int64_t> padRegShape(srcShape.begin(), srcShape.end());
      int npReg = -1;
      for (int d = 0; d < (int)padRegShape.size(); d++) {
        if (!llvm::isPowerOf2_64(padRegShape[d])) {
          int64_t p = 1;
          while (p < padRegShape[d])
            p *= 2;
          padRegShape[d] = p;
          npReg = d;
        }
      }
      SmallVector<unsigned> instr(srcMma.getInstrShape().begin(),
                                  srcMma.getInstrShape().end());
      if (npReg >= 0 && instr.size() >= 2)
        instr[1] = (unsigned)padRegShape[npReg];
      auto padMma = ttg::NvidiaMmaEncodingAttr::get(
          ctx, srcMma.getVersionMajor(), srcMma.getVersionMinor(),
          srcMma.getWarpsPerCTA(), srcMma.getCGALayout(), instr);
      auto padRegTy = RankedTensorType::get(padRegShape,
                                            srcRtt.getElementType(), padMma);

      // --- padded shared destination (same encoding, padded shape+alloc) ---
      SmallVector<int64_t> padMemShape(memShape.begin(), memShape.end());
      for (int d = 0; d < (int)padMemShape.size(); d++)
        if (!llvm::isPowerOf2_64(padMemShape[d])) {
          int64_t p = 1;
          while (p < padMemShape[d])
            p *= 2;
          padMemShape[d] = p;
        }
      auto realAlloc = memDescType.getAllocShape();
      SmallVector<int64_t> padAlloc(realAlloc.begin(), realAlloc.end());
      for (int d = 0; d < (int)padAlloc.size(); d++)
        if (!llvm::isPowerOf2_64(padAlloc[d])) {
          int64_t p = 1;
          while (p < padAlloc[d])
            p *= 2;
          padAlloc[d] = p;
        }
      auto padMemTy = ttg::MemDescType::get(
          ctx, padMemShape, memDescType.getElementType(),
          memDescType.getEncoding(), memDescType.getMemorySpace(),
          memDescType.getMutableMemory(), padAlloc);

      auto regLayoutPad = ttg::toLinearLayout(padRegTy);
      auto sharedLayoutPad = ttg::toLinearLayout(padMemTy);
      auto cvt = regLayoutPad.invertAndCompose(sharedLayoutPad);
      cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
      auto applyOff = [&](int reg, int lane, int warp) -> int64_t {
        auto out = cvt.apply({{kReg, reg}, {kLane, lane}, {kWarp, warp}});
        return out.front().second;
      };
      int nLaneBits = cvt.getInDimSizeLog2(kLane);
      int nWarpBits = cvt.getInDimSizeLog2(kWarp);

      // real split-layout register coords → real register-array index
      auto realTbl = getRegCoordTable(srcRtt);
      std::map<SmallVector<int>, int> realCoordToReg;
      for (int i = 0; i < nElems && i < (int)realTbl.regCoords.size(); i++)
        realCoordToReg[realTbl.regCoords[i]] = i;

      emit("// register→shared store (non-pow2 n80, padded invertAndCompose)");
      emit("{");
      indent();
      std::string lwExpr = "0";
      for (int b = 0; b < nLaneBits; b++) {
        int64_t d = applyOff(0, 1 << b, 0);
        if (d != 0)
          lwExpr += " ^ (((lane_id >> " + std::to_string(b) + ") & 1) * " +
                    std::to_string(d) + ")";
      }
      for (int b = 0; b < nWarpBits; b++) {
        int64_t d = applyOff(0, 0, 1 << b);
        if (d != 0)
          lwExpr += " ^ (((warp_id >> " + std::to_string(b) + ") & 1) * " +
                    std::to_string(d) + ")";
      }
      emit("int _lw = " + lwExpr + ";");
      // colgroup-stride rescale (transposed multi-colgroup n80 store).
      // The padded invertAndCompose addresses the smem buffer using the PADDED
      // (pow2) alloc, so the colgroup stride = padAlloc[outer]*W. The REAL
      // buffer uses realAlloc[outer]*W. For a single-colgroup buffer every
      // within-colgroup address is < colstridePad so this is an exact no-op;
      // for a multi-colgroup TRANSPOSED store (where the OUTER dim is padded,
      // e.g. [80,128]→pad[128,128], 2 colgroups) the padded colgroup stride
      // 128*W overshoots the real 80*W → OOB. Because colstridePad is pow2 and
      // every within-colgroup offset is strictly < colstridePad in the padded
      // space, we recover (colgroup, within) by /,% on the full runtime address
      // _lw^off and rebuild with the real stride. (XOR is safe to split here:
      // padded within-colgroup addresses occupy the low bits below the pow2
      // colstridePad boundary, the colgroup index the high bits.)
      // The non-pow2 (padded) dim is ALWAYS the physical colgroup-scaling dim:
      // the swizzle-inner dim is a multiple of W (pow2) and is never padded, so
      // the only padded dim is the "rows" count that scales the colgroup stride
      // (rows*W). Use the dim where realAlloc != padAlloc.
      int64_t colstridePad = 0, colstrideReal = 0;
      if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(
              memDescType.getEncoding())) {
        int elemBytes =
            (int)(memDescType.getElementType().getIntOrFloatBitWidth() / 8);
        int W = elemBytes > 0 ? nvmma.getSwizzlingByteWidth() / elemBytes : 0;
        if (W > 0 && realAlloc.size() == padAlloc.size()) {
          for (int d = 0; d < (int)padAlloc.size(); d++)
            if (realAlloc[d] != padAlloc[d]) {
              colstridePad = padAlloc[d] * (int64_t)W;
              colstrideReal = realAlloc[d] * (int64_t)W;
              break;
            }
        }
      }
      bool rescaleCol =
          colstridePad > 0 && colstrideReal > 0 && colstridePad != colstrideReal;
      int colShiftPad = 0;
      if (rescaleCol)
        colShiftPad = (int)llvm::Log2_64((uint64_t)colstridePad);
      int nPad = getElemsPerThread(padRegTy);
      for (int i = 0; i < nPad; i++) {
        auto c = regLayoutPad.apply(
            {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock0, 0}});
        SmallVector<int> key;
        bool inReal = true;
        for (int d = 0; d < (int)c.size(); d++) {
          int v = (int)c[d].second;
          key.push_back(v);
          if (d < (int)srcShape.size() && v >= (int)srcShape[d])
            inReal = false;
        }
        if (!inReal)
          continue;
        auto it = realCoordToReg.find(key);
        if (it == realCoordToReg.end())
          continue;
        int64_t off = applyOff(i, 0, 0);
        if (rescaleCol) {
          emit("{ int _f = _lw ^ " + std::to_string(off) + "; " +
               std::string(dstVar) + "[((_f >> " + std::to_string(colShiftPad) +
               ") * " + std::to_string(colstrideReal) + ") + (_f & " +
               std::to_string(colstridePad - 1) + ")] = " +
               getElemExpr(val, std::to_string(it->second)) + "; }");
        } else {
          emit(std::string(dstVar) + "[_lw ^ " + std::to_string(off) + "] = " +
               getElemExpr(val, std::to_string(it->second)) + ";");
        }
      }
      dedent();
      emit("}");
      return;
    }
  }

  auto regLayout = ttg::toLinearLayout(srcRtt);

  // SwizzledShared destinations: this emitter's convention is PLAIN
  // order-derived addressing that IGNORES the XOR swizzle — the read side
  // (emitLocalLoad generic path, cp.async store side, emitDot smem alias) all
  // address swizzled_shared buffers with plain strides. Using the true
  // swizzled linear layout here would scramble data for any plain reader
  // (e.g. test_split_subview round-trip). Strides come from the per-CTA
  // PARENT alloc shape so subslice views store at parent-strided offsets
  // (the subview base var already carries the subslice origin offset).
  if (auto swizEnc = dyn_cast<ttg::SwizzledSharedEncodingAttr>(
          memDescType.getEncoding())) {
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    int rank = srcRtt.getRank();
    auto allocShape = memDescType.getAllocShape().take_back(rank);
    auto allocPerCTA = ttg::getShapePerCTA(swizEnc, allocShape);
    auto order = swizEnc.getOrder();
    SmallVector<int64_t> strides(rank, 1);
    int64_t s = 1;
    for (int oi = 0; oi < rank; oi++) {
      strides[order[oi]] = s;
      s *= allocPerCTA[order[oi]];
    }
    const auto &bases = regLayout.getBases();
    const auto &laneBases = bases.find(kLane)->second;
    const auto &warpBases = bases.find(kWarp)->second;
    emit("// register→shared store (plain order-derived, matches plain readers)");
    emit("{");
    indent();
    std::string baseExpr = "0";
    for (size_t lb = 0; lb < laneBases.size(); lb++) {
      int64_t delta = 0;
      for (size_t d = 0; d < laneBases[lb].size(); d++)
        delta += laneBases[lb][d] * strides[d];
      if (delta != 0)
        baseExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " +
                    std::to_string(delta);
    }
    for (size_t wb = 0; wb < warpBases.size(); wb++) {
      int64_t delta = 0;
      for (size_t d = 0; d < warpBases[wb].size(); d++)
        delta += warpBases[wb][d] * strides[d];
      if (delta != 0)
        baseExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " +
                    std::to_string(delta);
    }
    emit("int _pb = " + baseExpr + ";");
    for (int i = 0; i < nElems; i++) {
      auto coords = regLayout.apply(
          {{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      int64_t regOff = 0;
      for (size_t d = 0; d < coords.size(); d++)
        regOff += coords[d].second * strides[d];
      emit(std::string(dstVar) + "[_pb + " + std::to_string(regOff) + "] = " +
           getElemExpr(val, std::to_string(i)) + ";");
    }
    dedent();
    emit("}");
    return;
  }

  auto sharedLayout = ttg::toLinearLayout(memDescType);
  auto cvt = regLayout.invertAndCompose(sharedLayout);
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

  auto applyOff = [&](int reg, int lane, int warp) -> int64_t {
    auto out = cvt.apply({{kReg, reg}, {kLane, lane}, {kWarp, warp}});
    return out.front().second;
  };

  int nLaneBits = cvt.getInDimSizeLog2(kLane);
  int nWarpBits = cvt.getInDimSizeLog2(kWarp);

  emit("// register→shared store (layout-aware via invertAndCompose)");
  emit("{");
  indent();
  std::string lwExpr = "0";
  bool lwEven = true; // track whether _lw is provably even (bit0 always 0)
  for (int b = 0; b < nLaneBits; b++) {
    int64_t d = applyOff(0, 1 << b, 0);
    if (d != 0) {
      lwExpr += " ^ (((lane_id >> " + std::to_string(b) + ") & 1) * " +
                std::to_string(d) + ")";
      if (d & 1)
        lwEven = false;
    }
  }
  for (int b = 0; b < nWarpBits; b++) {
    int64_t d = applyOff(0, 0, 1 << b);
    if (d != 0) {
      lwExpr += " ^ (((warp_id >> " + std::to_string(b) + ") & 1) * " +
                std::to_string(d) + ")";
      if (d & 1)
        lwEven = false;
    }
  }
  emit("int _lw = " + lwExpr + ";");
  // Element byte width of the destination shared buffer.
  int elemBits = memDescType.getElementType().getIntOrFloatBitWidth();
  // Vectorize byte stores into 16-bit stores when a consecutive register pair
  // maps to an aligned adjacent offset pair (off, off+1) with off even and _lw
  // even — then the two bytes form one aligned 2-byte unit, low byte = reg i.
  // Matches PTX's STS.U16 packing (halves the store-instruction count vs the
  // per-byte STS.U8 path) for the fp8 epilogue O→shared store.
  for (int i = 0; i < nElems;) {
    int64_t off = applyOff(i, 0, 0);
    if (elemBits == 8 && lwEven && i + 1 < nElems && (off % 2 == 0) &&
        applyOff(i + 1, 0, 0) == off + 1) {
      emit("*(unsigned short*)&" + std::string(dstVar) + "[_lw ^ " +
           std::to_string(off) + "] = (unsigned short)((unsigned char)(" +
           getElemExpr(val, std::to_string(i)) + ") | ((unsigned char)(" +
           getElemExpr(val, std::to_string(i + 1)) + ") << 8));");
      i += 2;
    } else {
      emit(std::string(dstVar) + "[_lw ^ " + std::to_string(off) + "] = " +
           getElemExpr(val, std::to_string(i)) + ";");
      i += 1;
    }
  }
  dedent();
  emit("}");
}

void CUDACodeGen::emitLocalLoad(ttg::LocalLoadOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  auto rtt = cast<RankedTensorType>(result.getType());
  int nElems = getElemsPerThread(rtt);
  auto cudaType = getCUDAType(rtt.getElementType());
  auto var = newVar("sld");
  valueToVar[result] = var;
  auto srcVar = getVar(src);

  // Determine the encoding of the destination (result)
  auto dstEnc = rtt.getEncoding();
  auto srcMemType = cast<ttg::MemDescType>(src.getType());
  auto srcEnc = srcMemType.getEncoding();
  bool isNvmmaShared = isa<ttg::NVMMASharedEncodingAttr>(srcEnc);
  bool isMMAResult = isa<ttg::NvidiaMmaEncodingAttr>(dstEnc);

  if (isNvmmaShared && isMMAResult) {
    // Loading from nvmma_shared into MMA registers: use ldmatrix-style addressing
    // For now, use simple per-element loads (TODO: ldmatrix)
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + var + "[_i] = " + srcVar + "[tid * " + std::to_string(nElems) + " + _i];");
  } else {
    // Use LinearLayout to compute per-register shared memory offsets
    // This must match the addressing used in emitAsyncCopyG2L / emitLocalAlloc
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");

    auto dstLL = ttg::toLinearLayout(rtt);
    auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");
    int rank = rtt.getRank();
    // Per-CTA register convention (see emitBlockedStoreToSmem): the smem tile
    // and register coords cover only this CTA's slice when CTASplitNum>1.
    auto shape = ttg::getShapePerCTA(dstEnc, rtt.getShape());
    // Compute strides for flattening. Must match the store side
    // (emitAsyncCopyG2L / emitLocalAlloc): honor the source shared encoding's
    // `order` so a K-contiguous (order=[0,1]) operand tile is read from the
    // same flat offsets it was written to. Identical to row-major for the
    // common order=[rank-1,...,0].
    // Strides come from the PARENT alloc shape (trailing `rank` dims): for a
    // memdesc_subslice view (e.g. 64x64 of a 128x128 tile) the physical row
    // stride is the parent's, not the view's. For full-tile views (pipeline
    // memdesc_index) alloc trailing dims == view shape, so this is a no-op.
    auto strideShape = ttg::getShapePerCTA(
        srcEnc, srcMemType.getAllocShape().take_back(rank));
    llvm::SmallVector<int64_t> strides(rank, 1);
    if (auto swizEnc = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(srcEnc)) {
      auto order = swizEnc.getOrder();
      int64_t s = 1;
      for (int oi = 0; oi < rank; oi++) {
        int d = order[oi];
        strides[d] = s;
        s *= strideShape[d];
      }
    } else {
      strides[rank - 1] = 1;
      for (int d = rank - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * strideShape[d + 1];
    }
    // Compute per-register addresses using LL bases
    auto regBases = dstLL.getBases().find(kReg);
    auto laneBases = dstLL.getBases().find(kLane);
    auto warpBases = dstLL.getBases().find(kWarp);
    int nRegBits = (regBases != dstLL.getBases().end()) ? regBases->second.size() : 0;
    int nLaneBits = (laneBases != dstLL.getBases().end()) ? laneBases->second.size() : 0;
    int nWarpBits = (warpBases != dstLL.getBases().end()) ? warpBases->second.size() : 0;

    // When the source is NVMMA-swizzled shared (e.g. a TMA-filled operand tile),
    // a logical (row, col) does NOT live at the plain row-major offset
    // row*physCols + col: the hardware stores it XOR-swizzled. The plain-offset
    // path below is only correct for unswizzled shared (cp.async / swizzle==0).
    // For HMMA-v2 small tiles whose dot operands come from TMA, we must invert
    // the same swizzle the store side applies (emitAsyncCopyG2L), otherwise the
    // operands are scrambled and the dot produces garbage. Mirror that formula.
    auto nvmmaSrc = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(srcEnc);
    int swizzleBytes = nvmmaSrc ? nvmmaSrc.getSwizzlingByteWidth() : 0;
    if (nvmmaSrc && swizzleBytes > 0 && rank == 2) {
      int bytesPerElem = rtt.getElementType().getIntOrFloatBitWidth() / 8;
      bool isTransposed = nvmmaSrc.getTransposed();
      llvm::SmallVector<int64_t> physShape(shape.begin(), shape.end());
      if (isTransposed)
        std::swap(physShape[0], physShape[1]);
      int physCols = (int)physShape[1];
      int physRows = (int)physShape[0];
      int vec = 16 / bytesPerElem;
      int perPhase = std::max(1, (int)(128 / (physCols * bytesPerElem)));
      int maxPhase = swizzleBytes / 16;
      int elemsPerSwizzlingRow = swizzleBytes / bytesPerElem;
      bool tiled = physCols > elemsPerSwizzlingRow;
      int rowDim = isTransposed ? 1 : 0;
      int colDim = isTransposed ? 0 : 1;

      // Runtime row/col base from lane + warp bits.
      std::string rowBase = "0", colBase = "0";
      for (int lb = 0; lb < nLaneBits; lb++) {
        int dr = laneBases->second[lb][rowDim];
        int dc = laneBases->second[lb][colDim];
        if (dr != 0)
          rowBase += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(dr);
        if (dc != 0)
          colBase += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(dc);
      }
      for (int wb = 0; wb < nWarpBits; wb++) {
        int dr = warpBases->second[wb][rowDim];
        int dc = warpBases->second[wb][colDim];
        if (dr != 0)
          rowBase += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(dr);
        if (dc != 0)
          colBase += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(dc);
      }
      // Per-register compile-time row/col offsets.
      std::string regRow = "const int _reg_row[] = {";
      std::string regCol = "const int _reg_col[] = {";
      for (int i = 0; i < nElems; i++) {
        if (i > 0) { regRow += ", "; regCol += ", "; }
        auto coords = dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        regRow += std::to_string(coords[rowDim].second);
        regCol += std::to_string(coords[colDim].second);
      }
      regRow += "};";
      regCol += "};";

      emit("{");
      indent();
      emit("int _rowBase = " + rowBase + ";");
      emit("int _colBase = " + colBase + ";");
      emit(regRow);
      emit(regCol);
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
      indent();
      emit("int _r = _rowBase + _reg_row[_i];");
      emit("int _c = _colBase + _reg_col[_i];");
      emit("int _ph = (_r / " + std::to_string(perPhase) + ") % " + std::to_string(maxPhase) + ";");
      if (tiled) {
        emit("int _tk = _c / " + std::to_string(elemsPerSwizzlingRow) + ";");
        emit("int _cit = _c % " + std::to_string(elemsPerSwizzlingRow) + ";");
        emit("int _sc = _cit ^ (_ph * " + std::to_string(vec) + ");");
        emit(var + "[_i] = " + srcVar + "[_tk * " +
             std::to_string(physRows * elemsPerSwizzlingRow) + " + _r * " +
             std::to_string(elemsPerSwizzlingRow) + " + _sc];");
      } else {
        emit("int _sc = _c ^ (_ph * " + std::to_string(vec) + ");");
        emit(var + "[_i] = " + srcVar + "[_r * " + std::to_string(physCols) + " + _sc];");
      }
      dedent();
      emit("}");
      dedent();
      emit("}");
      return;
    }

    // Swizzled nvmma_shared with rank != 2 (e.g. rank-3+ TMA descriptor tiles
    // where TMALowering picked swizzle_mode != 0): the per-dim stride formula
    // above cannot express the XOR swizzle, so use the shared linear layout
    // directly via invertAndCompose (mirrors emitLayoutAwareSharedStore).
    if (nvmmaSrc && swizzleBytes > 0) {
      auto kOffset = mlir::StringAttr::get(rtt.getContext(), "offset");
      auto sharedLL = ttg::toLinearLayout(srcMemType);
      auto cvt = dstLL.invertAndCompose(sharedLL);
      cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
      auto applyOff = [&](int reg, int lane, int warp) -> int64_t {
        auto out = cvt.apply({{kReg, reg}, {kLane, lane}, {kWarp, warp}});
        return out.front().second;
      };
      emit("// shared→register load (swizzle-aware via invertAndCompose)");
      emit("{");
      indent();
      std::string lwExpr = "0";
      for (int b = 0; b < cvt.getInDimSizeLog2(kLane); b++) {
        int64_t d = applyOff(0, 1 << b, 0);
        if (d != 0)
          lwExpr += " ^ (((lane_id >> " + std::to_string(b) + ") & 1) * " +
                    std::to_string(d) + ")";
      }
      for (int b = 0; b < cvt.getInDimSizeLog2(kWarp); b++) {
        int64_t d = applyOff(0, 0, 1 << b);
        if (d != 0)
          lwExpr += " ^ (((warp_id >> " + std::to_string(b) + ") & 1) * " +
                    std::to_string(d) + ")";
      }
      emit("int _lw = " + lwExpr + ";");
      for (int i = 0; i < nElems; i++) {
        emit(var + "[" + std::to_string(i) + "] = " + srcVar + "[_lw ^ " +
             std::to_string(applyOff(i, 0, 0)) + "];");
      }
      dedent();
      emit("}");
      return;
    }

    // Build base offset expression for lane and warp (runtime)
    emit("{");
    indent();
    // Compute the lane+warp contribution as a runtime expression
    std::string baseExpr = "0";
    for (int lb = 0; lb < nLaneBits; lb++) {
      auto &coords = laneBases->second[lb];
      int flatDelta = 0;
      for (int d = 0; d < rank; d++)
        flatDelta += coords[d] * strides[d];
      if (flatDelta != 0)
        baseExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(flatDelta);
    }
    for (int wb = 0; wb < nWarpBits; wb++) {
      auto &coords = warpBases->second[wb];
      int flatDelta = 0;
      for (int d = 0; d < rank; d++)
        flatDelta += coords[d] * strides[d];
      if (flatDelta != 0)
        baseExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(flatDelta);
    }
    emit("int _sld_base = " + baseExpr + ";");

    // For each register, compute the constant offset
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    // Build per-register offset as compile-time constant via switch
    // For small nElems, unroll manually
    if (nElems <= 32) {
      // Emit as array of constants
      std::string offsets = "const int _reg_offsets[] = {";
      for (int i = 0; i < nElems; i++) {
        if (i > 0) offsets += ", ";
        auto coords = dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        int flatOff = 0;
        for (size_t d = 0; d < coords.size(); d++)
          flatOff += coords[d].second * strides[d];
        offsets += std::to_string(flatOff);
      }
      offsets += "};";
      dedent();
      emit("}");
      dedent();
      emit("}");
      // Re-emit with the simple approach
      emit("{");
      indent();
      emit("int _sld_base = " + baseExpr + ";");
      emit(offsets);
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
      emit("    " + var + "[_i] = " + srcVar + "[_sld_base + _reg_offsets[_i]];");
    } else {
      // For larger nElems, compute per-register offset from bits
      std::string regExpr = "0";
      for (int rb = 0; rb < nRegBits; rb++) {
        auto &coords = regBases->second[rb];
        int flatDelta = 0;
        for (int d = 0; d < rank; d++)
          flatDelta += coords[d] * strides[d];
        if (flatDelta != 0)
          regExpr += " + ((_i >> " + std::to_string(rb) + ") & 1) * " + std::to_string(flatDelta);
      }
      emit(var + "[_i] = " + srcVar + "[_sld_base + " + regExpr + "];");
      dedent();
      emit("}");
    }
    dedent();
    emit("}");
  }
  blockSync();
}
void CUDACodeGen::emitLocalDealloc(ttg::LocalDeallocOp op) {
  emit("// local_dealloc");
  peakSharedMem = std::max(peakSharedMem, sharedMemOffset);
  // Free the buffer by popping the bump pointer down to *this* buffer's
  // offset, not to 0. Resetting to 0 would let a subsequently-allocated
  // buffer alias a still-live lower buffer (e.g. the resident Q tile in
  // flash-attention, which stays live across both attention stages while the
  // stage-1 K/V buffers are deallocated before stage-2 reallocates). Using the
  // minimum across a group of consecutive deallocs frees the whole contiguous
  // freed region regardless of dealloc order, while preserving any lower
  // buffer that is not being freed.
  liveSmemTop.erase(op.getSrc());
  auto it = valueToSmemOffset.find(op.getSrc());
  if (it != valueToSmemOffset.end())
    sharedMemOffset = std::min(sharedMemOffset, it->second);
}
// Lowest offset a new shared buffer may take: the top (offset+size) of the
// highest still-live buffer, or 0 if nothing is live. Used by emitLocalAlloc
// after retiring dead sourced tiles so the new buffer sits above everything
// still live (never aliasing a live tile, even with non-LIFO free order).
int CUDACodeGen::recomputeSmemFloor() {
  int floor = 0;
  for (auto &kv : liveSmemTop)
    floor = std::max(floor, kv.second);
  return floor;
}
void CUDACodeGen::emitConvertLayout(ttg::ConvertLayoutOp op) {
  auto result = op.getResult();
  auto src = op.getSrc();
  auto rtt = cast<RankedTensorType>(result.getType());
  auto srcRtt = cast<RankedTensorType>(src.getType());
  int nSrc = getElemsPerThread(src);
  int nDst = getElemsPerThread(rtt);
  auto srcEnc = srcRtt.getEncoding();
  auto dstEnc = rtt.getEncoding();
  Type elemType = rtt.getElementType();
  auto cudaType = getCUDAType(elemType);
  auto var = newVar("cvt");
  valueToVar[result] = var;
  auto srcVar = getVar(src);

  // Pointer tensors are represented as a *deferred* addptr (base + offset[])
  // and never materialized as a pointer array. A convert_layout on such a
  // tensor must reshuffle the underlying i32 OFFSET array (not a non-existent
  // pointer array), then re-defer the result with the same base. Every code
  // path below fills `var` as an indexable int array/pointer, so registering
  // the deferred result up-front covers all return paths.
  bool srcIsDeferredPtr = (bool)deferredAddPtr.count(src);
  std::string deferBase;
  if (srcIsDeferredPtr) {
    auto &di = deferredAddPtr[src];
    deferBase = di.first;
    srcVar = di.second; // int[nSrc] offset array
    elemType = mlir::IntegerType::get(rtt.getContext(), 32);
    cudaType = "int";
    deferredAddPtr[result] = {deferBase, var};
  }

  // Only use the manual MMA→blocked path for WGMMA v3; MMA v2 uses the
  // generic LL-based path because its column interleaving differs.
  bool isMMAv3ToBlocked = false;
  if (auto mmaEnc = dyn_cast<ttg::NvidiaMmaEncodingAttr>(srcEnc))
    isMMAv3ToBlocked = mmaEnc.getVersionMajor() >= 3 && isa<ttg::BlockedEncodingAttr>(dstEnc);
  if (isMMAv3ToBlocked && rtt.getRank() >= 2) {
    auto mmaEnc = cast<ttg::NvidiaMmaEncodingAttr>(srcEnc);
    // Per-CTA shape: with CTASplitNum>1 each CTA stages only its slice.
    auto cvtShapePerCTA = ttg::getShapePerCTA(rtt);
    int M = cvtShapePerCTA[0], N = cvtShapePerCTA[1];
    int MInstr = mmaEnc.getInstrShape()[0];
    auto wpc = mmaEnc.getWarpsPerCTA();
    int wpc0 = wpc[0], wpc1 = wpc.size() > 1 ? wpc[1] : 1;
    int NPerWarp = N / wpc1;
    emit("// convert_layout: #mma -> #blocked (" + std::to_string(M) + "x" + std::to_string(N) + ")");
    emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
    emit("{");
    indent();
    int64_t totalElems = M * N;
    int eb = getTypeSizeInBytes(elemType);
    // This scratch is a pure temporary: it is written by the MMA store and
    // fully read back into `var` registers before the block closes, so the
    // smem is dead afterwards. Save the bump pointer and restore it after the
    // readback so the next allocation reuses this space instead of stacking on
    // top of it (otherwise each epilogue convert_layout permanently grows peak
    // smem and blows the H800 232448-byte limit on persistent fp8 matmul).
    int savedSmemOffsetMmaCvt = sharedMemOffset;
    int smemOff = (sharedMemOffset + 127) & ~127;
    // If a full-tensor scratch would blow the H800 dynamic-smem cap, chunk the
    // roundtrip over row ranges: store rows [lo,hi) → sync → read back → sync,
    // repeated. Scratch is then only chunkRows*N*eb bytes.
    const int kSmemCap = 232448;
    int chunkRows = M;
    while (chunkRows > 8 && chunkRows % 2 == 0 &&
           smemOff + (int64_t)chunkRows * N * eb > kSmemCap)
      chunkRows /= 2;
    int nChunks = (M + chunkRows - 1) / chunkRows;
    sharedMemOffset = smemOff + (int64_t)chunkRows * N * eb;
    if (sharedMemOffset > peakSharedMem) peakSharedMem = sharedMemOffset;
    emit(cudaType + "* _cvt = (" + cudaType + "*)(shared_mem + " + std::to_string(smemOff) + ");");
    for (int ck = 0; ck < nChunks; ck++) {
    int ckLo = ck * chunkRows;
    int ckHi = std::min<int>(M, ckLo + chunkRows);
    std::string stGuard =
        nChunks > 1 ? ("if (_mma_row >= " + std::to_string(ckLo) +
                       " && _mma_row < " + std::to_string(ckHi) + ") ")
                    : "";
    std::string stRow =
        nChunks > 1 ? ("(_mma_row - " + std::to_string(ckLo) + ")")
                    : "_mma_row";
    // MMA store: pack pairs of f16 into uint32 for st.shared.b32
    {
      // One M-rep tile spans MInstr rows per warp x wpc0 warps; register reps
      // along M stride by rowsPerRep (LL: warp bits below register-rep bits).
      // regsPerMTile derives from nSrc so it stays correct for any wpc0/wpc1.
      int rowsPerRep = MInstr * wpc0;
      int nMTiles = std::max(1, M / rowsPerRep);
      int regsPerMTile = nSrc / nMTiles;
      emit("// MMA→shared via packed uint32 stores");
      emit("{");
      indent();
      // For 2-byte elements (f16/bf16) pack two into a uint32 (b32 store);
      // for 1-byte elements (fp8) pack two into a uint16 (b16 store). The
      // row/col coordinates below are element coordinates either way, so the
      // packed-unit row stride is always N/2.
      bool isByteElem = (eb == 1);
      // 4-byte elements (fp32 accumulator output, e.g. split_k partials) are
      // NOT packed: each element is stored directly as one b32 at its (row,col)
      // with full row stride N. The packed path below (used for f16/bf16/fp8)
      // would reinterpret two narrow elements as one wide unit and corrupt the
      // value — the readback reads cudaType `float` at row*N+col.
      bool is4Byte = (eb == 4);
      if (isByteElem)
        emit("uint16_t* _cvt16 = (uint16_t*)_cvt;");
      else if (!is4Byte)
        emit("uint32_t* _cvt32 = (uint32_t*)_cvt;");
      // Decompose warp_id into M and N dimensions. NvidiaMma warp
      // linearization is dim0(M)-fastest (see WGMMA.cpp loadA/loadB:
      // warpM = warp % wpt[0], warpN = (warp / wpt[0]) % wpt[1]).
      if (wpc1 > 1) {
        emit("int _warp_m = warp_id % " + std::to_string(wpc0) + ";");
        emit("int _warp_n = (warp_id / " + std::to_string(wpc0) + ") % " +
             std::to_string(wpc1) + ";");
      } else {
        emit("int _warp_m = warp_id;");
      }
      emit("#pragma unroll");
      emit("for (int _r = 0; _r < " + std::to_string(nSrc / 2) + "; _r++) {");
      indent();
      // Pack two adjacent elements into one packed unit (narrow types only).
      if (isByteElem) {
        // fp8 storage is already a raw byte; pack two bytes into a uint16
        emit("uint32_t _lo = *(uint8_t*)&" + srcVar + "[2*_r];");
        emit("uint32_t _hi = *(uint8_t*)&" + srcVar + "[2*_r+1];");
        emit("uint16_t _packed = (uint16_t)(_lo | (_hi << 8));");
      } else if (!is4Byte) {
        // 2-byte elements (f16 OR bf16): the register already holds the value
        // in the destination type, so take its raw 16 bits directly. Casting to
        // __half VALUE-converts a __nv_bfloat16 register to fp16 (e.g. 5.0 bf16
        // -> 0x4500 fp16 bits), and since the readback reinterprets those bits
        // as the original type, a bf16 output would decode 0x4500 as 2^11=2048.
        emit("uint32_t _lo = *(uint16_t*)&" + srcVar + "[2*_r];");
        emit("uint32_t _hi = *(uint16_t*)&" + srcVar + "[2*_r+1];");
        emit("uint32_t _packed = _lo | (_hi << 16);");
      }
      // Compute address: registers 2*_r and 2*_r+1 are at (row, col) and (row, col+1)
      // which are adjacent in row-major → one uint32 at col/2
      if (nMTiles > 1) {
        emit("int _r2 = 2 * _r;");
        emit("int _m_tile = _r2 / " + std::to_string(regsPerMTile) + ";");
        emit("int _r_local = _r2 % " + std::to_string(regsPerMTile) + ";");
        emit("int _mma_row = _m_tile * " + std::to_string(rowsPerRep) +
             " + _warp_m * " + std::to_string(MInstr) +
             " + (lane_id / 4) + ((_r_local >> 1) & 1) * 8;");
        if (wpc1 > 1) {
          emit("int _mma_col = _warp_n * " + std::to_string(NPerWarp) + " + (lane_id % 4) * 2 + ((_r_local >> 2)) * 8;");
        } else {
          emit("int _mma_col = (lane_id % 4) * 2 + ((_r_local >> 2)) * 8;");
        }
      } else {
        emit("int _r2 = 2 * _r;");
        emit("int _mma_row = _warp_m * " + std::to_string(MInstr) + " + (lane_id / 4) + ((_r2 >> 1) & 1) * 8;");
        if (wpc1 > 1) {
          emit("int _mma_col = _warp_n * " + std::to_string(NPerWarp) + " + (lane_id % 4) * 2 + ((_r2 >> 2)) * 8;");
        } else {
          emit("int _mma_col = (lane_id % 4) * 2 + ((_r2 >> 2)) * 8;");
        }
      }
      if (isByteElem)
        emit(stGuard + "_cvt16[" + stRow + " * " + std::to_string(N / 2) + " + _mma_col / 2] = _packed;");
      else if (is4Byte) {
        // Direct, unpacked: two adjacent columns (col, col+1), row stride N.
        emit(stGuard + "_cvt[" + stRow + " * " + std::to_string(N) + " + _mma_col] = " + srcVar + "[2*_r];");
        emit(stGuard + "_cvt[" + stRow + " * " + std::to_string(N) + " + _mma_col + 1] = " + srcVar + "[2*_r+1];");
      } else
        emit(stGuard + "_cvt32[" + stRow + " * " + std::to_string(N / 2) + " + _mma_col / 2] = _packed;");
      dedent();
      emit("}");
      dedent();
      emit("}");
    }
    blockSync();
    // Blocked load
    auto dstBlk = cast<ttg::BlockedEncodingAttr>(dstEnc);
    if (nChunks > 1)
      emitBlockedLoadFromSmem(var, "_cvt", dstBlk, rtt.getShape(), cudaType,
                              nDst, ckLo, ckHi);
    else
      emitBlockedLoadFromSmem(var, "_cvt", dstBlk, rtt.getShape(), cudaType, nDst);
    blockSync();
    } // chunk loop
    // Scratch consumed — restore the bump pointer so this space is reusable.
    sharedMemOffset = savedSmemOffsetMmaCvt;
    dedent();
    emit("}");
  } else if (llvm::any_of(srcRtt.getShape(),
                          [](int64_t d) { return !llvm::isPowerOf2_64(d); }) ||
             llvm::any_of(rtt.getShape(),
                          [](int64_t d) { return !llvm::isPowerOf2_64(d); })) {
    // Native non-pow2 WGMMA (omni-style n80): the GF(2) LinearLayout aborts on
    // a non-pow2 shape dim, so the toLinearLayout-based fast/shuffle/smem paths
    // below cannot run. Use the split-layout-aware register-coordinate table.
    // The relevant convert is mma-accumulator(n80) → dot-operand-A(K=80) in a
    // chained WGMMA (dV=P^T·dO, dK=dS^T·Q): the accumulator D-fragment and the
    // A-operand fragment place each logical element on the SAME (lane,warp), so
    // it is a pure per-thread register permutation (matched by coordinate).
    auto srcTbl = getRegCoordTable(srcRtt);
    auto dstTbl = getRegCoordTable(rtt);
    if (srcTbl.laneBases != dstTbl.laneBases ||
        srcTbl.warpBases != dstTbl.warpBases ||
        srcTbl.blockBases != dstTbl.blockBases) {
      std::string msg = "[emit_cuda] non-pow2 convert_layout requires matching "
                        "lane/warp placement (cross-thread non-pow2 convert "
                        "unsupported)";
      op->emitError(msg);
      if (!emitFailed) { emitFailed = true; emitErrorMsg = msg; }
      return;
    }
    std::map<llvm::SmallVector<int>, int> srcCoordToReg;
    for (int s = 0; s < nSrc && s < (int)srcTbl.regCoords.size(); s++)
      srcCoordToReg.insert({srcTbl.regCoords[s], s}); // keep first occurrence
    // Opt-in (TRITON_BWD_INPLACE_RS_PACK=1, default OFF): when this bf16/f16
    // convert feeds ONLY WGMMA A-operands (RS mode), emit the result directly
    // as packed uint32 words (2 elems per word, in the wgmma A-operand pairing
    // order aVar[2i],aVar[2i+1]) and register it in packedU32Convert so
    // emitWarpGroupDot aliases _a_packed onto it with NO copy. This halves the
    // A-operand staging registers (cvt[N] bf16 + _a_packed[N/2] u32  ->  one
    // u32[N/2]), relieving the v13 80-row consumer register cliff that makes
    // ptxas serialize the async WGMMAs (C7512). Default OFF => tut-01/02 and
    // every other kernel stay byte-identical (packedU32Convert never gains a
    // bf16 entry, so the wgmma alias branch below is never taken for them).
    static const bool inplaceRsPack =
        (getenv("TRITON_BWD_INPLACE_RS_PACK") != nullptr) &&
        (std::string(getenv("TRITON_BWD_INPLACE_RS_PACK")) == "1");
    bool packToU32 = inplaceRsPack && (elemType.isBF16() || elemType.isF16()) &&
                     nDst > 0 && (nDst % 2 == 0) &&
                     (int)dstTbl.regCoords.size() >= nDst;
    if (packToU32)
      for (Operation *user : result.getUsers()) {
        auto dotOp = dyn_cast<ttng::WarpGroupDotOp>(user);
        if (!dotOp || dotOp.getA() != result) { packToU32 = false; break; }
      }
    if (packToU32) {
      emit("// convert_layout (non-pow2 register-local, PACKED u32 for wgmma A "
           "RS, opt-in TRITON_BWD_INPLACE_RS_PACK)");
      emit("uint32_t " + var + "[" + std::to_string(nDst / 2) + "];");
      for (int i = 0; i < nDst / 2; i++) {
        auto regOf = [&](int d) {
          auto it = srcCoordToReg.find(dstTbl.regCoords[d]);
          return (it != srcCoordToReg.end()) ? it->second : 0;
        };
        emit("{");
        indent();
        emit(cudaType + " _e0 = " + srcVar + "[" + std::to_string(regOf(2 * i)) + "];");
        emit(cudaType + " _e1 = " + srcVar + "[" + std::to_string(regOf(2 * i + 1)) + "];");
        emit(var + "[" + std::to_string(i) +
             "] = ((uint32_t)*(uint16_t*)&_e0) | ((uint32_t)*(uint16_t*)&_e1) << 16;");
        dedent();
        emit("}");
      }
      packedU32Convert[result] = nDst / 2;
    } else {
      emit("// convert_layout (non-pow2 register-local, no smem)");
      emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
      for (int d = 0; d < nDst && d < (int)dstTbl.regCoords.size(); d++) {
        auto it = srcCoordToReg.find(dstTbl.regCoords[d]);
        int s = (it != srcCoordToReg.end()) ? it->second : 0;
        emit(var + "[" + std::to_string(d) + "] = " + srcVar + "[" +
             std::to_string(s) + "];");
      }
    }
  } else {
    // Fast path: register-local conversion (no shared memory). If src and dst
    // place every logical element on the SAME (lane,warp) — i.e. identical
    // lane/warp basis deltas over the flat coordinate space — then no
    // cross-thread movement is needed and the convert is a pure per-thread
    // register permutation (plus element narrowing on assignment). This is the
    // MMA-accumulator → MMA-operand-A case in chained WGMMA (e.g. flash
    // attention P=softmax(QK) feeding P@V), which the generic smem roundtrip
    // would otherwise lower as ~128 STS + 128 LDS per loop iteration.
    bool emittedRegLocal = false;
    {
      auto srcLL = ttg::toLinearLayout(srcRtt);
      auto dstLL = ttg::toLinearLayout(rtt);
      auto kReg = mlir::StringAttr::get(srcRtt.getContext(), "register");
      auto kLane = mlir::StringAttr::get(srcRtt.getContext(), "lane");
      auto kWarp = mlir::StringAttr::get(srcRtt.getContext(), "warp");
      auto kBlock = mlir::StringAttr::get(srcRtt.getContext(), "block");
      // src and dst share the same tensor shape → identical flat strides.
      auto shp = rtt.getShape();
      SmallVector<int64_t> strides(shp.size());
      strides.back() = 1;
      for (int d = (int)shp.size() - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * shp[d + 1];
      auto deltaVec = [&](const auto &bases) {
        SmallVector<int64_t> out;
        for (auto &b : bases) {
          int64_t delta = 0;
          for (size_t d = 0; d < b.size(); d++) delta += b[d] * strides[d];
          out.push_back(delta);
        }
        return out;
      };
      auto regBaseOf = [&](auto &ll, int i) {
        auto coords = ll.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        int64_t base = 0;
        for (size_t d = 0; d < coords.size(); d++)
          base += coords[d].second * strides[d];
        return base;
      };
      const auto &srcB = srcLL.getBases();
      const auto &dstB = dstLL.getBases();
      auto srcLane = deltaVec(srcB.find(kLane)->second);
      auto srcWarp = deltaVec(srcB.find(kWarp)->second);
      auto dstLane = deltaVec(dstB.find(kLane)->second);
      auto dstWarp = deltaVec(dstB.find(kWarp)->second);
      if (srcLane == dstLane && srcWarp == dstWarp) {
        // Pure register permutation: for each dst reg, find a src reg holding
        // the same logical element (matching flat regBase).
        std::map<int64_t, int> srcByBase;
        for (int s = 0; s < nSrc; s++)
          srcByBase[regBaseOf(srcLL, s)] = s;
        SmallVector<int> dstToSrc(nDst, -1);
        bool allFound = true;
        for (int d = 0; d < nDst; d++) {
          auto it = srcByBase.find(regBaseOf(dstLL, d));
          if (it == srcByBase.end()) { allFound = false; break; }
          dstToSrc[d] = it->second;
        }
        if (allFound) {
          emit("// convert_layout (register-local, no smem)");
          emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
          for (int d = 0; d < nDst; d++)
            emit(var + "[" + std::to_string(d) + "] = " + srcVar + "[" +
                 std::to_string(dstToSrc[d]) + "];");
          emittedRegLocal = true;
        }
      }
    }
    if (emittedRegLocal) return;
    // Barrier-free warp-shuffle convert (the fp8 P->dot_op case in chained
    // WGMMA). The generic smem path below needs TWO __syncthreads per call.
    // When the convert is warp-local — every dst element lives in the SAME
    // warp as in src — data can move with __shfl_sync and ZERO barriers, like
    // the PTX backend does. Strictly guarded: 1-byte elements only, warp/block
    // mapping identical, source-lane affine in lane bits, and the set of lane
    // bits selecting the source register small (<=2). On ANY guard failure we
    // fall through to the correct smem path below.
    {
      int ebShfl = getTypeSizeInBytes(elemType);
      auto srcLL = ttg::toLinearLayout(srcRtt);
      auto dstLL = ttg::toLinearLayout(rtt);
      auto kReg = mlir::StringAttr::get(srcRtt.getContext(), "register");
      auto kLane = mlir::StringAttr::get(srcRtt.getContext(), "lane");
      auto kWarp = mlir::StringAttr::get(srcRtt.getContext(), "warp");
      auto kBlock = mlir::StringAttr::get(srcRtt.getContext(), "block");
      auto shp = rtt.getShape();
      SmallVector<int64_t> strides(shp.size());
      strides.back() = 1;
      for (int d = (int)shp.size() - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * shp[d + 1];
      auto deltaVec = [&](const auto &bases) {
        SmallVector<int64_t> out;
        for (auto &b : bases) {
          int64_t delta = 0;
          for (size_t d = 0; d < b.size(); d++) delta += b[d] * strides[d];
          out.push_back(delta);
        }
        return out;
      };
      auto offsetOf = [&](auto &ll, int reg, int lane) {
        auto coords =
            ll.apply({{kReg, reg}, {kLane, lane}, {kWarp, 0}, {kBlock, 0}});
        int64_t off = 0;
        for (size_t d = 0; d < coords.size(); d++)
          off += coords[d].second * strides[d];
        return off;
      };
      const auto &srcB = srcLL.getBases();
      const auto &dstB = dstLL.getBases();
      const int NL = 32;
      bool ok = ebShfl == 1 &&
                deltaVec(srcB.find(kWarp)->second) ==
                    deltaVec(dstB.find(kWarp)->second) &&
                deltaVec(srcB.find(kBlock)->second) ==
                    deltaVec(dstB.find(kBlock)->second);
      // offset -> (srcReg, srcLane) within the warp.
      std::map<int64_t, std::pair<int, int>> offToSrc;
      if (ok)
        for (int s = 0; s < nSrc; s++)
          for (int L = 0; L < NL; L++)
            offToSrc[offsetOf(srcLL, s, L)] = {s, L};
      // Per dst register, the analysed move: which source lane (slExpr), which
      // source register per selector pattern (patReg), and the runtime selector
      // pattern expression (patExpr, valued 0..nPat-1).
      struct DstMove {
        std::string slExpr;        // source lane expression
        std::string patExpr;       // runtime selector pattern (0..nPat-1)
        std::map<int, int> patReg; // pattern -> source register
      };
      std::vector<DstMove> mv(nDst);
      for (int d = 0; d < nDst && ok; d++) {
        std::vector<int> sReg(NL), sLane(NL);
        for (int L = 0; L < NL; L++) {
          auto it = offToSrc.find(offsetOf(dstLL, d, L));
          if (it == offToSrc.end()) { ok = false; break; }
          sReg[L] = it->second.first;
          sLane[L] = it->second.second;
        }
        if (!ok) break;
        // Fit srcLane as XOR-affine in lane bits.
        int c0 = sLane[0];
        int delta[5];
        for (int i = 0; i < 5; i++) delta[i] = sLane[1 << i] ^ c0;
        for (int L = 0; L < NL && ok; L++) {
          int pred = c0;
          for (int i = 0; i < 5; i++) if ((L >> i) & 1) pred ^= delta[i];
          if (pred != sLane[L]) ok = false;
        }
        if (!ok) break;
        // Lane bits that affect the source register.
        std::vector<int> sbits;
        for (int i = 0; i < 5; i++) {
          bool affects = false;
          for (int L = 0; L < NL; L++)
            if (sReg[L] != sReg[L ^ (1 << i)]) { affects = true; break; }
          if (affects) sbits.push_back(i);
        }
        if (sbits.size() > 2) { ok = false; break; }
        // srcReg must depend only on sbits; build pattern->reg.
        std::map<int, int> patReg;
        for (int L = 0; L < NL && ok; L++) {
          int pat = 0;
          for (size_t k = 0; k < sbits.size(); k++)
            pat |= ((L >> sbits[k]) & 1) << k;
          auto it = patReg.find(pat);
          if (it == patReg.end()) patReg[pat] = sReg[L];
          else if (it->second != sReg[L]) ok = false;
        }
        if (!ok) break;
        // Source-lane expression (XOR-affine in lane_id bits).
        std::string slExpr = std::to_string(c0);
        for (int i = 0; i < 5; i++)
          if (delta[i] != 0)
            slExpr += " ^ (((lane_id >> " + std::to_string(i) + ") & 1) ? " +
                      std::to_string(delta[i]) + " : 0)";
        std::string patExpr;
        for (size_t k = 0; k < sbits.size(); k++) {
          std::string term = "(((lane_id >> " + std::to_string(sbits[k]) +
                             ") & 1) << " + std::to_string(k) + ")";
          patExpr += (patExpr.empty() ? "" : " | ") + term;
        }
        if (patExpr.empty()) patExpr = "0";
        mv[d] = {std::move(slExpr), std::move(patExpr), std::move(patReg)};
      }
      // Group consecutive dst registers that read from the SAME source lane
      // into a single 32-bit shuffle word (a __shfl moves 32 bits regardless of
      // payload), packing up to 4 source bytes per word and extracting each
      // dst register's byte by its selector pattern. This matches the PTX
      // backend's one-shuffle-per-4-elements density.
      struct GrpInfo {
        std::string slExpr;   // source lane
        std::string packExpr; // packed source word _p
        int dStart, dEnd;
      };
      std::vector<GrpInfo> groups;
      std::vector<int> byteGroup(nDst, -1); // output byte -> group index
      std::vector<int> bytePosBase(nDst, 0);// memberBase of that byte in _p
      std::vector<std::vector<std::string>> body; // byte-path (fallback)
      // Aligned 32-bit source-word loads, interned and shared across groups: the
      // packed word _p for each group is gathered from these via one __byte_perm
      // rather than 4 per-byte loads + shifts + ORs. Adjacent groups reuse the
      // same words, so this collapses the convert's LOP3/SHF/PRMT down toward the
      // PTX backend's one-PRMT-per-word density.
      std::map<int, std::string> wordName;
      std::vector<std::string> wordDecl;
      auto internWord = [&](int w) -> std::string {
        auto it = wordName.find(w);
        if (it != wordName.end()) return it->second;
        // Prefix with this convert's result var: these decls are emitted at
        // function scope, so plain "_w0" collides when a kernel contains two
        // shuffle-converts (nvcc: "_w0 has already been declared").
        std::string n = "_w" + var + "_" + std::to_string(wordName.size());
        wordName[w] = n;
        wordDecl.push_back("unsigned " + n + " = *(const unsigned*)&" + srcVar +
                           "[" + std::to_string(w * 4) + "];");
        return n;
      };
      for (int d = 0; d < nDst && ok;) {
        int budget = 0, e = d;
        // Extend the group while the source lane matches and bytes fit in 32b.
        while (e < nDst && mv[e].slExpr == mv[d].slExpr &&
               budget + (int)mv[e].patReg.size() <= 4) {
          budget += (int)mv[e].patReg.size();
          e++;
        }
        std::vector<std::string> lines;
        lines.push_back("int _sl = " + mv[d].slExpr + ";");
        int base = 0;
        std::vector<int> memberBase;
        std::string oldPack;          // per-byte gather (fallback)
        int posSrc[4] = {-1, -1, -1, -1};
        for (int m = d; m < e; m++) {
          memberBase.push_back(base);
          byteGroup[m] = (int)groups.size();
          bytePosBase[m] = base;
          for (auto &kv : mv[m].patReg) {
            int pos = base + kv.first;
            if (pos < 4) posSrc[pos] = kv.second;
            std::string term = "((unsigned)(*(const unsigned char*)&" + srcVar +
                               "[" + std::to_string(kv.second) + "])";
            if (pos) term += " << " + std::to_string(pos * 8);
            term += ")";
            oldPack += (oldPack.empty() ? "" : " | ") + term;
          }
          base += (int)mv[m].patReg.size();
        }
        // Gather the (possibly non-contiguous) source bytes from <=2 aligned
        // 32-bit words with a single __byte_perm; fall back to the byte gather
        // if the bytes span more than 2 words.
        std::string packExpr;
        std::vector<int> gw;
        for (int p2 = 0; p2 < 4; p2++)
          if (posSrc[p2] >= 0) {
            int w = posSrc[p2] / 4;
            bool seen = false;
            for (int gwv : gw) if (gwv == w) seen = true;
            if (!seen) gw.push_back(w);
          }
        if (gw.size() >= 1 && gw.size() <= 2) {
          std::string Wa = internWord(gw[0]);
          std::string Wb = (gw.size() == 2) ? internWord(gw[1]) : Wa;
          unsigned sel = 0;
          for (int p2 = 0; p2 < 4; p2++) {
            int nib = 0;
            if (posSrc[p2] >= 0) {
              int w = posSrc[p2] / 4, bb = posSrc[p2] % 4;
              nib = (w == gw[0] ? 0 : 4) + bb;
            }
            sel |= (unsigned)nib << (4 * p2);
          }
          const char *hexd = "0123456789abcdef";
          std::string selStr = "0x";
          for (int n = 3; n >= 0; n--) selStr += hexd[(sel >> (4 * n)) & 0xf];
          packExpr = "__byte_perm(" + Wa + ", " + Wb + ", " + selStr + ")";
        } else {
          packExpr = oldPack;
        }
        groups.push_back({mv[d].slExpr, packExpr, d, e});
        lines.push_back("unsigned _p = " + packExpr + ";");
        lines.push_back("unsigned _s = __shfl_sync(0xffffffff, _p, _sl);");
        for (int m = d; m < e; m++) {
          int b = memberBase[m - d];
          std::string sh = "((" + std::to_string(b) + " + (" + mv[m].patExpr +
                           ")) * 8)";
          lines.push_back("*(unsigned char*)&" + var + "[" + std::to_string(m) +
                          "] = (unsigned char)((_s >> " + sh + ") & 0xff);");
        }
        body.push_back(std::move(lines));
        d = e;
      }
      // Fused path: when the convert result feeds ONLY a WGMMA A-operand
      // (RS-mode), emit the output directly as packed uint32 words (one per 4
      // fp8 elements, in wgmma A order) via a single __byte_perm per word that
      // combines the (<=2) shuffle results — instead of materializing a byte
      // array that the wgmma would then re-pack. This mirrors the PTX backend,
      // which produces the A operand words straight out of the shuffles with no
      // byte twiddling. Requires nDst%4==0 and no shuffle group straddling a
      // 4-byte window; otherwise falls back to the byte path below.
      bool fuse = ok && nDst > 0 && (nDst % 4 == 0);
      if (fuse)
        for (Operation *user : result.getUsers()) {
          auto dotOp = dyn_cast<ttng::WarpGroupDotOp>(user);
          if (!dotOp || dotOp.getA() != result) { fuse = false; break; }
        }
      if (fuse)
        for (auto &g : groups)
          if (g.dStart / 4 != (g.dEnd - 1) / 4) { fuse = false; break; }
      // Hoist the loop-invariant, per-thread source-lane and byte_perm-selector
      // expressions: they are functions of lane_id only and are identical (or
      // few-valued) across the unrolled windows, but nvcc does not reliably CSE
      // the lane_id bit extractions, leaving hundreds of redundant LOP3/SHF/PRMT
      // in the hot loop. Intern each distinct expr to a single declaration.
      std::vector<std::string> hoistDecl;
      std::map<std::string, std::string> slName, selName;
      auto internSl = [&](const std::string &e) -> std::string {
        auto it = slName.find(e);
        if (it != slName.end()) return it->second;
        std::string n = "_sl" + std::to_string(slName.size());
        slName[e] = n;
        hoistDecl.push_back("int " + n + " = " + e + ";");
        return n;
      };
      auto internSel = [&](const std::string &e) -> std::string {
        auto it = selName.find(e);
        if (it != selName.end()) return it->second;
        std::string n = "_sel" + std::to_string(selName.size());
        selName[e] = n;
        hoistDecl.push_back("unsigned " + n + " = " + e + ";");
        return n;
      };
      std::vector<std::vector<std::string>> packedBody;
      for (int p = 0; p < nDst / 4 && fuse; p++) {
        int srcG[2] = {-1, -1};
        int nS = 0, slot[4];
        for (int e2 = 0; e2 < 4; e2++) {
          int gi = byteGroup[4 * p + e2];
          int s = -1;
          for (int t = 0; t < nS; t++)
            if (srcG[t] == gi) s = t;
          if (s < 0) {
            if (nS == 2) { fuse = false; break; }
            srcG[nS] = gi;
            s = nS++;
          }
          slot[e2] = s;
        }
        if (!fuse) break;
        std::vector<std::string> lines;
        // Compute each distinct packed source word _p once (the two shuffles in
        // a window usually share the same _p), then one __shfl per source lane.
        std::map<std::string, std::string> pName;
        std::vector<std::string> sName(nS);
        for (int t = 0; t < nS; t++) {
          auto &g = groups[srcG[t]];
          auto pit = pName.find(g.packExpr);
          std::string pv;
          if (pit == pName.end()) {
            pv = "_p" + std::to_string(pName.size());
            pName[g.packExpr] = pv;
            lines.push_back("unsigned " + pv + " = " + g.packExpr + ";");
          } else {
            pv = pit->second;
          }
          std::string sv = "_s" + std::to_string(t);
          lines.push_back("unsigned " + sv + " = __shfl_sync(0xffffffff, " + pv +
                          ", " + internSl(g.slExpr) + ");");
          sName[t] = sv;
        }
        std::string sel;
        for (int e2 = 0; e2 < 4; e2++) {
          int m = 4 * p + e2;
          std::string posE = "(" + std::to_string(bytePosBase[m]) + " + (" +
                             mv[m].patExpr + "))";
          std::string nib = "((unsigned)(" + std::to_string(slot[e2] * 4) +
                            " + " + posE + ") << " + std::to_string(4 * e2) + ")";
          sel += (sel.empty() ? "" : " | ") + nib;
        }
        std::string src1 = (nS == 2) ? sName[1] : sName[0];
        lines.push_back(var + "[" + std::to_string(p) + "] = __byte_perm(" +
                        sName[0] + ", " + src1 + ", " + internSel(sel) + ");");
        packedBody.push_back(std::move(lines));
      }
      if (fuse) {
        emit("// convert_layout (warp-shuffle -> packed uint32 for wgmma A, "
             "no smem/barrier)");
        emit("uint32_t " + var + "[" + std::to_string(nDst / 4) + "];");
        for (auto &w : wordDecl) emit(w);
        for (auto &d : hoistDecl) emit(d);
        for (auto &lines : packedBody) {
          emit("{");
          indent();
          for (auto &l : lines) emit(l);
          dedent();
          emit("}");
        }
        packedU32Convert[result] = nDst / 4;
        return;
      }
      if (ok && nDst > 0) {
        emit("// convert_layout (warp-shuffle, no smem/barrier)");
        emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
        for (auto &w : wordDecl) emit(w);
        for (auto &lines : body) {
          emit("{");
          indent();
          for (auto &l : lines) emit(l);
          dedent();
          emit("}");
        }
        return;
      }
    }
    // Shared memory intermediary
    emit("// convert_layout via shared memory");
    emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
    emit("{");
    indent();
    // The scratch must span every shared-memory offset actually written by the
    // store (src layout) and read by the load (dst layout). Both are emitted as
    // `regBase + sum_of_set_{lane,warp}_bit_deltas`, with all deltas >= 0 (the
    // LinearLayout coords are non-negative and strides positive). So the maximum
    // accessed offset for a layout = max_reg(regBase) + sum(all lane deltas) +
    // sum(all warp deltas); the buffer needs maxOff+1 elements. Sizing by the
    // old `numWarps*32*max(nSrc,nDst)` over-allocated up to 64x for broadcast
    // converts (nSrc=1 -> nDst=64), e.g. a 1KB convert ballooned to 64KB and
    // blew the persistent-matmul epilogue past the smem limit.
    auto computeMaxOff = [&](RankedTensorType tt, int nRegs) -> int64_t {
      auto LL = ttg::toLinearLayout(tt);
      auto *ctx = tt.getContext();
      auto kReg = mlir::StringAttr::get(ctx, "register");
      auto kLane = mlir::StringAttr::get(ctx, "lane");
      auto kWarp = mlir::StringAttr::get(ctx, "warp");
      auto kBlock = mlir::StringAttr::get(ctx, "block");
      auto shp = tt.getShape();
      SmallVector<int64_t> strides(shp.size());
      if (!strides.empty()) {
        strides.back() = 1;
        for (int d = (int)shp.size() - 2; d >= 0; d--)
          strides[d] = strides[d + 1] * shp[d + 1];
      }
      const auto &bases = LL.getBases();
      auto sumDeltas = [&](mlir::StringAttr key) -> int64_t {
        int64_t s = 0;
        auto it = bases.find(key);
        if (it == bases.end()) return 0;
        for (const auto &b : it->second) {
          int64_t delta = 0;
          for (size_t d = 0; d < b.size() && d < strides.size(); d++)
            delta += b[d] * strides[d];
          if (delta > 0) s += delta;
        }
        return s;
      };
      int64_t laneSum = sumDeltas(kLane);
      int64_t warpSum = sumDeltas(kWarp);
      int64_t maxRegBase = 0;
      for (int i = 0; i < nRegs; i++) {
        auto coords = LL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        int64_t rb = 0;
        for (size_t d = 0; d < coords.size() && d < strides.size(); d++)
          rb += coords[d].second * strides[d];
        maxRegBase = std::max(maxRegBase, rb);
      }
      return maxRegBase + laneSum + warpSum;
    };
    int64_t smemElems =
        std::max(computeMaxOff(srcRtt, nSrc), computeMaxOff(rtt, nDst)) + 1;
    int eb = getTypeSizeInBytes(elemType);
    // Pure-temporary scratch (stored then read back into `var` before the block
    // closes). Save/restore the bump pointer so sequential convert_layouts reuse
    // this space rather than stacking — see the matching note in the MMA→blocked
    // path above (over-allocation blew the persistent fp8 epilogue past 232448).
    int savedSmemOffsetLLCvt = sharedMemOffset;
    int smemOff = (sharedMemOffset + 127) & ~127;
    sharedMemOffset = smemOff + smemElems * eb;
    if (sharedMemOffset > peakSharedMem) peakSharedMem = sharedMemOffset;
    emit(cudaType + "* _cvt = (" + cudaType + "*)(shared_mem + " + std::to_string(smemOff) + ");");
    // Store with source layout using LinearLayout
    {
      auto srcLL = ttg::toLinearLayout(srcRtt);
      auto kReg = mlir::StringAttr::get(srcRtt.getContext(), "register");
      auto kLane = mlir::StringAttr::get(srcRtt.getContext(), "lane");
      auto kWarp = mlir::StringAttr::get(srcRtt.getContext(), "warp");
      auto kBlock = mlir::StringAttr::get(srcRtt.getContext(), "block");
      // Compute per-register offsets and emit stores using LL
      // Flatten multi-dim coords to 1D index
      auto srcShape = srcRtt.getShape();
      emit("// Store src layout via LinearLayout");
      emit("{");
      indent();
      // Compute lane and warp contributions to offset
      const auto &srcBases = srcLL.getBases();
      // For each register, compute the flat offset as f(register, lane, warp)
      // offset = sum_over_dims(coord[d] * stride[d])
      // But we need to emit runtime code. Simplest: emit per-register constants.
      for (int i = 0; i < nSrc; i++) {
        auto regCoords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        // Compute lane/warp contribution as deltas
        // For simplicity: compute full offset formula at runtime
        // offset(reg, lane, warp) = LL.apply(reg, lane, warp) flattened
        // But this is expensive to emit. Use a simpler approach:
        // Store: _cvt[linear_offset] = srcVar[i] where linear_offset depends on tid
      }
      // Actually, use a generic approach: compute per-register flat offsets
      // and emit them as compile-time constants, with lane/warp as runtime vars
      // This is what emitOffsetForLayout does conceptually

      // Simpler approach: use the fact that the LL is linear:
      // offset(reg, lane, warp) = reg_offset[reg] XOR lane_offsets XOR warp_offsets
      // For the store, compute the flat 1D address from the multi-dim coords
      SmallVector<int64_t> strides(srcShape.size());
      strides.back() = 1;
      for (int d = srcShape.size() - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * srcShape[d + 1];

      // For each register, compute the base offset (at lane=0, warp=0)
      for (int i = 0; i < nSrc; i++) {
        auto coords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        int64_t regBase = 0;
        for (size_t d = 0; d < coords.size(); d++)
          regBase += coords[d].second * strides[d];

        // Lane contribution: for each lane bit, compute the delta
        // Warp contribution: for each warp bit, compute the delta
        // Build the offset expression: regBase + lane_delta + warp_delta
        // The LL is additive (XOR-based): apply lane and warp bits
        const auto &laneBases = srcBases.find(kLane)->second;
        const auto &warpBases = srcBases.find(kWarp)->second;

        // Compute lane offset XOR pattern
        std::string offsetExpr = std::to_string(regBase);
        // Add lane contribution
        for (size_t lb = 0; lb < laneBases.size(); lb++) {
          int64_t delta = 0;
          for (size_t d = 0; d < laneBases[lb].size(); d++)
            delta += laneBases[lb][d] * strides[d];
          if (delta != 0) {
            offsetExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
          }
        }
        // Add warp contribution
        for (size_t wb = 0; wb < warpBases.size(); wb++) {
          int64_t delta = 0;
          for (size_t d = 0; d < warpBases[wb].size(); d++)
            delta += warpBases[wb][d] * strides[d];
          if (delta != 0) {
            offsetExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
          }
        }
        emit("_cvt[" + offsetExpr + "] = " + srcVar + "[" + std::to_string(i) + "];");
      }
      dedent();
      emit("}");
    }
    blockSync();
    // Load with dest layout using LinearLayout
    {
      auto dstLL = ttg::toLinearLayout(rtt);
      auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
      auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
      auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
      auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");
      auto dstShape = rtt.getShape();
      SmallVector<int64_t> strides(dstShape.size());
      strides.back() = 1;
      for (int d = dstShape.size() - 2; d >= 0; d--)
        strides[d] = strides[d + 1] * dstShape[d + 1];
      const auto &dstBases = dstLL.getBases();
      const auto &laneBases = dstBases.find(kLane)->second;
      const auto &warpBases = dstBases.find(kWarp)->second;
      emit("// Load dst layout via LinearLayout");
      emit("{");
      indent();
      for (int i = 0; i < nDst; i++) {
        auto coords = dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        int64_t regBase = 0;
        for (size_t d = 0; d < coords.size(); d++)
          regBase += coords[d].second * strides[d];
        std::string offsetExpr = std::to_string(regBase);
        for (size_t lb = 0; lb < laneBases.size(); lb++) {
          int64_t delta = 0;
          for (size_t d = 0; d < laneBases[lb].size(); d++)
            delta += laneBases[lb][d] * strides[d];
          if (delta != 0)
            offsetExpr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
        }
        for (size_t wb = 0; wb < warpBases.size(); wb++) {
          int64_t delta = 0;
          for (size_t d = 0; d < warpBases[wb].size(); d++)
            delta += warpBases[wb][d] * strides[d];
          if (delta != 0)
            offsetExpr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
        }
        emit(var + "[" + std::to_string(i) + "] = _cvt[" + offsetExpr + "];");
      }
      dedent();
      emit("}");
    }
    blockSync();
    // Scratch consumed — restore the bump pointer so this space is reusable.
    sharedMemOffset = savedSmemOffsetLLCvt;
    dedent();
    emit("}");
  }
}
void CUDACodeGen::emitMemDescSubview(Operation *op) {
  if (op->getNumResults() == 0 || op->getNumOperands() == 0)
    return;
  auto result = op->getResult(0);
  auto base = op->getOperand(0);
  auto baseVar = getVar(base);

  // Extract result memdesc shape to compute stride
  auto memDescType = cast<ttg::MemDescType>(result.getType());
  auto shape = memDescType.getShape();
  auto elemType = memDescType.getElementType();
  int elemBytes = getTypeSizeInBytes(elemType);
  auto cudaType = getCUDAType(elemType);

  // memdesc_subslice: static per-dimension offsets given as an attribute (NOT a
  // runtime index operand). It selects a sub-tile of the SAME logical tensor —
  // e.g. the K-slices x[0:16], x[16:32], ... feeding successive K=16 WGMMAs in
  // the mxfp/scaled-dot path. The byte offset of the sub-tile origin must follow
  // the SAME swizzle-aware layout convention that emitWarpGroupDot uses when it
  // advances the WGMMA descriptor across K-tiles (see the b_byte_offset math in
  // emitWarpGroupDot); otherwise every slice aliases offset 0 and all K-chunks
  // multiply against the same operand slice.
  if (auto subsliceOp = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    ArrayRef<int32_t> offsets = subsliceOp.getOffsets();
    int64_t offsetBytes = 0;
    if (auto nvmma =
            dyn_cast<ttg::NVMMASharedEncodingAttr>(memDescType.getEncoding())) {
      // Swizzle-aware physical offset (rank-2). The contiguous dimension is K
      // (dim0) when transposed, else N (dim1). eRow = swizzle/elemBytes is the
      // width of a swizzling atom along the contiguous dimension.
      int swizzle = nvmma.getSwizzlingByteWidth();
      int eRow = std::max<int>(1, swizzle / std::max(1, elemBytes));
      int64_t off0 = offsets.size() > 0 ? offsets[0] : 0;
      int64_t off1 = offsets.size() > 1 ? offsets[1] : 0;
      if (nvmma.getTransposed()) {
        // physical [stride=dim1, contig=dim0]
        int64_t strideDim = shape.size() > 1 ? shape[1] : 1;
        offsetBytes = ((off0 / eRow) * (strideDim * eRow) + off1 * eRow +
                       (off0 % eRow)) * elemBytes;
      } else {
        // physical [stride=dim0, contig=dim1]. Slicing the contiguous (N) dim
        // past an eRow swizzle-chunk boundary is NOT a flat byte offset: each
        // eRow-wide column chunk occupies a separate contiguous block of
        // strideDim*eRow elements (mirrors emitWarpGroupDot's b_byte_offset
        // non-transposed path). Derive the chunk stride from the PARENT (alloc)
        // dims so it matches the stored swizzle layout; a flat off1*elemBytes
        // desyncs the sub-tile's swizzle from the data for off1 >= eRow. For
        // off1 == 0 (the common K-only-slice callers) both forms coincide.
        int rank2 = (int)shape.size();
        auto allocShape2 = memDescType.getAllocShape().take_back(rank2);
        SmallVector<int64_t> allocPerCTA2 =
            ttg::getShapePerCTA(memDescType.getEncoding(), allocShape2);
        int64_t strideDim = allocPerCTA2[0];
        int64_t contigFull = rank2 > 1 ? allocPerCTA2[1] : 1;
        int64_t eRowEff = std::min<int64_t>(contigFull, eRow);
        offsetBytes = ((off1 / eRowEff) * (strideDim * eRowEff) +
                       off0 * eRowEff + (off1 % eRowEff)) * elemBytes;
      }
    } else {
      // Non-swizzled / generic: offset within the PARENT buffer. Strides must
      // come from the alloc shape (trailing `rank` dims, per-CTA), not the
      // view shape — a 64x64 subview of a 128x128 tile has parent row stride
      // 128. Honor the shared encoding's order (the plain order-derived
      // convention used by local_store/local_load).
      int rank = (int)shape.size();
      auto allocShape = memDescType.getAllocShape().take_back(rank);
      SmallVector<int64_t> allocPerCTA = ttg::getShapePerCTA(
          memDescType.getEncoding(), allocShape);
      SmallVector<int64_t> strides(rank, 1);
      if (auto swiz = dyn_cast<ttg::SwizzledSharedEncodingAttr>(
              memDescType.getEncoding())) {
        auto order = swiz.getOrder();
        int64_t s = 1;
        for (int oi = 0; oi < rank; oi++) {
          strides[order[oi]] = s;
          s *= allocPerCTA[order[oi]];
        }
      } else {
        strides[rank - 1] = 1;
        for (int d = rank - 2; d >= 0; --d)
          strides[d] = strides[d + 1] * allocPerCTA[d + 1];
      }
      int64_t elemOffset = 0;
      for (int d = 0; d < rank; ++d) {
        int64_t off = d < (int)offsets.size() ? offsets[d] : 0;
        elemOffset += off * strides[d];
      }
      offsetBytes = elemOffset * elemBytes;
    }
    auto var = newVar("sv");
    valueToVar[result] = var;
    emit(cudaType + "* " + var + " = (" + cudaType + "*)((char*)" + baseVar +
         " + " + std::to_string(offsetBytes) + ");");
    return;
  }

  // memdesc_index / memdesc_subview: runtime index selects one full sub-tile out
  // of a leading (e.g. multi-buffered pipeline) dimension. Stride = full tile.
  int64_t stride = 1;
  for (auto s : shape) stride *= s;
  int byteStride = stride * elemBytes;
  std::string idxVar = "0";
  if (op->getNumOperands() > 1)
    idxVar = getVar(op->getOperand(1));

  auto var = newVar("sv");
  valueToVar[result] = var;
  emit(cudaType + "* " + var + " = (" + cudaType + "*)((char*)" + baseVar +
       " + " + idxVar + " * " + std::to_string(byteStride) + ");");
}
void CUDACodeGen::emitMemDescTrans(ttg::MemDescTransOp op) {
  // Transposed view — no data movement, just alias with transposed shape
  auto result = op.getResult();
  auto src = op.getSrc();
  valueToVar[result] = getVar(src);
  emit("// memdesc_trans: transposed view (no data movement)");
}
void CUDACodeGen::emitWarpGroupDot(ttng::WarpGroupDotOp op) {
  auto result = op.getResult();
  auto a = op.getA();
  auto b = op.getB();
  auto acc = op.getC();
  auto rtt = cast<RankedTensorType>(result.getType());
  // CGA: with CTASplitNum > 1 each CTA owns only its slice of the tensor; all
  // WGMMA tiling below is per-CTA, so use the per-CTA shape (e.g. M=512 with
  // CTASplitNum=[2,1] -> 256 rows per CTA).
  auto rShapePerCTA = ttg::getShapePerCTA(rtt);
  int M_block = rShapePerCTA[0], N_block = rShapePerCTA[1];

  // Detect RS mode: A is tensor (registers) vs memdesc (shared memory)
  bool isRS = isa<RankedTensorType>(a.getType());

  // Get B shared layout info
  auto bMemDesc = cast<ttg::MemDescType>(b.getType());
  auto bEnc = dyn_cast<ttg::NVMMASharedEncodingAttr>(bMemDesc.getEncoding());
  int trans_b = (bEnc && bEnc.getTransposed()) ? 0 : 1;
  int swizzle_b = bEnc ? bEnc.getSwizzlingByteWidth() : 128;

  // Get K from memdesc shape (per-CTA; K is never CTA-split for dot operands)
  int K_block = 32;
  if (isRS) {
    K_block = ttg::getShapePerCTA(
        cast<RankedTensorType>(a.getType()))[1];
  } else {
    auto aMemDesc = cast<ttg::MemDescType>(a.getType());
    K_block = ttg::getShapePerCTA(aMemDesc)[1];
  }

  // Get element types
  Type aElemType = isRS ? cast<RankedTensorType>(a.getType()).getElementType()
                        : cast<ttg::MemDescType>(a.getType()).getElementType();
  std::string aElem;
  if (aElemType.isF16()) aElem = "f16";
  else if (aElemType.isBF16()) aElem = "bf16";
  else if (isa<Float8E4M3FNType>(aElemType)) aElem = "e4m3";
  else if (isa<Float8E5M2Type>(aElemType)) aElem = "e5m2";
  else if (aElemType.isF32()) aElem = "tf32";
  else if (aElemType.isInteger(8)) aElem = "s8";
  else aElem = "tf32";
  // B element type may differ (e.g., e4m3 x e5m2)
  Type bElemType = cast<ttg::MemDescType>(b.getType()).getElementType();
  std::string bElem;
  if (bElemType.isF16()) bElem = "f16";
  else if (bElemType.isBF16()) bElem = "bf16";
  else if (isa<Float8E4M3FNType>(bElemType)) bElem = "e4m3";
  else if (isa<Float8E5M2Type>(bElemType)) bElem = "e5m2";
  else if (bElemType.isF32()) bElem = "tf32";
  else if (bElemType.isInteger(8)) bElem = "s8";
  else bElem = "tf32";
  bool isInt8 = (aElem == "s8");
  std::string cElem = isInt8 ? "s32" : "f32";

  bool isFp8 = (aElem == "e4m3" || aElem == "e5m2");
  int wgmma_k = (isFp8 || isInt8) ? 32 : ((aElem == "f16" || aElem == "bf16") ? 16 : 8);
  int wgmma_m = 64;
  // Multi-warp-group split comes from the result MMA encoding's warpsPerCTA,
  // NOT from numWarps alone: warpsPerCTA=[4,2] (e.g. 64x64 tile, num_warps=8)
  // splits the two warpgroups along N, while [8,1] splits along M. Warp
  // linearization for NvidiaMma v3 is dim0(M)-fastest (see WGMMA.cpp
  // loadA/loadB: warpM = warp % wpt[0], warpN = (warp / wpt[0]) % wpt[1]),
  // so warpgroup gM = wg % n_warp_groups_m, gN = wg / n_warp_groups_m.
  int wpcM = numWarps, wpcN = 1;
  if (auto mmaEnc = dyn_cast<ttg::NvidiaMmaEncodingAttr>(rtt.getEncoding())) {
    auto wpc = mmaEnc.getWarpsPerCTA();
    wpcM = wpc[0];
    wpcN = wpc.size() > 1 ? wpc[1] : 1;
  }
  int n_warp_groups_m = std::max(wpcM / 4, 1);
  int n_warp_groups_n = std::max(wpcN, 1);
  // Per-warpgroup N extent.
  int N_wg = N_block / n_warp_groups_n;
  // Valid WGMMA N dimensions: 8-256 (power of 2). Use full per-wg N when
  // possible. Integer (s32) wgmma tops out at n=224, so cap s8 at 128.
  int wgmma_n = std::min(N_wg, isInt8 ? 128 : 256);
  int n_n_tiles = N_wg / wgmma_n;
  int n_m_tiles = M_block / wgmma_m;
  int n_k_tiles = K_block / wgmma_k;
  int n_out_regs = (wgmma_m * wgmma_n) / 128;
  int m_tiles_per_wg = std::max(1, n_m_tiles / n_warp_groups_m);
  int total_acc = n_out_regs * m_tiles_per_wg * n_n_tiles;
  if (n_m_tiles % std::min(n_warp_groups_m, n_m_tiles) != 0 ||
      N_block % n_warp_groups_n != 0 || N_wg % wgmma_n != 0) {
    std::string msg = "[emit_cuda] unsupported wgmma warpgroup split: M_block=" +
                      std::to_string(M_block) + " N_block=" + std::to_string(N_block) +
                      " warpsPerCTA=[" + std::to_string(wpcM) + "," +
                      std::to_string(wpcN) + "]";
    op->emitError(msg);
    if (!emitFailed) {
      emitFailed = true;
      emitErrorMsg = msg;
    }
    return;
  }

  // Hopper fp8 wgmma accumulates C imprecisely (reduced-precision adds inside
  // the wgmma pipeline). When maxNumImpreciseAcc <= K_block the PTX backend
  // (WGMMA.cpp convertDot) accumulates each chunk of maxNumImpreciseAcc
  // K-elements into a temporary register accumulator (first wgmma of the
  // chunk uses scale-d=0) and folds it into the persistent f32 accumulator
  // with precise add.f32 instructions. Mirror that here: it is both the
  // numeric fix (test_dot_max_num_imprecise_acc tolerances) and what the
  // test's `ptx.count("add.f32")` introspection expects.
  uint32_t maxImpAcc = op.getMaxNumImpreciseAcc();
  bool needsPartialAcc =
      isFp8 && rtt.getElementType().isF32() && maxImpAcc <= (uint32_t)K_block;
  int k_tiles_per_chunk =
      needsPartialAcc ? std::max(1, (int)maxImpAcc / wgmma_k) : n_k_tiles;
  if (needsPartialAcc && isRS) {
    std::string msg = "[emit_cuda] maxNumImpreciseAcc partial accumulation is "
                      "not implemented for RS-mode wgmma";
    op->emitError(msg);
    if (!emitFailed) {
      emitFailed = true;
      emitErrorMsg = msg;
    }
    return;
  }

  auto accVar = getVar(acc);
  auto bVar = getVar(b);

  // Reuse accumulator variable in-place when safe — WGMMA modifies acc via
  // "+f" constraints, so we avoid a redundant array copy per loop iteration.
  // A splat-scalar accumulator (e.g. a `zeros` constant emitted as a scalar
  // `const float c = 0.0f`) cannot be aliased in-place: WGMMA indexes the
  // accumulator as `acc[0..n-1]`, so it must be materialized into a real
  // register array first, broadcasting the scalar to every element.
  bool accIsScalar = scalarValues.contains(acc);
  // f16 result type: match the PTX backend, which uses a true f16 accumulator
  // (wgmma .f16.f16.f16) when the dot result is f16 and both inputs are f16.
  // Two f16 accumulator values are packed per 32-bit register ("+r"); after the
  // wait we unpack into the __half result array consumers expect. (bf16 inputs
  // never accumulate in bf16, so they keep the f32-acc + convert path below.)
  bool resIsF16 = rtt.getElementType().isF16();
  bool f16Acc = resIsF16 && aElem == "f16" && bElem == "f16";
  if (f16Acc) cElem = "f16";
  if (isInt8 && !rtt.getElementType().isInteger(32)) {
    std::string msg = "[emit_cuda] s8 warp_group_dot expects an i32 result";
    op->emitError(msg);
    if (!emitFailed) {
      emitFailed = true;
      emitErrorMsg = msg;
    }
    return;
  }
  std::string accType = f16Acc ? "uint32_t" : (isInt8 ? "int" : "float");
  // # of accumulator registers per wgmma instruction / total array size.
  // f16 packs two accumulator values per 32-bit register.
  int nAccRegs = f16Acc ? n_out_regs / 2 : n_out_regs;
  int accArrSize = f16Acc ? total_acc / 2 : total_acc;
  std::string accConstr = (f16Acc || isInt8) ? "\"+r\"(" : "\"+f\"(";
  std::string var;
  if (acc.hasOneUse() && !accIsScalar && !resIsF16) {
    var = accVar;
    valueToVar[result] = accVar;
  } else {
    var = newVar("wgmma");
    valueToVar[result] = var;
    emit(accType + " " + var + "[" + std::to_string(accArrSize) + "];");
    emit("#pragma unroll");
    if (f16Acc) {
      // Pack pairs of f16 accumulator inputs (or 0 for a splat-zero acc).
      std::string pk =
          accIsScalar
              ? "0u"
              : ("((uint32_t)__half_as_ushort(" + accVar + "[2*_i]) | "
                 "((uint32_t)__half_as_ushort(" + accVar + "[2*_i+1]) << 16))");
      emit("for (int _i = 0; _i < " + std::to_string(accArrSize) + "; _i++) " +
           var + "[_i] = " + pk + ";");
    } else {
      emit("for (int _i = 0; _i < " + std::to_string(total_acc) + "; _i++) " +
           var + "[_i] = (" + accType + ")(" + (accIsScalar ? accVar : (accVar + "[_i]")) + ");");
    }
  }

  emit("// WGMMA: m" + std::to_string(wgmma_m) + "n" + std::to_string(wgmma_n) +
       "k" + std::to_string(wgmma_k) + "." + cElem + "." + aElem + "." + bElem +
       (isRS ? " (RS mode)" : " (SS mode)"));
  emit("{");
  indent();

  // B descriptor setup
  if (!isRS) {
    auto aVar = getVar(a);
    int swizzle_a = 128;
    bool a_is_transposed = false;
    int a_shape1 = K_block;
    // Physical leading dim of the A shared allocation. For a row-sliced sub-tile
    // (e.g. q_smem.slice(0,64) of a [128,128] buffer) the swizzle atoms are still
    // strided by the PARENT row count, not the sliced M. Using the logical M_block
    // here makes the descriptor's leadDimensionBaseOffset and the cross-atom K jump
    // too small (8192 vs 16384 @ BK=128), so k-steps past the first swizzle atom
    // read the wrong rows. Derive it from getAllocShape (matches the local_load
    // subview-stride convention at the non-swizzled path above).
    int a_lead_block = M_block;
    {
      auto aMemDesc = cast<ttg::MemDescType>(a.getType());
      auto aEnc = dyn_cast<ttg::NVMMASharedEncodingAttr>(aMemDesc.getEncoding());
      swizzle_a = aEnc ? aEnc.getSwizzlingByteWidth() : 64;
      a_is_transposed = aEnc && aEnc.getTransposed();
      a_shape1 = aMemDesc.getShape()[1];
      int aRank = (int)aMemDesc.getShape().size();
      auto aAllocShape = aMemDesc.getAllocShape().take_back(aRank);
      auto aAllocPerCTA = ttg::getShapePerCTA(aMemDesc.getEncoding(), aAllocShape);
      a_lead_block = aAllocPerCTA[0];
      if (a_is_transposed)
        a_shape1 = aAllocPerCTA[1];
    }
    // Transposed (M-contiguous) A in SS mode requires the imm-trans-a flag,
    // which only exists for 16-bit wgmma. The pipeline normalizes int8/fp8/
    // tf32 operands to K-major, so this should never trigger.
    if (a_is_transposed && !(aElem == "f16" || aElem == "bf16")) {
      std::string msg = "[emit_cuda] transposed A in SS-mode wgmma is only "
                        "supported for f16/bf16 (got " + aElem + ")";
      op->emitError(msg);
      if (!emitFailed) {
        emitFailed = true;
        emitErrorMsg = msg;
      }
      return;
    }
    emit("uint32_t smem_addr_a = (unsigned)__cvta_generic_to_shared(" + aVar + ");");
    emit("uint32_t smem_addr_b = (unsigned)__cvta_generic_to_shared(" + bVar + ");");
    emitBlank();

    bool b_is_transposed = bEnc && bEnc.getTransposed();
    int elem_bytes_a = (isFp8 || isInt8) ? 1 : ((aElem == "f16" || aElem == "bf16") ? 2 : 4);
    int elem_bytes_b = (bElem == "e4m3" || bElem == "e5m2" || bElem == "s8") ? 1 : ((bElem == "f16" || bElem == "bf16") ? 2 : 4);

    // _wg_m is hoisted to kernel level (computed once, outside any loop).

    // Collect all WGMMA PTX instructions and merge into a single asm block.
    // This gives ptxas full visibility for scheduling WGMMAs and descriptor
    // computation, avoiding the register-barrier effect of multiple 128-operand
    // asm volatile blocks that nvcc cannot schedule across.
    struct WgmmaInfo {
      int acc_base;
      int a_byte_offset; // -1 means use runtime _a_off
      int a_runtime_static; // for n_warp_groups_m > 1
      int b_byte_offset;
    };
    std::vector<WgmmaInfo> wgmmaOps;

    // Swizzle-aware row strides. In an NVMMA-swizzled tile the contiguous
    // (K) dimension is tiled into "swizzling rows" of `eRow` elements
    // (eRow = swizzleBytes/elemBytes). The stride between successive rows of
    // the *stride* dimension is `eRow` elements (not the full tile width), and
    // advancing K past a swizzling-row boundary jumps by `strideDim*eRow`
    // elements to the next swizzle atom. Using K_block as the row stride is
    // only correct when K_block == eRow (BK<=64 @ 128B swizzle); for wider K
    // tiles (e.g. BK=128) it doubles the stride and skips the atom jump,
    // reading every other row/col. See WGMMA.cpp DotOpMmaV3SmemLoader::smemLoad.
    int eRow_a = std::max(1, swizzle_a / elem_bytes_a);
    int eRow_b = std::max(1, swizzle_b / elem_bytes_b);
    // Transposed A (physical layout K(stride) × M(contig), swizzled along M):
    // advancing one wgmma_m tile crosses (wgmma_m/eRow_a) swizzle atoms of
    // (a_stride_dim*eRow_a) elements each. wgmma_m=64 is a multiple of eRow_a
    // (16/32/64), so the within-atom remainder is always 0.
    int a_stride_dim = a_is_transposed ? a_shape1 : a_lead_block;
    int a_m_stride = a_is_transposed
                         ? wgmma_m * a_stride_dim * elem_bytes_a
                         : wgmma_m * eRow_a * elem_bytes_a;
    int trans_a = a_is_transposed ? 1 : 0;
    bool isTf32 = (aElem == "tf32");
    // Integer wgmma takes only scale-d (no imm-scale-a/b, no trans flags).
    std::string immSuffix = isInt8 ? ", 1;"
                            : (isFp8 || isTf32)
                                ? ", 1, 1, 1;"
                                : ", 1, 1, 1, " + std::to_string(trans_a) +
                                      ", " + std::to_string(trans_b) + ";";

    uint64_t desc_a_template = getWGMMADescTemplate(swizzle_a, a_stride_dim);
    // For transposed B (physical layout N×K), the stride dimension is N = shape[1].
    // For non-transposed B (physical layout K×N), the stride dimension is K = shape[0].
    int b_stride_dim = b_is_transposed ? bMemDesc.getShape()[1] : bMemDesc.getShape()[0];
    uint64_t desc_b_template = getWGMMADescTemplate(swizzle_b, b_stride_dim);

    // Split descriptor templates into 32-bit halves to avoid 64-bit
    // arithmetic on the uniform pipe (UIADD3.X carry chains, USHF.R.U64).
    // The upper 32 bits are always constant; only the lower 32 bits
    // incorporate the shared memory address.
    uint32_t desc_a_lo = (uint32_t)(desc_a_template & 0xFFFFFFFF);
    uint32_t desc_a_hi = (uint32_t)(desc_a_template >> 32);
    uint32_t desc_b_lo = (uint32_t)(desc_b_template & 0xFFFFFFFF);
    uint32_t desc_b_hi = (uint32_t)(desc_b_template >> 32);
    char desc_a_lo_hex[12], desc_a_hi_hex[12], desc_b_lo_hex[12], desc_b_hi_hex[12];
    snprintf(desc_a_lo_hex, sizeof(desc_a_lo_hex), "0x%08Xu", desc_a_lo);
    snprintf(desc_a_hi_hex, sizeof(desc_a_hi_hex), "0x%08Xu", desc_a_hi);
    snprintf(desc_b_lo_hex, sizeof(desc_b_lo_hex), "0x%08Xu", desc_b_lo);
    snprintf(desc_b_hi_hex, sizeof(desc_b_hi_hex), "0x%08Xu", desc_b_hi);

    for (int m_rel = 0; m_rel < m_tiles_per_wg; m_rel++) {
      for (int n_tile = 0; n_tile < n_n_tiles; n_tile++) {
      int acc_base = m_rel * n_out_regs * n_n_tiles + n_tile * n_out_regs;
      for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
        int a_k_elem = k_tile * wgmma_k;
        // Transposed A: K is the stride (row) dim, each row is eRow_a elems.
        int a_k_offset = a_is_transposed
                             ? a_k_elem * eRow_a * elem_bytes_a
                             : ((a_k_elem / eRow_a) * (a_lead_block * eRow_a) +
                                (a_k_elem % eRow_a)) * elem_bytes_a;
        int b_byte_offset;
        if (b_is_transposed) {
          // B physical [N(stride), K(contig)], swizzled along K — same atom
          // layout as A. stride dim = b_stride_dim (= N).
          int b_k_elem = k_tile * wgmma_k;
          int b_n_elem = n_tile * wgmma_n;
          b_byte_offset = ((b_k_elem / eRow_b) * (b_stride_dim * eRow_b) +
                           b_n_elem * eRow_b +
                           (b_k_elem % eRow_b)) * elem_bytes_b;
        } else {
          // B physical [K(stride), N(contig)], swizzled along N. Each swizzle
          // chunk of eRow columns spans all K rows contiguously, so advancing
          // N past a chunk boundary jumps by b_stride_dim*eRow elements (see
          // WGMMA.cpp smemLoad leading_offset); K rows stride by eRow within
          // a chunk.
          int eRow_eff = std::min(N_block, eRow_b);
          int b_n_elem = n_tile * wgmma_n;
          int b_k_elem = k_tile * wgmma_k;
          b_byte_offset = ((b_n_elem / eRow_eff) * (b_stride_dim * eRow_eff) +
                           b_k_elem * eRow_eff +
                           (b_n_elem % eRow_eff)) * elem_bytes_b;
        }

        WgmmaInfo info;
        info.acc_base = acc_base;
        info.b_byte_offset = b_byte_offset;
        if (n_warp_groups_m > 1) {
          // mma v3 linear layout interleaves M: warpgroup bit -> row +64,
          // register rep m_rel -> row +64*n_warp_groups_m. So warpgroups are
          // NOT contiguous when m_tiles_per_wg > 1; each rep strides over all
          // warpgroups.
          info.a_byte_offset = -1;
          info.a_runtime_static = m_rel * n_warp_groups_m * a_m_stride + a_k_offset;
        } else {
          info.a_byte_offset = m_rel * a_m_stride + a_k_offset;
          info.a_runtime_static = 0;
        }
        wgmmaOps.push_back(info);
      }
      } // end n_tile
    }

    // Group WGMMAs by acc_base — WGMMAs sharing the same accumulator registers
    // (same m_rel, n_tile; different k_tiles) are merged into a single asm volatile
    // block. This eliminates inter-WGMMA scheduling barriers: instead of N separate
    // asm volatile blocks each with n_out_regs "+f" constraints, we get ONE block
    // with n_out_regs "+f" + 2*N descriptor "l" inputs. ptxas can then freely
    // interleave WGMMA instructions with descriptor computation.
    std::map<int, std::vector<size_t>> accGroups;
    for (size_t wi = 0; wi < wgmmaOps.size(); wi++) {
      accGroups[wgmmaOps[wi].acc_base].push_back(wi);
    }

    // Runtime warpgroup decomposition. _wg_m (kernel-level) is the LINEAR
    // warpgroup index (warp_id/4). With an N-split ([4,2]) or an M-broadcast
    // (more M warpgroups than M tiles), derive the effective M index; with an
    // N-split also derive the per-warpgroup B byte offset.
    std::string wgMExpr = "_wg_m";
    if (n_warp_groups_m > 1 &&
        (n_warp_groups_n > 1 || n_warp_groups_m > n_m_tiles)) {
      emit("const int _wg_am = (_wg_m % " + std::to_string(n_warp_groups_m) +
           ") % " + std::to_string(n_m_tiles) + ";");
      wgMExpr = "_wg_am";
    }
    if (n_warp_groups_n > 1) {
      // Byte stride between adjacent warpgroups' N slabs of the B tile.
      int b_wg_stride;
      if (b_is_transposed) {
        // N is the stride (row) dim: each row is eRow_b elements.
        b_wg_stride = N_wg * eRow_b * elem_bytes_b;
      } else if (N_wg % eRow_b == 0) {
        // N slab spans whole swizzle atoms.
        b_wg_stride = (N_wg / eRow_b) * b_stride_dim * eRow_b * elem_bytes_b;
      } else if (n_warp_groups_n * N_wg <= eRow_b) {
        // All N slabs live inside one swizzle atom row.
        b_wg_stride = N_wg * elem_bytes_b;
      } else {
        std::string msg =
            "[emit_cuda] unsupported N-split wgmma B offset: N_wg=" +
            std::to_string(N_wg) + " eRow_b=" + std::to_string(eRow_b);
        op->emitError(msg);
        if (!emitFailed) {
          emitFailed = true;
          emitErrorMsg = msg;
        }
        return;
      }
      emit("const int _b_wgoff = (_wg_m / " + std::to_string(n_warp_groups_m) +
           ") * " + std::to_string(b_wg_stride) + ";");
    }

    // With maxNumImpreciseAcc partial accumulation, each chunk of
    // k_tiles_per_chunk wgmmas accumulates into a temporary register array
    // (first wgmma of the chunk uses scale-d=0), then folds into the
    // persistent accumulator with precise add.f32 after a wait.
    std::string paccVar;
    if (needsPartialAcc) {
      paccVar = newVar("pacc");
      emit("float " + paccVar + "[" + std::to_string(n_out_regs) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(n_out_regs) + "; _i++) " +
           paccVar + "[_i] = 0.0f;");
    }

    for (auto& [acc_base, indices] : accGroups) {
      for (size_t chunk0 = 0; chunk0 < indices.size();
           chunk0 += (size_t)k_tiles_per_chunk) {
        size_t chunkEnd =
            std::min(chunk0 + (size_t)k_tiles_per_chunk, indices.size());
        size_t chunkLen = chunkEnd - chunk0;
      emit("{");
      indent();

      // Pre-compute all descriptors using 32-bit arithmetic + mov.b64 packing.
      // This avoids 64-bit carry chains on the uniform pipe.
      for (size_t gi = 0; gi < chunkLen; gi++) {
        auto& info = wgmmaOps[indices[chunk0 + gi]];
        std::string suffix = "_" + std::to_string(gi);
        if (n_warp_groups_m > 1) {
          // Per-warpgroup M offset is one wgmma_m tile (interleaved layout),
          // not m_tiles_per_wg tiles.
          emit("int _a_off" + suffix + " = " + wgMExpr + " * " +
               std::to_string(a_m_stride) +
               " + " + std::to_string(info.a_runtime_static) + ";");
          // Mask to the 18-bit smem window: in a CGA cluster the cvta'd
          // address carries the CTA rank in high bits, which would corrupt
          // the descriptor LBO field (PTX backend masks the same way).
          emit("uint32_t _a_lo" + suffix + " = (((smem_addr_a + _a_off" + suffix + ") & 0x3FFFFu) >> 4) | " +
               std::string(desc_a_lo_hex) + ";");
        } else {
          emit("uint32_t _a_lo" + suffix + " = (((smem_addr_a + " + std::to_string(info.a_byte_offset) +
               ") & 0x3FFFFu) >> 4) | " + std::string(desc_a_lo_hex) + ";");
        }
        std::string bOff = std::to_string(info.b_byte_offset);
        if (n_warp_groups_n > 1)
          bOff += " + _b_wgoff";
        emit("uint32_t _b_lo" + suffix + " = (((smem_addr_b + " + bOff +
             ") & 0x3FFFFu) >> 4) | " + std::string(desc_b_lo_hex) + ";");
        // Pack into 64-bit descriptors using mov.b64 to keep computation in 32-bit
        emit("uint64_t desc_a" + suffix + ";");
        emit("asm(\"mov.b64 %0, {%1, %2};\" : \"=l\"(desc_a" + suffix +
             ") : \"r\"(_a_lo" + suffix + "), \"r\"(" + std::string(desc_a_hi_hex) + "));");
        emit("uint64_t desc_b" + suffix + ";");
        emit("asm(\"mov.b64 %0, {%1, %2};\" : \"=l\"(desc_b" + suffix +
             ") : \"r\"(_b_lo" + suffix + "), \"r\"(" + std::string(desc_b_hi_hex) + "));");
      }

      // Build merged PTX with all WGMMAs in one asm volatile block
      // Operand layout: %0..%(n_out_regs-1) = "+f" accumulators
      //                 %(n_out_regs+2*i), %(n_out_regs+2*i+1) = "l" desc_a_i, desc_b_i
      std::string out_str;
      for (int i = 0; i < nAccRegs; i++) {
        if (i) out_str += ", ";
        out_str += "%" + std::to_string(i);
      }

      emit("asm volatile(\"wgmma.fence.sync.aligned;\");");

      // Emit all WGMMAs in a single asm volatile block
      emit("asm volatile(");
      indent();
      for (size_t gi = 0; gi < chunkLen; gi++) {
        int a_idx = nAccRegs + 2 * (int)gi;
        int b_idx = a_idx + 1;
        // Partial accumulation: the chunk's first wgmma drops the (stale)
        // temporary accumulator via scale-d=0 (fp8 imm layout: scale-d,
        // imm-scale-a, imm-scale-b).
        std::string imm = (needsPartialAcc && gi == 0) ? ", 0, 1, 1;" : immSuffix;
        std::string ptx = "wgmma.mma_async.sync.aligned.m" + std::to_string(wgmma_m) +
                          "n" + std::to_string(wgmma_n) + "k" + std::to_string(wgmma_k) +
                          "." + cElem + "." + aElem + "." + bElem + " {" + out_str +
                          "}, %" + std::to_string(a_idx) + ", %" + std::to_string(b_idx) +
                          imm;
        std::string suffix = (gi + 1 < chunkLen) ? "\\n\\t" : "";
        emit("\"" + ptx + suffix + "\"");
      }

      // Output operands: "+f" (f32) / "+r" (s32/packed-f16) accumulator regs.
      std::string outOps;
      int accIdxBase = f16Acc ? acc_base / 2 : acc_base;
      for (int i = 0; i < nAccRegs; i++) {
        if (i) outOps += ", ";
        if (i % 8 == 0 && i > 0) outOps += "\n              ";
        if (needsPartialAcc)
          outOps += "\"+f\"(" + paccVar + "[" + std::to_string(i) + "])";
        else
          outOps += accConstr + var + "[" +
                    std::to_string(accIdxBase + i) + "])";
      }
      emit(": " + outOps);

      // Input operands: "l" for all descriptor pairs
      std::string inOps;
      for (size_t gi = 0; gi < chunkLen; gi++) {
        if (gi) inOps += ", ";
        std::string suffix = "_" + std::to_string(gi);
        inOps += "\"l\"(desc_a" + suffix + "), \"l\"(desc_b" + suffix + ")";
      }
      emit(": " + inOps);
      dedent();
      emit(");");
      emit("asm volatile(\"wgmma.commit_group.sync.aligned;\");");
      if (needsPartialAcc) {
        // The temporary accumulator must be final before the precise adds.
        emit("asm volatile(\"wgmma.wait_group.sync.aligned 0;\");");
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(n_out_regs) + "; _i++)");
        emit("  asm(\"add.f32 %0, %0, %1;\" : \"+f\"(" + var + "[" +
             std::to_string(acc_base) + " + _i]) : \"f\"(" + paccVar + "[_i]));");
      }
      dedent();
      emit("}");
      } // end chunk
    }
    bool isAsync = op.getIsAsync();
    if (!isAsync)
      emit("asm volatile(\"wgmma.wait_group.sync.aligned 0;\");");
  } else {
    // RS mode: A in registers, B in shared memory
    auto aVar = getVar(a);
    emit("// RS mode: A from registers, B from shared memory");
    emit("uint32_t smem_addr_b = (unsigned)__cvta_generic_to_shared(" + bVar + ");");
    emitBlank();

    bool isTf32A = (aElem == "tf32");
    int a_regs_per_ktile = (16 * wgmma_k) / 32;
    int a_packed_per_ktile;
    // A register operand spans m_tiles_per_wg M-subtiles. The register array
    // aVar is laid out [m_subtile][k_tile][reg]; each M-subtile must be packed
    // into its own _a_packed slice so the per-m_rel wgmma below indexes the
    // correct P rows (without the m-subtile offset, all output M-subtiles would
    // reuse the first subtile's A operand).
    int a_regs_per_mtile = n_k_tiles * a_regs_per_ktile;
    if (isTf32A) {
      // tf32: each A register is 32-bit, pass directly as uint32
      a_packed_per_ktile = a_regs_per_ktile;
      int a_packed_per_mtile = n_k_tiles * a_packed_per_ktile;
      emit("uint32_t _a_packed[" + std::to_string(m_tiles_per_wg * a_packed_per_mtile) + "];");
      emit("#pragma unroll");
      emit("for (int _m = 0; _m < " + std::to_string(m_tiles_per_wg) + "; _m++) {");
      indent();
      emit("#pragma unroll");
      emit("for (int _k = 0; _k < " + std::to_string(n_k_tiles) + "; _k++) {");
      indent();
      emit("#pragma unroll");
      emit("for (int _j = 0; _j < " + std::to_string(a_packed_per_ktile) + "; _j++) {");
      indent();
      emit("float _f = " + aVar + "[_m * " + std::to_string(a_regs_per_mtile) +
           " + _k * " + std::to_string(a_regs_per_ktile) + " + _j];");
      emit("_a_packed[_m * " + std::to_string(a_packed_per_mtile) +
           " + _k * " + std::to_string(a_packed_per_ktile) + " + _j] = *(uint32_t*)&_f;");
      dedent();
      emit("}");
      dedent();
      emit("}");
      dedent();
      emit("}");
    } else if ((isFp8 || aElem == "bf16" || aElem == "f16") &&
               packedU32Convert.count(a)) {
      // Fused: the convert already produced A as packed uint32 words in exactly
      // this wgmma's A-operand order (see emitConvertLayout packed paths: fp8
      // warp-shuffle = 4 elems/word; bf16/f16 non-pow2 register-local opt-in =
      // 2 elems/word). Alias _a_packed onto it directly, skipping the
      // unpack/re-pack round-trip and its duplicate staging registers.
      a_packed_per_ktile = a_regs_per_ktile / (isFp8 ? 4 : 2);
      emit("uint32_t* _a_packed = " + aVar + ";");
    } else {
      // f16/bf16: pack 2 16-bit values per uint32 (kWidth=2).
      // fp8 (e4m3/e5m2): pack 4 8-bit values per uint32 (kWidth=4). The wgmma
      // fp8 A fragment supplies a0..a3 each holding 4 fp8; cvt100/aVar already
      // hold the fp8 elements in linear register order (DotOperand kWidth=4), so
      // every 4 consecutive elements form one packed uint32.
      int elems_per_reg = (isFp8 || isInt8) ? 4 : 2;
      int elem_bits = (isFp8 || isInt8) ? 8 : 16;
      a_packed_per_ktile = a_regs_per_ktile / elems_per_reg;
      int a_packed_per_mtile = n_k_tiles * a_packed_per_ktile;
      emit("uint32_t _a_packed[" + std::to_string(m_tiles_per_wg * a_packed_per_mtile) + "];");
      emit("#pragma unroll");
      emit("for (int _m = 0; _m < " + std::to_string(m_tiles_per_wg) + "; _m++) {");
      indent();
      emit("#pragma unroll");
      emit("for (int _k = 0; _k < " + std::to_string(n_k_tiles) + "; _k++) {");
      indent();
      emit("#pragma unroll");
      emit("for (int _j = 0; _j < " + std::to_string(a_packed_per_ktile) + "; _j++) {");
      indent();
      auto aCudaType = getCUDAType(aElemType);
      std::string baseIdx = "_m * " + std::to_string(a_regs_per_mtile) + " + _k * " +
                            std::to_string(a_regs_per_ktile) + " + _j * " +
                            std::to_string(elems_per_reg);
      std::string packExpr;
      const char *castTy = isFp8 ? "uint8_t" : "uint16_t";
      for (int e = 0; e < elems_per_reg; e++) {
        emit(aCudaType + " _e" + std::to_string(e) + " = " + aVar + "[" + baseIdx +
             " + " + std::to_string(e) + "];");
        std::string term = "((uint32_t)*(" + std::string(castTy) + "*)&_e" + std::to_string(e) + ")";
        if (e) term += " << " + std::to_string(e * elem_bits);
        if (e) packExpr += " | ";
        packExpr += term;
      }
      emit("_a_packed[_m * " + std::to_string(a_packed_per_mtile) +
           " + _k * " + std::to_string(a_packed_per_ktile) + " + _j] = " + packExpr + ";");
      dedent();
      emit("}");
      dedent();
      emit("}");
      dedent();
      emit("}");
    }
    emitBlank();

    int elem_bytes_b = (bElem == "e4m3" || bElem == "e5m2" || bElem == "s8")
                           ? 1
                           : ((bElem == "f16" || bElem == "bf16") ? 2 : 4);
    bool b_is_transposed = bEnc && bEnc.getTransposed();

    // N-split warpgroups (warpsPerCTA=[4,2] etc.): each warpgroup reads its
    // own N slab of the B tile (same derivation as the SS path).
    if (n_warp_groups_n > 1) {
      int eRow_b_rs = std::max(1, swizzle_b / elem_bytes_b);
      int b_stride_dim_rs =
          b_is_transposed ? bMemDesc.getShape()[1] : bMemDesc.getShape()[0];
      int b_wg_stride;
      if (b_is_transposed) {
        b_wg_stride = N_wg * eRow_b_rs * elem_bytes_b;
      } else if (N_wg % eRow_b_rs == 0) {
        b_wg_stride = (N_wg / eRow_b_rs) * b_stride_dim_rs * eRow_b_rs * elem_bytes_b;
      } else if (n_warp_groups_n * N_wg <= eRow_b_rs) {
        b_wg_stride = N_wg * elem_bytes_b;
      } else {
        std::string msg =
            "[emit_cuda] unsupported N-split RS wgmma B offset: N_wg=" +
            std::to_string(N_wg) + " eRow_b=" + std::to_string(eRow_b_rs);
        op->emitError(msg);
        if (!emitFailed) {
          emitFailed = true;
          emitErrorMsg = msg;
        }
        return;
      }
      emit("const int _b_wgoff = (_wg_m / " + std::to_string(n_warp_groups_m) +
           ") * " + std::to_string(b_wg_stride) + ";");
    }

    emit("asm volatile(\"wgmma.fence.sync.aligned;\");");
    emitBlank();

    for (int m_rel = 0; m_rel < m_tiles_per_wg; m_rel++) {
      int acc_base = m_rel * n_out_regs;
      for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
        int b_byte_offset;
        if (b_is_transposed) {
          // B physical [N(stride), K(contig)], swizzled along K. One swizzle
          // atom holds eRow_b K-elements; advancing past it jumps by the
          // leading-dim stride (b_stride_dim * eRow_b elements), same as the
          // SS-mode path above.
          int eRow_b = swizzle_b / elem_bytes_b;
          int b_k_elem = k_tile * wgmma_k;
          if (eRow_b > 0) {
            int b_stride_dim_rs = bMemDesc.getShape()[1];
            b_byte_offset = ((b_k_elem / eRow_b) * (b_stride_dim_rs * eRow_b) +
                             (b_k_elem % eRow_b)) * elem_bytes_b;
          } else {
            b_byte_offset = b_k_elem * elem_bytes_b;
          }
        } else {
          int b_row_stride = std::min(N_block, swizzle_b / elem_bytes_b) * elem_bytes_b;
          b_byte_offset = k_tile * wgmma_k * b_row_stride;
        }
        int b_stride_dim = b_is_transposed ? bMemDesc.getShape()[1] : bMemDesc.getShape()[0];
        uint64_t desc_b = getWGMMADescTemplate(swizzle_b, b_stride_dim);
        int a_packed_base =
            m_rel * (n_k_tiles * a_packed_per_ktile) + k_tile * a_packed_per_ktile;

        // PTX for RS mode
        std::string out_str;
        for (int i = 0; i < nAccRegs; i++) {
          if (i) out_str += ", ";
          out_str += "%" + std::to_string(i);
        }
        std::string a_reg_str;
        for (int i = 0; i < a_packed_per_ktile; i++) {
          if (i) a_reg_str += ", ";
          a_reg_str += "%" + std::to_string(nAccRegs + i);
        }
        int b_idx = nAccRegs + a_packed_per_ktile;
        // tf32 wgmma (like fp8/int8) takes no trans immediates.
        std::string rsImmSuffix = isInt8 ? ", 1;"
                                  : (isFp8 || isTf32A)
                                      ? ", 1, 1, 1;"
                                      : ", 1, 1, 1, " + std::to_string(trans_b) + ";";
        std::string ptx = "wgmma.mma_async.sync.aligned.m" + std::to_string(wgmma_m) +
                          "n" + std::to_string(wgmma_n) + "k" + std::to_string(wgmma_k) +
                          "." + cElem + "." + aElem + "." + bElem + " {" + out_str +
                          "}, {" + a_reg_str + "}, %" + std::to_string(b_idx) + rsImmSuffix;

        emit("{");
        indent();
        uint32_t rs_desc_b_lo = (uint32_t)(desc_b & 0xFFFFFFFF);
        uint32_t rs_desc_b_hi = (uint32_t)(desc_b >> 32);
        char rs_b_lo_hex[12], rs_b_hi_hex[12];
        snprintf(rs_b_lo_hex, sizeof(rs_b_lo_hex), "0x%08Xu", rs_desc_b_lo);
        snprintf(rs_b_hi_hex, sizeof(rs_b_hi_hex), "0x%08Xu", rs_desc_b_hi);
        std::string rsBOff = std::to_string(b_byte_offset);
        if (n_warp_groups_n > 1)
          rsBOff += " + _b_wgoff";
        // 18-bit mask: strip CGA CTA-rank bits from cvta'd smem address.
        emit("uint32_t _b_lo = (((smem_addr_b + " + rsBOff +
             ") & 0x3FFFFu) >> 4) | " + std::string(rs_b_lo_hex) + ";");
        emit("uint64_t desc_b;");
        emit("asm(\"mov.b64 %0, {%1, %2};\" : \"=l\"(desc_b) : \"r\"(_b_lo), \"r\"(" +
             std::string(rs_b_hi_hex) + "));");
        emit("asm volatile(");
        indent();
        emit("\"" + ptx + "\"");
        std::string outOps;
        int accIdxBase = f16Acc ? acc_base / 2 : acc_base;
        for (int i = 0; i < nAccRegs; i++) {
          if (i) outOps += ", ";
          if (i % 8 == 0 && i > 0) outOps += "\n              ";
          outOps += accConstr + var + "[" +
                    std::to_string(accIdxBase + i) + "])";
        }
        emit(": " + outOps);
        std::string inOps;
        for (int i = 0; i < a_packed_per_ktile; i++) {
          if (i) inOps += ", ";
          inOps += "\"r\"(_a_packed[" + std::to_string(a_packed_base + i) + "])";
        }
        inOps += ", \"l\"(desc_b)";
        emit(": " + inOps);
        dedent();
        emit(");");
        dedent();
        emit("}");
      }
    }
    emit("asm volatile(\"wgmma.commit_group.sync.aligned;\");");
    bool isAsync = op.getIsAsync();
    if (!isAsync)
      emit("asm volatile(\"wgmma.wait_group.sync.aligned 0;\");");
  }
  dedent();
  emit("}");
  if (resIsF16) {
    // Materialize the f16-typed result expected by consumers. If the dot was
    // async, force a wait first (degrades overlap, but f16-acc dots are not on
    // pipelined hot paths) so the accumulator registers are final first.
    if (op.getIsAsync())
      emit("asm volatile(\"wgmma.wait_group.sync.aligned 0;\"); // f16-acc: sync before convert");
    std::string outV = newVar("wgmmah");
    emit("__half " + outV + "[" + std::to_string(total_acc) + "];");
    emit("#pragma unroll");
    if (f16Acc) {
      // Unpack the two f16 lanes of each 32-bit accumulator register.
      emit("for (int _j = 0; _j < " + std::to_string(total_acc / 2) + "; _j++) {");
      emit("  " + outV + "[2*_j] = __ushort_as_half((uint16_t)(" + var + "[_j] & 0xffffu));");
      emit("  " + outV + "[2*_j+1] = __ushort_as_half((uint16_t)(" + var + "[_j] >> 16));");
      emit("}");
    } else {
      emit("for (int _i = 0; _i < " + std::to_string(total_acc) + "; _i++) " +
           outV + "[_i] = __float2half(" + var + "[_i]);");
    }
    valueToVar[result] = outV;
  }
}

void CUDACodeGen::emitWarpGroupDotWait(ttng::WarpGroupDotWaitOp op) {
  // Map results to input operands (wait is a synchronization point).
  // WarpGroupDotWaitOp is variadic: result #i is a pass-through of operand #i
  // (#0 = the async dot result, the rest are the tensors held live across the
  // wait, e.g. the loop-carried accumulator). Each result must inherit its own
  // operand's variable, not operand(0)'s.
  for (auto [result, operand] :
       llvm::zip(op.getResults(), op.getOperands()))
    valueToVar[result] = getVar(operand);
  int pendings = op.getPendings();
  emit("asm volatile(\"wgmma.wait_group.sync.aligned " + std::to_string(pendings) + ";\");");
  // Barrier after WGMMA wait: ensures all warp groups have finished reading
  // shared memory via WGMMA before any warp group starts cp.async that
  // overwrites the next shared memory stage. Without this, warp group 0 could
  // race ahead and start cp.async while warp group 1 is still in WGMMA.
  // Matches the bar.sync 0 that Triton's LLVM backend emits at this point.
  if (numWarps > 4) {
    blockSync();
  }
}

// emitcuda.named_barrier — inserted by WgPingpongPass to realize level6's
// inter-warpgroup ping-pong scheduling. Attributes:
//   barrier_id : i32  — PTX named-barrier index (0..15)
//   num_threads: i32  — number of threads participating in the barrier
//   is_arrive  : bool — true => bar.arrive (non-blocking), false => bar.sync
// bar.arrive lets a warpgroup signal the *other* warpgroup's barrier without
// blocking; bar.sync blocks until num_threads have arrived. Together two
// consumer warpgroups stagger so one's softmax (CUDA cores) overlaps the
// other's WGMMA (tensor cores).
void CUDACodeGen::emitNamedBarrier(mlir::Operation *op) {
  auto idAttr = op->getAttrOfType<mlir::IntegerAttr>("barrier_id");
  auto ntAttr = op->getAttrOfType<mlir::IntegerAttr>("num_threads");
  auto arriveAttr = op->getAttrOfType<mlir::BoolAttr>("is_arrive");
  int barId = idAttr ? (int)idAttr.getInt() : 0;
  int numThreads = ntAttr ? (int)ntAttr.getInt() : 0;
  bool isArrive = arriveAttr ? arriveAttr.getValue() : false;
  const char *opcode = isArrive ? "bar.arrive" : "bar.sync";
  emit("asm volatile(\"" + std::string(opcode) + " " +
       std::to_string(barId) + ", " + std::to_string(numThreads) + ";\");");
}

void CUDACodeGen::emitFenceAsyncShared(ttng::FenceAsyncSharedOp op) {
  if (hasPendingCpAsync) {
    emit("asm volatile(\"cp.async.commit_group;\");");
    emit("asm volatile(\"cp.async.wait_group 0;\");");
    blockSync();
    hasPendingCpAsync = false;
  }
  emit("asm volatile(\"fence.proxy.async.shared::cta;\");");
  blockSync();
}
void CUDACodeGen::emitInitBarrier(ttng::InitBarrierOp op) {
  auto barVar = getVar(op.getAlloc());
  int count = op.getCount();
  emit("if (threadIdx.x == 0) {");
  indent();
  emit("asm volatile(\"mbarrier.init.shared::cta.b64 [%0], %1;\" :: \"r\"((unsigned)__cvta_generic_to_shared(" +
       barVar + ")), \"r\"(" + std::to_string(count) + "));");
  dedent();
  emit("}");
  // Gap B (num_ctas>1): the cooperative cluster TMA multicast has each CTA's TMA
  // complete_tx land on the PEER CTA's mbarrier (mbarrier.arrive.shared::cluster
  // / the .multicast::cluster copy). For that remote arrive to be well-defined,
  // every cluster CTA must have FINISHED initializing its mbarriers and the init
  // writes must be made visible cluster-wide BEFORE any cross-CTA traffic. Mirror
  // gemm_06: `fence.mbarrier_init.release.cluster` after init, then a cluster-wide
  // barrier (barrier.cluster.arrive/wait) instead of the CTA-local __syncthreads.
  // Init runs once per kernel, so the extra cluster barrier is off the hot path.
  // num_ctas==1 is byte-identical (keeps the plain CTA blockSync).
  if (numCtas > 1) {
    emit("asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");");
    emit("asm volatile(\"barrier.cluster.arrive;\\nbarrier.cluster.wait;\" ::: "
         "\"memory\");");
  } else {
    blockSync();
  }
}

void CUDACodeGen::emitWaitBarrier(ttng::WaitBarrierOp op) {
  auto barVar = getVar(op.getAlloc());
  auto phaseVar = getVar(op.getPhase());
  // EXPERIMENT (env-gated, default off -> byte-identical): inside a WS partition
  // the pre-wait bar.sync is redundant for an mbarrier consumer wait — the
  // try_wait.parity spin is itself the acquire fence and the producer/consumer
  // mbarrier protocol already orders smem. gemm_06's math loop has no per-K-iter
  // sync. Skipping it removes one named-barrier per K-iter from the hot loop.
  static const bool leanWsSync = (getenv("TRITON_WS_LEAN_SYNC") != nullptr) &&
                                 (std::string(getenv("TRITON_WS_LEAN_SYNC")) == "1");
  if (!(leanWsSync && wsSyncBarrierId > 0))
    blockSync();
  emit("{");
  indent();
  emit("unsigned _mbar_addr = (unsigned)__cvta_generic_to_shared(" + barVar + ");");
  emit("unsigned _phase = (unsigned)" + phaseVar + ";");
  emit("asm volatile(");
  indent();
  emit("\"{\\n\"");
  emit("\".reg .pred P1;\\n\"");
  emit("\"WAIT_LOOP_%=:\\n\"");
  emit("\"mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\\n\"");
  emit("\"@!P1 bra WAIT_LOOP_%=;\\n\"");
  emit("\"}\\n\"");
  emit(":: \"r\"(_mbar_addr), \"r\"((int)_phase));");
  dedent();
  emit("}");
}

void CUDACodeGen::emitArriveBarrier(ttng::ArriveBarrierOp op) {
  auto barVar = getVar(op.getAlloc());
  // mbarrier.arrive decrements the pending-arrival count by exactly `count`,
  // so it must be issued by a SINGLE thread — not once per thread in the
  // warp(group). Stock Triton (ArriveBarrierOpConversion) elects thread 0 via
  // `@(threadId==0)` and appends the count only when >1. We mirror that using
  // the partition-local `tid` that emitWarpSpecialize shadows (so each WS
  // consumer region elects its own first thread); in non-WS code `tid` is the
  // raw threadIdx.x, electing global thread 0. Issuing it unguarded over an
  // N-thread warpgroup over-arrives an init-count-1 barrier and corrupts its
  // phase ("Barrier error: missing wait").
  std::string pred = "tid == 0";
  if (op.getPred())
    pred += " && (" + getVar(op.getPred()) + ")";
  uint32_t count = op.getCount();

  // Gap B: cluster-broadcast empty arrive. When the barrier's shared tile is
  // REPLICATED across a subset of cluster CTAs (free in its layout's kBlock
  // input, i.e. allocated with a replicated cga_layout) the arrive must reach
  // EVERY CTA's copy of the barrier — not just the local one. In a cluster GEMM
  // the representative CTA delivers the multicast operand into all peers' smem
  // and may only recycle that slot once ALL peers' consumers have released it;
  // a CTA-local arrive would let the representative overwrite a peer's still-live
  // buffer (corruption / phase desync / deadlock). Mirrors the stock pipeliner's
  // cluster-broadcast empty arrives. Each consumer maps the barrier address into
  // every peer of its broadcast group (mapa.shared::cluster) and issues a remote
  // mbarrier.arrive.shared::cluster; the barrier's init count must therefore be
  // scaled by the number of arriving CTAs.
  uint32_t bcastMask = 0;
  if (numCtas > 1) {
    if (auto barTy = dyn_cast<ttg::MemDescType>(op.getAlloc().getType())) {
      auto *ctx = barTy.getContext();
      auto kBlock = mlir::StringAttr::get(ctx, "block");
      bcastMask = ttg::toLinearLayout(barTy).getFreeVariableMasks().lookup(
                      kBlock) &
                  (numCtas - 1);
    }
  }

  // CUDA inline asm has no portable predicate-register constraint, so guard the
  // single-thread arrival with a C++ `if` (the same idiom emit uses for the
  // producer's expect_tx) rather than a PTX `@p` predicate operand.
  emit("if (" + pred + ") {");
  indent();
  if (bcastMask != 0) {
    uint32_t fixedMask = (~bcastMask) & (numCtas - 1);
    std::string cnt = count > 1 ? (", " + std::to_string(count)) : "";
    emit("unsigned _ba = (unsigned)__cvta_generic_to_shared(" + barVar + ");");
    emit("for (unsigned _cta = 0u; _cta < " + std::to_string(numCtas) +
         "u; ++_cta) {");
    indent();
    emit("if ((_cta & " + std::to_string(fixedMask) + "u) == (_cta_rank & " +
         std::to_string(fixedMask) + "u)) {");
    indent();
    emit("unsigned _rb;");
    emit("asm volatile(\"mapa.shared::cluster.u32 %0, %1, %2;\" : \"=r\"(_rb) "
         ": \"r\"(_ba), \"r\"(_cta));");
    emit("asm volatile(\"mbarrier.arrive.shared::cluster.b64 _, [%0]" + cnt +
         ";\" :: \"r\"(_rb)" + (count > 1 ? (", \"n\"(" + std::to_string(count) +
                                            ")")
                                         : std::string()) +
         ");");
    dedent();
    emit("}");
    dedent();
    emit("}");
  } else if (count > 1)
    emit("asm volatile(\"mbarrier.arrive.shared::cta.b64 _, [%0], %1;\" :: "
         "\"r\"((unsigned)__cvta_generic_to_shared(" + barVar + ")), \"n\"(" +
         std::to_string(count) + "));");
  else
    emit("asm volatile(\"mbarrier.arrive.shared::cta.b64 _, [%0];\" :: "
         "\"r\"((unsigned)__cvta_generic_to_shared(" + barVar + ")));");
  dedent();
  emit("}");
}

void CUDACodeGen::emitBarrierExpect(ttng::BarrierExpectOp op) {
  blockSync();  // MUST be before expect_tx
  auto barVar = getVar(op.getAlloc());
  int bytes = op.getSize();
  auto predVar = getVar(op.getPred());
  emit("if (threadIdx.x == 0) {");
  indent();
  emit("if (" + predVar + ") {");
  indent();
  emit("asm volatile(\"mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\" :: \"r\"((unsigned)__cvta_generic_to_shared(" +
       barVar + ")), \"r\"(" + std::to_string(bytes) + "));");
  dedent();
  emit("}");
  dedent();
  emit("}");
}

void CUDACodeGen::emitInvalBarrier(ttng::InvalBarrierOp op) {
  auto barVar = getVar(op.getAlloc());
  emit("if (threadIdx.x == 0)");
  emit("    asm volatile(\"mbarrier.inval.shared::cta.b64 [%0];\" :: \"r\"((unsigned)__cvta_generic_to_shared(" +
       barVar + ")));");
}

// gluon mbarrier.sync_cluster_init() lowers to three standalone ops:
// fence_mbarrier_init_release_cluster, cluster_arrive(relaxed), cluster_wait.
// They make every cluster CTA's mbarrier-init writes visible cluster-wide before
// any cross-CTA TMA multicast traffic (whose complete_tx lands on PEER mbarriers).
// All threads must execute the cluster barrier (it is a CTA-wide rendezvous that
// also synchronizes across the cluster), so these are emitted unguarded.
void CUDACodeGen::emitFenceMBarrierInitReleaseCluster(
    ttng::FenceMBarrierInitReleaseClusterOp op) {
  emit("asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");");
}

void CUDACodeGen::emitClusterArrive(ttng::ClusterArriveOp op) {
  if (op.getRelaxed())
    emit("asm volatile(\"barrier.cluster.arrive.relaxed;\" ::: \"memory\");");
  else
    emit("asm volatile(\"barrier.cluster.arrive;\" ::: \"memory\");");
}

void CUDACodeGen::emitClusterWait(ttng::ClusterWaitOp op) {
  emit("asm volatile(\"barrier.cluster.wait;\" ::: \"memory\");");
}

// Compute the per-copy plan for a TMA global<->local transfer of a swizzled
// NVMMA shared tile. The TMA bounding-box inner dimension must be <= the
// swizzle width (see getTMABlockShape), so a tile whose fast dimension exceeds
// the swizzle width is delivered as several "messages"; we issue one
// cp.async.bulk per message. The per-message shared-memory ELEMENT offset and
// the per-dimension global COORDINATE offsets follow the swizzled layout
// exactly — computed with the same linear layouts as the reference TMA
// lowering. Crucially, these layouts are built over the SMEM rank
// (shapePerCTA.size()), which can be SMALLER than the descriptor coordinate
// rank (op.getCoord().size()) for ragged/persistent TMA whose descriptors carry
// extra degenerate leading dimensions. The returned coordOff is indexed by smem
// dim and applied to the innermost coords.
TMACopyPlan CUDACodeGen::computeTMACopyPlan(ttg::MemDescType smemTy,
                                            int coordRank) {
  TMACopyPlan plan;
  plan.numCopies = 1;
  plan.smemRank = coordRank;
  auto *ctx = smemTy.getContext();
  if (auto nvmmaShared =
          dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(smemTy.getEncoding())) {
    auto shapePerCTA = ttg::getShapePerCTA(smemTy);
    auto blockShape = ttng::getTMABlockShape(smemTy, /*packedSize=*/true);
    int smemRank = (int)shapePerCTA.size();
    // Native non-pow2 WGMMA N (omni-style n80): a smem tile whose row count is
    // non-pow2 (e.g. q_smem[80,128]) has a perfectly valid TMA box (box dims may
    // be any value <= 256), but the LinearLayout copy-plan machinery below
    // requires pow2 out-dim sizes. When the non-pow2 dim is a SINGLE TMA message
    // (blockShape[d] == shapePerCTA[d] — the whole dim fits in one box), it
    // contributes nothing to numCopies / coordOff: the per-message GLOBAL
    // coordinate offsets of the OTHER dims (swizzle column groups) are
    // independent of that dim's exact size. So compute the plan on a pow2
    // stand-in (the dim rounded down to the nearest pow2).
    //
    // The per-message SHARED-MEMORY element offset, however, DOES depend on the
    // exact size of the substituted dim. NVMMA swizzled smem is column-group
    // major: the contiguous (swizzle) dim is split into W-wide groups and those
    // groups are the OUTERMOST stride, so colgroup_stride == (product of all
    // other dims) * W. Substituting a non-pow2 outer dim from R down to P shrinks
    // that stride by P/R (e.g. q_smem[72,128] colgroup1 lands at 72*64=4608, but
    // the P=64 stand-in computes 64*64=4096 — a 512-elem mismatch vs. what the
    // WGMMA B-descriptor LBO expects, corrupting every K>=swizzle_width step).
    // Messages only ever vary along NON-substituted dims (substituted dims are a
    // single message), so each substituted outer dim scales every smem offset by
    // exactly R/P; rescale the stand-in plan's shMemElemOffset accordingly.
    {
      bool needSub = false;
      llvm::SmallVector<int64_t> subShape(shapePerCTA.begin(),
                                          shapePerCTA.end());
      for (int d = 0; d < smemRank; ++d) {
        if (!llvm::isPowerOf2_64(shapePerCTA[d]) &&
            blockShape[d] == shapePerCTA[d]) {
          int64_t p = 1;
          while (p * 2 <= shapePerCTA[d])
            p *= 2;
          subShape[d] = p;
          needSub = true;
        }
      }
      if (needSub) {
        auto subTy = ttg::MemDescType::get(
            ctx, subShape, smemTy.getElementType(), smemTy.getEncoding(),
            smemTy.getMemorySpace(), smemTy.getMutableMemory(), subShape);
        auto subPlan = computeTMACopyPlan(subTy, coordRank);
        for (auto &off : subPlan.shMemElemOffset) {
          for (int d = 0; d < smemRank; ++d) {
            if (subShape[d] != shapePerCTA[d]) {
              // off is divisible by subShape[d] (it is a multiple of the
              // colgroup stride, which carries subShape[d] as a factor).
              off = off / subShape[d] * shapePerCTA[d];
            }
          }
        }
        return subPlan;
      }
    }
    plan.smemRank = smemRank;
    auto kMsg = mlir::StringAttr::get(ctx, "msg");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    auto outDimNames = standardOutDimNames(ctx, smemRank);
    LinearLayout msgToOffset = LinearLayout::empty();
    for (int dim = 0; dim < smemRank; ++dim)
      msgToOffset *= LinearLayout::strided1D(shapePerCTA[dim] / blockShape[dim],
                                             blockShape[dim], kMsg,
                                             outDimNames[dim]);
    auto ctaLayout = ttg::getCGALayout(smemTy.getEncoding());
    for (int i = 0; i < smemRank; ++i) {
      auto dim = ctaLayout.getCTAOrder()[i];
      msgToOffset *= LinearLayout::identity1D(ctaLayout.getCTASplitNum()[dim],
                                              kBlock, outDimNames[dim]);
    }
    auto smemLayout = ttg::toLinearLayout(smemTy);
    auto msgToShared = msgToOffset.invertAndCompose(smemLayout);
    plan.numCopies = msgToOffset.getInDimSize(kMsg);
    for (int c = 0; c < plan.numCopies; ++c) {
      auto sh = msgToShared.apply({{kMsg, c}, {kBlock, 0}});
      plan.shMemElemOffset.push_back(sh[0].second);
      auto off = msgToOffset.apply({{kMsg, c}, {kBlock, 0}});
      SmallVector<int> perDim(smemRank, 0);
      for (auto &kv : off) {
        for (int d = 0; d < smemRank; ++d)
          if (kv.first == outDimNames[d])
            perDim[d] = kv.second;
      }
      plan.coordOff.push_back(perDim);
    }
    // CGA: per-block-bit global coordinate deltas (applied at runtime as
    // ((_cta_rank >> bit) & 1) * delta, mirroring {kBlock, ctaId} in the
    // reference lowering).
    int blockBits = llvm::Log2_32(msgToOffset.getInDimSize(kBlock));
    for (int b = 0; b < blockBits; ++b) {
      auto off = msgToOffset.apply({{kMsg, 0}, {kBlock, 1 << b}});
      auto base = msgToOffset.apply({{kMsg, 0}, {kBlock, 0}});
      SmallVector<int> perDim(smemRank, 0);
      for (size_t k = 0; k < off.size(); ++k)
        for (int d = 0; d < smemRank; ++d)
          if (off[k].first == outDimNames[d])
            perDim[d] = off[k].second - base[k].second;
      plan.blockCoordDelta.push_back(perDim);
    }
    // CGA multicast candidate: bits of the cluster CTA rank over which the
    // destination smem tile is REPLICATED (free in smemLayout's kBlock input).
    // Mirrors the reference AsyncTMACopyGlobalToLocalOpConversion:
    //   maskCGABroadcast = smemLayout.getFreeVariableMasks().lookup(kBlock).
    // A nonzero mask means the same shared tile is delivered to several CTAs and
    // the TMA may be issued once with .multicast::cluster.
    plan.numCTAs = numCtas;
    if (numCtas > 1)
      plan.cgaBroadcastMask =
          smemLayout.getFreeVariableMasks().lookup(kBlock) & (numCtas - 1);
  } else {
    plan.shMemElemOffset.push_back(0);
    plan.coordOff.push_back(SmallVector<int>(coordRank, 0));
  }
  return plan;
}

void CUDACodeGen::emitAsyncTMACopyG2L(ttng::AsyncTMACopyGlobalToLocalOp op) {
  auto descVar = getVar(op.getDesc());
  // Device-built descriptors are CUtensorMap* (pointer into global scratch);
  // host descriptors are grid_constant CUtensorMap params (need &).
  std::string descAddr = pointerDescriptors.contains(op.getDesc())
                             ? ("(uint64_t)" + descVar)
                             : ("(uint64_t)&" + descVar);
  auto smemVar = getVar(op.getResult());
  auto barVar = getVar(op.getBarrier());
  auto predVar = getVar(op.getPred());
  auto coords = op.getCoord();
  int rank = (int)coords.size();
  SmallVector<std::string> coordVar(rank);
  for (int d = 0; d < rank; d++)
    coordVar[d] = getVar(coords[d]);

  auto smemTy = cast<ttg::MemDescType>(op.getResult().getType());
  TMACopyPlan plan = computeTMACopyPlan(smemTy, rank);

  // Gap A: cluster TMA multicast. When the IR marks this copy as multicast AND
  // the destination smem tile is replicated across a subset of cluster CTAs
  // (cgaBroadcastMask != 0), a single CTA — the representative (lowest-id) CTA
  // of each multicast group — issues one cp.async.bulk...multicast::cluster that
  // delivers the tile to every CTA in the group and signals each CTA's mbarrier.
  // Non-representative CTAs are predicated out, eliminating the redundant
  // per-CTA loads of replicated operands (e.g. A in a row-split cluster GEMM).
  // Mirrors AsyncTMACopyGlobalToLocalOpConversion + createTMAMulticastMask.
  //
  // NOTE: the stock pipeliner (LowerLoops::createTMAAsyncLoad) builds these ops
  // with multicast=false for descriptor-load GEMMs, so neither stock-PTX nor a
  // naive emit ever multicasts replicated operands — every CTA redundantly loads
  // them. When the destination tile is replicated (cgaBroadcastMask != 0) a
  // single representative CTA can deliver it to the whole group with one
  // .multicast::cluster message, halving the replicated operand's global
  // traffic. That is correct ONLY if the cluster marches the producer/consumer
  // pipeline in lockstep: the representative's complete_tx lands on REMOTE CTAs'
  // mbarriers at a fixed smem offset, so every group CTA must already have run
  // its expect_tx and be on the matching pipeline phase. emit's current pipeline
  // only has a CTA-local __syncthreads() — no cluster.arrive/wait — so CTAs run
  // asynchronously and the cross-CTA expect_tx/complete_tx ordering races,
  // underflowing the tx count and DEADLOCKING under sustained launches. Until
  // the pipeline gains cluster-scoped barriers (Gap B), multicast is opt-in via
  // TRITON_EMIT_MULTICAST=1 so the default path stays deadlock-free. Measured:
  // even when it does not hang, multicast was ~3% slower than the redundant-load
  // baseline for square num_ctas=2 GEMMs (compute-bound; the representative-only
  // issue serializes the load that two CTAs otherwise fetch in parallel).
  // Honor the IR's PER-OP multicast attribute. The frontend marks ONLY genuinely
  // cluster-replicated operands — whose global coords are cta-rank-INDEPENDENT
  // (e.g. B in a row-split GEMM: off_n = pid_n*BLOCK_N, same on every CTA) — with
  // multicast=true. Gating on the smem layout's cgaBroadcastMask ALONE is wrong:
  // a per-CTA operand (e.g. A, whose global row off_m = ... + cta_rank*CTA_M
  // depends on _cta_rank) can share a replicated smem LAYOUT yet must NOT be
  // multicast — the representative CTA would deliver its OWN rows to all peers
  // (observed: err≈501 on ws_v8). cgaBroadcastMask stays a necessary condition
  // (the destination tile must actually be replicated for one message to reach
  // all peers). The op's multicast attribute — set by the frontend cluster/TMA
  // descriptor setup — is the sole signal; there is no env-var override.
  bool multicast = op.getMulticast() && plan.cgaBroadcastMask != 0;
  // P3 MulticastPass tags rank-DEPENDENT loads that the frontend nonetheless
  // marked multicast as `cooperative`: each CTA issues its OWN half-tile load
  // (off_n + cta_rank*HALF_N) and broadcasts it to all peers, so after every
  // CTA's load each holds the full tile (gemm_06 cooperative split). For these
  // we must NOT representative-gate (every CTA issues), but we DO keep the
  // .multicast::cluster + _mc_mask so the broadcast still reaches all peers.
  bool cooperative = multicast && op->hasAttr("cooperative");
  uint32_t mcPattern = 1, mcFixedBits = 0;
  if (multicast) {
    uint32_t broadcastBits = plan.cgaBroadcastMask;
    int blockBits = llvm::Log2_32(plan.numCTAs);
    mcFixedBits = (~broadcastBits) & (plan.numCTAs - 1);
    for (int i = 0; i < blockBits; ++i)
      if ((mcFixedBits & (1u << i)) == 0)
        mcPattern |= (mcPattern << (1u << i));
  }

  // No blockSync here: an async_tma_copy is always preceded by a barrier_expect
  // (which emits the buffer-free blockSync before its expect_tx). The expect_tx
  // and this cp.async.bulk are both issued by thread 0 and ordered within that
  // thread, so a second CTA-wide barrier between them is redundant. Dropping it
  // halves the per-iteration TMA barriers (K+V: 4 -> 2).
  // COOPERATIVE SPLIT (gemm_06 lines 695-707): when a replicated operand's tile
  // is multicast and decomposes into numCopies TMA boxes evenly divisible by the
  // cluster size, DON'T have one representative CTA issue every box (which idles
  // the peer CTAs' TMA engines and serializes the DRAM read). Instead partition
  // the boxes: CTA r issues boxes [r*copiesPerCta, (r+1)*copiesPerCta), each with
  // .multicast::cluster + full mc_mask so every box still lands in EVERY CTA's
  // smem at its own (issuer-independent) offset. Net: both TMA engines read in
  // parallel, total B DRAM traffic = 1x, single contiguous buffer => one wide
  // WGMMA. expect_tx stays the FULL tile on each CTA's barrier (self boxes via
  // own multicast + peer boxes via peer multicast). Falls back to representative
  // broadcast when the box count is not cluster-divisible.
  bool coopSplit = multicast && plan.numCTAs > 1 &&
                   plan.numCopies >= (int)plan.numCTAs &&
                   (plan.numCopies % (int)plan.numCTAs == 0);
  int copiesPerCta = coopSplit ? plan.numCopies / (int)plan.numCTAs : 0;

  emit("if (threadIdx.x == 0) {");
  indent();
  std::string copyPred = predVar;
  if (multicast) {
    if (!coopSplit && !cooperative) {
      // Only the representative CTA of each multicast group issues the copy.
      copyPred = "(" + predVar + ") && ((_cta_rank & " +
                 std::to_string(plan.cgaBroadcastMask) + "u) == 0)";
    }
    emit("unsigned _mc_mask = " + std::to_string(mcPattern) +
         "u << (_cta_rank & " + std::to_string(mcFixedBits) + "u);");
  }
  emit("if (" + copyPred + ") {");
  indent();
  for (int c = 0; c < plan.numCopies; c++) {
    // Cooperative: box c is owned (issued) by exactly one CTA; the peers receive
    // it via multicast. owner is a compile-time constant per box.
    if (coopSplit) {
      int owner = c / copiesPerCta;
      emit("if (_cta_rank == " + std::to_string(owner) + "u) {");
      indent();
    }
    std::string smemOff = (plan.shMemElemOffset[c] == 0)
                              ? ""
                              : " + " + std::to_string(plan.shMemElemOffset[c]);
    // Coordinate operands are emitted innermost-first (reverse of IR order).
    // The per-copy offsets (smemRank entries) apply to the innermost smemRank
    // coords; any outer ragged coords get no offset. Mirrors the reference
    // AsyncTMACopyGlobalToLocalOpConversion.
    SmallVector<std::string> coordOps(rank);
    std::string braceList;
    for (int i = 0; i < rank; i++) {
      std::string ce = coordVar[rank - 1 - i];
      if (i < plan.smemRank) {
        int d = plan.smemRank - 1 - i;
        int o = plan.coordOff[c][d];
        std::string add;
        if (o != 0)
          add += " + " + std::to_string(o);
        for (size_t bb = 0; bb < plan.blockCoordDelta.size(); ++bb)
          if (plan.blockCoordDelta[bb][d] != 0)
            add += " + ((_cta_rank >> " + std::to_string(bb) + ") & 1) * " +
                   std::to_string(plan.blockCoordDelta[bb][d]);
        if (!add.empty())
          ce = "(" + ce + add + ")";
      }
      coordOps[i] = ce;
      braceList += "%" + std::to_string(i + 2);
      if (i != rank - 1)
        braceList += ", ";
    }
    std::string mcSuffix = multicast ? ".multicast::cluster" : "";
    std::string mcOperand =
        multicast ? (", %" + std::to_string(rank + 3)) : "";
    emit("asm volatile(");
    emit("    \"cp.async.bulk.tensor." + std::to_string(rank) +
         "d.shared::cluster.global.tile.mbarrier::complete_tx::bytes" +
         mcSuffix + " [%0], "
                    "[%1, {" +
         braceList + "}], [%" + std::to_string(rank + 2) + "]" + mcOperand +
         ";\\n\"");
    emit("    :: \"r\"((unsigned)__cvta_generic_to_shared(" + smemVar + smemOff +
         ")),");
    emit("       \"l\"(" + descAddr + "),");
    for (int i = 0; i < rank; i++)
      emit("       \"r\"(" + coordOps[i] + "),");
    if (multicast)
      emit("       \"r\"((unsigned)__cvta_generic_to_shared(" + barVar + ")),");
    else
      emit("       \"r\"((unsigned)__cvta_generic_to_shared(" + barVar + "))");
    if (multicast)
      emit("       \"h\"((unsigned short)_mc_mask)");
    emit(");");
    if (coopSplit) {
      dedent();
      emit("}");
    }
  }
  dedent();
  emit("}");
  dedent();
  emit("}");
}

void CUDACodeGen::emitAsyncTMACopyL2G(ttng::AsyncTMACopyLocalToGlobalOp op) {
  emitTMAStoreLike(op.getDesc(), op.getSrc(), op.getCoord(), "");
}

// ttng.async_tma_reduce: identical addressing to the plain TMA bulk store, but
// the instruction atomically reduces into global memory (mirrors the shared
// convertTMAStoreLikeOp in the reference LoadStoreOpToLLVM.cpp).
void CUDACodeGen::emitAsyncTMAReduce(ttng::AsyncTMAReduceOp op) {
  emitTMAStoreLike(op.getDesc(), op.getSrc(), op.getCoord(),
                   mlir::triton::stringifyDescriptorReduceKind(op.getKind()));
}

void CUDACodeGen::emitTMAStoreLike(Value desc, Value src,
                                   mlir::OperandRange coords,
                                   llvm::StringRef reduceKind) {
  auto descVar = getVar(desc);
  std::string descAddr = pointerDescriptors.contains(desc)
                             ? ("(uint64_t)" + descVar)
                             : ("(uint64_t)&" + descVar);
  auto smemVar = getVar(src);
  int rank = (int)coords.size();
  SmallVector<std::string> coordVar(rank);
  for (int d = 0; d < rank; d++)
    coordVar[d] = getVar(coords[d]);

  auto smemTy = cast<ttg::MemDescType>(src.getType());
  TMACopyPlan plan = computeTMACopyPlan(smemTy, rank);
  std::string inst =
      reduceKind.empty()
          ? "cp.async.bulk.tensor." + std::to_string(rank) +
                "d.global.shared::cta.bulk_group"
          : "cp.reduce.async.bulk.tensor." + std::to_string(rank) +
                "d.global.shared::cta." + reduceKind.str() + ".bulk_group";

  // Region-local leader: the C-output TMA store runs inside warp_specialize
  // consumer regions (threads 128-383), where a raw threadIdx.x==0 guard would
  // never fire — use the region-local `tid`.
  for (int c = 0; c < plan.numCopies; c++) {
    std::string smemOff = (plan.shMemElemOffset[c] == 0)
                              ? ""
                              : " + " + std::to_string(plan.shMemElemOffset[c]);
    // Coordinate operands are emitted innermost-first (reverse of IR order).
    // The per-copy offsets (smemRank entries) apply to the innermost smemRank
    // coords; any outer ragged coords get no offset. Mirrors G2L.
    SmallVector<std::string> coordOps(rank);
    std::string braceList;
    for (int i = 0; i < rank; i++) {
      std::string ce = coordVar[rank - 1 - i];
      if (i < plan.smemRank) {
        int d = plan.smemRank - 1 - i;
        int o = plan.coordOff[c][d];
        std::string add;
        if (o != 0)
          add += " + " + std::to_string(o);
        for (size_t bb = 0; bb < plan.blockCoordDelta.size(); ++bb)
          if (plan.blockCoordDelta[bb][d] != 0)
            add += " + ((_cta_rank >> " + std::to_string(bb) + ") & 1) * " +
                   std::to_string(plan.blockCoordDelta[bb][d]);
        if (!add.empty())
          ce = "(" + ce + add + ")";
      }
      coordOps[i] = ce;
      braceList += "%" + std::to_string(i + 1);
      if (i != rank - 1)
        braceList += ", ";
    }
    emit("if (tid == 0) {");
    indent();
    emit("asm volatile(");
    emit("    \"" + inst + " [%0, {" + braceList + "}], [%" +
         std::to_string(rank + 1) + "];\\n\"");
    emit("    :: \"l\"(" + descAddr + "),");
    for (int i = 0; i < rank; i++)
      emit("       \"r\"(" + coordOps[i] + "),");
    emit("       \"r\"((unsigned)__cvta_generic_to_shared(" + smemVar + smemOff +
         "))");
    emit(");");
    dedent();
    emit("}");
  }
  emit("asm volatile(\"cp.async.bulk.commit_group;\\n\");");
}
void CUDACodeGen::emitTMAStoreWait(ttng::TMAStoreWaitOp op) {
  int pendings = op.getPendings();
  emit("asm volatile(\"cp.async.bulk.wait_group.read " +
       llvm::Twine(pendings).str() + ";\");");
  // async_tma_store_wait has CTA-wide semantics: it gates reuse of the shared
  // buffer the TMA store read from. But cp.async.bulk.wait_group is PER-THREAD,
  // and the TMA store (async_tma_copy_local_to_global) is issued by thread 0
  // only — so on the other threads the wait is a no-op. Without a barrier the
  // non-issuing threads race ahead and overwrite the buffer (the next
  // local_store) while thread 0's TMA is still reading it, corrupting the
  // in-flight store. This breaks EPILOGUE_SUBTILE, where the two subtile halves
  // reuse the same scratch buffer back-to-back. __syncthreads() makes the wait
  // CTA-wide: every thread blocks until thread 0 confirms the TMA reads drained.
  blockSync();
}

// ── Device-side TMA descriptor creation (tl.make_tensor_descriptor) ──────────
// A global_scratch_alloc reserves a per-CTA byte region; the result is a
// CUtensorMap* once filled by tensormap_create.
void CUDACodeGen::emitGlobalScratchAlloc(ttg::GlobalScratchAllocOp op) {
  auto &offs = globalScratchOffsets[op.getOperation()];
  int &cur = globalScratchCursor[op.getOperation()];
  int offset = offs.empty() ? 0
                            : offs[std::min<int>(cur, (int)offs.size() - 1)];
  cur++;
  auto var = newVar("gscratch");
  valueToVar[op.getResult()] = var;
  emit("char* " + var + " = global_scratch + _gscratch_cta + " +
       std::to_string(offset) + ";");
}

// Build a CUtensorMap in shared memory via tensormap.replace.tile.* then copy
// it to the global scratch slot with tensormap.cp_fenceproxy. Mirrors the
// reference TensormapCreateOpConversion (TMAToLLVM.cpp).
void CUDACodeGen::emitTensormapCreate(ttng::TensormapCreateOp op) {
  auto descVar = getVar(op.getDescPtr());        // global scratch char*
  auto gAddrVar = getVar(op.getGlobalAddress()); // base ptr of the tensor
  int rank = op.getRank();
  auto boxDim = op.getBoxDim();
  auto globalDim = op.getGlobalDim();
  auto globalStride = op.getGlobalStride();
  auto elemStride = op.getElementStride();
  int elemType = (int)op.getElemType();
  int interleave = (int)op.getInterleaveLayout();
  int swizzle = (int)op.getSwizzleMode();
  int fill = (int)op.getFillMode();
  // The global-stride /16 workaround is a ptxas bug fix that only applies to
  // PTX ISA <= 8.5 (CUDA <= 12.5). On newer toolkits (e.g. CUDA 12.9 -> PTX
  // 8.7/8.8) applying it would wrongly shrink the stride by 16x and corrupt
  // device-side TMA descriptors. Mirror needsStrideWorkaround in TMAToLLVM.cpp.
  bool strideWorkaround = ptxVersion <= 85;

  emit("// tensormap_create (device-side TMA descriptor)");
  // Elect a region-local leader warp via `tid` (not threadIdx.x): inside a
  // warp_specialize consumer region the C descriptor is built by threads
  // 128-383, whose region-local tid restarts at 0 — a raw threadIdx.x guard
  // would never fire there.
  emit("if (tid < 32) {");
  indent();
  // Per-warpgroup scratch slot: under warp specialization producer/consumers
  // build descriptors concurrently, so each must own a disjoint slot keyed on
  // the GLOBAL warpgroup index (threadIdx.x/128), not the region-local tid.
  if (totalNumWarps > numWarps) {
    emit("char* _tma_slot = _tma_build + (threadIdx.x / 128) * 128;");
  } else {
    // Non-WS: transient 128-byte descriptor scratch in the DYNAMIC region at
    // the current allocation floor (dead after tensormap.cp_fenceproxy +
    // __syncthreads below) — mirrors PTX, avoids static smem (task #44).
    int tmaOff = (sharedMemOffset + 127) & ~127;
    peakSharedMem = std::max(peakSharedMem, tmaOff + 128);
    emit("char* _tma_slot = shared_mem + " + std::to_string(tmaOff) +
         "; // transient descriptor scratch");
  }
  emit("unsigned _tms = (unsigned)__cvta_generic_to_shared(_tma_slot);");
  // Zero-fill the 128-byte descriptor (32 lanes x 4 bytes).
  emit("((int*)_tma_slot)[tid] = 0;");
  emit("__syncwarp();");
  emit("if (tid == 0) {");
  indent();
  auto replace32 = [&](const std::string &field, const std::string &valExpr) {
    emit("asm volatile(\"tensormap.replace.tile." + field +
         ".shared::cta.b1024.b32 [%0], %1;\" :: \"r\"(_tms), \"r\"(" + valExpr +
         "));");
  };
  auto replace32Imm = [&](const std::string &field, int imm) {
    emit("asm volatile(\"tensormap.replace.tile." + field +
         ".shared::cta.b1024.b32 [%0], " + std::to_string(imm) +
         ";\" :: \"r\"(_tms));");
  };
  auto replaceOrd32 = [&](const std::string &field, int ord,
                          const std::string &valExpr) {
    emit("asm volatile(\"tensormap.replace.tile." + field +
         ".shared::cta.b1024.b32 [%0], " + std::to_string(ord) +
         ", %1;\" :: \"r\"(_tms), \"r\"(" + valExpr + "));");
  };
  auto replaceOrd64 = [&](const std::string &field, int ord,
                          const std::string &valExpr) {
    emit("asm volatile(\"tensormap.replace.tile." + field +
         ".shared::cta.b1024.b64 [%0], " + std::to_string(ord) +
         ", %1;\" :: \"r\"(_tms), \"l\"(" + valExpr + "));");
  };
  // global_address (b64), rank (b32 immediate)
  emit("asm volatile(\"tensormap.replace.tile.global_address.shared::cta.b1024."
       "b64 [%0], %1;\" :: \"r\"(_tms), \"l\"((uint64_t)" + gAddrVar + "));");
  replace32Imm("rank", rank - 1);
  for (int i = 0; i < rank; ++i)
    replaceOrd32("box_dim", i, getVar(boxDim[i]));
  for (int i = 0; i < rank; ++i)
    replaceOrd32("global_dim", i, getVar(globalDim[i]));
  // Reference (TensormapCreateOpConversion) emits only rank-1 strides
  // (loop `i + 1 < rank`); the variadic operand may carry more.
  for (int i = 0; i + 1 < rank && i < (int)globalStride.size(); ++i) {
    std::string sv = "(uint64_t)" + getVar(globalStride[i]);
    if (strideWorkaround)
      sv = "((uint64_t)" + getVar(globalStride[i]) + " >> 4)";
    replaceOrd64("global_stride", i, sv);
  }
  for (int i = 0; i < rank; ++i)
    replaceOrd32("element_stride", i, getVar(elemStride[i]));
  replace32Imm("elemtype", elemType);
  replace32Imm("interleave_layout", interleave);
  replace32Imm("swizzle_mode", swizzle);
  replace32Imm("fill_mode", fill);
  dedent();
  emit("}");
  emit("__syncwarp();");
  // Collectively (warp 0, .sync.aligned) copy shared descriptor to global slot.
  emit("asm volatile(\"tensormap.cp_fenceproxy.global.shared::cta.tensormap::"
       "generic.release.gpu.sync.aligned [%0], [%1], 128;\" :: \"l\"((uint64_t)" +
       descVar + "), \"r\"(_tms) : \"memory\");");
  dedent();
  emit("}");
  blockSync();
}

// Acquire fence so the just-written descriptor is visible to the TMA unit.
void CUDACodeGen::emitTensormapFenceproxyAcquire(
    ttng::TensormapFenceproxyAcquireOp op) {
  auto descVar = getVar(op.getDescPtr());
  // Region-local leader warp (see emitTensormapCreate): the C-descriptor fence
  // runs inside warp_specialize consumer regions where threadIdx.x >= 128.
  emit("if (tid < 32) {");
  indent();
  emit("asm volatile(\"fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\" "
       ":: \"l\"((uint64_t)" + descVar + ") : \"memory\");");
  // ptxas workaround: ensure the fence completes before use.
  emit("asm volatile(\"cp.async.bulk.commit_group;\");");
  emit("asm volatile(\"cp.async.bulk.wait_group.read 0;\");");
  dedent();
  emit("}");
  blockSync();
}

// reinterpret_tensor_descriptor: the raw global-scratch pointer IS the
// CUtensorMap*. Alias the result to it and mark it as a pointer descriptor so
// TMA copies pass the pointer directly (not its address).
void CUDACodeGen::emitReinterpretTensorDesc(ttng::ReinterpretTensorDescOp op) {
  auto srcVar = getVar(op.getRawDesc());
  valueToVar[op.getResult()] = srcVar;
  pointerDescriptors.insert(op.getResult());
}

// ── cp.async helper: emit cp.async instructions from global ptr to shared mem ──
void CUDACodeGen::emitCpAsyncToShared(
    Value ptrTensor, const std::string &dstVar,
    ttg::MemDescType dstMemDescType,
    Value mask, Value other) {
  // Resolve deferred addptr for the pointer tensor
  auto deferIt = deferredAddPtr.find(ptrTensor);
  std::string asyncBasePtr, asyncOffVar;
  bool asyncUseDeferredAddr = false;
  if (deferIt != deferredAddPtr.end()) {
    asyncBasePtr = deferIt->second.first;
    asyncOffVar = deferIt->second.second;
    asyncUseDeferredAddr = true;
    if (valueToVar.find(ptrTensor) == valueToVar.end())
      valueToVar[ptrTensor] = "/*deferred*/";
  }
  bool asyncUsePtrBased = false;
  std::string asyncPtrVar;
  auto ptrIt = ptrBasedDeferred.find(ptrTensor);
  if (ptrIt != ptrBasedDeferred.end()) {
    asyncUsePtrBased = true;
    asyncPtrVar = getVar(ptrTensor);
  }
  // Scalar-base deferred: precomputed 64-bit addresses + scalar delta
  bool asyncUseScalarBase = false;
  ScalarBaseInfo asyncSBI;
  auto sbiIt = scalarBaseDeferred.find(ptrTensor);
  if (sbiIt != scalarBaseDeferred.end()) {
    asyncUseScalarBase = true;
    asyncSBI = sbiIt->second;
    // Don't use the old deferred addr path — use precomputed addresses directly
  }
  auto srcVar = getVar(ptrTensor);

  // Source tensor type info (pointer tensor → pointee element type)
  auto srcTy = cast<RankedTensorType>(ptrTensor.getType());
  int nElems = getElemsPerThread(srcTy);
  auto srcElemType = srcTy.getElementType();
  Type elemType = srcElemType;
  if (auto ptrType = dyn_cast<tt::PointerType>(srcElemType))
    elemType = ptrType.getPointeeType();
  auto cudaType = getCUDAType(elemType);
  int bytesPerElem = elemType.getIntOrFloatBitWidth() / 8;

  bool hasMask = mask != nullptr;

  // Destination shape/encoding. Use the per-CTA shape: with num_ctas > 1 a
  // CTA-split shared buffer physically holds only this CTA's tile, and the
  // read side (emitLocalLoad) addresses it with per-CTA strides.
  auto dstEnc = dstMemDescType.getEncoding();
  auto dstShape = ttg::getShapePerCTA(dstEnc, dstMemDescType.getShape());
  auto nvmmaShared = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(dstEnc);

  // LinearLayout for source tensor's register→coordinate mapping
  auto srcLL = ttg::toLinearLayout(srcTy);
  auto kReg = mlir::StringAttr::get(srcTy.getContext(), "register");
  auto kLane = mlir::StringAttr::get(srcTy.getContext(), "lane");
  auto kWarp = mlir::StringAttr::get(srcTy.getContext(), "warp");
  auto kBlock = mlir::StringAttr::get(srcTy.getContext(), "block");
  const auto &srcBases = srcLL.getBases();
  const auto &laneBases = srcBases.find(kLane)->second;
  const auto &warpBases = srcBases.find(kWarp)->second;

  int rank = dstShape.size();
  // Honor the shared encoding's `order` (order[0] = fastest/contiguous dim).
  // For the common order=[rank-1,...,0] this is identical to plain row-major,
  // so only non-standard orders change behavior — e.g. a K-contiguous fp8 B
  // tile has order=[0,1] (dim0 fastest). Computing strides row-major there
  // mis-placed the data AND made the vectorized cp.async write a
  // non-contiguous, non-16B-aligned destination (misaligned-address crash).
  SmallVector<int64_t> strides(rank, 1);
  if (auto swizEnc = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(dstEnc)) {
    auto order = swizEnc.getOrder();
    int64_t s = 1;
    for (int oi = 0; oi < rank; oi++) {
      int d = order[oi];
      strides[d] = s;
      s *= dstShape[d];
    }
  } else if (nvmmaShared && rank == 2 && nvmmaShared.getTransposed()) {
    // Transposed nvmma: physically stored [dim1][dim0], so the contiguous
    // (stride-1) logical dim is dim0. Using row-major strides here let
    // register runs along dim1 pass the vectorization check, producing 16B
    // cp.asyncs whose shared destination is strided/unaligned (misaligned
    // address, e.g. test_dot_mulbroadcasted B tile).
    strides[0] = 1;
    strides[1] = dstShape[0];
  } else {
    strides.back() = 1;
    for (int d = rank - 2; d >= 0; d--)
      strides[d] = strides[d + 1] * dstShape[d + 1];
  }

  bool isTransposed = nvmmaShared && nvmmaShared.getTransposed();
  SmallVector<int64_t> physShape(dstShape.begin(), dstShape.end());
  if (isTransposed && rank == 2)
    std::swap(physShape[0], physShape[1]);

  int swizzleBytes = nvmmaShared ? nvmmaShared.getSwizzlingByteWidth() : 0;
  int vec = 0, perPhase = 1, maxPhase = 1;
  int elemsPerSwizzlingRow = 0;
  if (nvmmaShared && swizzleBytes > 0) {
    vec = 16 / bytesPerElem;
    perPhase = std::max(1, (int)(128 / (physShape.back() * bytesPerElem)));
    maxPhase = swizzleBytes / 16;
    elemsPerSwizzlingRow = swizzleBytes / bytesPerElem;
  }

  emit("// cp.async: global -> shared (" + std::to_string(nElems) +
       " elems/thread, tile-major + swizzle)");
  emit("{");
  indent();
  emit(cudaType + "* _dst = (" + cudaType + "*)" + dstVar + ";");

  // Group consecutive registers for vectorized cp.async.
  // Clamp the per-cp.async width to the pointer's actual alignment: the global
  // source address of a cp.async.{ca,cg} must be naturally aligned to the access
  // size. Register-contiguity alone is not enough — an odd row stride (e.g. a
  // weight tensor with n=700: row stride 1400B ≡ 8 mod 16) makes alternate rows
  // only 8-byte aligned, so a 16-byte cp.async would fault with a misaligned
  // address. getMaxVecWidth mirrors the PTX backend's getVectorSize.
  unsigned maxVecW = getMaxVecWidth(ptrTensor, hasMask ? mask : Value());
  SmallVector<std::pair<int, int>> groups;
  {
    int groupStart = 0;
    for (int i = 1; i <= nElems; i++) {
      bool contiguous = false;
      if (i < nElems) {
        auto prevCoords = srcLL.apply({{kReg, i-1}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        auto currCoords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        // Find which single dimension changes between consecutive registers.
        // Accept contiguity along ANY dimension (not just the last one),
        // since the fast dimension may be dim 0 (e.g., fp8 B tile with K-contiguous layout).
        contiguous = true;
        int changeDim = -1;
        for (int d = 0; d < rank; d++) {
          if (currCoords[d].second != prevCoords[d].second) {
            if (changeDim != -1) { contiguous = false; break; } // more than one dim changed
            changeDim = d;
            if (currCoords[d].second != prevCoords[d].second + 1) { contiguous = false; break; }
          }
        }
        if (changeDim == -1) contiguous = false; // no dim changed
        // A vectorized cp.async writes a contiguous run of shared bytes, so the
        // varying dimension must be the stride-1 (contiguous) dim of the dst
        // layout. Registers contiguous along a non-stride-1 source dim are NOT
        // contiguous in shared — vectorizing them would write the wrong (and
        // possibly misaligned) destination addresses.
        else if (strides[changeDim] != 1) contiguous = false;
      }
      if (!contiguous) {
        int count = i - groupStart;
        while (count > 0) {
          int effCount = std::min(count, (int)maxVecW);
          int vecBytes = effCount * bytesPerElem;
          int cpBytes = 0;
          if (vecBytes >= 16) cpBytes = 16;
          else if (vecBytes >= 8) cpBytes = 8;
          else if (vecBytes >= 4) cpBytes = 4;
          int cpElems = cpBytes > 0 ? cpBytes / bytesPerElem : effCount;
          groups.push_back({groupStart, cpElems});
          groupStart += cpElems;
          count -= cpElems;
        }
        groupStart = i;
      }
    }
  }

  int physCols = isTransposed ? physShape[1] : (rank >= 2 ? dstShape[1] : 1);
  int physRows = isTransposed ? physShape[0] : (rank >= 2 ? dstShape[0] : 1);

  // Swizzled nvmma_shared with rank != 2 (e.g. rank-3+ TMA descriptor tiles
  // where TMALowering picked swizzle_mode != 0): the per-dim linear stride
  // formula below cannot express the XOR swizzle, so compute per-register
  // shared offsets via the shared linear layout (invertAndCompose, mirrors
  // emitLayoutAwareSharedStore). Without this the TMA bulk store/engine reads
  // the buffer swizzled while cp.async wrote it linearly → scrambled data.
  bool swizzleNd = nvmmaShared && swizzleBytes > 0 && rank != 2;
  llvm::SmallVector<int64_t> ndRegOff;
  std::string ndLwExpr = "0";
  if (swizzleNd) {
    auto kOffset = mlir::StringAttr::get(srcTy.getContext(), "offset");
    auto sharedLL = ttg::toLinearLayout(dstMemDescType);
    auto cvt = srcLL.invertAndCompose(sharedLL);
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
    auto applyOff = [&](int reg, int lane, int warp) -> int64_t {
      auto out = cvt.apply({{kReg, reg}, {kLane, lane}, {kWarp, warp}});
      return out.front().second;
    };
    ndRegOff.resize(nElems);
    for (int i = 0; i < nElems; i++)
      ndRegOff[i] = applyOff(i, 0, 0);
    int64_t ndLwMask = 0; // OR of all lane/warp XOR deltas
    for (int b = 0; b < cvt.getInDimSizeLog2(kLane); b++) {
      int64_t d = applyOff(0, 1 << b, 0);
      if (d != 0) {
        ndLwExpr += " ^ (((lane_id >> " + std::to_string(b) + ") & 1) * " +
                    std::to_string(d) + ")";
        ndLwMask |= d;
      }
    }
    for (int b = 0; b < cvt.getInDimSizeLog2(kWarp); b++) {
      int64_t d = applyOff(0, 0, 1 << b);
      if (d != 0) {
        ndLwExpr += " ^ (((warp_id >> " + std::to_string(b) + ") & 1) * " +
                    std::to_string(d) + ")";
        ndLwMask |= d;
      }
    }
    // Re-split the global-contiguity groups so that within each group the
    // shared offsets are also consecutive AND the group is XOR-safe:
    // power-of-two length, base offset aligned to the length, and every
    // lane/warp XOR delta a multiple of the length — then for j < len,
    // (_lw ^ off) + j == _lw ^ (off + j), so a vectorized cp.async starting
    // at _dst[_lw ^ off] writes exactly the right shared bytes.
    SmallVector<std::pair<int, int>> ng;
    for (auto &[gs, gc] : groups) {
      int i = gs;
      while (i < gs + gc) {
        int run = 1;
        while (i + run < gs + gc && ndRegOff[i + run] == ndRegOff[i] + run)
          run++;
        int len = 1;
        while (len * 2 <= run) len *= 2;
        while (len > 1 && (((ndRegOff[i] & (len - 1)) != 0) ||
                           ((ndLwMask & (len - 1)) != 0)))
          len /= 2;
        ng.push_back({i, len});
        i += len;
      }
    }
    groups = std::move(ng);
  }

  // Mapping from LL register index to flat array index. The emitter's flat
  // per-thread arrays (make_range, broadcast, expand_dims, addptr offsets,
  // masks) are all built in compact LinearLayout register order — broadcasts
  // are validated against the LL ground truth in emitBroadcast — so the LL
  // register index IS the flat array index. (A previous "last-dim-fast"
  // remap here compensated for the old order-dependent broadcast heuristic;
  // with LL-validated broadcasts that remap corrupts order=[0,1] tensors,
  // e.g. shifting mxfp4 W-tile cp.async global addresses by 1 byte →
  // misaligned address.)
  SmallVector<int> regToFlat(nElems);
  for (int r = 0; r < nElems; r++)
    regToFlat[r] = r;

  // Vectorized source pre-load. When every cp.async group degenerated to a
  // 1-element scalar store — the GLOBAL source is contiguous but the swizzled
  // shared destination scatters it (e.g. a col-major int8 A tile: dim0 has
  // global stride 1 yet maps to stride-32 shared rows) — the per-element
  // `_dst[..] = *(base+off[i])` path emits N scalar ld.global.u8. The PTX
  // backend instead issues ld.global.v4 into registers, then scattered shared
  // stores. Mirror that: pre-load maximal source-contiguous runs into a
  // register temp via vector loads, and let the scalar swizzled stores below
  // read from the temp. Gated to the deferred-addr, unmasked, all-scalar case.
  bool allScalarSrc = true;
  for (auto &g : groups) {
    int cpB = g.second * bytesPerElem;
    if (cpB == 4 || cpB == 8 || cpB == 16) {
      allScalarSrc = false;
      break;
    }
  }
  bool useLdTemp = asyncUseDeferredAddr && !hasMask && allScalarSrc &&
                   (int)maxVecW * bytesPerElem >= 4 && nElems > 1;
  std::string ldVar;
  if (useLdTemp) {
    // Maximal source-contiguous runs: consecutive registers whose source
    // coordinate advances by +1 along a single dim (independent of the
    // destination stride that blocked cp.async vectorization).
    SmallVector<std::pair<int, int>> srcRuns;
    int rsStart = 0;
    for (int i = 1; i <= nElems; i++) {
      bool cont = false;
      if (i < nElems) {
        auto pc =
            srcLL.apply({{kReg, i - 1}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        auto cc =
            srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        cont = true;
        int cd = -1;
        for (int d = 0; d < rank; d++) {
          if (cc[d].second != pc[d].second) {
            if (cd != -1) {
              cont = false;
              break;
            }
            cd = d;
            if (cc[d].second != pc[d].second + 1) {
              cont = false;
              break;
            }
          }
        }
        if (cd == -1)
          cont = false;
      }
      if (!cont) {
        srcRuns.push_back({rsStart, i - rsStart});
        rsStart = i;
      }
    }
    ldVar = newVar("ldsrc");
    emit(cudaType + " " + ldVar + "[" + std::to_string(nElems) + "];");
    for (auto &[start, cnt] : srcRuns) {
      int rem = cnt, off = start;
      while (rem > 0) {
        int eff = std::min(rem, (int)maxVecW);
        int vb = eff * bytesPerElem, lb;
        if (vb >= 16)
          lb = 16;
        else if (vb >= 8)
          lb = 8;
        else if (vb >= 4)
          lb = 4;
        else
          lb = bytesPerElem;
        int le = lb / bytesPerElem;
        int flatI = regToFlat[off];
        std::string ga = "(const char*)" + asyncBasePtr + " + (unsigned int)(" +
                         asyncOffVar + "[" + std::to_string(flatI) + "] * " +
                         std::to_string(bytesPerElem) + ")";
        if (lb >= 4) {
          std::string vt = lb == 16 ? "uint4" : lb == 8 ? "uint2" : "unsigned";
          emit("{ " + vt + " _ldt = *(const " + vt + "*)(" + ga + ");");
          emit("  const " + cudaType + "* _ldp = (const " + cudaType +
               "*)&_ldt;");
          for (int j = 0; j < le; j++)
            emit("  " + ldVar + "[" + std::to_string(off + j) + "] = _ldp[" +
                 std::to_string(j) + "];");
          emit("}");
        } else {
          emit(ldVar + "[" + std::to_string(off) + "] = *((const " + cudaType +
               "*)(" + ga + "));");
        }
        off += le;
        rem -= le;
      }
    }
  }

  if (nvmmaShared && swizzleBytes > 0 && rank == 2) {
    int rowDim = isTransposed ? 1 : 0;
    int colDim = isTransposed ? 0 : 1;
    std::string rowBase = "0";
    for (size_t lb = 0; lb < laneBases.size(); lb++) {
      int delta = laneBases[lb][rowDim];
      if (delta != 0)
        rowBase += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
    }
    for (size_t wb = 0; wb < warpBases.size(); wb++) {
      int delta = warpBases[wb][rowDim];
      if (delta != 0)
        rowBase += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
    }
    std::string colBase = "0";
    for (size_t lb = 0; lb < laneBases.size(); lb++) {
      int delta = laneBases[lb][colDim];
      if (delta != 0)
        colBase += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
    }
    for (size_t wb = 0; wb < warpBases.size(); wb++) {
      int delta = warpBases[wb][colDim];
      if (delta != 0)
        colBase += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
    }
    emit("int _rowBase = " + rowBase + ";");
    emit("int _colBase = " + colBase + ";");
  } else if (swizzleNd) {
    emit("int _lw = " + ndLwExpr + ";");
  }

  // Pre-scan groups for base+offset shared memory address optimization.
  // Groups sharing the same swizzle phase and column can use a single base
  // __cvta_generic_to_shared + constant byte offsets, avoiding redundant
  // LEA instructions that stall on uniform register dependencies in SASS.
  SmallVector<int> saFamilyId;
  SmallVector<int> saByteOffset;
  if (nvmmaShared && swizzleBytes > 0 && rank == 2) {
    struct SaFamilyBase { int regRow, regCol, phase, tileK; };
    SmallVector<SaFamilyBase> saFamilyBases;
    int rowDim_ = isTransposed ? 1 : 0;
    int colDim_ = isTransposed ? 0 : 1;
    bool tiled_ = (int)physCols > elemsPerSwizzlingRow;

    for (size_t gi = 0; gi < groups.size(); gi++) {
      auto rc = srcLL.apply({{kReg, groups[gi].first}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      int rr = rc[rowDim_].second;
      int rcc = rc[colDim_].second;
      int ph = (rr / perPhase) % maxPhase;
      int tk = tiled_ ? rcc / elemsPerSwizzlingRow : 0;
      int cit = tiled_ ? rcc % elemsPerSwizzlingRow : rcc;

      int fid = -1;
      for (size_t fi = 0; fi < saFamilyBases.size(); fi++) {
        int famCit = tiled_ ? saFamilyBases[fi].regCol % elemsPerSwizzlingRow : saFamilyBases[fi].regCol;
        if (saFamilyBases[fi].phase == ph && saFamilyBases[fi].tileK == tk && famCit == cit) {
          fid = fi; break;
        }
      }
      if (fid == -1) {
        fid = saFamilyBases.size();
        saFamilyBases.push_back({rr, rcc, ph, tk});
      }

      int rowDiff = rr - saFamilyBases[fid].regRow;
      int elemStride = tiled_ ? elemsPerSwizzlingRow : physCols;
      saFamilyId.push_back(fid);
      saByteOffset.push_back(rowDiff * elemStride * bytesPerElem);
    }

    // Emit base address computation for each family
    for (size_t fi = 0; fi < saFamilyBases.size(); fi++) {
      auto &fb = saFamilyBases[fi];
      emit("unsigned _sa_base" + std::to_string(fi) + ";");
      if (tiled_) {
        emit("{ int _r = _rowBase + " + std::to_string(fb.regRow) + ";");
        emit("  int _c = _colBase + " + std::to_string(fb.regCol) + ";");
        emit("  int _tk = _c / " + std::to_string(elemsPerSwizzlingRow) + ";");
        emit("  int _cit = _c % " + std::to_string(elemsPerSwizzlingRow) + ";");
        emit("  int _ph = (_r / " + std::to_string(perPhase) + ") % " + std::to_string(maxPhase) + ";");
        emit("  int _sc = _cit ^ (_ph * " + std::to_string(vec) + ");");
        emit("  _sa_base" + std::to_string(fi) + " = __cvta_generic_to_shared(&_dst[_tk * " +
             std::to_string(physRows * elemsPerSwizzlingRow) + " + _r * " +
             std::to_string(elemsPerSwizzlingRow) + " + _sc]); }");
      } else {
        emit("{ int _r = _rowBase + " + std::to_string(fb.regRow) + ";");
        emit("  int _c = _colBase + " + std::to_string(fb.regCol) + ";");
        emit("  int _ph = (_r / " + std::to_string(perPhase) + ") % " + std::to_string(maxPhase) + ";");
        emit("  int _sc = _c ^ (_ph * " + std::to_string(vec) + ");");
        emit("  _sa_base" + std::to_string(fi) + " = __cvta_generic_to_shared(&_dst[_r * " +
             std::to_string((int)physCols) + " + _sc]); }");
      }
    }
  }

  // Check if all groups can use cp.async and we have base+offset optimization
  bool useBaseOffset = !saFamilyId.empty();
  bool allCpAsync = true;
  for (size_t _gi = 0; _gi < groups.size(); _gi++) {
    int cpBytes = groups[_gi].second * bytesPerElem;
    if (!(cpBytes >= 4 && cpBytes <= 16 &&
          (cpBytes == 4 || cpBytes == 8 || cpBytes == 16))) {
      allCpAsync = false;
      break;
    }
  }

  // Merged cp.async emission: emit all cp.async in a single asm volatile block
  // with inline PTX global address computation and predicate conversion. This
  // gives ptxas full scheduling visibility over all LDGSTS plus their address
  // computation (cvt + shl + add + setp + selp per group), allowing it to
  // interleave ALU with LDGSTS to hide latency. Without this merge, separate
  // asm volatile blocks and pre-computed C++ arrays prevent ptxas from filling
  // LDGSTS stall cycles with interleaved ALU work.
  if (allCpAsync && useBaseOffset && hasMask &&
      (asyncUseDeferredAddr || asyncUsePtrBased) && groups.size() >= 2) {
    // Phase 1: Collect per-group info without emitting cp.async
    size_t N = groups.size();
    int elemShift = 0;
    if (bytesPerElem == 2) elemShift = 1;
    else if (bytesPerElem == 4) elemShift = 2;
    else if (bytesPerElem == 8) elemShift = 3;
    else if (bytesPerElem == 1) elemShift = 0;

    // Emit shared address, pre-computed 64-bit global address, and mask variables.
    // Computing full 64-bit addresses in C++ (not PTX) gives nvcc/ptxas maximum
    // scheduling freedom for the cvt+add, while the asm block only contains
    // setp+selp+cp.async (3 PTX instructions per group, down from 5).
    for (size_t _gi = 0; _gi < N; _gi++) {
      int i = groups[_gi].first;
      std::string saExpr = "_sa_base" + std::to_string(saFamilyId[_gi]);
      if (saByteOffset[_gi] != 0)
        saExpr += " + " + std::to_string(saByteOffset[_gi]);
      emit("unsigned _cp_sa_" + std::to_string(_gi) + " = " + saExpr + ";");
      int flatI = regToFlat[i];
      // Pre-compute full 64-bit global address in C++.
      // This moves cvt.u64.u32+add.u64 out of the asm block, giving ptxas
      // full freedom to schedule them separately from the cp.async instructions.
      if (asyncUseScalarBase) {
        // Precomputed addr: addr = precomp[i] + (long long)delta (signed:
        // negative loop pointer updates must move the 64-bit base backward).
        emit("const void* _cp_ga_" + std::to_string(_gi) +
             " = (const void*)(" + asyncSBI.precompAddr + "[" + std::to_string(flatI) +
             "] + (long long)" + asyncSBI.deltaVar + ");");
      } else if (asyncUsePtrBased) {
        emit("const void* _cp_ga_" + std::to_string(_gi) +
             " = (const void*)" + asyncPtrVar + "[" + std::to_string(flatI) +
             "];");
      } else {
        emit("const void* _cp_ga_" + std::to_string(_gi) +
             " = (const char*)" + asyncBasePtr +
             " + (unsigned int)(" + asyncOffVar + "[" + std::to_string(flatI) +
             "] * " + std::to_string(bytesPerElem) + ");");
      }
      // Raw mask boolean (use regToFlat mapping)
      std::string maskE = getElemExpr(mask, std::to_string(flatI));
      emit("int _cp_mask_" + std::to_string(_gi) + " = (int)(" + maskE + ");");
    }

    // Phase 2: Build merged PTX asm block — only pred + cp.async per group
    std::string ptx = "{\\n\\t";
    ptx += ".reg .pred %%p;\\n\\t";
    ptx += ".reg .b32 %%pred;\\n\\t";

    {
      // Operand layout:
      //   %0..%(N-1)   = shared addrs (r)
      //   %N..%(2N-1)  = global addrs (l)
      //   %(2N)..%(3N-1) = mask bools (r)
      for (size_t i = 0; i < N; i++) {
        int cpBytes = groups[i].second * bytesPerElem;
        std::string variant = (cpBytes >= 16) ? "cg" : "ca";
        int saIdx = i;
        int gaIdx = N + i;
        int maskIdx = 2 * N + i;

        // Predicate: mask ? cpBytes : 0
        ptx += "setp.ne.b32 %%p, %" + std::to_string(maskIdx) + ", 0;\\n\\t";
        ptx += "selp.b32 %%pred, " + std::to_string(cpBytes) + ", 0, %%p;\\n\\t";
        // cp.async with pre-computed address
        ptx += "cp.async." + variant + ".shared.global [%" +
               std::to_string(saIdx) + "], [%" +
               std::to_string(gaIdx) + "], " +
               std::to_string(cpBytes) + ", %%pred;\\n\\t";
      }
      ptx += "}";

      // Build constraint string
      std::string constraints;
      // Shared addresses (r)
      for (size_t i = 0; i < N; i++) {
        if (i > 0) constraints += ", ";
        constraints += "\"r\"(_cp_sa_" + std::to_string(i) + ")";
      }
      // Global addresses (l)
      for (size_t i = 0; i < N; i++)
        constraints += ", \"l\"(_cp_ga_" + std::to_string(i) + ")";
      // Mask booleans (r)
      for (size_t i = 0; i < N; i++)
        constraints += ", \"r\"(_cp_mask_" + std::to_string(i) + ")";

      emit("asm volatile(\"" + ptx + "\" :: " + constraints + ");");
    }
  } else {
    // Per-group emission fallback (original path)
    for (size_t _gi = 0; _gi < groups.size(); _gi++) {
      auto &[groupStart, groupCount] = groups[_gi];
      int i = groupStart;
      auto regCoords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});

      std::string offsetExpr;
      if (swizzleNd) {
        // XOR-swizzled offset; group re-splitting above guarantees that for
        // any j < groupCount, (_lw ^ off) + j == _lw ^ (off + j).
        emit("{");
        offsetExpr = "(_lw ^ " + std::to_string(ndRegOff[i]) + ")";
      } else if (nvmmaShared && swizzleBytes > 0 && rank == 2) {
        int rowDim = isTransposed ? 1 : 0;
        int colDim = isTransposed ? 0 : 1;
        int regRow = regCoords[rowDim].second;
        int regCol = regCoords[colDim].second;

        if ((int)physCols > elemsPerSwizzlingRow) {
          emit("{ int _row = _rowBase + " + std::to_string(regRow) + ";");
          emit("  int _col = _colBase + " + std::to_string(regCol) + ";");
          emit("  int _tileK = _col / " + std::to_string(elemsPerSwizzlingRow) + ";");
          emit("  int _colInTile = _col % " + std::to_string(elemsPerSwizzlingRow) + ";");
          emit("  int _phase = (_row / " + std::to_string(perPhase) + ") % " + std::to_string(maxPhase) + ";");
          emit("  int _swCol = _colInTile ^ (_phase * " + std::to_string(vec) + ");");
          offsetExpr = "_tileK * " + std::to_string(physRows * elemsPerSwizzlingRow) +
                       " + _row * " + std::to_string(elemsPerSwizzlingRow) + " + _swCol";
        } else {
          emit("{ int _row = _rowBase + " + std::to_string(regRow) + ";");
          emit("  int _col = _colBase + " + std::to_string(regCol) + ";");
          emit("  int _phase = (_row / " + std::to_string(perPhase) + ") % " + std::to_string(maxPhase) + ";");
          emit("  int _swCol = _col ^ (_phase * " + std::to_string(vec) + ");");
          offsetExpr = "_row * " + std::to_string((int)physCols) + " + _swCol";
        }
      } else {
        emit("{");
        SmallVector<std::string> coordExprs(rank);
        for (int d = 0; d < rank; d++) {
          int regBase = regCoords[d].second;
          std::string expr = std::to_string(regBase);
          for (size_t lb = 0; lb < laneBases.size(); lb++) {
            int delta = laneBases[lb][d];
            if (delta != 0)
              expr += " + ((lane_id >> " + std::to_string(lb) + ") & 1) * " + std::to_string(delta);
          }
          for (size_t wb = 0; wb < warpBases.size(); wb++) {
            int delta = warpBases[wb][d];
            if (delta != 0)
              expr += " + ((warp_id >> " + std::to_string(wb) + ") & 1) * " + std::to_string(delta);
          }
          coordExprs[d] = expr;
        }
        offsetExpr = "";
        for (int d = 0; d < rank; d++) {
          if (d > 0) offsetExpr += " + ";
          offsetExpr += "(" + coordExprs[d] + ") * " + std::to_string(strides[d]);
        }
      }

      int cpBytes = groupCount * bytesPerElem;
      bool useCpAsync = (cpBytes >= 4 && cpBytes <= 16 &&
                         (cpBytes == 4 || cpBytes == 8 || cpBytes == 16));
      std::string cpAsyncVariant = (cpBytes >= 16) ? "cg" : "ca";

      if (useCpAsync) {
        int flatI = regToFlat[i];
        std::string globalAddr;
        if (asyncUseScalarBase) {
          globalAddr = "(const void*)(" + asyncSBI.precompAddr + "[" + std::to_string(flatI) +
                       "] + (long long)" + asyncSBI.deltaVar + ")";
        } else if (asyncUsePtrBased) {
          globalAddr = "(const void*)" + asyncPtrVar + "[" + std::to_string(flatI) + "]";
        } else if (asyncUseDeferredAddr) {
          globalAddr = "(const void*)(" + asyncBasePtr + " + " + asyncOffVar + "[" + std::to_string(flatI) + "])";
        } else if (scalarValues.contains(ptrTensor)) {
          // Uniform scalar pointer (e.g. tt.splat of a base ptr): srcVar holds
          // the pointer itself, not an array. Indexing srcVar[flatI] would
          // dereference to a value (invalid void* conversion in nvcc).
          globalAddr = "(const void*)" + srcVar;
        } else {
          globalAddr = "(const void*)" + srcVar + "[" + std::to_string(flatI) + "]";
        }
        emit("  {");
        if (useBaseOffset) {
          std::string saExpr = "_sa_base" + std::to_string(saFamilyId[_gi]);
          if (saByteOffset[_gi] != 0)
            saExpr += " + " + std::to_string(saByteOffset[_gi]);
          emit("    unsigned _sa = " + saExpr + ";");
        } else {
          emit("    unsigned _sa = __cvta_generic_to_shared(&_dst[" + offsetExpr + "]);");
        }
        if (hasMask) {
          std::string maskE = getElemExpr(mask, std::to_string(flatI));
          emit("    int _pred = " + maskE + " ? " +
               std::to_string(cpBytes) + " : 0;");
          emit("    asm volatile(\"cp.async." + cpAsyncVariant + ".shared.global [%0], [%1], " +
               std::to_string(cpBytes) + ", %2;\" :: \"r\"(_sa), \"l\"(" +
               globalAddr + "), \"r\"(_pred));");
        } else {
          emit("    asm volatile(\"cp.async." + cpAsyncVariant + ".shared.global [%0], [%1], " +
               std::to_string(cpBytes) + ";\" :: \"r\"(_sa), \"l\"(" +
               globalAddr + "));");
        }
        emit("  }");
      } else {
        // Synchronous fallback for non-vectorizable groups
        for (int j = 0; j < groupCount; j++) {
          int flatIJ = regToFlat[i + j];
          std::string srcDeref;
          if (useLdTemp) {
            // Read from the vector-loaded source temp (ld.global.v4 above).
            srcDeref = ldVar + "[" + std::to_string(i + j) + "]";
          } else if (asyncUseScalarBase) {
            srcDeref = "*(const " + cudaType + "*)(" + asyncSBI.precompAddr + "[" + std::to_string(flatIJ) +
                       "] + (long long)" + asyncSBI.deltaVar + ")";
          } else if (asyncUsePtrBased) {
            auto &[bpe, cudaT] = ptrBasedDeferred[ptrTensor];
            srcDeref = "*((" + cudaT + "*)" + asyncPtrVar + "[" + std::to_string(flatIJ) + "])";
          } else if (asyncUseDeferredAddr) {
            srcDeref = "*(" + asyncBasePtr + " + " + asyncOffVar + "[" + std::to_string(flatIJ) + "])";
          } else {
            srcDeref = "*" + srcVar + "[" + std::to_string(flatIJ) + "]";
          }
          if (hasMask) {
            std::string maskE = getElemExpr(mask, std::to_string(flatIJ));
            emit("  if (" + maskE + ")");
            emit("    _dst[" + offsetExpr + " + " + std::to_string(j) + "] = " + srcDeref + ";");
            emit("  else");
            if (other) {
              emit("    _dst[" + offsetExpr + " + " + std::to_string(j) + "] = " +
                   getElemExpr(other, std::to_string(flatIJ)) + ";");
            } else {
              emit("    _dst[" + offsetExpr + " + " + std::to_string(j) + "] = 0;");
            }
          } else {
            emit("  _dst[" + offsetExpr + " + " + std::to_string(j) + "] = " + srcDeref + ";");
          }
        }
      }
      emit("}");
    }
  }
  dedent();
  emit("}");
}

void CUDACodeGen::emitAsyncCopyG2L(ttg::AsyncCopyGlobalToLocalOp op) {
  auto src = op.getSrc();
  auto dst = op->getOperand(1);
  auto dstVar = getVar(dst);
  auto dstMemDesc = cast<ttg::MemDescType>(dst.getType());

  emitCpAsyncToShared(src, dstVar, dstMemDesc, op.getMask(), op.getOther());

  // Map the async token result (placeholder — tokens are consumed by commit/wait)
  if (op->getNumResults() > 0) {
    auto var = newVar("tok");
    valueToVar[op->getResult(0)] = var;
    emit("const int " + var + " = 0; // async token");
  }
}

void CUDACodeGen::emitAsyncCommitGroup(ttg::AsyncCommitGroupOp op) {
  // Try to merge commit_group into the preceding cp.async asm block
  // to eliminate a scheduling barrier between cp.async and commit.
  bool merged = false;
  for (int i = (int)lines.size() - 1; i >= 0; i--) {
    auto &line = lines[i];
    if (line.find("cp.async.cg") != std::string::npos &&
        line.find("asm volatile") != std::string::npos &&
        line.find("cp.async.commit_group") == std::string::npos) {
      // The PTX block ends with: ...;\n\t}" :: constraints);
      // We want to insert \n\tcp.async.commit_group; before the closing \n\t}
      // In the lines vector, \n\t appears as literal characters: backslash n backslash t
      std::string closing = "\\n\\t}";
      auto pos = line.rfind(closing);
      if (pos != std::string::npos) {
        std::string insert = "\\n\\tcp.async.commit_group;";
        line.insert(pos, insert);
        merged = true;
      }
      break;
    }
    if (line.find("asm volatile") != std::string::npos)
      break;
  }
  if (!merged)
    emit("asm volatile(\"cp.async.commit_group;\");");
  // Map the result token
  if (op->getNumResults() > 0) {
    auto var = newVar("tok");
    valueToVar[op->getResult(0)] = var;
    emit("const int " + var + " = 0; // async token placeholder");
  }
}

void CUDACodeGen::emitAsyncWait(ttg::AsyncWaitOp op) {
  int num = op.getNum();
  emit("asm volatile(\"cp.async.wait_group " + llvm::Twine(num).str() + ";\");");
  blockSync();
  // Map the result token
  if (op->getNumResults() > 0) {
    auto var = newVar("tok");
    valueToVar[op->getResult(0)] = var;
    emit("const int " + var + " = 0; // async token placeholder");
  }
}

uint64_t CUDACodeGen::getWGMMADescTemplate(int swizzleBytes,
                                            int strideDimSize) {
  uint64_t desc = 0;
  if (swizzleBytes == 128)
    desc |= (1ULL << 62);
  else if (swizzleBytes == 64)
    desc |= (2ULL << 62);
  else if (swizzleBytes == 32)
    desc |= (3ULL << 62);
  // strideDimensionBaseOffset = swizzleBytes >> 1
  desc |= (((uint64_t)(swizzleBytes >> 1)) & 0x3FFFULL) << 32;
  // leadDimensionBaseOffset = (swizzleBytes * strideDimSize) >> 4
  desc |= (((uint64_t)((swizzleBytes * strideDimSize) >> 4)) & 0x3FFFULL) << 16;
  return desc;
}

void CUDACodeGen::emitBlockedStoreToSmem(llvm::StringRef srcVar,
                                          llvm::StringRef smemVar,
                                          ttg::BlockedEncodingAttr enc,
                                          llvm::ArrayRef<int64_t> shape,
                                          llvm::StringRef elemType,
                                          int nElems) {
  if (shape.size() == 1) {
    int spt = enc.getSizePerThread()[0];
    int tpw = enc.getThreadsPerWarp()[0];
    int stride = tpw * enc.getWarpsPerCTA()[0] * spt;
    emit("{");
    indent();
    emit("int _base = (lane_id % " + std::to_string(tpw) + " + warp_id * " +
         std::to_string(tpw) + ") * " + std::to_string(spt) + ";");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    if (nElems == spt)
      emit("    " + smemVar.str() + "[_base + _i] = " + srcVar.str() + "[_i];");
    else
      emit("    " + smemVar.str() + "[_base + (_i % " + std::to_string(spt) +
           ") + (_i / " + std::to_string(spt) + ") * " + std::to_string(stride) + "] = " +
           srcVar.str() + "[_i];");
    dedent();
    emit("}");
  } else if (shape.size() >= 2) {
    // Register convention is PER-CTA (PTX-backend convention): with
    // CTASplitNum>1 each CTA holds only its slice, so rep counts and the
    // smem row stride must use shapePerCTA, not the full tensor shape.
    auto shapePerCTA = ttg::getShapePerCTA(enc, shape);
    int rows = shapePerCTA[0], cols = shapePerCTA[1];
    int spt0 = enc.getSizePerThread()[0], spt1 = enc.getSizePerThread()[1];
    int tpw0 = enc.getThreadsPerWarp()[0], tpw1 = enc.getThreadsPerWarp()[1];
    int wpc0 = enc.getWarpsPerCTA()[0], wpc1 = enc.getWarpsPerCTA()[1];
    // Honor the blocked encoding's order: order[0]==0 means dim0 is the
    // fastest-varying dim for lane/warp/register decomposition (column-major
    // outputs, e.g. test_dot trans epilogue).
    bool dim0Fast = (enc.getOrder()[0] == 0);
    int totalT0 = tpw0 * wpc0;
    int totalT1 = tpw1 * wpc1;
    int reps0 = std::max(1, rows / (totalT0 * spt0));
    int reps1 = std::max(1, cols / (totalT1 * spt1));
    int strideRep0 = spt1 * reps1 * spt0;
    emit("{");
    indent();
    if (dim0Fast) {
      emit("int _ld0 = lane_id % " + std::to_string(tpw0) + ";");
      emit("int _ld1 = lane_id / " + std::to_string(tpw0) + ";");
      emit("int _wd0 = warp_id % " + std::to_string(wpc0) + ";");
      emit("int _wd1 = warp_id / " + std::to_string(wpc0) + ";");
    } else {
      emit("int _ld0 = lane_id / " + std::to_string(tpw1) + ";");
      emit("int _ld1 = lane_id % " + std::to_string(tpw1) + ";");
      emit("int _wd0 = warp_id / " + std::to_string(wpc1) + ";");
      emit("int _wd1 = warp_id % " + std::to_string(wpc1) + ";");
    }
    emit("int _bp0 = _ld0 + _wd0 * " + std::to_string(tpw0) + ";");
    emit("int _bp1 = _ld1 + _wd1 * " + std::to_string(tpw1) + ";");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    if (dim0Fast) {
      emit("int _s0 = _i % " + std::to_string(spt0) + ";");
      emit("int _rp0 = (_i / " + std::to_string(spt0) + ") % " + std::to_string(reps0) + ";");
      emit("int _s1 = (_i / " + std::to_string(spt0 * reps0) + ") % " + std::to_string(spt1) + ";");
      emit("int _rp1 = _i / " + std::to_string(spt0 * reps0 * spt1) + ";");
    } else {
      emit("int _rp0 = _i / " + std::to_string(strideRep0) + ";");
      emit("int _s0 = (_i / " + std::to_string(spt1 * reps1) + ") % " + std::to_string(spt0) + ";");
      emit("int _rp1 = (_i / " + std::to_string(spt1) + ") % " + std::to_string(reps1) + ";");
      emit("int _s1 = _i % " + std::to_string(spt1) + ";");
    }
    emit("int _row = _bp0 * " + std::to_string(spt0) + " + _s0 + _rp0 * " + std::to_string(totalT0 * spt0) + ";");
    emit("int _col = _bp1 * " + std::to_string(spt1) + " + _s1 + _rp1 * " + std::to_string(totalT1 * spt1) + ";");
    emit(smemVar.str() + "[_row * " + std::to_string(cols) + " + _col] = " + srcVar.str() + "[_i];");
    dedent();
    emit("}");
    dedent();
    emit("}");
  } else {
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + smemVar.str() + "[tid * " + std::to_string(nElems) + " + _i] = " + srcVar.str() + "[_i];");
  }
}

void CUDACodeGen::emitBlockedLoadFromSmem(llvm::StringRef dstVar,
                                           llvm::StringRef smemVar,
                                           ttg::BlockedEncodingAttr enc,
                                           llvm::ArrayRef<int64_t> shape,
                                           llvm::StringRef elemType,
                                           int nElems, int rowLo, int rowHi) {
  // Optional row window [rowLo, rowHi): the smem buffer holds only those rows
  // (chunked convert_layout scratch) — guard accesses and rebase the row.
  bool windowed = rowHi >= 0;
  if (shape.size() == 1) {
    int spt = enc.getSizePerThread()[0];
    int tpw = enc.getThreadsPerWarp()[0];
    int stride = tpw * enc.getWarpsPerCTA()[0] * spt;
    emit("{");
    indent();
    emit("int _base = (lane_id % " + std::to_string(tpw) + " + warp_id * " +
         std::to_string(tpw) + ") * " + std::to_string(spt) + ";");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    if (nElems == spt)
      emit("    " + dstVar.str() + "[_i] = " + smemVar.str() + "[_base + _i];");
    else
      emit("    " + dstVar.str() + "[_i] = " + smemVar.str() + "[_base + (_i % " + std::to_string(spt) +
           ") + (_i / " + std::to_string(spt) + ") * " + std::to_string(stride) + "];");
    dedent();
    emit("}");
  } else if (shape.size() >= 2) {
    // Per-CTA register convention — see emitBlockedStoreToSmem.
    auto shapePerCTA = ttg::getShapePerCTA(enc, shape);
    int rows = shapePerCTA[0], cols = shapePerCTA[1];
    int spt0 = enc.getSizePerThread()[0], spt1 = enc.getSizePerThread()[1];
    int tpw0 = enc.getThreadsPerWarp()[0], tpw1 = enc.getThreadsPerWarp()[1];
    int wpc0 = enc.getWarpsPerCTA()[0], wpc1 = enc.getWarpsPerCTA()[1];
    int totalT0 = tpw0 * wpc0;
    int reps1 = std::max(1, cols / (tpw1 * wpc1 * spt1));
    int reps0 = std::max(1, rows / (tpw0 * wpc0 * spt0));
    int strideRep0 = spt1 * reps1 * spt0;
    int totalT1 = tpw1 * wpc1;
    // order[0]==0 means dim0 is the fastest-varying dim (column-major
    // register/lane/warp decomposition) — mirror emitBlockedStoreToSmem.
    bool dim0Fast = (enc.getOrder()[0] == 0);

    // Determine element size for vectorized loads
    int elemBits = 16; // default
    if (elemType == "float" || elemType == "int" || elemType == "uint32_t")
      elemBits = 32;
    else if (elemType == "__half" || elemType == "__nv_bfloat16")
      elemBits = 16;
    int elemBytesLocal = elemBits / 8;

    // Try vectorized load: spt1 contiguous elements in the inner dimension
    // (invalid when dim0Fast: registers are not contiguous along dim1).
    int vecLoadBytes = spt1 * elemBytesLocal;
    bool useVecLoad =
        !dim0Fast && (spt1 >= 4) && (vecLoadBytes == 8 || vecLoadBytes == 16);
    int nGroups = nElems / spt1; // number of vectorized load groups

    emit("{");
    indent();
    if (dim0Fast) {
      emit("int _ld0 = lane_id % " + std::to_string(tpw0) + ";");
      emit("int _ld1 = lane_id / " + std::to_string(tpw0) + ";");
      emit("int _wd0 = warp_id % " + std::to_string(wpc0) + ";");
      emit("int _wd1 = warp_id / " + std::to_string(wpc0) + ";");
    } else {
      emit("int _ld0 = lane_id / " + std::to_string(tpw1) + ";");
      emit("int _ld1 = lane_id % " + std::to_string(tpw1) + ";");
      emit("int _wd0 = warp_id / " + std::to_string(wpc1) + ";");
      emit("int _wd1 = warp_id % " + std::to_string(wpc1) + ";");
    }
    emit("int _bp0 = _ld0 + _wd0 * " + std::to_string(tpw0) + ";");
    emit("int _bp1 = _ld1 + _wd1 * " + std::to_string(tpw1) + ";");

    if (useVecLoad && elemBytesLocal == 2) {
      // Vectorized load: read spt1 contiguous f16/bf16 elements at once
      std::string vecType = (vecLoadBytes == 16) ? "uint4" : "uint2";
      int nU32 = vecLoadBytes / 4;
      emit("// Vectorized shared load: " + std::to_string(spt1) + " x " +
           elemType.str() + " per load (" + std::to_string(nGroups) + " loads)");
      emit("#pragma unroll");
      emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
      indent();
      // Compute row and column base for this group
      emit("int _gi = _g * " + std::to_string(spt1) + ";");
      emit("int _rp0 = _gi / " + std::to_string(strideRep0) + ";");
      emit("int _s0 = (_gi / " + std::to_string(spt1 * reps1) + ") % " + std::to_string(spt0) + ";");
      emit("int _rp1 = (_gi / " + std::to_string(spt1) + ") % " + std::to_string(reps1) + ";");
      emit("int _row = _bp0 * " + std::to_string(spt0) + " + _s0 + _rp0 * " + std::to_string(totalT0 * spt0) + ";");
      emit("int _col = _bp1 * " + std::to_string(spt1) + " + _rp1 * " + std::to_string(totalT1 * spt1) + ";");
      if (windowed) {
        emit("if (_row >= " + std::to_string(rowLo) + " && _row < " +
             std::to_string(rowHi) + ") {");
        indent();
        emit("_row -= " + std::to_string(rowLo) + ";");
      }
      emit(vecType + " _ld = *(" + vecType + "*)&" + smemVar.str() +
           "[_row * " + std::to_string(cols) + " + _col];");
      // Unpack uint4/uint2 into f16 elements
      for (int u = 0; u < nU32; u++) {
        std::string field;
        if (nU32 == 1) field = "_ld";
        else if (nU32 == 2) field = (u == 0) ? "_ld.x" : "_ld.y";
        else field = (u == 0) ? "_ld.x" : (u == 1) ? "_ld.y" : (u == 2) ? "_ld.z" : "_ld.w";
        int e0 = u * 2, e1 = u * 2 + 1;
        emit("*(uint16_t*)&" + dstVar.str() + "[_gi+" + std::to_string(e0) +
             "] = (uint16_t)" + field + ";");
        emit("*(uint16_t*)&" + dstVar.str() + "[_gi+" + std::to_string(e1) +
             "] = (uint16_t)(" + field + " >> 16);");
      }
      if (windowed) {
        dedent();
        emit("}");
      }
      dedent();
      emit("}");
    } else {
      // Scalar fallback
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
      indent();
      if (dim0Fast) {
        emit("int _s0 = _i % " + std::to_string(spt0) + ";");
        emit("int _rp0 = (_i / " + std::to_string(spt0) + ") % " + std::to_string(reps0) + ";");
        emit("int _s1 = (_i / " + std::to_string(spt0 * reps0) + ") % " + std::to_string(spt1) + ";");
        emit("int _rp1 = _i / " + std::to_string(spt0 * reps0 * spt1) + ";");
      } else {
        emit("int _rp0 = _i / " + std::to_string(strideRep0) + ";");
        emit("int _s0 = (_i / " + std::to_string(spt1 * reps1) + ") % " + std::to_string(spt0) + ";");
        emit("int _rp1 = (_i / " + std::to_string(spt1) + ") % " + std::to_string(reps1) + ";");
        emit("int _s1 = _i % " + std::to_string(spt1) + ";");
      }
      emit("int _row = _bp0 * " + std::to_string(spt0) + " + _s0 + _rp0 * " + std::to_string(totalT0 * spt0) + ";");
      emit("int _col = _bp1 * " + std::to_string(spt1) + " + _s1 + _rp1 * " + std::to_string(totalT1 * spt1) + ";");
      if (windowed)
        emit("if (_row >= " + std::to_string(rowLo) + " && _row < " +
             std::to_string(rowHi) + ") " + dstVar.str() + "[_i] = " +
             smemVar.str() + "[(_row - " + std::to_string(rowLo) + ") * " +
             std::to_string(cols) + " + _col];");
      else
        emit(dstVar.str() + "[_i] = " + smemVar.str() + "[_row * " + std::to_string(cols) + " + _col];");
      dedent();
      emit("}");
    }
    dedent();
    emit("}");
  } else {
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + dstVar.str() + "[_i] = " + smemVar.str() + "[tid * " + std::to_string(nElems) + " + _i];");
  }
}
std::string CUDACodeGen::getBroadcastIndexExpr(Value src,
                                                RankedTensorType dstTy,
                                                int srcN, int dstN) {
  return "_i % " + std::to_string(srcN);
}
void CUDACodeGen::emitSwizzledStoreToSmem(llvm::StringRef srcVar,
                                           llvm::StringRef dstVar,
                                           ttg::BlockedEncodingAttr srcEnc,
                                           llvm::ArrayRef<int64_t> shape,
                                           llvm::StringRef elemType,
                                           int swizzleByteWidth) {
  emit("// TODO: emitSwizzledStoreToSmem not yet ported to C++");
}

#undef STUB_EMIT

} // namespace mlir::triton
