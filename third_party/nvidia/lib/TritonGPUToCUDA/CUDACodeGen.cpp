/// CUDACodeGen: Core code generation from TTGIR/NVGPUIR to CUDA C++ source.
///
/// This translates MLIR operations to CUDA C++ code by walking the IR and
/// emitting equivalent C++ for each operation. Unlike the Python regex-based
/// emitter, this directly accesses MLIR's typed op/attribute/type API.

#include "CUDACodeGen.h"
#include <cmath>
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
                         int32_t numCtas)
    : module(module), capability(capability), numWarps(numWarps),
      numCtas(numCtas) {}

// ═══════════════════════════════════════════════════════════════════════
// Emit infrastructure
// ═══════════════════════════════════════════════════════════════════════

void CUDACodeGen::emit(const llvm::Twine &line) {
  std::string s;
  for (int i = 0; i < indentLevel; i++)
    s += "    ";
  s += line.str();
  lines.push_back(std::move(s));
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
  auto it = valueToVar.find(val);
  if (it != valueToVar.end())
    return it->second;
  // Auto-create for block arguments
  auto var = newVar("arg");
  valueToVar[val] = var;
  return var;
}

std::string CUDACodeGen::getElemExpr(Value val, const std::string &idx) {
  auto var = getVar(val);
  if (scalarValues.contains(val))
    return var; // scalar — no indexing needed
  // Check deferred addptr — return base + offset[idx] expression
  auto it = deferredAddPtr.find(val);
  if (it != deferredAddPtr.end())
    return "(" + it->second.first + " + " + it->second.second + "[" + idx + "])";
  return var + "[" + idx + "]";
}

std::string CUDACodeGen::getPtrElemExpr(Value val, const std::string &idx) {
  // For deferred addptr, dereference: *(base + offset[idx])
  auto it = deferredAddPtr.find(val);
  if (it != deferredAddPtr.end())
    return "*(" + it->second.first + " + " + it->second.second + "[" + idx + "])";
  if (scalarValues.contains(val))
    return "*" + getVar(val);
  return "*" + getVar(val) + "[" + idx + "]";
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

bool CUDACodeGen::isTensorValue(Value val) {
  return isa<RankedTensorType>(val.getType());
}

llvm::ArrayRef<int64_t> CUDACodeGen::getTensorShape(Value val) {
  if (auto rtt = dyn_cast<RankedTensorType>(val.getType()))
    return rtt.getShape();
  return {};
}

int CUDACodeGen::getElemsPerThread(Value val) {
  if (auto rtt = dyn_cast<RankedTensorType>(val.getType()))
    return getElemsPerThread(rtt);
  return 1;
}

int CUDACodeGen::getElemsPerThread(RankedTensorType ty) {
  auto encoding = ty.getEncoding();
  auto shape = ty.getShape();

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
    int result = 1;
    for (unsigned d = 0; d < shape.size(); d++) {
      int spt = blocked.getSizePerThread()[d];
      int tpw = blocked.getThreadsPerWarp()[d];
      int wpc = blocked.getWarpsPerCTA()[d];
      int threadsInDim = tpw * wpc;
      if (shape[d] == 1) {
        result *= 1;
      } else {
        int elems = std::max(1L, shape[d] / (threadsInDim * spt)) * spt;
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
      if (slice.getDim() == 1)
        return 2;
    }
  }

  if (auto mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(encoding)) {
    if (shape.size() >= 2) {
      int64_t M = shape[0], N = shape[1];
      return std::max({M * N / 128, M / 32, (int64_t)1});
    }
  }

  // DotOperandEncoding: use LinearLayout for correct element count
  if (isa<ttg::DotOperandEncodingAttr>(encoding)) {
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

  // Walk functions in the module
  module.walk([&](tt::FuncOp func) {
    kernelName = func.getName().str();
    int numThreads = numWarps * 32;

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

    emit("extern \"C\" __global__ void __launch_bounds__(" +
         llvm::Twine(numThreads) + ")");
    emit(kernelName + "(" + argList + ") {");
    indent();

    emit("// Thread indexing");
    emit("const int tid = threadIdx.x;");
    emit("const int warp_id = tid / 32;");
    emit("const int lane_id = tid % 32;");
    emitBlank();
    emit("extern __shared__ char shared_mem[];");
    emitBlank();

    // Emit function body
    for (auto &block : func.getBody()) {
      emitBlock(block);
    }

    emit("return;");
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

  return result;
}

void CUDACodeGen::emitBlock(Block &block) {
  for (auto &op : block) {
    emitOp(&op);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Op dispatch — TypeSwitch over all supported ops
// ═══════════════════════════════════════════════════════════════════════

void CUDACodeGen::emitOp(Operation *op) {
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

  // Math ops
  if (op->getDialect()->getNamespace() == "math")
    return emitMathOp(op);

  // Triton core ops
  if (auto getPidOp = dyn_cast<tt::GetProgramIdOp>(op))
    return emitGetProgramId(getPidOp);
  if (auto getNumOp = dyn_cast<tt::GetNumProgramsOp>(op)) {
    auto var = newVar("np");
    valueToVar[getNumOp.getResult()] = var;
    auto axis = getNumOp.getAxisAsInt();
    std::string dim = axis == 0 ? "x" : (axis == 1 ? "y" : "z");
    emit("const int " + var + " = gridDim." + dim + ";");
    return;
  }
  if (auto makeRangeOp = dyn_cast<tt::MakeRangeOp>(op))
    return emitMakeRange(makeRangeOp);
  if (auto splatOp = dyn_cast<tt::SplatOp>(op))
    return emitSplat(splatOp);
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
  if (auto externOp = dyn_cast<tt::ExternElementwiseOp>(op))
    return emitExternElementwise(externOp);
  if (auto atomicOp = dyn_cast<tt::AtomicRMWOp>(op))
    return emitAtomicRMW(atomicOp);
  if (auto casOp = dyn_cast<tt::AtomicCASOp>(op))
    return emitAtomicCAS(casOp);
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
    if (nDst == nSrc) {
      emit(cudaType + "* " + var + " = " + srcVar + "; // reshape (alias)");
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
      emit("__syncthreads();");
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
      emit("__syncthreads();");
      dedent();
      emit("}");
    }
    return;
  }
  if (auto printOp = dyn_cast<tt::PrintOp>(op))
    return emitPrint(printOp);
  if (auto mulhiOp = dyn_cast<tt::MulhiUIOp>(op))
    return emitMulhiUI(mulhiOp);

  // SCF control flow
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return emitScfFor(forOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return emitScfIf(ifOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return emitScfWhile(whileOp);
  if (isa<scf::YieldOp, scf::ConditionOp>(op))
    return; // Handled by parent op

  // GPU barrier
  if (isa<mlir::gpu::BarrierOp>(op)) {
    emit("__syncthreads();");
    return;
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
  if (auto tmaCopyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op))
    return emitAsyncTMACopyG2L(tmaCopyOp);
  if (auto tmaCopyOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return emitAsyncTMACopyL2G(tmaCopyOp);
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

  // Unhandled op — emit a TODO comment with full op name
  emit("// TODO: unhandled op: " +
       llvm::Twine(op->getName().getStringRef()) + " [" +
       llvm::Twine(op->getName().getDialectNamespace()) + "]");
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

        emit(cudaType + " " + var + "[" + llvm::Twine(nElems).str() + "];");
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + llvm::Twine(nElems).str() +
             "; _i++) " + var + "[_i] = " + val + ";");
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
  auto var = newVar("a");
  valueToVar[result] = var;

  auto lhsVar = getVar(op->getOperand(0));
  auto rhsVar = getVar(op->getOperand(1));

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
           opName.contains("maxsi"))
    cppOp = "max";
  else if (opName.contains("minnumf") || opName.contains("minimumf") ||
           opName.contains("minsi"))
    cppOp = "min";

  auto cudaType = getCUDAType(
      isTensor
          ? cast<RankedTensorType>(result.getType()).getElementType()
          : result.getType());

  if (isTensor) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    // If both inputs are scalar (splat), result is also scalar
    bool lhsScalar = scalarValues.contains(lhs);
    bool rhsScalar = scalarValues.contains(rhs);
    bool resultScalar = lhsScalar && rhsScalar;

    int nElems = getElemsPerThread(result);
    std::string lhsE = getElemExpr(lhs, "_i");
    std::string rhsE = getElemExpr(rhs, "_i");

    if (resultScalar) {
      scalarValues.insert(result);
      // Scalar op — no array needed
      if (cppOp == "ceildiv") {
        emit("const " + cudaType + " " + var + " = ((" + lhsE +
             ") + (" + rhsE + ") - 1) / (" + rhsE + ");");
      } else if (cppOp == "floordiv") {
        emit("const " + cudaType + " " + var + " = (" + lhsE + " / " + rhsE +
             ") - ((" + lhsE + " % " + rhsE + " != 0) && ((" + lhsE +
             " ^ " + rhsE + ") < 0));");
      } else if (cppOp == "max" || cppOp == "min" || cppOp == "fmod") {
        std::string fn = cppOp;
        if (cudaType == "float") {
          if (cppOp == "max") fn = "fmaxf";
          else if (cppOp == "min") fn = "fminf";
          else if (cppOp == "fmod") fn = "fmodf";
        }
        emit("const " + cudaType + " " + var + " = " + fn + "(" + lhsE + ", " + rhsE + ");");
      } else {
        emit("const " + cudaType + " " + var + " = (" + lhsE + " " + cppOp + " " + rhsE + ");");
      }
    } else {
      emit(cudaType + " " + var + "[" + llvm::Twine(nElems).str() + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + llvm::Twine(nElems).str() +
           "; _i++)");
      if (cppOp == "ceildiv") {
        emit("    " + var + "[_i] = ((" + lhsE + ") + (" + rhsE + ") - 1) / (" + rhsE + ");");
      } else if (cppOp == "floordiv") {
        emit("    " + var + "[_i] = (" + lhsE + " / " + rhsE +
             ") - ((" + lhsE + " % " + rhsE + " != 0) && ((" + lhsE +
             " ^ " + rhsE + ") < 0));");
      } else if (cppOp == "max" || cppOp == "min" || cppOp == "fmod") {
        std::string fn = cppOp;
        if (cudaType == "float") {
          if (cppOp == "max") fn = "fmaxf";
          else if (cppOp == "min") fn = "fminf";
          else if (cppOp == "fmod") fn = "fmodf";
        }
        emit("    " + var + "[_i] = " + fn + "(" + lhsE + ", " + rhsE + ");");
      } else {
        // For unsigned integer ops, cast operands to unsigned type
        bool isUnsignedOp = isa<arith::DivUIOp, arith::RemUIOp, arith::ShRUIOp,
                                arith::MaxUIOp, arith::MinUIOp>(op);
        if (isUnsignedOp) {
          Type elemType = cast<RankedTensorType>(result.getType()).getElementType();
          auto utype = getUnsignedType(elemType);
          emit("    " + var + "[_i] = (" + cudaType + ")((" + utype + ")" +
               lhsE + " " + cppOp + " (" + utype + ")" + rhsE + ");");
        } else {
          emit("    " + var + "[_i] = (" + lhsE + " " + cppOp + " " + rhsE + ");");
        }
      }
    }
  } else {
    if (cppOp == "ceildiv") {
      emit("const " + cudaType + " " + var + " = ((" + lhsVar +
           ") + (" + rhsVar + ") - 1) / (" + rhsVar + ");");
    } else if (cppOp == "floordiv") {
      // floordiv for signed: a/b rounded towards -inf
      emit("const " + cudaType + " " + var + " = (" + lhsVar + " / " + rhsVar +
           ") - ((" + lhsVar + " % " + rhsVar + " != 0) && ((" + lhsVar +
           " ^ " + rhsVar + ") < 0));");
    } else if (cppOp == "max" || cppOp == "min" || cppOp == "fmod") {
      std::string fn = cppOp;
      if (cudaType == "float") {
        if (cppOp == "max") fn = "fmaxf";
        else if (cppOp == "min") fn = "fminf";
        else if (cppOp == "fmod") fn = "fmodf";
      }
      emit("const " + cudaType + " " + var + " = " + fn + "(" + lhsVar +
           ", " + rhsVar + ");");
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
  auto var = newVar("a");
  valueToVar[result] = var;

  auto srcVar = getVar(src);
  Type resultElemType =
      isTensor ? cast<RankedTensorType>(result.getType()).getElementType()
               : result.getType();
  auto cudaType = getCUDAType(resultElemType);

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    bool srcScalar = scalarValues.contains(src);
    std::string srcE = getElemExpr(src, "_i");

    if (srcScalar) {
      scalarValues.insert(result);
      if (isa<arith::ExtUIOp, arith::UIToFPOp, arith::IndexCastUIOp>(op)) {
        Type srcElemType =
            cast<RankedTensorType>(src.getType()).getElementType();
        auto unsignedSrc = getUnsignedType(srcElemType);
        emit("const " + cudaType + " " + var + " = ((" + cudaType + ")(" + unsignedSrc + ")" + srcE + ");");
      } else {
        emit("const " + cudaType + " " + var + " = ((" + cudaType + ")" + srcE + ");");
      }
    } else {
      emit(cudaType + " " + var + "[" + llvm::Twine(nElems).str() + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + llvm::Twine(nElems).str() + "; _i++)");
      if (isa<arith::ExtUIOp, arith::UIToFPOp, arith::IndexCastUIOp>(op)) {
        Type srcElemType =
            cast<RankedTensorType>(src.getType()).getElementType();
        auto unsignedSrc = getUnsignedType(srcElemType);
        emit("    " + var + "[_i] = ((" + cudaType + ")(" + unsignedSrc + ")" + srcE + ");");
      } else {
        emit("    " + var + "[_i] = ((" + cudaType + ")" + srcE + ");");
      }
    }
  } else {
    if (isa<arith::ExtUIOp, arith::UIToFPOp, arith::IndexCastUIOp>(op)) {
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
  auto var = newVar("a");
  valueToVar[result] = var;

  auto lhsVar = getVar(op->getOperand(0));
  auto rhsVar = getVar(op->getOperand(1));

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

    std::string lhsE = getElemExpr(lhs, "_i");
    std::string rhsE = getElemExpr(rhs, "_i");
    if (isUnsignedCmp) {
      Type lhsElemType = cast<RankedTensorType>(lhs.getType()).getElementType();
      auto utype = getUnsignedType(lhsElemType);
      lhsE = "(" + utype + ")" + lhsE;
      rhsE = "(" + utype + ")" + rhsE;
    }

    int nElems = getElemsPerThread(result);
    if (resultScalar) {
      scalarValues.insert(result);
      emit("const bool " + var + " = (" + lhsE + " " + cmpOp + " " + rhsE + ");");
    } else {
      emit("bool " + var + "[" + llvm::Twine(nElems).str() + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + llvm::Twine(nElems).str() + "; _i++)");
      emit("    " + var + "[_i] = (" + lhsE + " " + cmpOp + " " + rhsE + ");");
    }
  } else {
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

void CUDACodeGen::emitArithSelect(arith::SelectOp op) {
  auto result = op.getResult();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("a");
  valueToVar[result] = var;

  auto condVar = getVar(op.getCondition());
  auto trueVar = getVar(op.getTrueValue());
  auto falseVar = getVar(op.getFalseValue());

  Type elemType = isTensor
                      ? cast<RankedTensorType>(result.getType()).getElementType()
                      : result.getType();
  auto cudaType = getCUDAType(elemType);

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    // Condition may be a scalar bool (i1) even when true/false are tensors
    bool condIsTensor = isa<RankedTensorType>(op.getCondition().getType());
    bool condIsScalar = !condIsTensor || scalarValues.contains(op.getCondition());
    std::string condExpr = condIsScalar ? condVar : (condVar + "[_i]");
    std::string trueExpr = scalarValues.contains(op.getTrueValue())
                               ? trueVar : (trueVar + "[_i]");
    std::string falseExpr = scalarValues.contains(op.getFalseValue())
                                ? falseVar : (falseVar + "[_i]");
    emit(cudaType + " " + var + "[" + llvm::Twine(nElems).str() + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + llvm::Twine(nElems).str() + "; _i++)");
    emit("    " + var + "[_i] = (" + condExpr + " ? " + trueExpr +
         " : " + falseExpr + ");");
  } else {
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
  auto var = newVar("m");
  valueToVar[result] = var;

  auto srcVar = getVar(src);
  llvm::StringRef opName = op->getName().getStringRef();

  // Map math ops to CUDA functions
  std::string fn = "unknown";
  if (opName.contains("exp2"))
    fn = "exp2f";
  else if (opName.contains("exp"))
    fn = "expf";
  else if (opName.contains("log2"))
    fn = "log2f";
  else if (opName.contains("log"))
    fn = "logf";
  else if (opName.contains("sqrt"))
    fn = "sqrtf";
  else if (opName.contains("rsqrt"))
    fn = "rsqrtf";
  else if (opName.contains("abs"))
    fn = "fabsf";
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

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    emit(cudaType + " " + var + "[" + llvm::Twine(nElems).str() + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + llvm::Twine(nElems).str() + "; _i++)");
    emit("    " + var + "[_i] = " + fn + "(" + srcVar + "[_i]);");
  } else {
    emit("const " + cudaType + " " + var + " = " + fn + "(" + srcVar + ");");
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

  // Use LinearLayout to compute per-register values as f(register, lane, warp)
  auto ll = ttg::toLinearLayout(rtt);
  auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
  auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
  auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
  auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");
  const auto &bases = ll.getBases();
  const auto &laneBases = bases.find(kLane)->second;
  const auto &warpBases = bases.find(kWarp)->second;

  emit("int " + var + "[" + llvm::Twine(nElems).str() + "];");
  emit("{");
  indent();

  for (int i = 0; i < nElems; i++) {
    auto coords = ll.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    int regBase = coords[0].second; // dim0 coordinate at lane=0, warp=0

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

      // Compute broadcast index expression
      std::string idxExpr = "_i % " + std::to_string(srcN);
      auto dstEnc2 = rtt.getEncoding();
      if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(dstEnc2)) {
        if (rtt.getRank() >= 2) {
          auto spt1 = blocked.getSizePerThread()[1];
          auto tpw1 = blocked.getThreadsPerWarp()[1];
          auto wpc1 = blocked.getWarpsPerCTA()[1];
          int totalT1 = tpw1 * wpc1;
          int reps1 = std::max((int64_t)1, rtt.getShape()[1] / (totalT1 * spt1));
          int elemsInner = spt1 * reps1;
          if (srcRtt.getShape()[1] == 1 && rtt.getShape()[1] > 1)
            idxExpr = "_i / " + std::to_string(elemsInner);
          else if (srcRtt.getShape()[0] == 1 && rtt.getShape()[0] > 1)
            idxExpr = "_i % " + std::to_string(elemsInner);
        }
      }

      emit("int " + bcastOff + "[" + std::to_string(nElems) + "];");
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
           bcastOff + "[_i] = " + srcOff + "[" + idxExpr + "];");
      deferredAddPtr[result] = {base, bcastOff};
      return;
    }
  }

  auto var = newVar("bcast");
  valueToVar[result] = var;

  auto srcVar = getVar(src);
  auto cudaType = getCUDAType(rtt.getElementType());

  emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");

  // For non-3D cases, use the original broadcast logic (well-tested for 2D)
  auto dstEnc = rtt.getEncoding();
  if (rtt.getRank() <= 2) {
    emit("#pragma unroll");
    std::string idxExpr = "_i % " + std::to_string(srcN);
    if (isa<ttg::NvidiaMmaEncodingAttr>(dstEnc)) {
      if (srcN == 2 && nElems > 2)
        idxExpr = "(_i >> 1) & 1";
    } else if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(dstEnc)) {
      if (rtt.getRank() >= 2) {
        auto spt1 = blocked.getSizePerThread()[1];
        auto tpw1 = blocked.getThreadsPerWarp()[1];
        auto wpc1 = blocked.getWarpsPerCTA()[1];
        int totalT1 = tpw1 * wpc1;
        int reps1 = std::max((int64_t)1, rtt.getShape()[1] / (totalT1 * spt1));
        int elemsInner = spt1 * reps1;
        if (srcRtt.getShape()[1] == 1 && rtt.getShape()[1] > 1)
          idxExpr = "_i / " + std::to_string(elemsInner);
        else if (srcRtt.getShape()[0] == 1 && rtt.getShape()[0] > 1)
          idxExpr = "_i % " + std::to_string(elemsInner);
      }
    }
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) " +
         var + "[_i] = " + srcVar + "[" + idxExpr + "];");
    return;
  }

  // For 3D+ cases, use LinearLayout for correct mapping
  // Use LinearLayout to compute the exact mapping from dst register to src register
  // Broadcast: src and dst have same encoding but different shapes (some src dims are 1)
  auto dstLL = ttg::toLinearLayout(rtt);
  auto srcLL = ttg::toLinearLayout(srcRtt);
  auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
  auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
  auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
  auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");

  // Build mapping: for each dst register, find which src register has the same
  // coordinates on the non-broadcast dimensions
  // The broadcast dims have src.shape[d] == 1, so we modulo the dst coord by src.shape[d]
  auto srcShape = srcRtt.getShape();

  // For each dst register, compute its tensor coords, modulo src shape, then find src register
  // Pre-compute src register → coord map
  std::map<SmallVector<int>, int> srcCoordToReg;
  for (int i = 0; i < srcN; i++) {
    auto coords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<int> key;
    for (auto &c : coords) key.push_back(c.second);
    srcCoordToReg[key] = i;
  }

  // Emit per-element mapping
  emit("#pragma unroll");
  bool usedLoop = false;
  // Check if a simple expression works (common case: 2D with clear broadcast pattern)
  // Otherwise use per-element constants
  SmallVector<int> dstToSrc(nElems, 0);
  bool allSame = true;
  for (int i = 0; i < nElems; i++) {
    auto dstCoords = dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<int> srcKey;
    for (size_t d = 0; d < dstCoords.size(); d++)
      srcKey.push_back(dstCoords[d].second % std::max((int64_t)1, srcShape[d]));
    auto it = srcCoordToReg.find(srcKey);
    dstToSrc[i] = (it != srcCoordToReg.end()) ? it->second : 0;
    if (dstToSrc[i] != dstToSrc[0]) allSame = false;
  }

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

  auto srcVar = getVar(src);
  auto cudaType = getCUDAType(rtt.getElementType());

  if (nElems == srcN) {
    emit(cudaType + "* " + var + " = " + srcVar + "; // expand_dims (alias)");
  } else {
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");

    // Use LinearLayout to compute mapping from dst register to src register
    int axis = op.getAxis();
    auto dstLL = ttg::toLinearLayout(rtt);
    auto srcLL = ttg::toLinearLayout(srcRtt);
    auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");

    // Build src register → coord mapping (without the expanded dim)
    std::map<SmallVector<int>, int> srcCoordToReg;
    for (int i = 0; i < srcN; i++) {
      auto coords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      SmallVector<int> key;
      for (auto &c : coords) key.push_back(c.second);
      srcCoordToReg[key] = i;
    }

    // For each dst register, strip the expanded dim and find src register
    SmallVector<int> dstToSrc(nElems, 0);
    for (int i = 0; i < nElems; i++) {
      auto dstCoords = dstLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      SmallVector<int> srcKey;
      for (int d = 0; d < (int)dstCoords.size(); d++) {
        if (d != axis) srcKey.push_back(dstCoords[d].second);
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
      // Both scalar → result is scalar
      scalarValues.insert(result);
      emit(cudaType + " " + var + " = " + ptrExpr + " + " + offExpr + ";");
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
    emit(cudaType + " " + var + " = " + ptrVar + " + " + offVar + ";");
  }
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
      if (isa_and_nonnull<ttg::NVMMASharedEncodingAttr>(destEnc)) {
        // Defer this load — will be fused with local_alloc via cp.async
        deferredCpAsyncLoads.insert(result);
        return;
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

    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    std::string ptrDeref = getPtrElemExpr(ptr, "_i");
    if (mask) {
      std::string maskE = getElemExpr(mask, "_i");
      if (other) {
        std::string otherE = getElemExpr(other, "_i");
        emit("if (" + maskE + ") {");
        emit("    " + var + "[_i] = " + ptrDeref + ";");
        emit("} else {");
        emit("    " + var + "[_i] = " + otherE + ";");
        emit("}");
      } else {
        emit("if (" + maskE + ")");
        emit("    " + var + "[_i] = " + ptrDeref + ";");
      }
    } else {
      emit(var + "[_i] = " + ptrDeref + ";");
    }
    dedent();
    emit("}");
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

    // Try vectorized store: blocked layout with contiguous inner dimension
    int vecWidth = 1;
    if (auto blk = dyn_cast<ttg::BlockedEncodingAttr>(valRtt.getEncoding())) {
      auto spt = blk.getSizePerThread();
      if (spt.size() >= 2) {
        int innerSpt = spt[spt.size() - 1];
        // Maximum 16 bytes per vectorized store (uint4)
        int maxVec = 16 / elemBytes;
        vecWidth = std::min(innerSpt, maxVec);
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

    // Helper to emit a predicated PTX store via setp + @p
    auto emitPtxStore = [&](const std::string &ptxOp, int nU32,
                            bool hasMask) {
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
          args += ", \"r\"(_r" + std::to_string(i) + ")";
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
          args += ", \"r\"(_r" + std::to_string(i) + ")";
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
        if (nU32 == 1) ptxOp = "st.global.b32";
        else if (nU32 == 2) ptxOp = "st.global.v2.b32";
        else ptxOp = "st.global.v4.b32";

        emit("// Vectorized store via PTX: " + std::to_string(vecWidth) + " x " +
             getCUDAType(elemType) + " per store (" + std::to_string(nGroups) + " stores)");
        emit("#pragma unroll");
        emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
        indent();
        emit("int _base = _g * " + std::to_string(vecWidth) + ";");
        // Pack elements into uint32 registers
        for (int i = 0; i < nU32; i++) {
          int e0 = i * 2, e1 = i * 2 + 1;
          emit("uint32_t _r" + std::to_string(i) + " = "
               "*(uint16_t*)&" + valVar + "[_base+" + std::to_string(e0) + "] | "
               "((uint32_t)*(uint16_t*)&" + valVar + "[_base+" + std::to_string(e1) + "] << 16);");
        }
        if (mask) {
          auto maskVar = getVar(mask);
          emit("int _pred = (int)" + maskVar + "[_base + " + std::to_string(vecWidth - 1) + "];");
        }
        emitPtxStore(ptxOp, nU32, mask != nullptr);
        dedent();
        emit("}");
        return;
      }
    }
    if (vecWidth >= 2 && elemType.isF32()) {
      // Vectorized store for f32 using inline PTX with predication
      int nGroups = nElems / vecWidth;
      std::string ptxOp;
      if (vecWidth == 2) ptxOp = "st.global.v2.b32";
      else ptxOp = "st.global.v4.b32";

      emit("// Vectorized store via PTX: " + std::to_string(vecWidth) + " x f32 per store");
      emit("#pragma unroll");
      emit("for (int _g = 0; _g < " + std::to_string(nGroups) + "; _g++) {");
      indent();
      emit("int _base = _g * " + std::to_string(vecWidth) + ";");
      // Bitcast f32 to uint32 for PTX
      for (int i = 0; i < vecWidth; i++) {
        emit("uint32_t _r" + std::to_string(i) + " = "
             "__float_as_uint(" + valVar + "[_base+" + std::to_string(i) + "]);");
      }
      if (mask) {
        auto maskVar = getVar(mask);
        emit("int _pred = (int)" + maskVar + "[_base + " + std::to_string(vecWidth - 1) + "];");
      }
      emitPtxStore(ptxOp, vecWidth, mask != nullptr);
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
      if (mask) {
        std::string maskE = getElemExpr(mask, "_i");
        emit("if (" + maskE + ")");
        emit("    " + ptrDeref + " = " + valVar + "[_i];");
      } else {
        emit(ptrDeref + " = " + valVar + "[_i];");
      }
    }
    dedent();
    emit("}");
  } else {
    emit("*" + ptrVar + " = " + valVar + ";");
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
  int M = rtt.getShape()[0], N = rtt.getShape()[1];
  int K = aRtt.getShape()[1];
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

  auto var = newVar("dot");
  valueToVar[result] = var;

  emit("// tt.dot via shared memory FMA (" + std::to_string(M) + "x" +
       std::to_string(N) + "x" + std::to_string(K) + ")");
  emit(outType + " " + var + "[" + std::to_string(nOut) + "];");
  emit("#pragma unroll");
  emit("for (int _i = 0; _i < " + std::to_string(nOut) + "; _i++) " +
       var + "[_i] = " + accVar + "[_i];");
  emit("{");
  indent();

  // Allocate shared memory for A and B
  int aBytes = getTypeSizeInBytes(aElemType);
  int bBytes = getTypeSizeInBytes(bElemType);
  int smemOffA = (sharedMemOffset + 127) & ~127;
  int smemOffB = (smemOffA + M * K * aBytes + 127) & ~127;
  sharedMemOffset = smemOffB + K * N * bBytes;
  if (sharedMemOffset > peakSharedMem)
    peakSharedMem = sharedMemOffset;

  emit(aType + "* _dotA = (" + aType + "*)(shared_mem + " +
       std::to_string(smemOffA) + ");");
  emit(bType + "* _dotB = (" + bType + "*)(shared_mem + " +
       std::to_string(smemOffB) + ");");

  // Store A to shared memory using LinearLayout
  {
    auto aLL = ttg::toLinearLayout(aRtt);
    auto kReg = mlir::StringAttr::get(aRtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(aRtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(aRtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(aRtt.getContext(), "block");
    auto aShape = aRtt.getShape();
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
    emit("_dotA[_aoff] = " + aVar + "[_i];");
    dedent();
    emit("}");
  }

  // Store B to shared memory using LinearLayout
  {
    auto bLL = ttg::toLinearLayout(bRtt);
    auto kReg = mlir::StringAttr::get(bRtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(bRtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(bRtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(bRtt.getContext(), "block");
    auto bShape = bRtt.getShape();
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
    emit("_dotB[_boff] = " + bVar + "[_i];");
    dedent();
    emit("}");
  }

  emit("__syncthreads();");

  // Compute output elements: each thread reads A and B from shared memory
  {
    auto outLL = ttg::toLinearLayout(rtt);
    auto kReg = mlir::StringAttr::get(rtt.getContext(), "register");
    auto kLane = mlir::StringAttr::get(rtt.getContext(), "lane");
    auto kWarp = mlir::StringAttr::get(rtt.getContext(), "warp");
    auto kBlock = mlir::StringAttr::get(rtt.getContext(), "block");
    auto outShape = rtt.getShape();
    const auto &outBases = outLL.getBases();
    const auto &outLaneBases = outBases.find(kLane)->second;
    const auto &outWarpBases = outBases.find(kWarp)->second;

    emit("// FMA: each thread accumulates its output elements");
    emit("#pragma unroll");
    emit("for (int _o = 0; _o < " + std::to_string(nOut) + "; _o++) {");
    indent();

    // Compute (row, col) for this output element
    emit("int _row, _col;");
    emit("switch (_o) {");
    for (int i = 0; i < nOut; i++) {
      auto coords =
          outLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
      // coords is [(dim0, val0), (dim1, val1)]
      int64_t row = coords[0].second;
      int64_t col = coords[1].second;
      emit("case " + std::to_string(i) + ": _row = " +
           std::to_string(row) + "; _col = " + std::to_string(col) +
           "; break;");
    }
    emit("default: _row = 0; _col = 0; break;");
    emit("}");

    // Add lane and warp contributions to row/col
    for (size_t lb = 0; lb < outLaneBases.size(); lb++) {
      int64_t rowDelta = outLaneBases[lb][0];
      int64_t colDelta = outLaneBases[lb][1];
      if (rowDelta != 0)
        emit("_row += ((lane_id >> " + std::to_string(lb) +
             ") & 1) * " + std::to_string(rowDelta) + ";");
      if (colDelta != 0)
        emit("_col += ((lane_id >> " + std::to_string(lb) +
             ") & 1) * " + std::to_string(colDelta) + ";");
    }
    for (size_t wb = 0; wb < outWarpBases.size(); wb++) {
      int64_t rowDelta = outWarpBases[wb][0];
      int64_t colDelta = outWarpBases[wb][1];
      if (rowDelta != 0)
        emit("_row += ((warp_id >> " + std::to_string(wb) +
             ") & 1) * " + std::to_string(rowDelta) + ";");
      if (colDelta != 0)
        emit("_col += ((warp_id >> " + std::to_string(wb) +
             ") & 1) * " + std::to_string(colDelta) + ";");
    }

    // Inner product: sum A[row][k] * B[k][col] for k in [0, K)
    emit(outType + " _sum = " + var + "[_o];");
    emit("#pragma unroll");
    emit("for (int _k = 0; _k < " + std::to_string(K) + "; _k++) {");
    indent();
    // Cast to output type for accumulation
    if (aElemType.isF16() || aElemType.isBF16()) {
      emit("_sum += (" + outType + ")_dotA[_row * " + std::to_string(K) +
           " + _k] * (" + outType + ")_dotB[_k * " + std::to_string(N) +
           " + _col];");
    } else {
      emit("_sum += _dotA[_row * " + std::to_string(K) +
           " + _k] * _dotB[_k * " + std::to_string(N) + " + _col];");
    }
    dedent();
    emit("}");
    emit(var + "[_o] = _sum;");

    dedent();
    emit("}");
  }
  emit("__syncthreads();");

  dedent();
  emit("}");
}

void CUDACodeGen::emitReduce(tt::ReduceOp op) {
  auto result = op->getResult(0);
  auto src = op.getOperands()[0];
  int axis = op.getAxis();
  int srcElems = getElemsPerThread(src);
  int numResults = op->getNumResults();
  auto var = newVar("red");
  valueToVar[result] = var;

  // For multi-result reduces (e.g., max-with-indices), map all results
  std::string var1;
  if (numResults > 1) {
    var1 = newVar("red");
    valueToVar[op->getResult(1)] = var1;
  }

  // Determine reduce op from combiner region
  std::string reduceOp = "add";
  bool hasSelect = false;
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
    else if (name.contains("cmpf") || name.contains("cmpi"))
      hasSelect = true;
  }
  // For multi-result reduces with compare+select, detect the comparison direction
  if (numResults > 1 && hasSelect) {
    for (auto &combOp : op.getCombineOp().front()) {
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
  } else if (reduceOp == "max") {
    if (srcElemType.isIntOrIndex()) {
      int bits = srcElemType.isIndex() ? 64 : srcElemType.getIntOrFloatBitWidth();
      if (bits == 8) identity = "(-128)";
      else if (bits == 16) identity = "(-32768)";
      else if (bits == 64) identity = "LLONG_MIN";
      else identity = "INT_MIN";
    } else {
      identity = "(-INFINITY)";
    }
  } else if (reduceOp == "min") {
    if (srcElemType.isIntOrIndex()) {
      int bits = srcElemType.isIndex() ? 64 : srcElemType.getIntOrFloatBitWidth();
      if (bits == 8) identity = "127";
      else if (bits == 16) identity = "32767";
      else if (bits == 64) identity = "LLONG_MAX";
      else identity = "INT_MAX";
    } else {
      identity = "INFINITY";
    }
  }

  // Reduce expression
  auto redExpr = [&](const std::string &a, const std::string &b) -> std::string {
    if (reduceOp == "add")
      return "(" + a + " + " + b + ")";
    if (reduceOp == "max") {
      if (srcElemType.isF32() || srcElemType.isF16() || srcElemType.isBF16())
        return "fmaxf(" + a + ", " + b + ")";
      if (srcElemType.isF64())
        return "fmax(" + a + ", " + b + ")";
      return "max(" + a + ", " + b + ")";
    }
    if (reduceOp == "min") {
      if (srcElemType.isF32() || srcElemType.isF16() || srcElemType.isBF16())
        return "fminf(" + a + ", " + b + ")";
      if (srcElemType.isF64())
        return "fmin(" + a + ", " + b + ")";
      return "min(" + a + ", " + b + ")";
    }
    if (reduceOp == "xor")
      return "(" + a + " ^ " + b + ")";
    return a;
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

    // Group registers by their coordinate with reduce axis set to 0
    // Key: coordinate with axis zeroed -> list of register indices
    std::map<SmallVector<int>, SmallVector<int>> reduceGroups;
    for (int i = 0; i < srcElems; i++) {
      SmallVector<int> key = regOffsets[i];
      if (axis < (int)key.size()) key[axis] = 0;
      reduceGroups[key].push_back(i);
    }

    // Assign output indices to each group
    int outIdx = 0;
    std::map<SmallVector<int>, int> keyToOutIdx;
    for (auto &[key, indices] : reduceGroups) {
      keyToOutIdx[key] = outIdx++;
    }
    assert(outIdx == resultElems && "reduce group count must match result elements");

    // Emit per-thread reduction using the computed grouping
    emit("// Initialize reduction accumulators");
    emit("#pragma unroll");
    emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++) {");
    emit("    " + var + "[_o] = " + identity + ";");
    if (isMultiResult2D) emit("    " + var1 + "[_o] = 0;");
    emit("}");

    for (auto &[key, indices] : reduceGroups) {
      int oi = keyToOutIdx[key];
      for (int si : indices) {
        if (isMultiResult2D) {
          std::string cmpOp = (reduceOp == "max") ? ">" : "<";
          emit("if (" + srcVar + "[" + std::to_string(si) + "] " + cmpOp + " " +
               var + "[" + std::to_string(oi) + "]) { " +
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
                emit("        if (_ov " + cmpOp + " " + var + "[_o]) { " + var + "[_o] = _ov; " + var1 + "[_o] = _oi; }");
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

    // Cross-warp reduction
    {
      // Compute warps with unique data along reduce axis
      auto srcShape = srcRtt.getShape();
      int warpsAlongAxis = 1;
      if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(srcEnc)) {
        auto wpc = blocked.getWarpsPerCTA();
        auto tpw = blocked.getThreadsPerWarp();
        auto spt = blocked.getSizePerThread();
        int elemsPerWarpOnAxis = tpw[axis] * spt[axis];
        warpsAlongAxis = std::min((int)wpc[axis],
                                  (int)((srcShape[axis] + elemsPerWarpOnAxis - 1) / elemsPerWarpOnAxis));
      } else {
        warpsAlongAxis = ttg::getWarpsPerCTA(srcRtt.getEncoding(), srcShape)[axis];
      }
      if (warpsAlongAxis > 1) {
        // Per-thread cross-warp reduction via shared memory
        // Each thread stores its values; all threads participate in the reduce
        int totalThreads = numWarps * 32;
        emit("// Cross-warp reduction along axis " + std::to_string(axis) +
             " (" + std::to_string(warpsAlongAxis) + " warps with unique data)");
        emit("{");
        indent();
        auto bufName = "_xw_" + var;
        int bufSize = totalThreads * resultElems;
        emit("__shared__ " + accType + " " + bufName + "[" + std::to_string(bufSize) + "];");
        std::string bufName1;
        if (isMultiResult2D) {
          bufName1 = "_xw_" + var1;
          emit("__shared__ " + cuda1Type + " " + bufName1 + "[" + std::to_string(bufSize) + "];");
        }
        // Each thread stores its partial result
        emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++)");
        emit("    " + bufName + "[tid * " + std::to_string(resultElems) + " + _o] = " + var + "[_o];");
        if (isMultiResult2D) {
          emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++)");
          emit("    " + bufName1 + "[tid * " + std::to_string(resultElems) + " + _o] = " + var1 + "[_o];");
        }
        emit("__syncthreads();");
        // Each thread reduces its own position across warps with unique data
        emit("for (int _o = 0; _o < " + std::to_string(resultElems) + "; _o++) {");
        indent();
        emit(var + "[_o] = " + bufName + "[lane_id * " + std::to_string(resultElems) + " + _o];");
        if (isMultiResult2D)
          emit(var1 + "[_o] = " + bufName1 + "[lane_id * " + std::to_string(resultElems) + " + _o];");
        emit("for (int _w = 1; _w < " + std::to_string(warpsAlongAxis) + "; _w++) {");
        indent();
        emit("int _src = _w * 32 * " + std::to_string(resultElems) + " + lane_id * " + std::to_string(resultElems) + " + _o;");
        if (isMultiResult2D) {
          std::string cmpOp = (reduceOp == "max") ? ">" : "<";
          emit("if (" + bufName + "[_src] " + cmpOp + " " + var + "[_o]) {");
          emit("    " + var + "[_o] = " + bufName + "[_src];");
          emit("    " + var1 + "[_o] = " + bufName1 + "[_src]; }");
        } else {
          emit(var + "[_o] = " + redExpr(var + "[_o]", bufName + "[_src]") + ";");
        }
        dedent();
        emit("}");
        dedent();
        emit("}");
        dedent();
        emit("}");
      }
    }
  } else {
    // Scalar result — full reduction
    bool isMultiResult = numResults > 1 && !src1Var.empty();

    // Determine how many threads actually hold valid data
    // For a 1D tensor with blocked layout, only totalT0 threads have data
    int totalActiveThreads = numWarps * 32; // default: all threads active
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
    bool needsThreadGuard = totalActiveThreads < numWarps * 32;

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
    std::string srcElem = promoteToFloat ? ("(float)" + srcVar + "[_i]") : (srcVar + "[_i]");
    emit("#pragma unroll");
    if (isMultiResult) {
      // Multi-result: track both value and index
      emit("for (int _i = 0; _i < " + std::to_string(srcElems) + "; _i++) {");
      indent();
      std::string cmpOp = (reduceOp == "max") ? ">" : "<";
      emit("if (" + srcElem + " " + cmpOp + " " + var + ") {");
      indent();
      emit(var + " = " + srcElem + ";");
      emit(var1 + " = " + src1Var + "[_i];");
      dedent();
      emit("}");
      dedent();
      emit("}");
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
      emit("if (_other_val " + cmpOp + " " + var + ") { " +
           var + " = _other_val; " + var1 + " = _other_idx; }");
      dedent();
      emit("}");
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

    // Cross-warp reduction
    if (numWarps > 1) {
      emit("// Cross-warp reduction");
      emit("{");
      indent();
      auto bufName = "_wb_" + var;
      emit("__shared__ " + accType + " " + bufName + "[" +
           std::to_string(numWarps) + "];");
      if (isMultiResult) {
        auto bufName1 = "_wb_" + var1;
        emit("__shared__ " + cuda1Type + " " + bufName1 + "[" +
             std::to_string(numWarps) + "];");
        emit("if (lane_id == 0) { " + bufName + "[warp_id] = " + var + "; " +
             bufName1 + "[warp_id] = " + var1 + "; }");
        emit("__syncthreads();");
        emit("if (warp_id == 0) {");
        indent();
        emit(var + " = (lane_id < " + std::to_string(numWarps) +
             ") ? " + bufName + "[lane_id] : " + identity + ";");
        emit(var1 + " = (lane_id < " + std::to_string(numWarps) +
             ") ? " + bufName1 + "[lane_id] : 0;");
        emit("for (int _off = 16; _off > 0; _off /= 2) {");
        indent();
        emit(accType + " _other_val = __shfl_xor_sync(0xffffffff, " + var + ", _off);");
        emit(cuda1Type + " _other_idx = __shfl_xor_sync(0xffffffff, " + var1 + ", _off);");
        std::string cmpOp = (reduceOp == "max") ? ">" : "<";
        emit("if (_other_val " + cmpOp + " " + var + ") { " +
             var + " = _other_val; " + var1 + " = _other_idx; }");
        dedent();
        emit("}");
        dedent();
        emit("}");
        emit("// Broadcast");
        emit("if (warp_id == 0 && lane_id == 0) { " + bufName + "[0] = " + var + "; " +
             bufName1 + "[0] = " + var1 + "; }");
        emit("__syncthreads();");
        emit(var + " = " + bufName + "[0];");
        emit(var1 + " = " + bufName1 + "[0];");
      } else {
        emit("if (lane_id == 0) " + bufName + "[warp_id] = " + var + ";");
        emit("__syncthreads();");
        emit("if (warp_id == 0) {");
        indent();
        emit(var + " = (lane_id < " + std::to_string(numWarps) +
             ") ? " + bufName + "[lane_id] : " + accIdentity + ";");
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
        emit("__syncthreads();");
        emit(var + " = " + bufName + "[0];");
      }
      dedent();
      emit("}");
    }
  }
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
    if (sElemType == rElemType) {
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
         var + "[_i] = (" + cudaType + ")(uint64_t)" + srcVar + "[_i];");
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

  std::string convertExpr;
  if (srcIsFp8 && !dstIsFp8) {
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
      // fp32 → fp8: go through fp16
      convertExpr = "__nv_cvt_halfraw_to_fp8(*(__half_raw*)&(__half(SRC)), __NV_SATFINITE, " + fp8Type + ")";
    }
  } else {
    // fp8 → fp8 or same type: use raw cast
    convertExpr = "(" + dstCudaType + ")SRC";
  }

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    emit(dstCudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    std::string expr = convertExpr;
    // Replace SRC with srcVar[_i]
    auto pos = expr.find("SRC");
    while (pos != std::string::npos) {
      expr.replace(pos, 3, srcVar + "[_i]");
      pos = expr.find("SRC", pos + srcVar.size() + 4);
    }
    emit("    " + var + "[_i] = " + expr + ";");
  } else {
    emit("const " + dstCudaType + " " + var + " = " +
         [&]{ auto e = convertExpr; auto p = e.find("SRC"); if (p != std::string::npos) e.replace(p, 3, srcVar); return e; }() + ";");
  }
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
  if (it != nvMap.end()) symbol = it->second;
  Type rElemType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto cudaType = getCUDAType(rElemType);
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    std::string args;
    for (unsigned i = 0; i < op.getNumOperands(); i++) {
      if (i) args += ", ";
      args += getVar(op.getOperand(i)) + "[_i]";
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

void CUDACodeGen::emitAtomicRMW(tt::AtomicRMWOp op) {
  auto result = op.getResult();
  auto var = newVar("atom");
  valueToVar[result] = var;
  auto ptrVar = getVar(op.getPtr());
  auto valVar = getVar(op.getVal());
  bool isTensor = isa<RankedTensorType>(result.getType());
  Type elemType = isTensor ? cast<RankedTensorType>(result.getType()).getElementType() : result.getType();
  auto cudaType = getCUDAType(elemType);
  std::string fn = "atomicAdd";
  auto rmw = op.getAtomicRmwOp();
  if (rmw == tt::RMWOp::ADD || rmw == tt::RMWOp::FADD) fn = "atomicAdd";
  else if (rmw == tt::RMWOp::MAX) fn = "atomicMax";
  else if (rmw == tt::RMWOp::MIN) fn = "atomicMin";
  else if (rmw == tt::RMWOp::AND) fn = "atomicAnd";
  else if (rmw == tt::RMWOp::OR) fn = "atomicOr";
  else if (rmw == tt::RMWOp::XOR) fn = "atomicXor";
  else if (rmw == tt::RMWOp::XCHG) fn = "atomicExch";
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    if (op.getMask()) emit("if (" + getVar(op.getMask()) + "[_i])");
    emit("    " + var + "[_i] = " + fn + "(" + ptrVar + "[_i], " + valVar + "[_i]);");
    dedent();
    emit("}");
  } else {
    emit(cudaType + " " + var + " = " + fn + "(" + ptrVar + ", " + valVar + ");");
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

  if (isTensor) {
    int nElems = getElemsPerThread(result);
    emit(cudaType + " " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + var + "[_i] = atomicCAS(" + ptrVar + "[_i], " +
         cmpVar + "[_i], " + valVar + "[_i]);");
  } else {
    emit(cudaType + " " + var + " = atomicCAS(" + ptrVar + ", " +
         cmpVar + ", " + valVar + ");");
  }
}

void CUDACodeGen::emitPrint(tt::PrintOp op) {
  emit("// tt.print (skipped)");
}

void CUDACodeGen::emitMulhiUI(tt::MulhiUIOp op) {
  auto result = op.getResult();
  bool isTensor = isa<RankedTensorType>(result.getType());
  auto var = newVar("mulhi");
  valueToVar[result] = var;
  auto aVar = getVar(op.getOperand(0));
  auto bVar = getVar(op.getOperand(1));
  if (isTensor) {
    int nElems = getElemsPerThread(result);
    emit("unsigned " + var + "[" + std::to_string(nElems) + "];");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + var + "[_i] = __umulhi((unsigned)" + aVar + "[_i], (unsigned)" + bVar + "[_i]);");
  } else {
    emit("unsigned " + var + " = __umulhi((unsigned)" + aVar + ", (unsigned)" + bVar + ");");
  }
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

      // Check if init value is a deferred addptr — carry offsets, not pointers
      auto deferIt = deferredAddPtr.find(iterArgs[i]);
      if (deferIt != deferredAddPtr.end()) {
        std::string base = deferIt->second.first;
        std::string initOff = deferIt->second.second;
        // Carry int offsets instead of __half* pointers (saves 50% registers)
        emit("int " + iterVar + "[" + std::to_string(nElems) + "];");
        emit("for (int _t = 0; _t < " + std::to_string(nElems) +
             "; _t++) " + iterVar + "[_t] = " + initOff + "[_t];");
        // Register as deferred so downstream ops use base+offset
        deferredAddPtr[regionIterArgs[i]] = {base, iterVar};
        iterArgDeferredBase[regionIterArgs[i]] = base;
        if (i < op.getNumResults())
          deferredAddPtr[op.getResult(i)] = {base, iterVar};
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

  // Emit for loop
  emit("for (int " + ivVar + " = " + lbVar + "; " + ivVar + " < " + ubVar +
       "; " + ivVar + " += " + stepVar + ") {");
  indent();

  // Emit body
  auto &body = op.getBody()->getOperations();
  for (auto &bodyOp : body) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
      // Update iter_args
      for (unsigned i = 0; i < yieldOp.getNumOperands(); i++) {
        if (i < regionIterArgs.size()) {
          auto iterVar = getVar(regionIterArgs[i]);
          auto yieldVal = yieldOp.getOperand(i);
          auto srcVar = getVar(yieldVal);
          bool isTensor =
              isa<RankedTensorType>(yieldVal.getType());
          if (isTensor) {
            int nElems = getElemsPerThread(yieldVal);
            // Handle deferred addptr in yield
            auto deferIt = deferredAddPtr.find(yieldVal);
            if (deferIt != deferredAddPtr.end()) {
              auto baseIt = iterArgDeferredBase.find(regionIterArgs[i]);
              if (baseIt != iterArgDeferredBase.end() &&
                  baseIt->second == deferIt->second.first) {
                // Same base: just copy offsets (int, not __half*)
                emit("#pragma unroll");
                emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                     "; _t++) " + iterVar + "[_t] = " +
                     deferIt->second.second + "[_t];");
              } else {
                // Different base: materialize full pointer
                emit("#pragma unroll");
                emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                     "; _t++) " + iterVar + "[_t] = " +
                     deferIt->second.first + " + " + deferIt->second.second + "[_t];");
              }
            } else if (scalarValues.contains(yieldVal)) {
              emit("#pragma unroll");
              emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                   "; _t++) " + iterVar + "[_t] = " + srcVar + ";");
            } else {
              emit("#pragma unroll");
              emit("for (int _t = 0; _t < " + std::to_string(nElems) +
                   "; _t++) " + iterVar + "[_t] = " + srcVar + "[_t];");
            }
          } else {
            if (iterVar != srcVar)
              emit(iterVar + " = " + srcVar + ";");
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
               "; _t++) " + resultVars[i] + "[_t] = " + yieldedVar + "[_t];");
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
                 "; _t++) " + resultVars[i] + "[_t] = " + yieldedVar + "[_t];");
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
           "; _t++) " + iterVar + "[_t] = " + initVar + "[_t];");
    } else {
      auto cudaType = getCUDAType(inits[i].getType());
      emit(cudaType + " " + iterVar + " = " + initVar + ";");
    }
  }

  // Create condition output variables + map results
  llvm::SmallVector<std::string> condOutVars;
  for (unsigned i = 0; i < inits.size(); i++) {
    auto condVar = newVar("wcond");
    condOutVars.push_back(condVar);
    bool isTensor = isa<RankedTensorType>(inits[i].getType());
    if (isTensor) {
      int nElems = getElemsPerThread(inits[i]);
      Type elemType = cast<RankedTensorType>(inits[i].getType()).getElementType();
      emit(getCUDAType(elemType) + " " + condVar + "[" + std::to_string(nElems) + "];");
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
        bool isTensor = isa<RankedTensorType>(condArgs[i].getType());
        if (isTensor) {
          int nElems = getElemsPerThread(condArgs[i]);
          emit("for (int _t = 0; _t < " + std::to_string(nElems) +
               "; _t++) " + condOutVars[i] + "[_t] = " + srcVar + "[_t];");
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
          int nElems = getElemsPerThread(yieldOp.getOperand(i));
          emit("#pragma unroll");
          emit("for (int _t = 0; _t < " + std::to_string(nElems) +
               "; _t++) " + iterVars[i] + "[_t] = " + srcVar + "[_t];");
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

  // Align to 128 bytes for WGMMA
  int offset = (sharedMemOffset + 127) & ~127;
  sharedMemOffset = offset + byteSize;

  auto var = newVar("smem");
  valueToVar[result] = var;
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
        // MMA layout → shared memory via stmatrix
        emit("// MMA→shared via stmatrix (swizzle=128B)");
        emit("{");
        indent();
        emit("uint32_t _base = ((tid << 7) & 0x780) | ((tid << 4) & 0x70);");
        emit("_base = (_base ^ (tid & 0x10)) | ((tid << 6) & 0x1800);");
        emit("char* _smem_base = (char*)(shared_mem + " + std::to_string(offset) + ");");
        // Pack f32→f16 pairs into uint32 for stmatrix
        emit("uint32_t _packed[64];");
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < 64; _i++) {");
        indent();
        emit("__half _h0 = (__half)" + srcVar + "[2*_i];");
        emit("__half _h1 = (__half)" + srcVar + "[2*_i+1];");
        emit("uint32_t _lo = *(uint16_t*)&_h0;");
        emit("uint32_t _hi = *(uint16_t*)&_h1;");
        emit("_packed[_i] = _lo | (_hi << 16);");
        dedent();
        emit("}");
        // 16 stmatrix calls: 4 n-groups × 4 sub-tiles
        int smemOffsets[] = {0, 16384, 8192, 24576};
        for (int nGrp = 0; nGrp < 4; nGrp++) {
          for (int sub = 0; sub < 4; sub++) {
            int packedStart = nGrp * 4 + sub * 16;
            int xorVal = (nGrp ^ 1) * 32;  // regPermForDivide: swap adjacent n-groups
            int smemOff = smemOffsets[sub];
            std::string addrExpr = xorVal ? ("_base ^ " + std::to_string(xorVal)) : "_base";
            emit("asm volatile(\"stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\"");
            emit("    :: \"r\"((unsigned)__cvta_generic_to_shared(_smem_base + (" +
                 addrExpr + ") + " + std::to_string(smemOff) + ")),");
            emit("       \"r\"(_packed[" + std::to_string(packedStart) + "]), \"r\"(_packed[" +
                 std::to_string(packedStart+1) + "]),");
            emit("       \"r\"(_packed[" + std::to_string(packedStart+2) + "]), \"r\"(_packed[" +
                 std::to_string(packedStart+3) + "]));");
          }
        }
        dedent();
        emit("}");
      } else if (auto blocked = dyn_cast_or_null<ttg::BlockedEncodingAttr>(srcEnc)) {
        if (nvmmaShared && shape.size() == 2) {
          // Swizzled store: blocked → nvmma_shared
          int swizzleBytes = nvmmaShared.getSwizzlingByteWidth();
          int elemBits = nvmmaShared.getElementBitWidth();
          int rows = shape[0], cols = shape[1];
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

          emit("// Swizzled store: blocked→nvmma_shared (" +
               std::to_string(rows) + "x" + std::to_string(cols) + " " +
               cudaType + ", swizzle=" + std::to_string(swizzleBytes) + "B)");
          emit("{");
          indent();
          emit("int _lane_d0 = lane_id / " + std::to_string(tpw1) + ";");
          emit("int _lane_d1 = lane_id % " + std::to_string(tpw1) + ";");
          emit("int _warp_d0 = warp_id / " + std::to_string(wpc1) + ";");
          emit("int _warp_d1 = warp_id % " + std::to_string(wpc1) + ";");
          emit("int _pos0 = _lane_d0 + _warp_d0 * " + std::to_string(tpw0) + ";");
          emit("int _pos1 = _lane_d1 + _warp_d1 * " + std::to_string(tpw1) + ";");
          emit("#pragma unroll");
          emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
          indent();
          emit("int _rep0 = _i / " + std::to_string(strideRep0) + ";");
          emit("int _s0 = (_i / " + std::to_string(spt1 * reps1) + ") % " + std::to_string(spt0) + ";");
          emit("int _rep1 = (_i / " + std::to_string(spt1) + ") % " + std::to_string(reps1) + ";");
          emit("int _s1 = _i % " + std::to_string(spt1) + ";");
          emit("int _row = _pos0 * " + std::to_string(spt0) + " + _s0 + _rep0 * " + std::to_string(totalT0 * spt0) + ";");
          emit("int _col = _pos1 * " + std::to_string(spt1) + " + _s1 + _rep1 * " + std::to_string(totalT1 * spt1) + ";");
          {
            // nvmma_shared uses tile-major layout when cols > elemsPerSwizzlingRow
            int elemsPerSwizzlingRow = swizzleBytes / elemBytes;
            if (cols > elemsPerSwizzlingRow) {
              // Multi-tile: each tile is rows × elemsPerSwizzlingRow
              emit("int _tileK = _col / " + std::to_string(elemsPerSwizzlingRow) + ";");
              emit("int _colInTile = _col % " + std::to_string(elemsPerSwizzlingRow) + ";");
              if (maxPhase > 1) {
                emit("int _phase = (_row / " + std::to_string(perPhase) + ") % " + std::to_string(maxPhase) + ";");
                emit("int _swizzled_col = _colInTile ^ (_phase * " + std::to_string(vec) + ");");
                emit(var + "[_tileK * " + std::to_string(rows * elemsPerSwizzlingRow) +
                     " + _row * " + std::to_string(elemsPerSwizzlingRow) + " + _swizzled_col] = " + srcVar + "[_i];");
              } else {
                emit(var + "[_tileK * " + std::to_string(rows * elemsPerSwizzlingRow) +
                     " + _row * " + std::to_string(elemsPerSwizzlingRow) + " + _colInTile] = " + srcVar + "[_i];");
              }
            } else {
              // Single tile: simple row-major with swizzle
              if (maxPhase > 1) {
                emit("int _phase = (_row / " + std::to_string(perPhase) + ") % " + std::to_string(maxPhase) + ";");
                emit("int _swizzled_col = _col ^ (_phase * " + std::to_string(vec) + ");");
                emit(var + "[_row * " + std::to_string(cols) + " + _swizzled_col] = " + srcVar + "[_i];");
              } else {
                emit(var + "[_row * " + std::to_string(cols) + " + _col] = " + srcVar + "[_i];");
              }
            }
          }
          dedent();
          emit("}");
          dedent();
          emit("}");
          emit("__syncthreads();");
        } else {
          // Linear store
          emit("// Store to shared memory (linear)");
          emit("#pragma unroll");
          emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
          emit("    " + var + "[tid * " + std::to_string(nElems) + " + _i] = " + srcVar + "[_i];");
          emit("__syncthreads();");
        }
      } else {
        // Fallback
        emit("// Store to shared memory (linear fallback)");
        emit("#pragma unroll");
        emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
        emit("    " + var + "[tid * " + std::to_string(nElems) + " + _i] = " + srcVar + "[_i];");
        emit("__syncthreads();");
      }
    }
  }
}

void CUDACodeGen::emitLocalStore(ttg::LocalStoreOp op) {
  auto val = op.getSrc();
  auto dst = op.getDst();
  auto valVar = getVar(val);
  auto dstVar = getVar(dst);
  bool isTensor = isa<RankedTensorType>(val.getType());

  if (isTensor) {
    int nElems = getElemsPerThread(val);
    emit("// local_store to shared memory");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++)");
    emit("    " + dstVar + "[tid * " + std::to_string(nElems) + " + _i] = " + valVar + "[_i];");
    emit("__syncthreads();");
  } else {
    emit(dstVar + "[tid] = " + valVar + ";");
  }
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
    auto shape = rtt.getShape();
    // Compute strides for flattening
    llvm::SmallVector<int64_t> strides(rank);
    strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; d--)
      strides[d] = strides[d + 1] * shape[d + 1];
    // Compute per-register addresses using LL bases
    auto regBases = dstLL.getBases().find(kReg);
    auto laneBases = dstLL.getBases().find(kLane);
    auto warpBases = dstLL.getBases().find(kWarp);
    int nRegBits = (regBases != dstLL.getBases().end()) ? regBases->second.size() : 0;
    int nLaneBits = (laneBases != dstLL.getBases().end()) ? laneBases->second.size() : 0;
    int nWarpBits = (warpBases != dstLL.getBases().end()) ? warpBases->second.size() : 0;

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
  emit("__syncthreads();");
}
void CUDACodeGen::emitLocalDealloc(ttg::LocalDeallocOp op) {
  emit("// local_dealloc");
  peakSharedMem = std::max(peakSharedMem, sharedMemOffset);
  sharedMemOffset = 0;
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

  // Only use the manual MMA→blocked path for WGMMA v3; MMA v2 uses the
  // generic LL-based path because its column interleaving differs.
  bool isMMAv3ToBlocked = false;
  if (auto mmaEnc = dyn_cast<ttg::NvidiaMmaEncodingAttr>(srcEnc))
    isMMAv3ToBlocked = mmaEnc.getVersionMajor() >= 3 && isa<ttg::BlockedEncodingAttr>(dstEnc);
  bool isSameSpt = false;
  if (auto srcBlk = dyn_cast<ttg::BlockedEncodingAttr>(srcEnc))
    if (auto dstBlk = dyn_cast<ttg::BlockedEncodingAttr>(dstEnc))
      isSameSpt = (srcBlk.getSizePerThread() == dstBlk.getSizePerThread());

  if (isMMAv3ToBlocked && rtt.getRank() >= 2) {
    auto mmaEnc = cast<ttg::NvidiaMmaEncodingAttr>(srcEnc);
    int M = rtt.getShape()[0], N = rtt.getShape()[1];
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
    int smemOff = (sharedMemOffset + 127) & ~127;
    sharedMemOffset = smemOff + totalElems * eb;
    if (sharedMemOffset > peakSharedMem) peakSharedMem = sharedMemOffset;
    emit(cudaType + "* _cvt = (" + cudaType + "*)(shared_mem + " + std::to_string(smemOff) + ");");
    // MMA store: pack pairs of f16 into uint32 for vectorized st.shared.b32
    {
      int regsPerMTile = (64 * N) / 128;
      int nMTiles = (M + 63) / 64;
      emit("// MMA→shared via packed uint32 stores");
      emit("{");
      indent();
      emit("uint32_t* _cvt32 = (uint32_t*)_cvt;");
      // Decompose warp_id into M and N dimensions
      if (wpc1 > 1) {
        emit("int _warp_m = warp_id / " + std::to_string(wpc1) + ";");
        emit("int _warp_n = warp_id % " + std::to_string(wpc1) + ";");
      } else {
        emit("int _warp_m = warp_id;");
      }
      emit("#pragma unroll");
      emit("for (int _r = 0; _r < " + std::to_string(nSrc / 2) + "; _r++) {");
      indent();
      // Pack two adjacent f16 into uint32
      emit("__half _h0 = (__half)" + srcVar + "[2*_r];");
      emit("__half _h1 = (__half)" + srcVar + "[2*_r+1];");
      emit("uint32_t _lo = *(uint16_t*)&_h0;");
      emit("uint32_t _hi = *(uint16_t*)&_h1;");
      emit("uint32_t _packed = _lo | (_hi << 16);");
      // Compute address: registers 2*_r and 2*_r+1 are at (row, col) and (row, col+1)
      // which are adjacent in row-major → one uint32 at col/2
      if (nMTiles > 1) {
        emit("int _r2 = 2 * _r;");
        emit("int _m_tile = _r2 / " + std::to_string(regsPerMTile) + ";");
        emit("int _r_local = _r2 % " + std::to_string(regsPerMTile) + ";");
        emit("int _mma_row = _m_tile * 64 + _warp_m * " + std::to_string(MInstr) +
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
      emit("_cvt32[_mma_row * " + std::to_string(N / 2) + " + _mma_col / 2] = _packed;");
      dedent();
      emit("}");
      dedent();
      emit("}");
    }
    emit("__syncthreads();");
    // Blocked load
    auto dstBlk = cast<ttg::BlockedEncodingAttr>(dstEnc);
    emitBlockedLoadFromSmem(var, "_cvt", dstBlk, rtt.getShape(), cudaType, nDst);
    emit("__syncthreads();");
    dedent();
    emit("}");
  } else if (nSrc == nDst && isSameSpt) {
    emit("// convert_layout (alias — same sizePerThread)");
    emit(cudaType + "* " + var + " = " + srcVar + ";");
  } else {
    // Shared memory intermediary
    emit("// convert_layout via shared memory");
    emit(cudaType + " " + var + "[" + std::to_string(nDst) + "];");
    emit("{");
    indent();
    int64_t totalElems = 1;
    for (auto s : rtt.getShape()) totalElems *= s;
    // Shared memory needs space for all threads, not just tensor elements
    int64_t smemElems = std::max(totalElems, (int64_t)(numWarps * 32 * std::max(nSrc, nDst)));
    int eb = getTypeSizeInBytes(elemType);
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
    emit("__syncthreads();");
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
    emit("__syncthreads();");
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
  int64_t stride = 1;
  for (auto s : shape) stride *= s;
  int byteStride = stride * elemBytes;

  // Get index operand (second operand)
  std::string idxVar = "0";
  if (op->getNumOperands() > 1)
    idxVar = getVar(op->getOperand(1));

  auto cudaType = getCUDAType(elemType);
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
  int M_block = rtt.getShape()[0], N_block = rtt.getShape()[1];

  // Detect RS mode: A is tensor (registers) vs memdesc (shared memory)
  bool isRS = isa<RankedTensorType>(a.getType());

  // Get B shared layout info
  auto bMemDesc = cast<ttg::MemDescType>(b.getType());
  auto bEnc = dyn_cast<ttg::NVMMASharedEncodingAttr>(bMemDesc.getEncoding());
  int trans_b = (bEnc && bEnc.getTransposed()) ? 0 : 1;
  int swizzle_b = bEnc ? bEnc.getSwizzlingByteWidth() : 128;

  // Get K from memdesc shape
  int K_block = 32;
  if (isRS) {
    K_block = cast<RankedTensorType>(a.getType()).getShape()[1];
  } else {
    auto aMemDesc = cast<ttg::MemDescType>(a.getType());
    K_block = aMemDesc.getShape()[1];
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
  else aElem = "tf32";
  // B element type may differ (e.g., e4m3 x e5m2)
  Type bElemType = cast<ttg::MemDescType>(b.getType()).getElementType();
  std::string bElem;
  if (bElemType.isF16()) bElem = "f16";
  else if (bElemType.isBF16()) bElem = "bf16";
  else if (isa<Float8E4M3FNType>(bElemType)) bElem = "e4m3";
  else if (isa<Float8E5M2Type>(bElemType)) bElem = "e5m2";
  else if (bElemType.isF32()) bElem = "tf32";
  else bElem = "tf32";
  std::string cElem = "f32";

  bool isFp8 = (aElem == "e4m3" || aElem == "e5m2");
  int wgmma_k = isFp8 ? 32 : ((aElem == "f16" || aElem == "bf16") ? 16 : 8);
  int wgmma_m = 64;
  // Triton PTX backend uses HGMMA.64x256x16 with .b32 regs for n=256.
  // nvcc can't match this: "+f" asm constraints allocate .f32 regs (less
  // flexible than .b32), and "+r" with uint32_t* cast adds type-punning
  // overhead. Cap at 128 for now; proper fix requires generating PTX directly.
  int wgmma_n = std::min(N_block, 128);
  int n_n_tiles = N_block / wgmma_n;
  int n_m_tiles = M_block / wgmma_m;
  int n_k_tiles = K_block / wgmma_k;
  int n_out_regs = (wgmma_m * wgmma_n) / 128;
  int total_acc = n_out_regs * n_m_tiles * n_n_tiles;

  auto var = newVar("wgmma");
  valueToVar[result] = var;
  auto accVar = getVar(acc);
  auto bVar = getVar(b);

  emit("// WGMMA: m" + std::to_string(wgmma_m) + "n" + std::to_string(wgmma_n) +
       "k" + std::to_string(wgmma_k) + "." + cElem + "." + aElem + "." + bElem +
       (isRS ? " (RS mode)" : " (SS mode)"));
  emit("float " + var + "[" + std::to_string(total_acc) + "];");
  emit("#pragma unroll");
  emit("for (int _i = 0; _i < " + std::to_string(total_acc) + "; _i++) " +
       var + "[_i] = " + accVar + "[_i];");
  emit("{");
  indent();

  // B descriptor setup
  if (!isRS) {
    auto aVar = getVar(a);
    int swizzle_a = 128;
    if (!isRS) {
      auto aMemDesc = cast<ttg::MemDescType>(a.getType());
      auto aEnc = dyn_cast<ttg::NVMMASharedEncodingAttr>(aMemDesc.getEncoding());
      swizzle_a = aEnc ? aEnc.getSwizzlingByteWidth() : 64;
    }
    emit("uint32_t smem_addr_a = (unsigned)__cvta_generic_to_shared(" + aVar + ");");
    emit("uint32_t smem_addr_b = (unsigned)__cvta_generic_to_shared(" + bVar + ");");
    emitBlank();

    bool b_is_transposed = bEnc && bEnc.getTransposed();
    int elem_bytes_a = isFp8 ? 1 : ((aElem == "f16" || aElem == "bf16") ? 2 : 4);
    int elem_bytes_b = (bElem == "e4m3" || bElem == "e5m2") ? 1 : ((bElem == "f16" || bElem == "bf16") ? 2 : 4);

    emit("asm volatile(\"wgmma.fence.sync.aligned;\");");
    emitBlank();

    for (int m_tile = 0; m_tile < n_m_tiles; m_tile++) {
      for (int n_tile = 0; n_tile < n_n_tiles; n_tile++) {
      int acc_base = m_tile * n_out_regs * n_n_tiles + n_tile * n_out_regs;
      for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
        int a_byte_offset = m_tile * wgmma_m * K_block * elem_bytes_a + k_tile * wgmma_k * elem_bytes_a;
        int b_byte_offset;
        if (b_is_transposed)
          b_byte_offset = k_tile * wgmma_k * elem_bytes_b + n_tile * wgmma_n * K_block * elem_bytes_b;
        else {
          int b_row_stride = std::min(N_block, swizzle_b / elem_bytes_b) * elem_bytes_b;
          b_byte_offset = k_tile * wgmma_k * b_row_stride + n_tile * wgmma_n * elem_bytes_b;
        }

        uint64_t desc_a = getWGMMADescTemplate(swizzle_a, M_block);
        // B descriptor stride = physical alloc's non-fast-moving dimension.
        // For transposed B (fastMovingDim=0): physAlloc[1] = memdesc.shape[0]
        // For non-transposed B (fastMovingDim=1): physAlloc[0] = memdesc.shape[0]
        int b_stride_dim = bMemDesc.getShape()[0];
        uint64_t desc_b = getWGMMADescTemplate(swizzle_b, b_stride_dim);

        // Build PTX
        std::string out_str;
        for (int i = 0; i < n_out_regs; i++) {
          if (i) out_str += ", ";
          out_str += "%" + std::to_string(i);
        }
        int a_idx = n_out_regs, b_idx = n_out_regs + 1;
        bool isTf32 = (aElem == "tf32");
        // fp8 and tf32: 3 immediates (scale_d, imm_scale_a, imm_scale_b)
        // f16/bf16: 5 immediates (scale_d, imm_scale_a, imm_scale_b, unused=0, trans_b)
        std::string immSuffix = (isFp8 || isTf32) ? ", 1, 1, 1;" : ", 1, 1, 1, 0, " + std::to_string(trans_b) + ";";
        std::string ptx = "wgmma.mma_async.sync.aligned.m" + std::to_string(wgmma_m) +
                          "n" + std::to_string(wgmma_n) + "k" + std::to_string(wgmma_k) +
                          "." + cElem + "." + aElem + "." + bElem + " {" + out_str +
                          "}, %" + std::to_string(a_idx) + ", %" + std::to_string(b_idx) +
                          immSuffix;

        emit("{");
        indent();
        char desc_a_hex[20], desc_b_hex[20];
        snprintf(desc_a_hex, sizeof(desc_a_hex), "0x%016lXULL", desc_a);
        snprintf(desc_b_hex, sizeof(desc_b_hex), "0x%016lXULL", desc_b);
        emit("uint64_t desc_a = ((uint64_t)((smem_addr_a + " + std::to_string(a_byte_offset) +
             ") >> 4)) | " + std::string(desc_a_hex) + ";");
        emit("uint64_t desc_b = ((uint64_t)((smem_addr_b + " + std::to_string(b_byte_offset) +
             ") >> 4)) | " + std::string(desc_b_hex) + ";");

        emit("asm volatile(");
        indent();
        emit("\"" + ptx + "\"");

        // Output operands
        std::string outOps;
        for (int i = 0; i < n_out_regs; i++) {
          if (i) outOps += ", ";
          if (i % 8 == 0 && i > 0) outOps += "\n              ";
          outOps += "\"+f\"(" + var + "[" + std::to_string(acc_base + i) + "])";
        }
        emit(": " + outOps);
        emit(": \"l\"(desc_a), \"l\"(desc_b)");
        dedent();
        emit(");");
        dedent();
        emit("}");
      }
      } // end n_tile
    }
    emit("asm volatile(\"wgmma.commit_group.sync.aligned;\");");
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
    if (isTf32A) {
      // tf32: each A register is 32-bit, pass directly as uint32
      a_packed_per_ktile = a_regs_per_ktile;
      emit("uint32_t _a_packed[" + std::to_string(n_k_tiles * a_packed_per_ktile) + "];");
      emit("#pragma unroll");
      emit("for (int _k = 0; _k < " + std::to_string(n_k_tiles) + "; _k++) {");
      indent();
      emit("#pragma unroll");
      emit("for (int _j = 0; _j < " + std::to_string(a_packed_per_ktile) + "; _j++) {");
      indent();
      emit("float _f = " + aVar + "[_k * " + std::to_string(a_regs_per_ktile) + " + _j];");
      emit("_a_packed[_k * " + std::to_string(a_packed_per_ktile) + " + _j] = *(uint32_t*)&_f;");
      dedent();
      emit("}");
      dedent();
      emit("}");
    } else {
      // f16/bf16/fp8: pack pairs of 16-bit values into uint32
      a_packed_per_ktile = a_regs_per_ktile / 2;
      emit("uint32_t _a_packed[" + std::to_string(n_k_tiles * a_packed_per_ktile) + "];");
      emit("#pragma unroll");
      emit("for (int _k = 0; _k < " + std::to_string(n_k_tiles) + "; _k++) {");
      indent();
      emit("#pragma unroll");
      emit("for (int _j = 0; _j < " + std::to_string(a_packed_per_ktile) + "; _j++) {");
      indent();
      auto aCudaType = getCUDAType(aElemType);
      emit(aCudaType + " _h0 = " + aVar + "[_k * " + std::to_string(a_regs_per_ktile) + " + _j * 2];");
      emit(aCudaType + " _h1 = " + aVar + "[_k * " + std::to_string(a_regs_per_ktile) + " + _j * 2 + 1];");
      emit("uint32_t _lo = *(uint16_t*)&_h0;");
      emit("uint32_t _hi = *(uint16_t*)&_h1;");
      emit("_a_packed[_k * " + std::to_string(a_packed_per_ktile) + " + _j] = _lo | (_hi << 16);");
      dedent();
      emit("}");
      dedent();
      emit("}");
    }
    emitBlank();

    int elem_bytes_b = (bElem == "f16" || bElem == "bf16") ? 2 : 4;
    bool b_is_transposed = bEnc && bEnc.getTransposed();

    emit("asm volatile(\"wgmma.fence.sync.aligned;\");");
    emitBlank();

    for (int m_tile = 0; m_tile < n_m_tiles; m_tile++) {
      int acc_base = m_tile * n_out_regs;
      for (int k_tile = 0; k_tile < n_k_tiles; k_tile++) {
        int b_byte_offset;
        if (b_is_transposed)
          b_byte_offset = k_tile * wgmma_k * elem_bytes_b;
        else {
          int b_row_stride = std::min(N_block, swizzle_b / elem_bytes_b) * elem_bytes_b;
          b_byte_offset = k_tile * wgmma_k * b_row_stride;
        }
        int b_stride_dim = bMemDesc.getShape()[0];
        uint64_t desc_b = getWGMMADescTemplate(swizzle_b, b_stride_dim);
        int a_packed_base = k_tile * a_packed_per_ktile;

        // PTX for RS mode
        std::string out_str;
        for (int i = 0; i < n_out_regs; i++) {
          if (i) out_str += ", ";
          out_str += "%" + std::to_string(i);
        }
        std::string a_reg_str;
        for (int i = 0; i < a_packed_per_ktile; i++) {
          if (i) a_reg_str += ", ";
          a_reg_str += "%" + std::to_string(n_out_regs + i);
        }
        int b_idx = n_out_regs + a_packed_per_ktile;
        std::string rsImmSuffix = isFp8 ? ", 1, 1, 1;" : ", 1, 1, 1, " + std::to_string(trans_b) + ";";
        std::string ptx = "wgmma.mma_async.sync.aligned.m" + std::to_string(wgmma_m) +
                          "n" + std::to_string(wgmma_n) + "k" + std::to_string(wgmma_k) +
                          "." + cElem + "." + aElem + "." + bElem + " {" + out_str +
                          "}, {" + a_reg_str + "}, %" + std::to_string(b_idx) + rsImmSuffix;

        emit("{");
        indent();
        char desc_b_hex[20];
        snprintf(desc_b_hex, sizeof(desc_b_hex), "0x%016lXULL", desc_b);
        emit("uint64_t desc_b = ((uint64_t)((smem_addr_b + " + std::to_string(b_byte_offset) +
             ") >> 4)) | " + std::string(desc_b_hex) + ";");
        emit("asm volatile(");
        indent();
        emit("\"" + ptx + "\"");
        std::string outOps;
        for (int i = 0; i < n_out_regs; i++) {
          if (i) outOps += ", ";
          if (i % 8 == 0 && i > 0) outOps += "\n              ";
          outOps += "\"+f\"(" + var + "[" + std::to_string(acc_base + i) + "])";
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
}

void CUDACodeGen::emitWarpGroupDotWait(ttng::WarpGroupDotWaitOp op) {
  // Map results to input operands (wait is a synchronization point)
  auto src = op.getOperand(0);
  auto var = getVar(src);
  for (auto result : op.getResults())
    valueToVar[result] = var;
  int pendings = op.getPendings();
  emit("asm volatile(\"wgmma.wait_group.sync.aligned " + std::to_string(pendings) + ";\");");
}
void CUDACodeGen::emitFenceAsyncShared(ttng::FenceAsyncSharedOp op) {
  if (hasPendingCpAsync) {
    emit("asm volatile(\"cp.async.commit_group;\");");
    emit("asm volatile(\"cp.async.wait_group 0;\");");
    emit("__syncthreads();");
    hasPendingCpAsync = false;
  }
  emit("asm volatile(\"fence.proxy.async.shared::cta;\");");
  // Barrier after fence to ensure all threads' stmatrix/shared writes are
  // visible before a subsequent TMA L2G from thread 0.
  emit("__syncthreads();");
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
  emit("__syncthreads();");
}

void CUDACodeGen::emitWaitBarrier(ttng::WaitBarrierOp op) {
  auto barVar = getVar(op.getAlloc());
  auto phaseVar = getVar(op.getPhase());
  emit("__syncthreads();");
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
  emit("asm volatile(\"mbarrier.arrive.shared::cta.b64 _, [%0];\" :: \"r\"((unsigned)__cvta_generic_to_shared(" +
       barVar + ")));");
}

void CUDACodeGen::emitBarrierExpect(ttng::BarrierExpectOp op) {
  emit("__syncthreads();");  // MUST be before expect_tx
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

void CUDACodeGen::emitAsyncTMACopyG2L(ttng::AsyncTMACopyGlobalToLocalOp op) {
  auto descVar = getVar(op.getDesc());
  auto smemVar = getVar(op.getResult());
  auto barVar = getVar(op.getBarrier());
  auto predVar = getVar(op.getPred());
  auto coords = op.getCoord();
  std::string coord0 = coords.size() > 0 ? getVar(coords[0]) : "0";
  std::string coord1 = coords.size() > 1 ? getVar(coords[1]) : "0";

  // TMA copies the entire tile in one instruction — the hardware handles
  // swizzle and tiles wider than the swizzle width internally.
  emit("__syncthreads();");
  emit("if (threadIdx.x == 0) {");
  indent();
  emit("if (" + predVar + ") {");
  indent();
  emit("asm volatile(");
  emit("    \"cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\\n\"");
  emit("    :: \"r\"((unsigned)__cvta_generic_to_shared(" + smemVar + ")),");
  emit("       \"l\"((uint64_t)&" + descVar + "),");
  emit("       \"r\"(" + coord1 + "), \"r\"(" + coord0 + "),");
  emit("       \"r\"((unsigned)__cvta_generic_to_shared(" + barVar + "))");
  emit(");");
  dedent();
  emit("}");
  dedent();
  emit("}");
}

void CUDACodeGen::emitAsyncTMACopyL2G(ttng::AsyncTMACopyLocalToGlobalOp op) {
  auto descVar = getVar(op.getDesc());
  auto smemVar = getVar(op.getSrc());
  auto coords = op.getCoord();
  std::string coord0 = coords.size() > 0 ? getVar(coords[0]) : "0";
  std::string coord1 = coords.size() > 1 ? getVar(coords[1]) : "0";

  // Determine if copy splitting is needed.
  // When the tile's fast dimension exceeds the swizzle width, the shared memory
  // layout is "tiled" (sub-blocks), so we must issue multiple TMA L2G
  // instructions — one per column slice of swizzle-width elements.
  auto srcMemDesc = cast<ttg::MemDescType>(op.getSrc().getType());
  auto srcShape = srcMemDesc.getShape();
  auto srcEnc = srcMemDesc.getEncoding();
  int elemBytes = srcMemDesc.getElementType().getIntOrFloatBitWidth() / 8;

  int nCopies = 1;
  int maxContigCols = 0;
  int smemCopyBytes = 0;
  if (auto nvmmaShared = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(srcEnc)) {
    int swizzleBytes = nvmmaShared.getSwizzlingByteWidth();
    if (swizzleBytes > 0 && srcShape.size() == 2) {
      maxContigCols = swizzleBytes / elemBytes;
      int tileCols = srcShape[1];  // fast (innermost) dimension
      nCopies = (tileCols + maxContigCols - 1) / maxContigCols;
      int tileRows = srcShape[0];
      smemCopyBytes = tileRows * maxContigCols * elemBytes;
    }
  }

  for (int c = 0; c < nCopies; c++) {
    std::string smemOffset = (c == 0) ? "" : " + " + std::to_string(c * smemCopyBytes);
    // The split is along the fast (innermost) dimension = coord1 in the IR
    // TMA instruction coord order: {innermost, outermost} = {coord1, coord0}
    std::string coord1Adj = (c == 0) ? coord1 : "(" + coord1 + " + " + std::to_string(c * maxContigCols) + ")";
    emit("if (threadIdx.x == 0) {");
    indent();
    emit("asm volatile(");
    emit("    \"cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\\n\"");
    emit("    :: \"l\"((uint64_t)&" + descVar + "),");
    emit("       \"r\"(" + coord1Adj + "), \"r\"(" + coord0 + "),");
    emit("       \"r\"((unsigned)__cvta_generic_to_shared((char*)" + smemVar + smemOffset + "))");
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

  // Destination shape/encoding
  auto dstShape = dstMemDescType.getShape();
  auto dstEnc = dstMemDescType.getEncoding();
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
  SmallVector<int64_t> strides(rank);
  strides.back() = 1;
  for (int d = rank - 2; d >= 0; d--)
    strides[d] = strides[d + 1] * dstShape[d + 1];

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

  // Group consecutive registers for vectorized cp.async
  SmallVector<std::pair<int, int>> groups;
  {
    int groupStart = 0;
    for (int i = 1; i <= nElems; i++) {
      bool contiguous = false;
      if (i < nElems) {
        auto prevCoords = srcLL.apply({{kReg, i-1}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        auto currCoords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
        contiguous = true;
        for (int d = 0; d < rank - 1; d++) {
          if (currCoords[d].second != prevCoords[d].second) { contiguous = false; break; }
        }
        if (contiguous && currCoords[rank-1].second != prevCoords[rank-1].second + 1)
          contiguous = false;
      }
      if (!contiguous) {
        int count = i - groupStart;
        while (count > 0) {
          int vecBytes = count * bytesPerElem;
          int cpBytes = 0;
          if (vecBytes >= 16) cpBytes = 16;
          else if (vecBytes >= 8) cpBytes = 8;
          else if (vecBytes >= 4) cpBytes = 4;
          int cpElems = cpBytes > 0 ? cpBytes / bytesPerElem : count;
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
  }

  for (auto &[groupStart, groupCount] : groups) {
    int i = groupStart;
    auto regCoords = srcLL.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});

    std::string offsetExpr;
    if (nvmmaShared && swizzleBytes > 0 && rank == 2) {
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
      std::string globalAddr;
      if (asyncUseDeferredAddr) {
        globalAddr = "(const void*)(" + asyncBasePtr + " + " + asyncOffVar + "[" + std::to_string(i) + "])";
      } else {
        globalAddr = "(const void*)" + srcVar + "[" + std::to_string(i) + "]";
      }
      emit("  {");
      emit("    unsigned _sa = __cvta_generic_to_shared(&_dst[" + offsetExpr + "]);");
      if (hasMask) {
        std::string maskE = getElemExpr(mask, std::to_string(i));
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
        std::string srcDeref;
        if (asyncUseDeferredAddr) {
          srcDeref = "*(" + asyncBasePtr + " + " + asyncOffVar + "[" + std::to_string(i + j) + "])";
        } else {
          srcDeref = "*" + srcVar + "[" + std::to_string(i + j) + "]";
        }
        if (hasMask) {
          std::string maskE = getElemExpr(mask, std::to_string(i + j));
          emit("  if (" + maskE + ")");
          emit("    _dst[" + offsetExpr + " + " + std::to_string(j) + "] = " + srcDeref + ";");
          emit("  else");
          if (other) {
            auto otherVar = getVar(other);
            emit("    _dst[" + offsetExpr + " + " + std::to_string(j) + "] = " +
                 otherVar + "[" + std::to_string(i + j) + "];");
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
  emit("__syncthreads();");
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
    int rows = shape[0], cols = shape[1];
    int spt0 = enc.getSizePerThread()[0], spt1 = enc.getSizePerThread()[1];
    int tpw0 = enc.getThreadsPerWarp()[0], tpw1 = enc.getThreadsPerWarp()[1];
    int wpc0 = enc.getWarpsPerCTA()[0], wpc1 = enc.getWarpsPerCTA()[1];
    int totalT0 = tpw0 * wpc0;
    int reps1 = std::max(1, cols / (tpw1 * wpc1 * spt1));
    int strideRep0 = spt1 * reps1 * spt0;
    int totalT1 = tpw1 * wpc1;
    emit("{");
    indent();
    emit("int _ld0 = lane_id / " + std::to_string(tpw1) + ";");
    emit("int _ld1 = lane_id % " + std::to_string(tpw1) + ";");
    emit("int _wd0 = warp_id / " + std::to_string(wpc1) + ";");
    emit("int _wd1 = warp_id % " + std::to_string(wpc1) + ";");
    emit("int _bp0 = _ld0 + _wd0 * " + std::to_string(tpw0) + ";");
    emit("int _bp1 = _ld1 + _wd1 * " + std::to_string(tpw1) + ";");
    emit("#pragma unroll");
    emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
    indent();
    emit("int _rp0 = _i / " + std::to_string(strideRep0) + ";");
    emit("int _s0 = (_i / " + std::to_string(spt1 * reps1) + ") % " + std::to_string(spt0) + ";");
    emit("int _rp1 = (_i / " + std::to_string(spt1) + ") % " + std::to_string(reps1) + ";");
    emit("int _s1 = _i % " + std::to_string(spt1) + ";");
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
      emit("    " + dstVar.str() + "[_i] = " + smemVar.str() + "[_base + _i];");
    else
      emit("    " + dstVar.str() + "[_i] = " + smemVar.str() + "[_base + (_i % " + std::to_string(spt) +
           ") + (_i / " + std::to_string(spt) + ") * " + std::to_string(stride) + "];");
    dedent();
    emit("}");
  } else if (shape.size() >= 2) {
    int rows = shape[0], cols = shape[1];
    int spt0 = enc.getSizePerThread()[0], spt1 = enc.getSizePerThread()[1];
    int tpw0 = enc.getThreadsPerWarp()[0], tpw1 = enc.getThreadsPerWarp()[1];
    int wpc0 = enc.getWarpsPerCTA()[0], wpc1 = enc.getWarpsPerCTA()[1];
    int totalT0 = tpw0 * wpc0;
    int reps1 = std::max(1, cols / (tpw1 * wpc1 * spt1));
    int strideRep0 = spt1 * reps1 * spt0;
    int totalT1 = tpw1 * wpc1;

    // Determine element size for vectorized loads
    int elemBits = 16; // default
    if (elemType == "float" || elemType == "int" || elemType == "uint32_t")
      elemBits = 32;
    else if (elemType == "__half" || elemType == "__nv_bfloat16")
      elemBits = 16;
    int elemBytesLocal = elemBits / 8;

    // Try vectorized load: spt1 contiguous elements in the inner dimension
    int vecLoadBytes = spt1 * elemBytesLocal;
    bool useVecLoad = (spt1 >= 4) && (vecLoadBytes == 8 || vecLoadBytes == 16);
    int nGroups = nElems / spt1; // number of vectorized load groups

    emit("{");
    indent();
    emit("int _ld0 = lane_id / " + std::to_string(tpw1) + ";");
    emit("int _ld1 = lane_id % " + std::to_string(tpw1) + ";");
    emit("int _wd0 = warp_id / " + std::to_string(wpc1) + ";");
    emit("int _wd1 = warp_id % " + std::to_string(wpc1) + ";");
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
      dedent();
      emit("}");
    } else {
      // Scalar fallback
      emit("#pragma unroll");
      emit("for (int _i = 0; _i < " + std::to_string(nElems) + "; _i++) {");
      indent();
      emit("int _rp0 = _i / " + std::to_string(strideRep0) + ";");
      emit("int _s0 = (_i / " + std::to_string(spt1 * reps1) + ") % " + std::to_string(spt0) + ";");
      emit("int _rp1 = (_i / " + std::to_string(spt1) + ") % " + std::to_string(reps1) + ";");
      emit("int _s1 = _i % " + std::to_string(spt1) + ";");
      emit("int _row = _bp0 * " + std::to_string(spt0) + " + _s0 + _rp0 * " + std::to_string(totalT0 * spt0) + ";");
      emit("int _col = _bp1 * " + std::to_string(spt1) + " + _s1 + _rp1 * " + std::to_string(totalT1 * spt1) + ";");
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
