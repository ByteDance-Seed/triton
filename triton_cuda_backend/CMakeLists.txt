# Out-of-tree build for the CUDA emitter plugin (emit_cuda.so).
#
# This does NOT touch the Triton source tree. It consumes MLIR/LLVM through
# their installed CMake config packages (find_package), and the Triton dialect
# headers through the source tree + the tablegen-generated headers under
# Triton's build dir (which must be built first -- that is inherent to any
# out-of-tree MLIR plugin, not a workaround).
#
# Usage:
#   cmake -S triton_cuda_backend -B triton_cuda_backend/build
#   cmake --build triton_cuda_backend/build
#
# Auto-detected (override with -D... if needed):
#   MLIR_DIR           lib/cmake/mlir inside the LLVM install
#   TRITON_BUILD_DIR   the build/cmake.<platform>-<python> dir
cmake_minimum_required(VERSION 3.20)
project(triton_cuda_backend LANGUAGES CXX)

set(TRITON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "Triton repo root")

# --- Locate MLIR/LLVM ---------------------------------------------------------
# Prefer an explicit MLIR_DIR; else glob the pip-managed LLVM install.
if(NOT DEFINED MLIR_DIR)
  file(GLOB _llvm_roots "$ENV{HOME}/.triton/llvm/*/lib/cmake/mlir")
  if(_llvm_roots)
    list(GET _llvm_roots 0 MLIR_DIR)
  endif()
endif()
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# --- Locate Triton's build dir (for tablegen-generated headers) ---------------
if(NOT DEFINED TRITON_BUILD_DIR)
  file(GLOB _tb "${TRITON_ROOT}/build/cmake.*")
  list(LENGTH _tb _tb_n)
  if(_tb_n EQUAL 0)
    message(FATAL_ERROR "No ${TRITON_ROOT}/build/cmake.* found; build libtriton "
                        "first, or pass -DTRITON_BUILD_DIR=...")
  elseif(_tb_n GREATER 1)
    message(FATAL_ERROR "Multiple build dirs found; pick one with "
                        "-DTRITON_BUILD_DIR=...:\n  ${_tb}")
  endif()
  list(GET _tb 0 TRITON_BUILD_DIR)
endif()
message(STATUS "Using Triton build dir: ${TRITON_BUILD_DIR}")

set(LIBTRITON "${TRITON_ROOT}/python/triton/_C/libtriton.so")
if(NOT EXISTS "${LIBTRITON}")
  message(FATAL_ERROR "libtriton.so not found at ${LIBTRITON}; build Triton first")
endif()

# --- The plugin module --------------------------------------------------------
add_library(emit_cuda MODULE
  csrc/CUDACodeGen.cpp
  csrc/TritonGPUToCUDA.cpp
  csrc/EmitCudaPlugin.cpp)

target_include_directories(emit_cuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc        # emitter's own headers
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
  # Triton source headers (not exported via any CMake package)
  ${TRITON_ROOT}/include
  ${TRITON_ROOT}
  ${TRITON_ROOT}/third_party
  ${TRITON_ROOT}/third_party/nvidia/include
  ${TRITON_ROOT}/python/src
  # Triton tablegen-generated headers (only exist after building Triton)
  ${TRITON_BUILD_DIR}/include
  ${TRITON_BUILD_DIR}/third_party
  ${TRITON_BUILD_DIR}/third_party/nvidia/include)

target_compile_definitions(emit_cuda PRIVATE
  ${LLVM_DEFINITIONS}
  __STDC_FORMAT_MACROS
  TRITON_EXT_ENABLED=1)

# Match Triton/MLIR's build settings (no RTTI/exceptions; hide all but the
# plugin C entry point). -Wno-attributes silences the visibility-attribute
# warning Triton headers trip with default visibility.
target_compile_features(emit_cuda PRIVATE cxx_std_17)
target_compile_options(emit_cuda PRIVATE
  -fno-rtti -fno-exceptions
  -fvisibility=hidden -fvisibility-inlines-hidden
  -Wno-attributes)

# Link only the one MLIR static lib that libtriton does NOT export
# (ControlFlowToSCF, used by liftCFGToSCF). Every other MLIR/dialect symbol is
# resolved from libtriton.so at dlopen time, so link the archive file directly
# rather than the imported target (whose INTERFACE deps would drag in the whole
# static MLIR closure and risk ODR clashes with libtriton's copy).
target_link_libraries(emit_cuda PRIVATE
  ${LLVM_LIBRARY_DIR}/libMLIRControlFlowToSCF.a
  ${LIBTRITON})
set_target_properties(emit_cuda PROPERTIES
  BUILD_RPATH "${TRITON_ROOT}/python/triton/_C"
  INSTALL_RPATH "${TRITON_ROOT}/python/triton/_C")

# Emit `emit_cuda.so` (no lib prefix) straight into the Python package so an
# editable install picks it up.
set_target_properties(emit_cuda PROPERTIES
  PREFIX ""
  OUTPUT_NAME emit_cuda
  SUFFIX ".so"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/triton_cuda_backend")
