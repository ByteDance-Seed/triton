#!/bin/bash
# Build the fully out-of-tree CUDA emitter plugin (emit_cuda.so).
#
# The emitter sources live entirely in this package (csrc/) -- the Triton source
# tree is UNMODIFIED. We reuse the -I/-D compile flags of an unrelated in-tree
# nvidia source (pulled from compile_commands.json, so we inherit the exact
# MLIR/LLVM/dialect include search paths of the build), add -I<csrc> for the
# emitter's own headers and -fvisibility=hidden so only the TRITON_PLUGIN_API
# entry point `tritonGetPluginInfo` is exported, then compile+link against the
# already-built libtriton.so (which must be built with TRITON_EXT_ENABLED=1 so
# its symbols are resolvable at dlopen time).
set -euo pipefail

PKG=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # .../triton_cuda_backend
REPO=$(cd "$PKG/.." && pwd)                          # Triton repo root
BUILD=$REPO/build/cmake.linux-x86_64-cpython-3.11
SRCDIR=$PKG/csrc
OBJDIR=$PKG/build                                    # gitignored intermediates
LIBTRITON_DIR=$REPO/python/triton/_C
OUT=$PKG/triton_cuda_backend/emit_cuda.so
mkdir -p "$OBJDIR"

# Template in-tree source whose compile command supplies the common Triton/MLIR
# include + define flags. Any nvidia lib source works; it is NOT compiled here.
TEMPLATE=third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp

FLAGS=$(python3 - "$BUILD/compile_commands.json" "$TEMPLATE" <<'PY'
import json, sys, shlex
d = json.load(open(sys.argv[1]))
needle = sys.argv[2]
for e in d:
    if e["file"].endswith(needle):
        body = e["command"].split(" -o ", 1)[0]
        toks = shlex.split(body)[1:]  # drop /usr/bin/c++
        print(" ".join(toks))
        break
else:
    sys.exit("template source not found in compile_commands.json: " + needle)
PY
)

echo "FLAGS: $FLAGS"
EXTRA="-I$SRCDIR -fvisibility=hidden -fvisibility-inlines-hidden"

OBJS=()
for src in CUDACodeGen.cpp TritonGPUToCUDA.cpp EmitCudaPlugin.cpp; do
    obj=$OBJDIR/${src%.cpp}.o
    echo "=== compiling $src ==="
    /usr/bin/c++ $FLAGS $EXTRA -c "$SRCDIR/$src" -o "$obj"
    OBJS+=("$obj")
done

# The emitter uses MLIR's ControlFlowToSCF conversion (TritonGPUToCUDA.cpp).
# libtriton does not export that lib, so statically link it into the plugin.
# Static archives must follow the objects that reference them on the link line.
LLVM_LIB=$(ls -d /root/.triton/llvm/*/lib | head -1)

echo "=== linking $OUT ==="
/usr/bin/c++ -shared -fvisibility=hidden -o "$OUT" "${OBJS[@]}" \
    -L"$LLVM_LIB" -lMLIRControlFlowToSCF \
    -L"$LIBTRITON_DIR" -ltriton -Wl,-rpath,"$LIBTRITON_DIR"

echo "=== done: $OUT ==="
ls -la "$OUT"
echo "=== exported symbols (should be ~just tritonGetPluginInfo) ==="
nm -D --defined-only "$OUT" | grep -i pluginfo || nm -D --defined-only "$OUT" | grep -i tritonGetPlugin || true
