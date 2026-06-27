# triton_cuda_backend

A **non-invasive, out-of-tree** CUDA-C++ backend for Triton. When activated, sm90
(Hopper) kernels are lowered as `TTGIR → CUDA C++ → PTX (nvcc) → cubin (ptxas)`
instead of going through Triton's stock LLVM/PTX path.

The Triton source tree is **never modified** (`git diff` vs upstream over the
Triton tree is empty). Everything lives in this package:

- `csrc/` — the C++ emitter, built into an out-of-tree plugin `emit_cuda.so`
  that Triton loads through its official plugin ABI (`tritonGetPluginInfo`).
- `triton_cuda_backend/` — the Python package: it installs the official
  `add_stages_inspection_hook` to redirect the pipeline, plus a runtime patch
  enabling the per-launch `emit_cuda=True` kwarg.
- `CMakeLists.txt` — builds the plugin.

---

## Prerequisites

Run everything **inside the container** (`zhengsize-vibecuda`); the LLVM/MLIR
toolchain and `nvcc`/`ptxas` live there, not on the host.

- `cmake` (>= 3.20), `ninja`
- An LLVM/MLIR install under `~/.triton/llvm/*` (Triton's pinned LLVM; provides
  the MLIR CMake config package that the plugin's `find_package(MLIR)` consumes)
- CUDA toolkit (`nvcc`, `ptxas`) — auto-located via `$PATH`, `$CUDA_HOME`, or
  `/usr/local/cuda/bin`
- A GPU of compute capability sm90 (e.g. H800)

All paths below are relative to the **Triton repo root** (the parent of this
directory), referred to as `$TRITON_ROOT`.

---

## Build

This is a two-phase build: build Triton first (it produces `libtriton.so` and the
tablegen-generated dialect headers the plugin includes), then build the plugin.

### Phase 1 — build Triton

The plugin links against `libtriton.so`, so libtriton **must** be built with
`TRITON_EXT_ENABLED=1` (this gives its MLIR symbols default visibility so the
plugin can resolve them at `dlopen`).

> **`TRITON_EXT_ENABLED` is a build-time flag, NOT a source modification.** It only
> flips symbol visibility (`-fvisibility=default` for the MLIR/LLVM symbols) so an
> out-of-tree `dlopen`'d plugin can resolve them; it changes **zero** lines of Triton
> source. Build a *pristine* Triton checkout with this flag set — `git status` stays
> clean. The flag is upstream Triton's own (it exists precisely to support external
> plugins like this one).

```bash
cd $TRITON_ROOT
TRITON_EXT_ENABLED=1 \
TRITON_APPEND_CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-Wno-error=attributes" \
pip install -e . --no-build-isolation -v
```

`-Wno-error=attributes` is required when building with **GCC**: with
`TRITON_EXT_ENABLED=1` (default visibility) GCC turns a visibility warning in
`python/src/gluon_ir.cc` into a hard error. The flag downgrades just that one
warning; it does **not** modify any Triton source. (Triton's CMake is written
for clang, which does not hit this; if a full `clang`/`clang++` is available you
can instead use `TRITON_BUILD_WITH_CLANG_LLD=1` and drop the flag.)

This produces:
- `$TRITON_ROOT/python/triton/_C/libtriton.so`
- `$TRITON_ROOT/build/cmake.<platform>-cpython-<ver>/` (tablegen headers)

### Phase 2 — build the plugin

```bash
cd $TRITON_ROOT/triton_cuda_backend
cmake -S . -B build -G Ninja
cmake --build build
```

CMake auto-detects `MLIR_DIR` (`~/.triton/llvm/*/lib/cmake/mlir`) and the Triton
build dir; override with `-DMLIR_DIR=...` / `-DTRITON_BUILD_DIR=...` if needed.
The output is `triton_cuda_backend/emit_cuda.so` (exports only
`tritonGetPluginInfo`).

### Install the Python package

```bash
cd $TRITON_ROOT/triton_cuda_backend
python3 -m pip install --force-reinstall --no-deps --no-build-isolation .
rm -rf ~/.triton/cache   # always clear the kernel cache after a rebuild
```

---

## Non-invasive guarantee (zero Triton source edits)

The Triton source tree stays **pristine** — `git status` under `lib/` and
`python/triton/` is clean. Everything this backend needs lives out-of-tree in
`triton_cuda_backend/`:

| Coupling to Triton | How (no source patch) |
| --- | --- |
| Route sm90 kernels to the CUDA emitter | official `knobs.runtime.add_stages_inspection_hook` (set in `_stages.py`) |
| `emit_cuda=True` per-launch kwarg | runtime wrap of `JITFunction.run` (pops the kwarg before binding); overrides the `TRITON_EMIT_CUDA` env var |
| The emitter itself | `emit_cuda.so`, loaded via Triton's official plugin ABI (`TRITON_PLUGIN_PATHS` / `tritonGetPluginInfo`) |
| Symbol resolution for that `.so` | `TRITON_EXT_ENABLED=1` **build flag** (visibility only — see above) |
| Extra Gluon builtins (e.g. `tma.async_reduce_shared_to_global`) | injected at `import` by `_gluon_ext.py` (monkeypatch onto the gluon `tma` module); wraps **pristine** upstream bindings (`create_async_tma_reduce`, `DESCRIPTOR_REDUCE_KIND`, `AsyncTMAReduceOp`), so **no libtriton rebuild** for these |

There are **no** edits to Triton's MLIR passes, dialects, verifiers, or Python
source. (The tutorials use only power-of-2 WGMMA tiles, so no pow2-constraint
relaxation is needed; native non-pow2 WGMMA N=80 is the one capability that would
require a core C++ patch, and the shipped tutorials don't use it.)

---

## Activation

Two equivalent ways to route sm90 kernels through the CUDA emitter:

### 1. Environment variable (process-wide)

```bash
TRITON_EMIT_CUDA=1 python3 your_script.py
```

Every sm90 kernel in the process is emitted as CUDA.

### 2. Per-launch kwarg

```python
import triton

out = kernel[grid](x, y, out, n, BLOCK=1024, emit_cuda=True)
```

Only this launch is emitted as CUDA; everything else keeps the stock PTX path.
No environment variable and no explicit `import triton_cuda_backend` are needed —
the package is installed with a `.pth` autoloader that wires up the `emit_cuda=`
kwarg the moment `triton` is first imported. The kwarg overrides the env var for
that launch, and CUDA-emitted kernels use a distinct cache key, so toggling never
returns a stale binary.

> Non-sm90 GPUs always keep the stock pipeline, so a globally-set
> `TRITON_EMIT_CUDA=1` is safe on non-Hopper hardware.

### Dumping the generated CUDA

```bash
TRITON_CUDA_DUMP=/tmp/cuda_out TRITON_EMIT_CUDA=1 python3 your_script.py
ls /tmp/cuda_out/*.cu
```

A `.cu` file is written for every kernel that goes through the emitter — a handy
way to confirm activation actually took effect.

---

## Test

The smoke test compiles and runs an elementwise kernel and checks the result
against a PyTorch reference.

```bash
cd $TRITON_ROOT

# Per-launch kwarg path (the test passes emit_cuda=True; no env var needed):
python3 triton_cuda_backend/tests/smoke.py

# Or force the whole process through the emitter:
TRITON_EMIT_CUDA=1 python3 triton_cuda_backend/tests/smoke.py
```

Expected output:

```
mode=PLUGIN allclose=True maxerr=<small>
```

`mode=PLUGIN` confirms the out-of-tree `emit_cuda.so` plugin was loaded.

For broader validation, run the official tutorials under the emitter, e.g.:

```bash
TRITON_EMIT_CUDA=1 python3 python/tutorials/09-persistent-matmul.py
```

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `gluon_ir.cc ... 'GluonLayouts' declared with greater visibility ... [-Werror=attributes]` | Building Triton with GCC + `TRITON_EXT_ENABLED=1`. Add `TRITON_APPEND_CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-Wno-error=attributes"` (see Phase 1). |
| `undefined symbol: ...ControlFlowToSCFTransformation...` when loading the plugin | libtriton does not export that MLIR lib; the plugin static-links `libMLIRControlFlowToSCF.a` (handled by `CMakeLists.txt`). Rebuild the plugin. |
| `emit_cuda plugin pass not registered` / `Keyword argument emit_cuda ... unrecognised` | The plugin `.so` was not on `TRITON_PLUGIN_PATHS` before libtriton loaded, or libtriton was not built with `TRITON_EXT_ENABLED=1`. Rebuild Phase 1 with the flag, then reinstall this package. Ensure the package is imported before/at first `import triton`. |
| `mode=PYBIND` or no `.cu` produced | Activation did not take effect. Confirm `TRITON_EMIT_CUDA=1` is set or `emit_cuda=True` is passed, and that the target is sm90. |
| Emitter changes have no effect | Stale kernel cache. `rm -rf ~/.triton/cache` and re-run. |
