"""Non-invasive CUDA-C++ backend stages for Triton (sm90a).

This module replaces the default LLVM/PTX lowering with a CUDA-C++ emitter
pipeline (TTGIR -> CUDA C++ -> PTX via nvcc -> cubin via ptxas) **without
patching any Triton core file**.  It is wired in through the official
``knobs.runtime.add_stages_inspection_hook`` extension point, which the nvidia
backend invokes at the end of ``CUDABackend.add_stages`` with the live
``stages`` dict.

Activation, either of:
  * ``TRITON_EMIT_CUDA=1`` (process-wide), or
  * a per-launch ``kernel[grid](..., emit_cuda=True)`` kwarg (see
    ``_install_run_patch``), which overrides the env var for that launch.
Only sm90 (Hopper) targets are redirected; every other arch keeps the stock
pipeline so a globally-set env var is still safe on non-Hopper GPUs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading

from triton import knobs

# Per-launch activation flag set by the ``emit_cuda=`` kernel kwarg (see
# ``_install_run_patch``). Thread-local so concurrent launches don't clobber
# each other. ``None`` means "unset -> fall back to the env var".
_tls = threading.local()


def _emit_active():
    """Whether the CUDA emitter should fire for the launch being compiled.

    A per-launch ``emit_cuda=`` kwarg (thread-local) takes precedence; absent
    that, the process-wide ``TRITON_EMIT_CUDA=1`` env var decides.
    """
    flag = getattr(_tls, "emit_cuda", None)
    if flag is not None:
        return bool(flag)
    return os.environ.get("TRITON_EMIT_CUDA", "") == "1"


# Per-launch emit_cuda feature flags exposed as kernel kwargs (popped before
# binding by ``_install_run_patch``). Each maps a kwarg name to its thread-local
# attribute. These drive the structural IR->IR passes (persistent tile loop,
# epilogue overlap); the emitter itself stays a pure printer.
_FEATURE_KWARGS = ("persistent", "epilogue_overlap", "multicast", "wg_pingpong")


def _feature_flag(name):
    """Current per-launch value of a structural feature flag (default False)."""
    return bool(getattr(_tls, name, False))


def _resolve_tool(name):
    """Locate a CUDA toolchain binary (nvcc/ptxas).

    Prefer ``$PATH``; otherwise fall back to the standard CUDA install layout so
    the plugin works even when the interpreter was launched without CUDA on
    ``PATH`` (common under test runners and service shells).
    """
    found = shutil.which(name)
    if found:
        return found
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", name))
    candidates += [
        f"/usr/local/cuda/bin/{name}",
        f"/usr/local/cuda-12/bin/{name}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return name  # let subprocess raise a clear FileNotFoundError
# Module-level helpers shipped by the stock nvidia backend (present in 3.7.0).
from triton.backends.nvidia.compiler import (
    get_ptx_version_from_options,
    sm_arch_from_capability,
)


def _translate_ttgir_to_cuda(src, capability, options):
    """Run the emitter as an out-of-tree MLIR pass loaded via TRITON_PLUGIN_PATHS.

    The plugin ABI exposes no new pybind function, and ``ModuleOp`` has no
    string-attribute getter in the Python bindings, so the ``emit_cuda`` pass
    hands its results back through a temp file whose path we pass as the 5th
    pass argument. Requires libtriton built with ``TRITON_EXT_ENABLED=1`` and
    ``TRITON_PLUGIN_PATHS`` pointing at ``emit_cuda.so``.

    File format (see EmitCudaPlugin.cpp):
      line 1: "OK" or "ERR"
      on ERR: remainder = error message
      on OK:  line 2 = kernel name
              line 3 = "<shared> <scratchSize> <scratchAlign> <numWarps>"
              remainder = CUDA C++ source (verbatim)
    """
    from triton._C.libtriton import ir, passes
    if not hasattr(passes, "plugin") or not hasattr(passes.plugin, "emit_cuda"):
        raise RuntimeError(
            "emit_cuda plugin pass not registered. Ensure TRITON_PLUGIN_PATHS "
            "points at emit_cuda.so and libtriton was built with "
            "TRITON_EXT_ENABLED=1.")
    ptx_version = get_ptx_version_from_options(options, capability)
    with tempfile.NamedTemporaryFile(delete=False, mode="r",
                                     suffix=".cudaresult") as fout:
        out_path = fout.name
    try:
        pm = ir.pass_manager(src.context)
        args = [str(capability), str(options.num_warps), str(options.num_ctas),
                str(ptx_version), out_path,
                str(int(_feature_flag("persistent"))),
                str(int(_feature_flag("epilogue_overlap"))),
                str(int(_feature_flag("multicast"))),
                str(int(_feature_flag("wg_pingpong")))]
        passes.plugin.emit_cuda(pm, args)
        pm.run(src, "emit_cuda")
        with open(out_path) as f:
            payload = f.read()
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

    if not payload:
        raise RuntimeError("emit_cuda plugin produced no result")
    status, _, rest = payload.partition("\n")
    if status == "ERR":
        raise RuntimeError(rest or "emit_cuda plugin failed")
    if status != "OK":
        raise RuntimeError(f"emit_cuda plugin produced malformed result: "
                           f"{payload[:200]!r}")
    # line 2 = name, line 3 = sizes, remainder = source body.
    name_line, _, rest = rest.partition("\n")
    sizes_line, _, cuda_src = rest.partition("\n")
    shared, scratch_size, scratch_align, num_warps = (
        int(x) for x in sizes_line.split())
    return {
        "cuda_src": cuda_src,
        "kernel_name": name_line,
        "shared_mem_size": shared,
        "global_scratch_size": scratch_size,
        "global_scratch_align": scratch_align or 1,
        "num_warps": num_warps,
    }


def _make_ptx(src, metadata, options, capability):
    """TTGIR module -> CUDA C++ (C++ pass) -> PTX (nvcc).

    Folded into a single 'ptx' stage so the compiler framework never sees a
    novel 'cuda' asm extension -- that keeps ``compiler/compiler.py`` untouched.
    The emitter-side metadata that the LLVM path would normally populate
    (shared, name, num_warps, global scratch) is filled in here.
    """
    result = _translate_ttgir_to_cuda(src, capability, options)
    cuda_src = result["cuda_src"]

    metadata["name"] = result["kernel_name"]
    metadata["shared"] = result["shared_mem_size"]
    # Warp-specialized kernels launch more warps than options.num_warps; the
    # emitter reports the real total so the launcher sizes blockDim correctly.
    ws_num_warps = result.get("num_warps", 0)
    if ws_num_warps and ws_num_warps > options.num_warps:
        metadata["num_warps"] = ws_num_warps
    if result.get("global_scratch_size", 0):
        metadata["global_scratch_size"] = result["global_scratch_size"]
        metadata["global_scratch_align"] = result.get("global_scratch_align", 1) or 1
    for key in ("tmem_size", "global_scratch_size", "global_scratch_align",
                "profile_scratch_size", "profile_scratch_align", "maxntid"):
        metadata.setdefault(key, 0)

    dump_path = os.environ.get("TRITON_CUDA_DUMP")
    if dump_path:
        if dump_path.endswith("/") or os.path.isdir(dump_path):
            os.makedirs(dump_path, exist_ok=True)
            out = os.path.join(dump_path, result["kernel_name"] + ".cu")
        else:
            out = dump_path
        with open(out, "w") as f:
            f.write(cuda_src)

    arch = sm_arch_from_capability(capability)
    fmad = ['--fmad=true'] if options.enable_fp_fusion else ['--fmad=false']
    lineinfo = [] if knobs.compilation.disable_line_info else ['-lineinfo']
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.cu') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
        fsrc.write(cuda_src)
        fsrc.flush()
        ptx_path = fsrc.name + '.ptx'
        # --use_fast_math implies --ftz=true; override back to --ftz=false so
        # FP32 keeps Triton's non-FTZ denormal behaviour (matches mul.f32),
        # while still getting the fast transcendental intrinsics.
        nvcc_cmd = [
            _resolve_tool('nvcc'), '-ptx', f'--gpu-architecture={arch}', '-O3',
            '--use_fast_math', '--ftz=false', *fmad, *lineinfo,
            '-std=c++17', fsrc.name, '-o', ptx_path,
        ]
        try:
            subprocess.run(nvcc_cmd, check=True, close_fds=False,
                           stdout=flog, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            with open(flog.name) as lf:
                log = lf.read()
            raise RuntimeError(
                f"`nvcc` failed with error code {e.returncode}\n"
                f"`nvcc` command: {' '.join(nvcc_cmd)}\n`nvcc` log:\n{log}\n"
                f"CUDA source (first 2000 chars):\n{cuda_src[:2000]}\n")
        finally:
            for p in (fsrc.name, flog.name):
                if os.path.exists(p):
                    os.remove(p)
        with open(ptx_path) as f:
            ptx = f.read()
        if os.path.exists(ptx_path):
            os.remove(ptx_path)
    names = re.findall(r"\.visible \.entry ([a-zA-Z_][a-zA-Z0-9_]*)", ptx)
    assert len(names) == 1, f"expected 1 kernel, found {names}"
    metadata["name"] = names[0]
    return ptx


def _make_cubin(src, metadata, options, capability):
    """PTX (nvcc-generated) -> cubin (system ptxas)."""
    arch = sm_arch_from_capability(capability)
    fmad = [] if options.enable_fp_fusion else ['--fmad=false']
    lineinfo = [] if knobs.compilation.disable_line_info else ['-lineinfo']
    # Pin the per-thread register baseline so warp-specialized setmaxnreg.dec/inc
    # match the emitter's assumed baseline; otherwise ptxas picks an arbitrary
    # value and setmaxnreg faults at launch.
    maxreg = ['--maxrregcount=255']
    if 'setmaxnreg' in src and '.maxnreg' not in src:
        total_threads = metadata.get('num_warps', options.num_warps) * 32
        baseline = (65536 // total_threads) & ~7
        maxreg = [f'--maxrregcount={baseline}']
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
        fsrc.write(src)
        fsrc.flush()
        cubin_path = fsrc.name + '.cubin'
        ptxas_cmd = [
            _resolve_tool('ptxas'), f'--gpu-name={arch}', '-O3', *maxreg, *fmad, *lineinfo,
            fsrc.name, '-o', cubin_path,
        ]
        try:
            subprocess.run(ptxas_cmd, check=True, close_fds=False,
                           stdout=flog, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            with open(flog.name) as lf:
                log = lf.read()
            raise RuntimeError(
                f"`ptxas` failed with error code {e.returncode}\n"
                f"`ptxas` command: {' '.join(ptxas_cmd)}\n`ptxas` log:\n{log}\n")
        finally:
            for p in (fsrc.name, flog.name):
                if os.path.exists(p):
                    os.remove(p)
        with open(cubin_path, 'rb') as f:
            cubin = f.read()
        if os.path.exists(cubin_path):
            os.remove(cubin_path)
    return cubin


def cuda_stages_hook(*args):
    """Dual-purpose ``add_stages_inspection_hook``.

    Triton 3.7.0 invokes this single hook object two different ways:

    1. With **no arguments** (jit.py / compiler.py) to obtain a
       ``(cache_key, cache_hash)`` pair that is folded into the kernel cache key.
       This keeps CUDA-emitted kernels in a distinct cache bucket from the stock
       PTX ones so toggling ``TRITON_EMIT_CUDA`` never returns a stale binary.
    2. With ``(backend, stages, options, language, capability)`` at the end of
       ``CUDABackend.add_stages`` to actually rewrite the pipeline.

    We dispatch on ``len(args)``.
    """
    if len(args) == 0:
        # Cache-keying call. Only differentiate when the emitter is active. Fold
        # the structural feature flags into the key so toggling persistent /
        # epilogue_overlap never returns a stale binary.
        if _emit_active():
            feats = ",".join(f"{k}={int(_feature_flag(k))}"
                             for k in _FEATURE_KWARGS)
            key = f"cuda-emit-sm90;{feats}"
            return (key, key)
        return ("", "")

    # Stage-rewriting call.
    backend, stages, options, language, capability = args
    if not _emit_active():
        return
    if capability // 10 != 9:
        return
    stages.pop("llir", None)
    stages["ptx"] = lambda src, metadata: _make_ptx(src, metadata, options, capability)
    stages["cubin"] = lambda src, metadata: _make_cubin(src, metadata, options, capability)


_run_patched = False


def _install_run_patch():
    """Make ``kernel[grid](..., emit_cuda=True)`` a per-launch activation toggle.

    Triton's ``JITFunction.run`` binds every kwarg to the kernel signature
    (``jit.py`` ``binder(*args, **kwargs)``), so an undeclared ``emit_cuda``
    kwarg would raise. We wrap ``run`` to pop ``emit_cuda`` *before* binding and
    expose it to ``cuda_stages_hook`` via the thread-local; the wrapped call
    happens before both the cache-key hook and stage rewrite, so the choice is
    reflected in the cache key (no stale binary). Pure runtime patch -- no
    Triton source is modified.
    """
    global _run_patched
    if _run_patched:
        return
    from triton.runtime.jit import JITFunction

    _orig_run = JITFunction.run

    # All emit_cuda control kwargs popped before signature binding. ``emit_cuda``
    # toggles the backend; the rest are structural feature flags.
    _ctrl_kwargs = ("emit_cuda",) + _FEATURE_KWARGS

    def _run(self, *args, **kwargs):
        present = [k for k in _ctrl_kwargs if k in kwargs]
        if not present:
            return _orig_run(self, *args, **kwargs)
        prev = {k: getattr(_tls, k, None) for k in _ctrl_kwargs}
        for k in present:
            setattr(_tls, k, bool(kwargs.pop(k)))
        try:
            return _orig_run(self, *args, **kwargs)
        finally:
            for k, v in prev.items():
                setattr(_tls, k, v)

    JITFunction.run = _run
    _run_patched = True


def register():
    """Install the hook and the ``emit_cuda=`` kwarg patch. Idempotent."""
    _install_run_patch()
    if knobs.runtime.add_stages_inspection_hook is cuda_stages_hook:
        return
    knobs.runtime.add_stages_inspection_hook = cuda_stages_hook
