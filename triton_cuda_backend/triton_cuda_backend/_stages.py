"""Non-invasive CUDA-C++ backend stages for Triton (sm90a).

This module replaces the default LLVM/PTX lowering with a CUDA-C++ emitter
pipeline (TTGIR -> CUDA C++ -> PTX via nvcc -> cubin via ptxas) **without
patching any Triton core file**.  It is wired in through the official
``knobs.runtime.add_stages_inspection_hook`` extension point, which the nvidia
backend invokes at the end of ``CUDABackend.add_stages`` with the live
``stages`` dict.

Activation: set ``TRITON_EMIT_CUDA=1``.  Only sm90 (Hopper) targets are
redirected; every other arch keeps the stock pipeline so a globally-set env var
is still safe on non-Hopper GPUs.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from triton import knobs


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
    """Call the C++ emitter exposed on the libtriton ``nvidia`` module."""
    from triton._C.libtriton import nvidia
    ptx_version = get_ptx_version_from_options(options, capability)
    return nvidia.translate_ttgir_to_cuda(
        src, capability, options.num_warps, options.num_ctas, ptx_version)


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
        # Cache-keying call. Only differentiate when the emitter is active.
        if os.environ.get("TRITON_EMIT_CUDA", "") == "1":
            return ("cuda-emit-sm90", "cuda-emit-sm90")
        return ("", "")

    # Stage-rewriting call.
    backend, stages, options, language, capability = args
    if os.environ.get("TRITON_EMIT_CUDA", "") != "1":
        return
    if capability // 10 != 9:
        return
    stages.pop("llir", None)
    stages["ptx"] = lambda src, metadata: _make_ptx(src, metadata, options, capability)
    stages["cubin"] = lambda src, metadata: _make_cubin(src, metadata, options, capability)


def register():
    """Install the hook. Idempotent."""
    if knobs.runtime.add_stages_inspection_hook is cuda_stages_hook:
        return
    knobs.runtime.add_stages_inspection_hook = cuda_stages_hook
