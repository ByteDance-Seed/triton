"""Non-invasive CUDA-C++ backend plugin for Triton.

Importing this package installs the ``add_stages_inspection_hook`` (and a
``JITFunction.run`` patch enabling the per-launch ``emit_cuda=True`` kwarg) that
make the nvidia backend emit CUDA C++ (compiled by nvcc/ptxas) instead of going
through LLVM/PTX, for sm90 targets. Activate with ``TRITON_EMIT_CUDA=1`` or the
``kernel[grid](..., emit_cuda=True)`` kwarg.

It patches no Triton source file. The emitter ships as an out-of-tree shared
library (``emit_cuda.so``) loaded via Triton's official plugin ABI
(``TRITON_PLUGIN_PATHS``); the only other coupling is the official runtime
``add_stages_inspection_hook``. Requires a libtriton built with
``TRITON_EXT_ENABLED=1`` so the plugin can resolve its symbols.
"""
import os

# Register the emitter plugin .so on TRITON_PLUGIN_PATHS *before* libtriton is
# imported (loadPlugins() runs at libtriton module-init). This import must come
# before ._stages, which pulls in triton.
_PLUGIN_SO = os.path.join(os.path.dirname(__file__), "emit_cuda.so")
if os.environ.get("TRITON_CUDA_PLUGIN", "1") == "1" and os.path.exists(_PLUGIN_SO):
    _existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    if _PLUGIN_SO not in _existing.split(":"):
        os.environ["TRITON_PLUGIN_PATHS"] = (
            f"{_PLUGIN_SO}:{_existing}" if _existing else _PLUGIN_SO)
    os.environ.setdefault("TRITON_CUDA_PLUGIN", "1")

from ._stages import register, cuda_stages_hook  # noqa: E402,F401

register()
