"""Startup autoloader (installed at site-packages root, triggered by a .pth).

Kept deliberately tiny and *Triton-free* until the env gate passes, so that it
is inert during pip/cmake builds and on non-emit Python invocations.  Only when
``TRITON_EMIT_CUDA=1`` does it import the plugin package, which registers the
``add_stages_inspection_hook``.  Any failure is reported but never aborts the
interpreter (a .pth that raises would break every Python process).
"""
import os
import sys

if os.environ.get("TRITON_EMIT_CUDA", "") == "1":
    try:
        import triton_cuda_backend  # noqa: F401  -- import registers the hook
    except Exception as e:  # pragma: no cover - defensive
        print(f"[triton_cuda_backend] autoload skipped: {e}", file=sys.stderr)
