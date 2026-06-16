"""Startup autoloader (installed at site-packages root, triggered by a .pth).

Two activation paths are wired up here, both without importing Triton at
interpreter startup (so this stays inert during pip/cmake builds and in any
process that never touches Triton):

1. ``TRITON_EMIT_CUDA=1`` -- eagerly import the plugin package, redirecting
   *every* sm90 kernel to the CUDA emitter.
2. ``kernel[grid](..., emit_cuda=True)`` -- a per-launch opt-in. For the kwarg
   to be accepted, the plugin's ``JITFunction.run`` patch must be installed; we
   arrange that lazily by hooking the *first* ``import triton`` and registering
   right after it finishes. No env var or explicit ``import triton_cuda_backend``
   is needed.

The emitter ``.so`` must be on ``TRITON_PLUGIN_PATHS`` before libtriton loads
(its module-init runs ``loadPlugins``); we set that env var cheaply at startup
(string ops + ``find_spec``, no Triton import), so it is in place by the time
Triton is imported under either path.

Any failure here is reported but never aborts the interpreter -- a .pth that
raises would break every Python process in the environment.
"""
import os
import sys


def _plugin_so():
    """Locate ``emit_cuda.so`` without importing the (Triton-heavy) package."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("triton_cuda_backend")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    so = os.path.join(list(spec.submodule_search_locations)[0], "emit_cuda.so")
    return so if os.path.exists(so) else None


def _ensure_plugin_path():
    """Put the emitter .so on TRITON_PLUGIN_PATHS (idempotent, no import)."""
    so = _plugin_so()
    if not so:
        return
    existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    if so not in existing.split(":"):
        os.environ["TRITON_PLUGIN_PATHS"] = f"{so}:{existing}" if existing else so
    os.environ.setdefault("TRITON_CUDA_PLUGIN", "1")


def _register_now():
    try:
        import triton_cuda_backend  # noqa: F401 -- import installs hook + kwarg patch
    except Exception as e:  # pragma: no cover - defensive
        print(f"[triton_cuda_backend] autoload skipped: {e}", file=sys.stderr)


class _TritonImportHook:
    """One-shot meta-path finder that installs the plugin right after Triton's
    own import completes, then removes itself. Delegates the real import to the
    normal machinery so it cannot break ``import triton``."""

    def find_spec(self, name, path=None, target=None):
        if name != "triton":
            return None
        # Drop ourselves so the real finders resolve Triton (no recursion).
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        import importlib.util
        spec = importlib.util.find_spec("triton")
        if spec is None or spec.loader is None:
            return None
        _orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=_orig_exec):
            _orig(module)
            _register_now()

        try:
            spec.loader.exec_module = exec_module
        except Exception:
            # Loader rejects monkeypatching; fall back to registering after the
            # normal import returns (best effort).
            return None
        return spec


# --- run at interpreter startup (cheap; no Triton import) --------------------
_ensure_plugin_path()

if os.environ.get("TRITON_EMIT_CUDA", "") == "1":
    # Eager path: emit for every sm90 kernel.
    _register_now()
elif "triton" in sys.modules:
    # Triton already imported before us (unusual under a .pth); register now so
    # the emit_cuda= kwarg still works.
    _register_now()
else:
    # Lazy path: install the kwarg patch the moment Triton is first imported.
    sys.meta_path.insert(0, _TritonImportHook())
