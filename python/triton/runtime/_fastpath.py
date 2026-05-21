"""Host-overhead reduction helpers for Triton's launch fast path.

This module collects the *semantically-identical* optimisations that
Triton's launch path benefits from:

* Memoised ``str(options)`` for the kernel-cache key
  (``dict_repr``).  The launch-options dict reappears unchanged on
  every warm launch and ``str(d)`` is ~1 μs per call.
* Skipping ``launch_metadata`` construction when no enter/exit hook is
  registered (the call returns ``None`` anyway, but constructing the
  ``*bound_args.values()`` tuple is ~3 μs per launch).
* Routing through the optional C ``BinderState`` (``_binder_c``) that
  re-implements the per-launch argument binder in C.  This is the main
  contributor to the reduction in 1 ms GIL-induced spikes during
  multi-thread workloads.

None of these change the observable behaviour or numerical output of
any Triton kernel; they only reduce Python-side host overhead.

The flags are read **once at import time** so the launch hot path has
no per-call environment lookup.  All ``TRITON_FAST_LEGACY_*`` env vars
are documented escape hatches that revert one specific optimisation
back to the upstream behaviour, useful only for debugging / bisection.
"""

from __future__ import annotations

import functools
import os

# Optional C-extension accelerator: the whole ``dynamic_func`` re-implemented
# in C (per-kernel ``BinderState`` object).  Built automatically by
# ``setup.py`` via the ``triton.runtime._binder_c`` Extension declaration;
# ``build_binder_c.py`` is kept as a dev convenience for fast iteration.
# Missing -> ``JITFunction.run`` silently falls back to the exec'd Python
# ``dynamic_func``.
try:
    from . import _binder_c as _bc  # type: ignore[attr-defined]
    HAS_BINDER_C = True
except ImportError:  # pragma: no cover - optional accelerator
    _bc = None
    HAS_BINDER_C = False


def _flag(name: str) -> bool:
    v = os.environ.get(name, "")
    return v.lower() in ("1", "true", "yes", "on")


def _flag_default_on(name: str) -> bool:
    """Read an env flag whose default is *on* (i.e. the optimisation is
    active unless the user explicitly disables it with ``0``/``false``)."""
    v = os.environ.get(name, "").lower()
    if v == "":
        return True
    return v in ("1", "true", "yes", "on")


# ----- Master switch --------------------------------------------------------

# When ``TRITON_HOST_OPT=0`` (or ``false`` / ``no`` / ``off``) the
# host-overhead optimisations are completely disabled and
# ``JITFunction.run`` falls back to a verbatim copy of the upstream
# Triton ``run()`` method preserved as ``JITFunction._run_upstream``.
# Tracing (``TRITON_HOST_TRACE=1``) and every ``TRITON_FAST_*`` flag are
# also ignored in this mode — the runtime behaves exactly like an
# unpatched upstream Triton.
#
# Default: ``1`` (optimisations active).
HOST_OPT_ENABLED: bool = _flag_default_on("TRITON_HOST_OPT")


# ----- Always-on, semantically-identical optimisations ---------------------
#
# Each of the following can be reverted to upstream behaviour with a
# ``TRITON_FAST_LEGACY_*`` env var, but the default is *on* because none
# of them changes observable behaviour.  They are all forced off when
# the master switch ``TRITON_HOST_OPT`` is 0, so that path reproduces
# upstream Triton verbatim.

# Skip building ``launch_metadata`` when no enter/exit hook is registered.
# ``CompiledKernel.launch_metadata`` itself returns ``None`` in that case,
# but the call + ``*bound_args.values()`` unpacking still costs ~3 μs /
# launch.  When a hook is installed the method is called normally.
SKIP_LAUNCH_METADATA_WHEN_NO_HOOK: bool = HOST_OPT_ENABLED and not _flag(
    "TRITON_FAST_LEGACY_LAUNCH_METADATA")

# Use the C ``BinderState`` (whole-``dynamic_func`` replacement) instead of
# the exec'd Python function.  The C path returns identical
# ``(bound_args, specialization, options)`` tuples; uncommon argument types
# (TensorDescriptor, tuples, custom dtypes, ...) are transparently delegated
# back to the Python ``specialize_impl`` so semantics never diverge.
# Default: on when the master switch is on AND the extension is built.
# Disable with ``TRITON_FAST_LEGACY_BINDER=1`` to fall back to the exec'd
# Python ``dynamic_func``.
USE_C_BINDER: bool = (
    HOST_OPT_ENABLED and HAS_BINDER_C and not _flag("TRITON_FAST_LEGACY_BINDER")
)


def describe() -> str:
    """Human-readable summary of the active flags. Printed by the
    validation script so users can verify what's actually enabled."""
    if not HOST_OPT_ENABLED:
        return ("TRITON_HOST_OPT=0  -> JITFunction.run = _run_upstream "
                "(verbatim upstream; all TRITON_FAST_* and TRITON_HOST_TRACE "
                "ignored)")
    return (
        f"TRITON_HOST_OPT=1  -> JITFunction.run = _run_fast/_run_traced\n"
        f"  TRITON_HOST_TRACE        = {os.environ.get('TRITON_HOST_TRACE', '0')}\n"
        f"  SKIP_LAUNCH_METADATA     = {SKIP_LAUNCH_METADATA_WHEN_NO_HOOK}\n"
        f"  USE_C_BINDER             = {USE_C_BINDER} (HAS_BINDER_C={HAS_BINDER_C})"
    )


# ---- Helpers --------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _options_repr_cached(items: tuple) -> str:
    """Inner memoised stringifier keyed by a hashable items tuple."""
    return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in items) + "}"


def dict_repr(d) -> str:
    """``str(d)`` but with content memoisation, for the small launch-options
    dict that reappears every launch.

    ``d`` is the user-supplied ``**kwargs`` dict captured by the binder. Its
    contents are usually a handful of small ints / strings; producing it via
    ``str(d)`` costs ~1 μs even when the value never changes. Hashing the
    sorted items tuple costs ~100 ns, so memoisation is a net win whenever
    we get more than one launch with the same kwargs.

    Falls back to plain ``str(d)`` if any value isn't hashable.
    """
    try:
        items = tuple(d.items())
        return _options_repr_cached(items)
    except TypeError:
        return str(d)
