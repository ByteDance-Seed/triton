"""Minimal-overhead host-side tracer for Triton's launch / compile pipeline.

This module is intentionally tiny, dependency-free, and zero-cost when disabled.

Activation
----------
Set ``TRITON_HOST_TRACE=1`` to enable the tracer. The output file may be selected via
``TRITON_HOST_TRACE_FILE`` (defaults to ``./triton_host_trace.json``). The file is written
in the Chrome Trace Format and can be opened directly in https://ui.perfetto.dev .

Programmatic control:

    from triton.runtime import host_trace
    host_trace.enable("/tmp/foo.json")     # or
    host_trace.enable()                    # uses env-configured path
    ... run kernels ...
    host_trace.flush()                     # optional, also runs at process exit

API
---
``trace_scope(name, cat="triton")`` — context manager; ``ENABLED`` is checked once,
so the disabled cost is one attribute load + one branch.

``traced(name=None, cat="triton")`` — decorator equivalent of ``trace_scope``.

``instant(name, cat="triton")`` — record a single instantaneous point.

Design notes
------------
- Events are stored in a flat ``list`` of tuples and serialised lazily, so the cost
  per event is one ``perf_counter_ns()`` + one ``list.append`` (~150-300 ns on a fast box).
- We do *not* call any thread/PID lookup inline; we cache them per-thread.
- When disabled, ``trace_scope`` returns a singleton no-op context object whose
  ``__enter__`` / ``__exit__`` are noop methods so the JIT can specialise it well.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import List, Optional, Tuple

__all__ = [
    "ENABLED",
    "trace_scope",
    "traced",
    "instant",
    "enable",
    "disable",
    "flush",
    "is_enabled",
]


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "")
    return v.lower() in ("1", "true", "yes", "on")


# Resolved once at import time. Programmatic `enable()` / `disable()` may flip it later.
#
# Note: tracing is force-disabled when ``TRITON_HOST_OPT=0`` so the
# upstream-verbatim mode does not pay for any instrumentation (the
# scopes embedded in ``compiler.py`` / ``autotuner.py`` would otherwise
# still call ``__enter__`` / ``__exit__`` on a no-op object).
def _host_opt_disabled() -> bool:
    v = os.environ.get("TRITON_HOST_OPT", "").lower()
    return v in ("0", "false", "no", "off")


ENABLED: bool = (not _host_opt_disabled()) and _env_truthy("TRITON_HOST_TRACE")
_OUTPUT_FILE: str = os.environ.get("TRITON_HOST_TRACE_FILE", "triton_host_trace.json")

# Event tuple format: (name, cat, ts_us, dur_us, pid, tid, args_or_None)
_Event = Tuple[str, str, int, int, int, int, Optional[dict]]
_EVENTS: List[_Event] = []
_LOCK = threading.Lock()
_FLUSHED = False

_PID = os.getpid()
_TID_CACHE = threading.local()


def _get_tid() -> int:
    tid = getattr(_TID_CACHE, "tid", None)
    if tid is None:
        tid = threading.get_ident()
        _TID_CACHE.tid = tid
    return tid


class _NullScope:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


_NULL_SCOPE = _NullScope()


class _ActiveScope:
    """Live tracing scope. Records duration on exit."""

    __slots__ = ("_name", "_cat", "_start", "_args")

    def __init__(self, name: str, cat: str, args: Optional[dict] = None):
        self._name = name
        self._cat = cat
        self._args = args
        self._start = 0  # set in __enter__

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter_ns()
        dur_us = (end - self._start) // 1000
        ts_us = self._start // 1000
        _EVENTS.append(
            (self._name, self._cat, ts_us, dur_us, _PID, _get_tid(), self._args))
        return False


def trace_scope(name: str, cat: str = "triton", args: Optional[dict] = None):
    """Return a context manager that records a duration event when enabled."""
    if not ENABLED:
        return _NULL_SCOPE
    return _ActiveScope(name, cat, args)


def traced(name: Optional[str] = None, cat: str = "triton"):
    """Decorator equivalent of ``trace_scope``."""

    def deco(fn):
        # When disabled, return the original function unwrapped to keep zero overhead.
        if not ENABLED:
            return fn
        ev_name = name or f"{fn.__module__}.{fn.__qualname__}"

        def wrapper(*args, **kwargs):
            with _ActiveScope(ev_name, cat):
                return fn(*args, **kwargs)

        wrapper.__wrapped__ = fn
        wrapper.__name__ = getattr(fn, "__name__", ev_name)
        wrapper.__qualname__ = getattr(fn, "__qualname__", ev_name)
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return deco


def instant(name: str, cat: str = "triton", args: Optional[dict] = None) -> None:
    """Record an instantaneous event (Chrome Trace ``ph="i"``).

    For simplicity we just emit a zero-duration ``X`` event; Perfetto renders it as a
    thin marker which is the desired visual.
    """
    if not ENABLED:
        return
    ts_us = time.perf_counter_ns() // 1000
    _EVENTS.append((name, cat, ts_us, 0, _PID, _get_tid(), args))


def _flush_locked(path: str) -> None:
    global _FLUSHED
    events_json = []
    for name, cat, ts, dur, pid, tid, args in _EVENTS:
        ev = {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": pid,
            "tid": tid,
        }
        if args:
            ev["args"] = args
        events_json.append(ev)
    payload = {
        "traceEvents": events_json,
        "displayTimeUnit": "ns",
        "otherData": {
            "triton_pid": _PID,
            "triton_host_trace": "1",
        },
    }
    # Best-effort directory creation.
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    _FLUSHED = True


def flush(path: Optional[str] = None) -> str:
    """Write all collected events to ``path`` (default: configured file)."""
    target = path or _OUTPUT_FILE
    with _LOCK:
        _flush_locked(target)
    return target


def enable(path: Optional[str] = None) -> None:
    """Programmatically enable tracing.

    Note: if tracing was disabled at import time the ``@traced`` decorators that already
    ran have been resolved to the identity function and will NOT begin recording.
    Use the env var (``TRITON_HOST_TRACE=1``) for full coverage.
    """
    global ENABLED, _OUTPUT_FILE, _FLUSHED
    ENABLED = True
    _FLUSHED = False
    if path is not None:
        _OUTPUT_FILE = path


def disable() -> None:
    global ENABLED
    ENABLED = False


def is_enabled() -> bool:
    return ENABLED


def _atexit_flush() -> None:
    if not ENABLED or _FLUSHED or not _EVENTS:
        return
    try:
        _flush_locked(_OUTPUT_FILE)
    except Exception:
        # Never let the tracer take down the interpreter at exit.
        pass


atexit.register(_atexit_flush)
