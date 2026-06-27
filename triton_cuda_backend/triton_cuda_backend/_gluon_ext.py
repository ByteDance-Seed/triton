"""Non-invasive Gluon extensions injected by the emit_cuda plugin.

These add Python-only builtins onto upstream Gluon modules at import time, so the
Triton source tree stays pristine (no diff).  Everything here wraps *pristine*
libtriton bindings — e.g. ``GluonOpBuilder.create_async_tma_reduce`` (gluon_ir.cc)
and ``ir.DESCRIPTOR_REDUCE_KIND`` (ir.cc), both upstream — so no libtriton rebuild
is required either.
"""

_REDUCE_KIND_MAP = {
    "add": "ADD", "and": "AND", "or": "OR", "xor": "XOR",
    "max": "MAX", "min": "MIN", "inc": "INC", "dec": "DEC",
}


def inject_async_reduce_shared_to_global():
    """Add ``tma.async_reduce_shared_to_global`` (TMA bulk reduce, no auto-wait).

    Idempotent; a no-op if the symbol already exists (e.g. a tree that still
    carries the in-source version).  Used by the tutorial-03 backward dQ reduce.
    """
    from triton.experimental.gluon.language.nvidia.hopper import tma as _tma
    if hasattr(_tma, "async_reduce_shared_to_global"):
        return
    from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

    @builtin
    def async_reduce_shared_to_global(tensor_desc, coord, src, kind="add", _semantic=None):
        """Asynchronously reduce a shared-memory tile into global memory via a TMA
        bulk reduce (cp.reduce.async.bulk.tensor...). Unlike ``tensor_descriptor.
        atomic_add`` (which lowers to DescriptorReduceOp and gets a forced
        synchronous store-wait), this creates AsyncTMAReduceOp directly: NO
        auto-wait is inserted, so the reduce overlaps subsequent compute and must
        be drained explicitly with ``store_wait(pendings)``. The required
        async-proxy fence is inserted automatically by ProxyFenceInsertion."""
        if _semantic.builder.options.enable_iisan:
            _tma._emit_alignment_check(tensor_desc, coord, "async_reduce_shared_to_global",
                                       "innermost coordinate", _semantic=_semantic)
        kind = _unwrap_if_constexpr(kind)
        if isinstance(kind, str):
            from triton._C.libtriton import ir as _ir
            kind = getattr(_ir.DESCRIPTOR_REDUCE_KIND, _REDUCE_KIND_MAP[kind.lower()])
        coord = _semantic._convert_to_ir_values(coord, require_i64=False)
        _semantic.builder.create_async_tma_reduce(kind, tensor_desc.handle, coord, src.handle)

    _tma.async_reduce_shared_to_global = async_reduce_shared_to_global
    try:
        if "async_reduce_shared_to_global" not in _tma.__all__:
            _tma.__all__.append("async_reduce_shared_to_global")
    except (AttributeError, TypeError):
        pass
