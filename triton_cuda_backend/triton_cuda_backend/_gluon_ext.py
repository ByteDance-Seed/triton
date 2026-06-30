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


_im2col_launcher_installed = False


def inject_im2col_launcher():
    """Make the launcher build a HW im2col CUtensorMap for TensorDescriptors tagged
    with ``_im2col_params`` (set by the conv host glue). This is the host half of
    the backend TMA-im2col support (the device half is Im2colRewritePass + the
    emitter). Non-invasive: wraps the nvidia driver's ``make_tensordesc_arg`` and
    reuses Triton's own ``fill_tma_descriptor_im2col`` (cuTensorMapEncodeIm2col),
    so untagged descriptors are byte-identical to before. Idempotent.

    _im2col_params = (N,D,H,W,Cin, kT,kH,kW, pT,pH,pW, sT,sH,sW, BM,BK): the conv
    geometry. The descriptor's globalDim is {C,W,H,D,N} (the driver reverses the
    NDHWC shape we pass), corners encode the padding window, channelsPerPixel=BK,
    pixelsPerColumn=BM, elementStrides encode the stride. See SKILL.md."""
    global _im2col_launcher_installed
    if _im2col_launcher_installed:
        return
    try:
        from triton.backends.nvidia import driver as _drv
    except Exception:
        return
    import triton as _triton
    _orig = _drv.make_tensordesc_arg

    def _make(arg, metadata):
        p = getattr(arg, "_im2col_params", None)
        if p is None:
            return _orig(arg, metadata)
        N, D, H, W, Cin, kT, kH, kW, pT, pH, pW, sT, sH, sW, BM, BK = p
        util = _triton.runtime.driver.active.utils
        cu = util.fill_tma_descriptor_im2col(
            arg.base.data_ptr(), metadata["swizzle"], metadata["elem_size"],
            _drv.TMA_DTYPE_DEVICE_TO_HOST[metadata["elem_type"]],
            [BM, BK], [N, D, H, W, Cin],
            [D * H * W * Cin, H * W * Cin, W * Cin, Cin, 1], 0,
            [-pT, -pH, -pW], [pT - (kT - 1), pH - (kH - 1), pW - (kW - 1)],
            BK, BM, [1, sT, sH, sW, 1])
        return [cu, *arg.shape, *arg.strides]

    _drv.make_tensordesc_arg = _make
    _im2col_launcher_installed = True
