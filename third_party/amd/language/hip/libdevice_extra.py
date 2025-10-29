from triton.language import core
from triton_dist.language import core as dist_core

@core.extern
def load(ptr, semantic="monotonic", scope="agent", _semantic=None):
    """
    semantic should be one of ["monotonic", "acquire"]
    scope should be one of ["workgroup", "agent", "system]
    """
    return dist_core.extern_elementwise(
        "", "", [ptr], {
            (core.pointer_type(core.dtype("int32")), ): (f"__load_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("int32")),
            (core.pointer_type(core.dtype("int64")), ): (f"__load_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i64", core.dtype("int64")),
            (core.pointer_type(core.dtype("uint32")), ): (f"__load_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("uint32")),
            (core.pointer_type(core.dtype("uint64")), ): (f"__load_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("uint64")),
        }, is_pure=False, _semantic=_semantic)

@core.extern
def store(ptr, val, semantic="monotonic", scope="agent", _semantic=None):
    """
    semantic should be one of ["monotonic", "release"]
    scope should be one of ["workgroup", "agent", "system]
    """
    if isinstance(val, core.constexpr):
        core.static_assert(val.value == int, _semantic=_semantic)
    else:
        core.static_assert(val.dtype == core.dtype("int32") or val.dtype == core.dtype("int64") or val.dtype == core.dtype("uint32"), _semantic=_semantic)
    return dist_core.extern_elementwise(
        "", "", [ptr, core.cast(val, dtype=ptr.dtype.element_ty, _semantic=_semantic)], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32")): (f"__store_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("int32")),
            (core.pointer_type(core.dtype("uint32")), core.dtype("uint32")): (f"__store_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("uint32")),
            (core.pointer_type(core.dtype("int64")), core.dtype("int64")): (f"__store_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i64", core.dtype("int64")),
            (core.pointer_type(core.dtype("uint64")), core.dtype("uint64")): (f"__store_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i64", core.dtype("uint64")),
        }, is_pure=False, _semantic=_semantic)

@core.extern
def atomic_add(ptr, val, semantic="monotonic", scope="agent", _semantic=None):
    """
    semantic should be one of ["monotonic", "release", "acquire", "acq_rel]
    scope should be one of ["workgroup", "agent", "system]
    """
    if isinstance(val, core.constexpr):
        core.static_assert(val.value == int, _semantic=_semantic)
    else:
        core.static_assert(val.dtype == core.dtype("int32") or val.dtype == core.dtype("int64"), _semantic=_semantic)
    return dist_core.extern_elementwise(
        "", "", [ptr, core.cast(val, dtype=ptr.dtype.element_ty, _semantic=_semantic)], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32")): (f"__atomic_add_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("int32")),
            (core.pointer_type(core.dtype("uint32")), core.dtype("uint32")): (f"__atomic_add_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("uint32")),
            (core.pointer_type(core.dtype("int64")), core.dtype("int64")): (f"__atomic_add_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i64", core.dtype("int64")),
            (core.pointer_type(core.dtype("uint64")), core.dtype("uint64")): (f"__atomic_add_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}_i64", core.dtype("uint64")),
        }, is_pure=False, _semantic=_semantic)

@core.extern
def atomic_cas(ptr, val, target_val, semantic="monotonic", scope="agent", _semantic=None):
    """
    run this atomic:
    {
        old = *ptr;
        if (old == val) *ptr = target_value;
        return old;
    }

    semantic should be one of ["monotonic", "release", "acquire", "acq_rel]
    scope should be one of ["workgroup", "agent", "system]
    semantic only works when compare `old == val` success
    """
    if isinstance(val, core.constexpr):
        core.static_assert(val.value == int, _semantic=_semantic)
    else:
        core.static_assert(val.dtype == core.dtype("int32") or val.dtype == core.dtype("int64"), _semantic=_semantic)

    if isinstance(target_val, core.constexpr):
        core.static_assert(target_val.value == int, _semantic=_semantic)
    else:
        core.static_assert(target_val.dtype == core.dtype("int32") or target_val.dtype == core.dtype("int64"), _semantic=_semantic)

    return dist_core.extern_elementwise(
        "", "", [ptr, core.cast(val, dtype=ptr.dtype.element_ty, _semantic=_semantic), core.cast(target_val, dtype=ptr.dtype.element_ty, _semantic=_semantic)], {
            (core.pointer_type(core.dtype("int32")), core.dtype("int32"), core.dtype("int32")): (f"__atom_cas_{core._unwrap_if_constexpr(semantic)}_monotonic_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("int32")),
            (core.pointer_type(core.dtype("uint32")), core.dtype("uint32"), core.dtype("uint32")): (f"__atom_cas_{core._unwrap_if_constexpr(semantic)}_monotonic_{core._unwrap_if_constexpr(scope)}_i32", core.dtype("uint32")),
            (core.pointer_type(core.dtype("int64")), core.dtype("int64"), core.dtype("int64")): (f"__atom_cas_{core._unwrap_if_constexpr(semantic)}_monotonic_{core._unwrap_if_constexpr(scope)}_i64", core.dtype("int64")),
            (core.pointer_type(core.dtype("uint64")), core.dtype("uint64"), core.dtype("uint64")): (f"__atom_cas_{core._unwrap_if_constexpr(semantic)}_monotonic_{core._unwrap_if_constexpr(scope)}_i64", core.dtype("uint64")),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def sync_grid(_semantic=None):
    return dist_core.extern_elementwise(
        "", "", [], {
            tuple(): ("__ockl_grid_sync", core.dtype("int32")), # does not return
        }, is_pure=False, _semantic=_semantic
    )

@core.extern
def smid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "", "", [], {
            tuple(): ("__extra_smid", core.dtype("int32")), # does not return
        }, is_pure=True, _semantic=_semantic
    )

@core.extern
def seid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "", "", [], {
            tuple(): ("__extra_seid", core.dtype("int32")), # does not return
        }, is_pure=True, _semantic=_semantic
    )

@core.extern
def cuid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "", "", [], {
            tuple(): ("__extra_cuid", core.dtype("int32")), # does not return
        }, is_pure=True, _semantic=_semantic
    )

@core.extern
def xccid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "", "", [], {
            tuple(): ("__extra_xccid", core.dtype("int32")), # does not return
        }, is_pure=True, _semantic=_semantic
    )

@core.extern
def clock(_semantic=None):
    return dist_core.extern_elementwise(
        "", "", [], {
            tuple(): ("__extra_clock", core.dtype("uint64")), # does not return
        }, is_pure=False, _semantic=_semantic
    )

@core.extern
def wallclock(_semantic=None):
    """ strange that wallclock is not in unit of nanosecond, but 10*nanosecond. no doc found for that """
    return dist_core.extern_elementwise(
        "", "", [], {
            tuple(): ("__extra_wallclock", core.dtype("uint64")), # does not return
        }, is_pure=False, _semantic=_semantic
    )

@core.extern
def fence(semantic="monotonic", scope="agent", _semantic=None):
    return dist_core.extern_elementwise(

        "", "", [], {
            tuple(): (f"__extra_fence_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}", core.dtype("int32")), # does not return
        }, is_pure=False, _semantic=_semantic
    )
