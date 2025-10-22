################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import triton
import triton.language as tl
from triton.language import core


def _str_to_gpu_shfl_mode(mode_str):
    # The order of shfl modes is from (llvm-project/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td)
    ALL_SHFL_MODES = ["xor", "up", "down", "idx"]

    if mode_str not in ALL_SHFL_MODES:
        raise RuntimeError(f"unexpected gpu shuffle mode, expecte: {ALL_SHFL_MODES}, but got: {mode_str}")

    return ALL_SHFL_MODES.index(mode_str)


@core.extern
def laneid(_semantic=None):
    return core.tensor(_semantic.builder.create_laneid(), core.int32)


@core.extern
def tid(axis, _semantic=None):
    assert axis <= 2 and axis >= 0
    axis_to_xyz = ["x", "y", "z"]
    calleeName = f"llvm.amdgcn.workitem.id.{axis_to_xyz[axis]}"
    return core.extern_elementwise("", "", [], {
        (): (calleeName, core.dtype("int32")),
    }, is_pure=True, _semantic=_semantic)


@core.extern
def __syncthreads(_semantic=None):
    return core.tensor(_semantic.builder.create_barrier(), core.void)


@core.extern
def __shfl_sync_with_mode_i32(
    value,
    offset,
    mode: core.constexpr = "up",
    width: int = 64,
    _semantic=None,
):
    shfl_mode = _str_to_gpu_shfl_mode(mode.value)
    return core.tensor(_semantic.builder.create_warp_shuffle(value, offset, width, shfl_mode), value.dtype)


@triton.jit
def __shfl_sync_i32(value, laneid):
    return __shfl_sync_with_mode_i32(value, laneid, "idx", 64)


@triton.jit
def __shfl_up_sync_i32(value, offset):
    return __shfl_sync_with_mode_i32(value, offset, "up", 64)


@triton.jit
def __shfl_down_sync_i32(value, offset):
    return __shfl_sync_with_mode_i32(value, offset, "down", 64)


@triton.jit
def __shfl_xor_sync_i32(value, offset):
    return __shfl_sync_with_mode_i32(value, offset, "xor", 64)
