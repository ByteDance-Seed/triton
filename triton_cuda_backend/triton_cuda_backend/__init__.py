"""Non-invasive CUDA-C++ backend plugin for Triton.

Importing this package installs the ``add_stages_inspection_hook`` that makes
the nvidia backend emit CUDA C++ (compiled by nvcc/ptxas) instead of going
through LLVM/PTX, for sm90 targets, when ``TRITON_EMIT_CUDA=1`` is set.

It patches no Triton source file -- the only coupling is the official runtime
hook and the additive ``nvidia.translate_ttgir_to_cuda`` C++ entry point.
"""
from ._stages import register, cuda_stages_hook  # noqa: F401

register()
