from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, nvidia
try:
    from triton._C.libtriton import distributed
except ImportError:
    distributed = None
from triton import knobs
from triton.runtime.errors import PTXASError

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


def get_ptxas() -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas


@functools.lru_cache()
def get_ptxas_version():
    mock_ver = knobs.nvidia.mock_ptx_version
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas().path, "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor

    if major >= 13:
        base_ptx = 90
        return base_ptx + (major - 13) * 10 + minor

    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        cuda_version = get_ptxas().version
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.6 is the max version supported by llvm c1188642.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(86, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):
    # TODO: Handle non-"a" sms
    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    ptx_options: str = None
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""
    emit_cuda: bool = False  # If True, emit CUDA C++ code instead of going through LLVM/PTX

    def __post_init__(self):
        # TRITON_EMIT_CUDA=1 forces CUDA emitter for all kernels
        if os.environ.get("TRITON_EMIT_CUDA", "") == "1" and not self.emit_cuda:
            object.__setattr__(self, 'emit_cuda', True)
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.nvidia.libdevice_path or str(default_libdir / 'libdevice.10.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def get_target_name(self, options) -> str:
        capability = self._parse_arch(options.arch)
        return f"cuda:{capability}"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def parse_options(self, opts) -> Any:
        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires NVIDIA SM90+ (Hopper). "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        distributed.ir.load_dialects(ctx) if distributed else None
        nvidia.load_dialects(ctx)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        if capability // 10 < 9:
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        # Set maxnreg on all kernels, if it was provided.
        if opt.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        cluster_info = nvidia.ClusterInfo()
        if opt.cluster_dims is not None:
            cluster_info.clusterDimX = opt.cluster_dims[0]
            cluster_info.clusterDimY = opt.cluster_dims[1]
            cluster_info.clusterDimZ = opt.cluster_dims[2]
        pm = ir.pass_manager(mod.context)
        dump_enabled = pm.enable_debug()
        # TritonDistributed Extension (or standard path)
        if distributed:
            distributed.passes.ttir.add_convert_to_ttgpuir_ext(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        else:
            passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        if capability // 10 >= 8:
            passes.ttgpuir.add_f32_dot_tc(pm)
        # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        passes.ttgpuir.add_accelerate_matmul(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        passes.ttir.add_loop_aware_cse(pm)
        if capability // 10 in [8, 9]:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        elif capability // 10 >= 10:
            passes.ttgpuir.add_fuse_nested_loops(pm)
            passes.common.add_canonicalizer(pm)
            passes.ttir.add_triton_licm(pm)
            passes.ttgpuir.add_optimize_accumulator_init(pm)
            passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
            nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
            passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
            passes.ttgpuir.add_schedule_loops(pm)
            passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
            passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
            passes.ttgpuir.add_combine_tensor_select_and_if(pm)
            # hoist again and allow hoisting out of if statements
            passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
            nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
        else:
            passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.ttgpuir.add_prefetch(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        passes.ttgpuir.add_coalesce_async_copy(pm)
        nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.common.add_symbol_dce(pm)
        if capability // 10 >= 9:
            nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
        nvidia.passes.ttnvgpuir.add_lower_mma(pm)
        passes.common.add_sccp(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)

        pm.run(mod)
        metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        tensordesc_meta = mod.get_tensordesc_metadata()
        metadata["tensordesc_meta"] = tensordesc_meta
        return mod

    def gluon_to_ttgir(self, src, metadata, options, capability):
        mod = src
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.gluon.add_inliner(pm)
        passes.gluon.add_resolve_auto_encodings(pm)
        passes.common.add_sccp(pm)
        passes.ttir.add_loop_aware_cse(pm)
        passes.gluon.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        pm.run(mod)
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def make_llir(self, src, metadata, options, capability):
        ptx_version = get_ptx_version_from_options(options, self.target.arch)

        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()

        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        passes.ttgpuir.add_allocate_warp_groups(pm)
        passes.convert.add_scf_to_cf(pm)
        nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
        nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        if knobs.compilation.enable_experimental_consan:
            # Call ConcurrencySanitizerPass here, before allocating global scratch memory but after allocating tensor and shared
            passes.ttgpuir.add_concurrency_sanitizer(pm)
        passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
        # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
        # TritonDistributed Extension: Distributed/SIMT Dialect -> LLVM
        if distributed:
            distributed.passes.ttgpuir.nvidia.add_convert_triton_distributed_to_llvm(pm, capability, ptx_version)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.convert.add_nvvm_to_llvm(pm)
        if not knobs.compilation.disable_line_info:
            passes.llvmir.add_di_scope(pm)
        if CUDABackend.instrumentation:
            CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        if knobs.compilation.enable_asan:
            raise RuntimeError(
                "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        llvm_mod = llvm.to_module(mod, context)
        proc = sm_arch_from_capability(capability)
        features = get_features(options, self.target.arch)
        triple = 'nvptx64-nvidia-cuda'
        nvidia.set_short_ptr()
        llvm.attach_datalayout(llvm_mod, triple, proc, features)
        nvidia.set_nvvm_reflect_ftz(llvm_mod)

        if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        # warp-specialization mutates num_warps
        total_num_warps = src.get_int_attr("ttg.total-num-warps")
        if total_num_warps is not None:
            metadata["num_warps"] = total_num_warps
        metadata["shared"] = src.get_int_attr("ttg.shared")
        metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
        metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    def make_ptx(self, src, metadata, opt, capability):
        ptx_version = get_ptx_version_from_options(opt, self.target.arch)

        triple = 'nvptx64-nvidia-cuda'
        proc = sm_arch_from_capability(capability)
        features = get_features(opt, self.target.arch)
        ret = llvm.translate_to_asm(src, triple, proc, features, [], opt.enable_fp_fusion, False)
        # Find kernel names (there should only be one)
        names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        assert len(names) == 1
        metadata["name"] = names[0]
        # post-process
        ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
        # Remove the debug flag that prevents ptxas from optimizing the code
        ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        if knobs.nvidia.dump_nvptx:
            print("// -----// NVPTX Dump //----- //")
            print(ret)
        return ret

    def make_cubin(self, src, metadata, opt, capability):
        ptxas = get_ptxas().path
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            debug_info = []
            if knobs.compilation.disable_line_info:
                # This option is ignored if used without -lineinfo
                debug_info += ["-lineinfo", "-suppress-debug-info"]
            elif knobs.nvidia.disable_ptxas_opt:
                # Synthesize complete debug info
                debug_info += ["-g"]
            else:
                # Only emit line info
                debug_info += ["-lineinfo"]

            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
            arch = sm_arch_from_capability(capability)

            # Disable ptxas optimizations if requested
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []

            # Accept more ptxas options if provided
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            ptxas_cmd = [
                ptxas, *debug_info, *fmad, '-v', *disable_opt, *ptx_extra_options, f'--gpu-name={arch}', fsrc.name,
                '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                error = (f"{error}\n"
                         f"`ptxas` stderr:\n{log}\n"
                         f'Repro command: {" ".join(ptxas_cmd)}\n')

                print(f"""

================================================================
{error}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise PTXASError(error)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def make_cuda(self, src, metadata, options, capability):
        """Translate TTGIR module to CUDA C++ source code via C++ MLIR pass."""
        # HMMA v2 (nvidia_mma versionMajor=2, small tiles e.g. block_m=16) is
        # handled by the generic LinearLayout-based emitDot/emitLocalLoad path
        # (shared-memory FMA). Validated correct for f16/bf16/fp8 across tile
        # shapes and warp counts. No guard needed.
        # Dump TTGIR for debugging
        ttgir_dump = os.environ.get("TRITON_CUDA_DUMP_TTGIR")
        if ttgir_dump:
            print("=== TTGIR (input to CUDA emitter) ===")
            print(src)
            print("=== END TTGIR ===")
        from triton._C.libtriton import nvidia
        # The emitter gates a ptxas global-stride workaround on the PTX ISA
        # version (needed for PTX <= 8.5 / CUDA <= 12.5). Pass the version that
        # the system nvcc/ptxas will target so device-side TMA descriptors are
        # encoded correctly on both old and new toolkits.
        ptx_version = get_ptx_version_from_options(options, capability)
        result = nvidia.translate_ttgir_to_cuda(
            src, capability, options.num_warps, options.num_ctas, ptx_version)
        cuda_src = result["cuda_src"]
        metadata["name"] = result["kernel_name"]
        metadata["shared"] = result["shared_mem_size"]
        # Warp-specialized kernels launch more warps than options.num_warps
        # (base producer warps + the partition/consumer warpgroups). The emitter
        # reports the true total so the launcher sizes blockDim.x = 32*num_warps
        # to cover every warp-group; without this the consumer warps never run.
        ws_num_warps = result.get("num_warps", 0)
        if ws_num_warps and ws_num_warps > options.num_warps:
            metadata["num_warps"] = ws_num_warps
        # Device-side TMA descriptors allocate per-CTA global scratch; the
        # runtime sizes the buffer as grid*num_ctas*global_scratch_size.
        if result.get("global_scratch_size", 0):
            metadata["global_scratch_size"] = result["global_scratch_size"]
            metadata["global_scratch_align"] = result.get("global_scratch_align", 1) or 1
        if os.environ.get("TRITON_CUDA_DEBUG"):
            print(f"[CUDA emitter] shared_mem_size={result['shared_mem_size']}, kernel={result['kernel_name']}")
        # Dump for debugging
        dump_path = os.environ.get("TRITON_CUDA_DUMP")
        if dump_path:
            # If dump_path is a directory (or ends with '/'), write one file per
            # kernel name so concurrent/multi-kernel compiles don't clobber each
            # other (useful for debugging specific failing kernels).
            if dump_path.endswith("/") or os.path.isdir(dump_path):
                os.makedirs(dump_path, exist_ok=True)
                out = os.path.join(dump_path, result["kernel_name"] + ".cu")
            else:
                out = dump_path
            with open(out, "w") as f:
                f.write(cuda_src)
        # Set metadata that would normally come from LLVM lowering
        for key in ("tmem_size", "global_scratch_size", "global_scratch_align",
                    "profile_scratch_size", "profile_scratch_align", "maxntid"):
            metadata.setdefault(key, 0)
        return cuda_src

    def make_ptx_from_cuda(self, src, metadata, options, capability):
        """Compile CUDA source code to PTX using nvcc -ptx."""
        arch = sm_arch_from_capability(capability)
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.cu') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            ptx_path = fsrc.name + '.ptx'

            fmad = ['--fmad=false'] if not options.enable_fp_fusion else ['--fmad=true']

            # Source-level line info: the emitter writes #line directives that
            # point back at the Triton source; -lineinfo makes nvcc/ptxas carry
            # them through to the SASS line table (mirrors make_cubin).
            lineinfo = [] if knobs.compilation.disable_line_info else ['-lineinfo']

            nvcc_cmd = [
                'nvcc', '-ptx',
                f'--gpu-architecture={arch}',
                '-O3',
                '--use_fast_math',
                # --use_fast_math implies --ftz=true, but Triton FP32 arithmetic
                # is non-FTZ (LLVM fmul -> mul.f32 keeps denormals). Explicit
                # --ftz=false overrides the implied flush while keeping the fast
                # intrinsic substitutions (exp2f -> ex2.approx etc.).
                '--ftz=false',
                *fmad,
                *lineinfo,
                '-std=c++17',
                fsrc.name,
                '-o', ptx_path,
            ]

            try:
                subprocess.run(nvcc_cmd, check=True, close_fds=False,
                             stdout=flog, stderr=subprocess.STDOUT)
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                # Also read the CUDA source for debugging
                cuda_src_debug = src[:2000] if len(src) > 2000 else src
                if os.path.exists(flog.name):
                    os.remove(flog.name)
                raise RuntimeError(
                    f"`nvcc` failed with error code {e.returncode}\n"
                    f"`nvcc` command: {' '.join(nvcc_cmd)}\n"
                    f"`nvcc` log:\n{log}\n"
                    f"CUDA source (first 2000 chars):\n{cuda_src_debug}\n"
                )

            with open(ptx_path, 'r') as f:
                ptx = f.read()
            if os.path.exists(ptx_path):
                os.remove(ptx_path)
        names = re.findall(r"\.visible \.entry ([a-zA-Z_][a-zA-Z0-9_]*)", ptx)
        assert len(names) == 1
        metadata["name"] = names[0]
        return ptx

    def make_cubin_from_ptx(self, src, metadata, options, capability):
        """Assemble nvcc-generated PTX to cubin with the system ptxas.

        Uses the CUDA-toolkit ptxas from PATH (same toolkit as nvcc) rather
        than the Triton-bundled one, so the PTX ISA version nvcc emits is
        always understood.
        """
        arch = sm_arch_from_capability(capability)
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            cubin_path = fsrc.name + '.cubin'

            fmad = ['--fmad=false'] if not options.enable_fp_fusion else []
            lineinfo = [] if knobs.compilation.disable_line_info else ['-lineinfo']

            # Warp-specialized kernels use setmaxnreg.dec/inc to redistribute the
            # SM register file between the producer and consumer warpgroups. Those
            # instructions are only valid if the kernel's per-thread register
            # baseline equals the value the emitter assumed when computing the
            # dec/inc amounts (upstream sets nvvm.maxnreg = 65536/totalThreads&~7).
            # If the kernel already carries a .maxnreg directive (emitted via
            # __maxnreg__ from ttg.maxnreg), that takes care of it; otherwise
            # pin the baseline via --maxrregcount, or ptxas picks an arbitrary
            # baseline and setmaxnreg faults at launch.
            maxreg = ['--maxrregcount=255']
            if 'setmaxnreg' in src and '.maxnreg' not in src:
                total_threads = metadata.get('num_warps', options.num_warps) * 32
                baseline = (65536 // total_threads) & ~7
                maxreg = [f'--maxrregcount={baseline}']

            ptxas_cmd = [
                'ptxas',
                f'--gpu-name={arch}',
                '-O3',
                *maxreg,
                *fmad,
                *lineinfo,
                fsrc.name,
                '-o', cubin_path,
            ]

            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False,
                             stdout=flog, stderr=subprocess.STDOUT)
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)
                raise RuntimeError(
                    f"`ptxas` failed with error code {e.returncode}\n"
                    f"`ptxas` command: {' '.join(ptxas_cmd)}\n"
                    f"`ptxas` log:\n{log}\n"
                )

            with open(cubin_path, 'rb') as f:
                cubin = f.read()
            if os.path.exists(cubin_path):
                os.remove(cubin_path)
        return cubin

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        # The CUDA emitter targets sm90a (WGMMA/TMA/mbarrier). Other
        # architectures (e.g. sm100 tcgen05/tmem) are out of its scope, so an
        # explicit non-sm90 target compiles through the standard path even
        # when TRITON_EMIT_CUDA=1 is set globally.
        if options.emit_cuda and capability // 10 == 9:
            # CUDA emitter path: TTGIR -> CUDA -> PTX (nvcc) -> cubin (ptxas)
            stages["cuda"] = lambda src, metadata: self.make_cuda(src, metadata, options, capability)
            stages["ptx"] = lambda src, metadata: self.make_ptx_from_cuda(src, metadata, options, capability)
            stages["cubin"] = lambda src, metadata: self.make_cubin_from_ptx(src, metadata, options, capability)
        else:
            # Standard path: TTGIR -> LLVM-IR -> PTX -> cubin
            stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
            stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, self.target.arch)
            stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, self.target.arch)

    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version()
        return f'{version}-{self.target.arch}'
