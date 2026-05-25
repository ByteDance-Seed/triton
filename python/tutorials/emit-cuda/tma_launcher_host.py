# Launcher class: <class 'triton.backends.nvidia.driver.CudaLauncher'>
# Source file: /usr/local/lib/python3.11/dist-packages/triton/backends/nvidia/driver.py

class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        tensordesc_meta = getattr(metadata, "tensordesc_meta", None)
        src = make_launcher(constants, signature, tensordesc_meta)
        mod = compile_module_from_src(
            src=src,
            name="__triton_launcher",
            library_dirs=library_dirs(),
            include_dirs=include_dirs,
            libraries=libraries,
        )

        self.num_ctas = functools.reduce(operator.mul, metadata.cluster_dims, 1)
        self.launch = wrap_handle_tensordesc(mod.launch, signature, tensordesc_meta)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.profile_scratch_size = metadata.profile_scratch_size
        self.profile_scratch_align = metadata.profile_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid
        self.launch_pdl = metadata.launch_pdl

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):

        def allocate_scratch(size, align, allocator):
            if size > 0:
                grid_size = gridX * gridY * gridZ
                alloc_size = grid_size * self.num_ctas * size
                alloc_fn = allocator.get()
                return alloc_fn(alloc_size, align, stream)
            return None

        global_scratch = allocate_scratch(self.global_scratch_size, self.global_scratch_align, _allocation._allocator)
        profile_scratch = allocate_scratch(self.profile_scratch_size, self.profile_scratch_align,
                                           _allocation._profile_allocator)
        self.launch(gridX, gridY, gridZ, stream, function, self.launch_cooperative_grid, self.launch_pdl,
                    global_scratch, profile_scratch, *args)
