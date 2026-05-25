# __call__ of <class 'triton.backends.nvidia.driver.CudaLauncher'>

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
