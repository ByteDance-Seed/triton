// Factory declarations for the out-of-tree IR->IR transform passes that run
// BEFORE the emit-cuda lowering pass. These restructure a *simple* (non
// persistent, non epilogue-overlapped) warp-specialized GEMM in TTGIR into the
// gemm_06-class structure that the CUDA emitter then lowers verbatim.
//
// The emitter (CUDACodeGen.cpp) is a pure printer; all algorithmic structure is
// introduced here so it lives in the compiler, not in a hand-written kernel.
#ifndef TRITON_CUDA_BACKEND_PASSES_H
#define TRITON_CUDA_BACKEND_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace triton_cuda {

// P1: persistent-ize each warp_specialize region by wrapping
// {swizzle + K-loop + epilogue} in an outer grid-stride tile loop, threading
// the pipeline ring (index, phase) across tiles. Correctness-preserving for any
// launch grid; becomes persistent once the grid is shrunk to ~num_SMs.
std::unique_ptr<mlir::Pass> createPersistentTilePass();

// P2: in consumer partitions (post-P1), defer the TMA-store wait to the top of
// the next tile iteration so store(t) overlaps mainloop(t+1).
std::unique_ptr<mlir::Pass> createEpilogueOverlapPass();

// P3: in a cluster (num_ctas>1) GEMM, set the multicast attr on the TMA
// global->shared loads whose coordinates are independent of %cluster_ctarank
// (one load serves all CTAs of the CGA). No-op on non-cluster kernels.
std::unique_ptr<mlir::Pass> createMulticastPass();

// P4: inter-warpgroup ping-pong scheduling for the 2-consumer FlashAttention
// mainloop. Inserts emitcuda.named_barrier ops (bar.sync/bar.arrive on private
// scheduler barrier IDs) so the two consumer warpgroups stagger: one WG's
// softmax (CUDA cores) overlaps the other's WGMMA (tensor cores). Mirrors
// level6's warp_scheduler_barrier_sync/arrive. Only fires on a warp_specialize
// with exactly 2 consumer partitions whose mainloop has >=2 WGMMA dots.
std::unique_ptr<mlir::Pass> createWgPingpongPass();

} // namespace triton_cuda

#endif // TRITON_CUDA_BACKEND_PASSES_H
