#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

// Helper macros
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 90
#endif

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

extern "C" __global__ void __launch_bounds__(128)
matmul_kernel_tma(const __grid_constant__ CUtensorMap v1, int v2, int v3, int64_t v4, int64_t v5, const __grid_constant__ CUtensorMap v6, int v7, int v8, int64_t v9, int64_t v10, const __grid_constant__ CUtensorMap v11, int v12, int v13, int64_t v14, int64_t v15, int v16, int v17, int v18) {
    // Thread indexing
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    extern __shared__ char shared_mem[];

    float c19[128];
    #pragma unroll
    for (int _i = 0; _i < 128; _i++) c19[_i] = 0.0f;
    const int c20 = 31;
    const int c21 = 127;
    const int c22 = 1;
    const int c23 = 0;
    const int c24 = 32;
    const int c25 = 128;
    const int c26 = -1;
    const int c27 = 2;
    const int c28 = 3;
    const int pid29 = blockIdx.x;
    const int a30 = (v16 + c21);
    const int a31 = (a30 / c25);
    const int a32 = (pid29 % a31);
    const int a33 = (pid29 / a31);
    const int a34 = (a32 * c25);
    const int a35 = (a33 * c25);
    const int a36 = (v18 + c20);
    const int a37 = (a36 / c24);
    __half* smem38 = (__half*)(shared_mem + 0);
    __half* smem39 = (__half*)(shared_mem + 24576);
    int64_t* smem40 = (int64_t*)(shared_mem + 49152);
    int64_t* sv41 = (int64_t*)((char*)smem40 + c23 * 8);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(sv41)), "r"(1));
    }
    __syncthreads();
    int64_t* sv42 = (int64_t*)((char*)smem40 + c22 * 8);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(sv42)), "r"(1));
    }
    __syncthreads();
    int64_t* sv43 = (int64_t*)((char*)smem40 + c27 * 8);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(sv43)), "r"(1));
    }
    __syncthreads();
    const bool a44 = (a37 > c23);
    if (threadIdx.x == 0) {
        if (a44) {
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(sv41)), "r"(16384));
        }
    }
    __half* sv45 = (__half*)((char*)smem38 + c23 * 8192);
    // TMA: cp.async.bulk.tensor.2d global→shared
    if (threadIdx.x == 0) {
        if (a44) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                :: "r"((unsigned)__cvta_generic_to_shared(sv45)),
                   "l"((uint64_t)&v1),
                   "r"(c23), "r"(a34),
                   "r"((unsigned)__cvta_generic_to_shared(sv41))
            );
        }
    }
    __half* sv46 = (__half*)((char*)smem39 + c23 * 8192);
    // TMA: cp.async.bulk.tensor.2d global→shared
    if (threadIdx.x == 0) {
        if (a44) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                :: "r"((unsigned)__cvta_generic_to_shared(sv46)),
                   "l"((uint64_t)&v6),
                   "r"(a35), "r"(c23),
                   "r"((unsigned)__cvta_generic_to_shared(sv41))
            );
        }
    }
    const bool a47 = (a37 > c22);
    if (threadIdx.x == 0) {
        if (a47) {
            asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(sv42)), "r"(16384));
        }
    }
    __half* sv48 = (__half*)((char*)smem38 + c22 * 8192);
    // TMA: cp.async.bulk.tensor.2d global→shared
    if (threadIdx.x == 0) {
        if (a47) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                :: "r"((unsigned)__cvta_generic_to_shared(sv48)),
                   "l"((uint64_t)&v1),
                   "r"(c24), "r"(a34),
                   "r"((unsigned)__cvta_generic_to_shared(sv42))
            );
        }
    }
    __half* sv49 = (__half*)((char*)smem39 + c22 * 8192);
    // TMA: cp.async.bulk.tensor.2d global→shared
    if (threadIdx.x == 0) {
        if (a47) {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                :: "r"((unsigned)__cvta_generic_to_shared(sv49)),
                   "l"((uint64_t)&v6),
                   "r"(a35), "r"(c24),
                   "r"((unsigned)__cvta_generic_to_shared(sv42))
            );
        }
    }
    float iter51[128];
    for (int _t = 0; _t < 128; _t++) iter51[_t] = c19[_t];
    int iter52 = c22;
    int iter53 = c26;
    int iter54 = c23;
    for (int iv50 = c23; iv50 < a37; iv50 += c22) {
        const int a55 = (a37 - c27);
        const bool a56 = (iv50 < a55);
        const int a57 = (iter53 + c22);
        const bool a58 = (a57 >= c28);
        const int a59 = (a58 ? c23 : a57);
        const int a60 = (iter54 ^ c22);
        const int a61 = (a58 ? a60 : iter54);
        int64_t* sv62 = (int64_t*)((char*)smem40 + a59 * 8);
        {
            uint32_t bar_addr = (unsigned)__cvta_generic_to_shared(sv62);
            asm volatile(
                "{\n"
                ".reg .pred P1;\n"
                "WAIT_LOOP_%=:\n"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                "@!P1 bra WAIT_LOOP_%=;\n"
                "}\n"
                :: "r"(bar_addr), "r"((int)a61));
        }
        __half* sv63 = (__half*)((char*)smem39 + a59 * 8192);
        __half* sv64 = (__half*)((char*)smem38 + a59 * 8192);
        // WGMMA: m64n128k16.f32.f16.f16
        // 2 M-tiles x 2 K-tiles, 64 regs/tile
        float wgmma65[128];
        #pragma unroll
        for (int _i = 0; _i < 128; _i++) wgmma65[_i] = iter51[_i];
        {
            // Construct WGMMA shared memory descriptors
            uint32_t smem_addr_a = (unsigned)__cvta_generic_to_shared(sv64);
            uint32_t smem_addr_b = (unsigned)__cvta_generic_to_shared(sv63);

            asm volatile("wgmma.fence.sync.aligned;");

            {
                uint64_t desc_a = ((uint64_t)((smem_addr_a + 0) >> 4)) | 0x8000002002000000ULL;
                uint64_t desc_b = ((uint64_t)((smem_addr_b + 0) >> 4)) | 0x4000004001000000ULL;
                asm volatile(
                    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 1;"
                    : "+f"(wgmma65[0]), "+f"(wgmma65[1]), "+f"(wgmma65[2]), "+f"(wgmma65[3]), "+f"(wgmma65[4]), "+f"(wgmma65[5]), "+f"(wgmma65[6]), "+f"(wgmma65[7])
                      , "+f"(wgmma65[8]), "+f"(wgmma65[9]), "+f"(wgmma65[10]), "+f"(wgmma65[11]), "+f"(wgmma65[12]), "+f"(wgmma65[13]), "+f"(wgmma65[14]), "+f"(wgmma65[15])
                      , "+f"(wgmma65[16]), "+f"(wgmma65[17]), "+f"(wgmma65[18]), "+f"(wgmma65[19]), "+f"(wgmma65[20]), "+f"(wgmma65[21]), "+f"(wgmma65[22]), "+f"(wgmma65[23])
                      , "+f"(wgmma65[24]), "+f"(wgmma65[25]), "+f"(wgmma65[26]), "+f"(wgmma65[27]), "+f"(wgmma65[28]), "+f"(wgmma65[29]), "+f"(wgmma65[30]), "+f"(wgmma65[31])
                      , "+f"(wgmma65[32]), "+f"(wgmma65[33]), "+f"(wgmma65[34]), "+f"(wgmma65[35]), "+f"(wgmma65[36]), "+f"(wgmma65[37]), "+f"(wgmma65[38]), "+f"(wgmma65[39])
                      , "+f"(wgmma65[40]), "+f"(wgmma65[41]), "+f"(wgmma65[42]), "+f"(wgmma65[43]), "+f"(wgmma65[44]), "+f"(wgmma65[45]), "+f"(wgmma65[46]), "+f"(wgmma65[47])
                      , "+f"(wgmma65[48]), "+f"(wgmma65[49]), "+f"(wgmma65[50]), "+f"(wgmma65[51]), "+f"(wgmma65[52]), "+f"(wgmma65[53]), "+f"(wgmma65[54]), "+f"(wgmma65[55])
                      , "+f"(wgmma65[56]), "+f"(wgmma65[57]), "+f"(wgmma65[58]), "+f"(wgmma65[59]), "+f"(wgmma65[60]), "+f"(wgmma65[61]), "+f"(wgmma65[62]), "+f"(wgmma65[63])
                    : "l"(desc_a), "l"(desc_b), "n"(1)
                );
            }
            {
                uint64_t desc_a = ((uint64_t)((smem_addr_a + 32) >> 4)) | 0x8000002002000000ULL;
                uint64_t desc_b = ((uint64_t)((smem_addr_b + 4096) >> 4)) | 0x4000004001000000ULL;
                asm volatile(
                    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 1;"
                    : "+f"(wgmma65[0]), "+f"(wgmma65[1]), "+f"(wgmma65[2]), "+f"(wgmma65[3]), "+f"(wgmma65[4]), "+f"(wgmma65[5]), "+f"(wgmma65[6]), "+f"(wgmma65[7])
                      , "+f"(wgmma65[8]), "+f"(wgmma65[9]), "+f"(wgmma65[10]), "+f"(wgmma65[11]), "+f"(wgmma65[12]), "+f"(wgmma65[13]), "+f"(wgmma65[14]), "+f"(wgmma65[15])
                      , "+f"(wgmma65[16]), "+f"(wgmma65[17]), "+f"(wgmma65[18]), "+f"(wgmma65[19]), "+f"(wgmma65[20]), "+f"(wgmma65[21]), "+f"(wgmma65[22]), "+f"(wgmma65[23])
                      , "+f"(wgmma65[24]), "+f"(wgmma65[25]), "+f"(wgmma65[26]), "+f"(wgmma65[27]), "+f"(wgmma65[28]), "+f"(wgmma65[29]), "+f"(wgmma65[30]), "+f"(wgmma65[31])
                      , "+f"(wgmma65[32]), "+f"(wgmma65[33]), "+f"(wgmma65[34]), "+f"(wgmma65[35]), "+f"(wgmma65[36]), "+f"(wgmma65[37]), "+f"(wgmma65[38]), "+f"(wgmma65[39])
                      , "+f"(wgmma65[40]), "+f"(wgmma65[41]), "+f"(wgmma65[42]), "+f"(wgmma65[43]), "+f"(wgmma65[44]), "+f"(wgmma65[45]), "+f"(wgmma65[46]), "+f"(wgmma65[47])
                      , "+f"(wgmma65[48]), "+f"(wgmma65[49]), "+f"(wgmma65[50]), "+f"(wgmma65[51]), "+f"(wgmma65[52]), "+f"(wgmma65[53]), "+f"(wgmma65[54]), "+f"(wgmma65[55])
                      , "+f"(wgmma65[56]), "+f"(wgmma65[57]), "+f"(wgmma65[58]), "+f"(wgmma65[59]), "+f"(wgmma65[60]), "+f"(wgmma65[61]), "+f"(wgmma65[62]), "+f"(wgmma65[63])
                    : "l"(desc_a), "l"(desc_b), "n"(1)
                );
            }
            {
                uint64_t desc_a = ((uint64_t)((smem_addr_a + 4096) >> 4)) | 0x8000002002000000ULL;
                uint64_t desc_b = ((uint64_t)((smem_addr_b + 0) >> 4)) | 0x4000004001000000ULL;
                asm volatile(
                    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 1;"
                    : "+f"(wgmma65[64]), "+f"(wgmma65[65]), "+f"(wgmma65[66]), "+f"(wgmma65[67]), "+f"(wgmma65[68]), "+f"(wgmma65[69]), "+f"(wgmma65[70]), "+f"(wgmma65[71])
                      , "+f"(wgmma65[72]), "+f"(wgmma65[73]), "+f"(wgmma65[74]), "+f"(wgmma65[75]), "+f"(wgmma65[76]), "+f"(wgmma65[77]), "+f"(wgmma65[78]), "+f"(wgmma65[79])
                      , "+f"(wgmma65[80]), "+f"(wgmma65[81]), "+f"(wgmma65[82]), "+f"(wgmma65[83]), "+f"(wgmma65[84]), "+f"(wgmma65[85]), "+f"(wgmma65[86]), "+f"(wgmma65[87])
                      , "+f"(wgmma65[88]), "+f"(wgmma65[89]), "+f"(wgmma65[90]), "+f"(wgmma65[91]), "+f"(wgmma65[92]), "+f"(wgmma65[93]), "+f"(wgmma65[94]), "+f"(wgmma65[95])
                      , "+f"(wgmma65[96]), "+f"(wgmma65[97]), "+f"(wgmma65[98]), "+f"(wgmma65[99]), "+f"(wgmma65[100]), "+f"(wgmma65[101]), "+f"(wgmma65[102]), "+f"(wgmma65[103])
                      , "+f"(wgmma65[104]), "+f"(wgmma65[105]), "+f"(wgmma65[106]), "+f"(wgmma65[107]), "+f"(wgmma65[108]), "+f"(wgmma65[109]), "+f"(wgmma65[110]), "+f"(wgmma65[111])
                      , "+f"(wgmma65[112]), "+f"(wgmma65[113]), "+f"(wgmma65[114]), "+f"(wgmma65[115]), "+f"(wgmma65[116]), "+f"(wgmma65[117]), "+f"(wgmma65[118]), "+f"(wgmma65[119])
                      , "+f"(wgmma65[120]), "+f"(wgmma65[121]), "+f"(wgmma65[122]), "+f"(wgmma65[123]), "+f"(wgmma65[124]), "+f"(wgmma65[125]), "+f"(wgmma65[126]), "+f"(wgmma65[127])
                    : "l"(desc_a), "l"(desc_b), "n"(1)
                );
            }
            {
                uint64_t desc_a = ((uint64_t)((smem_addr_a + 4128) >> 4)) | 0x8000002002000000ULL;
                uint64_t desc_b = ((uint64_t)((smem_addr_b + 4096) >> 4)) | 0x4000004001000000ULL;
                asm volatile(
                    "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 1;"
                    : "+f"(wgmma65[64]), "+f"(wgmma65[65]), "+f"(wgmma65[66]), "+f"(wgmma65[67]), "+f"(wgmma65[68]), "+f"(wgmma65[69]), "+f"(wgmma65[70]), "+f"(wgmma65[71])
                      , "+f"(wgmma65[72]), "+f"(wgmma65[73]), "+f"(wgmma65[74]), "+f"(wgmma65[75]), "+f"(wgmma65[76]), "+f"(wgmma65[77]), "+f"(wgmma65[78]), "+f"(wgmma65[79])
                      , "+f"(wgmma65[80]), "+f"(wgmma65[81]), "+f"(wgmma65[82]), "+f"(wgmma65[83]), "+f"(wgmma65[84]), "+f"(wgmma65[85]), "+f"(wgmma65[86]), "+f"(wgmma65[87])
                      , "+f"(wgmma65[88]), "+f"(wgmma65[89]), "+f"(wgmma65[90]), "+f"(wgmma65[91]), "+f"(wgmma65[92]), "+f"(wgmma65[93]), "+f"(wgmma65[94]), "+f"(wgmma65[95])
                      , "+f"(wgmma65[96]), "+f"(wgmma65[97]), "+f"(wgmma65[98]), "+f"(wgmma65[99]), "+f"(wgmma65[100]), "+f"(wgmma65[101]), "+f"(wgmma65[102]), "+f"(wgmma65[103])
                      , "+f"(wgmma65[104]), "+f"(wgmma65[105]), "+f"(wgmma65[106]), "+f"(wgmma65[107]), "+f"(wgmma65[108]), "+f"(wgmma65[109]), "+f"(wgmma65[110]), "+f"(wgmma65[111])
                      , "+f"(wgmma65[112]), "+f"(wgmma65[113]), "+f"(wgmma65[114]), "+f"(wgmma65[115]), "+f"(wgmma65[116]), "+f"(wgmma65[117]), "+f"(wgmma65[118]), "+f"(wgmma65[119])
                      , "+f"(wgmma65[120]), "+f"(wgmma65[121]), "+f"(wgmma65[122]), "+f"(wgmma65[123]), "+f"(wgmma65[124]), "+f"(wgmma65[125]), "+f"(wgmma65[126]), "+f"(wgmma65[127])
                    : "l"(desc_a), "l"(desc_b), "n"(1)
                );
            }
            asm volatile("wgmma.commit_group.sync.aligned;");
        }
        asm volatile("wgmma.wait_group.sync.aligned 1;");
        const int a66 = (iter52 + c22);
        const bool a67 = (a66 >= c28);
        const int a68 = (a67 ? c23 : a66);
        const int a69 = (iv50 + c27);
        const int a70 = (a69 * c24);
        int64_t* sv71 = (int64_t*)((char*)smem40 + a68 * 8);
        if (threadIdx.x == 0) {
            if (a56) {
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(sv71)), "r"(16384));
            }
        }
        __half* sv72 = (__half*)((char*)smem38 + a68 * 8192);
        // TMA: cp.async.bulk.tensor.2d global→shared
        if (threadIdx.x == 0) {
            if (a56) {
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                    :: "r"((unsigned)__cvta_generic_to_shared(sv72)),
                       "l"((uint64_t)&v1),
                       "r"(a70), "r"(a34),
                       "r"((unsigned)__cvta_generic_to_shared(sv71))
                );
            }
        }
        __half* sv73 = (__half*)((char*)smem39 + a68 * 8192);
        // TMA: cp.async.bulk.tensor.2d global→shared
        if (threadIdx.x == 0) {
            if (a56) {
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
                    :: "r"((unsigned)__cvta_generic_to_shared(sv73)),
                       "l"((uint64_t)&v6),
                       "r"(a35), "r"(a70),
                       "r"((unsigned)__cvta_generic_to_shared(sv71))
                );
            }
        }
        #pragma unroll
        for (int _t = 0; _t < 128; _t++) iter51[_t] = wgmma65[_t];
        iter52 = a68;
        iter53 = a59;
        iter54 = a61;
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;");
    if (threadIdx.x == 0)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"((unsigned)__cvta_generic_to_shared(sv41)));
    if (threadIdx.x == 0)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"((unsigned)__cvta_generic_to_shared(sv42)));
    if (threadIdx.x == 0)
        asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"((unsigned)__cvta_generic_to_shared(sv43)));
    // local_dealloc (no-op in CUDA)
    // local_dealloc (no-op in CUDA)
    // local_dealloc (no-op in CUDA)
    __half a74[128];
    #pragma unroll
    for (int _i = 0; _i < 128; _i++)
        a74[_i] = ((__half)iter51[_i]);
    __half* smem75 = (__half*)(shared_mem + 49280);
    // Store to shared memory for WGMMA
    #pragma unroll
    for (int _i = 0; _i < 128; _i++)
        smem75[tid * 128 + _i] = a74[_i];
    asm volatile("fence.proxy.async.shared::cta;");
    // TMA: cp.async.bulk.tensor.2d shared→global + commit
    if (threadIdx.x == 0) {
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];\n"
            :: "l"((uint64_t)&v11),
               "r"(a35), "r"(a34),
               "r"((unsigned)__cvta_generic_to_shared(smem75))
        );
        asm volatile("cp.async.bulk.commit_group;");
    }
    asm volatile("cp.async.bulk.wait_group.read 0;");
    return;
}