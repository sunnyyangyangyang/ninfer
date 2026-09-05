#include "ops/common/act_quant_i8.cuh"

#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

namespace {

constexpr int kActGroupK = 64;

static __global__ void act_quant_i8_kernel(const __nv_bfloat16* __restrict__ x,
                                    std::int8_t* __restrict__ x_q,
                                    __half* __restrict__ x_scale, std::int32_t k,
                                    std::int32_t t, std::int32_t kg) {
    const int warp_id = static_cast<int>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane    = static_cast<int>(threadIdx.x) & 31;
    if (warp_id >= t * kg) { return; }
    const int tt = warp_id / kg;
    const int gg = warp_id % kg;
    const __nv_bfloat16* row = x + static_cast<std::int64_t>(tt) * k + gg * kActGroupK;

    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float v = __bfloat162float(row[lane * 2 + i]);
        m             = fmaxf(m, fabsf(v));
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d)); }

    const bool   active     = m > 0.0f;
    const float  scale      = active ? m / 127.0f : 1.0f;
    const __half half_scale = __float2half_rn(scale);
    x_scale[static_cast<std::int64_t>(tt) * kg + gg] = half_scale;
    const float inv = active ? (127.0f / m) : 1.0f;

    std::int8_t* out_row = x_q + static_cast<std::int64_t>(tt) * k + gg * kActGroupK;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int off  = lane * 2 + i;
        const int code = active ? __float2int_rn(__bfloat162float(row[off]) * inv) : 0;
        out_row[off]   = static_cast<std::int8_t>(code);
    }
}

} // namespace

void act_quant_i8_launch(const __nv_bfloat16* x, std::int8_t* x_q, __half* x_scale,
                         std::int32_t k, std::int32_t t, cudaStream_t stream) {
    const std::int32_t kg    = k / kActGroupK;
    const std::int32_t pairs = t * kg;
    const std::int32_t block = 256;
    const std::int32_t grid  = div_up(pairs, block / 32);
    act_quant_i8_kernel<<<grid, block, 0, stream>>>(x, x_q, x_scale, k, t, kg);
}

void* act_quant_i8_scratch(std::int64_t bytes, cudaStream_t stream) {
    (void)stream;
    static void* buf   = nullptr;
    static std::int64_t cap = 0;
    if (bytes > cap) {
        if (buf != nullptr) { cudaFree(buf); }
        void* grown = nullptr;
        cudaMalloc(&grown, static_cast<std::size_t>(bytes));
        if (grown == nullptr) { throw std::bad_alloc(); }
        buf = grown;
        cap = bytes;
    }
    return buf;
}

} // namespace ninfer::ops::detail
