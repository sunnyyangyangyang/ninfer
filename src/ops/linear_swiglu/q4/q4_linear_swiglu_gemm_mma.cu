#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma_i8.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using GateUpC40Cfg  = GemmCfg<64, 40, 64, 64, 8, 2, 1, false, true, true>;
using GateUpC48Cfg  = GemmCfg<64, 48, 64, 64, 8, 2, 1, false, true, true>;
using GateUpC128Cfg = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;

template <class Cfg, bool Full>
void launch_folded(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = Cfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, Cfg::BN)));
    if constexpr (Full) {
        q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, true><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    } else {
        q4_linear_swiglu_mma_split_half_pair_kernel<Cfg, false><<<grid, Cfg::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], t, weight.padded_shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Cfg, bool Full>
void launch_folded_i8(const Tensor& x_q, const Tensor& x_scale, const Weight& weight, Tensor& out,
                      cudaStream_t stream) {
    constexpr int PM = Cfg::BM / 2;
    const int t      = x_q.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, Cfg::BN)));
    q4_linear_swiglu_mma_split_half_pair_i8_kernel<Cfg, Full>
        <<<grid, Cfg::THREADS, 0, stream>>>(static_cast<const std::int8_t*>(x_q.data),
                                            static_cast<const std::uint16_t*>(x_scale.data),
                                            static_cast<const std::uint8_t*>(weight.qdata),
                                            static_cast<const std::uint8_t*>(weight.scales),
                                            static_cast<__nv_bfloat16*>(out.data), out.ne[0],
                                            x_q.ne[0], t, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Cfg>
void launch_route_i8(const Tensor& x_q, const Tensor& x_scale, const Weight& weight, Tensor& out,
                     cudaStream_t stream) {
    const bool full = (x_q.ne[1] % Cfg::BN) == 0;
    for_each_token_slice(x_q.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor xq_slice  = x_q.slice(1, offset, count);
        const Tensor xs_slice  = x_scale.slice(1, offset, count);
        Tensor out_slice       = out.slice(1, offset, count);
        if (full) {
            launch_folded_i8<Cfg, true>(xq_slice, xs_slice, weight, out_slice, stream);
        } else {
            launch_folded_i8<Cfg, false>(xq_slice, xs_slice, weight, out_slice, stream);
        }
    });
}

template <class Cfg>
void launch_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (x.ne[1] % Cfg::BN) == 0;
    for_each_token_slice(x.ne[1], Cfg::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_folded<Cfg, true>(x_slice, weight, out_slice, stream);
        } else {
            launch_folded<Cfg, false>(x_slice, weight, out_slice, stream);
        }
    });
}

} // namespace

void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& weight,
                                                          Tensor& out, cudaStream_t stream) {
    launch_route<GateUpC128Cfg>(x, weight, out, stream);
}

void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& weight,
                                                         Tensor& out, cudaStream_t stream) {
    launch_route<GateUpC40Cfg>(x, weight, out, stream);
}

void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& weight,
                                                         Tensor& out, cudaStream_t stream) {
    launch_route<GateUpC48Cfg>(x, weight, out, stream);
}



void q4_linear_swiglu_mma_split_half_pair_r32_c128_i8_launch(const Tensor& x_q, const Tensor& x_scale,
                                                             const Weight& weight, Tensor& out,
                                                             cudaStream_t stream) {
    launch_route_i8<GateUpC128Cfg>(x_q, x_scale, weight, out, stream);
}

} // namespace ninfer::ops::detail
