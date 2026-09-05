#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/rowsplit_grouped_mma_i8.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                               Tensor& out) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > weight.n ||
        out.ne[0] != row_count) {
        throw std::invalid_argument("Q4/Q5 attention input grouped MMA row view is invalid");
    }
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(row_begin) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5G64_F16S
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(row_begin) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(row_begin) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,     high,      scales, static_cast<__nv_bfloat16*>(out.data),
        row_count, out.ne[0], 0,      weight.qtype == QType::Q5G64_F16S,
    };
}

template <class Schedule, RowSplitGroupedMmaCodec Codec>
void launch_pair(bool full, const Tensor& x, RowSplitGroupedMmaJob first,
                 RowSplitGroupedMmaJob second, cudaStream_t stream) {
    const int tiles = div_up(first.n, Schedule::BM) + div_up(second.n, Schedule::BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));
    RowSplitGroupedMmaJob empty{};

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first, second, empty, empty, x.ne[0], cols,
                                                     x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, RowSplitGroupedMmaCodec Codec>
void launch_pair_i8(bool full, const Tensor& x_q, const Tensor& x_scale, RowSplitGroupedMmaJob first,
                    RowSplitGroupedMmaJob second, cudaStream_t stream) {
    const int tiles = div_up(first.n, Schedule::BM) + div_up(second.n, Schedule::BM);
    const int cols  = x_q.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));
    RowSplitGroupedMmaJob empty{};

    if (full) {
        rowsplit_grouped_mma_i8_kernel<Schedule, true, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const std::int8_t*>(x_q.data),
                                                     static_cast<const std::uint16_t*>(x_scale.data),
                                                     first, second, empty, empty, x_q.ne[0], cols,
                                                     x_q.ne[0]);
    } else {
        rowsplit_grouped_mma_i8_kernel<Schedule, false, Codec, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const std::int8_t*>(x_q.data),
                                                     static_cast<const std::uint16_t*>(x_scale.data),
                                                     first, second, empty, empty, x_q.ne[0], cols,
                                                     x_q.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, RowSplitGroupedMmaCodec Codec, class Schedule2, RowSplitGroupedMmaCodec Codec2>
void launch_slice_i8(const Tensor& x_q, const Tensor& x_scale, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     cudaStream_t stream) {
    constexpr std::int32_t kQueryRows = 6144;
    const bool full                   = (x_q.ne[1] % 128) == 0;
    const RowSplitGroupedMmaJob jq = make_job(query_key_weight, 0, kQueryRows, q);
    const RowSplitGroupedMmaJob jk = make_job(query_key_weight, kQueryRows, k.ne[0], k);
    const RowSplitGroupedMmaJob jg = make_job(gate_value_weight, 0, gate.ne[0], gate);
    const RowSplitGroupedMmaJob jv = make_job(gate_value_weight, gate.ne[0], v.ne[0], v);
    launch_pair_i8<Schedule, Codec>(full, x_q, x_scale, jq, jk, stream);
    launch_pair_i8<Schedule2, Codec2>(full, x_q, x_scale, jg, jv, stream);
}

template <class Schedule, class Schedule2>
void launch_i8(const Tensor& x_q, const Tensor& x_scale, const Weight& query_key_weight,
               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
               cudaStream_t stream) {
    constexpr std::int32_t kSliceCols = 128;
    for_each_token_slice(x_q.ne[1], kSliceCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor xq_slice    = x_q.slice(1, offset, count);
        const Tensor xs_slice    = x_scale.slice(1, offset, count);
        Tensor q_slice           = q.slice(1, offset, count);
        Tensor gate_slice        = gate.slice(1, offset, count);
        Tensor k_slice           = k.slice(1, offset, count);
        Tensor v_slice           = v.slice(1, offset, count);
        launch_slice_i8<Schedule, RowSplitGroupedMmaCodec::Q4, Schedule2, RowSplitGroupedMmaCodec::Q5>(
            xq_slice, xs_slice, query_key_weight, gate_value_weight, q_slice, gate_slice, k_slice,
            v_slice, stream);
    });
}

template <class Schedule>
void launch_slice(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
                  Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    const bool full = (x.ne[1] % Schedule::BN) == 0;
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q4>(
        full, x, make_job(query_key_weight, 0, 6144, q), make_job(query_key_weight, 6144, 1024, k),
        stream);
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q5>(
        full, x, make_job(gate_value_weight, 0, 6144, gate),
        make_job(gate_value_weight, 6144, 1024, v), stream);
}

template <class Schedule>
void launch(const Tensor& x, const Weight& query_key_weight, const Weight& gate_value_weight,
            Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kSliceCols = 128;
    for_each_token_slice(x.ne[1], kSliceCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor gate_slice    = gate.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        launch_slice<Schedule>(x_slice, query_key_weight, gate_value_weight, q_slice, gate_slice,
                               k_slice, v_slice, stream);
    });
}

using MmaR16C64S3 = GemmCfg<16, 64, 64, 16, 16, 3, 1, false, true, true>;
using MmaR32C64S4 = GemmCfg<32, 64, 64, 16, 16, 4, 1, false, true, true>;

} // namespace

void q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream) {
    launch<MmaR16C64S3>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
}

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream) {
    launch<MmaR32C64S4>(x, query_key_weight, gate_value_weight, q, gate, k, v, stream);
}



void q4_q5_attn_input_grouped_mma_i8_r16_c64_s3_launch(const Tensor& x_q, const Tensor& x_scale,
                                                       const Weight& query_key_weight,
                                                       const Weight& gate_value_weight, Tensor& q,
                                                       Tensor& gate, Tensor& k, Tensor& v,
                                                       cudaStream_t stream) {
    launch_i8<MmaR16C64S3, MmaR16C64S3>(x_q, x_scale, query_key_weight, gate_value_weight, q, gate,
                                        k, v, stream);
}

void q4_q5_attn_input_grouped_mma_i8_r32_c64_s4_launch(const Tensor& x_q, const Tensor& x_scale,
                                                       const Weight& query_key_weight,
                                                       const Weight& gate_value_weight, Tensor& q,
                                                       Tensor& gate, Tensor& k, Tensor& v,
                                                       cudaStream_t stream) {
    launch_i8<MmaR32C64S4, MmaR32C64S4>(x_q, x_scale, query_key_weight, gate_value_weight, q, gate,
                                        k, v, stream);
}

} // namespace ninfer::ops::detail
