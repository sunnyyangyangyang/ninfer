#pragma once

// Q4G64 RowSplit x INT8-MMA GEMM (int8 activation route).
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// Same tile schedule and shared-memory layout as the BF16 route, with the
// weight codes expanded in shared memory to raw int8 pairs (u16 = two signed
// int8 values) instead of dequantized BF16, and the activations pre-quantized
// to INT8 with one FP16 scale per (token, 64-K group).  Contraction runs on
// mma.sync m16n8k32 s8 with exact int32 accumulation; each 64-wide K tile
// flushes the tile accumulator through the per-row weight scale and the
// per-column activation scale into the FP32 result accumulator.  The final
// epilogue is the BF16 store (or fused epilogue) of the FP32 accumulator, so
// the only new numerical profile versus the BF16 route is the int8 group
// quantization of the activations.
//
// The shared arrays As/Bs are declared as in the BF16 kernel and re-viewed as
// uint16/int8: the int8 data occupies the low half of each array, so the
// schedule's shared-memory budget is unchanged.

#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// clang-format off
template <class Schedule_, bool Full>
__global__ __launch_bounds__(Schedule_::kThreads, Schedule_::kLaunchBoundsMinBlocks)
void q4_rowsplit_gemm_mma_i8_kernel(
    const std::int8_t* __restrict__ x_q,
    const std::uint16_t* __restrict__ x_scale,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k) {
    // clang-format on
    using Schedule       = Schedule_;
    constexpr bool kFull = Full;
    constexpr int BM     = Schedule::kBlockRows;
    constexpr int BN     = Schedule::kBlockCols;
    constexpr int BK     = Schedule::kBlockK;
    constexpr int WM     = Schedule::kWarpRows;
    constexpr int WN     = Schedule::kWarpCols;
    constexpr int MT     = Schedule::kMmaRows;
    constexpr int NT     = Schedule::kMmaCols;
    constexpr int KSUB   = BK / 32;
    constexpr int S      = Schedule::kPipelineStages;
    constexpr int GPB    = Schedule::kGroupsPerK;
    constexpr int SB     = Schedule::kScaleBytes;
    constexpr int KB2    = BK / 2;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * GPB * Q4RowSplitStorage::kCodeBytesPerGroup];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * GPB * SB];
    __shared__ __align__(16) std::uint16_t Xs[S][BN];

    auto* Asu = reinterpret_cast<std::uint16_t*>(As);
    auto* Bs8 = reinterpret_cast<std::uint8_t*>(Bs);

    const int groups_per_row = padded_k / Q4RowSplitStorage::kGroupK;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int warp_row       = warp / Schedule::kWarpGridCols;
    const int warp_col       = warp % Schedule::kWarpGridCols;
    const int mma_row        = lane >> 2;
    const int mma_col        = lane & 3;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    float accum[MT][NT][4];
    int acc_int[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                accum[mi][ni][c] = 0.0f;
                acc_int[mi][ni][c] = 0;
            }
        }
    }

    const int k_tiles = padded_k / BK;

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
        const int x8 = BK / 16;
#pragma unroll 1
        for (int item = tid; item < BN * x8; item += Schedule::kThreads) {
            const int local_col = item / x8;
            const int k16       = item - local_col * x8;
            const int kk        = k0 + k16 * 16;
            const int col       = col0 + local_col;
            auto* dst           = &Bs8[stage * (BN * BK) + local_col * BK + 2 * q4_mma_swizzle_k64(local_col, k16 * 8)];
            if constexpr (kFull) {
                cp_async<16, Schedule::kActivationCache>(dst,
                                                         &x_q[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                if (col < cols && kk + 16 <= k) {
                    cp_async<16, Schedule::kActivationCache>(
                        dst, &x_q[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        const int g = k_tile;
#pragma unroll 1
        for (int tl = tid; tl < BN; tl += Schedule::kThreads) {
            const int col = col0 + tl;
            Xs[stage][tl] = (kFull || col < cols) ? x_scale[static_cast<std::int64_t>(col) * groups_per_row + g]
                                           : static_cast<std::uint16_t>(0x3C00u);
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int group0 = (k_tile * BK) / Q4RowSplitStorage::kGroupK;
#pragma unroll 1
        for (int item = tid; item < BM * GPB * 2; item += Schedule::kThreads) {
            const int row_group = item >> 1;
            const int half      = item & 1;
            const int local_row = row_group / GPB;
            const int group     = row_group - local_row * GPB;
            const int row       = row0 + local_row;
            auto* dst           = &Cr[stage][row_group * Q4RowSplitStorage::kCodeBytesPerGroup + half * 16];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                cp_async<16, Schedule::kQuantCache>(
                    dst, &codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + half * 16]);
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                    cp_async<16, Schedule::kQuantCache>(
                        dst,
                        &codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + half * 16]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }

#pragma unroll 1
        for (int row_group = tid; row_group < BM * GPB; row_group += Schedule::kThreads) {
            const int local_row   = row_group / GPB;
            const int group       = row_group - local_row * GPB;
            const int row         = row0 + local_row;
            const int scale_group = group0 + group;
            auto* dst             = &Sr[stage][row_group * SB];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                if constexpr (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32) {
                    const int aligned_group = scale_group & ~1;
                    const std::int64_t aligned_index =
                        static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                    if (aligned_group + 1 < groups_per_row) {
                        cp_async<4>(
                            dst, &scales[aligned_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(
                            &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                }
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                    if constexpr (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32) {
                        const int aligned_group = scale_group & ~1;
                        const std::int64_t aligned_index =
                            static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                        if (aligned_group + 1 < groups_per_row) {
                            cp_async<4>(
                                dst,
                                &scales[aligned_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                        } else {
                            *reinterpret_cast<std::uint16_t*>(dst) =
                                *reinterpret_cast<const std::uint16_t*>(
                                    &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                            *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                        }
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Q4RowSplitStorage::kScaleBytesPerGroup]);
                    }
                } else {
                    dst[0] = 0;
                    dst[1] = 0;
                    if constexpr (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32) {
                        dst[2] = 0;
                        dst[3] = 0;
                    }
                }
            }
        }
    };

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_quant(stage, k_tile);
    };

    auto expand_weight = [&](int stage, int k_tile) {
        for (int local_row = warp; local_row < BM; local_row += Schedule::kWarps) {
            auto* dst = &Asu[local_row * KB2];
            if constexpr (!kFull) {
                if (row0 + local_row >= rows) {
                    for (int c = 0; c < KB2 / 8; ++c) {
                        store_vec(&Asu[local_row * KB2 + c * 8], make_int4(0, 0, 0, 0));
                    }
                    continue;
                }
            }
            const int group          = k_tile % GPB;
            const std::uint8_t packed = Cr[stage][local_row * GPB * Q4RowSplitStorage::kCodeBytesPerGroup
                                                 + group * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
            const int v0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
            const int v1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
            const std::uint16_t pair =
                static_cast<std::uint16_t>(v1) << 8 | static_cast<std::uint16_t>(static_cast<std::uint8_t>(v0));
            // u16 column space (one pair per column), same space as the ldmatrix below.
            dst[q4_mma_swizzle_k64(local_row, group * (Q4RowSplitStorage::kGroupK / 2) + lane)] = pair;
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        cp_wait<S - 1>();
        __syncthreads();

        expand_weight(stage, k_tile);
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_frag)[MT][4], unsigned(&b_frag)[NT][2]) {
            const int ku = k_step * (BK / 4);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_row * WM + mi * 16 + a_row_offset;
                const int col = ku + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&Asu[row * KB2 + q4_mma_swizzle_k64(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_col * WN + ni * 8 + b_inner_row;
                const int col = ku + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&Bs8[stage * (BN * BK) + row * BK + 2 * q4_mma_swizzle_k64(row, col)]));
            }
        };

        if constexpr (Schedule::kFragmentPipeline == Q4FragmentPipeline::PingPong) {
            unsigned a_frag[2][MT][4];
            unsigned b_frag[2][NT][2];
            load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                const int current = ki & 1;
                const int next    = (ki + 1) & 1;
                if (ki + 1 < KSUB) { load_fragments(ki + 1, a_frag[next], b_frag[next]); }
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_s8(acc_int[mi][ni][0], acc_int[mi][ni][1], acc_int[mi][ni][2],
                               acc_int[mi][ni][3], a_frag[current][mi][0], a_frag[current][mi][1],
                               a_frag[current][mi][2], a_frag[current][mi][3],
                               b_frag[current][ni][0], b_frag[current][ni][1]);
                    }
                }
            }
        } else {
            unsigned a_frag[MT][4];
            unsigned b_frag[NT][2];
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                load_fragments(ki, a_frag, b_frag);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_s8(acc_int[mi][ni][0], acc_int[mi][ni][1], acc_int[mi][ni][2],
                               acc_int[mi][ni][3], a_frag[mi][0], a_frag[mi][1], a_frag[mi][2],
                               a_frag[mi][3], b_frag[ni][0], b_frag[ni][1]);
                    }
                }
            }
        }

        {
            const int scale_off =
                (Schedule::kScaleLoadMode == Q4ScaleLoad::Pair32) ? ((k_tile & 1) * SB / 2) : 0;
            const int group = k_tile * GPB;
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int r0_local = warp_row * WM + mi * 16 + mma_row;
                const float w0 =
                    __half2float(*reinterpret_cast<const __half*>(&Sr[stage][(r0_local) * SB + scale_off]));
                const float w1 = __half2float(
                    *reinterpret_cast<const __half*>(&Sr[stage][(r0_local + 8) * SB + scale_off]));
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int c0 = warp_col * WN + ni * 8 + 2 * mma_col;
                    const float x0 = __half2float(*reinterpret_cast<const __half*>(&Xs[stage][c0]));
                    const float x1 = __half2float(*reinterpret_cast<const __half*>(&Xs[stage][c0 + 1]));
                    const float f00 = w0 * x0;
                    const float f01 = w0 * x1;
                    const float f10 = w1 * x0;
                    const float f11 = w1 * x1;
                    accum[mi][ni][0] += f00 * static_cast<float>(acc_int[mi][ni][0]);
                    accum[mi][ni][1] += f01 * static_cast<float>(acc_int[mi][ni][1]);
                    accum[mi][ni][2] += f10 * static_cast<float>(acc_int[mi][ni][2]);
                    accum[mi][ni][3] += f11 * static_cast<float>(acc_int[mi][ni][3]);
                    acc_int[mi][ni][0] = 0;
                    acc_int[mi][ni][1] = 0;
                    acc_int[mi][ni][2] = 0;
                    acc_int[mi][ni][3] = 0;
                }
            }
            (void)group;
        }

        __syncthreads();
        const int prefetch_tile = k_tile + S;
        if (prefetch_tile < k_tiles) { stage_inputs(stage, prefetch_tile); }
        cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int output_row0 = row0 + warp_row * WM + mi * 16 + mma_row;
        const int output_row1 = output_row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int output_col0 = col0 + warp_col * WN + ni * 8 + 2 * mma_col;
            const int output_col1 = output_col0 + 1;
            const float* values   = accum[mi][ni];
            if constexpr (kFull) {
                out[static_cast<std::int64_t>(output_col0) * rows + output_row0] =
                    __float2bfloat16_rn(values[0]);
                out[static_cast<std::int64_t>(output_col1) * rows + output_row0] =
                    __float2bfloat16_rn(values[1]);
                out[static_cast<std::int64_t>(output_col0) * rows + output_row1] =
                    __float2bfloat16_rn(values[2]);
                out[static_cast<std::int64_t>(output_col1) * rows + output_row1] =
                    __float2bfloat16_rn(values[3]);
            } else {
                if (output_row0 < rows) {
                    if (output_col0 < cols) {
                        out[static_cast<std::int64_t>(output_col0) * rows + output_row0] =
                            __float2bfloat16_rn(values[0]);
                    }
                    if (output_col1 < cols) {
                        out[static_cast<std::int64_t>(output_col1) * rows + output_row0] =
                            __float2bfloat16_rn(values[1]);
                    }
                }
                if (output_row1 < rows) {
                    if (output_col0 < cols) {
                        out[static_cast<std::int64_t>(output_col0) * rows + output_row1] =
                            __float2bfloat16_rn(values[2]);
                    }
                    if (output_col1 < cols) {
                        out[static_cast<std::int64_t>(output_col1) * rows + output_row1] =
                            __float2bfloat16_rn(values[3]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
