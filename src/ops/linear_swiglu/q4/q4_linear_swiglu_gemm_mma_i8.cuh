#pragma once

// Folded gate/up GEMM, int8-MMA route.
//
// Same folded layout and SiLU epilogue as the BF16 kernel, with raw int8
// weight pairs in shared memory (u16 = two signed int8 values, exact
// expansion of the Q4 nibbles), INT8 activations pre-quantized per
// (token, 64-K group), mma.sync m16n8k32 s8 contraction, and a per-K-tile
// flush of the int32 tile accumulator through the weight row scale and the
// activation column scale into the FP32 accumulator.  A logical BM=64 weight tile is laid out as 32 gate
// rows followed by their 32 matching up rows.  One normal BM64 tensor-core
// contraction therefore produces both projections in one accumulator array;
// the warp's first and second row halves pair directly in the SiLU epilogue.

#include "ops/common/math.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Cfg, bool FullTiles>
__global__
__launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void q4_linear_swiglu_mma_split_half_pair_i8_kernel(
    const std::int8_t* __restrict__ x_q, const std::uint16_t* __restrict__ x_scale,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t intermediate, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    constexpr int BM   = Cfg::BM;
    constexpr int BN   = Cfg::BN;
    constexpr int BK   = Cfg::BK;
    constexpr int WM   = Cfg::WM;
    constexpr int WN   = Cfg::WN;
    constexpr int MT   = Cfg::MT;
    constexpr int NT   = Cfg::NT;
    constexpr int KSUB = BK / 32;
    constexpr int KB2  = BK / 2;
    constexpr int S    = Cfg::STAGES;
    constexpr int SB   = Cfg::SCALE_BYTES;
    constexpr int PM   = BM / 2;
    static_assert(BK == 64, "folded gate/up Q4 kernel requires one group per K tile");
    static_assert(BM == 64 && WM == 64 && MT == 4,
                  "folded gate/up mapping requires one 64-row warp tile");

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * 32];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * SB];
    __shared__ __align__(16) std::uint16_t Xs[S][BN];

    const int kg   = padded_k >> 6;
    auto* Asu       = reinterpret_cast<std::uint16_t*>(As);
    auto* Bs8 = reinterpret_cast<std::uint8_t*>(Bs);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wn   = warp;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int m0   = static_cast<int>(blockIdx.x) * PM;
    const int t0   = static_cast<int>(blockIdx.y) * BN;

    float acc[MT][NT][4];
    int acc_int[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[mi][ni][c]       = 0.0f;
                acc_int[mi][ni][c]   = 0;
            }
        }
    }

    const int NKT      = padded_k / BK;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    auto stage_load_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
        const int x8 = BK / 16;
#pragma unroll 1
        for (int c = tid; c < BN * x8; c += Cfg::THREADS) {
            const int tl   = c / x8;
            const int k16  = c - tl * x8;
            const int col  = t0 + tl;
            const int kk   = k0 + k16 * 16;
            auto* dst      = &Bs8[stage * (BN * BK) + tl * BK + 2 * gemm_swz64(tl, k16 * 8)];
            if constexpr (FullTiles) {
                gemm_cp_async<16, Cfg>(dst, &x_q[static_cast<std::int64_t>(col) * k + kk]);
            } else if (col < t && kk + 16 <= k) {
                gemm_cp_async<16, Cfg>(dst, &x_q[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int tl = tid; tl < BN; tl += Cfg::THREADS) {
            const int col = t0 + tl;
            Xs[stage][tl] = (FullTiles || col < t)
                  ? x_scale[static_cast<std::int64_t>(col) * kg + kt]
                  : static_cast<std::uint16_t>(0x3C00u);
        }
    };

    auto global_row = [&](int row) { return m0 + (row & (PM - 1)) + (row / PM) * intermediate; };

    auto stage_load_quant = [&](int stage, int kt) {
        const int g = (kt * BK) >> 6;
#pragma unroll 1
        for (int c = tid; c < BM * 2; c += Cfg::THREADS) {
            const int row  = c >> 1;
            const int half = c & 1;
            const int grow = global_row(row);
            auto* dst      = &Cr[stage][row * 32 + half * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &codes[gi * 32 + half * 16]);
            } else if (m0 + (row & (PM - 1)) < intermediate) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_cp_async<16, Cfg>(dst, &codes[gi * 32 + half * 16]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
#pragma unroll 1
        for (int row = tid; row < BM; row += Cfg::THREADS) {
            const int grow = global_row(row);
            auto* dst      = &Sr[stage][row * SB];
            if constexpr (FullTiles) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if (aligned_g + 1 < kg) {
                    gemm_cp_async<4, Cfg>(dst, &scales[aligned_gi * 2]);
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&scales[gi * 2]);
                    *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                }
            } else if (m0 + (row & (PM - 1)) < intermediate) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if (aligned_g + 1 < kg) {
                    gemm_cp_async<4, Cfg>(dst, &scales[aligned_gi * 2]);
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&scales[gi * 2]);
                    *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                }
            } else {
                *reinterpret_cast<std::uint32_t*>(dst) = 0;
            }
        }
    };

    auto stage_load = [&](int stage, int kt) {
        stage_load_x(stage, kt);
        stage_load_quant(stage, kt);
    };

    auto expand_to_As = [&](int stage, int kt) {
        (void)kt;
        for (int row = warp; row < BM; row += Cfg::WARPS) {
            if constexpr (!FullTiles) {
                if (m0 + (row & (PM - 1)) >= intermediate) {
                    for (int c = 0; c < KB2 / 8; ++c) {
                        store_vec(&Asu[row * KB2 + c * 8], make_int4(0, 0, 0, 0));
                    }
                    continue;
                }
            }
            const std::uint8_t packed = Cr[stage][row * 32 + lane];
            const int v0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
            const int v1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
            const std::uint16_t pair =
                static_cast<std::uint16_t>(v1) << 8 | static_cast<std::uint16_t>(static_cast<std::uint8_t>(v0));
            // u16 column space (one pair per column), same space as the ldmatrix below.
            Asu[row * KB2 + gemm_swz64(row, lane)] = pair;
        }
    };

#pragma unroll
    for (int s = 0; s < S; ++s) {
        if (s < NKT) { stage_load(s, s); }
        ninfer::ops::cp_commit();
    }

    for (int it = 0; it < NKT; ++it) {
        const int stage = it % S;
        ninfer::ops::cp_wait<S - 1>();
        __syncthreads();
        expand_to_As(stage, it);
        __syncthreads();

        unsigned af[MT][4];
        unsigned bf[NT][2];
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int ku = ki * (BK / 4);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int arow = mi * 16 + a_rowoff;
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            smem_addr(&Asu[arow * KB2 + gemm_swz64(arow, ku + a_coloff)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int brow = wn * WN + ni * 8 + b_rin;
                ldmatrix_x2(bf[ni][0], bf[ni][1],
                            smem_addr(&Bs8[stage * (BN * BK) + brow * BK + 2 * gemm_swz64(brow, ku + b_koff)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_s8(acc_int[mi][ni][0], acc_int[mi][ni][1], acc_int[mi][ni][2],
                           acc_int[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                           bf[ni][0], bf[ni][1]);
                }
            }
        }

        {
            const int scale_off = ((it * BK >> 6) & 1) * 2;
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int wrow0 = mi * 16 + gid;
                const float w0  = __half2float(*reinterpret_cast<const __half*>(&Sr[stage][wrow0 * SB + scale_off]));
                const float w1  = __half2float(*reinterpret_cast<const __half*>(&Sr[stage][(wrow0 + 8) * SB + scale_off]));
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int c0 = wn * WN + ni * 8 + 2 * lid;
                    const float x0 = __half2float(*reinterpret_cast<const __half*>(&Xs[stage][c0]));
                    const float x1 = __half2float(*reinterpret_cast<const __half*>(&Xs[stage][c0 + 1]));
                    acc[mi][ni][0] += (w0 * x0) * static_cast<float>(acc_int[mi][ni][0]);
                    acc[mi][ni][1] += (w0 * x1) * static_cast<float>(acc_int[mi][ni][1]);
                    acc[mi][ni][2] += (w1 * x0) * static_cast<float>(acc_int[mi][ni][2]);
                    acc[mi][ni][3] += (w1 * x1) * static_cast<float>(acc_int[mi][ni][3]);
                    acc_int[mi][ni][0] = 0;
                    acc_int[mi][ni][1] = 0;
                    acc_int[mi][ni][2] = 0;
                    acc_int[mi][ni][3] = 0;
                }
            }
        }

        __syncthreads();
        const int next = it + S;
        if (next < NKT) { stage_load(stage, next); }
        ninfer::ops::cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < MT / 2; ++mi) {
        const int r0 = m0 + mi * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int cc0 = t0 + wn * WN + ni * 8 + 2 * lid;
            const int cc1 = cc0 + 1;
            auto store    = [&](int col, int row, float gv, float uv) {
                out[static_cast<std::int64_t>(col) * intermediate + row] =
                    __float2bfloat16_rn(silu(gv) * uv);
            };
            if constexpr (FullTiles) {
                store(cc0, r0, acc[mi][ni][0], acc[mi + MT / 2][ni][0]);
                store(cc1, r0, acc[mi][ni][1], acc[mi + MT / 2][ni][1]);
                store(cc0, r1, acc[mi][ni][2], acc[mi + MT / 2][ni][2]);
                store(cc1, r1, acc[mi][ni][3], acc[mi + MT / 2][ni][3]);
            } else {
                if (r0 < intermediate) {
                    if (cc0 < t) { store(cc0, r0, acc[mi][ni][0], acc[mi + MT / 2][ni][0]); }
                    if (cc1 < t) { store(cc1, r0, acc[mi][ni][1], acc[mi + MT / 2][ni][1]); }
                }
                if (r1 < intermediate) {
                    if (cc0 < t) { store(cc0, r1, acc[mi][ni][2], acc[mi + MT / 2][ni][2]); }
                    if (cc1 < t) { store(cc1, r1, acc[mi][ni][3], acc[mi + MT / 2][ni][3]); }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
