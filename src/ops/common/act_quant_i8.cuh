#pragma once

// Shared activation pre-quantization for the int8-MMA GEMM routes.
//
// Quantizes the BF16 activation tile x[T][K] (token-major) to INT8 codes plus
// one FP16 scale per (token, 64-K group), aligned to the weight quant groups
// so each 64-wide K tile of an int8-MMA GEMM folds exactly one scale product.
//
//   x_q[t][k]     = round(x[t][k] / scale[t][g])     (exact for |x| <= absmax)
//   x_scale[t][g] = absmax(x[t][g*64..g*64+64]) / 127  (FP16, 1.0 when absmax == 0)
//
// The group width (64) is fixed by the Q4/Q5 weight group size.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void act_quant_i8_launch(const __nv_bfloat16* x, std::int8_t* x_q, __half* x_scale,
                         std::int32_t k, std::int32_t t, cudaStream_t stream);

// Growable static scratch for activation staging when no WorkspaceArena is
// available (workspace-less op overloads). Returns a device pointer valid for
// at least `bytes`; the buffer only grows, never shrinks.
void* act_quant_i8_scratch(std::int64_t bytes, cudaStream_t stream);

} // namespace ninfer::ops::detail
