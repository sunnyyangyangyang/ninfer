#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "ops/common/act_quant_i8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 3> kRoutes{{
    {{1, 16}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{17, 20}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaI8R16C64S3},
    {{21, kAnyCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaI8R32C64S4},
}};

constexpr bool catalog_is_closed() noexcept {
    return kRoutes[0].cols.first == 1 && kRoutes[0].cols.last + 1 == kRoutes[1].cols.first &&
           kRoutes[1].cols.last + 1 == kRoutes[2].cols.first && kRoutes[2].cols.last == kAnyCols;
}

static_assert(catalog_is_closed(), "attention input routes must be exact and closed");

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r16.c64.s3";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaI8R16C64S3:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.i8.r16.c64.s3";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaI8R32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.i8.r32.c64.s4";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        return {route.schedule};
    }
    throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena* ws, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaI8R16C64S3:
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaI8R32C64S4: {
        const std::int32_t kk = x.ne[0];
        const std::int32_t tt = x.ne[1];
        DeviceArena::Scope scratch_scope(ws != nullptr ? ws->scope() : DeviceArena::Scope());
        Tensor x_q, x_scale;
        if (ws != nullptr) {
            x_q     = ws->alloc(DType::I8, {kk, tt});
            x_scale = ws->alloc(DType::FP16, {kk / 64, tt});
        } else {
            const std::int64_t need = static_cast<std::int64_t>(tt) * kk +
                                      static_cast<std::int64_t>(tt) * (kk / 64) * 2;
            char* base = static_cast<char*>(act_quant_i8_scratch(need, stream));
            x_q        = Tensor(base, DType::I8, {kk, tt});
            x_scale    = Tensor(base + static_cast<std::int64_t>(tt) * kk, DType::FP16, {kk / 64, tt});
        }
        act_quant_i8_launch(static_cast<const __nv_bfloat16*>(x.data), static_cast<std::int8_t*>(x_q.data),
                            static_cast<__half*>(x_scale.data), kk, tt, stream);
        if (plan.schedule == Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaI8R16C64S3) {
            q4_q5_attn_input_grouped_mma_i8_r16_c64_s3_launch(x_q, x_scale, query_key_weight,
                                                              gate_value_weight, q, gate, k, v, stream);
        } else {
            q4_q5_attn_input_grouped_mma_i8_r32_c64_s4_launch(x_q, x_scale, query_key_weight,
                                                              gate_value_weight, q, gate, k, v, stream);
        }
        return;
    }
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, WorkspaceArena* ws, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v,
                                  ws, stream);
}

} // namespace ninfer::ops::detail
