#pragma once

#include <tuple>
#include <vector>
#include <optional>
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "aclnn/opdev/platform.h"
#include "ms_extension/all.h"

#include "config.hpp"
#include "event.hpp"

namespace deep_ep {

struct Buffer {
    int64_t rank, rdma_rank, nvl_rank;
    int64_t num_ranks, num_rdma_ranks, num_nvl_ranks;
    op::SocVersion soc_version;

    int64_t num_nvl_bytes;
    int64_t num_rdma_bytes;

    bool low_latency_mode = false;
    bool is_padding = false;
    int padding_cnt = 0;
    ms::Tensor ori_x;
    ms::Tensor new_topk_idx;
    ms::Tensor new_scales;
    ms::Tensor notify_send_data;
    ms::Tensor send_token_idx_small;
    int notify_send_data_size;

    int64_t shared_expert_rank_num;
    int64_t shared_expert_num = 1;

private:
    std::string moe_all_to_all_group_name;
    void *shmem_ptr = nullptr;
    bool shmem_enable;

    int device_id;

    HcclComm ep_comm;

    bool available = false;

public:
    Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
           std::string moe_all_to_all_group_name);

    ~Buffer() noexcept(false);

    bool is_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    std::tuple<ms::Tensor, std::optional<ms::Tensor>, ms::Tensor, ms::Tensor, std::optional<EventHandle>>
    get_dispatch_layout(const ms::Tensor &topk_idx, int num_experts, std::optional<EventHandle> &previous_event,
                        bool async, bool allocate_on_comm_stream);

    ms::Tensor get_notify_send_data();

    std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<ms::Tensor>, std::optional<ms::Tensor>,
               std::vector<int>, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor,
               std::optional<EventHandle>>
    intranode_dispatch(const ms::Tensor &x, const std::optional<ms::Tensor> &x_scales,
                       const std::optional<ms::Tensor> &topk_idx, const std::optional<ms::Tensor> &topk_weights,
                       const std::optional<ms::Tensor> &num_tokens_per_rank, const ms::Tensor &is_token_in_rank,
                       const std::optional<ms::Tensor> &num_tokens_per_expert, int cached_num_recv_tokens,
                       const std::optional<ms::Tensor> &cached_rank_prefix_matrix,
                       const std::optional<ms::Tensor> &cached_channel_prefix_matrix,
                       const std::optional<ms::Tensor> &dispatch_wait_recv_cost_stats, int expert_alignment,
                       int num_worst_tokens, const Config &config, std::optional<EventHandle> &previous_event,
                       bool async, bool allocate_on_comm_stream, bool use_quant);

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<EventHandle>> intranode_combine(
        const ms::Tensor &x, const ms::Tensor &topk_idx, const std::optional<ms::Tensor> &topk_weights,
        const ms::Tensor &src_idx, const ms::Tensor &send_head, const ms::Tensor &put_offset,
        const std::optional<ms::Tensor> &combine_send_cost_stats);

    std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<ms::Tensor>, std::optional<ms::Tensor>,
               std::vector<int>, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor,
               std::optional<EventHandle>>
    internode_dispatch(const ms::Tensor &x, const std::optional<ms::Tensor> &x_scales,
                       const std::optional<ms::Tensor> &topk_idx, const std::optional<ms::Tensor> &topk_weights,
                       const std::optional<ms::Tensor> &num_tokens_per_rank,
                       const std::optional<ms::Tensor> &num_tokens_per_rdma_rank,
                       const ms::Tensor &is_token_in_rank, const std::optional<ms::Tensor> &num_tokens_per_expert,
                       const Config &config, std::optional<EventHandle> &previous_event, bool async,
                       bool allocate_on_comm_stream, bool use_quant);

    std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<EventHandle>> internode_combine(
        const ms::Tensor &x, const ms::Tensor &topk_idx, const std::optional<ms::Tensor> &topk_weights,
        const ms::Tensor &src_idx, const ms::Tensor &send_head, const ms::Tensor &offsetInner,
        const ms::Tensor &offsetOuter, const ms::Tensor &countOuter, const ms::Tensor &expand_scales);

    std::tuple<ms::Tensor, std::optional<ms::Tensor>, ms::Tensor, ms::Tensor, ms::Tensor, std::optional<EventHandle>,
               std::optional<std::function<void()>>>
    low_latency_dispatch(const ms::Tensor &x, const ms::Tensor &topk_idx,
                         const std::optional<ms::Tensor> &cumulative_local_expert_recv_stats,
                         int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, bool use_fp8, bool round_scale,
                         bool use_ue8m0, bool async, bool return_recv_hook);

    std::tuple<ms::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
        const ms::Tensor &x, const ms::Tensor &topk_idx, const ms::Tensor &topk_weights, const ms::Tensor &src_info,
        const ms::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
        const ms::Tensor &packed_recv_count, bool zero_copy, bool async, bool return_recv_hook,
        const std::optional<ms::Tensor> &out);

    std::vector<ms::Tensor> fused_deep_moe(const ms::Tensor &x, const ms::Tensor &expertIds,
                                           const ms::Tensor &gmm1PermutedWeight,
                                           const ms::Tensor &gmm1PermutedWeightScale, const ms::Tensor &gmm2Weight,
                                           const ms::Tensor &gmm2WeightScale, const ms::Tensor &expertScalesOptional,
                                           int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
                                           int quant_mode);
};
}  // namespace deep_ep
