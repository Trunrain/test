#include <set>
#include <functional>
#include <optional>
#include <memory>

#include "ms_extension/all.h"

#include "deep_ep.hpp"
#include "config.hpp"
#include "event.hpp"

namespace mindspore {

std::shared_ptr<deep_ep::Buffer> deep_ep_buffer_create(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes,
                                                        int64_t num_rdma_bytes, bool low_latency_mode,
                                                        std::string moe_all_to_all_group_name)
{
    return std::make_shared<deep_ep::Buffer>(rank, num_ranks, num_nvl_bytes, num_rdma_bytes,
                                              low_latency_mode, moe_all_to_all_group_name);
}

bool deep_ep_buffer_is_available(std::shared_ptr<deep_ep::Buffer> &buffer)
{
    return buffer->is_available();
}

int deep_ep_buffer_get_num_rdma_ranks(std::shared_ptr<deep_ep::Buffer> &buffer)
{
    return buffer->get_num_rdma_ranks();
}

int deep_ep_buffer_get_rdma_rank(std::shared_ptr<deep_ep::Buffer> &buffer)
{
    return buffer->get_rdma_rank();
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, ms::Tensor, ms::Tensor, std::optional<deep_ep::EventHandle>>
deep_ep_buffer_get_dispatch_layout(std::shared_ptr<deep_ep::Buffer> &buffer, const ms::Tensor &topk_idx,
                                    int num_experts, std::optional<deep_ep::EventHandle> &previous_event,
                                    bool async, bool allocate_on_comm_stream)
{
    return buffer->get_dispatch_layout(topk_idx, num_experts, previous_event, async, allocate_on_comm_stream);
}

ms::Tensor deep_ep_buffer_get_notify_send_data(std::shared_ptr<deep_ep::Buffer> &buffer)
{
    return buffer->get_notify_send_data();
}

void deep_ep_buffer_clean_low_latency_buffer(std::shared_ptr<deep_ep::Buffer> &buffer,
                                              int num_max_dispatch_tokens_per_rank, int hidden, int num_experts)
{
    buffer->clean_low_latency_buffer(num_max_dispatch_tokens_per_rank, hidden, num_experts);
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<ms::Tensor>, std::optional<ms::Tensor>,
           std::vector<int>, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor,
           std::optional<deep_ep::EventHandle>>
deep_ep_buffer_intranode_dispatch(std::shared_ptr<deep_ep::Buffer> &buffer, const ms::Tensor &x,
                                   const std::optional<ms::Tensor> &x_scales,
                                   const std::optional<ms::Tensor> &topk_idx,
                                   const std::optional<ms::Tensor> &topk_weights,
                                   const std::optional<ms::Tensor> &num_tokens_per_rank,
                                   const ms::Tensor &is_token_in_rank,
                                   const std::optional<ms::Tensor> &num_tokens_per_expert,
                                   int cached_num_recv_tokens,
                                   const std::optional<ms::Tensor> &cached_rank_prefix_matrix,
                                   const std::optional<ms::Tensor> &cached_channel_prefix_matrix,
                                   const std::optional<ms::Tensor> &dispatch_wait_recv_cost_stats,
                                   int expert_alignment, int num_worst_tokens, const deep_ep::Config &config,
                                   std::optional<deep_ep::EventHandle> &previous_event, bool async,
                                   bool allocate_on_comm_stream, bool use_quant)
{
    return buffer->intranode_dispatch(x, x_scales, topk_idx, topk_weights, num_tokens_per_rank, is_token_in_rank,
                                       num_tokens_per_expert, cached_num_recv_tokens, cached_rank_prefix_matrix,
                                       cached_channel_prefix_matrix, dispatch_wait_recv_cost_stats, expert_alignment,
                                       num_worst_tokens, config, previous_event, async, allocate_on_comm_stream,
                                       use_quant);
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<deep_ep::EventHandle>>
deep_ep_buffer_intranode_combine(std::shared_ptr<deep_ep::Buffer> &buffer, const ms::Tensor &x,
                                  const ms::Tensor &topk_idx, const std::optional<ms::Tensor> &topk_weights,
                                  const ms::Tensor &src_idx, const ms::Tensor &send_head,
                                  const ms::Tensor &put_offset,
                                  const std::optional<ms::Tensor> &combine_send_cost_stats)
{
    return buffer->intranode_combine(x, topk_idx, topk_weights, src_idx, send_head, put_offset, combine_send_cost_stats);
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<ms::Tensor>, std::optional<ms::Tensor>,
           std::vector<int>, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor,
           std::optional<deep_ep::EventHandle>>
deep_ep_buffer_internode_dispatch(std::shared_ptr<deep_ep::Buffer> &buffer, const ms::Tensor &x,
                                   const std::optional<ms::Tensor> &x_scales,
                                   const std::optional<ms::Tensor> &topk_idx,
                                   const std::optional<ms::Tensor> &topk_weights,
                                   const std::optional<ms::Tensor> &num_tokens_per_rank,
                                   const std::optional<ms::Tensor> &num_tokens_per_rdma_rank,
                                   const ms::Tensor &is_token_in_rank,
                                   const std::optional<ms::Tensor> &num_tokens_per_expert,
                                   const deep_ep::Config &config, std::optional<deep_ep::EventHandle> &previous_event,
                                   bool async, bool allocate_on_comm_stream, bool use_quant)
{
    return buffer->internode_dispatch(x, x_scales, topk_idx, topk_weights, num_tokens_per_rank,
                                       num_tokens_per_rdma_rank, is_token_in_rank, num_tokens_per_expert, config,
                                       previous_event, async, allocate_on_comm_stream, use_quant);
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<deep_ep::EventHandle>>
deep_ep_buffer_internode_combine(std::shared_ptr<deep_ep::Buffer> &buffer, const ms::Tensor &x,
                                  const ms::Tensor &topk_idx, const std::optional<ms::Tensor> &topk_weights,
                                  const ms::Tensor &src_idx, const ms::Tensor &send_head,
                                  const ms::Tensor &offsetInner, const ms::Tensor &offsetOuter,
                                  const ms::Tensor &countOuter, const ms::Tensor &expand_scales)
{
    return buffer->internode_combine(x, topk_idx, topk_weights, src_idx, send_head, offsetInner, offsetOuter,
                                      countOuter, expand_scales);
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, ms::Tensor, ms::Tensor, ms::Tensor, std::optional<deep_ep::EventHandle>,
           std::optional<std::function<void()>>>
deep_ep_buffer_low_latency_dispatch(std::shared_ptr<deep_ep::Buffer> &buffer, const ms::Tensor &x,
                                     const ms::Tensor &topk_idx,
                                     const std::optional<ms::Tensor> &cumulative_local_expert_recv_stats,
                                     int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, bool use_fp8,
                                     bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook)
{
    return buffer->low_latency_dispatch(x, topk_idx, cumulative_local_expert_recv_stats,
                                         num_max_dispatch_tokens_per_rank, num_experts, use_fp8, round_scale,
                                         use_ue8m0, async, return_recv_hook);
}

std::tuple<ms::Tensor, std::optional<deep_ep::EventHandle>, std::optional<std::function<void()>>>
deep_ep_buffer_low_latency_combine(std::shared_ptr<deep_ep::Buffer> &buffer, const ms::Tensor &x,
                                    const ms::Tensor &topk_idx, const ms::Tensor &topk_weights,
                                    const ms::Tensor &src_info, const ms::Tensor &layout_range,
                                    int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
                                    const ms::Tensor &packed_recv_count, bool zero_copy, bool async,
                                    bool return_recv_hook, const std::optional<ms::Tensor> &out)
{
    return buffer->low_latency_combine(x, topk_idx, topk_weights, src_info, layout_range,
                                        num_max_dispatch_tokens_per_rank, num_experts, packed_recv_count,
                                        zero_copy, async, return_recv_hook, out);
}

std::vector<ms::Tensor> deep_ep_buffer_fused_deep_moe(std::shared_ptr<deep_ep::Buffer> &buffer,
                                                       const ms::Tensor &x, const ms::Tensor &expert_ids,
                                                       const ms::Tensor &gmm1_permuted_weight,
                                                       const ms::Tensor &gmm1_permuted_weight_scale,
                                                       const ms::Tensor &gmm2_weight,
                                                       const ms::Tensor &gmm2_weight_scale,
                                                       const ms::Tensor &expert_scales_optional,
                                                       int64_t num_max_dispatch_tokens_per_rank,
                                                       int64_t num_experts, int quant_mode)
{
    return buffer->fused_deep_moe(x, expert_ids, gmm1_permuted_weight, gmm1_permuted_weight_scale, gmm2_weight,
                                   gmm2_weight_scale, expert_scales_optional, num_max_dispatch_tokens_per_rank,
                                   num_experts, quant_mode);
}

}  // namespace mindspore

PYBIND11_MODULE(MS_EXTENSION_NAME, m)
{
    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(), pybind11::arg("num_sms") = 20,
             pybind11::arg("num_max_nvl_chunked_send_tokens") = 6, pybind11::arg("num_max_nvl_chunked_recv_tokens") = 256,
             pybind11::arg("num_max_rdma_chunked_send_tokens") = 6, pybind11::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &mindspore::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, std::string>())
        .def("is_available", &mindspore::deep_ep_buffer_is_available)
        .def("get_num_rdma_ranks", &mindspore::deep_ep_buffer_get_num_rdma_ranks)
        .def("get_rdma_rank", &mindspore::deep_ep_buffer_get_rdma_rank)
        .def("get_dispatch_layout", &mindspore::deep_ep_buffer_get_dispatch_layout)
        .def("get_notify_send_data", &mindspore::deep_ep_buffer_get_notify_send_data)
        .def("clean_low_latency_buffer", &mindspore::deep_ep_buffer_clean_low_latency_buffer)
        .def("intranode_dispatch", &mindspore::deep_ep_buffer_intranode_dispatch)
        .def("intranode_combine", &mindspore::deep_ep_buffer_intranode_combine)
        .def("internode_dispatch", &mindspore::deep_ep_buffer_internode_dispatch)
        .def("internode_combine", &mindspore::deep_ep_buffer_internode_combine)
        .def("low_latency_dispatch", &mindspore::deep_ep_buffer_low_latency_dispatch)
        .def("low_latency_combine", &mindspore::deep_ep_buffer_low_latency_combine)
        .def("fused_deep_moe", &mindspore::deep_ep_buffer_fused_deep_moe);
}
