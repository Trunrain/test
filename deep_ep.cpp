#include <memory>
#include <cmath>

#include "hccl/hccl.h"
#include "exception.hpp"
#include "deep_ep.hpp"
#include "shmem.hpp"

namespace deep_ep {
constexpr int PADDING_SIZE = 1;
constexpr size_t HCOMM_NAME_LEN = 128;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr int LOCAL_RANK_SIZE = 8;
constexpr int MAX_BATCH_SIZE = 4096;
constexpr int EXPERT_DATA_SIZE = 1 + MAX_BATCH_SIZE;
constexpr int A3_MAX_HCCS_PEERS = 384;
constexpr int A2_MAX_HCCS_PEERS = 8;

ms::Tensor create_tensor_from_shmem(const std::vector<int64_t> &shape, ms::DType dtype, ms::Device &device)
{
    int64_t numel = 1;
    for (auto v : shape) {
        if (v <= 0) {
            throw std::runtime_error("invalid shape dimension");
        }
        numel *= v;
    }

    size_t ele_size = dtype.GetSize();
    if (ele_size == 0) {
        throw std::runtime_error("invalid dtype element size");
    }

    size_t bytes = static_cast<size_t>(numel) * ele_size;

    void *dev_ptr = shmem_malloc(bytes);
    if (!dev_ptr) {
        throw std::runtime_error("shmem_malloc failed");
    }

    auto tensor = ms::Tensor::from_blob(dev_ptr, shape, dtype, device,
                                         [](void *ptr) { shmem_free(ptr); });

    return tensor;
}

Buffer::Buffer(int64_t rank, int64_t num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode,
               std::string moe_all_to_all_group_name)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      moe_all_to_all_group_name(moe_all_to_all_group_name)
{
    rdma_rank = rank;
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks);

    this->shared_expert_rank_num = get_value_from_env("MOE_SHARED_EXPERT_RANK_NUM", 0);

    soc_version = op::GetCurrentPlatformInfo().GetSocVersion();
    num_rdma_ranks = 1;
    num_nvl_ranks = num_ranks;
    rdma_rank = rank;
    nvl_rank = rank;
    if (soc_version == op::SocVersion::ASCEND910B) {
        num_rdma_ranks = std::max(static_cast<int64_t>(1), num_ranks / A2_MAX_HCCS_PEERS);
        num_nvl_ranks = std::min(num_ranks, static_cast<int64_t>(A2_MAX_HCCS_PEERS));
        rdma_rank = rank / A2_MAX_HCCS_PEERS;
        nvl_rank = rank % A2_MAX_HCCS_PEERS;
    }

    shmem_enable = (get_value_from_env("DEEPEP_SHMEM_ENABLE", 0) == 1);
    if (shmem_enable) {
        size_t local_mem_size = 8 * 1024 * 1024 * 1024UL;
        size_t meta_data_size = 100 * 1024 * 1024UL;
        size_t ele_size = sizeof(int32_t);
        size_t num_of_int32 = meta_data_size / ele_size;

        EP_HOST_ASSERT(rank == internode::init(rank, num_ranks, local_mem_size, "tcp://127.0.0.1:11222"));
        shmem_ptr = internode::alloc(num_of_int32, ele_size);
    } else {
        if (moe_all_to_all_group_name.empty()) {
            char *ranktable_file = std::getenv("RANK_TABLE_FILE");
            EP_HOST_ASSERT(ranktable_file != nullptr);
            ACL_CHECK(aclrtGetDevice(&device_id));

            HCCL_CHECK(HcclCommInitClusterInfo(ranktable_file, device_id, &ep_comm));
        } else {
            EP_HOST_ASSERT(moe_all_to_all_group_name.size() < HCOMM_NAME_LEN);
        }
    }
}

Buffer::~Buffer() noexcept(false)
{
    if (shmem_enable) {
        internode::free(shmem_ptr);
        internode::finalize();
    }
}

bool Buffer::is_available() const
{
    return available;
}

int Buffer::get_num_rdma_ranks() const
{
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const
{
    return rdma_rank;
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, ms::Tensor, ms::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const ms::Tensor &topk_idx, int num_experts, std::optional<EventHandle> &previous_event,
                            bool async, bool allocate_on_comm_stream)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    this->new_topk_idx = topk_idx;
    if (topk_idx.size(0) < PADDING_SIZE) {
        this->is_padding = true;
        this->padding_cnt = PADDING_SIZE - topk_idx.size(0);
        std::vector<ms::Tensor> topk_blocks;
        if (topk_idx.size(0) != 0) {
            topk_blocks.emplace_back(topk_idx);
        }
        int topk = static_cast<int>(topk_idx.size(1));
        for (int i = 0; i < this->padding_cnt; i++) {
            ms::Tensor tmp_topk = ms::Tensor::arange(0, topk, topk_idx.dtype(), topk_idx.device()).reshape({1, topk});
            topk_blocks.emplace_back(tmp_topk);
        }
        this->new_topk_idx = ms::Tensor::concat(topk_blocks, 0);
    }

    const int num_tokens = new_topk_idx.size(0);
    const int num_topk = new_topk_idx.size(1);
    const int local_ranksize = LOCAL_RANK_SIZE;
    auto server_num = num_ranks / local_ranksize;

    auto device = new_topk_idx.device();
    ms::Tensor num_tokens_per_expert;
    if (shmem_enable) {
        num_tokens_per_expert = create_tensor_from_shmem(std::vector<int64_t>{num_experts}, ms::DType::Int32, device);
        num_tokens_per_expert.fill_(0);
    } else {
        num_tokens_per_expert = ms::Tensor::zeros({num_experts}, ms::DType::Int32, device);
    }
    auto num_tokens_per_rank = ms::Tensor::zeros({num_ranks}, ms::DType::Int32, device);
    auto is_token_in_rank = ms::Tensor::zeros({num_tokens, num_ranks}, ms::DType::Int32, device);
    const int notify_send_data_size =
        num_experts * EXPERT_DATA_SIZE + server_num + MAX_BATCH_SIZE * (1 + 2 * server_num + num_experts);
    auto send_token_idx_small = ms::Tensor::zeros({num_tokens, num_topk}, ms::DType::Int32, device);
    auto notify_send_data = ms::Tensor::zeros({notify_send_data_size}, ms::DType::Int32, device);

    EXEC_NPU_CMD(aclnnDispatchLayout, new_topk_idx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize,
                 num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, notify_send_data, send_token_idx_small);

    this->notify_send_data = notify_send_data;
    this->send_token_idx_small = send_token_idx_small;
    this->notify_send_data_size = notify_send_data_size;

    std::optional<ms::Tensor> num_tokens_per_rdma_rank = std::nullopt;
    std::optional<EventHandle> output_event = std::nullopt;

    return std::make_tuple(num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank,
                           output_event);
}

ms::Tensor Buffer::get_notify_send_data()
{
    return this->notify_send_data;
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts)
{
    return;
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<ms::Tensor>, std::optional<ms::Tensor>,
           std::vector<int>, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor,
           std::optional<EventHandle>>
Buffer::intranode_dispatch(const ms::Tensor &x, const std::optional<ms::Tensor> &x_scales,
                           const std::optional<ms::Tensor> &topk_idx, const std::optional<ms::Tensor> &topk_weights,
                           const std::optional<ms::Tensor> &num_tokens_per_rank, const ms::Tensor &is_token_in_rank,
                           const std::optional<ms::Tensor> &num_tokens_per_expert, int cached_num_recv_tokens,
                           const std::optional<ms::Tensor> &cached_rank_prefix_matrix,
                           const std::optional<ms::Tensor> &cached_channel_prefix_matrix,
                           const std::optional<ms::Tensor> &dispatch_wait_recv_cost_stats, int expert_alignment,
                           int num_worst_tokens, const Config &config, std::optional<EventHandle> &previous_event,
                           bool async, bool allocate_on_comm_stream, bool use_quant)
{
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    ms::Tensor expert_ids = new_topk_idx.to(ms::DType::Int32);
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t quant_mode = use_quant ? DYNAMIC_SCALES : NO_SCALES;
    auto recv_topk_idx = std::optional<ms::Tensor>();
    auto recv_topk_weights = std::optional<ms::Tensor>();
    std::optional<EventHandle> event;
    auto rank_prefix_matrix = ms::Tensor::empty({num_ranks, num_ranks}, ms::DType::Int32, x.device());
    auto channel_prefix_matrix = ms::Tensor::empty({num_ranks, num_channels}, ms::DType::Int32, x.device());
    auto recv_channel_prefix_matrix = ms::Tensor::empty({num_ranks, num_channels}, ms::DType::Int32, x.device());

    ms::Tensor new_x = x;
    if (topk_idx->size(0) < PADDING_SIZE) {
        this->is_padding = true;
        this->padding_cnt = PADDING_SIZE - topk_idx->size(0);
        std::vector<ms::Tensor> x_blocks;
        if (topk_idx->size(0) != 0) {
            x_blocks.emplace_back(x);
        } else {
            this->ori_x = x.clone();
        }
        for (int i = 0; i < this->padding_cnt; i++) {
            ms::Tensor tmp_x = ms::Tensor::ones({1, x.size(1)}, x.dtype(), x.device()) * (i + 1) * 2;
            x_blocks.emplace_back(tmp_x);
        }
        new_x = ms::Tensor::concat(x_blocks, 0);
    }

    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    EP_HOST_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert->size(0));
    auto num_local_experts = static_cast<int>(num_experts / num_ranks);

    int num_topk = 0;
    EP_HOST_ASSERT(topk_idx.has_value());
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
    EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
    EP_HOST_ASSERT(num_tokens == new_topk_idx.size(0));
    EP_HOST_ASSERT(num_topk == topk_weights->size(1));

    int send_per_group = 3;
    auto send_data = ms::Tensor::empty({num_experts * send_per_group}, ms::DType::Int32, x.device());
    int64_t send_count = send_per_group * num_local_experts * num_ranks;
    auto send_data_offset = ms::Tensor::empty({num_experts}, ms::DType::Int32, x.device());
    ms::Tensor recv_data = ms::Tensor::empty({num_experts * send_per_group}, ms::DType::Int32, x.device());

    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    ms::Tensor total_recv_token = ms::Tensor::empty({1}, ms::DType::Int32, x.device());
    ms::Tensor recv_offset = ms::Tensor::empty({num_experts}, ms::DType::Int32, x.device());
    ms::Tensor recv_count = ms::Tensor::empty({num_experts}, ms::DType::Int32, x.device());
    ms::Tensor max_bs = ms::Tensor::empty({1}, ms::DType::Int32, x.device());
    ms::Tensor recv_tokens_per_expert = ms::Tensor::empty({num_local_experts}, ms::DType::Int32, x.device());
    ms::Tensor expert_global_offset = ms::Tensor::empty({num_local_experts}, ms::DType::Int32, x.device());
    ms::Tensor srcrank_in_expert_offset = ms::Tensor::empty({num_local_experts * num_ranks}, ms::DType::Int32, x.device());
    ms::Tensor r_in_srcrank_offset = ms::Tensor::empty({num_local_experts * num_ranks}, ms::DType::Int32, x.device());

    int64_t local_rank_size = num_ranks;
    int64_t local_rank_id = rank % local_rank_size;
    auto new_num_tokens_per_expert = num_tokens_per_expert.value();
    std::vector<int> num_recv_tokens_per_expert_list;
    int expert_token_nums_type = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);

    EXEC_NPU_CMD(aclnnNotifyDispatch, send_data, new_num_tokens_per_expert, send_count, num_tokens,
                 hcom_ep_name, num_ranks, rank, local_rank_size, local_rank_id, send_data_offset, recv_data, recv_count,
                 recv_offset, expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset, total_recv_token,
                 max_bs, recv_tokens_per_expert);

    auto send_token_idx_small = this->send_token_idx_small;
    int64_t real_max_bs = static_cast<int64_t>(std::max(max_bs.item<int>(), static_cast<int>(num_worst_tokens)));
    int64_t global_bs = static_cast<int64_t>(std::min(static_cast<int64_t>(MAX_BATCH_SIZE), real_max_bs) * num_ranks);

    int64_t trt = total_recv_token.item<int>();
    int num_recv_tokens = (trt == 0) ? 1 : trt;
    auto expandx_out = use_quant ? ms::Tensor::empty({num_recv_tokens, hidden}, ms::DType::Int8, x.device())
                                 : ms::Tensor::empty({num_recv_tokens, hidden}, x.dtype(), x.device());
    auto dynamic_scales_out = ms::Tensor::empty({num_recv_tokens}, ms::DType::Float32, x.device());
    auto expand_idx_out = ms::Tensor::empty({num_recv_tokens * 3}, ms::DType::Int32, x.device());
    if (topk_idx.has_value()) {
        recv_topk_idx = ms::Tensor::empty({trt, num_topk}, topk_idx->dtype(), topk_idx->device());
        recv_topk_weights = ms::Tensor::empty({trt, num_topk}, topk_weights->dtype(), topk_weights->device());
    }

    EXEC_NPU_CMD(aclnnCamMoeDispatchNormal, new_x, expert_ids, send_data_offset, send_token_idx_small,
                 recv_offset, recv_count, expert_global_offset, srcrank_in_expert_offset,
                 r_in_srcrank_offset, hcom_ep_name, num_ranks, rank, hcom_ep_name, tp_size, tp_rank, num_experts,
                 quant_mode, real_max_bs, global_bs, expandx_out, dynamic_scales_out,
                 expand_idx_out, dispatch_wait_recv_cost_stats.value_or(ms::Tensor()));

    auto recv_token_per_exp_cpu = recv_tokens_per_expert.to(ms::DeviceType::CPU);
    auto recv_token_per_exp_ptr = recv_token_per_exp_cpu.data_ptr<int32_t>();

    int token_cnt = 0;
    for (int local_e = 0; local_e < num_local_experts; ++local_e) {
        int current_tokens = static_cast<int>(recv_token_per_exp_ptr[local_e]);
        token_cnt = (expert_token_nums_type == 0) ? token_cnt + current_tokens : current_tokens;
        num_recv_tokens_per_expert_list.emplace_back(token_cnt);
    }

    auto recv_count_one_dim = recv_count.sum(0, false).to(ms::DType::Int32);
    return {expandx_out, dynamic_scales_out, recv_topk_idx, recv_topk_weights,
            num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix,
            recv_channel_prefix_matrix, expand_idx_out, recv_count_one_dim, event};
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<EventHandle>>
Buffer::intranode_combine(const ms::Tensor &x, const ms::Tensor &topk_idx,
                          const std::optional<ms::Tensor> &topk_weights, const ms::Tensor &src_idx,
                          const ms::Tensor &send_head, const ms::Tensor &put_offset,
                          const std::optional<ms::Tensor> &combine_send_cost_stats)
{
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    ms::Tensor recv_x = x;
    auto topk_idx_int32 = topk_idx.to(ms::DType::Int32);
    ms::Tensor token_src_info = src_idx;
    ms::Tensor ep_send_counts = send_head;
    auto device = x.device();

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);
    ms::Tensor expert_scales = topk_weights.value_or(ms::Tensor::ones({num_tokens, num_topk}, ms::DType::Float32, device));

    int64_t hidden = static_cast<int>(recv_x.size(1));
    ms::Tensor tp_send_counts = ms::Tensor::empty({1}, ms::DType::Int32, device);
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t moe_expert_number = send_head.size(0);

    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    auto combined_x = ms::Tensor::empty({expert_scales.size(0), hidden}, x.dtype(), x.device());
    std::optional<ms::Tensor> recv_topk_weights;
    std::optional<EventHandle> event;

    EXEC_NPU_CMD(aclnnCamMoeCombineNormal, recv_x, token_src_info, ep_send_counts, expert_scales,
                 topk_idx_int32, tp_send_counts, hcom_ep_name, num_ranks, rank, hcom_ep_name, tp_world_size,
                 tp_rankId, moe_expert_number, combined_x, combine_send_cost_stats.value_or(ms::Tensor()));

    return {combined_x, recv_topk_weights, event};
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<ms::Tensor>, std::optional<ms::Tensor>,
           std::vector<int>, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor, ms::Tensor,
           std::optional<EventHandle>>
Buffer::internode_dispatch(
    const ms::Tensor &x, const std::optional<ms::Tensor> &x_scales, const std::optional<ms::Tensor> &topk_idx,
    const std::optional<ms::Tensor> &topk_weights, const std::optional<ms::Tensor> &num_tokens_per_rank,
    const std::optional<ms::Tensor> &num_tokens_per_rdma_rank, const ms::Tensor &is_token_in_rank,
    const std::optional<ms::Tensor> &num_tokens_per_expert, const Config &config,
    std::optional<EventHandle> &previous_event, bool async, bool allocate_on_comm_stream, bool use_quant)
{
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    ms::Tensor new_x = x;
    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    EP_HOST_ASSERT(new_x.dim() == 2 and new_x.is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);

    auto num_tokens = static_cast<int>(new_x.size(0)), hidden = static_cast<int>(new_x.size(1));
    auto num_experts = static_cast<int64_t>(num_tokens_per_expert->size(0));
    auto num_local_experts = static_cast<int>(num_experts / num_ranks);

    int num_topk = 0;
    EP_HOST_ASSERT(topk_idx.has_value());
    ms::Tensor expert_ids = topk_idx.value().to(ms::DType::Int32);
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
    EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
    EP_HOST_ASSERT(num_tokens == expert_ids.size(0));
    EP_HOST_ASSERT(num_topk == topk_weights->size(1));

    auto device = x.device();
    ms::Tensor new_topk_weights = topk_weights.value_or(ms::Tensor::ones({num_tokens, num_topk}, ms::DType::Float32, device));

    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t expertShardType = 0;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 0;
    int64_t expertTokenNumsType = 0;
    int64_t quant_mode = use_quant ? DYNAMIC_SCALES : NO_SCALES;
    int64_t global_bs = static_cast<int64_t>(MAX_BATCH_SIZE * num_ranks);
    ms::Tensor xActiveMask = ms::Tensor::empty({1}, ms::DType::Int32, x.device());
    auto expertTokenNums = ms::Tensor::zeros({1}, ms::DType::Int64, x.device());
    auto epRecvCount = ms::Tensor::zeros({1}, ms::DType::Int32, x.device());
    auto tpRecvCount = ms::Tensor::zeros({1}, ms::DType::Int32, x.device());
    ms::Tensor dispatch_wait_recv_cost_stats_out;

    int64_t local_rank_size = A2_MAX_HCCS_PEERS;
    int32_t server_num = num_ranks / local_rank_size;
    int64_t local_rank_id = rank % local_rank_size;
    auto new_num_tokens_per_expert = num_tokens_per_expert.value();
    std::vector<int> num_recv_tokens_per_expert_list;
    int expert_token_nums_type = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);

    auto new_send_data = this->notify_send_data;
    int send_count = this->notify_send_data_size;

    auto send_data_offset = ms::Tensor::empty({num_experts}, ms::DType::Int32, x.device());
    ms::Tensor tmp_data = ms::Tensor::empty({send_count * num_ranks}, ms::DType::Int32, x.device());
    ms::Tensor recv_data = ms::Tensor::empty({send_count * num_ranks}, ms::DType::Int32, x.device());
    ms::Tensor token_server_idx = ms::Tensor::empty({MAX_BATCH_SIZE, server_num}, ms::DType::Int32, x.device());
    ms::Tensor token_unique_per_server = ms::Tensor::empty({server_num}, ms::DType::Int32, x.device());
    ms::Tensor ep_rank_token_cnt = ms::Tensor::empty({num_experts, num_ranks}, ms::DType::Int32, x.device());
    ms::Tensor recv_tokens_per_expert = ms::Tensor::empty({num_local_experts}, ms::DType::Int64, x.device());
    ms::Tensor src_offset_rank_token_idx = ms::Tensor::empty({num_experts, num_ranks, MAX_BATCH_SIZE}, ms::DType::Int32, x.device());
    ms::Tensor dst_offset_rank_token_idx = ms::Tensor::empty({num_experts, num_ranks, MAX_BATCH_SIZE}, ms::DType::Int32, x.device());
    ms::Tensor offset_inner = ms::Tensor::empty({2, MAX_BATCH_SIZE, num_experts}, ms::DType::Int32, x.device());
    ms::Tensor count_outer = ms::Tensor::empty({MAX_BATCH_SIZE}, ms::DType::Int32, x.device());
    ms::Tensor expand_idx = ms::Tensor::empty({MAX_BATCH_SIZE, num_experts}, ms::DType::Int32, x.device());
    ms::Tensor total_recv_token = ms::Tensor::empty({1}, ms::DType::Int32, x.device());

    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    EXEC_NPU_CMD(aclnnNotifyDispatchA2, new_send_data, new_num_tokens_per_expert, tmp_data, send_count,
                 num_tokens, num_topk, num_experts, hcom_ep_name, num_ranks, rank, local_rank_size, local_rank_id,
                 send_data_offset, recv_data, token_server_idx, token_unique_per_server, ep_rank_token_cnt,
                 recv_tokens_per_expert, src_offset_rank_token_idx, dst_offset_rank_token_idx, offset_inner,
                 count_outer, expand_idx, total_recv_token);

    int total_count = total_recv_token.item<int>();
    int num_recv_tokens = (total_count == 0) ? 1 : total_count;

    auto expandx_out = use_quant ? ms::Tensor::empty({num_recv_tokens, hidden}, ms::DType::Int8, x.device())
                                 : ms::Tensor::empty({num_recv_tokens, hidden}, x.dtype(), x.device());
    auto dynamic_scales_out = ms::Tensor::empty({num_recv_tokens}, ms::DType::Float32, x.device());
    auto expand_scales = ms::Tensor::empty({num_recv_tokens}, ms::DType::Float32, x.device());
    std::optional<ms::Tensor> recv_topk_idx, recv_topk_weights;
    if (topk_idx.has_value()) {
        recv_topk_idx = ms::Tensor::empty({total_count, num_topk}, topk_idx->dtype(), topk_idx->device());
        recv_topk_weights = ms::Tensor::empty({total_count, num_topk}, topk_weights->dtype(), topk_weights->device());
    }

    EXEC_NPU_CMD(aclnnDispatchNormalA2, new_x, expert_ids, x_scales.value_or(ms::Tensor()), xActiveMask, new_topk_weights,
                 token_server_idx, token_unique_per_server, ep_rank_token_cnt, src_offset_rank_token_idx,
                 dst_offset_rank_token_idx, hcom_ep_name, num_ranks, rank, num_experts, hcom_ep_name, tp_size,
                 tp_rank, expertShardType, sharedExpertNum, sharedExpertRankNum, quant_mode, global_bs,
                 expertTokenNumsType, expandx_out, dynamic_scales_out, expand_idx, expertTokenNums,
                 epRecvCount, expand_scales, dispatch_wait_recv_cost_stats_out);

    auto recv_token_per_exp_cpu = recv_tokens_per_expert.to(ms::DeviceType::CPU);
    auto recv_token_per_exp_ptr = recv_token_per_exp_cpu.data_ptr<int64_t>();

    int token_cnt = 0;
    for (int local_e = 0; local_e < num_local_experts; ++local_e) {
        int current_tokens = static_cast<int>(recv_token_per_exp_ptr[local_e]);
        token_cnt = (expert_token_nums_type == 0) ? token_cnt + current_tokens : current_tokens;
        num_recv_tokens_per_expert_list.emplace_back(token_cnt);
    }

    return {expandx_out, dynamic_scales_out, recv_topk_idx, recv_topk_weights,
            num_recv_tokens_per_expert_list, expand_idx, ep_rank_token_cnt,
            offset_inner, token_server_idx, count_outer, expand_scales, std::optional<EventHandle>()};
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, std::optional<EventHandle>> Buffer::internode_combine(
    const ms::Tensor &x, const ms::Tensor &topk_idx, const std::optional<ms::Tensor> &topk_weights,
    const ms::Tensor &src_idx, const ms::Tensor &send_head, const ms::Tensor &offsetInner,
    const ms::Tensor &offsetOuter, const ms::Tensor &countOuter, const ms::Tensor &expand_scales)
{
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    ms::Tensor recv_x = x;
    ms::Tensor expert_ids = topk_idx.to(ms::DType::Int32);
    ms::Tensor expand_idx = src_idx;
    ms::Tensor ep_send_counts = send_head;
    auto device = x.device();

    int64_t hidden = static_cast<int>(recv_x.size(1));
    ms::Tensor tp_send_counts = ms::Tensor::empty({1}, ms::DType::Int32, device);
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t moe_expert_number = send_head.size(0);
    int64_t global_bs = static_cast<int64_t>(MAX_BATCH_SIZE * num_ranks);

    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    auto combined_x = ms::Tensor::empty({topk_idx.size(0), hidden}, x.dtype(), x.device());
    ms::Tensor expert_scales = ms::Tensor::empty({1}, ms::DType::Float32, x.device());
    std::optional<ms::Tensor> recv_topk_weights;
    std::optional<EventHandle> event;
    ms::Tensor x_active_mask, activation_scale, weight_scale, group_list;
    int64_t expert_shared_type = 0;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    ms::Tensor combine_send_cost_stats_out;

    EXEC_NPU_CMD(aclnnMoeDistributeCombineA2, recv_x, expert_ids, expand_idx, ep_send_counts, expert_scales,
                 tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales,
                 offsetInner, offsetOuter, countOuter, hcom_ep_name, num_ranks, rank, moe_expert_number,
                 hcom_ep_name, tp_world_size, tp_rankId, expert_shared_type, shared_expert_num, shared_expert_rank_num,
                 global_bs, out_dtype, comm_quant_mode, group_list_type, combined_x, combine_send_cost_stats_out);

    return {combined_x, recv_topk_weights, event};
}

std::tuple<ms::Tensor, std::optional<ms::Tensor>, ms::Tensor, ms::Tensor, ms::Tensor, std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const ms::Tensor &x, const ms::Tensor &topk_idx,
                             const std::optional<ms::Tensor> &cumulative_local_expert_recv_stats,
                             int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts, bool use_fp8,
                             bool round_scale, bool use_ue8m0, bool async, bool return_recv_hook)
{
    EP_HOST_ASSERT(low_latency_mode);
    ms::Tensor new_x = x;
    EP_HOST_ASSERT(num_max_dispatch_tokens_per_rank >= x.size(0));

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    int32_t num_local_experts = num_experts / (num_ranks - shared_expert_rank_num);
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;
    auto num_max_tokens = 0;
    if (rank < shared_expert_rank_num) {
        num_max_tokens = global_bs / shared_expert_rank_num;
        num_local_experts = 1;
    } else {
        num_max_tokens = global_bs * std::min(num_topk, num_local_experts);
    }
    auto max_size = std::max(num_tokens * num_topk, num_max_tokens * 128);

    auto device = x.device();
    auto packed_recv_x = ms::Tensor::empty({num_max_tokens, hidden}, use_fp8 ? ms::DType::Int8 : ms::DType::BFloat16, device);
    auto packed_recv_x_scales = ms::Tensor::empty({num_max_tokens}, ms::DType::Float32, device);
    auto expandIdx = ms::Tensor::empty({max_size}, ms::DType::Int32, device);

    int32_t server_num = num_ranks / LOCAL_RANK_SIZE;
    ms::Tensor ep_recv_count = ms::Tensor::empty({num_local_experts * num_ranks}, ms::DType::Int32, device);
    auto tp_recv_count = ms::Tensor::empty({1}, ms::DType::Int32, device);
    auto packed_recv_count = ms::Tensor::empty({num_local_experts}, ms::DType::Int64, device);
    ms::Tensor scales;
    ms::Tensor active_mask;
    int enable_neg_one = get_value_from_env("MOE_ENABLE_TOPK_NEG_ONE", 0);
    int64_t quant_mode = use_fp8 ? 2 : 0;
    int64_t tp_size = 1;
    int64_t tp_rank = 0;
    int64_t expert_shard_type = 0;
    int64_t expert_token_nums_type = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);
    char *comm_alg;

    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }
    char hcom_tp_name[HCOMM_NAME_LEN] = {0};
    std::optional<EventHandle> event;
    bool isLayered = false;

    if (soc_version == op::SocVersion::ASCEND910B) {
        const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
        const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
        if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr && strcmp(hcclIntraPcieEnable, "1") == 0 &&
            strcmp(hcclIntraRoceEnable, "0") == 0) {
            isLayered = true;
            int64_t recv_count_tensor_size = num_experts + 2 * global_bs * num_topk * server_num;
            ep_recv_count = ms::Tensor::empty({recv_count_tensor_size}, ms::DType::Int32, device);
        }
    }

    if (soc_version == op::SocVersion::ASCEND910B) {
        comm_alg = "fullmesh";
    } else {
        comm_alg = "fullmesh_v1";
    }

    if (enable_neg_one) {
        EP_HOST_ASSERT(isLayered == false);
        active_mask = (topk_idx >= 0).to(ms::DType::Bool);
    }

    EXEC_NPU_CMD(aclnnMoeDistributeDispatchV2, new_x, topk_idx, scales, active_mask, hcom_ep_name,
                 num_ranks, rank, num_experts, hcom_tp_name, tp_size, tp_rank, expert_shard_type, shared_expert_num,
                 shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_alg, packed_recv_x,
                 packed_recv_x_scales, expandIdx, packed_recv_count, ep_recv_count, tp_recv_count);

    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, expandIdx, ep_recv_count,
            event, std::function<void()>([] {})};
}

std::tuple<ms::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const ms::Tensor &x, const ms::Tensor &topk_idx, const ms::Tensor &topk_weights, const ms::Tensor &src_info,
    const ms::Tensor &layout_range, int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
    const ms::Tensor &packed_recv_count, bool zero_copy, bool async, bool return_recv_hook,
    const std::optional<ms::Tensor> &out)
{
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.dtype() == ms::DType::BFloat16);
    EP_HOST_ASSERT(num_max_dispatch_tokens_per_rank >= topk_idx.size(0));

    char hcom_ep_name[HCOMM_NAME_LEN];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }
    char hcom_tp_name[HCOMM_NAME_LEN] = {0};

    auto device = x.device();
    ms::Tensor expand_x = x;
    ms::Tensor expert_ids = topk_idx;
    ms::Tensor expand_idx = src_info;
    ms::Tensor ep_send_counts = layout_range;
    ms::Tensor expert_scales = topk_weights;
    ms::Tensor tp_send_counts = ms::Tensor::empty({1}, ms::DType::Int32, device);
    ms::Tensor x_active_mask, activation_scale, weight_scale, group_list, expand_scales;
    int enable_neg_one = get_value_from_env("MOE_ENABLE_TOPK_NEG_ONE", 0);
    int64_t tp_world_size = 1;
    int64_t tp_rankId = 0;
    int64_t expert_shared_type = 0;
    int64_t global_bs = num_max_dispatch_tokens_per_rank * num_ranks;
    int64_t out_dtype = 0;
    int64_t comm_quant_mode = 0;
    int64_t group_list_type = 0;
    bool isLayered = false;
    char *comm_alg;
    ms::Tensor combine_send_cost_stats_out;

    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));
    auto hidden = static_cast<int>(x.size(1));
    ms::Tensor shared_expert_x{nullptr};
    ms::Tensor combined_x = ms::Tensor::empty({num_combined_tokens, hidden}, x.dtype(), x.device());
    std::optional<EventHandle> event;

    if (soc_version == op::SocVersion::ASCEND910B) {
        const char *hcclIntraPcieEnable = getenv("HCCL_INTRA_PCIE_ENABLE");
        const char *hcclIntraRoceEnable = getenv("HCCL_INTRA_ROCE_ENABLE");
        if (hcclIntraPcieEnable != nullptr && hcclIntraRoceEnable != nullptr && strcmp(hcclIntraPcieEnable, "1") == 0 &&
            strcmp(hcclIntraRoceEnable, "0") == 0) {
            isLayered = true;
        }
    }

    if (soc_version == op::SocVersion::ASCEND910B) {
        comm_alg = "fullmesh";
    } else {
        comm_alg = "fullmesh_v1";
    }

    if (enable_neg_one) {
        EP_HOST_ASSERT(isLayered == false);
        x_active_mask = (expert_ids >= 0).to(ms::DType::Bool);
    }

    EXEC_NPU_CMD(aclnnMoeDistributeCombineV2, expand_x, expert_ids, expand_idx, ep_send_counts, expert_scales,
                 tp_send_counts, x_active_mask, activation_scale, weight_scale, group_list, expand_scales,
                 shared_expert_x, hcom_ep_name, num_ranks, rank, num_experts, hcom_tp_name, tp_world_size, tp_rankId,
                 expert_shared_type, shared_expert_num, shared_expert_rank_num, global_bs, out_dtype, comm_quant_mode,
                 group_list_type, comm_alg, combined_x, combine_send_cost_stats_out);

    return {combined_x, event, std::function<void()>([] {})};
}

std::vector<ms::Tensor> Buffer::fused_deep_moe(const ms::Tensor &x, const ms::Tensor &expert_ids,
                                               const ms::Tensor &gmm1_permuted_weight,
                                               const ms::Tensor &gmm1_permuted_weight_scale,
                                               const ms::Tensor &gmm2_weight, const ms::Tensor &gmm2_weight_scale,
                                               const ms::Tensor &expert_scales_optional,
                                               int64_t num_max_dispatch_tokens_per_rank, int64_t num_experts,
                                               int quant_mode)
{
    EP_HOST_ASSERT(expert_ids.dim() == 2);
    EP_HOST_ASSERT(expert_scales_optional.dim() == 2);

    char hcom_ep_name[128];
    if (!moe_all_to_all_group_name.empty()) {
        std::memcpy(hcom_ep_name, moe_all_to_all_group_name.data(), moe_all_to_all_group_name.size() + 1);
    } else {
        HCCL_CHECK(HcclGetCommName(ep_comm, hcom_ep_name));
    }

    int64_t global_bs = std::max(expert_ids.size(0), num_max_dispatch_tokens_per_rank) * num_ranks;

    auto x_shape = x.sizes();
    int h = x_shape[1];
    int bs = expert_ids.size(0);

    ms::Tensor output = ms::Tensor::empty({bs, h}, x.dtype(), x.device());

    bool is_shared_expert = (rank < shared_expert_rank_num);
    int64_t num_local_experts = is_shared_expert ? 1 : num_experts / (num_ranks - shared_expert_rank_num);
    ms::Tensor ep_recv_count = ms::Tensor::empty({num_local_experts * num_ranks}, expert_ids.dtype(), expert_ids.device());

    EXEC_NPU_CMD(aclnnFusedDeepMoe, x, expert_ids, gmm1_permuted_weight, gmm1_permuted_weight_scale,
                 gmm2_weight, gmm2_weight_scale, static_cast<const std::nullptr_t &>(nullptr),
                 expert_scales_optional, hcom_ep_name, num_ranks, rank, num_experts, shared_expert_num,
                 shared_expert_rank_num, quant_mode, global_bs, output, ep_recv_count);

    return {output, ep_recv_count};
}
}  // namespace deep_ep
