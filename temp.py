    @staticmethod
    def backward_deepep(ctx, *args):
        """
        Backward pass for MoE layer using DeepEP dispatch/combine.
        
        DeepEP's dispatch and combine are pure communication operators.
        The backward flow is:
        1. grad_output -> combine (distribute gradients to experts)
        2. experts backward
        3. dispatch (collect gradients from experts) -> grad_hidden_states
        
        Args:
            ctx: Context with saved tensors
            args[0]: Gradient from output (grad_output)
            args[1]: Gradient from bias (not used)
        """
        # Unpack saved tensors according to forward order:
        # 0: scores (route_graph)
        # 1: scores_detach (detach_scores)
        # 2: indices (routing_map)
        # 3-7: DeepEP tensors (probs_f16, res_send_token_small, tokens_per_expert, new_topk_idx, ...)
        #       saved in alltoall_token_permutation_new_deepep
        # 8: expert_output (experts_graph)
        # 9: expert_inputs_detach[0] (detached inputs for experts)
        # 10: l_aux (or None)
        # 11: l_aux_detach (or None)
        # 12: hidden_states (detach_input)
        # 13: share_experts_output (share_experts_graph)
        # 14: global_input_tokens_local_experts_indices (or None)
        global_args = get_args()

        output_splits = ctx.output_splits
        input_splits = ctx.input_splits
        router_topk = ctx.router_topk
        n_shared_experts = ctx.n_shared_experts
        moe_zero_memory = ctx.moe_zero_memory
        moe_experts_pipeline_degree = ctx.moe_experts_pipeline_degree
        moe_tp_extend_ep = global_args.moe_tp_extend_ep
        moe_hierarchical_alltoallv = global_args.moe_hierarchical_alltoallv
        shared_expert_gate = ctx.shared_expert_gate
        input_splits_tp_ep = ctx.input_splits_tp_ep

        saved = ctx.saved_tensors
        route_graph = saved[0]
        detach_scores = saved[1]
        indices = saved[2]
        
        # DeepEP specific tensors (saved in alltoall_token_permutation_new_deepep)
        probs_f16 = saved[3]
        res_send_token_small = saved[4]
        tokens_per_expert = saved[5]
        new_topk_idx = saved[6]
        
        experts_graph = saved[7]
        expert_inputs_detach = saved[8]
        l_aux_graph = saved[9]
        l_aux_detach = saved[10]
        detach_input = saved[11]
        share_experts_graph = saved[12]
        
        # (route_graph, detach_scores,
        #  indices, probs_f16,
        #  res_send_token_small, 
        #  experts_graph, expert_inputs_detach,
        #  l_aux_graph, l_aux_detach,tokens_per_expert,
        #  new_topk_idx,
        #  detach_input, share_experts_graph,
        #  global_input_tokens_local_experts_indices
        #  ) = ctx.saved_tensors

        print(f"debug backward_deepep get saved", flush=True)
        
        backward_ag_shared = None
        backward_ag_shared_handle = None
        if n_shared_experts:
            if get_tensor_model_parallel_world_size() > 1 and not shared_expert_gate:
                _, backward_ag_shared, backward_ag_shared_handle = async_all_gather(
                    args[0], get_tensor_model_parallel_group()
                )
            else:
                backward_ag_shared = args[0]
                backward_ag_shared_handle = None
        ep_group = parallel_state.get_expert_model_parallel_group()
        if moe_tp_extend_ep:
            ep_group = parallel_state.get_tensor_and_expert_parallel_group()
        deepep_buffer = DeepEPBuffer.get_deepep_buffer(ep_group)
        print(f"debug backward_deepep get deepep_buffer", flush=True)
        
        # Step 1: Backward through combine (distribute gradients to experts)
        grad_output = args[0]
        # Reshape grad_output to match combine input shape
        grad_output_reshaped = grad_output.view(-1, grad_output.shape[-1])
        # Use combine in reverse to distribute gradients to experts
        grad_expert_output = deepep_buffer.combine(
            grad_output_reshaped, 
            indices, 
            None,  # src_idx - not needed for backward
            None,  # put_offset - not needed for backward
            res_send_token_small, 
            probs_f16, 
            new_topk_idx
        )
        print(f"debug backward_deepep combine", flush=True)
        
        # Step 2: Backward through experts
        backward_func(experts_graph, grad_expert_output)
        print(f"debug backward_deepep backward_func", flush=True)
        
        # Step 3: Backward through dispatch (collect gradients from experts)
        # Get gradient from expert input (from detached inputs)
        grad_expert_input = expert_inputs_detach.grad
        if grad_expert_input is not None:
            # Use dispatch in reverse to collect gradients
            grad_hidden_states = deepep_buffer.dispatch(
                grad_expert_input, 
                indices, 
                tokens_per_expert, 
                res_send_token_small, 
                None,  # topk_weights - not needed for backward
                new_topk_idx
            )[0]  # Get the first return value (expandx_out)
            
            # Reshape back to original input shape
            grad_hidden_states = grad_hidden_states.view(detach_input.shape)
            # Set the gradient directly
            detach_input.grad = grad_hidden_states
        
        # Handle shared experts backward
        if n_shared_experts and share_experts_graph is not None:
            if backward_ag_shared_handle is not None:
                backward_ag_shared_handle.wait()
            share_experts_graph.backward(backward_ag_shared)
            share_experts_graph = None
            if backward_ag_shared_handle is not None:
                backward_ag_shared.untyped_storage().resize_(0)
        
        if l_aux_graph is not None:
            l_aux_graph.backward(l_aux_detach.grad, retain_graph=True)

        route_graph = None
        
        grad_output = detach_input.grad
        
        return grad_output, None, detach_scores.grad, None
