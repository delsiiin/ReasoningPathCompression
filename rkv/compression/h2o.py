import torch
import torch.nn as nn
from . import compute_attention_scores


class H2O:
    def __init__(
        self,
        budget=128,
        window_size=8,
        record_kept_token_indices=False,
        layer_idx=None,
        model_config=None,
        model_type=None,
        mode=None,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = 1

        self.layer_idx = layer_idx
        self.model_config = model_config
        self.model_type = model_type
        self.mode = mode

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
    ):
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.budget:
            return key_states, value_states
        else:
            query_states = query_states[:, :, -1:, :]
            attn_weights = compute_attention_scores(query_states, key_states)

            attn_weights_sum = (
                nn.functional.softmax(
                    attn_weights[:, :, : -self.window_size],
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            )

            # shape: (bsz, num_kv_heads, budget - window_size)
            if self.mode == "record_indices":
                indices = attn_weights_sum.topk(
                self.model_config.observation_topk, dim=-1
            ).indices
            else:
                indices = attn_weights_sum.topk(
                    self.budget - self.window_size, dim=-1
                ).indices

            

            if self.mode == "record_indices":

                # Create directory if it doesn't exist
                folder_path = f'/home/yangx/ReasoningPathCompression/observation/topk_indices/{self.model_type}/h2o'
                import os 
                os.makedirs(folder_path, exist_ok=True)

                # Save indices to file
                save_path = f'{folder_path}/topk_indices_layer_{self.layer_idx}_observe_{self.model_config.observation_length}_top_{self.model_config.observation_topk}.pt'
                torch.save(indices, save_path)

            #####################################################
            ###### Store evicted token indices start ############
            #####################################################
            # shape: (num_kv_heads, budget - window_size)
            if self.record_kept_token_indices:
                indices_cl = indices.clone().squeeze(0).to("cpu")
                recent_window_indices = torch.arange(
                    kv_cache_len - self.window_size, kv_cache_len, device="cpu"
                ).expand(indices_cl.shape[0], -1)
                cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)

                if self.evicted_token_num > 0:
                    prev_indices = self.kept_token_indices[-1]
                    mask = cur_indices < self.budget

                    for i in range(cur_indices.shape[0]):
                        positions = torch.where(mask[i])[0]

                        # For each position, get the value and use it as an index into prev_indices
                        for pos in positions:
                            val = cur_indices[i, pos].item()
                            cur_indices[i, pos] = prev_indices[i, val]

                    # For values >= self.budget, add the evicted token count
                    cur_indices[~mask] += self.evicted_token_num

                self.kept_token_indices.append(cur_indices)
                self.evicted_token_num += kv_cache_len - self.budget
            ######################################################

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states