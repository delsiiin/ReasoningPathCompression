import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from .qwen2_vanilla import (
    apply_rotary_pos_emb,
    repeat_kv,
    Qwen2Attention
)
from transformers.utils import (
    logging,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from rpc.rpc_utils import init_rpc

import math

logger = logging.get_logger(__name__)

class Qwen2RPCAttention(Qwen2Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_rpc(self)
        self.verbose = False

        self.row_sum_accu = None
        self.col_sum_accu = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
       
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            if q_len != 1:
                # save cache length for prefill
                self.kv_cluster.prompt_len = key_states.size()[-2]
                self.kv_cluster.num_comp = 0

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        if q_len == 1:

            if self.row_sum_accu is None:
                self.row_sum_accu = torch.sum(attn_weights[..., 1, : self.kv_cluster.prompt_len], dim=-1)
                self.row_sum_accu = self.row_sum_accu.mean(dim=1)
            else:
                cur_row_sum = torch.sum(attn_weights[..., 1, : self.kv_cluster.prompt_len], dim=-1)
                cur_row_sum = cur_row_sum.mean(dim=1)
                self.row_sum_accu = torch.cat([self.row_sum_accu, cur_row_sum], dim=-1)
            
            if self.col_sum_accu is None:
                self.col_sum_accu = attn_weights[..., 1, self.kv_cluster.prompt_len :]
                self.col_sum_accu = self.col_sum_accu.mean(dim=1)
            else:
                prev_col_sum = F.pad(self.col_sum_accu, pad=(0, q_len), mode='constant', value=0)
                prev_col_sum = prev_col_sum.mean(dim=1)
                self.col_sum_accu = attn_weights[..., 1, self.kv_cluster.prompt_len :] + prev_col_sum
                
            # cannot use 'past_key_value.get_seq_length'
            target_length = past_key_value.key_cache[self.layer_idx].size()[-2] - self.kv_cluster.prompt_len

            if target_length == self.kv_cluster.budget_cot - 1:

                # support gqa
                if self.kv_cluster.aggregation == 'none':
                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)
                
                key_states_compress, value_states_compress = self.kv_cluster.compress_kv(key_states, value_states, self.row_sum_accu, self.col_sum_accu)
                
                # replace with compressed cache
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
                
                self.kv_cluster.num_comp += 1
            

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value