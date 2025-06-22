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
    Qwen2Attention,
    Qwen2Model,
)
from transformers.utils import (
    logging,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from rpc.rpc_utils_window import init_rpc

import math

logger = logging.get_logger(__name__)

def restore_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 (batch, num_attention_heads, seqlen, head_dim) 的张量还原为 
    (batch, num_key_value_heads, seqlen, head_dim)，其中 num_key_value_heads = num_attention_heads // n_rep
    """
    batch, num_attention_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    assert num_attention_heads % n_rep == 0, "num_attention_heads must be divisible by n_rep"
    num_key_value_heads = num_attention_heads // n_rep
    return hidden_states.view(batch, num_key_value_heads, n_rep, seqlen, head_dim)[:, :, 0, :, :]

class Qwen2RPCAttention(Qwen2Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_rpc(self)
        self.verbose = False

        self.layer_budget_importance = None

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
                self.row_sum_accu = None
                self.col_sum_accu = None
    
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
        attn_output = torch.matmul(attn_weights, value_states)

        if q_len == 1:

            if self.row_sum_accu is None:
                self.row_sum_accu = torch.mean(attn_weights[..., 0, : self.kv_cluster.prompt_len], dim=-1)
                self.row_sum_accu = self.row_sum_accu.unsqueeze(-1)
            else:
                cur_row_sum = torch.mean(attn_weights[..., 0, : self.kv_cluster.prompt_len], dim=-1)
                cur_row_sum = cur_row_sum.unsqueeze(-1)
                self.row_sum_accu = torch.cat([self.row_sum_accu, cur_row_sum], dim=-1)
                
            # cannot use 'past_key_value.get_seq_length'
            target_length = past_key_value.key_cache[self.layer_idx].size()[-2] - self.kv_cluster.prompt_len - (self.kv_cluster.num_comp * self.kv_cluster.cp_cot) 
                

            if target_length >= self.kv_cluster.budget_cot - 1 - self.kv_cluster.R and target_length < self.kv_cluster.budget_cot - 1:

                if self.col_sum_accu is None:
                    self.col_sum_accu = attn_weights[..., 0, self.kv_cluster.prompt_len : self.kv_cluster.prompt_len + (self.kv_cluster.num_comp * self.kv_cluster.cp_cot) + self.kv_cluster.budget_cot - 1 - self.kv_cluster.R]
                else:
                    prev_col_sum = self.col_sum_accu
                    # if self.kv_cluster.num_comp == 1:
                    #     print(prev_col_sum.shape, attn_weights[..., 0, self.kv_cluster.prompt_len :].shape)
                    self.col_sum_accu = attn_weights[..., 0, self.kv_cluster.prompt_len : self.kv_cluster.prompt_len + (self.kv_cluster.num_comp * self.kv_cluster.cp_cot) + self.kv_cluster.budget_cot - 1 - self.kv_cluster.R] + prev_col_sum

            if target_length >= self.kv_cluster.budget_cot - 1:

                key_states = restore_kv(key_states, self.num_key_value_groups)
                value_states = restore_kv(value_states, self.num_key_value_groups)
                
                key_states_compress, value_states_compress, self.row_sum_accu = self.kv_cluster.compress_kv(key_states, value_states, self.row_sum_accu, self.col_sum_accu, self.num_key_value_groups)

                self.col_sum_accu = None
                
                # replace with compressed cache
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
                
                self.kv_cluster.num_comp += 1
        
        else:
            
            if self.config.mode == "dynamic_layer_budget":
                self.layer_budget_importance = attn_weights[..., :self.kv_cluster.prompt_len, :self.kv_cluster.prompt_len].sum(dim=-2).var(dim=-1).mean(dim=0) # refer to D2O (针对单样本复制多个打batch，不支持不同输入打batch)

                var_max = self.layer_budget_importance.max()
                var_min = self.layer_budget_importance.min()

                self.layer_budget_importance = (self.layer_budget_importance - var_min) / (var_max - var_min + 1e-6)

                self.layer_budget_importance = self.layer_budget_importance.mean(1) 
            else:
                pass



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
    

class Qwen2RPCModel(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if self.config.mode == "dynamic_layer_budget":
            layer_budget_allocator = []

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            if hidden_states.shape[-2] > 1 and self.config.mode == "dynamic_layer_budget":
                layer_budget_allocator.append(decoder_layer.self_attn.layer_budget_importance)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if hidden_states.shape[-2] > 1 and self.config.mode == "dynamic_layer_budget":

            layer_budget_allocator = torch.stack(layer_budget_allocator)

            weights = torch.softmax(-layer_budget_allocator, dim=0)

            layer_budgets = weights

            layer_budgets = torch.clamp(layer_budgets, min=0.01, max=1.0)

            for idx, decoder_layer in enumerate(self.layers):
                decoder_layer.self_attn.kv_cluster.cp_ratio = layer_budgets[idx]
                decoder_layer.self_attn.kv_cluster.cp_cot = layer_budgets[idx] * decoder_layer.self_attn.kv_cluster.budget_cot
                decoder_layer.self_attn.kv_cluster.cp_ans = layer_budgets[idx] * decoder_layer.self_attn.kv_cluster.budget_ans

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    