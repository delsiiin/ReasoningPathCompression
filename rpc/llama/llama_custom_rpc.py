import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    LlamaAttention,
    LlamaModel
)
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs
from .llama_config import LlamaConfig
from transformers.utils import (
    logging,
)
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from rpc.rpc_utils_rpc import init_rpc

logger = logging.get_logger(__name__)

def Llama_RPC_init(self, config: LlamaConfig, layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
    )
    self.k_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.v_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
    )
    init_rpc(self)
    self.verbose = False


def Llama_RPC_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        q_len = hidden_states.size(1)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            
        if past_key_value is not None:
        
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    
            # NOTE: decoding update
            if q_len == 1:
                
                # cannot use 'past_key_value.get_seq_length'
                target_length = past_key_value.key_cache[self.layer_idx].size()[-2] - self.kv_cluster.prompt_len - (self.kv_cluster.num_comp * self.kv_cluster.T) - self.kv_cluster.R
                
                if self.kv_cluster.selectors == 'recent' and target_length > self.kv_cluster.P - self.kv_cluster.R:
                    # cache recent query states as selectors
                    self.kv_cluster.cache_recent(query_states)
                
                if target_length == self.kv_cluster.P - 1:

                    # support gqa
                    if self.kv_cluster.aggregation == 'none':
                        key_states = repeat_kv(key_states, self.num_key_value_groups)
                        value_states = repeat_kv(value_states, self.num_key_value_groups)
                    
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            
                    key_states_compress, value_states_compress = self.kv_cluster.compress_kv(key_states, value_states, query_states)
                    
                    # replace with compressed cache
                    past_key_value.key_cache[self.layer_idx] = key_states_compress
                    past_key_value.value_cache[self.layer_idx] = value_states_compress
                    
                    self.kv_cluster.num_comp += 1

                    if self.verbose:
                        if self.layer_idx == 31:
                            print(f"\n[RECOMPRESS] Num Recompression: {self.num_recompression}, Current Seqlen: {past_key_value.get_seq_length()}\n")
                else:
                    if self.kv_cluster.aggregation == 'none':
                        key_states = repeat_kv(key_states, self.num_key_value_groups)
                        value_states = repeat_kv(value_states, self.num_key_value_groups)
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            else:
                past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                # save cache length for prefill
                self.kv_cluster.prompt_len = key_states.size()[-2]
                self.kv_cluster.num_comp = 0

                if self.verbose:
                    if self.layer_idx == 31:
                        print(f"[Prefill] Num Recompression: {self.num_recompression}, Prompt Len: {past_key_value.get_seq_length()}")
                

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights