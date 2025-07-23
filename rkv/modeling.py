import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Callable
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
# from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
# from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .compression import (
    R1KV,
    SnapKV,
    StreamingLLM,
    H2O,
    AnalysisKV
)

KV_COMPRESSION_MAP = {
    "rkv": R1KV,
    "snapkv": SnapKV,
    "streamingllm": StreamingLLM,
    "h2o": H2O,
    "analysiskv": AnalysisKV
}

logger = logging.get_logger(__name__)

def LlamaAttention_init(
    self, config: LlamaConfig, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.num_key_value_heads = config.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta

    self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    # =============== New logic end =================

def LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            if self.config.update_kv is True:
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # if self.kv_cluster.num_comp == 1 and self.layer_idx<2:
    #     breakpoint()

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask[:,:key_states.shape[1]] if attention_mask is not None else attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def Qwen2Attention_init(
    self, config: Qwen2Config, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
    )
    self.k_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.v_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    # =============== New logic end =================

def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end ===============

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )
            if self.config.update_kv is True:
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    # flashattention2 kernel handles it automatically
    # repeat k/v heads if n_kv_heads < n_heads
    #key_states = repeat_kv(key_states, self.num_key_value_groups)
    #value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()


# Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # Reashape to the expected shape for Flash Attention
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask[:,:key_states.shape[1]] if attention_mask is not None else attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# def Qwen3Attention_init(
#     self, config: Qwen3Config, layer_idx: int, compression_config: dict
# ):
#         nn.Module.__init__(self)
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = self.head_dim**-0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = True

#         self.q_proj = nn.Linear(
#             config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.v_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.o_proj = nn.Linear(
#             config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
#         )
#         self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
#         self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
#         self.sliding_window = config.sliding_window
#         if not (
#             self.config.use_sliding_window
#             and getattr(self.config, "sliding_window", None) is not None
#             and self.layer_idx >= self.config.max_window_layers
#         ):
#             self.sliding_window = None

#         # =============== New logic start ===============
#         self.config.update(compression_config)
#         self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
#             **compression_config["method_config"]
#         )
#         # =============== New logic end =================

# def Qwen3Attention_forward(
#     self,
#     hidden_states: torch.Tensor,
#     position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#     attention_mask: Optional[torch.Tensor],
#     past_key_value: Optional[Cache] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     **kwargs: Unpack[FlashAttentionKwargs],
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     input_shape = hidden_states.shape[:-1]
#     hidden_shape = (*input_shape, -1, self.head_dim)

#     query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#     key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#     value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#     cos, sin = position_embeddings
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


#     if past_key_value is not None:
#         cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

#         # =============== Enable Query Cache ============
#         if not hasattr(past_key_value, "query_cache"):
#             past_key_value.query_cache = {}

#         if self.layer_idx not in past_key_value.query_cache:
#             # prefill stage
#             bsz, n_heads, _, head_dim = query_states.shape
#             past_key_value.query_cache[self.layer_idx] = torch.empty(
#                 bsz, n_heads, 0, head_dim
#             )
#             past_key_value.query_cache[self.layer_idx] = query_states[
#                 :, :, -self.config.method_config["window_size"] :, :
#             ]
#         else:
#             # Add current query to cache
#             past_key_value.query_cache[self.layer_idx] = torch.cat(
#                 (past_key_value.query_cache[self.layer_idx], query_states), dim=2
#             )  # [batch, n_q_heads, seq_len, head_dim]

#             # Keep only window_size most recent queries
#             window_size = self.config.method_config["window_size"]
#             if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
#                 past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
#                     self.layer_idx
#                 ][:, :, -window_size:, :]
#         # =============== Enable Query Cache end =========

#         # =============== decoding-time compression start ===============
#         cached_queries = past_key_value.query_cache[self.layer_idx]
#         if self.config.compression is None:
#             key_states_compress, value_states_compress = self.kv_cluster.update_kv(
#                 key_states,
#                 cached_queries,  # Use cached queries instead of current query
#                 value_states,
#             )

#             past_key_value.update(
#                 key_states_compress,
#                 value_states_compress,
#                 self.layer_idx,
#                 cache_kwargs,
#             )

#         elif self.config.compression is True:
#             key_states, value_states = past_key_value.update(
#                 key_states,
#                 value_states,
#                 self.layer_idx,
#                 cache_kwargs,
#             )

#             key_states_compress, value_states_compress = self.kv_cluster.update_kv(
#                 key_states,
#                 cached_queries,  # Use cached queries instead of current query
#                 value_states,
#             )

#             past_key_value.key_cache[self.layer_idx] = key_states_compress
#             past_key_value.value_cache[self.layer_idx] = value_states_compress
#         else:
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )
#         # =============== decoding-time compression end ===============

#     attention_interface: Callable = eager_attention_forward
#     if self.config._attn_implementation != "eager":
#         if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
#             logger.warning_once(
#                 "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
#                 'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#             )
#         else:
#             attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

#     attn_output, attn_weights = attention_interface(
#         self,
#         query_states,
#         key_states,
#         value_states,
#         attention_mask,
#         dropout=0.0 if not self.training else self.attention_dropout,
#         scaling=self.scaling,
#         sliding_window=self.sliding_window,  # diff with Llama
#         **kwargs,
#     )

#     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#     attn_output = self.o_proj(attn_output)
#     return attn_output, attn_weights

def CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # sample-level statistics
    if input_ids.shape[-1] > 1:
        if self.config.compression_content == "think":
            self.after_think = False

    if not hasattr(self, "length"):
        self.length = input_ids.shape[1]
    else:
        self.length += input_ids.shape[1]

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

    # =============== Step-level Compression logic start ===============
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and self.after_think == False:
        self.after_think = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = self.length % self.config.divide_length == 0
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and self.after_think == True:
        is_newline = False

    # Set compression flag for all layers at once
    for layer in self.model.layers:
        layer.self_attn.config.compression = is_newline
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )