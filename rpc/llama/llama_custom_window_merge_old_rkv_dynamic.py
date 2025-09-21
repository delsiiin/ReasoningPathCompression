import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from .llama_vanilla import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaAttention,
    LlamaModel,
    LlamaForCausalLM,
    _prepare_4d_causal_attention_mask_with_cache_position
)
from transformers.utils import (
    logging,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from rpc.rpc_utils_window_merge_old_rkv_dynamic import init_rpc

import math

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)

def calculate_entropy(attention_scores):
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
    entropy= entropy.to(dtype=torch.float32)
    return entropy

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


class LlamaRPCAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_rpc(self)
        self.verbose = False

        self.cache_mode = "compression"  # options: vanilla, compression

        self.layer_budget_importance = None

        self.row_sum_accu = None
        self.col_sum_accu = None


    def forward(
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
                self.kv_cluster.threshold = None
                self.row_sum_accu = None
                self.col_sum_accu = None
                self.cache_mode = "compression"
                self.question_cache = query_states
                if hasattr(self, "_cot_done_printed"):
                    delattr(self, "_cot_done_printed")
    
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        if self.cache_mode == "vanilla":

            if self.layer_idx == 0 and not hasattr(self, "_cot_done_printed"):
                print("\033[33mCoT Done!!! Start Answering...\033[0m")
                self._cot_done_printed = True

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        elif self.cache_mode == "compression":

            if q_len == 1:
                    
                # cannot use 'past_key_value.get_seq_length'
                target_length = past_key_value.key_cache[self.layer_idx].size()[-2] - self.kv_cluster.prompt_len - self.kv_cluster.budget_cot 
                    
                if target_length > self.kv_cluster.buffer_cot - self.kv_cluster.R:
                    
                    self.kv_cluster.cache_recent(query_states)

                    # partial_attn_weights = nn.functional.softmax(attn_weights[..., 0, self.kv_cluster.prompt_len:self.kv_cluster.prompt_len + self.kv_cluster.budget_cot + self.kv_cluster.buffer_cot - self.kv_cluster.R], dim=-1, dtype=torch.float32).to(query_states.dtype)

                    # try:
                    #     if self.col_sum_accu is None:
                    #         self.col_sum_accu = partial_attn_weights
                    #     else:
                    #         prev_col_sum = self.col_sum_accu
                    #         if prev_col_sum.shape[-1] != partial_attn_weights.shape[-1]:
                    #             self.col_sum_accu = partial_attn_weights
                    #         else:
                    #             self.col_sum_accu = prev_col_sum + partial_attn_weights
                    # except Exception as e:
                    #     print(f"Error when updating col_sum_accu, target_length={target_length}, {self.layer_budget_importance}")
                    #     raise

                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_states)

                # if self.row_sum_accu is None:
                #     self.row_sum_accu = torch.sum(attn_weights[..., 0, : self.kv_cluster.prompt_len], dim=-1)
                #     self.row_sum_accu = self.row_sum_accu.unsqueeze(-1)
                # else:
                #     cur_row_sum = torch.sum(attn_weights[..., 0, : self.kv_cluster.prompt_len], dim=-1)
                #     cur_row_sum = cur_row_sum.unsqueeze(-1)
                #     self.row_sum_accu = torch.cat([self.row_sum_accu, cur_row_sum], dim=-1)
                
                if target_length == self.kv_cluster.buffer_cot:

                    if "dynamic" in self.config.mode:
                        
                        self.layer_budget_importance = calculate_entropy(attn_weights[..., self.kv_cluster.prompt_len:])
                        # if not self.cal_importance and self.layer_budget_importance is None:
                        #     self.layer_budget_importance = calculate_entropy(attn_weights) # refer to D2O (针对单样本复制多个打batch，不支持不同输入打batch)
                        #     self.cal_importance = True
                        # else:
                        #     self.cal_importance = True
                    
                    question = self.question_cache

                    bsz, num_heads, ques_len, head_dim = question.shape
            
                    ques_attn_weights = torch.matmul(question, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                    # no need to deal with attention mask

                    ques_attn_weights = nn.functional.softmax(ques_attn_weights[:, :, :, self.kv_cluster.prompt_len:self.kv_cluster.prompt_len + self.kv_cluster.budget_cot + self.kv_cluster.buffer_cot - self.kv_cluster.R], dim=-1, dtype=torch.float32).to(question.dtype)
                    ques_attn_weights_sum = ques_attn_weights.sum(dim = -2)

                    key_states = restore_kv(key_states, self.num_key_value_groups)
                    value_states = restore_kv(value_states, self.num_key_value_groups)
                    
                    key_states_compress, value_states_compress= self.kv_cluster.compress_kv(key_states, value_states, ques_attn_weights_sum, self.num_key_value_groups)

                    # self.col_sum_accu = None
                    
                    # replace with compressed cache
                    past_key_value.key_cache[self.layer_idx] = key_states_compress
                    past_key_value.value_cache[self.layer_idx] = value_states_compress
                    
                    self.kv_cluster.num_comp += 1

                    # print(f"\033[32mnum_comp: {self.kv_cluster.num_comp}, layer: {self.layer_idx}\033[0m")
                    if self.kv_cluster.cp_cot < self.kv_cluster.budget_cot:
                        self.kv_cluster.budget_cot = self.kv_cluster.cp_cot
            
            else:
                
                # upcast attention to fp32
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, value_states)

                if self.layer_idx == 0:
                    print("\033[33mInput Done!!! Start Compressing CoT...\033[0m")

                if "dynamic" in self.config.mode:
                    self.layer_budget_importance = calculate_entropy(attn_weights) # refer to D2O (针对单样本复制多个打batch，不支持不同输入打batch)
                    # self.cal_importance = True


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


class LlamaRPCModel(LlamaModel):
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

        if "dynamic" in self.config.mode and hidden_states.shape[-2] > 1:
            self.layer_budget_allocator = {f"layer_{i}": None for i in range(len(self.layers))}

        for l_idx, decoder_layer in enumerate(self.layers):
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

            if "dynamic" in self.config.mode and decoder_layer.self_attn.layer_budget_importance is not None:
                # if decoder_layer.self_attn.cal_importance:
                self.layer_budget_allocator.update({f"layer_{l_idx}": decoder_layer.self_attn.layer_budget_importance})

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if "dynamic" in self.config.mode and all(v is not None for v in self.layer_budget_allocator.values()):

            if hidden_states.shape[-2] > 1:
                if not hasattr(self, "total_kv_size"):
                    self.total_kv_size = self.layers[0].self_attn.kv_cluster.budget_cot * len(self.layers)
                else:
                    pass

                layer_budgets = {f"layer_{i}": None for i in range(len(self.layers))}

                for key, importance in self.layer_budget_allocator.items():
                    layer_budgets.update({key: int(importance/sum(self.layer_budget_allocator.values())*self.total_kv_size)})
            
                # layer_budgets = adjust_budgets(layer_budgets, self.total_kv_size, seq_len-self.window_size,  self.num_layers)

                for idx, decoder_layer in enumerate(self.layers):
                    decoder_layer.self_attn.kv_cluster.budget_cot = layer_budgets[f"layer_{idx}"]
                    decoder_layer.self_attn.kv_cluster.cp_cot = layer_budgets[f"layer_{idx}"]
                    # decoder_layer.self_attn.cal_importance = False
                    decoder_layer.self_attn.layer_budget_importance = None

            else:
                
                layer_budgets = {f"layer_{i}": None for i in range(len(self.layers))}

                for key, importance in self.layer_budget_allocator.items():
                    layer_budgets.update({key: int(importance/sum(self.layer_budget_allocator.values())*self.total_kv_size)})
                
                # layer_budgets = adjust_budgets(layer_budgets, self.total_kv_size, seq_len-self.window_size,  self.num_layers)

                for idx, decoder_layer in enumerate(self.layers):
                    if decoder_layer.self_attn.kv_cluster.budget_cot <= layer_budgets[f"layer_{idx}"]:
                        decoder_layer.self_attn.kv_cluster.budget_cot = layer_budgets[f"layer_{idx}"]
                        decoder_layer.self_attn.kv_cluster.cp_cot = layer_budgets[f"layer_{idx}"]
                    else:
                        decoder_layer.self_attn.kv_cluster.cp_cot = layer_budgets[f"layer_{idx}"]

                    # decoder_layer.self_attn.cal_importance = False
                    decoder_layer.self_attn.layer_budget_importance = None

            # print(f"\033[32mLayer budgets: {layer_budgets}\033[0m")

            self.layer_budget_allocator = {f"layer_{i}": None for i in range(len(self.layers))}


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
    
class LlamaRPCForCausalLM(LlamaForCausalLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        ###################################################################
        if 128014 in model_inputs["input_ids"]: # check for </> token
            
            for layer in self.model.layers:
                layer.self_attn.cache_mode = "vanilla"
        ###################################################################

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def forward(
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()


        # =============== Step-level Compression logic start ===============
        # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
        predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

        self.CoT_done = (
            predicted_token_ids[0].cpu().item() in self.CoT_done_token_ids
        )

        if self.config.divide_method == "new_line":
            is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
        elif self.config.divide_method == "step_length":
            is_newline = self.length % self.config.divide_length == 0
        else:
            raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

        # Set compression flag for all layers at once
        if self.CoT_done == True:
            for layer in self.model.layers:
                layer.self_attn.cache_mode = "vanilla"
        elif is_newline:
            for layer in self.model.layers:
                layer.self_attn.cache_mode = "compression"
        else:
            for layer in self.model.layers:
                layer.self_attn.cache_mode = "compression"
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