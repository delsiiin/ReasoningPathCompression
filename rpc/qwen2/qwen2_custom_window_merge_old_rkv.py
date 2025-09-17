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
    Qwen2ForCausalLM,
    _prepare_4d_causal_attention_mask_with_cache_position
)
from transformers.utils import (
    logging,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from rpc.rpc_utils_window_merge_old_rkv import init_rpc

import math
import numpy as np

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

class Qwen2RPCAttention(Qwen2Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_rpc(self)
        self.verbose = False

        self.cache_mode = "vanilla"  # options: vanilla, compression

        self.layer_budget_importance = None

        self.row_sum_accu = None
        self.col_sum_accu = None

        self.question_cache = None
        
        self.step_start_indices = [0]
        self.is_new_step = 1

    def cal_similarity(
        self,
        key_states,
        threshold=0.5,
        retain_ratio=0.2,
        retain_direction="last",
    ):
        k = key_states[0]
        num_heads = k.shape[0]

        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

        for h in range(num_heads):
            similarity_cos[h].fill_diagonal_(0.0)

        # shape: [num_heads, seq_len, seq_len]
        similarity_mask = similarity_cos > threshold

        seq_len = similarity_mask.size(-1)
        k = int(seq_len * retain_ratio)

        indices = torch.where(
            similarity_mask,
            torch.arange(similarity_mask.size(-1), device=similarity_mask.device),
            torch.zeros_like(similarity_mask, dtype=torch.long),
        )

        # find the last True index in each row
        if retain_direction == "last":
            similarity_retain = torch.max(indices, dim=-1)[0]

        # find the first True index in each row
        elif retain_direction == "first":
            similarity_retain = torch.min(indices, dim=-1)[0]

        # keep the last_percent% elements
        elif retain_direction == "last_percent":
            similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]

        # keep the first_percent% elements
        elif retain_direction == "first_percent":
            similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]

        # create indices for zeroing
        batch_idx = (
            torch.arange(num_heads).unsqueeze(1).repeat(1, similarity_retain.size(1))
        )
        seq_idx = torch.arange(similarity_retain.size(1)).unsqueeze(0).repeat(num_heads, 1)



        # print("cos shape:", similarity_cos.shape)  # 期望 [B, S, S?] 之类
        # print("batch_idx:", batch_idx, type(batch_idx))
        # print("seq_idx:", seq_idx, type(seq_idx))
        # print("retain shape/dtype/dev:", getattr(similarity_retain, "shape", None),
        #     getattr(similarity_retain, "dtype", None), getattr(similarity_retain, "device", None))

        # # 1) 保证 batch_idx / seq_idx 是标量（而非向量索引）
        # if torch.is_tensor(batch_idx):
        #     assert batch_idx.ndim == 0, "batch_idx 应为标量"
        # if torch.is_tensor(seq_idx):
        #     assert seq_idx.ndim == 0, "seq_idx 应为标量"

        # # 2) 索引类型与范围
        # assert similarity_retain.dtype in (torch.long, torch.int64) or similarity_retain.dtype == torch.bool
        # if similarity_retain.dtype != torch.bool:
        #     mn = int(similarity_retain.min().item())
        #     mx = int(similarity_retain.max().item())
        #     print("retain min/max:", mn, mx)
        #     assert mn >= 0, "存在负索引（如 -1）"
        #     assert mx < similarity_cos.size(2), "存在越界索引"

        # # 3) 设备一致
        # assert similarity_cos.device == (similarity_retain.device if torch.is_tensor(similarity_retain) else similarity_cos.device)




        # zero the specified positions in similarity_cos
        similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

        return similarity_cos.mean(dim=1).softmax(dim=-1)
   
    

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
                self.kv_cluster.threshold = None
                self.row_sum_accu = None
                self.col_sum_accu = None
                self.layer_budget_importance = None
                self.cache_mode = "vanilla"
                self.question_cache = query_states
                self.step_start_indices = [0]
                self.is_new_step = 1
                if hasattr(self, "_cot_done_printed"):
                    delattr(self, "_cot_done_printed")
    
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # if self.cache_mode == "vanilla":

        #     if self.layer_idx == 0 and not hasattr(self, "_cot_done_printed"):
        #         print("\033[33mCoT Done!!! Start Answering...\033[0m")
        #         self._cot_done_printed = True

        #     # upcast attention to fp32
        #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #     attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        #     attn_output = torch.matmul(attn_weights, value_states)

        if q_len == 1:

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if self.is_new_step == len(self.step_start_indices):
                self.kv_cluster.cache_recent(query_states)
            else:
                self.is_new_step += 1
                selectors = self.kv_cluster.cached_recent
                self.kv_cluster.cached_recent = None
                self.kv_cluster.cache_recent(query_states)

            # cannot use 'past_key_value.get_seq_length'
            target_length = past_key_value.key_cache[self.layer_idx].size()[-2] - self.kv_cluster.prompt_len 
                
            if target_length >= self.kv_cluster.budget_cot and self.cache_mode == "compression":
                
                question = self.question_cache

                bsz, num_heads, ques_len, head_dim = question.shape

        
                ques_attn_weights = torch.matmul(question, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                # no need to deal with attention mask

                ques_attn_weights = nn.functional.softmax(ques_attn_weights[:, :, :, self.kv_cluster.prompt_len:], dim=-1, dtype=torch.float32).to(question.dtype)
                ques_attn_weights_sum = ques_attn_weights.sum(dim = -2)

                key_states = restore_kv(key_states, self.num_key_value_groups)
                value_states = restore_kv(value_states, self.num_key_value_groups)
                
                key_states_compress, value_states_compress= self.kv_cluster.compress_kv(key_states, value_states, ques_attn_weights_sum, self.col_sum_accu, self.num_key_value_groups, self.step_start_indices, selectors)

                if key_states_compress.size(-2) - self.kv_cluster.prompt_len >= self.kv_cluster.budget_cot:

                    print(key_states_compress, "key_states_compress before second compression")
                    
                    similarity_cos = self.cal_similarity(
                        key_states_compress[..., self.kv_cluster.prompt_len:, :],
                        retain_ratio=0.2,
                        retain_direction="last",
                    )[..., :-selectors.shape[-2]]
                    print(similarity_cos.shape, "similarity_cos")
                    # 选择similarity_cos中最小的budget_cot个位置
                    _, min_indices = torch.topk(similarity_cos, k=self.kv_cluster.budget_cot - selectors.shape[-2], dim=-1, largest=False)
                    
                    # 对索引进行排序以保持原始顺序
                    min_indices_sorted, _ = torch.sort(min_indices, dim=-1)
                    print(min_indices_sorted.shape, "min_indices_sorted")
                    
                    # 添加prompt部分的索引
                    prompt_indices = torch.arange(self.kv_cluster.prompt_len, device=key_states_compress.device).unsqueeze(0).unsqueeze(0).expand(bsz, key_states_compress.size(1), -1)
                    
                    # 将压缩部分的索引调整到正确的位置（加上prompt_len偏移）
                    compress_indices = min_indices_sorted + self.kv_cluster.prompt_len

                    # 添加最后selectors.shape[-2]个位置的索引
                    recent_indices = torch.arange(key_states_compress.size(-2) - selectors.shape[-2], 
                                                key_states_compress.size(-2), 
                                                device=key_states_compress.device).unsqueeze(0).unsqueeze(0).expand(bsz, key_states_compress.size(1), -1)
                    
                    # 合并所有索引
                    final_indices = torch.cat([prompt_indices, compress_indices.unsqueeze(0), recent_indices], dim=-1)

                    print(final_indices.shape, "final_indices")
                    
                    # 扩展索引维度以便gather操作
                    gather_indices = final_indices.unsqueeze(-1).expand(-1, key_states_compress.size(1), -1, key_states_compress.size(-1))
                    
                    # 进行再次压缩
                    key_states_compress = key_states_compress.gather(dim=2, index=gather_indices)
                    value_states_compress = value_states_compress.gather(dim=2, index=gather_indices)

                
                # replace with compressed cache
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
                
                self.kv_cluster.num_comp += 1
        
        else:

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if self.layer_idx == 0:
                print("\033[33mInput Done!!! Start Compressing CoT...\033[0m")


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

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)


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
    

class Qwen2RPCForCausalLM(Qwen2ForCausalLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
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

        if input_ids.shape[-1] > 1:
            self.is_newline = False
            self.current_step_len = 0
            self.step_confidences = []
            self.early_exit = False
            self.CoT_done = False

        # =============== Step-level Compression logic start ===============
        # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
        predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

        self.CoT_done = (
            predicted_token_ids[0].cpu().item() in self.CoT_done_token_ids
        )

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Calculate entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_probs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Get top-k probabilities
        k = getattr(self.config, 'topk_size', 20)  # Default to top-20 if not specified
        # Get top-k probabilities
        topk_probs, _ = torch.topk(probs, k, dim=-1)
        # Calculate log of top-k probabilities and negative mean
        epsilon = 1e-8  # Avoid log(0)
        topk_log_probs = torch.log(topk_probs + epsilon)
        neg_mean_topk_log_prob = -torch.mean(topk_log_probs, dim=-1)

        # print(f"Predicted Token ID: {predicted_token_ids[0].item()}, Entropy: {entropy.item():.4f}, Neg Mean Top-{k} Log Prob: {neg_mean_topk_log_prob.item():.4f}")

        self.current_step_len += 1
        self.step_confidences.append(neg_mean_topk_log_prob.item())

        if self.is_newline and entropy >= 0.3 and not self.CoT_done and not self.early_exit and self.current_step_len > 5:
            for layer in self.model.layers:
                layer.self_attn.step_start_indices.append(self.current_step_len-1+layer.self_attn.step_start_indices[-1])
            
            # Check if average confidence is below threshold for early exiting
            # if len(self.step_confidences) > 0:
            #     avg_confidence = sum(self.step_confidences) / len(self.step_confidences)
            #     if avg_confidence < 8:
            #         print("early exiting", avg_confidence)
            #         self.early_exit = True
            #     else:
            #         self.step_confidences = []
            if not self.early_exit:
                for layer in self.model.layers:
                    layer.self_attn.cache_mode = "compression"
            self.is_newline = False
            self.current_step_len = 0
        else:
            for layer in self.model.layers:
                layer.self_attn.cache_mode = "vanilla"
            self.is_newline = False

        if predicted_token_ids[0].item() in self.newline_token_ids:
            self.is_newline = True

        # Set compression flag for all layers at once
        if self.CoT_done == True:
            for layer in self.model.layers:
                layer.self_attn.cache_mode = "vanilla"
            if not hasattr(self, '_cot_done_printed'):
                print("\033[33mCoT Done!!! Start Answering...\033[0m")
                self._cot_done_printed = True
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