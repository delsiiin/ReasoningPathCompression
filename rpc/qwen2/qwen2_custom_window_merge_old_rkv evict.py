import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
    Qwen2Attention,
    Qwen2Model
)
from .qwen2_config import Qwen2Config
from transformers.utils import (
    logging,
    TransformersKwargs
)
from transformers.processing_utils import Unpack
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
from rpc.step_lens_optimizer import (
    update_step_lens_optimized,
    build_final_indices_optimized,
    efficient_gather_operation
)

logger = logging.get_logger(__name__)

def calculate_entropy(attention_scores):
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
    entropy= entropy.to(dtype=torch.float32)
    return entropy

def cal_similarity(
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

    # zero the specified positions in similarity_cos
    similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

    return similarity_cos.mean(dim=1).softmax(dim=-1)


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

def Qwen2_Ours_init(self, config: Qwen2Config, layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
    self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
    init_rpc(self)
    self.verbose = False

    self.cache_mode = "vanilla"  # options: vanilla, compression

    self.layer_budget_importance = None

    self.row_sum_accu = None
    self.col_sum_accu = None

    self.question_cache = None

    self.step_lens = []
    self.is_new_step = False

def Qwen2_Ours_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    bsz, q_len, _ = hidden_states.size()
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        self.step_lens = []
        self.is_new_step = False
        self.current_step_len = 0
        if hasattr(self, "_cot_done_printed"):
            delattr(self, "_cot_done_printed")

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if q_len == 1:

            self.kv_cluster.cache_recent(query_states)
            if self.kv_cluster.cached_recent.shape[-2] > self.kv_cluster.R:
                self.kv_cluster.cached_recent = self.kv_cluster.cached_recent[:, :, -self.kv_cluster.R :, :]

            if not self.is_new_step:
                self.current_step_len += 1
            else:
                current_step_len = self.current_step_len
                self.current_step_len = 0
                self.is_new_step = False

            # cannot use 'past_key_value.get_seq_length'
            target_length = past_key_value.key_cache[self.layer_idx].size()[-2] - self.kv_cluster.prompt_len 

            if target_length >= self.kv_cluster.budget_cot and self.cache_mode == "compression":

                # print(target_length, "target_length")

                # if self.layer_idx == 5:
                #     print(selectors.shape[-2], "selectors.shape")
                
                # question = self.question_cache

                # bsz, num_heads, ques_len, head_dim = question.shape

        
                # ques_attn_weights = torch.matmul(question, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                # # no need to deal with attention mask

                # ques_attn_weights = nn.functional.softmax(ques_attn_weights[:, :, :, self.kv_cluster.prompt_len:], dim=-1, dtype=torch.float32).to(question.dtype)
                # ques_attn_weights_sum = ques_attn_weights.sum(dim = -2)

                ques_attn_weights_sum = None

                key_states_compress, value_states_compress, updated_step_lens= self.kv_cluster.compress_kv(key_states, value_states, ques_attn_weights_sum, self.col_sum_accu, self.num_key_value_groups, self.step_lens, current_step_len)

                self.step_lens = updated_step_lens

                if key_states_compress.size(-2) - self.kv_cluster.prompt_len > self.kv_cluster.budget_cot:

                    # print(key_states_compress, "key_states_compress before second compression")
                    
                    similarity_cos = cal_similarity(
                        key_states_compress,  # 包装成列表格式以匹配函数期望
                        retain_ratio=0.2,
                        retain_direction="last",
                    )[..., :-current_step_len]

                    # print(similarity_cos.shape, "similarity_cos")
                    # 选择similarity_cos中最小的budget_cot个位置
                    min_indices = torch.topk(similarity_cos[..., self.kv_cluster.prompt_len:], k=self.kv_cluster.budget_cot - current_step_len, dim=-1, largest=False).indices
                    
                    # 对索引进行排序以保持原始顺序
                    min_indices_sorted = torch.sort(min_indices, dim=-1).values
                    # print(min_indices_sorted.shape, "min_indices_sorted")


                    
                    # 使用优化的索引构建和gather操作
                    device = key_states_compress.device
                    seq_len = key_states_compress.size(-2)
                    num_heads = key_states_compress.size(1)
                    recent_len = current_step_len
                    
                    # 高效构建final_indices
                    final_indices = build_final_indices_optimized(
                        bsz, num_heads, self.kv_cluster.prompt_len,
                        min_indices_sorted, seq_len, recent_len, device
                    )
                    
                    # 高效进行gather操作
                    key_states_compress, value_states_compress = efficient_gather_operation(
                        key_states_compress, value_states_compress, final_indices
                    )

                    # print(key_states_compress.shape, "key_states_compress after second compression")
                    
                    # 高效更新step_lens - 使用优化的向量化实现
                    # step_lens_start_time = time.time()
                    if len(self.step_lens) > 0:
                        self.step_lens = update_step_lens_optimized(
                            self.step_lens, 
                            min_indices_sorted[0], 
                            min_indices_sorted.device
                        )
                        
                        # step_lens_end_time = time.time()
                        # step_lens_update_time = step_lens_end_time - step_lens_start_time
                        # if self.layer_idx == 0:  # 只在第一层打印，避免输出过多
                        #     print(f"Step_lens update time: {step_lens_update_time:.4f}s")

                
                # replace with compressed cache
                past_key_value.key_cache[self.layer_idx] = key_states_compress
                past_key_value.value_cache[self.layer_idx] = value_states_compress
                
                self.kv_cluster.num_comp += 1
        
        else:

            if self.layer_idx == 0:
                print("\033[33mInput Done!!! Start Compressing CoT...\033[0m")

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # Change Mask [TODO]
    if q_len == 1 and self.config._attn_implementation == "eager":
        attention_mask = None

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen2_Ours_CausalLM_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if input_ids.shape[-1] > 1:
            self.is_newline = False
            self.current_step_len = 0
            self.step_confidences = []
            self.early_exit = False
            self.CoT_done = False

        # =============== Step-level Compression logic start ===============
        # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
        predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

        if not self.CoT_done:
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

        # # Get top-k probabilities
        # k = getattr(self.config, 'topk_size', 20)  # Default to top-20 if not specified
        # # Get top-k probabilities
        # topk_probs, _ = torch.topk(probs, k, dim=-1)
        # # Calculate log of top-k probabilities and negative mean
        # epsilon = 1e-8  # Avoid log(0)
        # topk_log_probs = torch.log(topk_probs + epsilon)
        # neg_mean_topk_log_prob = -torch.mean(topk_log_probs, dim=-1)

        # print(f"Predicted Token ID: {predicted_token_ids[0].item()}, Entropy: {entropy.item():.4f}, Neg Mean Top-{k} Log Prob: {neg_mean_topk_log_prob.item():.4f}")

        self.current_step_len += 1
        # self.step_confidences.append(neg_mean_topk_log_prob.item())

        if self.is_newline and not self.CoT_done and not self.early_exit and self.current_step_len >= self.model.layers[0].self_attn.kv_cluster.R:
            for layer in self.model.layers:
                layer.self_attn.step_lens.append(self.current_step_len)
                layer.self_attn.is_new_step = True
            
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )