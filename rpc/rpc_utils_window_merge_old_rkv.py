import warnings

import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union, Any,Dict
from transformers.cache_utils import Cache, DynamicCache
from flash_attn import flash_attn_func
# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv for gqa_support
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def calculate_entropy(attention_scores):
    attention_scores = attention_scores.to(torch.float32)
    entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))  
    entropy= entropy.to(dtype=torch.float32)
    return entropy

def scatter_reduce_ema(input, dim, index, src_merge, merge_weights):
    # 假设 k_hh_recent, merged_indices, merge_weights, k_hh_merged 已定义
    # dim = 2

    # Step 1: 初始化目标张量
    weighted_sum = torch.zeros_like(input)
    count = torch.zeros_like(input)

    # Step 2: scatter_add 累加 merge_weights * k_hh_merged 到对应 index
    weighted_sum.scatter_add_(
        dim=dim,
        index=index,
        src=src_merge * merge_weights
    )

    count.scatter_add_(
        dim=2,
        index=index,
        src=merge_weights
    )

    # Step 4: include_self=True → 加入原始 k_hh_recent 的值
    weighted_sum += input
    
    count += torch.ones_like(input)

    # Step 5: 求平均，避免除以 0
    weighted_sum = weighted_sum / (count + 1e-6)

    # Step 5: 求平均，避免除以 0
    return weighted_sum

class RPCCluster():
    def __init__(self,
                 layer_idx = None, 
                 P=1024,
                 R=32,
                 c=4,
                 selectors='recent', # options prompt, new, recent
                 aggregation='all', # all, group, none
                 kernel_size=7, 
                 pooling='avgpool',
                 num_key_value_groups=1,
                 budget_cot=4096,
                 buffer_cot=128,
                 budget_ans=1024,
                 cp_ratio=0.25
                 ):

        self.layer_idx = layer_idx

        # compression arguments
        self.budget_cot = budget_cot
        self.buffer_cot = buffer_cot
        self.budget_ans = budget_ans
        self.cp_ratio = cp_ratio
        self.cp_cot = int(budget_cot*cp_ratio)
        self.cp_ans = int(budget_ans*cp_ratio)
        self.prompt_len = 0
        self.num_comp = 0
        self.R = R

        self.kernel_size = kernel_size
        self.pooling = pooling

        self.selectors = selectors

        self.cached_prompt = None
        self.cached_recent = None
        self.threshold = None

        # support gqa
        self.aggregation = aggregation
        self.num_key_value_groups = num_key_value_groups
        self.agg_func = 'mean'

        
    def cache_recent(self, current_query_states):
        if self.cached_recent is None:
            self.cached_recent = current_query_states
        else:
            self.cached_recent = torch.cat([self.cached_recent, current_query_states], dim=-2)

    def compress_kv(self, origin_key_states, origin_value_states, row_sum_accu, col_sum_accu, num_key_value_groups, step_start_indices, selectors):

        # # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        
        bsz, num_heads, q_len, head_dim = selectors.shape

  
        attn_weights = torch.matmul(selectors, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        # no need to deal with attention mask

        attn_weights = nn.functional.softmax(attn_weights[:, :, :, self.prompt_len:-selectors.shape[-2]], dim=-1, dtype=torch.float32).to(selectors.dtype)
        print(attn_weights.shape, "attn_weights_shape")
        col_sum_accu = attn_weights

        # print(attn_weights_sum, 1111111111)
            
        # origin_row_sum_accu = row_sum_accu

        # row_sum_accu = row_sum_accu[..., :-self.R]
        
        # if self.aggregation == 'all':

        #     row_sum_accu = row_sum_accu.view(row_sum_accu.shape[0], -1, 1, row_sum_accu.shape[-1])
        #     col_sum_accu = col_sum_accu.view(col_sum_accu.shape[0], -1, 1, col_sum_accu.shape[-1])

        #     if self.agg_func == 'max':
        #         row_sum_accu = row_sum_accu.max(dim=1).values
        #         col_sum_accu = col_sum_accu.max(dim=1).values
        #     elif self.agg_func == 'mean':
        #         row_sum_accu = row_sum_accu.mean(dim=1)
        #         col_sum_accu = col_sum_accu.mean(dim=1)
        #     else:
        #         raise ValueError('agg_func not supported')
        
        # elif self.aggregation == 'group':

        #     row_sum_accu = row_sum_accu.view(row_sum_accu.shape[0], -1, num_key_value_groups, row_sum_accu.shape[-1])
        #     col_sum_accu = col_sum_accu.view(col_sum_accu.shape[0], -1, num_key_value_groups, col_sum_accu.shape[-1])

        #     if self.agg_func == 'max':
        #         row_sum_accu = row_sum_accu.max(dim=-2).values
        #         col_sum_accu = col_sum_accu.max(dim=-2).values
        #     elif self.agg_func == 'mean':
        #         row_sum_accu = row_sum_accu.mean(dim=-2)
        #         col_sum_accu = col_sum_accu.mean(dim=-2)
        #     else:
        #         raise ValueError('agg_func not supported')

        # if self.pooling == 'avgpool':
        #     row_attn_cache = F.avg_pool1d(row_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        #     col_attn_cache = F.avg_pool1d(col_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        # elif self.pooling == 'maxpool':
        #     row_attn_cache = F.max_pool1d(row_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        #     col_attn_cache = F.max_pool1d(col_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        # else:
        #     raise ValueError('Pooling method not supported')

        alpha = 0.9
        row_col_sum = col_sum_accu.mean(dim=1) #NOT SURE IF THIS IS CORRECT, NEED TO CHECK
        print(row_col_sum.shape, "row_col_sum")

        # 保存原始的row_col_sum用于topk操作
        original_row_col_sum = row_col_sum
        
        # 根据step_start_indices对每个分区在row_col_sum的最后一维求和，累计除最后一段的其他step
        if step_start_indices is not None:
            print(step_start_indices, "step_start_indices")
            partitioned_sums = []
            # 只处理除最后一个step之外的所有step
            for i in range(len(step_start_indices) - 2):  # 排除最后一个step
                start_idx = step_start_indices[i]
                end_idx = step_start_indices[i + 1]
                partition_sum = row_col_sum[..., start_idx:end_idx].sum(dim=-1).mean(dim=1).unsqueeze(1)  # 计算每个分区的和并保持维度
                partitioned_sums.append(partition_sum)
                print(partition_sum.shape, "partition_sum")
            step_scores = torch.cat(partitioned_sums, dim=-1)  # 形状: (bsz, n_steps-1) 排除最后一个step
            print(step_scores, "step_scores")
            # 在step级别取topk，现在k_steps应该基于排除最后一个step后的数量
            k_steps = step_scores.size(-1) - 1 if step_scores.size(-1) > 1 else step_scores.size(-1)
            topk_step_values, topk_step_indices = torch.topk(step_scores, k=k_steps, dim=-1, largest=True)
            
            # 将step索引映射回原始序列索引
            selected_indices = []
            for batch_idx in range(topk_step_indices.size(0)):
                batch_indices = []
                for step_idx in topk_step_indices[batch_idx]:
                    step_idx = step_idx.item()
                    start_pos = step_start_indices[step_idx]
                    end_pos = step_start_indices[step_idx + 1] if step_idx + 1 < len(step_start_indices) else original_row_col_sum.size(-1)
                    # 在该step内选择所有token
                    step_token_indices = torch.arange(start_pos, end_pos, device=original_row_col_sum.device)
                    print(step_token_indices, "step_token_indices")
                    batch_indices.append(step_token_indices)
                selected_indices.append(torch.cat(batch_indices))

            # 对selected_indices的最后一维进行排序
            for batch_idx in range(len(selected_indices)):
                selected_indices[batch_idx] = torch.sort(selected_indices[batch_idx])[0]

            print(selected_indices, "selected_indices")
            
            # 重新组织索引张量
            max_tokens = max(len(idx) for idx in selected_indices)
            indices_tensor = torch.zeros(topk_step_indices.size(0), max_tokens, 
                                       dtype=torch.long, device=original_row_col_sum.device)
            
            for batch_idx, idx_list in enumerate(selected_indices):
                # 再次确保索引不超出past_seq_len
                indices_tensor[batch_idx, :len(idx_list)] = idx_list
                
            # 扩展维度以便后续的gather操作
            # 需要为所有head复制相同的索引，并扩展最后一维
            indices = indices_tensor.unsqueeze(1).expand(-1, origin_key_states.size(1), -1).unsqueeze(-1).expand(-1, -1, -1, origin_key_states.size(-1))
            print(indices.shape, "final_indices_shape")
        # self.cached_recent = None # for next compress

        # support gqa
        if self.aggregation == 'all' or 'group':
            k_prompt = origin_key_states[:, :, :self.prompt_len, :]
            v_prompt = origin_value_states[:, :, :self.prompt_len, :]
        
            k_past_compress = origin_key_states[:, :, self.prompt_len:-selectors.shape[-2], :].gather(dim = 2, index = indices)
            v_past_compress = origin_value_states[:, :, self.prompt_len:-selectors.shape[-2], :].gather(dim = 2, index = indices)

            print(origin_key_states[:, :, self.prompt_len:-selectors.shape[-2], :].shape, "k_past_compress_shape")

            k_cur = origin_key_states[:, :, -selectors.shape[-2]:, :]
            v_cur = origin_value_states[:, :, -selectors.shape[-2]:, :]

        else:
            k_prompt = key_states[:, :, :self.prompt_len, :]
            v_prompt = value_states[:, :, :self.prompt_len, :]

            k_past_compress = key_states[:, :, self.prompt_len:, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, self.prompt_len:, :].gather(dim = 2, index = indices)

            # k_cur = key_states[:, :, -self.R:, :]
            # v_cur = value_states[:, :, -self.R:, :]

        key_states = torch.cat([k_prompt, k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_prompt, v_past_compress, v_cur], dim = 2)


        return key_states, value_states
   

def init_rpc(self):

    # init only once
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = RPCCluster(
            layer_idx = self.layer_idx,
            P = 1024,
            R = 32,
            c = 4,
            selectors='recent', # options: new, recent
            aggregation='all', # options: all, group, none
            kernel_size = 7,
            pooling = 'avgpool',
            num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads,
            budget_cot=4096,
            buffer_cot=128,
            budget_ans=1024,
            cp_ratio=0.25
            )