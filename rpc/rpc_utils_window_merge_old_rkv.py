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
        self.agg_func = 'max'

        
    def cache_recent(self, current_query_states):
        if self.cached_recent is None:
            self.cached_recent = current_query_states
        else:
            self.cached_recent = torch.cat([self.cached_recent, current_query_states], dim=-2)

    def compress_kv(self, origin_key_states, origin_value_states, row_sum_accu, col_sum_accu, num_key_value_groups):

        # selectors = self.cached_recent

        # # # support gqa
        # key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        # value_states = repeat_kv(origin_value_states, self.num_key_value_groups)
        
        # bsz, num_heads, q_len, head_dim = selectors.shape

  
        # attn_weights = torch.matmul(selectors, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        # # no need to deal with attention mask

        # attn_weights = nn.functional.softmax(attn_weights[:, :, :, self.prompt_len:-self.R], dim=-1, dtype=torch.float32).to(selectors.dtype)
        # attn_weights_sum = attn_weights.sum(dim = -2)

        # print(attn_weights_sum, 1111111111)
            
        # origin_row_sum_accu = row_sum_accu

        # row_sum_accu = row_sum_accu[..., :-self.R]
        
        if self.aggregation == 'all':

            row_sum_accu = row_sum_accu.view(row_sum_accu.shape[0], -1, 1, row_sum_accu.shape[-1])
            col_sum_accu = col_sum_accu.view(col_sum_accu.shape[0], -1, 1, col_sum_accu.shape[-1])

            if self.agg_func == 'max':
                row_sum_accu = row_sum_accu.max(dim=1).values
                col_sum_accu = col_sum_accu.max(dim=1).values
            elif self.agg_func == 'mean':
                row_sum_accu = row_sum_accu.mean(dim=1)
                col_sum_accu = col_sum_accu.mean(dim=1)
            else:
                raise ValueError('agg_func not supported')
        
        elif self.aggregation == 'group':

            row_sum_accu = row_sum_accu.view(row_sum_accu.shape[0], -1, num_key_value_groups, row_sum_accu.shape[-1])
            col_sum_accu = col_sum_accu.view(col_sum_accu.shape[0], -1, num_key_value_groups, col_sum_accu.shape[-1])

            if self.agg_func == 'max':
                row_sum_accu = row_sum_accu.max(dim=-2).values
                col_sum_accu = col_sum_accu.max(dim=-2).values
            elif self.agg_func == 'mean':
                row_sum_accu = row_sum_accu.mean(dim=-2)
                col_sum_accu = col_sum_accu.mean(dim=-2)
            else:
                raise ValueError('agg_func not supported')

        if self.pooling == 'avgpool':
            row_attn_cache = F.avg_pool1d(row_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            col_attn_cache = F.avg_pool1d(col_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            row_attn_cache = F.max_pool1d(row_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            col_attn_cache = F.max_pool1d(col_sum_accu, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        alpha = 0.9
        row_col_sum = alpha * (col_attn_cache / self.R) + (1 - alpha) * (row_attn_cache / self.prompt_len) #NOT SURE IF THIS IS CORRECT, NEED TO CHECK
        indices = row_col_sum.topk(self.budget_cot-self.R, dim=-1, largest=True).indices.sort(dim=-1).values
        # row_indices = row_attn_cache.topk(self.cp_cot, dim=-1, largest=True).indices.sort(dim=-1).values
        # col_indices = col_attn_cache.topk(self.cp_cot, dim=-1, largest=True).indices.sort(dim=-1).values
        # indices = torch.cat([row_indices, col_indices], dim=-1) 
        # indices = torch.sort(indices, dim=-1).values       # 排序
        # indices = torch.unique(indices, dim=-1)            # 去重

        # self.cached_recent = None # for next compress

        # need check
        if self.aggregation == 'all':
            batch, n_head, slen = indices.shape
            sum_indices = indices[:, None, :, :].expand(batch, origin_key_states.size(1) * num_key_value_groups, n_head, slen)
            sum_indices = sum_indices.reshape(batch, n_head * origin_key_states.size(1) * num_key_value_groups, slen)
        elif self.aggregation == 'group':
            batch, n_head, slen = indices.shape
            sum_indices = indices[:, None, :, :].expand(batch, num_key_value_groups, n_head, slen)
            sum_indices = sum_indices.reshape(batch, n_head * num_key_value_groups, slen)
        # row_sum_accu = torch.cat([origin_row_sum_accu.gather(dim = 2, index = sum_indices), origin_row_sum_accu[..., -self.R:]], dim=-1)

        bsz, num_heads, _, head_dim = origin_key_states.shape  
        # indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        target_key_states = origin_key_states[:, :, self.prompt_len:-self.R, :]
        target_value_states = origin_value_states[:, :, self.prompt_len:-self.R, :]

        mask = torch.zeros(target_key_states.shape[:-1], dtype=torch.bool).to(target_key_states.device)
        mask = mask.scatter(-1, indices, 1)

        k_hh_recent = target_key_states[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = target_value_states[mask].view(bsz, num_heads, -1, head_dim)
        
        # applying merge here
        # breakpoint()
        k_hh_pruned = target_key_states[~mask].view(bsz, num_heads, -1, head_dim)
        v_hh_pruned = target_value_states[~mask].view(bsz, num_heads, -1, head_dim)
        # similarity = k_hh_pruned @ k_hh_recent.transpose(-1, -2) # dot product
        similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, head_dim)) @ ((k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, head_dim))).transpose(-1, -2)) # cosin
        max_values, max_indices = similarity.max(dim=-1)
        # breakpoint()   
        if self.threshold == None:
            self.threshold = max_values.mean()
        else:
            # self.threshold = (self.threshold + max_values.mean()) / 2
            # breakpoint()
            self.threshold = 0.3 * self.threshold + 0.7 * max_values.mean()  # 0.3 0.7
        filter_indices = (max_values.mean(1)>=self.threshold).squeeze(0)
        merged_indices = max_indices[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, head_dim)
        merge_weights = max_values[..., filter_indices].unsqueeze(-1).repeat(1, 1, 1, head_dim)

        k_hh_merged = k_hh_pruned[..., filter_indices, :]
        # k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=merge_weights*k_hh_merged, reduce='mean', include_self=True)
        k_hh_recent = scatter_reduce_ema(input=k_hh_recent, dim=2, index=merged_indices, src_merge=k_hh_merged, merge_weights=merge_weights)
    
        v_hh_merged = v_hh_pruned[..., filter_indices, :]
        # v_hh_recent = torch.scatter_reduce(input=v_hh_recent, dim=2, index=merged_indices, src=merge_weights*v_hh_merged, reduce='mean', include_self=True)
        v_hh_recent = scatter_reduce_ema(input=v_hh_recent, dim=2, index=merged_indices, src_merge=v_hh_merged, merge_weights=merge_weights)
    

        # support gqa
        if self.aggregation == 'all' or 'group':
            k_prompt = origin_key_states[:, :, :self.prompt_len, :]
            v_prompt = origin_value_states[:, :, :self.prompt_len, :]

            # k_past_compress = origin_key_states[:, :, self.prompt_len:-self.R, :].gather(dim = 2, index = indices)
            # v_past_compress = origin_value_states[:, :, self.prompt_len:-self.R, :].gather(dim = 2, index = indices)

            k_past_compress = k_hh_recent
            v_past_compress = v_hh_recent
            
            k_cur = origin_key_states[:, :, -self.R:, :]
            v_cur = origin_value_states[:, :, -self.R:, :]

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