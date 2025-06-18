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

def set_rpc_config(
    model,
    P=1024,
    R=32,
    c=4,
    selectors='recent',
    aggregation='all',
    kernel_size=7, 
    pooling='avgpool',
    budget_cot=4096,
    budget_ans=1024,
    cp_ratio=0.25
    ):

    layers = len(model.model.layers)

    for i in range(layers):
        model.model.layers[i].self_attn.kv_cluster.budget_cot = budget_cot
        model.model.layers[i].self_attn.kv_cluster.cp_ratio = cp_ratio
        model.model.layers[i].self_attn.kv_cluster.cp_cot = int(budget_cot*cp_ratio)
        model.model.layers[i].self_attn.kv_cluster.budget_ans = budget_ans
        model.model.layers[i].self_attn.kv_cluster.cp_ans = int(budget_ans*cp_ratio)
        model.model.layers[i].self_attn.kv_cluster.R = R
        model.model.layers[i].self_attn.kv_cluster.selectors = selectors
        model.model.layers[i].self_attn.kv_cluster.aggregation = aggregation
        model.model.layers[i].self_attn.kv_cluster.kernel_size = kernel_size
        model.model.layers[i].self_attn.kv_cluster.pooling = pooling

    print(f"[RPC Config][CoT Budget={budget_cot}, Ans Budget={budget_ans}, Compression ratio={cp_ratio}][selectors={selectors}, aggregation={aggregation}]",  flush=True)

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
                 budget_ans=1024,
                 cp_ratio=0.25
                 ):

        self.layer_idx = layer_idx

        # compression arguments
        self.budget_cot = budget_cot
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

        # support gqa
        self.aggregation = aggregation
        self.num_key_value_groups = num_key_value_groups
        self.agg_func = 'mean'

        
    def cache_recent(self, current_query_states):
        if self.cached_recent is None:
            self.cached_recent = current_query_states
        else:
            self.cached_recent = torch.cat([self.cached_recent, current_query_states], dim=-2)

    def compress_kv(self, origin_key_states, origin_value_states, row_sum_accu, col_sum_accu, num_key_value_groups):

        origin_col_sum_accu = col_sum_accu
        origin_row_sum_accu = row_sum_accu

        row_sum_accu = row_sum_accu[..., :-self.R]
        col_sum_accu = col_sum_accu[..., :-self.R]

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

        row_indices = row_attn_cache.topk(self.cp_cot, dim=-1, largest=True).indices.sort(dim=-1).values
        col_indices = col_attn_cache.topk(self.cp_cot, dim=-1, largest=True).indices.sort(dim=-1).values
        indices = torch.cat([row_indices, col_indices], dim=-1) 
        indices = torch.sort(indices, dim=-1).values       # 排序
        indices = torch.unique(indices, dim=-1)            # 去重

        # need check
        if self.aggregation == 'all':
            batch, n_head, slen = indices.shape
            sum_indices = indices[:, None, :, :].expand(batch, origin_key_states.size(1) * num_key_value_groups, n_head, slen)
            sum_indices = sum_indices.reshape(batch, n_head * origin_key_states.size(1) * num_key_value_groups, slen)
        elif self.aggregation == 'group':
            batch, n_head, slen = indices.shape
            sum_indices = indices[:, None, :, :].expand(batch, num_key_value_groups, n_head, slen)
            sum_indices = sum_indices.reshape(batch, n_head * num_key_value_groups, slen)
        col_sum_accu = torch.cat([origin_col_sum_accu.gather(dim = 2, index = sum_indices), origin_col_sum_accu[..., -self.R:]], dim=-1)
        row_sum_accu = torch.cat([origin_row_sum_accu.gather(dim = 2, index = sum_indices), origin_row_sum_accu[..., -self.R:]], dim=-1)

        head_dim = origin_key_states.shape[-1]        
        indices = indices.unsqueeze(-1).expand(-1, origin_key_states.size(1), -1, head_dim)


        # support gqa
        if self.aggregation == 'all' or 'group':
            k_prompt = origin_key_states[:, :, :self.prompt_len, :]
            v_prompt = origin_value_states[:, :, :self.prompt_len, :]

            k_past_compress = origin_key_states[:, :, self.prompt_len:, :].gather(dim = 2, index = indices)
            v_past_compress = origin_value_states[:, :, self.prompt_len:, :].gather(dim = 2, index = indices)
            
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


        return key_states, value_states, col_sum_accu, row_sum_accu
   

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
            budget_ans=1024,
            cp_ratio=0.25
            )