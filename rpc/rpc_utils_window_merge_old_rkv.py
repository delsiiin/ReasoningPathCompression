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

from rpc.step_lens_optimizer import (
    update_step_lens_optimized,
    build_final_indices_optimized,
    efficient_gather_operation
)

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
        self.threshold_token = None  # 第二个EMA阈值，用于token级别的相似度

        # support gqa
        self.aggregation = aggregation
        self.num_key_value_groups = num_key_value_groups
        self.agg_func = 'mean'

        
    def cache_recent(self, current_query_states):
        if self.cached_recent is None:
            self.cached_recent = current_query_states
        else:
            self.cached_recent = torch.cat([self.cached_recent, current_query_states], dim=-2)

    def compress_kv(self, origin_key_states, origin_value_states, row_sum_accu, col_sum_accu, num_key_value_groups, step_lens, current_step_len):

        bsz, num_heads, q_len, head_dim = origin_key_states.shape

        # # support gqa
        key_states = repeat_kv(origin_key_states, self.num_key_value_groups)
        value_states = repeat_kv(origin_value_states, self.num_key_value_groups)

        selectors = self.cached_recent

        # 记录注意力权重计算时间
        # attn_start_time = time.time()
        attn_weights = torch.matmul(selectors, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        # no need to deal with attention mask

        attn_weights = nn.functional.softmax(attn_weights[:, :, :, self.prompt_len:-current_step_len], dim=-1, dtype=torch.float32).to(selectors.dtype)
        # attn_end_time = time.time()
        # attn_time = attn_end_time - attn_start_time
        # if self.layer_idx == 0:  # 只在第一层打印，避免输出过多
        #     print(f"Attention weights calculation time: {attn_time:.4f}s")
        
        # print(attn_weights.shape, "attn_weights_shape")
        col_sum_accu = attn_weights
        
        row_col_sum = col_sum_accu.mean(dim=1) #NOT SURE IF THIS IS CORRECT, NEED TO CHECK
        # print(row_col_sum.shape, "row_col_sum")
        
            # 根据step_lens对每个分区在row_col_sum的最后一维求和，累计除最后一段的其他step
        if step_lens is not None:
            # start_time = time.time()
            # print(step_lens, "step_lens")
            # 使用向量化计算step起始索引
            step_lens_tensor = torch.tensor(step_lens[:-1], device=row_col_sum.device)
            step_start_indices = torch.cat([torch.zeros(1, dtype=torch.long, device=row_col_sum.device), 
                                           torch.cumsum(step_lens_tensor, dim=0)])
            
            # 向量化计算所有step的partition_sum
            partitioned_sums = []
            for i in range(len(step_lens) - 1):  # 排除最后一个step
                start_idx = step_start_indices[i].item()
                end_idx = (step_start_indices[i] + step_lens[i]).item() if isinstance(step_lens[i], int) else (step_start_indices[i] + step_lens[i]).item()
                partition_sum = row_col_sum[..., start_idx:end_idx].sum(dim=-1).mean(dim=1).unsqueeze(1)
                partitioned_sums.append(partition_sum)
            step_scores = torch.cat(partitioned_sums, dim=-1)  # 形状: (bsz, n_steps-1) 排除最后一个step
            # print(step_scores, "step_scores")
            # 在step级别取topk，现在k_steps应该基于排除最后一个step后的数量
            k_steps = step_scores.size(-1) - 1 if step_scores.size(-1) > 1 else step_scores.size(-1)
            topk_step_values, topk_step_indices = torch.topk(step_scores, k=k_steps, dim=-1, largest=True, sorted=False)

            # if self.layer_idx == 0:
            #     print(step_lens, "step_lens")
            
            # 将step索引映射回原始序列索引 - 向量化处理
            selected_indices = []
            for batch_idx in range(topk_step_indices.size(0)):
                # 获取该batch的所有step索引
                batch_step_indices = topk_step_indices[batch_idx]  # (k_steps,)
                # 向量化获取起始和结束位置
                start_positions = step_start_indices[batch_step_indices]
                step_lengths = torch.tensor([step_lens[idx.item()] for idx in batch_step_indices], 
                                           device=row_col_sum.device)
                end_positions = start_positions + step_lengths
                
                # 生成所有token索引
                batch_indices = []
                for i in range(len(batch_step_indices)):
                    step_token_indices = torch.arange(start_positions[i].item(), 
                                                     end_positions[i].item(), 
                                                     device=row_col_sum.device)
                    batch_indices.append(step_token_indices)
                selected_indices.append(torch.cat(batch_indices))

            # 更新step_lens，仅保留topk选出的step，按照原step顺序排列
            # 对于batch中的第一个样本，选择topk的step索引（假设batch中所有样本的step选择相同）
            selected_step_indices = topk_step_indices[0]  # 保持在GPU上
            # 对选中的step索引按照原始顺序排序，保持在GPU上
            selected_step_indices_sorted = torch.sort(selected_step_indices).values
            # 直接使用GPU张量进行索引操作，避免CPU转换
            selected_step_lens = [step_lens[i.item()] for i in selected_step_indices_sorted] + [step_lens[-1]]  # 保留最后一个step
            step_lens = selected_step_lens

            # if self.layer_idx == 0:
            #     print(selected_step_indices_sorted, "selected_step_indices_sorted")
            #     print(step_lens, "step_lens_updated")

            # 对selected_indices的最后一维进行排序
            for batch_idx in range(len(selected_indices)):
                selected_indices[batch_idx] = torch.sort(selected_indices[batch_idx])[0]

            # if self.layer_idx == 0:
            #     print(selected_indices, "selected_indices")
            
            if selected_indices[0].shape[-1] <= self.budget_cot - current_step_len:

                # 直接对selected_indices进行维度扩展，用于gather操作
                # 将每个batch的索引扩展到 (num_heads, seq_len, head_dim) 的形状
                indices = []
                for idx_list in selected_indices:
                    # 扩展维度: (seq_len,) -> (num_heads, seq_len, head_dim)
                    expanded_idx = idx_list.unsqueeze(0).unsqueeze(-1).expand(
                        origin_key_states.size(1), -1, origin_key_states.size(-1)
                    )
                    indices.append(expanded_idx)
                indices = torch.stack(indices, dim=0)  # 堆叠成 (batch_size, num_heads, seq_len, head_dim)
                # print(indices.shape, "final_indices_shape")
                
                # end_time = time.time()
                # compression_time = end_time - start_time
                # if self.layer_idx == 0:  # 只在第一层打印，避免输出过多
                #     print(f"Step compression time: {compression_time:.4f}s")

            else:
                # 当候选 token 超过预算：
                # 1) 在 col_sum_accu 中排除 selected_indices 对应元素
                # 2) 从剩余元素中选取 topk，k = self.budget_cot - current_step_len
                # 3) 依据选出的 token 构造 gather 所需的 indices

                # 计算每个历史 token 的分数：此处保留 head 维度，稍后再做聚合
                # token_scores 形状: (bsz, n_head, L)
                token_scores = col_sum_accu.sum(dim = -2)
                device = token_scores.device

                # 根据 topk_step_values 对 token_scores 进行加权
                # 创建一个与 token_scores 形状匹配的权重张量
                step_weights_expanded = torch.ones_like(token_scores)
                # 使用向量化计算step起始索引
                step_lens_tensor = torch.tensor(step_lens[:-1], device=device)
                step_start_indices_tensor = torch.cat([torch.zeros(1, dtype=torch.long, device=device), 
                                                       torch.cumsum(step_lens_tensor, dim=0)])
                
                # 向量化处理权重分配
                for batch_idx in range(bsz):
                    for step_idx in range(len(topk_step_values[batch_idx])):
                        start_pos = step_start_indices_tensor[step_idx].item()
                        end_pos = start_pos + step_lens[step_idx]
                        step_value = topk_step_values[batch_idx, step_idx]
                        # 使用切片和广播一次性赋值
                        step_weights_expanded[batch_idx, :, start_pos:end_pos] = step_value
                
                # 应用权重
                token_scores = token_scores * step_weights_expanded

                bsz, n_head, L = token_scores.shape
                
                dtype = token_scores.dtype

                if self.aggregation == 'all':

                    token_scores = token_scores.view(token_scores.shape[0], -1, 1, token_scores.shape[-1])

                    if self.agg_func == 'max':
                        token_scores = token_scores.max(dim=1).values
                    elif self.agg_func == 'mean':
                        token_scores = token_scores.mean(dim=1)
                    else:
                        raise ValueError('agg_func not supported')
                
                elif self.aggregation == 'group':

                    token_scores = token_scores.view(token_scores.shape[0], -1, num_key_value_groups, token_scores.shape[-1])

                    if self.agg_func == 'max':
                        token_scores = token_scores.max(dim=-2).values
                    elif self.agg_func == 'mean':
                        token_scores = token_scores.mean(dim=-2)
                    else:
                        raise ValueError('agg_func not supported')

                if self.pooling == 'avgpool':
                    token_scores = F.avg_pool1d(token_scores, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    token_scores = F.max_pool1d(token_scores, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')

                # 预算中留给历史 token 的数量
                remaining_k = max(int(self.budget_cot - current_step_len), 0)
                remaining_k = min(remaining_k, L)  # 不超过可选长度

                indices_list = []
                # 对 batch 中每个样本独立计算
                for b in range(bsz):
                    # 构造屏蔽 mask: True 表示可选，False 表示需要排除
                    mask = torch.zeros(L, dtype=torch.bool, device=device)
                    # 排除已选 step 的 token（这些是相对 self.prompt_len:-self.R 窗口的索引）
                    mask[selected_indices[b]] = True

                    # 若没有预算或没有可选元素，则构造空 indices
                    if remaining_k == 0 or mask.sum().item() == 0:
                        selected_tok_idx = torch.empty(0, dtype=torch.long, device=device)
                    else:
                        # 将被排除位置的分数置为 -inf，确保不会被 topk 选中
                        # token_scores[b] 形状: (n_head, L)，按最后一维进行屏蔽
                        masked_scores = token_scores[b].clone()  # (n_head, L)
                        masked_scores[:, ~mask] = float('-inf')

                        # # 在 head 维上做聚合以得到 1D 的 token 分数（可选 mean 或 max，这里使用 max 更保守）
                        # combined_scores = masked_scores.max(dim=0).values  # (L,)

                        # 处理可选数量不足的情况：只取可选数量与 remaining_k 的较小值
                        k_eff = min(remaining_k, int(mask.sum().item()))
                        topk_idx = torch.topk(masked_scores, k=k_eff, dim=-1, largest=True).indices  # (k_eff,)
                        # 排序以保持时间顺序
                        selected_tok_idx = torch.sort(topk_idx).values

                    indices_list.append(selected_tok_idx)

                # 高效更新step_lens - 使用优化的向量化实现
                # step_lens_start_time = time.time()
                if len(step_lens) > 0:
                    step_lens = update_step_lens_optimized(
                        step_lens, 
                        selected_tok_idx[0], 
                        selected_tok_idx.device
                    )
                    
                    # step_lens_end_time = time.time()
                    # step_lens_update_time = step_lens_end_time - step_lens_start_time
                    # if self.layer_idx == 0:  # 只在第一层打印，避免输出过多
                    #     print(f"Step_lens update time: {step_lens_update_time:.4f}s")

                # 堆叠为 (bsz, num_heads, k_effm)
                indices = torch.stack(indices_list, dim=0)
                
                # indices = indices.unsqueeze(-1).expand(-1, origin_key_states.size(1), -1, head_dim)

                # Merging
                target_key_states = origin_key_states[:, :, self.prompt_len:-current_step_len, :]
                target_value_states = origin_value_states[:, :, self.prompt_len:-current_step_len, :]

                mask = torch.zeros(target_key_states.shape[:-1], dtype=torch.bool).to(target_key_states.device)
                mask = mask.scatter(-1, indices, 1)

                k_hh_recent = target_key_states[mask].view(bsz, num_heads, -1, head_dim)
                v_hh_recent = target_value_states[mask].view(bsz, num_heads, -1, head_dim)
                
                # applying merge here
                # breakpoint()
                k_hh_pruned = target_key_states[~mask].view(bsz, num_heads, -1, head_dim)
                v_hh_pruned = target_value_states[~mask].view(bsz, num_heads, -1, head_dim)
                
                # 计算语义相似度（余弦相似度）
                k_hh_pruned_norm = k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, head_dim)
                k_hh_recent_norm = k_hh_recent / torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, head_dim)

                # 根据step_lens在k_hh_recent_norm中对每个step求centroid
                # 使用向量化计算step起始索引（只计算一次，后续复用）
                step_lens_tensor = torch.tensor(step_lens, device=k_hh_recent_norm.device)
                step_start_indices_all = torch.cat([torch.zeros(1, dtype=torch.long, device=k_hh_recent_norm.device), 
                                                    torch.cumsum(step_lens_tensor[:-1], dim=0)])
                
                # 向量化计算所有centroids
                centroids = []
                for i in range(len(step_lens)):  # 包括所有step
                    start_idx = step_start_indices_all[i].item()
                    end_idx = start_idx + step_lens[i]
                    # 在最后一维求均值得到centroid
                    step_centroid = k_hh_recent_norm[:, :, start_idx:end_idx, :].mean(dim=2, keepdim=True)
                    centroids.append(step_centroid)
                
                # 拼接所有centroids: (bsz, num_heads, n_steps, head_dim)
                centroids = torch.cat(centroids, dim=2)

                # 计算k_hh_pruned_norm与centroids的相似度
                # k_hh_pruned_norm: (bsz, num_heads, n_pruned, head_dim)
                # centroids: (bsz, num_heads, n_steps, head_dim)
                # similarity_to_centroids: (bsz, num_heads, n_pruned, n_steps)
                similarity_to_centroids = k_hh_pruned_norm @ centroids.transpose(-1, -2)
                similarity_to_centroids = similarity_to_centroids.to(k_hh_recent.dtype)

                max_values, max_indices = similarity_to_centroids.max(dim=-1)  # (bsz, num_heads, n_pruned)

                # 根据EMA阈值统计大于阈值数量最多的一个step
                # 首先需要设定一个阈值用于判断
                if self.threshold == None:
                    # 初始化阈值为相似度的平均值
                    self.threshold = max_values.mean()
                else:
                    # 使用EMA更新阈值
                    self.threshold = 0.3 * self.threshold + 0.7 * max_values.mean()
                
                # 对每个step统计大于阈值的token数量
                # (bsz, num_heads, n_pruned, n_steps) -> (bsz, n_steps)
                above_threshold = (similarity_to_centroids > self.threshold).float().mean(dim=1).sum(dim=1)
                
                # 找到数量最多的step索引
                most_similar_step_idx = above_threshold.argmax(dim=-1)  # (bsz,)
                
                # 提取最相似step在k_hh_recent中的token（复用step_start_indices_all）
                # 计算每个pruned token与该step中所有token的相似度，然后进行合并
                for b in range(bsz):
                    step_idx = most_similar_step_idx[b].item()
                    start_idx = step_start_indices_all[step_idx].item()
                    end_idx = start_idx + step_lens[step_idx]
                    
                    # 提取该step的所有token: (num_heads, step_len, head_dim)
                    step_tokens = k_hh_recent_norm[b, :, start_idx:end_idx, :]
                    
                    # 计算pruned token与该step中所有token的相似度
                    # (num_heads, n_pruned, head_dim) @ (num_heads, head_dim, step_len) -> (num_heads, n_pruned, step_len)
                    similarity_to_step_tokens = k_hh_pruned_norm[b] @ step_tokens.transpose(-1, -2)
                    similarity_to_step_tokens = similarity_to_step_tokens.to(k_hh_recent.dtype)

                    # 对每个pruned token找到最相似的step token
                    max_values, max_indices = similarity_to_step_tokens.max(dim=-1)  # (num_heads, n_pruned)
                    
                    # 使用第二个EMA阈值
                    if self.threshold_token is None:
                        # 初始化第二个阈值为token级别相似度的平均值
                        self.threshold_token = max_values.mean()
                    else:
                        # 使用EMA更新第二个阈值
                        self.threshold_token = 0.3 * self.threshold_token + 0.7 * max_values.mean()
                    
                    # 筛选相似度大于第二个阈值的token
                    filter_mask = max_values.mean(0) >= self.threshold_token  # (n_pruned,)
                    
                    if filter_mask.sum().item() > 0:
                        # 将相对索引转换为绝对索引
                        absolute_indices = max_indices[:, filter_mask] + start_idx  # (num_heads, n_filtered)
                        merged_indices = absolute_indices.unsqueeze(-1).repeat(1, 1, head_dim)  # (num_heads, n_filtered, head_dim)
                        
                        # 提取对应的权重和k/v
                        merge_weights = max_values[:, filter_mask].unsqueeze(-1).repeat(1, 1, head_dim)  # (num_heads, n_filtered, head_dim)
                        k_hh_merged = k_hh_pruned[b, :, filter_mask, :]  # (num_heads, n_filtered, head_dim)
                        v_hh_merged = v_hh_pruned[b, :, filter_mask, :]  # (num_heads, n_filtered, head_dim)
                        
                        # 执行合并操作
                        k_hh_recent[b:b+1] = torch.scatter_reduce(
                            input=k_hh_recent[b:b+1], 
                            dim=2, 
                            index=merged_indices.unsqueeze(0), 
                            src=merge_weights.unsqueeze(0) * k_hh_merged.unsqueeze(0), 
                            reduce='mean', 
                            include_self=True
                        )
                        
                        v_hh_recent[b:b+1] = torch.scatter_reduce(
                            input=v_hh_recent[b:b+1], 
                            dim=2, 
                            index=merged_indices.unsqueeze(0), 
                            src=merge_weights.unsqueeze(0) * v_hh_merged.unsqueeze(0), 
                            reduce='mean', 
                            include_self=True
                        )
        
        # support gqa
        if self.aggregation == 'all' or 'group':
            k_prompt = origin_key_states[:, :, :self.prompt_len, :]
            v_prompt = origin_value_states[:, :, :self.prompt_len, :]

            if selected_indices[0].shape[-1] <= self.budget_cot - current_step_len:
        
                k_past_compress = origin_key_states[:, :, self.prompt_len:-current_step_len, :].gather(dim = 2, index = indices)
                v_past_compress = origin_value_states[:, :, self.prompt_len:-current_step_len, :].gather(dim = 2, index = indices)

            else:
            
                k_past_compress = k_hh_recent
                v_past_compress = v_hh_recent

            # print(origin_key_states[:, :, self.prompt_len:-selectors.shape[-2], :].shape, "k_past_compress_shape")

            k_cur = origin_key_states[:, :, -current_step_len:, :]
            v_cur = origin_value_states[:, :, -current_step_len:, :]

        else:
            k_prompt = key_states[:, :, :self.prompt_len, :]
            v_prompt = value_states[:, :, :self.prompt_len, :]

            k_past_compress = key_states[:, :, self.prompt_len:, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, self.prompt_len:, :].gather(dim = 2, index = indices)

            # k_cur = key_states[:, :, -self.R:, :]
            # v_cur = value_states[:, :, -self.R:, :]

        key_states = torch.cat([k_prompt, k_past_compress, k_cur], dim = 2)
        value_states = torch.cat([v_prompt, v_past_compress, v_cur], dim = 2)


        return key_states, value_states, step_lens
   

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