"""
高效的 step_lens 更新和索引操作的优化实现
使用 Triton 和 PyTorch 优化版本
"""

import torch
import time
from typing import List, Tuple, Optional

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available, falling back to PyTorch optimized version")


def update_step_lens_optimized(
    step_lens: List[int],
    retained_indices: torch.Tensor,
    device: torch.device
) -> List[int]:
    """
    高效更新 step_lens 的优化版本
    
    Args:
        step_lens: 原始的step长度列表
        retained_indices: 保留的索引张量 [num_retained]
        device: 设备
    
    Returns:
        更新后的step_lens列表
    """
    if len(step_lens) == 0:
        return step_lens
    
    # 使用torch.cumsum高效计算step的起始索引
    step_lens_tensor = torch.tensor(step_lens[:-1], device=device, dtype=torch.long)
    step_start_indices = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long), 
        torch.cumsum(step_lens_tensor, dim=0)
    ])
    # 正确计算step_end_indices: 每个step的结束位置 = 下一个step的开始位置
    step_end_indices = step_start_indices[1:]
    
    # # 向量化计算每个step中保留的token数量
    # if TRITON_AVAILABLE and retained_indices.numel() > 1000:  # 大数据时使用Triton
    #     retained_counts = _count_tokens_per_step_triton(
    #         retained_indices, step_start_indices[:-1], step_end_indices
    #     )
    # else:
    # PyTorch优化版本
    retained_counts = _count_tokens_per_step_pytorch(
        retained_indices, step_start_indices[:-1], step_end_indices
    )
    
    # 转换为列表，只保留非零长度的step
    new_step_lens = []
    for count in retained_counts:
        count_val = count.item() if isinstance(count, torch.Tensor) else count
        if count_val > 0:
            new_step_lens.append(count_val)
    
    # 保留最后一个step（recent tokens）
    new_step_lens.append(step_lens[-1])
    return new_step_lens


def _count_tokens_per_step_pytorch(
    retained_indices: torch.Tensor,
    step_starts: torch.Tensor,
    step_ends: torch.Tensor
) -> torch.Tensor:
    """PyTorch优化版本的token计数"""
    # 使用broadcasting计算每个index属于哪个step
    # retained_indices: [num_retained], step_starts: [num_steps]
    step_masks = (retained_indices.unsqueeze(1) >= step_starts.unsqueeze(0)) & \
                (retained_indices.unsqueeze(1) < step_ends.unsqueeze(0))
    
    # 统计每个step的保留数量
    return step_masks.sum(dim=0)


if TRITON_AVAILABLE:
    @triton.jit
    def _count_tokens_kernel(
        retained_indices_ptr,
        step_starts_ptr,
        step_ends_ptr,
        counts_ptr,
        num_retained: tl.constexpr,
        num_steps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for counting tokens per step"""
        step_id = tl.program_id(0)
        
        if step_id >= num_steps:
            return
        
        # Load step boundaries
        step_start = tl.load(step_starts_ptr + step_id)
        step_end = tl.load(step_ends_ptr + step_id)
        
        count = 0
        
        # Process retained indices in blocks
        for block_start in range(0, num_retained, BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, num_retained)
            block_size = block_end - block_start
            
            # Load a block of retained indices
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < block_end
            indices = tl.load(retained_indices_ptr + offsets, mask=mask, other=0)
            
            # Check which indices fall within this step
            in_step = (indices >= step_start) & (indices < step_end) & mask
            count += tl.sum(in_step.to(tl.int32))
        
        # Store result
        tl.store(counts_ptr + step_id, count)

    def _count_tokens_per_step_triton(
        retained_indices: torch.Tensor,
        step_starts: torch.Tensor,
        step_ends: torch.Tensor
    ) -> torch.Tensor:
        """Triton优化版本的token计数"""
        num_retained = retained_indices.numel()
        num_steps = step_starts.numel()
        
        # 创建输出张量
        counts = torch.zeros(num_steps, dtype=torch.int32, device=retained_indices.device)
        
        # 选择合适的block size
        BLOCK_SIZE = triton.next_power_of_2(min(1024, num_retained))
        
        # 启动kernel
        grid = (num_steps,)
        _count_tokens_kernel[grid](
            retained_indices,
            step_starts,
            step_ends,
            counts,
            num_retained=num_retained,
            num_steps=num_steps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return counts.long()


def build_final_indices_optimized(
    bsz: int,
    num_heads: int,
    prompt_len: int,
    compress_indices: torch.Tensor,
    seq_len: int,
    recent_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    高效构建final_indices
    
    Args:
        bsz: batch size
        num_heads: 注意力头数
        prompt_len: prompt长度
        compress_indices: 压缩后的索引
        seq_len: 序列总长度
        recent_len: recent tokens长度
        device: 设备
    
    Returns:
        final_indices张量 [bsz, num_heads, final_len]
    """
    # 计算最终索引的总长度
    final_len = prompt_len + compress_indices.size(-1) + recent_len
    
    # 预分配final_indices张量
    final_indices = torch.empty(bsz, num_heads, final_len, dtype=torch.long, device=device)
    
    # 高效填充各部分索引
    # Prompt部分
    final_indices[:, :, :prompt_len] = torch.arange(prompt_len, device=device).view(1, 1, -1)
    
    # 压缩部分
    compress_end = prompt_len + compress_indices.size(-1)
    final_indices[:, :, prompt_len:compress_end] = (compress_indices + prompt_len).unsqueeze(0)
    
    # Recent部分
    final_indices[:, :, compress_end:] = torch.arange(
        seq_len - recent_len, seq_len, device=device
    ).view(1, 1, -1)
    
    return final_indices


def efficient_gather_operation(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    final_indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    高效的gather操作
    
    Args:
        key_states: key状态张量
        value_states: value状态张量
        final_indices: 索引张量
    
    Returns:
        压缩后的(key_states, value_states)
    """
    # 扩展索引维度以便gather操作
    gather_indices = final_indices.unsqueeze(-1).expand(
        -1, key_states.size(1), -1, key_states.size(-1)
    )
    
    # 进行压缩
    key_compressed = key_states.gather(dim=2, index=gather_indices)
    value_compressed = value_states.gather(dim=2, index=gather_indices)
    
    return key_compressed, value_compressed


def apply_step_weights(token_scores, step_lens, step_start_indices, topk_step_indices, topk_step_values):
    """
    Apply weights to token scores based on step values in a vectorized manner.

    Args:
        token_scores (torch.Tensor): The scores for each token. Shape: (bsz, n_head, L)
        step_lens (list): A list of lengths for each step.
        step_start_indices (list): A list of start indices for each step.
        topk_step_indices (torch.Tensor): The indices of the top-k steps. Shape: (bsz, k_steps)
        topk_step_values (torch.Tensor): The values of the top-k steps. Shape: (bsz, k_steps)

    Returns:
        torch.Tensor: The token scores with applied weights.
    """
    bsz, n_head, L = token_scores.shape
    device = token_scores.device

    # 1. Convert lists to tensors
    step_lens_tensor = torch.tensor(step_lens, device=device)
    step_start_indices_tensor = torch.tensor(step_start_indices, device=device)

    # 2. Get start positions and lengths for each top-k step
    selected_step_starts = step_start_indices_tensor[topk_step_indices]  # (bsz, k_steps)
    selected_step_lens = step_lens_tensor[topk_step_indices]            # (bsz, k_steps)

    # 3. Build indices and weights
    max_len = selected_step_lens.max()
    seq_indices = torch.arange(max_len, device=device).unsqueeze(0).unsqueeze(0) # (1, 1, max_len)

    # Create a mask to identify valid tokens within each step
    mask = seq_indices < selected_step_lens.unsqueeze(-1)  # (bsz, k_steps, max_len)

    # Calculate absolute token positions in the L dimension
    token_indices = selected_step_starts.unsqueeze(-1) + seq_indices  # (bsz, k_steps, max_len)

    # 4. Create weight tensor and populate using scatter_
    step_weights_expanded = torch.ones_like(token_scores)
    
    # Prepare values and indices for scatter
    values_to_scatter = topk_step_values.unsqueeze(-1).expand(-1, -1, max_len)
    
    # Filter valid values and indices
    valid_values = values_to_scatter[mask]
    valid_indices = token_indices[mask]

    # Prepare batch indices for scatter
    bsz_indices = torch.arange(bsz, device=device).view(bsz, 1, 1).expand(-1, topk_step_indices.shape[1], max_len)[mask]
    
    # Use a temporary tensor for scattering, ensuring dtype matches token_scores
    temp_weights = torch.ones(bsz, n_head, L, device=device, dtype=token_scores.dtype)
    
    for b in range(bsz):
        batch_mask = (bsz_indices == b)
        if batch_mask.any():
            batch_indices = valid_indices[batch_mask]
            batch_values = valid_values[batch_mask]
            
            # Expand to match n_head dimension
            scatter_indices = batch_indices.unsqueeze(0).expand(n_head, -1)
            # Ensure scatter_values has the same dtype as temp_weights
            scatter_values = batch_values.unsqueeze(0).expand(n_head, -1).to(temp_weights.dtype)
            
            temp_weights[b].scatter_(1, scatter_indices.long(), scatter_values)

    # Assign values from the temporary tensor using a mask
    updated_mask = (temp_weights != 1.0)
    step_weights_expanded[updated_mask] = temp_weights[updated_mask]

    # Apply weights
    return token_scores * step_weights_expanded