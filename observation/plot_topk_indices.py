#!/usr/bin/env python3
"""
增强版可视化脚本：绘制 topk_indices 和 topk_indices_induced 的比较
包括更好的可视化效果和统计分析
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# 设置matplotlib参数
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_tensor_data(dir_path, layer_idx):
    """加载指定层的张量数据"""
    file_pattern = f"topk_indices_layer_{layer_idx}_observe_1024_top_512.pt"
    file_path = os.path.join(dir_path, file_pattern)
    
    if os.path.exists(file_path):
        tensor = torch.load(file_path, map_location='cpu')
        return tensor
    else:
        print(f"File not found: {file_path}")
        return None

def plot_comparison_enhanced(layer_idx, save_path=None):
    """绘制增强版比较图，包含更多统计信息"""
    
    # 定义两个目录路径
    dir1 = "/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/fullkv"
    dir2 = "/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/induced"
    
    # 加载两个目录的数据
    tensor1 = load_tensor_data(dir1, layer_idx)
    tensor2 = load_tensor_data(dir2, layer_idx)
    
    if tensor1 is None or tensor2 is None:
        print(f"Cannot load data for layer {layer_idx}")
        return
    
    # 确保张量形状正确
    assert tensor1.shape == (1, 8, 512), f"Tensor1 shape incorrect: {tensor1.shape}"
    assert tensor2.shape == (1, 8, 512), f"Tensor2 shape incorrect: {tensor2.shape}"
    
    # 创建8个子图
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Layer {layer_idx} - TopK Indices Comparison\n' + 
                 'Blue: topk_indices | Red: topk_indices_induced | Green: Overlap', 
                 fontsize=16, y=0.98)
    
    # 展平axes数组便于索引
    axes = axes.flatten()
    
    # 统计信息
    overlap_stats = []
    
    # 为每个子图（对应张量的第二个维度）绘制散点图
    for head_idx in range(8):
        ax = axes[head_idx]
        
        # 提取数据 (1, 8, 512) -> (512,)
        data1 = tensor1[0, head_idx, :].numpy()
        data2 = tensor2[0, head_idx, :].numpy()
        
        # 计算重叠
        overlap = np.intersect1d(data1, data2)
        overlap_ratio = len(overlap) / len(data1)
        overlap_stats.append(overlap_ratio)
        
        # 创建x坐标（表示在512个位置中的索引）
        x_coords1 = np.arange(len(data1))
        x_coords2 = np.arange(len(data2))
        
        # 绘制散点图 - 使用更小的点
        ax.scatter(x_coords1, data1, alpha=0.6, s=0.5, label='topk_indices', color='blue')
        ax.scatter(x_coords2, data2, alpha=0.6, s=0.5, label='topk_indices_induced', color='red')
        
        # 高亮重叠的点
        if len(overlap) > 0:
            # 找到重叠点在原数据中的位置
            overlap_mask1 = np.isin(data1, overlap)
            overlap_mask2 = np.isin(data2, overlap)
            
            ax.scatter(x_coords1[overlap_mask1], data1[overlap_mask1], 
                      s=1, color='green', alpha=0.8, label='Overlap')
        
        # 设置子图标题和标签
        ax.set_title(f'Head {head_idx} (Overlap: {overlap_ratio:.2%})')
        ax.set_xlabel('Selected Index (0-511)')
        ax.set_ylabel('Position in Sequence (0-1023)')
        ax.set_ylim(-10, 1033)
        ax.grid(True, alpha=0.3)
        
        # 只在第一个子图显示图例
        if head_idx == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # 在图的底部添加统计信息
    stats_text = f"Average Overlap Ratio: {np.mean(overlap_stats):.2%} | " + \
                f"Min: {np.min(overlap_stats):.2%} | Max: {np.max(overlap_stats):.2%}"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, weight='bold')
    
    # 保存图片
    if save_path is None:
        save_path = f"/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/topk_comparison_enhanced_layer_{layer_idx}.png"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced comparison saved to: {save_path}")
    plt.close()  # 关闭图形以节省内存

def plot_distribution_comparison(layer_idx):
    """绘制位置分布比较图"""
    
    # 定义两个目录路径
    dir1 = "/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/fullkv"
    dir2 = "/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/induced"
    
    # 加载数据
    tensor1 = load_tensor_data(dir1, layer_idx)
    tensor2 = load_tensor_data(dir2, layer_idx)
    
    if tensor1 is None or tensor2 is None:
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Layer {layer_idx} - Position Distribution Histograms', fontsize=16)
    
    axes = axes.flatten()
    
    for head_idx in range(8):
        ax = axes[head_idx]
        
        data1 = tensor1[0, head_idx, :].numpy()
        data2 = tensor2[0, head_idx, :].numpy()
        
        # 绘制直方图
        ax.hist(data1, bins=50, alpha=0.6, label='topk_indices', color='blue', density=True)
        ax.hist(data2, bins=50, alpha=0.6, label='topk_indices_induced', color='red', density=True)
        
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Position in Sequence')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = f"/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/topk_distribution_layer_{layer_idx}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distribution comparison saved to: {save_path}")
    plt.close()

def analyze_layer_statistics(layer_idx):
    """分析单层的统计信息"""
    
    # 定义两个目录路径
    dir1 = "/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/fullkv"
    dir2 = "/home/yangx/ReasoningPathCompression/observation/topk_indices/llama3/induced"

    # 加载数据
    tensor1 = load_tensor_data(dir1, layer_idx)
    tensor2 = load_tensor_data(dir2, layer_idx)
    
    if tensor1 is None or tensor2 is None:
        return
    
    print(f"\n=== Layer {layer_idx} Statistics ===")
    
    for head_idx in range(8):
        data1 = tensor1[0, head_idx, :].numpy()
        data2 = tensor2[0, head_idx, :].numpy()
        
        # 计算统计信息
        overlap = np.intersect1d(data1, data2)
        overlap_ratio = len(overlap) / len(data1)
        
        mean1, std1 = np.mean(data1), np.std(data1)
        mean2, std2 = np.mean(data2), np.std(data2)
        
        print(f"Head {head_idx}:")
        print(f"  Overlap: {len(overlap)}/512 ({overlap_ratio:.2%})")
        print(f"  topk_indices: mean={mean1:.1f}, std={std1:.1f}")
        print(f"  topk_indices_induced: mean={mean2:.1f}, std={std2:.1f}")

if __name__ == "__main__":
    # 选择要可视化的层
    layers_to_plot = [0, 1, 2, 15, 16, 30, 31]
    
    for layer_idx in layers_to_plot:
        print(f"\nProcessing layer {layer_idx}...")
        
        # 生成增强版比较图
        plot_comparison_enhanced(layer_idx)
        
        # 生成分布比较图
        plot_distribution_comparison(layer_idx)
        
        # 打印统计信息
        analyze_layer_statistics(layer_idx)
    
    print(f"\n所有图片已保存到 /home/yangx/ReasoningPathCompression/observation/topk_indices 目录")
