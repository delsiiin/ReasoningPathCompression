#!/usr/bin/env python3
"""
增强版可视化脚本：绘制 topk_indices 和 topk_indices_induced 的比较
包括更好的可视化效果和统计分析
支持命令行参数配置模型类型和输出层
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from glob import glob

# 设置matplotlib参数
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_tensor_data(base_dir, model_type, layer_idx, data_type, observation_length=1024, topk=512):
    """
    加载指定层的张量数据
    
    Args:
        base_dir: 基础目录路径
        model_type: 模型类型 (如 'llama3', 'qwen2')
        layer_idx: 层索引
        data_type: 数据类型 ('fullkv' 或 'induced')
        observation_length: 观察长度 (默认: 1024)
        topk: topk值 (默认: 512)
    """
    file_pattern = f"topk_indices_layer_{layer_idx}_observe_{observation_length}_top_{topk}.pt"
    dir_path = os.path.join(base_dir, model_type, data_type)
    file_path = os.path.join(dir_path, file_pattern)
    
    if os.path.exists(file_path):
        tensor = torch.load(file_path, map_location='cpu')
        return tensor
    else:
        print(f"File not found: {file_path}")
        return None

def plot_comparison_enhanced(base_dir, model_type, layer_idx, output_dir, observation_length=1024, topk=512):
    """绘制增强版比较图，包含更多统计信息"""
    
    # 加载两个目录的数据
    tensor1 = load_tensor_data(base_dir, model_type, layer_idx, 'fullkv', observation_length, topk)
    tensor2 = load_tensor_data(base_dir, model_type, layer_idx, 'induced', observation_length, topk)
    
    if tensor1 is None or tensor2 is None:
        print(f"Cannot load data for layer {layer_idx}")
        return
    
    # 确保张量形状正确 - 动态检查topk大小
    expected_shape = (1, 8, topk)
    assert tensor1.shape == expected_shape, f"Tensor1 shape incorrect: {tensor1.shape}, expected: {expected_shape}"
    assert tensor2.shape == expected_shape, f"Tensor2 shape incorrect: {tensor2.shape}, expected: {expected_shape}"
    
    # 创建8个子图
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'{model_type.upper()} Layer {layer_idx} - TopK Indices Comparison\n' + 
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
        ax.set_xlabel(f'Selected Index (0-{topk-1})')
        ax.set_ylabel(f'Position in Sequence (0-{observation_length-1})')
        ax.set_ylim(-10, observation_length + 10)
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
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"topk_comparison_enhanced_{model_type}_layer_{layer_idx}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced comparison saved to: {save_path}")
    plt.close()  # 关闭图形以节省内存

def plot_distribution_comparison(base_dir, model_type, layer_idx, output_dir, observation_length=1024, topk=512):
    """绘制位置分布比较图"""
    
    # 加载数据
    tensor1 = load_tensor_data(base_dir, model_type, layer_idx, 'fullkv', observation_length, topk)
    tensor2 = load_tensor_data(base_dir, model_type, layer_idx, 'induced', observation_length, topk)
    
    if tensor1 is None or tensor2 is None:
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'{model_type.upper()} Layer {layer_idx} - Position Distribution Histograms', fontsize=16)
    
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
    
    save_path = os.path.join(output_dir, f"topk_distribution_{model_type}_layer_{layer_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distribution comparison saved to: {save_path}")
    plt.close()

def analyze_layer_statistics(base_dir, model_type, layer_idx, observation_length=1024, topk=512):
    """分析单层的统计信息"""
    
    # 加载数据
    tensor1 = load_tensor_data(base_dir, model_type, layer_idx, 'fullkv', observation_length, topk)
    tensor2 = load_tensor_data(base_dir, model_type, layer_idx, 'induced', observation_length, topk)
    
    if tensor1 is None or tensor2 is None:
        return
    
    print(f"\n=== {model_type.upper()} Layer {layer_idx} Statistics ===")
    
    for head_idx in range(8):
        data1 = tensor1[0, head_idx, :].numpy()
        data2 = tensor2[0, head_idx, :].numpy()
        
        # 计算统计信息
        overlap = np.intersect1d(data1, data2)
        overlap_ratio = len(overlap) / len(data1)
        
        mean1, std1 = np.mean(data1), np.std(data1)
        mean2, std2 = np.mean(data2), np.std(data2)
        
        print(f"Head {head_idx}:")
        print(f"  Overlap: {len(overlap)}/{topk} ({overlap_ratio:.2%})")
        print(f"  topk_indices: mean={mean1:.1f}, std={std1:.1f}")
        print(f"  topk_indices_induced: mean={mean2:.1f}, std={std2:.1f}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='绘制 topk_indices 和 topk_indices_induced 的比较图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --model llama3 --layers 0 1 15 16
  %(prog)s --model qwen2 --layers 5-10 --output-dir ./results
  %(prog)s --model llama3 --all-layers
        """
    )
    
    # 必要参数
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='模型类型 (如: llama3, qwen2)')
    
    # 层选择参数 (互斥组)
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument('--layers', '-l', type=str, nargs='+',
                            help='要处理的层索引，可以是单个数字、范围(如5-10)或多个值')
    layer_group.add_argument('--all-layers', action='store_true',
                            help='处理所有可用的层')
    
    # 可选参数
    parser.add_argument('--base-dir', '-b', type=str,
                        default='/home/yangx/ReasoningPathCompression/observation/topk_indices',
                        help='数据文件的基础目录路径 (默认: %(default)s)')
    
    parser.add_argument('--output-dir', '-o', type=str,
                        help='输出图片的目录 (默认: 基础目录下的模型子目录)')
    
    parser.add_argument('--observation-length', type=int, default=1024,
                        help='观察序列长度 (默认: %(default)s)')
    
    parser.add_argument('--topk', type=int, default=512,
                        help='TopK值，即选择的位置数量 (默认: %(default)s)')
    
    parser.add_argument('--skip-distribution', action='store_true',
                        help='跳过分布图的生成')
    
    parser.add_argument('--skip-statistics', action='store_true',
                        help='跳过统计信息的打印')
    
    return parser.parse_args()

def parse_layer_specification(layer_specs):
    """解析层规格说明，支持范围和单个值"""
    layers = []
    
    for spec in layer_specs:
        if '-' in spec and not spec.startswith('-'):
            # 处理范围，如 "5-10"
            start, end = map(int, spec.split('-', 1))
            layers.extend(range(start, end + 1))
        else:
            # 处理单个值
            layers.append(int(spec))
    
    return sorted(set(layers))  # 去重并排序

def get_available_layers(base_dir, model_type, observation_length=1024, topk=512):
    """获取指定模型的所有可用层"""
    fullkv_dir = os.path.join(base_dir, model_type, 'fullkv')
    if not os.path.exists(fullkv_dir):
        return []
    
    # 查找所有匹配的文件
    pattern = os.path.join(fullkv_dir, f"topk_indices_layer_*_observe_{observation_length}_top_{topk}.pt")
    files = glob(pattern)
    
    # 提取层号
    layers = []
    for file_path in files:
        filename = os.path.basename(file_path)
        # 从文件名中提取层号
        parts = filename.split('_')
        if len(parts) >= 4 and parts[2] == 'layer':
            try:
                layer_idx = int(parts[3])
                layers.append(layer_idx)
            except ValueError:
                continue
    
    return sorted(layers)

if __name__ == "__main__":
    args = parse_arguments()
    
    # 检查基础目录是否存在
    if not os.path.exists(args.base_dir):
        print(f"错误: 基础目录不存在: {args.base_dir}")
        exit(1)
    
    # 检查模型目录是否存在
    model_dir = os.path.join(args.base_dir, args.model)
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        exit(1)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = model_dir
    
    # 确定要处理的层
    if args.all_layers:
        layers_to_plot = get_available_layers(args.base_dir, args.model, args.observation_length, args.topk)
        if not layers_to_plot:
            print(f"错误: 在 {model_dir} 中未找到任何数据文件")
            exit(1)
        print(f"找到 {len(layers_to_plot)} 个可用层: {layers_to_plot}")
    else:
        layers_to_plot = parse_layer_specification(args.layers)
    
    print(f"开始处理模型: {args.model}")
    print(f"观察长度: {args.observation_length}")
    print(f"TopK值: {args.topk}")
    print(f"要处理的层: {layers_to_plot}")
    print(f"输出目录: {output_dir}")
    
    # 处理每一层
    success_count = 0
    for layer_idx in layers_to_plot:
        print(f"\n处理第 {layer_idx} 层...")
        
        try:
            # 生成增强版比较图
            plot_comparison_enhanced(args.base_dir, args.model, layer_idx, output_dir, 
                                   args.observation_length, args.topk)
            
            # 生成分布比较图
            if not args.skip_distribution:
                plot_distribution_comparison(args.base_dir, args.model, layer_idx, output_dir,
                                           args.observation_length, args.topk)
            
            # 打印统计信息
            if not args.skip_statistics:
                analyze_layer_statistics(args.base_dir, args.model, layer_idx,
                                        args.observation_length, args.topk)
            
            success_count += 1
            
        except Exception as e:
            print(f"处理第 {layer_idx} 层时出错: {e}")
            continue
    
    print(f"\n完成! 成功处理了 {success_count}/{len(layers_to_plot)} 个层")
    print(f"所有图片已保存到: {output_dir}")
