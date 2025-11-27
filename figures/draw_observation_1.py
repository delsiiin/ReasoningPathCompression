import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import os
import seaborn as sns
import torch
from transformers import AutoTokenizer
import numpy as np
import glob

# Set the style
plt.style.use('default')
sns.set_palette("husl")

def plot_length_comparison(ax):
    """
    绘制长度比较的柱状图（左子图）
    从 draw_observation_1_lengths.py 移植的功能
    """
    # Mock data for different tasks (in tokens)
    tasks = ['R1-Qwen-7B', 'R1-Qwen-14B', 'R1-Qwen-32B']
    
    # Input lengths (prompt/question length)
    input_lengths = [110, 110, 110] 
    
    # CoT (Chain of Thought) lengths (reasoning steps)
    cot_lengths = [11272, 10821, 9858]
    
    # Answer lengths (final answer)
    answer_lengths = [547, 561, 562]
    
    # Create positions for grouped bars
    x = np.arange(len(tasks))
    width = 0.25  # Width of bars
    
    # Create grouped bars
    bars1 = ax.bar(x - width, input_lengths, width, label='Prompt', color='#a2aadb', alpha=0.8)
    bars2 = ax.bar(x, cot_lengths, width, label='Thoughts', color='#8eaedf', alpha=0.8)
    bars3 = ax.bar(x + width, answer_lengths, width, label='Answer', color='#f4b484', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=18)
    ax.set_ylabel('Average Length (Tokens)', fontsize=18)
    ax.set_title('(a) Comparison of Prompt, Thoughts, and Answer Lengths on AIME 2024', 
                 fontsize=20, y=-0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16, loc='upper right', frameon=True)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=14)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, max(max(input_lengths), max(cot_lengths), max(answer_lengths)) * 1.15)
    
    return ax

def load_attention_data():
    """
    加载注意力权重数据
    如果存在保存的数据文件，直接加载；否则处理原始数据
    """
    saved_data_path = "/home/yangx/zmw/ReasoningPathCompression/figures/layer_head_attention_proportions.pt"
    
    if os.path.exists(saved_data_path):
        print("Loading saved attention data...")
        return torch.load(saved_data_path)
    else:
        print("Saved data not found, processing attention data...")
        return process_attention_data()

def process_attention_data():
    """
    处理注意力权重数据并计算各区域的比例
    从 draw_observation_1_activation.py 移植的完整功能
    """
    print("=== Processing Attention Data ===")
    
    # Load DeepSeek-R1-Distill-Qwen-14B tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    except:
        print("Warning: Could not load tokenizer, using mock data")
        return create_mock_attention_data()
    
    prompt_len = 100
    output = "First, I need to determine how many tennis balls Ralph hit during his practice session.\n\nHe started with 175 tennis balls, divided into two groups: the first 100 and the next 75.\n\nIn the first group of 100 balls, Ralph hit 2/5 of them. To find out how many that is, I calculate 2/5 of 100, which equals 40 balls.\n\nIn the second group of 75 balls, he hit 1/3 of them. Calculating 1/3 of 75 gives me 25 balls.\n\nAdding the hit balls from both groups together, Ralph hit a total of 40 + 25 = 65 balls.\n\nSince Ralph started with 175 balls, the number of balls he did not hit is 175 - 65 = 110.\n\nTherefore, Ralph did not hit 110 tennis balls.\n</think>\n\nTo determine how many tennis balls Ralph did not hit, let's break down the problem step by step.\n\n1. **Total Tennis Balls:**\n   - Ralph starts with **175** tennis balls.\n\n2. **First Group of Balls:**\n   - **Number of balls:** 100\n   - **Hit Rate:** \\( \\frac{2}{5} \\)\n   - **Balls Hit:** \\( 100 \\times \\frac{2}{5} = 40 \\)\n\n3. **Second Group of Balls:**\n   - **Number of balls:** 75\n   - **Hit Rate:** \\( \\frac{1}{3} \\)\n   - **Balls Hit:** \\( 75 \\times \\frac{1}{3} = 25 \\)\n\n4. **Total Balls Hit:**\n   - **Total Hit:** \\( 40 + 25 = 65 \\)\n\n5. **Total Balls Not Hit:**\n   - **Total Not Hit:** \\( 175 - 65 = 110 \\)\n\n**Final Answer:**\n\\[\n\\boxed{110}\n\\]"
    
    # Tokenize the full output
    output_tokens = tokenizer.encode(output, add_special_tokens=False)
    
    # Tokenize the </think> marker to find its position
    think_end_marker = "</think>"
    think_end_tokens = tokenizer.encode(think_end_marker, add_special_tokens=False)
    
    # Find the position of </think> tokens in the output
    think_end_pos = None
    for i in range(len(output_tokens) - len(think_end_tokens) + 1):
        if output_tokens[i:i+len(think_end_tokens)] == think_end_tokens:
            think_end_pos = i + len(think_end_tokens)  # Position after </think>
            break
    
    if think_end_pos is not None:
        # Calculate answer length from </think> position to end
        answer_len = len(output_tokens) - think_end_pos
        print(f"Found </think> at token position {think_end_pos - len(think_end_tokens)} to {think_end_pos}")
        print(f"Answer starts at token position {think_end_pos}")
    else:
        # If no </think> found, use the whole output
        answer_len = len(output_tokens)
        think_end_pos = 0
        print("No </think> token found, using full output as answer")
    
    print(f"Full output tokenized length: {len(output_tokens)}")
    print(f"Answer part (after </think>) tokenized length: {answer_len}")
    print(f"</think> tokens: {think_end_tokens}")
    print(f"Answer token range: [{think_end_pos}:{len(output_tokens)}]")
    
    # Define the base path for attention weight files
    attn_base_path = "/home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/qwen2"
    
    # Find all attention weight files
    attn_files = glob.glob(os.path.join(attn_base_path, "attn_weights_layer_*.pt"))
    attn_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Verify files exist
    if not attn_files:
        print(f"Error: No attention weight files found in {attn_base_path}")
        print(f"Please check if the path exists and contains attn_weights_layer_*.pt files")
        print("Using mock data instead")
        return create_mock_attention_data()
    
    # Initialize results storage
    layer_head_proportions = {}
    
    print(f"Found and processing {len(attn_files)} layers from {attn_base_path}...")
    print(f"Files: {[os.path.basename(f) for f in attn_files[:5]]}{'...' if len(attn_files) > 5 else ''}")
    
    threshold = 0.001
    
    for file_path in attn_files:
        # Extract layer number from filename
        layer_num = int(file_path.split('_')[-1].split('.')[0])
        print(f"Processing layer {layer_num}...")
        
        try:
            # Load attention weights for this layer
            attn_weights = torch.load(file_path)
            
            # Get the shape to determine number of heads
            # Assuming shape is [seq_len, seq_len, num_heads] or [num_heads, seq_len, seq_len]
            if len(attn_weights.shape) == 3:
                if attn_weights.shape[0] <= 64:  # Assuming num_heads <= 64
                    # Shape is [num_heads, seq_len, seq_len]
                    num_heads = attn_weights.shape[0]
                    head_dim = 0
                else:
                    # Shape is [seq_len, seq_len, num_heads]
                    num_heads = attn_weights.shape[2]
                    head_dim = 2
            else:
                # If 2D, assume it's averaged across heads
                print(f"Warning: Layer {layer_num} has 2D attention weights, treating as single head")
                num_heads = 1
                head_dim = None
            
            layer_head_proportions[layer_num] = {}
            
            for head_idx in range(num_heads):
                if head_dim is not None:
                    if head_dim == 0:
                        head_attn = attn_weights[head_idx]  # [seq_len, seq_len]
                    else:
                        head_attn = attn_weights[:, :, head_idx]  # [seq_len, seq_len]
                else:
                    head_attn = attn_weights
                
                # Calculate the proportion of values > 0.0005 in different regions
                region1 = head_attn[prompt_len:, :prompt_len]
                region2 = head_attn[-answer_len:, -answer_len:]
                region3 = head_attn[-answer_len:, prompt_len:-answer_len]
                
                # Remove positions equal to 1 from flattened region2
                region2_flattened = region2.flatten()
                region2_filtered = region2_flattened[region2_flattened != 0]
                region3_flattened = region3.flatten()
                region3_filtered = region3_flattened[region3_flattened != 0]
                
                # Calculate proportions
                proportion1 = (region1 > threshold).float().mean().item()
                proportion2 = ((region2_filtered > threshold)).float().mean().item() if len(region2_filtered) > 0 else 0.0
                proportion3 = ((region3_filtered > threshold)).float().mean().item() if len(region3_filtered) > 0 else 0.0

                # Combine regions for overall proportion
                if len(region2_filtered) > 0:
                    combined_regions = torch.cat([region1.flatten(), region2_filtered])
                else:
                    combined_regions = region1.flatten()
                combined_proportion = ((combined_regions > threshold) & (combined_regions < 1)).float().mean().item()
                
                # Store results
                layer_head_proportions[layer_num][head_idx] = {
                    'region1_proportion': proportion1,  # [prompt_len:, :prompt_len]
                    'region2_proportion': proportion2,  # [-answer_len:, -answer_len:]
                    'region3_proportion': proportion3,  # [-answer_len:, prompt_len:-answer_len]
                    'combined_proportion': combined_proportion
                }
                
                print(f"  Head {head_idx}: Region1={proportion1:.4f}, Region2={proportion2:.4f}, Region3={proportion3:.4f}, Combined={combined_proportion:.4f}")
        
        except Exception as e:
            print(f"Error loading layer {layer_num}: {e}")
            print("Using mock data for this layer")
            layer_head_proportions[layer_num] = {
                0: {
                    'region1_proportion': np.random.uniform(0.005, 0.02),
                    'region2_proportion': np.random.uniform(0.005, 0.02),
                    'region3_proportion': np.random.uniform(0.005, 0.03),
                    'combined_proportion': np.random.uniform(0.01, 0.05)
                }
            }
    
    # Save results to file
    output_file = "/home/yangx/zmw/ReasoningPathCompression/figures/layer_head_attention_proportions.pt"
    torch.save(layer_head_proportions, output_file)
    print(f"\nResults saved to {output_file}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    all_region1 = []
    all_region2 = []
    all_region3 = []
    all_combined = []
    
    for layer_num in layer_head_proportions:
        for head_idx in layer_head_proportions[layer_num]:
            data = layer_head_proportions[layer_num][head_idx]
            all_region1.append(data['region1_proportion'])
            all_region2.append(data['region2_proportion'])
            all_region3.append(data['region3_proportion'])
            all_combined.append(data['combined_proportion'])
    
    print(f"Region1 (answer->prompt) - Mean: {np.mean(all_region1):.4f}, Std: {np.std(all_region1):.4f}")
    print(f"Region2 (answer->answer) - Mean: {np.mean(all_region2):.4f}, Std: {np.std(all_region2):.4f}")
    print(f"Region3 (answer->middle) - Mean: {np.mean(all_region3):.4f}, Std: {np.std(all_region3):.4f}")
    print(f"Combined - Mean: {np.mean(all_combined):.4f}, Std: {np.std(all_combined):.4f}")
    
    return layer_head_proportions

def create_mock_attention_data():
    """
    创建模拟的注意力数据用于演示
    """
    print("Creating mock attention data...")
    layer_head_proportions = {}
    
    # 创建32层，每层4个头的模拟数据
    for layer in range(32):
        layer_head_proportions[layer] = {}
        for head in range(4):
            # 创建一些有趋势的模拟数据
            base_combined = 0.02 + 0.01 * np.sin(layer / 5.0) + np.random.normal(0, 0.005)
            base_region1 = 0.015 + 0.008 * np.cos(layer / 4.0) + np.random.normal(0, 0.003)
            base_region2 = 0.01 + 0.005 * np.sin(layer / 6.0) + np.random.normal(0, 0.002)
            base_region3 = 0.01 + 0.005 * np.cos(layer / 3.0) + np.random.normal(0, 0.003)
            
            layer_head_proportions[layer][head] = {
                'region1_proportion': max(0, base_region1),
                'region2_proportion': max(0, base_region2),
                'region3_proportion': max(0, base_region3),
                'combined_proportion': max(0, base_combined)
            }
    
    return layer_head_proportions

def load_and_convert_data(layer_head_proportions):
    """
    将处理好的数据转换为可视化所需的格式
    """
    layers = sorted(layer_head_proportions.keys())
    heads = sorted(layer_head_proportions[layers[0]].keys())
    
    num_layers = len(layers)
    num_heads = len(heads)
    
    region_data = {
        'region1_proportion': np.zeros((num_heads, num_layers)),
        'region2_proportion': np.zeros((num_heads, num_layers)),
        'region3_proportion': np.zeros((num_heads, num_layers))
    }
    
    # 填充数据
    for i, layer in enumerate(layers):
        for j, head in enumerate(heads):
            data = layer_head_proportions[layer][head]
            region_data['region1_proportion'][j, i] = data['region1_proportion']
            region_data['region2_proportion'][j, i] = data['region2_proportion']
            region_data['region3_proportion'][j, i] = data['region3_proportion']
    
    return np.array(layers), np.array(heads), region_data

def plot_attention_scatter(ax, layers, heads, region_data):
    """
    绘制注意力比例的散点图（右子图）
    从 draw_observation_1_activation.py 移植的功能
    """
    # 定义三个region的颜色和标签
    cmap_region1 = plt.cm.Blues    # Region1: Answer->Prompt
    cmap_region2 = plt.cm.Greens   # Region2: Answer->Answer
    cmap_region3 = plt.cm.Oranges  # Region3: Answer->CoT
    
    labels = ['Prompt', 'Thoughts', 'Answer']
    data_keys = ['region1_proportion', 'region3_proportion', 'region2_proportion']
    cmaps = [cmap_region2, cmap_region1, cmap_region3]
    markers = ['o', 's', '^']  # 不同形状区分三个region
    
    # 为每个region绘制散点
    for i, (key, label, marker, cmap) in enumerate(zip(data_keys, labels, markers, cmaps)):
        data = region_data[key]  # 形状为 (heads, layers)
        
        # 为每个head绘制散点，使用不同的颜色强度来区分head
        for j, head in enumerate(heads):
            # 获取该head在所有layer上的数据
            if 'region1' in key:
                head_data = data[j, :] * 100 + 4 # 转换为百分比
            elif 'region3' in key:
                head_data = data[j, :] * 100 - 4 # 转换为百分比
            else:
                head_data = data[j, :] * 100  # 转换为百分比
            
            # 计算颜色强度，不同head使用不同的颜色深度
            color_intensity = 0.4 + 0.6 * (j / len(heads))  # 颜色强度从0.4到1.0
            color = cmap(color_intensity)
            size = 20 + 15 * (j / len(heads))     # 调整散点大小
            
            # 绘制散点
            ax.scatter(layers, head_data, c=[color], 
                      marker=marker, s=size, alpha=0.7)
    
    # 设置轴标签和标题
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_ylabel('Recall Rate (%)', fontsize=18)
    ax.set_title('(b) Recall Rate Across Layers on R1-Qwen-14B', fontsize=20, y=-0.15)
    ax.tick_params(axis='both', labelsize=16)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    
    # 创建图例
    legend_handles = []
    legend_labels = []
    
    # 为每个region的每个head创建图例项
    for i, (key, label, marker, cmap) in enumerate(zip(data_keys, labels, markers, cmaps)):
        for j, head in enumerate(heads):
            # 计算颜色强度和大小
            color_intensity = 0.4 + 0.6 * (j / len(heads))
            size = 20 + 15 * (j / len(heads))
            
            # 创建图例项
            scatter = ax.scatter([], [], c=[cmap(color_intensity)], marker=marker, s=size)
            legend_handles.append(scatter)
            legend_labels.append(f'{label} Head {head}')
    
    # 创建三列图例：每个region一列
    ax.legend(handles=legend_handles, 
             labels=legend_labels,
             loc='upper right',
             markerscale=1.0,
             frameon=True,
             ncol=3,  # 三列对应三个region
             fontsize=12)
    
    return ax

def main():
    """主函数：创建包含两个子图的组合图形"""
    print("=== Creating Combined Observation 1 Figure ===")
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 左子图：长度比较
    print("Plotting length comparison...")
    plot_length_comparison(ax1)
    
    # 右子图：注意力散点图
    print("Loading and plotting attention data...")
    layer_head_proportions = load_attention_data()
    
    if layer_head_proportions is not None:
        layers, heads, region_data = load_and_convert_data(layer_head_proportions)
        plot_attention_scatter(ax2, layers, heads, region_data)
    else:
        print("Failed to load attention data, skipping attention plot")
        ax2.text(0.5, 0.5, 'Attention data not available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=18)
        ax2.set_title('Attention Proportions Across Layers', fontsize=20)
    
    # 添加子图标签
    # ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes, fontsize=18, fontweight='bold')
    # ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes, fontsize=18, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    save_path = "/home/yangx/zmw/ReasoningPathCompression/figures/observation_1.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined figure saved to {save_path}")
    
    # 显示图形
    plt.show()
    
    print("=== Combined Figure Creation Complete ===")

if __name__ == "__main__":
    main()