import seaborn as sns
import matplotlib.pyplot as plt
import torch
import argparse
import random
import os


def draw_heat_map_grid(model, num_layers):
    # 随机选择8层
    random.seed(43)  # 设置随机种子以保证可重现性
    selected_layers = sorted(random.sample(range(10, min(50, num_layers)), 8))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, layer in enumerate(selected_layers):
        # 加载相似度数据
        attn_weights = torch.load(
            f"/home/yangx/zmw/ReasoningPathCompression/observation/sim_heat_map_token/{model}/similarity_layer_{layer}.pt"
        )
        
        # 选择一个区域进行可视化（可以根据需要调整）
        # 这里选择中间的100x100区域
        start_pos = max(0, attn_weights.shape[0] // 2 - 50)
        end_pos = min(attn_weights.shape[0], start_pos + 100)
        attn_weights_subset = attn_weights[start_pos:end_pos, start_pos:end_pos]
        
        # 绘制热力图
        sns.heatmap(
            attn_weights_subset.detach().to(torch.float).cpu().numpy(),
            cmap='Blues',
            vmin=0,
            vmax=1,
            xticklabels=False,
            yticklabels=False,
            square=True,
            ax=axes[idx],
            cbar=True
        )
        
        axes[idx].set_title(f'Layer {layer}', fontsize=14)
        axes[idx].set_xlabel('Key Position', fontsize=12)
        axes[idx].set_ylabel('Key Position', fontsize=12)
    
    plt.tight_layout()
    
    # 保存为PNG格式
    folder_path = '/home/yangx/zmw/ReasoningPathCompression/observation/'
    os.makedirs(folder_path, exist_ok=True)
    
    output_filename = f"{folder_path}/similarity_grid_{model}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"图像已保存到: {output_filename}")
    print(f"选择的层: {selected_layers}")
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw similarity heat maps in 2x4 grid")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., llama3, qwen2, qwq)")
    parser.add_argument("--num_layers", type=int, required=True, help="The total number of layers in the model")
    args = parser.parse_args()

    draw_heat_map_grid(args.model, args.num_layers)
