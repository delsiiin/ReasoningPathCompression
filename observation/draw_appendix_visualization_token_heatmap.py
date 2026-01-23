import seaborn as sns
import matplotlib.pyplot as plt
import torch
import random
import os
import glob


def draw_multi_heatmaps():
    # 获取所有pt文件
    pt_files = glob.glob("/home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/llama3/attn_weights_layer_*.pt")
    
    # 随机选择8个文件
    selected_files = random.sample(pt_files, 8)
    selected_files.sort()  # 按文件名排序
    
    # 创建2行4列的子图
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Token-level Attention Heatmaps (R1-Llama-8B)', fontsize=16, y=0.995)
    
    # 遍历选择的文件并绘制热力图
    for idx, file_path in enumerate(selected_files):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        # 加载注意力权重
        attn_weights = torch.load(file_path)
        
        # 从文件名中提取层索引
        layer_idx = os.path.basename(file_path).replace('attn_weights_layer_', '').replace('.pt', '')
        
        # 绘制热力图
        sns.heatmap(
            attn_weights.detach().to(torch.float).cpu().numpy(),
            cmap='Blues',
            vmin=0,
            vmax=0.01,
            xticklabels=False,
            yticklabels=False,
            square=True,
            ax=ax,
            cbar=True
        )
        
        ax.set_title(f'Layer {layer_idx}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.tight_layout()
    
    # 保存图像
    output_folder = '/home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/llama3'
    os.makedirs(output_folder, exist_ok=True)
    output_path = f"{output_folder}/multi_layer_heatmap.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='png')
    print(f"图像已保存到: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    # 设置随机种子以便结果可复现（可选）
    random.seed(42)
    
    draw_multi_heatmaps()
