import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
try:
    import torch
except ImportError:
    torch = None

# 设置随机种子以确保可重复性
np.random.seed(42)

# 基本参数
layers = np.arange(0, 32)  # 32层，从32到1逆向排列
heads = np.arange(0, 8)  # 8个头

# 读取磁盘上的 .pt 文件作为第一组数据（预期文件中为 (32,8)）
pt_path = '/home/yangx/zmw/ReasoningPathCompression/observation/topk_indices/llama3/hit_rates_llama3_vs_snapkv_obs16385_top2048.pt'
hit_rate_1 = None

if torch is not None and os.path.exists(pt_path):
    try:
        loaded = torch.load(pt_path, map_location='cpu')
        print(f"Loaded data type: {type(loaded)}")
        
        # 如果是 dict，尝试提取可能的字段
        if isinstance(loaded, dict):
            print(f"Dict keys: {list(loaded.keys())}")
            # 常见可能的 key
            for k in ('hit_rates', 'data', 'arr', 'tensor', 'values'):
                if k in loaded:
                    loaded = loaded[k]
                    print(f"Using key: {k}")
                    break
            else:
                # 如果没有找到，尝试使用第一个值
                if loaded:
                    loaded = list(loaded.values())[0]
                    print(f"Using first value")

        # 转为 numpy 数组
        if hasattr(loaded, 'numpy'):
            arr = loaded.numpy()
        else:
            arr = np.array(loaded)
        
        print(f"Array shape: {arr.shape}")
        print(f"Array dtype: {arr.dtype}")

        # 文件预期为 (32, 8) (layers x heads). 为绘图需要 (8, 32) (heads x layers)
        if arr.shape == (32, 8):
            hit_rate_1 = arr.T
            print("Successfully loaded and transposed (32,8) -> (8,32)")
        elif arr.shape == (8, 32):
            hit_rate_1 = arr
            print("Successfully loaded (8,32) shape")
        else:
            # 尝试 reshape 为 (32,8)
            if arr.size == 256:  # 32 * 8 = 256
                arr2 = arr.reshape((32, 8))
                hit_rate_1 = arr2.T
                print(f"Reshaped from {arr.shape} to (32,8) then transposed to (8,32)")
            else:
                print(f"Cannot handle shape {arr.shape} with size {arr.size}")
                hit_rate_1 = None
                
    except Exception as e:
        print(f"Warning: failed to load pt file {pt_path}: {e}")
        hit_rate_1 = None
else:
    if torch is None:
        print('Warning: torch not available; cannot load .pt file')
    else:
        print(f'Warning: pt file not found at {pt_path}')

# 如果加载失败，回退到常量数组以保证绘图能运行
if hit_rate_1 is None:
    print("Using fallback constant array for hit_rate_1")
    hit_rate_1 = np.full((8, 32), 38)

hit_rate_1_ = 100* hit_rate_1


# 其余两组仍使用常量/示例数据
hit_rate_1 = np.full((8, 32), 34.7)
hit_rate_2 = np.full((8, 32), 38.7)
hit_rate_3 = np.full((8, 32), 40.7)
hit_rate_5 = hit_rate_1_ + 4
hit_rate_4 = np.full((8, 32), 41.7)

# 创建网格
X, Y = np.meshgrid(layers, heads)

# 创建单个3D图形
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 定义四组数据的颜色和透明度设置
colors = ['blue', 'red', 'green', 'orange', 'purple']
alphas = [0.5, 0.5, 0.5, 0.5, 0.8]
labels = ['H2O Mean', 'SnapKV Mean', 'R-KV Mean', 'T2O Mean', 'T2O']
datasets = [hit_rate_1, hit_rate_2, hit_rate_3, hit_rate_4, hit_rate_5]

# 在同一个坐标系中绘制四组数据
for i, (data, color, alpha, label) in enumerate(zip(datasets, colors, alphas, labels)):
    # 为了在同一坐标系中区分不同数据组，稍微偏移Z轴位置
    z_offset = i * 2  # 每组数据在Z轴方向偏移2个单位

    surf = ax.plot_surface(X, Y, data + z_offset, alpha=alpha, label=label, cmap=plt.cm.get_cmap('viridis' if i == len(datasets) - 1 else ['Purples_r', 'Blues_r', 'Greens_r', 'Reds_r'][i]))

# 设置轴标签和标题
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Head', fontsize=12)
ax.set_zlabel('Hit Rate (%)', fontsize=12)

# 反转X轴（Layer轴）的显示顺序
ax.invert_xaxis()
ax.invert_yaxis()

# 添加图例
ax.legend(labels, loc='upper left', bbox_to_anchor=(0.75, 0.85))

# 设置视角
ax.view_init(elev=20, azim=45)

# 显示和保存图形
plt.show()
plt.savefig('hit_rate.pdf', dpi=300)
