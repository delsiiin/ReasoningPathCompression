import torch
import matplotlib.pyplot as plt

# 1. 加载 .pt 文件
tensor_path = '/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/r1-7b/gsm8k/grad_attn_tensor_gsm8k.pt'  # 替换为你的文件路径
tensor = torch.load(tensor_path)

min_val = torch.min(tensor).item()

min_val = round(min_val, 2)

tensor = tensor - min_val

# tensor = tensor.softmax(-1)

tensor = tensor / tensor.sum(1).item()


# 2. 检查形状是否为 (1, 28)
if tensor.shape != (1, 28):
    raise ValueError(f"预期张量形状为 (1, 28)，但实际为 {tensor.shape}")

# 3. 提取数据并转换为一维列表
data = tensor.squeeze().tolist()  # 去除 batch 维，转换为 [28]

# 4. 绘制折线图
plt.figure(figsize=(10, 4))
plt.plot(range(28), data, marker='o', linestyle='-')
plt.title('Tensor Values Line Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/r1-7b/gsm8k/layer_budget.pdf")
