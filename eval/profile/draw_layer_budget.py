import torch
import matplotlib.pyplot as plt

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run profiling on layer budget")
    parser.add_argument("--grad_path", type=str, required=True, help="Grad path")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    args = parser.parse_args()

    if "aime" in args.grad_path.lower():
        task = "aime"
    elif "gsm8k" in args.grad_path.lower():
        task = "gsm8k"
    elif "math500" in args.grad_path.lower():
        task = "math500"
    elif "gpqa" in args.grad_path.lower():
        task = "gpqa"

    # 1. 加载 .pt 文件
    tensor = torch.load(args.grad_path)

    min_val = torch.min(tensor).item()

    min_val = round(min_val, 2)

    tensor = tensor - min_val

    # tensor = tensor.softmax(-1)

    tensor = tensor / tensor.sum(1).item()

    # 3. 提取数据并转换为一维列表
    data = tensor.squeeze().tolist() 

    # 4. 绘制折线图
    plt.figure(figsize=(10, 4))
    plt.plot(range(28), data, marker='o', linestyle='-')
    plt.title('Layer Budget')
    plt.xlabel('Layer ID')
    plt.ylabel('Mean L2 Norm')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/{args.model}/{task}/layer_budget.pdf")
