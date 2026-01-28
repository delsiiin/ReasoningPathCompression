# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# 1) 横轴刻度（每个任务可以有不同的横坐标值）
kv_budgets = {
    "AIME24": [128, 256, 512, 1024, 2048, 4096],
    "MATH-500": [128, 256, 512, 1024, 2048],
    "LiveCodeBench": [128, 256, 512, 1024, 2048, 4096],
    "GPQA": [128, 256, 512, 1024, 2048, 4096],
}

# 2) 数据组织方式：
# datasets -> models -> { 'full_cache': float,
#                         'StreamingLLM': [y1..y5],
#                         'H2O': [y1..y5],
#                         'D2O': [y1..y5] }
#
# ⚠️把下面的示例数值替换为你的真实结果即可。
data = {
    "AIME24": {
        "R1-Qwen-7B": {
            "full_cache": 55.5,
            "R-KV":         [5.4, 19.0, 31.8, 53.1, 55.5, 55.5],
            "T2O":         [6.8, 21.6, 33.8, 54.9, 55.5, 55.5],
            "ThinKV":       [6.1, 20.3, 31.2, 55.3, 55.5, 55.5],
        },
        "R1-Qwen-14B": {
            "full_cache": 69.7,
            "R-KV":         [6.8, 15.2, 33, 46.7, 69.7, 69.7],
            "T2O":         [9.2, 18.3, 35.7, 48.8, 69.7, 69.7],
            "ThinKV":       [8.0, 16.8, 32.5, 49.2, 69.7, 69.7],
        },
        "R1-Qwen-32B": {
            "full_cache":72.6,
            "R-KV":         [6.6, 15.9, 41.0, 51.1, 72.6, 72.6],
            "T2O":         [8.1, 20.1, 44.9, 55.3, 72.6, 72.6],
            "ThinKV":       [7.4, 18.0, 45.5, 50.8, 72.6, 72.6],
        },
        "R1-Llama-8B": {
            "full_cache": 50,
            "R-KV":         [0.5, 13.8, 28.9, 46.7, 50, 50],
            "T2O":         [0.8, 16.1, 32.0, 49.4, 50, 51.6],
            "ThinKV":       [0.7, 15.0, 30.5, 48.1, 50, 50.8],
        },
        "GPT-OSS-20B": {
            "full_cache": 73.33,
            "R-KV":         [6.8, 33.4, 54.33, 60.5, 73.33, 73.33],
            "T2O":         [8.2, 34.9, 57.6, 62.8, 73.33, 73.33],
            "ThinKV":       [7.5, 34.2, 56.0, 61.7, 73.33, 73.33],
        },
        "QwQ-32B": {
            "full_cache": 79.5,
            "R-KV":         [6.4, 25.9, 49.8, 55.2, 78.1, 79.5],
            "T2O":         [8.0, 28.5, 51.7, 58.2, 79.5, 79.5],
            "ThinKV":       [7.2, 27.2, 50.8, 56.7, 78.8, 79.5],
        },
        "Qwen3-30B": {
            "full_cache": 97.2,
            "R-KV":         [6.3, 57.3, 70.8, 80.8, 97.2, 97.2],
            "T2O":         [8.5, 61.7, 75.5, 82.7, 97.2, 97.2],
            "ThinKV":       [7.4, 59.5, 73.2, 81.8, 97.2, 97.2],
        },
    },
    "LiveCodeBench": {
        "R1-Qwen-7B": {
            "full_cache": 37.6,
            "R-KV":         [0.9, 15.0, 24.0, 34.4, 37.6, 37.6],
            "T2O":         [1.3, 16.4, 25.9, 36.0, 37.6, 37.6],
            "ThinKV":       [1.1, 15.7, 26.3, 36.4, 37.6, 37.6],
        },
        "R1-Qwen-14B": {
            "full_cache": 53.1,
            "R-KV":         [1.2, 25.4, 30.2, 35.8, 51.0, 53.1],
            "T2O":         [1.8, 28.7, 31.8, 37.2, 53.0, 53.1],
            "ThinKV":       [1.5, 27.1, 31.0, 36.5, 52.0, 53.1],
        },
        "R1-Qwen-32B": {
            "full_cache": 57.2,
            "R-KV":         [1.5, 19.8, 28.2, 37.2, 56.2, 57.2],
            "T2O":         [2.3, 22.4, 31.8, 38.2, 57.2, 57.2],
            "ThinKV":       [1.9, 21.1, 27.8, 38.6, 57.2, 57.2],
        },
        "R1-Llama-8B": {
            "full_cache": 32.14,
            "R-KV":         [2.34, 15.8, 20.67, 28.78, 30.14, 32.14],
            "T2O":         [2.76, 18.4, 22.14, 29.97, 32.14, 32.14],
            "ThinKV":       [2.55, 17.1, 21.4, 29.4, 31.1, 32.14],
        },
        "GPT-OSS-20B": {
            "full_cache": 77.8,
            "R-KV":         [1.2, 22.5, 52.8, 64.4, 76.6, 77.8],
            "T2O":         [1.5, 24.6, 54.8, 66.5, 75.6, 77.8],
            "ThinKV":       [1.4, 23.6, 53.8, 65.5, 76.1, 77.8],
        },
        "QwQ-32B": {
            "full_cache": 62.7,
            "R-KV":         [4.7, 23.8, 29.9, 39.1, 60.9, 62.7],
            "T2O":         [6.3, 31.2, 34.7, 42.4, 62.3, 62.7],
            "ThinKV":       [5.5, 27.5, 32.3, 40.8, 61.6, 62.7],
        },
        "Qwen3-30B": {
            "full_cache": 66.0,
            "R-KV":         [2.1, 27.6, 35.8, 55.8, 66.0, 66.0],
            "T2O":         [2.8, 32.2, 39.2, 57.8, 66.0, 66.0],
            "ThinKV":       [2.5, 29.9, 37.5, 56.8, 66.0, 66.0],
        },
    },
    "MATH-500": {
        "R1-Qwen-7B": {
            "full_cache": 92.8,
            "R-KV":         [42.8, 68.9, 87.3, 91.5, 93.2],
            "T2O":         [44.5, 71.2, 88.7, 92.1, 94.1],
            "ThinKV":       [43.7, 70.1, 86.8, 92.4, 94.3],
        },
        "R1-Qwen-14B": {
            "full_cache": 93.9,
            "R-KV":         [53.4, 74.8, 90.9, 93.9, 94.3],
            "T2O":         [54.0, 76.7, 91.3, 94.6, 95.4],
            "ThinKV":       [53.7, 75.8, 91.6, 94.8, 94.9],
        },
        "R1-Qwen-32B": {
            "full_cache": 94.3,
            "R-KV":         [38.4, 68.5, 86.7, 93.1, 94.7],
            "T2O":         [42.8, 73.2, 89.4, 95.6, 96.8],
            "ThinKV":       [40.6, 70.9, 88.1, 94.4, 95.8],
        },
        "R1-Llama-8B": {
            "full_cache": 89.1,
            "R-KV":         [50.7, 70.9, 86.2, 89.1, 89.6],
            "T2O":         [51.2, 72.8, 86.7, 89.8, 90.5],
            "ThinKV":       [51.0, 71.9, 85.8, 90.1, 90.1],
        },
        "GPT-OSS-20B": {
            "full_cache": 79.8,
            "R-KV":         [44.8, 64.2, 76.9, 79.5, 80.3],
            "T2O":         [46.3, 64.8, 78.1, 80.1, 81.1],
            "ThinKV":       [45.6, 64.5, 77.5, 79.8, 80.7],
        },
        "QwQ-32B": {
            "full_cache": 98.0,
            "R-KV":         [38.4, 69.2, 87.8, 98.0, 98.3],
            "T2O":         [43.8, 74.6, 91.2, 98.0, 99.4],
            "ThinKV":       [41.1, 71.9, 89.5, 98.0, 98.9],
        },
        "Qwen3-30B": {
            "full_cache": 97.2,
            "R-KV":         [41.8, 71.2, 88.6, 94.1, 97.2],
            "T2O":         [45.2, 75.8, 91.2, 95.8, 97.2],
            "ThinKV":       [43.5, 73.5, 89.9, 95.0, 97.2],
        },
    },
    "GPQA": {
        "R1-Qwen-7B": {
            "full_cache": 49.1,
            "R-KV":         [10.8, 22.8, 25.5, 39.8, 48.6, 49.1],
            "T2O":         [15.2, 25.4, 27.9, 40.9, 49.0, 49.1],
            "ThinKV":       [13.0, 24.1, 25.1, 39.2, 48.8, 49.1],
        },
        "R1-Qwen-14B": {
            "full_cache": 59.1,
            "R-KV":         [11.9, 21.2, 37.1, 51.6, 58.7, 59.1],
            "T2O":         [14.2, 25.2, 40.1, 52.6, 59.2, 59.1],
            "ThinKV":       [13.1, 23.2, 38.6, 52.1, 59.0, 59.1],
        },
        "R1-Qwen-32B": {
            "full_cache": 62.1,
            "R-KV":         [20.8, 30.2, 40.9, 53.8, 61.1, 62.1],
            "T2O":         [23.4, 34.6, 42.2, 54.9, 61.9, 62.1],
            "ThinKV":       [22.1, 32.4, 40.3, 55.2, 62.0, 62.1],
        },
        "R1-Llama-8B": {
            "full_cache": 49.0,
            "R-KV":         [18.2, 31.4, 35.3, 40.5, 48.7, 49.0],
            "T2O":         [22.1, 28.7, 35.8, 40.3, 49.1, 49.0],
            "ThinKV":       [20.2, 30.1, 35.6, 40.4, 48.9, 49.0],
        },
        "GPT-OSS-20B": {
            "full_cache": 71.5,
            "R-KV":         [22.9, 42.3, 53.1, 64.7, 70.9, 71.5],
            "T2O":         [28.4, 44.7, 56.8, 65.9, 71.5, 71.5],
            "ThinKV":       [25.7, 43.5, 55.0, 65.3, 71.2, 71.5],
        },
        "QwQ-32B": {
            "full_cache": 65.6,
            "R-KV":         [22.4, 36.8, 48.9, 59.8, 65.2, 65.6],
            "T2O":         [26.7, 41.3, 51.7, 60.9, 65.4, 65.6],
            "ThinKV":       [24.6, 39.1, 52.1, 61.3, 65.3, 65.6],
        },
        "Qwen3-30B": {
            "full_cache": 73.4,
            "R-KV":         [22.4, 37.8, 51.3, 66.8, 72.2, 73.4],
            "T2O":         [25.8, 42.1, 54.7, 68.2, 73.1, 73.4],
            "ThinKV":       [24.1, 40.0, 53.0, 67.5, 72.7, 73.4],
        },
    },
}

# 3) 画图
datasets_order = ["AIME24", "MATH-500", "LiveCodeBench", "GPQA"]
models_order = ["R1-Qwen-7B", "R1-Qwen-14B", "R1-Qwen-32B", "R1-Llama-8B", "GPT-OSS-20B", "QwQ-32B", "Qwen3-30B"]

fig, axes = plt.subplots(
    nrows=len(datasets_order), ncols=len(models_order),
    figsize=(24, 12), sharex=False
)

# 统一的样式（指定颜色以获得更好的区分度）
marker_map = {
    "R-KV": "d",
    "T2O": "*",
    "ThinKV": "s",
}
linestyle_map = {
    "R-KV": "-",
    "T2O": "-",
    "ThinKV": "-",
}
color_map = {
    "R-KV": "#a2dbb3",      # 深蓝色
    "ThinKV": "#8eaedf",       # 深红色
    "T2O": "#f4b484",    # 深绿色
}

# 用于放在整张图上的大图例
handles_for_legend = []
labels_for_legend = []

for r, ds in enumerate(datasets_order):
    for c, model in enumerate(models_order):
        ax = axes[r, c]
        series = data[ds][model]

        # Full Cache 虚线水平线
        fc = series["full_cache"]
        ax.axhline(fc, linestyle="--", linewidth=1.5, label="Full Cache", color="black")

        # 三条方法曲线
        for name in ["R-KV", "ThinKV", "T2O"]:
            y = series[name]
            x = range(len(kv_budgets[ds]))  # 使用等间隔的索引作为横坐标
            (ln,) = ax.plot(
                x, y,
                marker=marker_map[name], linestyle=linestyle_map[name],
                color=color_map[name],
                linewidth=2, markersize=5, label=name,
            )
            # 收集一次即可用于全局图例
            if r == 0 and c == 0:
                handles_for_legend.append(ln)
                labels_for_legend.append(name)

        # 设置横坐标刻度（等间隔位置，显示实际数值）
        ax.set_xticks(range(len(kv_budgets[ds])))
        ax.set_xticklabels(kv_budgets[ds], fontsize=11)
        
        # 轴、标题
        if r == len(datasets_order) - 1:
            ax.set_xlabel("KV Cache Budget", fontsize=12)
        if c == 0:
            # 前两行用 EM，最后一行用 bleu_acc（可按需改成你的指标名）
            ax.set_ylabel("Accuracy (%)", fontsize=12)

        ax.set_title(f"{ds} {model}", fontsize=13)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(True, alpha=0.3)

# 顶部合并图例（含 Full Cache 的虚线）
# 先手动创建一个与 axhline 样式一致的 Line2D 句柄
from matplotlib.lines import Line2D
full_cache_proxy = Line2D([0], [0], linestyle="--", linewidth=1.5, color="black")
handles = [full_cache_proxy] + handles_for_legend
labels = ["Full Cache"] + labels_for_legend

fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给顶部图例留空间
plt.show()
plt.savefig("exp_thinkv.pdf", dpi=300, bbox_inches="tight")
