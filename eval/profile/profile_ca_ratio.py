import json
import argparse
from tqdm import tqdm
import os
from pathlib import Path

import matplotlib.pyplot as plt

import copy

import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference on model with prompts from a jsonl file")
    parser.add_argument("--data_path", type=str, required=True, help="Data path")
    args = parser.parse_args()

    if "aime" in args.data_path.lower():
        task = "aime"
    elif "gsm8k" in args.data_path.lower():
        task = "gsm8k"
    elif "math500" in args.data_path.lower():
        task = "math500"
    elif "gpqa" in args.data_path.lower():
        task = "gpqa"
    
    # Load dataset
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    total_ca_ratio = 0.0

    ca_ratios = []

    for item in data:

        index = item['gen'][0].find("</think>")

        CoT = index + len("</think>")
        AnS = len(item['gen'][0]) - CoT

        ca_ratio = CoT / AnS
        ca_ratios.append(ca_ratio)
        total_ca_ratio += ca_ratio

    avg_ca_ratio = total_ca_ratio / len(data)
            
    print(f"The Average CoT to Answer Ratio for Task {task} is: {avg_ca_ratio}.")

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(ca_ratios, marker='o', markersize=2, label='ca_ratio')
    plt.axhline(y=avg_ca_ratio, color='red', linestyle='--', label=f'Avg: {avg_ca_ratio:.2f}')
    plt.title("CoT-to-Answer Ratio")
    plt.xlabel("Sample")
    plt.ylabel("ca_ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"/home/yangx/ReasoningPathCompression/eval/profile/{task}_ca_ratio.pdf")