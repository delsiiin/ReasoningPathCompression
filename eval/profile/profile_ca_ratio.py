import json
import argparse
from tqdm import tqdm
import os
from pathlib import Path

import matplotlib.pyplot as plt

import copy

import re

from transformers import AutoTokenizer

def draw_ca_ratios(ca_ratios, avg_ca_ratio, args, task):
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
    if args.bbh_subset:
        plt.savefig(f"/home/yangx/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}/{args.bbh_subset}_ca_ratio.pdf")
    else:
        plt.savefig(f"/home/yangx/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}_ca_ratio.pdf")

def draw_lens(CoT_lens, AnS_lens, args, task):
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(CoT_lens, marker='o', markersize=2, label='CoT_lens')
    plt.plot(AnS_lens, marker='s', markersize=2, label='AnS_lens')
    plt.title("CoT and Answer Lengths")
    plt.xlabel("Sample")
    plt.ylabel("Length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    if args.bbh_subset:
        plt.savefig(f"/home/yangx/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}/{args.bbh_subset}_lens.pdf")
    else:
        plt.savefig(f"/home/yangx/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}_lens.pdf")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run profiling on CoT to answer ratio")
    parser.add_argument("--data_path", type=str, required=True, help="Data path")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--bbh_subset", type=str, required=False, help="BBH task type")
    args = parser.parse_args()

    if "aime" in args.data_path.lower():
        task = "aime"
    elif "gsm8k" in args.data_path.lower():
        task = "gsm8k"
    elif "math500" in args.data_path.lower():
        task = "math500"
    elif "gpqa" in args.data_path.lower():
        task = "gpqa"
    elif "bbh" in args.data_path.lower():
        task = "bbh"


    # Load dataset
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    total_ca_ratio = 0.0

    ca_ratios = []

    CoT_lens = []
    AnS_lens = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side='right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    for item in data:

        index = item['gen'][0].find("</think>")

        CoT = index + len("</think>")
        AnS = len(item['gen'][0]) - CoT

        CoT = tokenizer(item['gen'][0][:CoT], return_tensors='pt')['input_ids'].size(-1)
        AnS = tokenizer(item["gen"][0][-AnS:], return_tensors='pt')['input_ids'].size(-1)

        CoT_lens.append(CoT)
        AnS_lens.append(AnS)

        ca_ratio = CoT / AnS
        ca_ratios.append(ca_ratio)
        total_ca_ratio += ca_ratio

    avg_ca_ratio = total_ca_ratio / len(data)
            
    print(f"The Average CoT to Answer Ratio for Task {task} is: {avg_ca_ratio}.")

    draw_ca_ratios(ca_ratios=ca_ratios, avg_ca_ratio=avg_ca_ratio, args=args, task=task)

    draw_lens(CoT_lens=CoT_lens, AnS_lens=AnS_lens, args=args, task=task)