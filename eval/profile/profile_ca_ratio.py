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
        plt.savefig(f"/home/yangx/zmw/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}/{args.bbh_subset}_ca_ratio.pdf")
    else:
        plt.savefig(f"/home/yangx/zmw/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}_ca_ratio.pdf")

def draw_lens(CoT_lens, AnS_lens, Prompt_lens, args, task):
    # 绘制折线图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(CoT_lens, marker='o', markersize=2, label='CoT_lens')
    plt.plot(AnS_lens, marker='s', markersize=2, label='AnS_lens')
    plt.plot(Prompt_lens, marker='^', markersize=2, label='Prompt_lens')
    plt.title("CoT, Answer, and Prompt Lengths")
    plt.xlabel("Sample")
    plt.ylabel("Length (tokens)")
    plt.grid(True)
    plt.legend()
    
    # 添加第二个子图专门显示 prompt 长度分布
    plt.subplot(2, 1, 2)
    plt.plot(Prompt_lens, marker='^', markersize=2, label='Prompt_lens', color='green')
    plt.axhline(y=sum(Prompt_lens)/len(Prompt_lens), color='red', linestyle='--', 
                label=f'Avg: {sum(Prompt_lens)/len(Prompt_lens):.2f}')
    plt.title("Prompt Length Distribution")
    plt.xlabel("Sample")
    plt.ylabel("Prompt Length (tokens)")
    plt.grid(True)  
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    if args.bbh_subset:
        plt.savefig(f"/home/yangx/zmw/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}/{args.bbh_subset}_lens.pdf")
    else:
        plt.savefig(f"/home/yangx/zmw/ReasoningPathCompression/eval/profile/ca_ratios/{args.model}/{task}_lens.pdf")

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

    total_samples = len(data)
    filtered_samples = 0
    total_ca_ratio = 0.0

    ca_ratios = []

    CoT_lens = []
    AnS_lens = []
    Prompt_lens = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side='right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    for item in data:

        # Calculate prompt length
        if 'prompt' in item:
            prompt_text = item['prompt']
        elif 'problem' in item:
            prompt_text = item['problem']
        elif 'question' in item:
            prompt_text = item['question']
        else:
            prompt_text = ""  # fallback if no prompt field found
        
        prompt_len = tokenizer(prompt_text, return_tensors='pt')['input_ids'].size(-1)

        index = item['gen'][0].find("</think>")

        CoT = index + len("</think>")
        AnS = len(item['gen'][0]) - CoT

        CoT = tokenizer(item['gen'][0][:CoT], return_tensors='pt')['input_ids'].size(-1)
        AnS = tokenizer(item["gen"][0][-AnS:], return_tensors='pt')['input_ids'].size(-1)

        # Filter out samples with answer length > 2000 tokens
        if AnS > 2000:
            filtered_samples += 1
            continue

        Prompt_lens.append(prompt_len)
        CoT_lens.append(CoT)
        AnS_lens.append(AnS)

        ca_ratio = CoT / AnS
        ca_ratios.append(ca_ratio)
        total_ca_ratio += ca_ratio

    valid_samples = len(ca_ratios)
    avg_ca_ratio = total_ca_ratio / valid_samples if valid_samples > 0 else 0
    avg_prompt_len = sum(Prompt_lens) / len(Prompt_lens) if len(Prompt_lens) > 0 else 0
    avg_cot_len = sum(CoT_lens) / len(CoT_lens) if len(CoT_lens) > 0 else 0
    avg_ans_len = sum(AnS_lens) / len(AnS_lens) if len(AnS_lens) > 0 else 0
            
    print(f"Total samples: {total_samples}")
    print(f"Filtered samples (Answer > 2000 tokens): {filtered_samples}")
    print(f"Valid samples: {valid_samples}")
    print(f"The Average CoT to Answer Ratio for Task {task} is: {avg_ca_ratio:.4f}.")
    print(f"The Average Prompt Length for Task {task} is: {avg_prompt_len:.2f} tokens.")
    print(f"The Average CoT Length for Task {task} is: {avg_cot_len:.2f} tokens.")
    print(f"The Average Answer Length for Task {task} is: {avg_ans_len:.2f} tokens.")

    draw_ca_ratios(ca_ratios=ca_ratios, avg_ca_ratio=avg_ca_ratio, args=args, task=task)

    draw_lens(CoT_lens=CoT_lens, AnS_lens=AnS_lens, Prompt_lens=Prompt_lens, args=args, task=task)