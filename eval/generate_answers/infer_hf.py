import json
import argparse
from tqdm import tqdm
import os
from pathlib import Path
from rich.progress import track

import copy
import concurrent.futures
import threading
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from rpc import enable_rpc, set_rpc_config
from eval.generate_answers.utils_hf import count_completed_samples, batched_generate

import torch.multiprocessing as mp
from datasets import load_dataset

from eval.generate_answers.utils_hf import format_gpqa_question

from rpc.llama.llama_config import LlamaConfig
from rpc.llama.llama_vanilla import LlamaForCausalLM
from rpc.qwen2.qwen2_config import Qwen2Config
from rpc.qwen2.qwen2_vanilla import Qwen2ForCausalLM

def gen_result_dp(data, batch_size, total_tasks, model_path, rpc, P, R, c, selectors, aggregation, kernel_size, pooling, output_file, top_k, rank, task, budget_cot, budget_ans, cp_ratio, mode):
    
    device = torch.device(f'cuda:{rank}')

    if rpc:
        enable_rpc(mode)

    if mode == "rpc" or mode is None:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'eager'

    if "qwen" in model_path.lower() or "qwq" in model_path.lower():
        config = Qwen2Config.from_pretrained(model_path)
        config.update({'mode':mode})
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            config=config
        ).to(device)
    elif "llama" in model_path.lower():
        config = LlamaConfig.from_pretrained(model_path)
        config.update({'mode':mode})
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            config=config
        ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'

    if rpc: 
        set_rpc_config(model=model,
                            P=P,
                            R=R,
                            c=c,
                            selectors=selectors,
                            aggregation=aggregation,
                            kernel_size=kernel_size,
                            pooling=pooling,
                            budget_cot=budget_cot,
                            budget_ans=budget_ans,
                            cp_ratio=cp_ratio,
                            mode=mode
                            )

    else:
        print(f"Full KV Cache Inference")

    for i in track(range(0, len(data), batch_size)):

        batch_dicts = data[i : i + batch_size] 

        processing = len(batch_dicts)
        # print(f"[Timestamp: {datetime.now()}][{total_tasks} samples remaining]")
        # print(f"[Timestamp: {datetime.now()}][{processing} samples on processing]")
        
        batched_generate(
            model=model,
            tokenizer=tokenizer,
            output_file=output_file,
            batch_dicts=batch_dicts,
            batch_size=batch_size,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            top_k=top_k,
            task=task
        )

        total_tasks -= processing

def gen_result(data, batch_size, total_tasks, model_path, rpc, P, R, c, selectors, aggregation, kernel_size, pooling, output_file, top_k, task, budget_cot, budget_ans, cp_ratio, mode):
    
    if rpc:
        enable_rpc(mode)

    if mode == "rpc" or mode is None:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'eager'

    if "qwen" in model_path.lower() or "qwq" in model_path.lower():
        config = Qwen2Config.from_pretrained(model_path)
        config.update({'mode':mode})
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            config=config,
            device_map="auto"
        )
    elif "llama" in model_path.lower():
        config = LlamaConfig.from_pretrained(model_path)
        config.update({'mode':mode})
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            config=config,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'

    if rpc: 
        set_rpc_config(model=model,
                            P=P,
                            R=R,
                            c=c,
                            selectors=selectors,
                            aggregation=aggregation,
                            kernel_size=kernel_size,
                            pooling=pooling,
                            budget_cot=budget_cot,
                            budget_ans=budget_ans,
                            cp_ratio=cp_ratio,
                            mode=mode
                            )

    else:
        print(f"Full KV Cache Inference")

    for i in track(range(0, len(data), batch_size)):

        batch_dicts = data[i : i + batch_size] 

        processing = len(batch_dicts)
        print(f"[Timestamp: {datetime.now()}][{total_tasks} samples remaining]")
        print(f"[Timestamp: {datetime.now()}][{processing} samples on processing]")
        
        batched_generate(
            model=model,
            tokenizer=tokenizer,
            output_file=output_file,
            batch_dicts=batch_dicts,
            batch_size=batch_size,
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            top_k=top_k,
            task=task
        )

        total_tasks -= processing

if __name__ == "__main__":

    set_seed(42)

    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Run inference on model with prompts from a jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch per single call")
    parser.add_argument("--model_path", type=str, default='Qwen/QwQ-32B', help="Model name")
    parser.add_argument("--rpc", action="store_true", help="Run RPC")
    parser.add_argument("--P", type=int, default=512, help="Compression period")
    parser.add_argument("--R", type=int, default=128, help="Size of selector window size")
    parser.add_argument("--c", type=int, default=128, help="Target compression ratio")
    parser.add_argument("--selectors", type=str, default='recent', help="Selection policy")
    parser.add_argument("--aggregation", type=str, default='group', help="Aggregation policy")
    parser.add_argument("--kernel_size", type=int, default=7, help="Local pooling size")
    parser.add_argument("--pooling", type=str, default='avgpool', help="Type of local pooling")

    parser.add_argument("--data_path", type=str, required=True, help="Data path")
    parser.add_argument("--test_data_num", type=int, required=False, default=None, help="Choose how many samples to test")
    parser.add_argument("--bbh_subset", type=str, required=False, help="BBH task type")
    parser.add_argument("--budget_cot", type=int, default=4096, help="Compression budget for CoT")
    parser.add_argument("--budget_ans", type=int, default=1024, help="Compression budget for answer")
    parser.add_argument("--cp_ratio", type=float, default=0.25, help="Target compression ratio")
    parser.add_argument("--mode", type=str, default=None, help="heatmap, rpc, ours_all_step, ours_window, ours_window_merge, ours_window_merge_new, dynamic_layer_budget. (None is for uniform allocation)")
    parser.add_argument("--data_parallel", action="store_true", help="whether use multi-processing")
    args = parser.parse_args()

    if 'qwq' in args.model_path.lower():
        top_k = 40
    else:
        top_k = None

    print(f"Using Model: {args.model_path}, therefore top_k={top_k}")

    if "aime" in args.data_path.lower():
        task = "aime"
    elif "ifeval" in args.data_path.lower():
        task = "ifeval"
    elif "livecodebench" in args.data_path.lower():
        task = "livecodebench"
    elif "gsm8k" in args.data_path.lower():
        task = "gsm8k"
    elif "math500" in args.data_path.lower():
        task = "math500"
    elif "gpqa" in args.data_path.lower():
        task = "gpqa"
    elif "bbh" in args.data_path.lower():
        task = "bbh"

    if os.path.exists(args.output_file):
        completed_counts = count_completed_samples(args.output_file, task)
        total_completed = sum(completed_counts.values())
        print(f"Found {total_completed} completed samples from previous run")
    else:
        output_dir = Path(args.output_file).parent
        os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, 'w', encoding='utf-8') as g:
            completed_counts = dict()
    
    # Load dataset
    if task == "gpqa":
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif task == "bbh":
        data = load_dataset(args.data_path, args.bbh_subset)
    elif task == "aime" or task == "ifeval" or task == "livecodebench":
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(l) for l in f]
    else:
        data = load_dataset(args.data_path)

    expanded_data = []
    if task == "aime" or task == "ifeval" or task == "livecodebench":
        for item in data:
            prompt = item['prompt']
            completed = completed_counts.get(prompt, 0)
            remaining = max(args.n_samples - completed, 0)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))
    elif task == "gsm8k":
        data = data['train'].select(range(600))
        for item in data:
            prompt = item['question']
            completed = completed_counts.get(prompt, 0)
            remaining = max(args.n_samples - completed, 0)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))
    elif task == "math500":
        for item in data['test']:
            prompt = item['problem']
            completed = completed_counts.get(prompt, 0)
            remaining = max(args.n_samples - completed, 0)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))
    elif task == "gpqa":
        for item in data['questions']:
            item['question'] = format_gpqa_question(item)
            prompt = item['question']
            completed = completed_counts.get(prompt, 0)
            remaining = max(args.n_samples - completed, 0)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))
    elif task == "bbh":
        for item in data['test']:
            prompt = item['input']
            completed = completed_counts.get(prompt, 0)
            remaining = max(args.n_samples - completed, 0)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))
        task = task + '/' + args.bbh_subset
    
    total_tasks = len(expanded_data)
    print(f"Total remaining samples to process: {total_tasks}")

    if args.test_data_num:
        expanded_data = expanded_data[:args.test_data_num]

    if args.data_parallel:

        world_size = torch.cuda.device_count()

        data_subsets = [expanded_data[i::world_size] for i in range(world_size)]

        processes = []
        for rank in range(world_size):
            p = mp.Process(target=gen_result_dp, args=(data_subsets[rank], args.batch_size, total_tasks, args.model_path, args.rpc, args.P, args.R, args.c, args.selectors, args.aggregation, args.kernel_size, args.pooling, args.output_file, top_k, rank, task, args.budget_cot, args.budget_ans, args.cp_ratio, args.mode))

            p.start()

            processes.append(p)

        for p in processes:
            p.join()
    
    else:

        gen_result(expanded_data, args.batch_size, total_tasks, args.model_path, args.rpc, args.P, args.R, args.c, args.selectors, args.aggregation, args.kernel_size, args.pooling, args.output_file, top_k, task, args.budget_cot, args.budget_ans, args.cp_ratio, args.mode)


    