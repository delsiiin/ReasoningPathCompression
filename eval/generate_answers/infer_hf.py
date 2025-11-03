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
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from rpc.qwen2.qwen2_config import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from rpc.qwen3.qwen3_config import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from rpc.gpt_oss.gpt_oss_config import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from rkv.monkeypatch import replace_llama, replace_qwen2

def gen_result(data, total_tasks, top_k, temperature, top_p, task, args, rank=None):
    """
    Unified function for both single-process and data-parallel inference.
    
    Args:
        data: Input data to process
        total_tasks: Total number of tasks
        top_k: Top-k parameter for generation
        task: Task type
        args: Arguments object
        rank: GPU rank for data parallel (None for single-process mode)
    """
    
    # Set up device based on whether we're in data parallel mode
    if rank is not None:
        device = torch.device(f'cuda:{rank}')
        device_map = None
        use_auto_device = False
    else:
        device = None
        device_map = "auto"
        use_auto_device = True

    if args.rpc:
        enable_rpc(args.mode)

    if args.mode == "rpc" or args.mode == "rkv" or args.mode is None:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'flash_attention_2'

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = 'left'

    if args.mode != "rkv":
        if "distill-qwen" in args.model_path.lower() or "qwq" in args.model_path.lower():
            config = Qwen2Config.from_pretrained(args.model_path)
            config.update({'mode':args.mode})
            config.update({'divide_method':args.divide_method})
            model = Qwen2ForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map=device_map
            )
            if not use_auto_device:
                model = model.to(device)
            
            model.newline_token_ids = [
                tokenizer.encode("\n")[-1],
                tokenizer.encode(".\n")[-1],
                tokenizer.encode(")\n")[-1],
                tokenizer.encode("\n\n")[-1],
                tokenizer.encode(".\n\n")[-1],
                tokenizer.encode(")\n\n")[-1],
            ]

            model.CoT_done_token_ids = [
                tokenizer.encode("</think>")[-1],
            ]
        elif "qwen3" in args.model_path.lower():
            config = Qwen3MoeConfig.from_pretrained(args.model_path)
            config.update({'mode':args.mode})
            config.update({'divide_method':args.divide_method})
            model = Qwen3MoeForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map=device_map
            )
            if not use_auto_device:
                model = model.to(device)
            
            model.newline_token_ids = [
                tokenizer.encode("\n")[-1],
                tokenizer.encode(".\n")[-1],
                tokenizer.encode(")\n")[-1],
                tokenizer.encode("\n\n")[-1],
                tokenizer.encode(".\n\n")[-1],
                tokenizer.encode(")\n\n")[-1],
            ]

            model.CoT_done_token_ids = [
                tokenizer.encode("</think>")[-1],
            ]
        elif "gpt" in args.model_path.lower():
            config = GptOssConfig.from_pretrained(args.model_path)
            config.update({'mode':args.mode})
            config.update({'divide_method':args.divide_method})
            model = GptOssForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map=device_map
            )
            if not use_auto_device:
                model = model.to(device)
            
            model.newline_token_ids = [
                tokenizer.encode("\n")[-1],
                tokenizer.encode(".\n")[-1],
                tokenizer.encode(")\n")[-1],
                tokenizer.encode("\n\n")[-1],
                tokenizer.encode(".\n\n")[-1],
                tokenizer.encode(")\n\n")[-1],
            ]

            model.CoT_done_token_ids = [
                tokenizer.encode("</think>")[-1],
            ]
        elif "llama" in args.model_path.lower():
            config = LlamaConfig.from_pretrained(args.model_path)
            config.update({'mode':args.mode})
            config.update({'divide_method':args.divide_method})
            model = LlamaForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map=device_map
            )
            if not use_auto_device:
                model = model.to(device)
            
            model.newline_token_ids = [
                tokenizer.encode("\n")[-1],
                tokenizer.encode(".\n")[-1],
                tokenizer.encode(")\n")[-1],
                tokenizer.encode("\n\n")[-1],
                tokenizer.encode(".\n\n")[-1],
                tokenizer.encode(")\n\n")[-1],
            ]

            model.CoT_done_token_ids = [
                tokenizer.encode("</think>")[-1],
            ]
    else:
        # ====== build compression config ======
        compression_config = {
            "method": args.mode,
            "method_config": {
                "budget": args.buffer_cot,
                "window_size": 8,
                "mix_lambda": 0.07,
                "retain_ratio": 0.2,
                "retain_direction": "last",
                "first_tokens": 4,
            },
            "compression": None,
            "update_kv": True
        }
        model_config = {
            "divide_method": "step_length",
            "divide_length": 128,
            "compression_content": "think",
        }

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=True, padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # apply monkey patch
        if args.mode.lower() != "fullkv":
            if "llama" in args.model_path.lower():
                replace_llama(compression_config)
            elif "qwen" in args.model_path.lower():
                replace_qwen2(compression_config)
            else:
                raise ValueError(f"Unsupported model: {args.model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            use_cache=True,
            attn_implementation=attn_implementation,
        )
        if not use_auto_device:
            model = model.to(device)
        model.eval()

        model.config.update(model_config)

        if args.mode.lower() != "fullkv":
            model.newline_token_ids = [
                tokenizer.encode("\n")[-1],
                tokenizer.encode(".\n")[-1],
                tokenizer.encode(")\n")[-1],
                tokenizer.encode("\n\n")[-1],
                tokenizer.encode(".\n\n")[-1],
                tokenizer.encode(")\n\n")[-1],
            ]

            model.after_think_token_ids = [
                tokenizer.encode("</think>")[-1],
            ]

    if args.rpc: 
        set_rpc_config(model=model,
                            P=args.P,
                            R=args.R,
                            c=args.c,
                            selectors=args.selectors,
                            aggregation=args.aggregation,
                            kernel_size=args.kernel_size,
                            pooling=args.pooling,
                            budget_cot=args.budget_cot,
                            buffer_cot=args.buffer_cot,
                            budget_ans=args.budget_ans,
                            cp_ratio=args.cp_ratio,
                            mode=args.mode
                            )

    elif args.mode == "rkv":
        print(f"RKV Cache Inference")
    else:
        print(f"Full KV Cache Inference")

    for i in track(range(0, len(data), args.batch_size)):

        batch_dicts = data[i : i + args.batch_size] 

        processing = len(batch_dicts)
        # Only print detailed logs in single-process mode
        if rank is None:
            print(f"[Timestamp: {datetime.now()}][{total_tasks} samples remaining]")
            print(f"[Timestamp: {datetime.now()}][{processing} samples on processing]")
        
        try:
            batched_generate(
                model=model,
                tokenizer=tokenizer,
                output_file=args.output_file,
                batch_dicts=batch_dicts,
                batch_size=args.batch_size,
                max_new_tokens=32768,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                task=task
            )
        except Exception as e:
            print(f"[Error] Batch generation failed: {str(e)}")
            print(f"[Warning] Skipping current batch and continuing...")

        torch.cuda.empty_cache()

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
    parser.add_argument("--buffer_cot", type=int, default=128, help="Newly generated token buffer for CoT")
    parser.add_argument("--budget_ans", type=int, default=1024, help="Compression budget for answer")
    parser.add_argument("--cp_ratio", type=float, default=0.25, help="Target compression ratio")
    parser.add_argument("--mode", type=str, default=None, help="heatmap, rpc, ours_all_step, ours_window, ours_window_merge, ours_window_merge_rkv, ours_window_merge_new, dynamic_layer_budget. (None is for uniform allocation)")
    parser.add_argument("--divide_method", type=str, default=None, help="new_line, step_length")
    parser.add_argument("--data_parallel", action="store_true", help="whether use multi-processing")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of data shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID of this instance")
    args = parser.parse_args()

    if 'qwq' in args.model_path.lower():
        top_k = 40
        temperature = 0.6
        top_p = 0.95
    elif "qwen3" in args.model_path.lower():
        top_k = 20
        temperature = 0.8
        top_p = 0.7
    elif "gpt" in args.model_path.lower():
        top_k = None
        temperature = 1
        top_p = 1
    else:
        top_k = None
        temperature = 0.6
        top_p = 0.95

    print(f"Using Model: {args.model_path}, therefore top_k={top_k}, temperature={temperature}, top_p={top_p}")

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
        if args.num_shards > 1:
            data['questions'] = data['questions'][args.shard_id::args.num_shards]
    elif task == "bbh":
        data = load_dataset(args.data_path, args.bbh_subset)
        if args.num_shards > 1:
            data['test'] = data['test'].shard(num_shards=args.num_shards, index=args.shard_id)
    elif task == "aime" or task == "ifeval" or task == "livecodebench":
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(l) for l in f]
        if args.num_shards > 1:
            data = data[args.shard_id::args.num_shards]
    elif task == "gsm8k":
        data = load_dataset(args.data_path, "main")
        if args.num_shards > 1 and 'test' in data:
            data['test'] = data['test'].shard(num_shards=args.num_shards, index=args.shard_id)
    else:
        data = load_dataset(args.data_path)
        if args.num_shards > 1 and 'test' in data:
            data['test'] = data['test'].shard(num_shards=args.num_shards, index=args.shard_id)

    expanded_data = []
    if task == "aime" or task == "ifeval" or task == "livecodebench":
        for item in data:
            prompt = item['prompt']
            completed = completed_counts.get(prompt, 0)
            remaining = max(args.n_samples - completed, 0)
            for _ in range(remaining):
                expanded_data.append(copy.deepcopy(item))
    elif task == "gsm8k":
        data = data['test']
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
            p = mp.Process(target=gen_result, args=(data_subsets[rank], total_tasks, top_k, temperature, top_p, task, args, rank))

            p.start()

            processes.append(p)

        for p in processes:
            p.join()
    
    else:

        gen_result(expanded_data, total_tasks, top_k, temperature, top_p, task, args)


    