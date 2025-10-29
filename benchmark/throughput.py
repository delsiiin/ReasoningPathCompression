import fire
import logging
from tqdm import tqdm

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpc import enable_rpc, set_rpc_config

# from utils.qwen2_norepeat import qwen2_flashattention2_norepeat_forward

# def monkeypatch():
#     transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flashattention2_norepeat_forward

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def average_excluding_min_max(numbers):
    if len(numbers) <= 2:
        return sum(numbers) / len(numbers)
    
    numbers_excluding_min_max = numbers.copy()
    numbers_excluding_min_max.remove(min(numbers))
    numbers_excluding_min_max.remove(max(numbers))

    return sum(numbers_excluding_min_max) / len(numbers_excluding_min_max)

def measure_throughput(
    model_path: str = "Qwen/QwQ-32B",
    rpc: bool = False,
    # RPC arguments
    P: int = 1024,
    R: int = 32,
    c: int = 4,
    selectors: str = 'recent',
    # experiment arguments
    batch_size: int = 16,
    input_len: int = 128,
    output_len: int = 32768,
    num_warmups: int = 1,
    num_runs: int = 3,
    output_file: str = None
    ):

    import datetime
    import os

    # Generate output file name if not provided
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        rpc_suffix = "_rpc" if rpc else "_baseline"
        output_file = f"/home/yangx/zmw/ReasoningPathCompression/benchmark/throughput_results_{model_name}{rpc_suffix}_{batch_size}_{output_len}_{timestamp}.txt"

    num_gpus = torch.cuda.device_count()

    attn_implementation = 'flash_attention_2'
    if rpc:
        enable_rpc()
        
    else:
        pass
        # monkeypatch()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    
    # Ensure pad_token is set for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rpc:
        set_rpc_config(model=model,
                            P=P,
                            R=R,
                            c=c,
                            selectors=selectors,
                            aggregation='all',
                            kernel_size=7,
                            pooling='avgpool'                            
                            )

    # Input Sequence      
    input_id = torch.ones((batch_size, input_len), dtype=torch.int64).to(model.device)
    attn_mask = torch.ones((batch_size, input_len), dtype=torch.int64).to(model.device)
    context_length = input_id.shape[-1]

    if num_warmups > 0:
        for i in range(num_warmups):
            print(f"Warm Up Run #{i}")

            with torch.no_grad():
                # Use model.generate() instead of manual token-by-token generation
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                generated_ids = model.generate(
                    input_ids=input_id,
                    attention_mask=attn_mask,
                    max_new_tokens=output_len,
                    min_new_tokens=output_len,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None,  # Disable EOS token to prevent early termination
                    use_cache=True
                )
                end_time.record()
                torch.cuda.synchronize()
                
        del generated_ids
        cleanup_memory()
    
    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(device=i)


    results_list = []

    for i in range(num_runs):
        print(f"Test Run #{i}")

        with torch.no_grad():
            # Use model.generate() instead of manual token-by-token generation
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            generated_ids = model.generate(
                input_ids=input_id,
                attention_mask=attn_mask,
                max_new_tokens=output_len,
                min_new_tokens=output_len,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,  # Disable EOS token to prevent early termination
                use_cache=True
            )
            end_time.record()
            torch.cuda.synchronize()
            
            # Calculate time in milliseconds
            total_time = start_time.elapsed_time(end_time)
            
        throughput = batch_size * output_len / (total_time / 1000)
        results_list.append(throughput)

        print(f"Generated IDs length: {generated_ids.shape}")

        del generated_ids
        cleanup_memory()

    avg_throughput = average_excluding_min_max(results_list)

    total_max_memory = 0
    for i in range(num_gpus):
        max_mem = torch.cuda.max_memory_allocated(device=i)
        total_max_memory += max_mem

    # Prepare results for both console and file output
    results_text = []
    results_text.append("=" * 60)
    results_text.append("THROUGHPUT BENCHMARK RESULTS")
    results_text.append("=" * 60)
    results_text.append(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"Model: {model_path}")
    results_text.append(f"Mode: {'RPC' if rpc else 'Baseline'}")
    
    if rpc:
        results_text.append(f"RPC Parameters:")
        results_text.append(f"  P={P}")
        results_text.append(f"  R={R}")
        results_text.append(f"  c={c}")
        results_text.append(f"  selectors={selectors}")
    
    results_text.append(f"Experiment Parameters:")
    results_text.append(f"  Batch Size: {batch_size}")
    results_text.append(f"  Input Length: {input_len}")
    results_text.append(f"  Output Length: {output_len}")
    results_text.append(f"  Number of Warm Up Runs: {num_warmups}")
    results_text.append(f"  Number of Test Runs: {num_runs}")
    results_text.append(f"")
    results_text.append(f"Individual Run Results (tokens/sec):")
    for i, throughput in enumerate(results_list):
        results_text.append(f"  Run {i+1}: {throughput:.2f}")
    results_text.append(f"")
    results_text.append(f"Average Throughput (tokens/sec): {avg_throughput:.2f}")
    results_text.append(f"Peak GPU Memory: {total_max_memory / 1000**2 / 1000:.2f} GB")
    results_text.append("=" * 60)

    # Print to console
    for line in results_text:
        print(line)

    # Save to file
    try:
        with open(output_file, 'w') as f:
            for line in results_text:
                f.write(line + '\n')
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving results to file: {e}")

    # Also print the old format for backward compatibility
    print(f"\nModel: {model_path}")
    # print(f"Mode: {mode}")

    if rpc:
        print(f"P={P}")
        print(f"R={R}")
        print(f"c={c}" )

    print(f"Batch Size={batch_size}")
    print(f"Input Length={input_len}, Output Length={output_len}")
    print(f"Number of Warm Up Runs={num_warmups}, Number of Test Runs={num_runs}")
    print(f"Average Throughput (tokens/sec)={avg_throughput:.2f}")
    print(f"Peak GPU Memory: {total_max_memory / 1000**2 / 1000:.2f} GB\n")


if __name__ == "__main__":
    fire.Fire(measure_throughput)

