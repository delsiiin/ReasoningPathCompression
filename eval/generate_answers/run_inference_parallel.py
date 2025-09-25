import subprocess
import sys
import os
from multiprocessing import Process
import argparse
import json

def run_inference(gpu_ids, shard_id, num_shards, common_args):
    """
    Function to run a single instance of infer_hf.py on a given set of GPUs.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    command = [
        sys.executable,
        "-m",
        "eval.generate_answers.infer_hf",
        "--shard_id", str(shard_id),
        "--num_shards", str(num_shards),
    ] + common_args
    
    print(f"Starting process for shard {shard_id} on GPUs {gpu_ids} with command: {' '.join(command)}")
    
    # Using subprocess.Popen to run the command
    process = subprocess.Popen(command, env=env, stdout=None, stderr=None, text=True)
    
    # Stream the output
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(f"[Shard {shard_id} | GPUs {gpu_ids}]: {line.strip()}")
    
    retcode = process.wait()
    if retcode != 0:
        print(f"Process for shard {shard_id} on GPUs {gpu_ids} exited with error code {retcode}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu_allocations",
        type=str,
        default="[[0, 1], [2, 3]]",
        help="JSON string for GPU allocations, e.g., '[[0, 1], [2, 3]]'"
    )
    
    args, common_args = parser.parse_known_args()
    
    try:
        gpu_allocations = json.loads(args.gpu_allocations)
    except json.JSONDecodeError:
        print("Error: Invalid format for --gpu_allocations. Please provide a valid JSON string.")
        sys.exit(1)
    
    num_shards = len(gpu_allocations)
    
    processes = []
    for i, gpu_ids in enumerate(gpu_allocations):
        shard_id = i
        p = Process(target=run_inference, args=(gpu_ids, shard_id, num_shards, common_args))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    print("All inference processes have completed.")
