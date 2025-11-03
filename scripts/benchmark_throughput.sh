# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# "Qwen/QwQ/-32B"

# DeepSeek-R1-Distill-Llama-8B with batch size 16
for bsz in 16
do
    for ol in 1024 2048 4096 8192 16384
    do  
        python -m benchmark.throughput \
        --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
        --batch_size $bsz \
        --output_len $ol \
        --rpc False \
        --num_runs 1
    done
done

# Qwen3-30B-A3B-Thinking-2507 with batch size 8
for bsz in 8
do
    for ol in 1024 2048 4096 8192 16384
    do  
        python -m benchmark.throughput \
        --model_path "Qwen/Qwen3-30B-A3B-Thinking-2507" \
        --batch_size $bsz \
        --output_len $ol \
        --rpc False \
        --num_runs 1
    done
done

# QwQ-32B with batch size 8
for bsz in 8
do
    for ol in 1024 2048 4096 8192
    do  
        python -m benchmark.throughput \
        --model_path "Qwen/QwQ-32B" \
        --batch_size $bsz \
        --output_len $ol \
        --rpc False \
        --num_runs 1
    done
done
