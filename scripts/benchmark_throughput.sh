# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# "Qwen/QwQ/-32B"

MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
for model in "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "Qwen/Qwen3-30B-A3B-Instruct-2507"
do
    for bsz in 4
    do
        for ol in 1024 2048 4096 8192 16384 32768
        do  
            python -m benchmark.throughput \
            --model_path $model \
            --batch_size $bsz \
            --output_len $ol \
            --rpc False

            # python -m benchmark.throughput \
            # --model_path $MODEL \
            # --rpc True \
            # --P 1024 \
            # --R 32 \
            # --c 4 \
            # --batch_size $bsz \
            # --output_len $ol
        done
    done
done


