
GPU_DEVICES=0
MODEL=/home/yangx/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_NICKNAME=r1-1.5b # qwq
N_SAMPLES=8
BSZ=8

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
        --input_file "eval/data/ifeval.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/ifeval_n1_full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL 
