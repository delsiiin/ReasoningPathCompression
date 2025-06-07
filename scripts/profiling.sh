GPU_DEVICES=0,1,2,3
MODEL=/home/yangx/DeepSeek-R1-Distill-Qwen-7B
MODEL_NICKNAME=r1-7b # qwq
N_SAMPLES=1
BSZ=1

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/gpqa/gpqa_diamond.json" \
        --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gpqa_n1_full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL 

python -m eval.profile.profile_ca_ratio \
        --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/r1-7b/profile_ca_ratio/gpqa_n1_full.jsonl"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024" \
        --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/aime24_n1_full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL 

python -m eval.profile.profile_ca_ratio \
        --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/r1-7b/profile_ca_ratio/aime24_n1_full.jsonl"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/gsm8k" \
        --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gsm8k_n1_full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL 

python -m eval.profile.profile_ca_ratio \
        --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/r1-7b/profile_ca_ratio/gsm8k_n1_full.jsonl"

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/MATH500" \
        --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/math500_n1_full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL 

python -m eval.profile.profile_ca_ratio \
        --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/r1-7b/profile_ca_ratio/math500_n1_full.jsonl"