# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# "Qwen/QwQ/-32B"
# "/home/yangx/DeepSeek-R1-Distill-Qwen-1.5B"

MODEL=/home/yangx/models/DeepSeek-R1-Distill-Qwen-7B
MODEL_NICKNAME=r1-7b # qwq
N_SAMPLES=4
BSZ=1
GPU_DEVICES=0,1,2,3

# full kv caches
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-full.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL --data_parallel


# rpc
# P=4096
# R=32
# c=4
# SELECTORS=recent
# AGGREGATION=all
# MODE=rpc

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/aime24_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL \
#         --rpc \
#         --P $P \
#         --R $R \
#         --c $c \
#         --selectors $SELECTORS \
#         --aggregation $AGGREGATION --mode $MODE --data_parallel

# ours
# BUDGET_COT=4096
# BUDGET_ANS=1024
# c=0.25
# R=32
# AGGREGATION=group
# MODE=ours_window

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL \
#         --rpc \
#         --R $R \
#         --budget_cot $BUDGET_COT \
#         --budget_ans $BUDGET_ANS \
#         --cp_ratio $c \
#         --aggregation $AGGREGATION --mode $MODE --data_parallel

# BUDGET_COT=4096
# BUDGET_ANS=1024
# c=0.25
# R=32
# AGGREGATION=group
# MODE=ours_window_merge

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL \
#         --rpc \
#         --R $R \
#         --budget_cot $BUDGET_COT \
#         --budget_ans $BUDGET_ANS \
#         --cp_ratio $c \
#         --aggregation $AGGREGATION --mode $MODE --data_parallel


BUDGET_COT=4096
BUFFER_COT=128
BUDGET_ANS=1024
c=0.25
R=32
AGGREGATION=group
MODE=ours_window_merge_rkv_dynamic

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --rpc \
        --R $R \
        --budget_cot $BUDGET_COT --buffer_cot $BUFFER_COT \
        --budget_ans $BUDGET_ANS \
        --cp_ratio $c \
        --aggregation $AGGREGATION --mode $MODE --data_parallel