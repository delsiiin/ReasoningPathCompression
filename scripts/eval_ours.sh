MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
MODEL_NICKNAME=r1-14b # qwq
N_SAMPLES=4
BSZ=1

BUDGET_COT=4096
BUFFER_COT=128
BUDGET_ANS=1024
c=0.25
R=8
AGGREGATION=group
MODE=ours_window_merge_rkv

python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUFFER_COT-$R-$AGGREGATION-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --rpc \
        --R $R \
        --budget_cot $BUDGET_COT --buffer_cot $BUFFER_COT \
        --budget_ans $BUDGET_ANS \
        --cp_ratio $c \
        --aggregation $AGGREGATION --mode $MODE --gpu_allocations "[[0, 1], [2, 3]]"

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/ifeval/ifeval.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/ifeval-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUFFER_COT-$R-$AGGREGATION-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --rpc \
        --R $R \
        --budget_cot $BUDGET_COT --buffer_cot $BUFFER_COT \
        --budget_ans $BUDGET_ANS \
        --cp_ratio $c \
        --aggregation $AGGREGATION --mode $MODE --gpu_allocations "[[0, 1], [2, 3]]"

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/gsm8k" \
        --output_file "eval/outputs/$MODEL_NICKNAME/gsm8k-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUFFER_COT-$R-$AGGREGATION-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --rpc \
        --R $R \
        --budget_cot $BUDGET_COT --buffer_cot $BUFFER_COT \
        --budget_ans $BUDGET_ANS \
        --cp_ratio $c \
        --aggregation $AGGREGATION --mode $MODE --gpu_allocations "[[0, 1], [2, 3]]"

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/MATH500" \
        --output_file "eval/outputs/$MODEL_NICKNAME/math500-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUFFER_COT-$R-$AGGREGATION-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --rpc \
        --R $R \
        --budget_cot $BUDGET_COT --buffer_cot $BUFFER_COT \
        --budget_ans $BUDGET_ANS \
        --cp_ratio $c \
        --aggregation $AGGREGATION --mode $MODE --gpu_allocations "[[0, 1], [2, 3]]"