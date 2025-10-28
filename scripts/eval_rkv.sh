MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_NICKNAME=r1-7b # qwq
N_SAMPLES=4
BSZ=1

BUFFER_COT=2048
MODE=rkv

python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-$BUFFER_COT-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --buffer_cot $BUFFER_COT \
        --mode $MODE --gpu_allocations "[[2], [3]]"


N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/ifeval/ifeval.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/ifeval-b-$BSZ-s-$N_SAMPLES-$BUFFER_COT-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --buffer_cot $BUFFER_COT \
        --mode $MODE --gpu_allocations "[[2], [3]]"

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/gsm8k" \
        --output_file "eval/outputs/$MODEL_NICKNAME/gsm8k-b-$BSZ-s-$N_SAMPLES-$BUFFER_COT-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --buffer_cot $BUFFER_COT \
        --mode $MODE --gpu_allocations "[[2], [3]]"

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/MATH500" \
        --output_file "eval/outputs/$MODEL_NICKNAME/math500-b-$BSZ-s-$N_SAMPLES-$BUFFER_COT-$MODE.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL \
        --buffer_cot $BUFFER_COT \
        --mode $MODE --gpu_allocations "[[2], [3]]"
