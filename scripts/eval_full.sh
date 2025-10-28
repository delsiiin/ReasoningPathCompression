MODEL=openai/gpt-oss-20b
MODEL_NICKNAME=oss-20b # qwq
N_SAMPLES=4
BSZ=1

python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/aime_2024/aime24.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL --gpu_allocations "[[0], [1], [2], [3]]"

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/ifeval/ifeval.jsonl" \
        --output_file "eval/outputs/$MODEL_NICKNAME/ifeval-b-$BSZ-s-$N_SAMPLES-full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL --gpu_allocations "[[0], [1], [2], [3]]"
        

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/gsm8k" \
        --output_file "eval/outputs/$MODEL_NICKNAME/gsm8k-b-$BSZ-s-$N_SAMPLES-full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL --gpu_allocations "[[0], [1], [2], [3]]"
        

N_SAMPLES=1
python -m eval.generate_answers.run_inference_parallel \
        --data_path "/home/yangx/zmw/ReasoningPathCompression/datasets/MATH500" \
        --output_file "eval/outputs/$MODEL_NICKNAME/math500-b-$BSZ-s-$N_SAMPLES-full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL --gpu_allocations "[[0], [1], [2], [3]]"