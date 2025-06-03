MODEL_NICKNAME=r1-1.5b
P=1024
R=32
c=4
SELECTORS=recent
AGGREGATION=all

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/aime24_n8_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/aime24_n8_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl \
# --task_name "math_opensource/aime24" > ./eval_res/$MODEL_NICKNAME/aime24_n8_$P-$R-$c-$SELECTORS-$AGGREGATION.txt

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/aime24_n8_full.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/aime24_n8_full.jsonl \
# --task_name "math_opensource/aime24" > ./eval_res/$MODEL_NICKNAME/aime24_n8_full.txt

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/ifeval_n1_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/ifeval_n1_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl \
# --task_name "ifeval" > ./eval_res/$MODEL_NICKNAME/ifeval_n1_$P-$R-$c-$SELECTORS-$AGGREGATION.txt

python  ./eval/eval.py \
--input_path ./outputs/$MODEL_NICKNAME/ifeval_n1_full.jsonl \
--cache_path ./eval_res/$MODEL_NICKNAME/ifeval_n1_full.jsonl \
--task_name "ifeval" > ./eval_res/$MODEL_NICKNAME/ifeval_n1_full.txt


# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/livecodebench_v5_n4_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/livecodebench_v5_n4_$P-$R-$c-$SELECTORS-$AGGREGATION.jsonl \
# --task_name "livecodebench" > ./eval_res/$MODEL_NICKNAME/livecodebench_v5_n4_$P-$R-$c-$SELECTORS-$AGGREGATION.txt