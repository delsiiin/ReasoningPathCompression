MODEL_NICKNAME=r1-7b
N_SAMPLES=4
BSZ=1

# aime24 full
# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --task_name "math_opensource/aime24" 

# aime24 rpc
# P=4096
# c=4
# R=32
# SELECTORS=recent
# AGGREGATION=all
# MODE=rpc

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/aime24_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/aime24_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --task_name "math_opensource/aime24"

# aime24 ours
# BUDGET_COT=4096
# BUDGET_ANS=1024
# c=0.25
# R=32
# AGGREGATION=group
# MODE=ours_window_merge_rkv

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/aime24-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --task_name "math_opensource/aime24" 




N_SAMPLES=1
BSZ=1

# ifeval full
# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/ifeval-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/ifeval-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --task_name "ifeval" 

# ifeval rpc
# P=4096
# c=4
# R=32
# SELECTORS=recent
# AGGREGATION=all
# MODE=rpc

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/ifeval_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/ifeval_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --task_name "ifeval"

# ifeval ours
# BUDGET_COT=4096
# BUDGET_ANS=1024
# c=0.25
# R=32
# AGGREGATION=group
# MODE=ours_window_merge_rkv

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/ifeval-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/ifeval-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --task_name "ifeval" 








N_SAMPLES=4
BSZ=1

# livecodebench full
# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/livecodebench-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/livecodebench-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --task_name "livecodebench" 

# livecodebench rpc
# P=4096
# c=4
# R=32
# SELECTORS=recent
# AGGREGATION=all
# MODE=rpc

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/livecodebench_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/livecodebench_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --task_name "livecodebench"

# livecodebench ours
# BUDGET_COT=4096
# BUDGET_ANS=1024
# c=0.25
# R=32
# AGGREGATION=group
# MODE=ours_window_merge_new

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/livecodebench-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/livecodebench-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --task_name "livecodebench" 



N_SAMPLES=1
BSZ=1

# # gsm8k full
# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/gsm8k-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/gsm8k-b-$BSZ-s-$N_SAMPLES-full.jsonl \
# --task_name "openai/gsm8k" 

# # gsm8k rpc
# P=4096
# c=4
# R=32
# SELECTORS=recent
# AGGREGATION=all
# MODE=rpc

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/gsm8k_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/gsm8k_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
# --task_name "openai/gsm8k"

# # gsm8k ours
# BUDGET_COT=4096
# BUDGET_ANS=1024
# c=0.25
# R=32
# AGGREGATION=group
# MODE=ours_window_merge_rkv

# python  ./eval/eval.py \
# --input_path ./outputs/$MODEL_NICKNAME/gsm8k-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --cache_path ./eval_res/$MODEL_NICKNAME/gsm8k-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
# --task_name "openai/gsm8k" 




N_SAMPLES=4
BSZ=1

# math500 full
python  ./eval/eval.py \
--input_path ./outputs/$MODEL_NICKNAME/math500-b-$BSZ-s-$N_SAMPLES-full.jsonl \
--cache_path ./eval_res/$MODEL_NICKNAME/math500-b-$BSZ-s-$N_SAMPLES-full.jsonl \
--task_name "HuggingFaceH4/math500" 

# math500 rpc
P=4096
c=4
R=32
SELECTORS=recent
AGGREGATION=all
MODE=rpc

python  ./eval/eval.py \
--input_path ./outputs/$MODEL_NICKNAME/math500_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
--cache_path ./eval_res/$MODEL_NICKNAME/math500_b-$BSZ-s-$N_SAMPLES-$P-$R-$c-$SELECTORS-$AGGREGATION-$MODE.jsonl \
--task_name "HuggingFaceH4/math500"

# math500 ours
BUDGET_COT=4096
BUDGET_ANS=1024
c=0.25
R=32
AGGREGATION=group
MODE=ours_window_merge_rkv

python  ./eval/eval.py \
--input_path ./outputs/$MODEL_NICKNAME/math500-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
--cache_path ./eval_res/$MODEL_NICKNAME/math500-b-$BSZ-s-$N_SAMPLES-$BUDGET_COT-$BUDGET_ANS-$c-$AGGREGATION-$MODE.jsonl \
--task_name "HuggingFaceH4/math500" 
