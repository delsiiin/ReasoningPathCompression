GPU_DEVICES=0,1,2,3
MODEL=/home/yangx/DeepSeek-R1-Distill-Qwen-7B
MODEL_NICKNAME=r1-7b # qwq

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
        --model_name $MODEL \
        --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/gsm8k" \
        --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gsm8k_n1_full.jsonl \
        --data_range 300

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
#         --model_name $MODEL \
#         --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/aime24" \
#         --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/aime24_n1_full.jsonl 

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
#         --model_name $MODEL \
#         --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/gpqa" \
#         --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gpqa_n1_full.jsonl 

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
#         --model_name $MODEL \
#         --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/math500" \
#         --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/math500_n1_full.jsonl 