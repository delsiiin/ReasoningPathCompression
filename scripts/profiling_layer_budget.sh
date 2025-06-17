GPU_DEVICES=0,1,2,3
MODEL=/home/yangx/DeepSeek-R1-Distill-Qwen-7B
MODEL_NICKNAME=r1-7b # qwq

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
#         --model_name $MODEL \
#         --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/gsm8k" \
#         --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gsm8k_n1_full.jsonl 

# python -m eval.profile.draw_layer_budget \
#         --grad_path "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/gsm8k/grad_attn_tensor_gsm8k.pt" \
#         --model $MODEL_NICKNAME

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
#         --model_name $MODEL \
#         --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/aime24" \
#         --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/aime24_n1_full.jsonl 

# python -m eval.profile.draw_layer_budget \
#         --grad_path "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/aime24/grad_attn_tensor_aime24.pt" \
#         --model $MODEL_NICKNAME

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
#         --model_name $MODEL \
#         --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/gpqa" \
#         --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gpqa_n1_full.jsonl 

# python -m eval.profile.draw_layer_budget \
#         --grad_path "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/gpqa/grad_attn_tensor_gpqa.pt" \
#         --model $MODEL_NICKNAME


# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
#         --model_name $MODEL \
#         --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/math500" \
#         --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/math500_n1_full.jsonl 

# python -m eval.profile.draw_layer_budget \
#         --grad_path "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/math500/grad_attn_tensor_math500.pt" \
#         --model $MODEL_NICKNAME





# BBH所有subset列表
BBH_SUBSETS=(
    "boolean_expressions"
    "causal_judgement"
    "date_understanding"
    "disambiguation_qa"
    "dyck_languages"
    "formal_fallacies"
    "geometric_shapes"
    "hyperbaton"
    "logical_deduction_five_objects"
)

# 循环处理每个subset
for BBH_SUBSET in "${BBH_SUBSETS[@]}"; do
    echo "Processing subset: $BBH_SUBSET"
    CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.profile.profile_layer_budget \
        --model_name $MODEL \
        --grad_dir "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/bbh/$BBH_SUBSET" \
        --data_path /home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/bbh_n1_full/$BBH_SUBSET.jsonl \
        --bbh_subset $BBH_SUBSET

    echo "Finished processing subset: $BBH_SUBSET"
    echo "----------------------------------------"

    python -m eval.profile.draw_layer_budget \
        --grad_path "/home/yangx/ReasoningPathCompression/eval/profile/grad_dir/$MODEL_NICKNAME/bbh/$BBH_SUBSET/grad_attn_tensor_bbh.pt" \
        --model $MODEL_NICKNAME \
        --bbh_subset $BBH_SUBSET

done

echo "All BBH subsets processing completed!"
