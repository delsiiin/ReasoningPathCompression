GPU_DEVICES=0,1,2,3
MODEL=/home/yangx/DeepSeek-R1-Distill-Qwen-7B
MODEL_NICKNAME=r1-7b # qwq
N_SAMPLES=1
BSZ=1

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/gpqa/gpqa_diamond.json" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gpqa_n1_full.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL 

# python -m eval.profile.profile_ca_ratio \
#         --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gpqa_n1_full.jsonl" \
#         --model $MODEL_NICKNAME --model_path $MODEL

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/aime_2024" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/aime24_n1_full.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL 

# python -m eval.profile.profile_ca_ratio \
#         --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/aime24_n1_full.jsonl" \
#         --model $MODEL_NICKNAME --model_path $MODEL

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
        --data_path "/home/yangx/ReasoningPathCompression/datasets/gsm8k" \
        --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gsm8k_n1_full.jsonl" \
        --n_samples $N_SAMPLES \
        --batch_size $BSZ \
        --model_path $MODEL 

python -m eval.profile.profile_ca_ratio \
        --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/gsm8k_n1_full.jsonl" \
        --model $MODEL_NICKNAME --model_path $MODEL

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/MATH500" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/math500_n1_full.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL 

# python -m eval.profile.profile_ca_ratio \
#         --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/math500_n1_full.jsonl" \
#         --model $MODEL_NICKNAME --model_path $MODEL



# BBH所有subset列表
# BBH_SUBSETS=(
#     "boolean_expressions"
#     "causal_judgement"
#     "date_understanding"
#     "disambiguation_qa"
#     "dyck_languages"
#     "formal_fallacies"
#     "geometric_shapes"
#     "hyperbaton"
#     "logical_deduction_five_objects"
#     "movie_recommendation"
#     "multistep_arithmetic_two"
#     "navigate"
#     "object_counting"
#     "penguins_in_a_table"
#     "reasoning_about_colored_objects"
#     "ruin_names"
#     "salient_translation_error_detection"
#     "snarks"
#     "sports_understanding"
#     "temporal_sequences"
#     "tracking_shuffled_objects_five_objects"
#     "web_of_lies"
#     "word_sorting"
# )

# # 循环处理每个subset
# for BBH_SUBSET in "${BBH_SUBSETS[@]}"; do
#     echo "Processing subset: $BBH_SUBSET"
    
#     CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m eval.generate_answers.infer_hf \
#         --data_path "/home/yangx/ReasoningPathCompression/datasets/bbh" \
#         --output_file "eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/bbh_n1_full/$BBH_SUBSET.jsonl" \
#         --n_samples $N_SAMPLES \
#         --batch_size $BSZ \
#         --model_path $MODEL \
#         --bbh_subset $BBH_SUBSET
    
#     echo "Finished processing subset: $BBH_SUBSET"
#     echo "----------------------------------------"
# done

# echo "All BBH subsets processing completed!"


# for BBH_SUBSET in "${BBH_SUBSETS[@]}"; do

#         python -m eval.profile.profile_ca_ratio \
#                 --data_path "/home/yangx/ReasoningPathCompression/eval/outputs/$MODEL_NICKNAME/profile_ca_ratio/bbh_n1_full/$BBH_SUBSET.jsonl" \
#                 --model $MODEL_NICKNAME --model_path $MODEL --bbh_subset $BBH_SUBSET

# done