# Vanilla Inference Observation

## Generate Attetion Score Using Heatmap Mode
CUDA_VISIBLE_DEVICES=0 python example.py --mode heatmap --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
(Need to change the kv length in llama_vanilla.py)

## Generate Token Entropy Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --mode entropy --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

## Generate Token Confidence Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --mode confidence --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

## Generate Token-wise Attention Map
python draw_heat_map.py --model llama --num_layers 32

## Generate Step-wise Attention Map
./run_plot_step_wise_attn_map.sh llama 0 31 0.1

## Generate Token Entropy
./run_plot_token_entropy.sh llama

## Generate Token Confidence
./run_plot_token_confidence.sh llama



# Compressed Inference Observation (streamingllm, h2o, snapkv, r-kv)

## Generate Token Entropy Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 4096 --rkv True --rkv_mode h2o --mode entropy --rkv_budget 1024

## Generate Token Confidence Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 4096 --rkv True --rkv_mode h2o --mode confidence --rkv_budget 1024

## Compare Important Indices w/ or w/o Answer Inducing
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 1500  --mode observation_window --observation_length 1024 --observation_topk 512 --window_size 8
CUDA_VISIBLE_DEVICES=0 python example.py --mode induce_answer --observation_length 1024 --observation_topk 512 
./run_plot_topk_indices.sh llama3 all 1024 512

## Generate Hit Rate
./run_plot_topk_indices.sh llama3 all 1024 512 true induced