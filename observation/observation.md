# Generate Attetion Score Using Heatmap Mode
CUDA_VISIBLE_DEVICES=0 python example.py --rpc False --mode heatmap --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
(Need to change the kv length in llama_vanilla.py)

# Generate Token Entropy Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --rpc False --mode entropy --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
(Need to change the kv length in llama_vanilla.py)

# Generate Token-wise Attention Map

# Generate Step-wise Attention Map
./run_plot_step_wise_attn_map.sh llama 0 31 0.05

# Generate Token Entropy
./run_plot_token_entropy.sh llama