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
./run_plot_token_entropy.sh llama h2o

## Generate Token Confidence Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 4096 --rkv True --rkv_mode h2o --mode confidence --rkv_budget 1024
./run_plot_token_confidence.sh llama h2o

## Compare Important Indices w/ Different Compressed Methods
### Vanilla Continue Gen (eg. 1024 -> 1152)
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 1152  --mode record_indices --observation_length 1024 --observation_topk 256
### Compression Methods
CUDA_VISIBLE_DEVICES=0 python example.py --rkv True --rkv_mode snapkv --mode record_indices --observation_length 1024 --observation_topk 256 --window_size 8 --rkv_budget 1024
### Induced Answer
CUDA_VISIBLE_DEVICES=0 python example.py --mode induce_answer --observation_length 1024 --observation_topk 256 --window_size 8

./run_plot_topk_indices.sh llama3 all 1024 256

## Generate Hit Rate
./run_plot_topk_indices.sh llama3 all 1024 256 true induced