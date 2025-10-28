# Vanilla Inference Observation (eager)

## Generate Attetion Score Using Heatmap Mode (token)
CUDA_VISIBLE_DEVICES=0 python example.py --mode token_heatmap --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
(Need to change the kv length in llama_vanilla.py)

## Generate Attetion Score Using Heatmap Mode (step)
CUDA_VISIBLE_DEVICES=0 python example.py --mode step_heatmap --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
(Need to change the kv length in llama_vanilla.py)


## Generate Token Entropy Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --mode entropy --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

## Generate Token Confidence Using Confidence Mode
CUDA_VISIBLE_DEVICES=0 python example.py --mode confidence --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

## Generate Token-wise Attention Map
python draw_heat_map.py --model llama3 --num_layers 32

## Generate Step-wise Attention Map
./run_plot_step_wise_attn_map.sh llama 0 31 0.1

## Generate Token Entropy
./run_plot_token_entropy.sh llama

## Generate Token Confidence
./run_plot_token_confidence.sh llama



# Compressed Inference Observation (streamingllm, h2o, snapkv, r-kv) (eager)

## Generate Token Entropy Using Entropy Mode
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 4096 --rkv True --rkv_mode h2o --mode entropy --rkv_budget 1024
./run_plot_token_entropy.sh llama h2o

## Generate Token Confidence Using Confidence Mode
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 4096 --rkv True --rkv_mode h2o --mode confidence --rkv_budget 1024
./run_plot_token_confidence.sh llama h2o

## Compare Important Indices w/ Different Compressed Methods
### Vanilla Continue Gen (eg. 1510 = 102 + 128*10 + 128, 1537 = 155(prompt) + 102 + 128*10)
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 1510  --mode record_indices --observation_length 1537 --observation_topk 256
### Compression Methods (eg. 1383 = 1537 - 155 + 1, 1537 = 155(prompt) + 102 + 128*10)
CUDA_VISIBLE_DEVICES=0 python example.py --rkv True --rkv_mode snapkv --mode record_indices --observation_length 1383 --observation_topk 256 --window_size 8 --rkv_budget 1537
### Induced Answer (eg. 1382 = 1537 - 155)
CUDA_VISIBLE_DEVICES=0 python example.py --mode induce_answer --observation_length 1382 --observation_topk 256 --window_size 8

./run_plot_topk_indices.sh llama3 all 1537 256

## Generate Hit Rate
./run_plot_topk_indices.sh llama3 all 1537 256 true induced




# Run Example New Method
CUDA_VISIBLE_DEVICES=0 python example.py --max_new_tokens 4096 --rpc True --rpc_mode ours_window_merge_rkv --rpc_budget_cot 256