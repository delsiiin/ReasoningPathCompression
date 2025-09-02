#!/bin/bash

# 简单的Step-wise Attention热力图生成脚本
# 用法: ./run_step_wise.sh MODEL START_LAYER END_LAYER
# MODEL: qwen 或 llama

MODEL=${1:-llama}
START_LAYER=${2:-0}
END_LAYER=${3:-31}
VMAX=${4:-0.05}

# 根据模型设置tokenizer和attention目录
if [ "$MODEL" = "qwen" ]; then
    TOKENIZER_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ATTN_DIR="attn_heat_map/step_wise/qwen2"
elif [ "$MODEL" = "llama" ]; then
    TOKENIZER_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    ATTN_DIR="attn_heat_map/step_wise/llama3"
else
    echo "错误: 不支持的模型 '$MODEL'. 请使用 'qwen' 或 'llama'"
    exit 1
fi

echo "使用模型: $MODEL"
echo "生成Layer $START_LAYER 到 $END_LAYER 的step-wise attention热力图..."

for layer in $(seq $START_LAYER $END_LAYER); do
    echo "处理Layer $layer..."
    python plot_step_wise_attn_map.py --tokenizer_name "$TOKENIZER_NAME" --attn_dir "$ATTN_DIR" --output_dir "$ATTN_DIR" --layer_id $layer --verbose --skip_answer --vmax $VMAX
done

echo "完成!"
