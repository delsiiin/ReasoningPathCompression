#!/bin/bash

# Token Entropy 绘图脚本

# 默认参数
MODEL_TYPE="llama"
INPUT_FILE="output.jsonl"

# 如果提供了参数，使用用户参数
if [ "$1" != "" ]; then
    MODEL_TYPE="$1"
fi

# 根据模型类型设置路径
case "$MODEL_TYPE" in
    "llama")
        TENSOR_PATH="token_entropy/llama3/entropy.pt"
        TOKENIZER_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        OUTPUT_PATH="token_entropy/llama3/token_entropy.pdf"
        ;;
    "qwen")
        TENSOR_PATH="token_entropy/qwen2/entropy.pt"
        TOKENIZER_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        OUTPUT_PATH="token_entropy/qwen2/token_entropy.pdf"
        ;;
    *)
        echo "错误: 不支持的模型类型 $MODEL_TYPE"
        echo "支持的类型: llama, qwen, glm"
        exit 1
        ;;
esac

# 检查文件是否存在
if [ ! -f "$TENSOR_PATH" ]; then
    echo "错误: 找不到张量文件 $TENSOR_PATH"
    exit 1
fi

# 运行Python脚本
echo "绘制token entropy图表..."
echo "张量文件: $TENSOR_PATH"
echo "输出文件: $OUTPUT_PATH"

python plot_token_entropy.py \
    --tensor_path "$TENSOR_PATH" \
    --input_file "$INPUT_FILE" \
    --output "$OUTPUT_PATH" \
    --tokenizer_name "$TOKENIZER_PATH" \
    --skip_answer --verbose

if [ $? -eq 0 ]; then
    echo "✓ 完成! 图片保存到: $OUTPUT_PATH"
else
    echo "✗ 绘图失败"
    exit 1
fi
