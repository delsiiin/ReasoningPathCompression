#!/bin/bash

# Token Entropy 绘图脚本

# 默认参数
MODEL_TYPE="llama"
INPUT_FILE="output.jsonl"
# 如果提供了第二个参数作为压缩方法，使用它作为后缀
if [ "$2" != "" ]; then
    INPUT_FILE="output_$2.jsonl"
fi

# 如果提供了参数，使用用户参数
if [ "$1" != "" ]; then
    MODEL_TYPE="$1"
fi

# 根据模型类型设置路径
case "$MODEL_TYPE" in
    "llama"|"llama3")
        # 如果提供了第二个参数作为压缩方法，使用它作为后缀
        if [ "$2" != "" ]; then
            TENSOR_PATH="token_entropy/llama3_$2/entropy.pt"
            OUTPUT_PATH="token_entropy/llama3_$2/token_entropy.pdf"
        else
            TENSOR_PATH="token_entropy/llama3/entropy.pt"
            OUTPUT_PATH="token_entropy/llama3/token_entropy.pdf"
        fi
        TOKENIZER_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        ;;
    "qwen"|"qwen2")
        # 如果提供了第二个参数作为压缩方法，使用它作为后缀
        if [ "$2" != "" ]; then
            TENSOR_PATH="token_entropy/qwen2_$2/entropy.pt"
            OUTPUT_PATH="token_entropy/qwen2_$2/token_entropy.pdf"
        else
            TENSOR_PATH="token_entropy/qwen2/entropy.pt"
            OUTPUT_PATH="token_entropy/qwen2/token_entropy.pdf"
        fi
        TOKENIZER_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        ;;
    "qwen3")
        if [ "$2" != "" ]; then
            TENSOR_PATH="token_entropy/qwen3_$2/entropy.pt"
            OUTPUT_PATH="token_entropy/qwen3_$2/token_entropy.pdf"
        else
            TENSOR_PATH="token_entropy/qwen3/entropy.pt"
            OUTPUT_PATH="token_entropy/qwen3/token_entropy.pdf"
        fi
        TOKENIZER_PATH="Qwen/Qwen3-30B-A3B"
        ;;
    "gpt"|"oss"|"gpt_oss")
        if [ "$2" != "" ]; then
            TENSOR_PATH="token_entropy/oss_$2/entropy.pt"
            OUTPUT_PATH="token_entropy/oss_$2/token_entropy.pdf"
        else
            TENSOR_PATH="token_entropy/oss/entropy.pt"
            OUTPUT_PATH="token_entropy/oss/token_entropy.pdf"
        fi
        TOKENIZER_PATH="openai/gpt-oss-20b"
        ;;
    *)
        echo "错误: 不支持的模型类型 $MODEL_TYPE"
        echo "支持的类型: llama, qwen/qwen2, qwen3, gpt/oss"
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
