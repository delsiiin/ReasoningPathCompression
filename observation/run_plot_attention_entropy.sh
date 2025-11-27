#!/bin/bash

# 批量处理qwen3文件夹中的所有注意力权重文件的执行脚本
# 作者: AI Assistant
# 功能: 为每一层生成注意力熵图表

echo "======================================"
echo "  注意力熵批量处理脚本 - qwen3"
echo "======================================"

# 设置路径变量
INPUT_DIR="/home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/qwen3"
OUTPUT_DIR="/home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/qwen3/attn_entropy"
INPUT_FILE="/home/yangx/zmw/ReasoningPathCompression/observation/output.jsonl"
TOKENIZER="Qwen/Qwen3-30B-A3B-Thinking-2507"
SCRIPT_DIR="/home/yangx/zmw/ReasoningPathCompression/observation"

echo "配置信息:"
echo "- 输入目录: $INPUT_DIR"
echo "- 输出目录: $OUTPUT_DIR"
echo "- 输入文件: $INPUT_FILE"
echo "- 分词器: $TOKENIZER"
echo ""

# 验证输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ 错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

# 验证输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "⚠️  警告: 输入文件不存在: $INPUT_FILE"
    echo "将在没有段落划分的模式下运行"
    EXTRA_ARGS="--no_paragraphs"
else
    echo "✓ 输入文件存在"
    EXTRA_ARGS=""
fi

# 验证脚本文件是否存在
if [ ! -f "$SCRIPT_DIR/plot_token_entropy.py" ]; then
    echo "❌ 错误: 脚本文件不存在: $SCRIPT_DIR/plot_token_entropy.py"
    exit 1
fi

# 检查输入目录中的文件数量
FILE_COUNT=$(ls "$INPUT_DIR"/attn_weights_layer_*.pt 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "❌ 错误: 在输入目录中没有找到任何 attn_weights_layer_*.pt 文件"
    exit 1
fi

echo "✓ 找到 $FILE_COUNT 个注意力权重文件"
echo ""

# 创建输出目录
echo "创建输出目录..."
mkdir -p "$OUTPUT_DIR"
if [ $? -eq 0 ]; then
    echo "✓ 输出目录创建成功: $OUTPUT_DIR"
else
    echo "❌ 错误: 无法创建输出目录"
    exit 1
fi

echo ""
echo "开始批量处理..."
echo "======================================"

# 切换到脚本目录
cd "$SCRIPT_DIR" || {
    echo "❌ 错误: 无法切换到脚本目录: $SCRIPT_DIR"
    exit 1
}

# 记录开始时间
START_TIME=$(date +%s)

# 执行批量处理
if [ -n "$EXTRA_ARGS" ]; then
    # 没有段落划分
    python plot_attention_entropy.py \
        --batch_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        $EXTRA_ARGS \
        --verbose \
        --skip_answer --entropy_stats
else
    # 有段落划分
    python plot_attention_entropy.py \
        --batch_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --input_file "$INPUT_FILE" \
        --tokenizer_name "$TOKENIZER" \
        --verbose \
        --skip_answer --entropy_stats
fi

# 获取执行结果
EXIT_CODE=$?

# 计算执行时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "======================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 批量处理完成！"
    
    # 统计生成的文件
    OUTPUT_COUNT=$(ls "$OUTPUT_DIR"/*.pdf 2>/dev/null | wc -l)
    echo "✓ 成功生成 $OUTPUT_COUNT 个图表文件"
    echo "✓ 输出目录: $OUTPUT_DIR"
    echo "✓ 执行时间: ${DURATION}秒"
    
    # 显示前几个生成的文件
    echo ""
    echo "生成的文件示例:"
    ls "$OUTPUT_DIR"/*.pdf 2>/dev/null | head -5 | while read file; do
        echo "  - $(basename "$file")"
    done
    
    if [ $OUTPUT_COUNT -gt 5 ]; then
        echo "  ... 以及其他 $((OUTPUT_COUNT - 5)) 个文件"
    fi
    
else
    echo "❌ 批量处理失败 (退出码: $EXIT_CODE)"
    echo "执行时间: ${DURATION}秒"
    
    # 检查是否有部分成功的文件
    PARTIAL_COUNT=$(ls "$OUTPUT_DIR"/*.pdf 2>/dev/null | wc -l)
    if [ $PARTIAL_COUNT -gt 0 ]; then
        echo "⚠️  部分成功: 已生成 $PARTIAL_COUNT 个文件"
    fi
fi

echo ""
echo "脚本执行完毕。"