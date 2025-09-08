#!/bin/bash
# 简易的topk_indices可视化脚本
# 用法: ./run_plot_topk_indices.sh [模型] [层] [观察长度] [topk] [hit-rate-only]

# 设置默认值
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_topk_indices.py"

# 帮助信息
show_help() {
    echo "用法:"
    echo "  ./run_plot_topk_indices.sh [模型] [层] [观察长度] [topk] [hit-rate-only]"
    echo ""
    echo "参数说明:"
    echo "  模型: llama3, qwen2 等"
    echo "  层: all, 单个数字(如 0), 多个数字(如 '0 1 15'), 范围(如 5-10)"
    echo "  观察长度: 序列长度，默认 1024"
    echo "  topk: 选择的位置数量，默认 512"
    echo "  hit-rate-only: true/false, 是否只计算hit rate，默认 false"
    echo ""
    echo "示例:"
    echo "  ./run_plot_topk_indices.sh llama3 0                    # 处理llama3第0层，默认参数"
    echo "  ./run_plot_topk_indices.sh llama3 all                  # 处理llama3所有层，默认参数"
    echo "  ./run_plot_topk_indices.sh qwen2 '0 1 15'              # 处理qwen2的0,1,15层"
    echo "  ./run_plot_topk_indices.sh llama3 5-10                 # 处理llama3的5到10层"
    echo "  ./run_plot_topk_indices.sh llama3 0 2048               # 处理llama3第0层，观察长度2048"
    echo "  ./run_plot_topk_indices.sh llama3 0 1024 256           # 处理llama3第0层，观察长度1024，topk=256"
    echo "  ./run_plot_topk_indices.sh llama3 0 1024 256 true      # 只计算hit rate"
    echo ""
    echo "默认值:"
    echo "  模型: llama3"
    echo "  层: all"
    echo "  观察长度: 1024"
    echo "  TopK: 512"
    echo "  Hit Rate Only: false"
}

# 检查帮助
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

# 设置参数
MODEL=${1:-llama3}
LAYERS=${2:-all}
OBS_LENGTH=${3:-1024}
TOPK=${4:-512}
HIT_RATE_ONLY=${5:-true}
REF_DATA_TYPE=${6:-induced}

# 构建命令
CMD_BASE="python3 $PYTHON_SCRIPT --model $MODEL --observation-length $OBS_LENGTH --topk $TOPK"

if [ "$LAYERS" = "all" ]; then
    CMD="$CMD_BASE --all-layers"
else
    CMD="$CMD_BASE --layers $LAYERS"
fi

# 添加hit-rate-only参数
if [ "$HIT_RATE_ONLY" = "true" ] || [ "$HIT_RATE_ONLY" = "1" ]; then
    CMD="$CMD --hit-rate-only --reference-data-type $REF_DATA_TYPE"
fi

echo "执行: $CMD"
echo "模型: $MODEL, 层: $LAYERS, 观察长度: $OBS_LENGTH, TopK: $TOPK, Hit Rate Only: $HIT_RATE_ONLY"
echo ""

# 执行命令
eval $CMD
