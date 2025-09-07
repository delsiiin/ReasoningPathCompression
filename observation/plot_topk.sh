#!/bin/bash
# 简易的topk_indices可视化脚本
# 用法: ./plot_topk.sh [模型] [层] [观察长度] [topk]

# 设置默认值
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_topk_indices.py"

# 帮助信息
show_help() {
    echo "用法:"
    echo "  ./plot_topk.sh [模型] [层] [观察长度] [topk]"
    echo ""
    echo "参数说明:"
    echo "  模型: llama3, qwen2 等"
    echo "  层: all, 单个数字(如 0), 多个数字(如 '0 1 15'), 范围(如 5-10)"
    echo "  观察长度: 序列长度，默认 1024"
    echo "  topk: 选择的位置数量，默认 512"
    echo ""
    echo "示例:"
    echo "  ./plot_topk.sh llama3 0                    # 处理llama3第0层，默认参数"
    echo "  ./plot_topk.sh llama3 all                  # 处理llama3所有层，默认参数"
    echo "  ./plot_topk.sh qwen2 '0 1 15'              # 处理qwen2的0,1,15层"
    echo "  ./plot_topk.sh llama3 5-10                 # 处理llama3的5到10层"
    echo "  ./plot_topk.sh llama3 0 2048               # 处理llama3第0层，观察长度2048"
    echo "  ./plot_topk.sh llama3 0 1024 256           # 处理llama3第0层，观察长度1024，topk=256"
    echo ""
    echo "默认值:"
    echo "  模型: llama3"
    echo "  层: all"
    echo "  观察长度: 1024"
    echo "  TopK: 512"
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

# 构建命令
if [ "$LAYERS" = "all" ]; then
    CMD="python3 $PYTHON_SCRIPT --model $MODEL --all-layers --observation-length $OBS_LENGTH --topk $TOPK"
else
    CMD="python3 $PYTHON_SCRIPT --model $MODEL --layers $LAYERS --observation-length $OBS_LENGTH --topk $TOPK"
fi

echo "执行: $CMD"
echo "模型: $MODEL, 层: $LAYERS, 观察长度: $OBS_LENGTH, TopK: $TOPK"
echo ""

# 执行命令
eval $CMD
