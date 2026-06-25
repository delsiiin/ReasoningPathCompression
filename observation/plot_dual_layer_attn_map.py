from transformers import AutoTokenizer, LlamaTokenizer
import torch
import argparse
import os
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.nn import functional as F

THOUGHT_DELTA_THRESHOLD = 0.05
MEAN_ENTROPY_COLOR = "darkgray"
THOUGHT_CATEGORY_COLORS = {
    "Progressive": "#a2aadb",
    "Backtracing": "#8eaedf",
    "Reflection": "#f4b484",
}
HARDCODED_THOUGHT_CATEGORIES = (
    "Progressive",
    "Reflection",
    "Backtracing",
    "Backtracing",
    "Progressive",
    "Reflection",
    "Reflection",
    "Backtracing",
    "Progressive",
)
FALLBACK_THOUGHT_CATEGORIES = ("Progressive", "Backtracing", "Reflection")
DEFAULT_TOKENIZER_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

def parse_args():
    parser = argparse.ArgumentParser(description="绘制mean entropy thought图与step-wise attention热力图")
    
    # 必需参数
    parser.add_argument("--input_file", "-i", type=str, default="output.jsonl",
                       help="输入的文本文件路径 (默认: output.jsonl)")
    
    # 分词器相关参数
    parser.add_argument("--tokenizer_name", "-t", type=str, 
                       default=DEFAULT_TOKENIZER_NAME,
                       help="分词器模型名称 (默认: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
    
    # Attention相关参数
    parser.add_argument("--attn_dir", type=str, 
                       default="/home/yangx/ReasoningPathCompression/observation/attn_heat_map_step/llama3",
                       help="attention权重文件目录路径")
    parser.add_argument("--layer_id", type=int, default=None,
                       help="要分析的层数 (默认: 10)")
    parser.add_argument("--layer_ids", nargs="+", type=int, default=None,
                       help="兼容旧参数；仅使用第一个层数")
    parser.add_argument("--skip_tokens", "-s", type=int, default=44,
                       help="跳过的prompt token数 (默认: 44)")
    parser.add_argument("--skip_answer", action="store_true", default=True,
                       help="是否跳过answer部分的注意力分数")

    # Entropy相关参数
    parser.add_argument("--entropy_input_file", type=str, default="output_entropy.jsonl",
                       help="entropy图使用的jsonl文件路径 (默认: output_entropy.jsonl)")
    parser.add_argument("--entropy_tensor_path", type=str, default="token_entropy/llama3/entropy.pt",
                       help="entropy张量文件路径 (默认: token_entropy/llama3/entropy.pt)")
    parser.add_argument("--entropy_dict_key", type=str, default=None,
                       help="如果entropy张量保存为字典，指定要提取的键名")
    parser.add_argument("--entropy_tokenizer_name", type=str, default=DEFAULT_TOKENIZER_NAME,
                       help="entropy图使用的分词器名称")
    parser.add_argument("--entropy_skip_answer", action="store_true", default=True,
                       help="是否跳过entropy图的answer部分")
    parser.add_argument("--mean_entropy_step_range", nargs=2, type=int, default=(18, 33),
                       metavar=("START_STEP", "END_STEP"),
                       help="mean entropy图要截取的原始step闭区间")

    # 输出相关参数
    parser.add_argument("--output_dir", "-o", type=str, default="/home/yangx/ReasoningPathCompression/observation/attn_heat_map_step/llama3",
                       help="输出的热力图文件夹")
    parser.add_argument("--figure_size", nargs=2, type=float, default=[18, 6.8],
                       help="图像大小 [宽, 高] (默认: [18, 6.8])")
    parser.add_argument("--vmax", type=float, default=0.05,
                       help="热力图颜色映射的最大值 (默认: 0.05)")
    parser.add_argument("--labelpad", type=float, default=0,
                       help="标签与坐标轴的距离 (默认: 20)")
    
    # 功能开关
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="显示详细的段落信息")
    parser.add_argument("--show_plot", action="store_true",
                       help="显示图像窗口")
    
    return parser.parse_args()

def get_layer_id(args):
    """获取要绘制的单个layer id。"""
    if args.layer_id is not None:
        return args.layer_id
    if args.layer_ids:
        return args.layer_ids[0]
    return 10

def load_entropy_values(tensor_path, dict_key=None):
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"张量文件不存在: {tensor_path}")

    tensor = torch.load(tensor_path, map_location="cpu")
    if isinstance(tensor, dict):
        if dict_key is None:
            raise ValueError(f"张量文件是字典类型，请使用 --entropy_dict_key 指定键；可用键: {list(tensor.keys())}")
        if dict_key not in tensor:
            raise KeyError(f"字典中不存在键 '{dict_key}'，可用的键: {list(tensor.keys())}")
        tensor = tensor[dict_key]

    if tensor.ndim != 2:
        raise ValueError(f"期望二维张量，但得到 {tensor.ndim} 维张量，形状: {tensor.shape}")

    values = tensor.detach().cpu().float().numpy()[0]
    print(f"entropy张量形状: {tensor.shape}")
    print(f"entropy压缩后的数据形状: {values.shape}")
    return values

def get_newline_token_ids(tokenizer):
    return [
        tokenizer.encode("\n")[-1],
        tokenizer.encode(".\n")[-1],
        tokenizer.encode(")\n")[-1],
        tokenizer.encode("\n\n")[-1],
        tokenizer.encode(".\n\n")[-1],
        tokenizer.encode(")\n\n")[-1],
    ]

def load_first_jsonl_record(input_file):
    if not input_file:
        raise ValueError("需要提供input_file")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)

    raise ValueError(f"输入文件为空: {input_file}")

def load_token_metadata(input_file, tokenizer_name=None):
    record = load_first_jsonl_record(input_file)
    generated_token_ids = record.get("generated_token_ids")
    newline_token_ids = record.get("newline_token_ids")

    if generated_token_ids is not None and newline_token_ids is not None:
        return [int(x) for x in generated_token_ids], [int(x) for x in newline_token_ids], record

    if tokenizer_name and record.get("decoded_output"):
        print("未找到 generated_token_ids/newline_token_ids，回退到 decoded_output 重新 tokenize")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        generated_token_ids = tokenizer(record["decoded_output"], add_special_tokens=False)["input_ids"]
        newline_token_ids = get_newline_token_ids(tokenizer)
        return generated_token_ids, newline_token_ids, record

    raise ValueError("input_file 中缺少 generated_token_ids/newline_token_ids，无法按 newline_token_ids 划分 step")

def find_subsequence(values, pattern):
    if not pattern or len(pattern) > len(values):
        return -1

    for start in range(len(values) - len(pattern) + 1):
        if values[start:start + len(pattern)] == pattern:
            return start
    return -1

def maybe_truncate_answer(token_ids, values, tokenizer_name, skip_answer):
    if not skip_answer:
        return token_ids, values
    if not tokenizer_name:
        print("Warning: --entropy_skip_answer 需要 entropy_tokenizer_name 才能定位 </think>，本次不截断")
        return token_ids, values

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    start = find_subsequence(token_ids, end_think_ids)
    if start < 0:
        print("未找到 </think> token，保留全部 entropy")
        return token_ids, values

    end = start + len(end_think_ids)
    print(f"遇到 </think>，entropy截断到 token 位置 {end}")
    return token_ids[:end], values[:end]

def add_step_stats(stats, values, start, end):
    step_values = values[start:end]
    step_values = step_values[np.isfinite(step_values)]
    if len(step_values) == 0:
        return

    stats.append({
        "step_id": len(stats) + 1,
        "start": start,
        "end": end,
        "mean": float(step_values.mean()),
        "max": float(step_values.max()),
        "min": float(step_values.min()),
        "std": float(step_values.std()),
    })

def attach_step_deltas(stats):
    prev_mean = None
    for item in stats:
        item["mean_delta"] = None if prev_mean is None else float(abs(item["mean"] - prev_mean))
        prev_mean = item["mean"]
    return stats

def build_step_stats(values, token_ids, newline_token_ids):
    length = min(len(values), len(token_ids))
    if length == 0:
        raise ValueError("没有可用于统计的 token entropy")
    if len(values) != len(token_ids):
        print(f"Warning: entropy长度({len(values)})和token长度({len(token_ids)})不一致，将使用前{length}个")

    values = np.asarray(values[:length], dtype=np.float64)
    token_ids = token_ids[:length]
    newline_token_ids = set(newline_token_ids)

    stats = []
    start = 0
    for idx, token_id in enumerate(token_ids):
        if token_id in newline_token_ids:
            add_step_stats(stats, values, start, idx + 1)
            start = idx + 1

    if start < length:
        add_step_stats(stats, values, start, length)

    if not stats:
        raise ValueError("未得到有效 step，无法绘图")
    attach_step_deltas(stats)
    return stats

def print_step_stats(stats, values, verbose=False):
    finite_values = np.asarray(values)[np.isfinite(values)]
    print(f"entropy有效step数: {len(stats)}")
    print(f"entropy有效数据点数: {len(finite_values)}")
    if len(finite_values) > 0:
        print(f"entropy值的范围: [{finite_values.min():.6f}, {finite_values.max():.6f}]")
        print(f"entropy均值: {finite_values.mean():.6f}, 标准差: {finite_values.std():.6f}")

    if verbose:
        print("\n=== Step entropy统计 ===")
        for item in stats:
            delta = item["mean_delta"]
            delta_text = "n/a" if delta is None else f"{delta:.6f}"
            print(
                f"step {item['step_id']}: token[{item['start']}:{item['end']}], "
                f"mean={item['mean']:.6f}, max={item['max']:.6f}, "
                f"min={item['min']:.6f}, std={item['std']:.6f}, delta={delta_text}"
            )
        print("=" * 50)

def select_step_range(stats, step_range=None):
    """返回闭区间 ``[start_step, end_step]`` 内的 step 统计。"""
    if not stats:
        raise ValueError("没有可用于绘制 mean entropy 的 step")

    first_step = stats[0]["step_id"]
    last_step = stats[-1]["step_id"]
    if step_range is None:
        start_step, end_step = first_step, last_step
    else:
        start_step, end_step = step_range
        if start_step > end_step:
            raise ValueError("mean entropy 的 step 区间起点不能大于终点")
        if start_step < first_step or end_step > last_step:
            raise ValueError(
                f"mean entropy 的 step 区间必须位于 [{first_step}, {last_step}]，"
                f"当前为 [{start_step}, {end_step}]"
            )

    selected = [
        item for item in stats
        if start_step <= item["step_id"] <= end_step
    ]
    if not selected:
        raise ValueError("指定的 mean entropy step 区间没有有效数据")
    return selected, start_step, end_step

def build_smooth_curve(x_values, y_values, points_per_interval=20):
    """用三次 Hermite 插值生成经过所有数据点的平滑曲线。"""
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if len(x_values) < 3:
        return x_values, y_values

    slopes = np.gradient(y_values, x_values)
    smooth_x = []
    smooth_y = []
    for idx in range(len(x_values) - 1):
        x0, x1 = x_values[idx], x_values[idx + 1]
        y0, y1 = y_values[idx], y_values[idx + 1]
        interval = x1 - x0
        t = np.linspace(0, 1, points_per_interval, endpoint=False)
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        smooth_x.extend(x0 + t * interval)
        smooth_y.extend(h00 * y0 + h10 * interval * slopes[idx] +
                        h01 * y1 + h11 * interval * slopes[idx + 1])

    smooth_x.append(x_values[-1])
    smooth_y.append(y_values[-1])
    return np.asarray(smooth_x), np.asarray(smooth_y)

def build_thought_groups(stats, delta_threshold=THOUGHT_DELTA_THRESHOLD):
    """按相邻 step 的 mean entropy 绝对差值划分连续 thought。"""
    if not stats:
        return []

    thoughts = []
    current_steps = [stats[0]]
    previous_mean = stats[0]["mean"]
    for item in stats[1:]:
        if abs(item["mean"] - previous_mean) >= delta_threshold:
            thoughts.append({
                "thought_id": len(thoughts),
                "start_step": current_steps[0]["step_id"],
                "end_step": current_steps[-1]["step_id"],
                "steps": current_steps,
            })
            current_steps = [item]
        else:
            current_steps.append(item)
        previous_mean = item["mean"]

    thoughts.append({
        "thought_id": len(thoughts),
        "start_step": current_steps[0]["step_id"],
        "end_step": current_steps[-1]["step_id"],
        "steps": current_steps,
    })
    return thoughts

def select_visible_thoughts(thoughts, start_step, end_step):
    """裁剪到目标区间，并将可见 thought 从 T0 重新编号。"""
    visible_thoughts = []
    for thought in thoughts:
        visible_steps = [
            item for item in thought["steps"]
            if start_step <= item["step_id"] <= end_step
        ]
        if visible_steps:
            visible_thoughts.append({
                **thought,
                "thought_id": len(visible_thoughts),
                "start_step": visible_steps[0]["step_id"],
                "end_step": visible_steps[-1]["step_id"],
                "steps": visible_steps,
            })
    return visible_thoughts

def get_thought_category(thought_id):
    """返回 Thought 色块使用的固定类别。"""
    if thought_id < len(HARDCODED_THOUGHT_CATEGORIES):
        return HARDCODED_THOUGHT_CATEGORIES[thought_id]
    return FALLBACK_THOUGHT_CATEGORIES[thought_id % len(FALLBACK_THOUGHT_CATEGORIES)]

def load_entropy_step_stats(tensor_path, input_file, tokenizer_name=None, dict_key=None,
                            skip_answer=False, verbose=False):
    values = load_entropy_values(tensor_path, dict_key)
    token_ids, newline_token_ids, _ = load_token_metadata(input_file, tokenizer_name)
    token_ids, values = maybe_truncate_answer(token_ids, values, tokenizer_name, skip_answer)
    stats = build_step_stats(values, token_ids, newline_token_ids)
    print_step_stats(stats, values, verbose)
    return stats

def draw_mean_entropy_subplot(stats, ax, strip_ax, step_range=None):
    """绘制指定 step 闭区间内按 thought 分段的 mean entropy。"""
    selected, start_step, end_step = select_step_range(stats, step_range)
    thoughts = select_visible_thoughts(
        build_thought_groups(stats), start_step, end_step
    )

    display_offset = selected[0]["step_id"]
    step_ids = [item["step_id"] - display_offset for item in selected]
    means = [item["mean"] for item in selected]
    smooth_step_ids, smooth_means = build_smooth_curve(step_ids, means)
    marker_interval = 20 if len(step_ids) >= 3 else 1
    ax.plot(
        smooth_step_ids,
        smooth_means,
        linewidth=2,
        alpha=0.9,
        color=MEAN_ENTROPY_COLOR,
        marker="o",
        markersize=6,
        markevery=marker_interval,
    )

    for thought in thoughts[1:]:
        boundary = thought["start_step"] - display_offset - 0.5
        ax.axvline(boundary, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.7)

    for thought in thoughts:
        start = thought["start_step"] - display_offset
        end = thought["end_step"] - display_offset
        step_count = end - start + 1
        category = get_thought_category(thought["thought_id"])
        color = THOUGHT_CATEGORY_COLORS[category]
        strip_ax.barh(
            0.5,
            step_count,
            left=start - 0.5,
            height=1,
            color=color,
            edgecolor="white",
            linewidth=1,
        )
        label = f"T{thought['thought_id']}"
        strip_ax.text(
            (start + end) / 2,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=16,
            clip_on=True,
        )

    ax.set_ylabel("Mean Entropy", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelbottom=False, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=category)
            for category, color in THOUGHT_CATEGORY_COLORS.items()
        ],
        loc="upper right",
        frameon=True,
        fontsize=16,
    )
    ax.set_xlim(-0.5, len(selected) - 0.5)
    strip_ax.set_ylim(0, 1)
    strip_ax.set_yticks([])
    strip_ax.tick_params(axis="x", labelsize=16)
    strip_ax.tick_params(axis="y", labelsize=16)
    strip_ax.set_ylabel("Thought", labelpad=18, fontsize=18)
    strip_ax.set_xlabel("Step ID", fontsize=18)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    strip_ax.yaxis.set_label_coords(-0.08, 0.5)
    strip_ax.spines[["top", "right", "left"]].set_visible(False)

def load_and_process_attention(attn_dir, layer_id, attn_score_tensor_template, skip_answer, prompt_len, cur_start_idx, tokenizer_name):
    """加载并处理指定层的attention权重"""
    save_path = f'{attn_dir}/attn_weights_layer_{layer_id}.pt'
    
    if not os.path.exists(save_path):
        print(f"错误：找不到attention权重文件 {save_path}")
        return None
    
    print(f"加载attention权重: {save_path}")
    attn_score_tensor = torch.load(save_path, map_location="cpu")
    
    # 跳过prompt
    if skip_answer:
        # Find the index where "</think>" appears
        before_ans = cur_start_idx + prompt_len
        
        if before_ans is not None:
            attn_score_tensor = attn_score_tensor[prompt_len:before_ans, prompt_len:before_ans]
            # 将上三角部分赋值为无穷小（不包括对角线）
            mask = torch.triu(torch.ones_like(attn_score_tensor, dtype=torch.bool), diagonal=1)
            attn_score_tensor = attn_score_tensor.masked_fill(mask, float('-inf'))
            if "gpt" in tokenizer_name.lower():
                attn_score_tensor = F.softmax(attn_score_tensor, dim=-1)
                attn_score_tensor = attn_score_tensor[..., :-1]
            else:
                attn_score_tensor = attn_score_tensor.softmax(dim=-1)
    else:
        attn_score_tensor = attn_score_tensor[prompt_len:, prompt_len:]
        # 将上三角部分赋值为无穷小（不包括对角线）
        mask = torch.triu(torch.ones_like(attn_score_tensor, dtype=torch.bool), diagonal=1)
        attn_score_tensor = attn_score_tensor.masked_fill(mask, float('-inf'))
        if "gpt" in tokenizer_name.lower():
            attn_score_tensor = F.softmax(attn_score_tensor, dim=-1)
            attn_score_tensor = attn_score_tensor[..., :-1]
        else:
            attn_score_tensor = attn_score_tensor.softmax(dim=-1)
    
    return attn_score_tensor

def compute_step_wise_attention(attn_score_tensor, para_token_start_idx_list, verbose=False):
    """计算step-wise attention分数"""
    step_wise_attn_score = []

    for idx in range(len(para_token_start_idx_list)-1):
        cur_step_attn_score = []
        # Ensure the range does not go out of bounds
        for i in range(1, len(para_token_start_idx_list) - idx -1):
            start_i = para_token_start_idx_list[idx+i]
            end_i = para_token_start_idx_list[idx+i+1]
            start_idx = para_token_start_idx_list[idx]
            end_idx = para_token_start_idx_list[idx+1]
            # Check if indices are within tensor bounds
            if end_i <= attn_score_tensor.shape[0] and end_idx <= attn_score_tensor.shape[1]:
                para_attn_scores = attn_score_tensor[start_i:end_i, start_idx:end_idx].sum(1).mean(0)
                cur_step_attn_score.append(para_attn_scores.item())
            else:
                if verbose:
                    print(f"跳过超出边界的切片: [{start_i}:{end_i}, {start_idx}:{end_idx}]")
        if len(cur_step_attn_score) > 0:
            step_wise_attn_score.append(cur_step_attn_score)
    
    return step_wise_attn_score

def create_attention_matrix(step_wise_attn_score):
    """构建attention矩阵"""
    max_len = len(step_wise_attn_score)
    matrix = np.zeros((max_len, max_len))

    for i, col in enumerate(step_wise_attn_score):
        for j, value in enumerate(col):
            matrix[i+j, i] = value

    # 创建mask来只显示下三角矩阵
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    return matrix, mask

def classify_scores_as_prb(scores):
    """将分数按大小分为P/B/R三档：高=P，中=B，低=R。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.full(scores.shape, "B", dtype=object)
    valid_mask = np.isfinite(scores)
    valid_scores = scores[valid_mask]

    if len(valid_scores) <= 1 or np.allclose(valid_scores, valid_scores[0]):
        return labels.tolist()

    order = np.argsort(valid_scores, kind="stable")
    valid_labels = np.full(valid_scores.shape, "B", dtype=object)
    low_cut = len(valid_scores) // 3
    high_cut = (2 * len(valid_scores)) // 3

    valid_labels[order[:low_cut]] = "R"
    valid_labels[order[low_cut:high_cut]] = "B"
    valid_labels[order[high_cut:]] = "P"
    labels[valid_mask] = valid_labels

    return labels.tolist()

def create_prb_axis_labels(matrix, mask):
    """根据当前热力图矩阵动态生成x/y轴的P/B/R标签。"""
    visible_matrix = np.asarray(matrix, dtype=float).copy()
    visible_matrix[mask] = np.nan

    x_scores = np.nanmean(visible_matrix, axis=0)
    y_scores = np.nanmean(visible_matrix, axis=1)

    return classify_scores_as_prb(x_scores), classify_scores_as_prb(y_scores)

def build_para_boundaries_from_token_ids(token_ids, newline_token_ids, verbose=False):
    para_token_len_list = []
    para_token_start_idx_list = []
    cur_start_idx = 0
    current_para_len = 0
    newline_token_ids = set(newline_token_ids)

    for token_id in token_ids:
        current_para_len += 1
        if token_id in newline_token_ids:
            para_token_len_list.append(current_para_len)
            para_token_start_idx_list.append(cur_start_idx)
            if verbose:
                print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}")
            cur_start_idx += current_para_len
            current_para_len = 0

    if current_para_len > 0:
        para_token_len_list.append(current_para_len)
        para_token_start_idx_list.append(cur_start_idx)
        if verbose:
            print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}")
        cur_start_idx += current_para_len

    para_token_start_idx_list.append(cur_start_idx)
    return para_token_len_list, para_token_start_idx_list, cur_start_idx

def main():
    args = parse_args()
    layer_id = get_layer_id(args)

    print("读取并处理mean entropy数据...")
    entropy_stats = load_entropy_step_stats(
        tensor_path=args.entropy_tensor_path,
        input_file=args.entropy_input_file,
        tokenizer_name=args.entropy_tokenizer_name,
        dict_key=args.entropy_dict_key,
        skip_answer=args.entropy_skip_answer,
        verbose=args.verbose,
    )

    # 读取jsonl文件内容
    print(f"读取文件: {args.input_file}")
    data = load_first_jsonl_record(args.input_file)
    prompt_len = data.get("context_length", 0)
    generated_token_ids = data.get("generated_token_ids")
    newline_token_ids = data.get("newline_token_ids")

    if generated_token_ids is not None and newline_token_ids is not None and not args.skip_answer:
        print("使用jsonl中的generated_token_ids/newline_token_ids划分attention段落")
        para_token_len_list, para_token_start_idx_list, cur_start_idx = build_para_boundaries_from_token_ids(
            [int(x) for x in generated_token_ids],
            [int(x) for x in newline_token_ids],
            args.verbose,
        )
    else:
        # 加载分词器
        print(f"加载分词器: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        text = data.get("decoded_output", "")

        # 先对整个文本进行tokenize
        print("对文本进行tokenize...")
        tokens = tokenizer(text)['input_ids']
        token_texts = [tokenizer.decode([token]) for token in tokens]

        para_token_len_list = []
        para_token_start_idx_list = []
        para_first_words = []  # 存储每段的第一个词
        cur_start_idx = 0
        current_para_len = 0

        newline_tokens = ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]

        # 遍历tokens，查找\n\n作为单独token的段落分隔符
        i = 0
        while i < len(token_texts):
            current_para_len += 1

            # 检查当前token是否为\n\n
            if token_texts[i] in newline_tokens:
                # 找到段落分隔符，记录当前段落
                para_token_len_list.append(current_para_len)
                para_token_start_idx_list.append(cur_start_idx)

                # 提取段落内容和第一个词
                para_text = tokenizer.decode(tokens[cur_start_idx:cur_start_idx + current_para_len])
                # 获取第一个有意义的词（跳过空格和特殊字符）
                first_word = ""
                for token_idx in range(cur_start_idx, cur_start_idx + current_para_len):
                    if token_idx < len(tokens):
                        word = tokenizer.decode([tokens[token_idx]]).strip()
                        if word and word not in [" ", "\n", "\t"] and len(word) > 0:
                            first_word = word[:10]  # 限制长度避免标签太长
                            break
                if not first_word:
                    first_word = f"Para{len(para_token_len_list)}"
                para_first_words.append(first_word)

                # 根据verbose参数决定是否打印段落内容
                if args.verbose:
                    print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}，第一个词 = {first_word}")
                    print(f"段落内容：{para_text}")
                    print("-" * 50)

                cur_start_idx += current_para_len
                current_para_len = 0

                if args.skip_answer and "</think>" in para_text:
                    print(args.skip_answer)
                    break

            i += 1

        # 处理最后一个段落（如果存在）
        if current_para_len > 0:
            para_token_len_list.append(current_para_len)
            para_token_start_idx_list.append(cur_start_idx)

            # 提取最后一个段落的第一个词
            para_text = tokenizer.decode(tokens[cur_start_idx:cur_start_idx + current_para_len])
            first_word = ""
            for token_idx in range(cur_start_idx, cur_start_idx + current_para_len):
                if token_idx < len(tokens):
                    word = tokenizer.decode([tokens[token_idx]]).strip()
                    if word and word not in [" ", "\n", "\t"] and len(word) > 0:
                        first_word = word[:10]  # 限制长度避免标签太长
                        break
            if not first_word:
                first_word = f"Para{len(para_token_len_list)}"
            para_first_words.append(first_word)

            # 根据verbose参数决定是否打印最后一个段落内容
            if args.verbose:
                print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}，第一个词 = {first_word}")
                print(f"段落内容：{para_text}")
                print("-" * 50)

            cur_start_idx += current_para_len

        para_token_start_idx_list.append(cur_start_idx)

    print(f"总token数：{cur_start_idx}")
    print(f"段落数：{len(para_token_len_list)}")

    # 确保attention目录存在
    os.makedirs(args.attn_dir, exist_ok=True)
    
    print(f"\n处理第 {layer_id} 层...")

    # 加载并处理attention权重
    attn_score_tensor = load_and_process_attention(
        args.attn_dir, layer_id, None, args.skip_answer,
        prompt_len, cur_start_idx, args.tokenizer_name
    )

    if attn_score_tensor is None:
        return

    # 计算step-wise attention分数
    print(f"计算第 {layer_id} 层的step-wise attention分数...")
    step_wise_attn_score = compute_step_wise_attention(
        attn_score_tensor, para_token_start_idx_list, args.verbose
    )

    # 构建attention矩阵
    matrix, mask = create_attention_matrix(step_wise_attn_score)
    x_labels, y_labels = create_prb_axis_labels(matrix, mask)

    # 绘制合并大图：左侧mean entropy thought，右侧step attention heatmap
    fig = plt.figure(figsize=tuple(args.figure_size))
    outer_grid = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.2], wspace=0.18)
    entropy_grid = outer_grid[0].subgridspec(
        2,
        1,
        height_ratios=[7, 1],
        hspace=0.06,
    )
    entropy_ax = fig.add_subplot(entropy_grid[0])
    strip_ax = fig.add_subplot(entropy_grid[1], sharex=entropy_ax)
    ax = fig.add_subplot(outer_grid[1])

    draw_mean_entropy_subplot(
        entropy_stats,
        entropy_ax,
        strip_ax,
        step_range=args.mean_entropy_step_range,
    )

    sns.heatmap(matrix, annot=False, cmap='RdYlBu_r', cbar=True,
           vmin=0, vmax=args.vmax, mask=mask, ax=ax, xticklabels=x_labels, yticklabels=y_labels)
    ax.tick_params(axis='x', rotation=0, labelsize=16)
    ax.tick_params(axis='y', rotation=90, labelsize=16)
    if ax.collections and ax.collections[0].colorbar is not None:
        ax.collections[0].colorbar.ax.tick_params(labelsize=16)
    ax.set_xlabel('Previous Thoughts', labelpad=args.labelpad, fontsize=18)
    ax.set_ylabel('Current Thoughts', labelpad=args.labelpad, fontsize=18)
    # ax.set_xticks([])  # 隐藏x轴刻度
    # ax.set_yticks([])  # 隐藏y轴刻度
    
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.18)
    entropy_pos = strip_ax.get_position()
    attention_pos = ax.get_position()
    fig.text(
        (entropy_pos.x0 + entropy_pos.x1) / 2,
        0.07,
        "(a) Sharp fluctuations in step entropy indicate thought transitions",
        ha="center",
        va="center",
        fontsize=20
    )
    fig.text(
        (attention_pos.x0 + attention_pos.x1) / 2,
        0.07,
        f"(b) Thought-level Attention (Layer {layer_id} Head 3)",
        ha="center",
        va="center",
        fontsize=20
    )

    if args.show_plot:
        plt.show()
    else:
        # 保存图像
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f'{args.output_dir}/combined_entropy_attention_layer_{layer_id}.pdf'
        fig.savefig(output_filename, bbox_inches='tight', dpi=300)
        print(f"合并热力图已保存至: {output_filename}")
    plt.close(fig)

if __name__ == "__main__":
    main()
