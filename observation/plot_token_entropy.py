import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer


THOUGHT_DELTA_THRESHOLD = 0.05


def load_entropy_values(tensor_path, dict_key=None):
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"张量文件不存在: {tensor_path}")

    tensor = torch.load(tensor_path, map_location="cpu")
    if isinstance(tensor, dict):
        if dict_key is None:
            print(f"张量文件是字典类型，可用的键: {list(tensor.keys())}")
            print("请使用 --dict-key 参数指定要使用的键")
            sys.exit(1)
        if dict_key not in tensor:
            raise KeyError(f"字典中不存在键 '{dict_key}'，可用的键: {list(tensor.keys())}")
        tensor = tensor[dict_key]

    if tensor.ndim != 2:
        raise ValueError(f"期望二维张量，但得到 {tensor.ndim} 维张量，形状: {tensor.shape}")

    values = tensor.detach().cpu().float().numpy()[0]
    print(f"张量形状: {tensor.shape}")
    print(f"压缩后的数据形状: {values.shape}")
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
        raise ValueError("需要提供包含 generated_token_ids 和 newline_token_ids 的 input_file")
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
        print("Warning: --skip_answer 需要 tokenizer_name 才能定位 </think>，本次不截断")
        return token_ids, values

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    start = find_subsequence(token_ids, end_think_ids)
    if start < 0:
        print("未找到 </think> token，保留全部 entropy")
        return token_ids, values

    end = start + len(end_think_ids)
    print(f"遇到 </think>，截断到 token 位置 {end}")
    return token_ids[:end], values[:end]


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


def attach_step_deltas(stats):
    prev_mean = None
    for item in stats:
        item["mean_delta"] = None if prev_mean is None else float(abs(item["mean"] - prev_mean))
        prev_mean = item["mean"]
    return stats


def build_delta_buckets(stats, bucket_size=0.1):
    if bucket_size <= 0:
        raise ValueError("bucket_size 必须大于 0")

    deltas = np.asarray(
        [item["mean_delta"] for item in stats if item.get("mean_delta") is not None],
        dtype=np.float64,
    )
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return []

    bucket_starts = np.floor(deltas / bucket_size) * bucket_size
    unique_starts, counts = np.unique(bucket_starts, return_counts=True)
    return [
        {
            "start": float(start),
            "end": float(start + bucket_size),
            "count": int(count),
            "label": f"[{start:.3f}, {start + bucket_size:.3f})",
        }
        for start, count in zip(unique_starts, counts)
    ]


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


def print_step_stats(stats, values, verbose=False):
    finite_values = np.asarray(values)[np.isfinite(values)]
    print(f"有效step数: {len(stats)}")
    print(f"有效数据点数: {len(finite_values)}")
    if len(finite_values) > 0:
        print(f"值的范围: [{finite_values.min():.6f}, {finite_values.max():.6f}]")
        print(f"均值: {finite_values.mean():.6f}, 标准差: {finite_values.std():.6f}")

    deltas = np.asarray(
        [item["mean_delta"] for item in stats if item.get("mean_delta") is not None],
        dtype=np.float64,
    )
    if len(deltas) > 0:
        print(f"相邻step mean entropy绝对差值范围: [{deltas.min():.6f}, {deltas.max():.6f}]")
        print(f"相邻step mean entropy绝对差值均值: {deltas.mean():.6f}, 标准差: {deltas.std():.6f}")

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


def print_delta_buckets(buckets):
    if not buckets:
        print("相邻step mean entropy差值bucket: 无")
        return

    print("\n=== 相邻step mean entropy绝对差值bucket频数 ===")
    for bucket in buckets:
        print(f"{bucket['label']}: {bucket['count']}")
    print("=" * 50)


def plot_step_stats(stats, output_path=None, title=None, bucket_size=0.1):
    import matplotlib.pyplot as plt

    step_ids = [item["step_id"] for item in stats]
    metric_specs = [
        ("mean", "Mean Entropy"),
        ("max", "Max Entropy"),
        ("min", "Min Entropy"),
        ("std", "Std Entropy"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, (metric, ylabel) in zip(axes.flat, metric_specs):
        y = [item[metric] for item in stats]
        ax.plot(step_ids, y, "o-", linewidth=1.4, markersize=3, alpha=0.85)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Step ID")
        ax.grid(True, alpha=0.3)

    delta_steps = [item["step_id"] for item in stats if item.get("mean_delta") is not None]
    deltas = [item["mean_delta"] for item in stats if item.get("mean_delta") is not None]
    delta_ax = axes[2, 0]
    delta_ax.axhline(y=0, color="gray", linewidth=1, alpha=0.6)
    delta_ax.plot(delta_steps, deltas, "o-", linewidth=1.4, markersize=3, alpha=0.85, color="#b6406b")
    delta_ax.set_title("Adjacent Mean Entropy Absolute Delta")
    delta_ax.set_xlabel("Step ID")
    delta_ax.set_ylabel("|Δ| Mean Entropy")
    delta_ax.grid(True, alpha=0.3)

    bucket_ax = axes[2, 1]
    buckets = build_delta_buckets(stats, bucket_size)
    if buckets:
        labels = [bucket["label"] for bucket in buckets]
        counts = [bucket["count"] for bucket in buckets]
        bucket_ax.bar(np.arange(len(buckets)), counts, color="#167a72", alpha=0.85)
        bucket_ax.set_xticks(np.arange(len(buckets)))
        bucket_ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    bucket_ax.set_title("Absolute Delta Bucket Frequency")
    bucket_ax.set_xlabel(f"|Δ| Mean Entropy Bucket (width={bucket_size:g})")
    bucket_ax.set_ylabel("Frequency")
    bucket_ax.grid(True, axis="y", alpha=0.3)

    if title is None:
        title = "Token Entropy Step Statistics"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()


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


def default_mean_entropy_output_path(output_path, start_step, end_step):
    """在主图旁生成 mean entropy 图的默认输出路径。"""
    if not output_path:
        return None

    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".png"
    return f"{root}_mean_entropy_steps_{start_step}-{end_step}{ext}"


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


def plot_mean_entropy(stats, step_range=None, output_path=None, title=None):
    """绘制指定 step 闭区间内按 thought 分段的 mean entropy。"""
    import matplotlib.pyplot as plt

    selected, start_step, end_step = select_step_range(stats, step_range)
    thoughts = select_visible_thoughts(
        build_thought_groups(stats), start_step, end_step
    )

    fig, (ax, strip_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [7, 1], "hspace": 0.06},
    )
    strip_colors = plt.get_cmap("Pastel2").colors
    display_offset = selected[0]["step_id"]
    step_ids = [item["step_id"] - display_offset for item in selected]
    means = [item["mean"] for item in selected]
    smooth_step_ids, smooth_means = build_smooth_curve(step_ids, means)
    ax.plot(smooth_step_ids, smooth_means, linewidth=1.5, alpha=0.9, color="#167a72")
    ax.scatter(step_ids, means, s=18, alpha=0.9, color="#167a72", zorder=3)

    for thought in thoughts[1:]:
        boundary = thought["start_step"] - display_offset - 0.5
        ax.axvline(boundary, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.7)

    for thought in thoughts:
        start = thought["start_step"] - display_offset
        end = thought["end_step"] - display_offset
        step_count = end - start + 1
        color = strip_colors[thought["thought_id"] % len(strip_colors)]
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
            fontsize=8,
            clip_on=True,
        )

    ax.set_ylabel("Mean Entropy")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, len(selected) - 0.5)
    strip_ax.set_ylim(0, 1)
    strip_ax.set_yticks([])
    strip_ax.set_ylabel("Thought", labelpad=18)
    strip_ax.set_xlabel("Step ID")
    strip_ax.spines[["top", "right", "left"]].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.11)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Mean entropy图片已保存到: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_token_entropy(tensor_path, dict_key=None, bins=50, output_path=None, title=None,
                       input_file=None, tokenizer_name=None, skip_answer=False, verbose=False,
                       delta_bucket_size=0.1, mean_entropy_step_range=None,
                       mean_entropy_output_path=None):
    """
    按 newline_token_ids 划分 step，并绘制每个 step 的 token entropy 统计。
    """
    _ = bins
    values = load_entropy_values(tensor_path, dict_key)
    token_ids, newline_token_ids, _ = load_token_metadata(input_file, tokenizer_name)
    token_ids, values = maybe_truncate_answer(token_ids, values, tokenizer_name, skip_answer)
    stats = build_step_stats(values, token_ids, newline_token_ids)
    buckets = build_delta_buckets(stats, delta_bucket_size)
    print_step_stats(stats, values, verbose)
    print_delta_buckets(buckets)
    plot_step_stats(stats, output_path, title, delta_bucket_size)

    _, start_step, end_step = select_step_range(stats, mean_entropy_step_range)
    if mean_entropy_output_path is None:
        mean_entropy_output_path = default_mean_entropy_output_path(
            output_path, start_step, end_step
        )
    plot_mean_entropy(
        stats,
        step_range=(start_step, end_step),
        output_path=mean_entropy_output_path,
        title=None if title is None else f"{title} - Mean Entropy",
    )


def main():
    parser = argparse.ArgumentParser(
        description="按newline_token_ids划分step，绘制token entropy统计图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python plot_token_entropy.py --tensor_path tensor.pt
  python plot_token_entropy.py --tensor_path tensor.pt --dict-key entropy_values
  python plot_token_entropy.py --tensor_path tensor.pt --output entropy_plot.png
  python plot_token_entropy.py --tensor_path tensor.pt --input_file output.jsonl --tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  python plot_token_entropy.py --tensor_path token_entropy/qwen3/entropy.pt --input_file output.jsonl --tokenizer_name Qwen/Qwen3-30B-A3B
  python plot_token_entropy.py --tensor_path tensor.pt --mean_entropy_step_range 20 80
        """
    )

    parser.add_argument(
        "--tensor_path",
        help="输入的张量文件路径 (.pt文件)",
        default="/home/zmw/idea/context_compression/ReasoningPathCompression/observation/token_entropy/llama3/entropy.pt",
    )

    parser.add_argument(
        "--dict-key",
        help="如果张量保存为字典，指定要提取的键名"
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="直方图的分箱数量 (保留兼容性，默认: 50)"
    )

    parser.add_argument(
        "--output", "-o",
        help="输出图片的路径 (支持 .png, .pdf, .svg 等格式)。如果不指定则显示图片",
        default="token_entropy.pdf"
    )

    parser.add_argument(
        "--title",
        help="图片标题 (如果不指定则使用默认标题)"
    )

    parser.add_argument(
        "--input_file", "-i",
        type=str,
        default="output.jsonl",
        help="输入的jsonl文件路径，用于step划分 (默认: output.jsonl)"
    )

    parser.add_argument(
        "--tokenizer_name", "-t",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="分词器模型名称 (默认: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)"
    )

    parser.add_argument(
        "--skip_answer",
        action="store_true",
        help="是否跳过answer部分的处理"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细的step统计信息"
    )

    parser.add_argument(
        "--delta_bucket_size",
        type=float,
        default=0.1,
        help="相邻step mean entropy绝对差值的bucket宽度 (默认: 0.1)"
    )

    parser.add_argument(
        "--mean_entropy_step_range", "--mean-entropy-step-range",
        nargs=2,
        type=int,
        metavar=("START_STEP", "END_STEP"),
        help=(
            "额外 mean entropy 图要截取的原始 step 闭区间（原图 step 从 1 开始）；"
            "额外图会从 0 重新显示"
        )
    )

    parser.add_argument(
        "--mean_entropy_output", "--mean-entropy-output",
        help=(
            "额外 mean entropy 图的输出路径；默认在 --output 同目录生成 "
            "<文件名>_mean_entropy_steps_<起点>-<终点>.<扩展名>"
        )
    )

    parser.add_argument(
        "--no_paragraphs",
        action="store_true",
        help="保留兼容性；新图仍需要input_file中的token元数据"
    )

    args = parser.parse_args()

    try:
        plot_token_entropy(
            tensor_path=args.tensor_path,
            dict_key=args.dict_key,
            bins=args.bins,
            output_path=args.output,
            title=args.title,
            input_file=None if args.no_paragraphs else args.input_file,
            tokenizer_name=args.tokenizer_name,
            skip_answer=args.skip_answer,
            verbose=args.verbose,
            delta_bucket_size=args.delta_bucket_size,
            mean_entropy_step_range=args.mean_entropy_step_range,
            mean_entropy_output_path=args.mean_entropy_output,
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
