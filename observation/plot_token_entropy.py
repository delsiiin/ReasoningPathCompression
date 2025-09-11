import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
from transformers import AutoTokenizer

def plot_token_entropy(tensor_path, dict_key=None, bins=50, output_path=None, title=None, 
                      input_file=None, tokenizer_name=None, skip_answer=False, verbose=False):
    """
    绘制token entropy的点线图，支持段落划分
    
    Args:
        tensor_path (str): 张量文件路径 (.pt文件)
        dict_key (str, optional): 如果张量保存为字典，指定要提取的键
        bins (int): 直方图的分箱数量（保留兼容性，实际不使用）
        output_path (str, optional): 输出图片的路径，如果不指定则显示图片
        title (str, optional): 图片标题
        input_file (str, optional): 输入的jsonl文件路径，用于段落划分
        tokenizer_name (str, optional): 分词器模型名称
        skip_answer (bool): 是否跳过answer部分
        verbose (bool): 是否显示详细信息
    """
    # 1) 读取二维张量
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"张量文件不存在: {tensor_path}")
    
    t = torch.load(tensor_path, map_location='cpu')
    
    # 如果是字典，提取指定的键
    if isinstance(t, dict):
        if dict_key is None:
            print(f"张量文件是字典类型，可用的键: {list(t.keys())}")
            print("请使用 --dict-key 参数指定要使用的键")
            sys.exit(1)
        if dict_key not in t:
            raise KeyError(f"字典中不存在键 '{dict_key}'，可用的键: {list(t.keys())}")
        t = t[dict_key]
    
    # 确保是二维张量
    if t.ndim != 2:
        raise ValueError(f"期望二维张量，但得到 {t.ndim} 维张量，形状: {t.shape}")
    
    print(f"张量形状: {t.shape}")
    
    # 2) 转为 NumPy 并压缩第一维
    t_np = t.detach().cpu().numpy()
    values = t_np[0, :]  # 压缩第一维，形状变为 (W,)
    print(f"压缩后的数据形状: {values.shape}")
    
    # 3) 段落划分（如果提供了input_file和tokenizer_name）
    para_boundaries = []
    para_first_words = []  # 存储每段的第一个词
    if input_file and tokenizer_name:
        print(f"加载分词器: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        print(f"读取文件: {input_file}")
        text = ""
        prompt_len = 0
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get("decoded_output", "")
                prompt_len = data.get("context_length", 0)
                break  # 只读取第一行
        
        # 对整个文本进行tokenize
        print("对文本进行tokenize...")
        tokens = tokenizer(text)['input_ids']
        token_texts = [tokenizer.decode([token]) for token in tokens]
        
        para_token_len_list = []
        para_token_start_idx_list = []
        cur_start_idx = 0
        current_para_len = 0
        
        newline_tokens = ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
        
        # 遍历tokens，查找段落分隔符
        i = 0
        while i < len(token_texts):
            current_para_len += 1
            
            # 检查当前token是否为段落分隔符
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
                
                if verbose:
                    print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}，第一个词 = {first_word}")
                    print(f"段落内容：{para_text}")
                    print("-" * 50)
                
                cur_start_idx += current_para_len
                current_para_len = 0

                # 检查是否跳过answer部分
                if skip_answer:
                    if "</think>" in para_text:
                        print("遇到</think>标签，停止处理")
                        break
            
            i += 1
        
        # 处理最后一个段落
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
            
            if verbose:
                print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}，第一个词 = {first_word}")
                print(f"段落内容：{para_text}")
                print("-" * 50)
            
            cur_start_idx += current_para_len
        
        para_token_start_idx_list.append(cur_start_idx)
        para_boundaries = para_token_start_idx_list
        
        print(f"总token数：{cur_start_idx}")
        print(f"段落数：{len(para_token_len_list)}")
        print(f"段落边界：{para_boundaries}")
        
        # 输出每个段落开头第一个token对应的entropy（总是显示）
        print("\n=== 每个段落开头第一个token的entropy值 ===")
        for i, start_idx in enumerate(para_token_start_idx_list[:-1]):  # 去掉最后一个边界
            if start_idx < len(values):
                entropy_val = values[start_idx]
                first_word = para_first_words[i] if i < len(para_first_words) else f"Para{i+1}"
                print(f"段落{i+1}: 位置{start_idx}, 第一个词='{first_word}', entropy={entropy_val:.6f}")
            else:
                print(f"段落{i+1}: 位置{start_idx} 超出数据范围")
        print("=" * 50)
        
        # 如果skip_answer为True，截断entropy数据
        if skip_answer and para_boundaries:
            max_position = para_boundaries[-1]  # 最后一个段落的结束位置
            if max_position < len(values):
                values = values[:max_position]
                print(f"截断后的数据长度: {len(values)}")
    
    # 4) 创建x轴坐标（token位置）
    x_positions = np.arange(len(values))
    
    # 清理非法值
    valid_mask = np.isfinite(values)
    x_clean = x_positions[valid_mask]
    y_clean = values[valid_mask]
    
    if len(y_clean) == 0:
        raise ValueError("张量中没有有效的有限值")
    
    print(f"有效数据点数: {len(y_clean)}")
    print(f"值的范围: [{y_clean.min():.6f}, {y_clean.max():.6f}]")
    print(f"均值: {y_clean.mean():.6f}, 标准差: {y_clean.std():.6f}")
    
    # 5) 绘制点线图
    plt.figure(figsize=(15, 6))
    plt.plot(x_clean, y_clean, 'o-', linewidth=1.0, markersize=2, alpha=0.8)
    
    # 添加段落分割线和标签
    if para_boundaries and para_first_words:
        for i, boundary in enumerate(para_boundaries[:-1]):  # 跳过最后一个边界
            if boundary < len(values):  # 确保边界在有效范围内
                # 添加垂直分割线
                if i > 0:  # 跳过第一个边界，因为它是起始位置
                    plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1)
                
                # 在段落边界处添加第一个词的标签
                if i < len(para_first_words):
                    # 计算标签位置：段落的中间位置
                    if i < len(para_boundaries) - 1:
                        label_x = (boundary + para_boundaries[i + 1]) / 2
                    else:
                        label_x = boundary + 10  # 如果是最后一段，稍微向右偏移
                    
                    # 获取该位置对应的y值，用于放置标签
                    if label_x < len(y_clean):
                        label_y = y_clean.max() * 0.9  # 放在图的上方
                    else:
                        label_y = y_clean.max() * 0.9
                    
                    plt.text(label_x, label_y, para_first_words[i], 
                            rotation=45, ha='center', va='bottom', 
                            fontsize=8, alpha=0.8, color='blue')
    
    plt.xlabel("Token Position")
    plt.ylabel("Token Entropy Value")
    
    if title is None:
        title = f"Token Entropy over Position (Shape: {t.shape})"
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存或显示图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="绘制token entropy张量值的点线图，支持段落划分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python plot_token_entropy.py --tensor_path tensor.pt
  python plot_token_entropy.py --tensor_path tensor.pt --dict-key entropy_values
  python plot_token_entropy.py --tensor_path tensor.pt --output entropy_plot.png
  python plot_token_entropy.py --tensor_path tensor.pt --input_file output.jsonl --tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
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
    
    # 段落划分相关参数
    parser.add_argument(
        "--input_file", "-i", 
        type=str, 
        default="output.jsonl",
        help="输入的jsonl文件路径，用于段落划分 (默认: observation/output.jsonl)"
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
        help="显示详细的段落信息"
    )
    
    parser.add_argument(
        "--no_paragraphs", 
        action="store_true",
        help="不进行段落划分，只绘制entropy曲线"
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
            tokenizer_name=None if args.no_paragraphs else args.tokenizer_name,
            skip_answer=args.skip_answer,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
