import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
from transformers import AutoTokenizer

def plot_token_confidence(tensor_path, dict_key=None, bins=50, output_path=None, title=None, 
                      input_file=None, tokenizer_name=None, skip_answer=False, verbose=False):
    """
    绘制token confidence的点线图，支持段落划分

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
                
                if verbose:
                    para_text = tokenizer.decode(tokens[cur_start_idx:cur_start_idx + current_para_len])
                    print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}")
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
            
            if verbose:
                para_text = tokenizer.decode(tokens[cur_start_idx:cur_start_idx + current_para_len])
                print(f"段落{len(para_token_len_list)}：token数 = {current_para_len}，开始索引 = {cur_start_idx}")
                print(f"段落内容：{para_text}")
                print("-" * 50)
            
            cur_start_idx += current_para_len
        
        para_token_start_idx_list.append(cur_start_idx)
        para_boundaries = para_token_start_idx_list
        
        print(f"总token数：{cur_start_idx}")
        print(f"段落数：{len(para_token_len_list)}")
        print(f"段落边界：{para_boundaries}")
        
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
    
    # 计算并输出每个段落的confidence平均值
    if para_boundaries and len(para_boundaries) > 1:
        print("\n" + "="*60)
        print("每个段落的confidence平均值:")
        print("="*60)
        
        for i in range(len(para_boundaries) - 1):
            start_idx = para_boundaries[i]
            end_idx = para_boundaries[i + 1]
            
            # 确保索引在有效范围内
            start_idx = max(0, min(start_idx, len(values) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(values)))
            
            # 提取该段落的confidence值
            para_values = values[start_idx:end_idx]
            
            # 计算有效值（排除nan和inf）
            valid_para_mask = np.isfinite(para_values)
            valid_para_values = para_values[valid_para_mask]
            
            if len(valid_para_values) > 0:
                para_mean = valid_para_values.mean()
                para_std = valid_para_values.std()
                print(f"段落 {i+1:2d}: 位置 [{start_idx:4d}, {end_idx:4d}), "
                      f"Token数: {end_idx - start_idx:3d}, "
                      f"平均Confidence: {para_mean:.6f}, "
                      f"标准差: {para_std:.6f}")
            else:
                print(f"段落 {i+1:2d}: 位置 [{start_idx:4d}, {end_idx:4d}), "
                      f"Token数: {end_idx - start_idx:3d}, "
                      f"平均Confidence: N/A (无有效值)")
        
        print("="*60)
    
    # 5) 绘制点线图
    plt.figure(figsize=(15, 6))
    plt.plot(x_clean, y_clean, 'o-', linewidth=1.0, markersize=2, alpha=0.8)
    
    # 添加段落分割线
    if para_boundaries:
        for boundary in para_boundaries[1:-1]:  # 跳过第一个和最后一个边界
            if boundary < len(values):  # 确保边界在有效范围内
                plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    plt.xlabel("Token Position")
    plt.ylabel("Token Confidence Value")
    
    if title is None:
        title = f"Token Confidence over Position (Shape: {t.shape})"
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
        description="绘制token confidence张量值的点线图，支持段落划分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python plot_token_confidence.py --tensor_path tensor.pt
  python plot_token_confidence.py --tensor_path tensor.pt --dict-key confidence_values
  python plot_token_confidence.py --tensor_path tensor.pt --output confidence_plot.png
  python plot_token_confidence.py --tensor_path tensor.pt --input_file output.jsonl --tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        """
    )
    
    parser.add_argument(
        "--tensor_path",
        help="输入的张量文件路径 (.pt文件)",
        default="/home/zmw/idea/context_compression/ReasoningPathCompression/observation/token_confidence/llama3/confidence.pt",
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
        plot_token_confidence(
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
