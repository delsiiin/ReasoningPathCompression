from transformers import AutoTokenizer, LlamaTokenizer
import torch
import argparse
import os
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="分析文本段落的token数并绘制step-wise attention热力图")
    
    # 必需参数
    parser.add_argument("--input_file", "-i", type=str, default="output.jsonl",
                       help="输入的文本文件路径 (默认: output.jsonl)")
    
    # 分词器相关参数
    parser.add_argument("--tokenizer_name", "-t", type=str, 
                       default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                       help="分词器模型名称 (默认: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
    
    # Attention相关参数
    parser.add_argument("--attn_dir", type=str, 
                       default="/home/zmw/idea/context_compression/ReasoningPathCompression/observation/attn_heat_map_step/llama3",
                       help="attention权重文件目录路径")
    parser.add_argument("--layer_id", "-l", type=int, default=15,
                       help="要分析的层数 (默认: 20)")
    parser.add_argument("--skip_tokens", "-s", type=int, default=44,
                       help="跳过的prompt token数 (默认: 44)")
    parser.add_argument("--skip_answer", action="store_true",
                       help="是否跳过answer部分的注意力分数")

    # 输出相关参数
    parser.add_argument("--output_dir", "-o", type=str, default="/home/zmw/idea/context_compression/ReasoningPathCompression/attn_heat_map_step/llama3",
                       help="输出的热力图文件夹")
    parser.add_argument("--figure_size", nargs=2, type=int, default=[10, 8],
                       help="图像大小 [宽, 高] (默认: [10, 8])")
    parser.add_argument("--vmax", type=float, default=0.05,
                       help="热力图颜色映射的最大值 (默认: 0.08)")
    
    # 功能开关
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="显示详细的段落信息")
    parser.add_argument("--show_plot", action="store_true",
                       help="显示图像窗口")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载分词器
    print(f"加载分词器: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # 读取jsonl文件内容
    print(f"读取文件: {args.input_file}")
    text = ""
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            text = data.get("decoded_output", "")
            prompt_len = data.get("context_length", 0)
            break  # 只读取第一行，如果需要处理多行请修改此处

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

    # 加载attention权重
    os.makedirs(args.attn_dir, exist_ok=True)
    save_path = f'{args.attn_dir}/attn_weights_layer_{args.layer_id}.pt'
    
    if not os.path.exists(save_path):
        print(f"错误：找不到attention权重文件 {save_path}")
        return
    
    print(f"加载attention权重: {save_path}")
    attn_score_tensor = torch.load(save_path)
    
    # 跳过prompt
    if args.skip_answer:
        # Find the index where "</think>" appears
        before_ans = cur_start_idx + prompt_len
        
        if before_ans is not None:
            attn_score_tensor = attn_score_tensor[prompt_len:before_ans, prompt_len:before_ans]
            # 将上三角部分赋值为无穷小（不包括对角线）
            mask = torch.triu(torch.ones_like(attn_score_tensor, dtype=torch.bool), diagonal=1)
            attn_score_tensor = attn_score_tensor.masked_fill(mask, float('-inf'))
            if "gpt" in args.tokenizer_name.lower():
                attn_score_tensor = F.softmax(attn_score_tensor, dim=-1)
                attn_score_tensor = attn_score_tensor[..., :-1]
            else:
                attn_score_tensor = attn_score_tensor.softmax(dim=-1)
    else:
        attn_score_tensor = attn_score_tensor[prompt_len:, prompt_len:]
        # 将上三角部分赋值为无穷小（不包括对角线）
        mask = torch.triu(torch.ones_like(attn_score_tensor, dtype=torch.bool), diagonal=1)
        attn_score_tensor = attn_score_tensor.masked_fill(mask, float('-inf'))
        if "gpt" in args.tokenizer_name.lower():
            attn_score_tensor = F.softmax(attn_score_tensor, dim=-1)
            attn_score_tensor = attn_score_tensor[..., :-1]
        else:
            attn_score_tensor = attn_score_tensor.softmax(dim=-1)

    # 计算step-wise attention分数
    print("计算step-wise attention分数...")
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
                if args.verbose:
                    print(f"跳过超出边界的切片: [{start_i}:{end_i}, {start_idx}:{end_idx}]")
        if len(cur_step_attn_score) > 0:
            step_wise_attn_score.append(cur_step_attn_score)

    # 构建下三角矩阵
    print("构建attention矩阵并绘制热力图...")
    max_len = len(step_wise_attn_score)
    matrix = np.zeros((max_len, max_len))

    for i, col in enumerate(step_wise_attn_score):
        for j, value in enumerate(col):
            matrix[i+j, i] = value

    # 创建mask来只显示下三角矩阵
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    # 创建标签列表，使用段落的第一个词
    x_labels = para_first_words[:max_len] if len(para_first_words) >= max_len else para_first_words + [f"Para{i+len(para_first_words)+1}" for i in range(max_len - len(para_first_words))]
    y_labels = para_first_words[1:max_len+1] if len(para_first_words) >= max_len+1 else para_first_words[1:] + [f"Para{i+len(para_first_words)}" for i in range(max_len+1 - len(para_first_words))]
    
    # 绘制热力图
    plt.figure(figsize=tuple(args.figure_size))
    sns.heatmap(matrix, annot=False, cmap='RdYlBu_r', cbar=True, vmin=0, vmax=args.vmax, mask=mask,
                xticklabels=x_labels, yticklabels=y_labels)
    plt.title(f'Step-wise Attention Score Heatmap (Layer {args.layer_id})')
    plt.xlabel('Previous Steps')
    plt.ylabel('Current Steps')
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签以避免重叠
    plt.yticks(rotation=0)  # y轴标签保持水平
    
    if args.show_plot:
        plt.show()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(f'{args.output_dir}/step_size_layer_{args.layer_id}.pdf', bbox_inches='tight', dpi=300)
    print(f"热力图已保存至: {args.output_dir}/layer_{args.layer_id}.pdf")
    plt.close()

if __name__ == "__main__":
    main()
    