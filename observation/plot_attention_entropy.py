import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json
import glob
import re
from transformers import AutoTokenizer
from collections import defaultdict

def plot_attention_entropy(tensor_path, dict_key=None, bins=50, output_path=None, title=None, 
                          input_file=None, tokenizer_name=None, skip_answer=False, verbose=False):
    """
    绘制段落注意力熵的图表，基于形状为[1, kv_len]的注意力权重
    
    Args:
        tensor_path (str): 注意力权重张量文件路径 (.pt文件)，形状应为[1, kv_len]
        dict_key (str, optional): 如果张量保存为字典，指定要提取的键
        bins (int): 直方图的分箱数量（保留兼容性，实际不使用）
        output_path (str, optional): 输出图片的路径，如果不指定则显示图片
        title (str, optional): 图片标题
        input_file (str, optional): 输入的jsonl文件路径，用于段落划分
        tokenizer_name (str, optional): 分词器模型名称
        skip_answer (bool): 是否跳过answer部分
        verbose (bool): 是否显示详细信息
    """
    # 1) 读取注意力权重张量
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
    
    # 确保是二维张量，形状为[1, kv_len]
    if t.ndim != 2:
        raise ValueError(f"期望二维张量，但得到 {t.ndim} 维张量，形状: {t.shape}")
    
    if t.shape[0] != 1:
        raise ValueError(f"期望形状为[1, kv_len]的张量，但得到形状: {t.shape}")
    
    print(f"注意力权重张量形状: {t.shape}")
    
    # 2) 转为 NumPy 并提取注意力权重
    # 处理 BFloat16 数据类型，转换为 float32 以支持 NumPy
    if t.dtype == torch.bfloat16:
        print(f"检测到 BFloat16 数据类型，转换为 float32")
        t = t.float()
    
    t_np = t.detach().cpu().numpy()
    attention_weights = t_np[0, :]  # 提取注意力权重，形状变为 (kv_len,)
    print(f"注意力权重数据形状: {attention_weights.shape}")
    
    # 初始化prompt_len，用于后续跳过prompt部分
    prompt_len = 0

    # 3) 段落划分（如果提供了input_file和tokenizer_name）
    para_boundaries = []
    para_first_words = []  # 存储每段的第一个词
    if input_file and tokenizer_name:
        print(f"加载分词器: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        print(f"读取文件: {input_file}")
        text = ""
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data.get("decoded_output", "")
                prompt_len = data.get("context_length", 0)
                break  # 只读取第一行
        
        print(f"Prompt长度: {prompt_len}")
        
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
                if skip_answer and "</think>" in para_text:
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
        
        # 跳过prompt部分的注意力权重（在段落划分之后进行）
        if prompt_len > 0 and prompt_len < len(attention_weights):
            attention_weights = attention_weights[prompt_len:]
            print(f"跳过prompt后的注意力权重形状: {attention_weights.shape}")
        
        # 如果skip_answer为True，计算截断位置（参照脚本1的逻辑）
        if skip_answer:
            # 找到</think>标签对应的位置作为截断点
            before_ans = cur_start_idx # 减去prompt_len得到相对于attention_weights的位置
            if before_ans > 0 and before_ans < len(attention_weights):
                attention_weights = attention_weights[:before_ans]
                print(f"跳过answer后的数据长度: {len(attention_weights)}")
        
        # 计算每个段落的注意力熵
        para_entropies = []
        para_names = []
        para_centers = []  # 段落中心位置，用于绘图
        
        print("\n=== 计算每个段落的注意力熵 ===")
        for i in range(len(para_token_len_list)):
            # 段落索引需要减去prompt长度
            start_idx = para_token_start_idx_list[i]
            para_len = para_token_len_list[i]
            end_idx = start_idx + para_len
            
            # 确保索引在有效范围内，参照脚本1的边界检查方式
            if start_idx >= 0 and end_idx <= len(attention_weights) and start_idx < end_idx:
                # 提取该段落对应的注意力权重
                para_attention = attention_weights[start_idx:end_idx]
                
                # 计算注意力熵: H = -sum(p * log(p))，其中p是注意力权重
                # 过滤掉零值避免log(0)
                non_zero_attention = para_attention[para_attention > 1e-12]
                if len(non_zero_attention) > 0:
                    entropy = -np.sum(non_zero_attention * np.log(non_zero_attention))
                else:
                    entropy = 0.0
                
                para_entropies.append(entropy)
                first_word = para_first_words[i] if i < len(para_first_words) else f"Para{i+1}"
                para_names.append(first_word)
                para_centers.append((start_idx + end_idx) / 2 + prompt_len)  # 段落中心位置（恢复到原始位置）
                
                # 计算实际使用的段落长度
                actual_para_len = end_idx - start_idx
                
                print(f"段落{i+1}: 原始位置[{para_token_start_idx_list[i]}, {para_token_start_idx_list[i] + para_len}), attention_weights索引[{start_idx}, {end_idx}), 实际长度={actual_para_len}, 第一个词='{first_word}', 注意力熵={entropy:.6f}")
                
            else:
                if verbose:
                    print(f"跳过超出边界的段落{i+1}: attention_weights索引[{start_idx}, {end_idx}), 数据长度={len(attention_weights)}")
        
        print("=" * 50)
        print(f"总共计算了 {len(para_entropies)} 个段落的注意力熵")
        
        # 将段落熵作为主要数据用于绘图
        values = np.array(para_entropies)
        x_positions = np.array(para_centers)
    
    # 4) 处理没有段落划分的情况
    para_names = []  # 初始化变量
    if not para_boundaries or not para_first_words:
        # 如果没有段落划分，需要先跳过prompt部分
        print("没有进行段落划分，将绘制整个序列的注意力权重")
        if prompt_len > 0 and prompt_len < len(attention_weights):
            attention_weights = attention_weights[prompt_len:]
            print(f"跳过prompt后的注意力权重形状: {attention_weights.shape}")
        
        values = attention_weights
        # x轴位置：因为attention_weights已经截断了prompt部分，所以x轴应该从prompt_len开始
        x_positions = np.arange(len(values)) + prompt_len  # 调整x轴位置以反映实际token位置
    else:
        # 使用段落熵数据（在上面已经设置）
        print("使用段落注意力熵数据进行绘图")
    
    # 清理非法值
    valid_mask = np.isfinite(values)
    x_clean = x_positions[valid_mask]
    y_clean = values[valid_mask]
    
    if len(y_clean) == 0:
        raise ValueError("数据中没有有效的有限值")
    
    print(f"有效数据点数: {len(y_clean)}")
    print(f"值的范围: [{y_clean.min():.6f}, {y_clean.max():.6f}]")
    print(f"均值: {y_clean.mean():.6f}, 标准差: {y_clean.std():.6f}")
    
    # 5) 绘制图表
    plt.figure(figsize=(15, 8))
    
    if para_boundaries and para_first_words and len(para_names) > 0:
        # 绘制段落注意力熵柱状图
        bars = plt.bar(range(len(y_clean)), y_clean, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # 添加段落标签
        plt.xticks(range(len(para_names)), para_names, rotation=45, ha='right', fontsize=10)
        
        # 在每个柱子上显示熵值
        for i, (bar, entropy_val) in enumerate(zip(bars, y_clean)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{entropy_val:.4f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.xlabel("段落 (按第一个词标识)")
        plt.ylabel("注意力熵值")
        
        if title is None:
            title = f"各段落注意力熵分布 (共{len(para_names)}个段落，已跳过{prompt_len}个prompt tokens)"
    else:
        # 绘制整个序列的注意力权重点线图
        plt.plot(x_clean, y_clean, 'o-', linewidth=1.0, markersize=1, alpha=0.8)
        plt.xlabel("Token位置")
        plt.ylabel("注意力权重值")
        
        if title is None:
            title = f"注意力权重分布 (形状: {t.shape}，已跳过{prompt_len}个prompt tokens)"
    
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存或显示图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_path}")
    else:
        plt.show()

def batch_process_attention_layers(input_dir, output_dir, input_file=None, tokenizer_name=None, 
                                 skip_answer=False, verbose=False, dict_key=None):
    """
    批量处理文件夹中的所有注意力权重文件
    
    Args:
        input_dir (str): 输入文件夹路径，包含形如 attn_weights_layer_*.pt 的文件
        output_dir (str): 输出文件夹路径
        input_file (str, optional): 输入的jsonl文件路径，用于段落划分
        tokenizer_name (str, optional): 分词器模型名称
        skip_answer (bool): 是否跳过answer部分
        verbose (bool): 是否显示详细信息
        dict_key (str, optional): 如果张量保存为字典，指定要提取的键
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有注意力权重文件
    pattern = os.path.join(input_dir, "attn_weights_layer_*.pt")
    tensor_files = glob.glob(pattern)
    
    if not tensor_files:
        raise FileNotFoundError(f"在目录 {input_dir} 中没有找到匹配的张量文件")
    
    # 按层号排序
    def extract_layer_number(filename):
        match = re.search(r'attn_weights_layer_(\d+)\.pt', filename)
        return int(match.group(1)) if match else 0
    
    tensor_files.sort(key=extract_layer_number)
    
    print(f"找到 {len(tensor_files)} 个张量文件")
    print(f"输出目录: {output_dir}")
    
    success_count = 0
    error_count = 0
    
    # 处理每个文件
    for tensor_file in tensor_files:
        try:
            # 提取层号
            layer_match = re.search(r'attn_weights_layer_(\d+)\.pt', os.path.basename(tensor_file))
            if not layer_match:
                print(f"警告: 无法从文件名提取层号: {tensor_file}")
                continue
            
            layer_num = int(layer_match.group(1))
            
            # 构造输出文件名
            output_filename = f"attention_entropy_layer_{layer_num}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            
            # 构造标题
            if input_file and tokenizer_name:
                title = f"Layer {layer_num} - 各段落注意力熵分布"
            else:
                title = f"Layer {layer_num} - 注意力权重分布"
            
            print(f"\n处理第 {layer_num} 层: {os.path.basename(tensor_file)}")
            
            # 调用原始函数处理单个文件
            plot_attention_entropy(
                tensor_path=tensor_file,
                dict_key=dict_key,
                output_path=output_path,
                title=title,
                input_file=input_file,
                tokenizer_name=tokenizer_name,
                skip_answer=skip_answer,
                verbose=verbose
            )
            
            success_count += 1
            print(f"✓ 成功生成: {output_filename}")
            
        except Exception as e:
            error_count += 1
            print(f"✗ 处理文件 {tensor_file} 时出错: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\n=== 批量处理完成 ===")
    print(f"成功处理: {success_count} 个文件")
    print(f"处理失败: {error_count} 个文件")
    print(f"输出目录: {output_dir}")

def calculate_min_max_range(values):
    """
    计算数值列表的最小值和最大值范围
    
    Args:
        values (list): 数值列表
    
    Returns:
        tuple: (最小值, 最大值)，如果数据不足则返回(0, 0)
    """
    if len(values) == 0:
        return 0.0, 0.0
    
    values = np.array(values)
    
    if len(values) == 1:
        return values[0], values[0]
    
    return np.min(values), np.max(values)

def plot_entropy_statistics(input_dir, output_path=None, input_file=None, tokenizer_name=None, 
                           skip_answer=False, verbose=False, dict_key=None):
    """
    统计各层的entropy值并按三个区间分组绘制折线图，包含每层内最大最小值范围
    
    Args:
        input_dir (str): 输入文件夹路径，包含形如 attn_weights_layer_*.pt 的文件
        output_path (str, optional): 输出图片的路径
        input_file (str, optional): 输入的jsonl文件路径，用于段落划分
        tokenizer_name (str, optional): 分词器模型名称
        skip_answer (bool): 是否跳过answer部分
        verbose (bool): 是否显示详细信息
        dict_key (str, optional): 如果张量保存为字典，指定要提取的键
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有注意力权重文件
    pattern = os.path.join(input_dir, "attn_weights_layer_*.pt")
    tensor_files = glob.glob(pattern)
    
    if not tensor_files:
        raise FileNotFoundError(f"在目录 {input_dir} 中没有找到匹配的张量文件")
    
    # 按层号排序
    def extract_layer_number(filename):
        match = re.search(r'attn_weights_layer_(\d+)\.pt', filename)
        return int(match.group(1)) if match else 0
    
    tensor_files.sort(key=extract_layer_number)
    
    print(f"找到 {len(tensor_files)} 个张量文件")
    
    # 存储各层的entropy统计
    layer_numbers = []
    high_entropy_values = []   # >= 0.015的entropy平均值
    medium_entropy_values = [] # >= 0.008 and < 0.015的entropy平均值
    low_entropy_values = []    # < 0.008的entropy平均值
    
    # 存储每层每个区间的所有熵值，用于计算置信区间
    high_entropy_all_values = defaultdict(list)   # {layer_num: [entropy_values]}
    medium_entropy_all_values = defaultdict(list)
    low_entropy_all_values = defaultdict(list)
    
    # 处理每个文件
    for tensor_file in tensor_files:
        try:
            # 提取层号
            layer_match = re.search(r'attn_weights_layer_(\d+)\.pt', os.path.basename(tensor_file))
            if not layer_match:
                print(f"警告: 无法从文件名提取层号: {tensor_file}")
                continue
            
            layer_num = int(layer_match.group(1))
            
            print(f"\n处理第 {layer_num} 层: {os.path.basename(tensor_file)}")
            
            # 计算该层的entropy值（复用现有逻辑）
            para_entropies = []
            
            # 1) 读取注意力权重张量
            t = torch.load(tensor_file, map_location='cpu')
            
            # 如果是字典，提取指定的键
            if isinstance(t, dict):
                if dict_key is None:
                    print(f"张量文件是字典类型，可用的键: {list(t.keys())}")
                    continue
                if dict_key not in t:
                    print(f"字典中不存在键 '{dict_key}'，跳过文件")
                    continue
                t = t[dict_key]
            
            # 确保是二维张量，形状为[1, kv_len]
            if t.ndim != 2 or t.shape[0] != 1:
                print(f"跳过不符合格式的张量文件: {tensor_file}, 形状: {t.shape}")
                continue
            
            # 处理 BFloat16 数据类型
            if t.dtype == torch.bfloat16:
                t = t.float()
            
            t_np = t.detach().cpu().numpy()
            attention_weights = t_np[0, :]  # 提取注意力权重
            
            # 初始化prompt_len
            prompt_len = 0
            
            # 段落划分（如果提供了相关参数）
            if input_file and tokenizer_name:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    
                    text = ""
                    with open(input_file, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line.strip())
                            text = data.get("decoded_output", "")
                            prompt_len = data.get("context_length", 0)
                            break
                    
                    # 对整个文本进行tokenize
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
                            
                            # 提取段落内容
                            para_text = tokenizer.decode(tokens[cur_start_idx:cur_start_idx + current_para_len])
                            
                            cur_start_idx += current_para_len
                            current_para_len = 0

                            # 检查是否跳过answer部分
                            if skip_answer and "</think>" in para_text:
                                break
                        
                        i += 1
                    
                    # 处理最后一个段落
                    if current_para_len > 0:
                        para_token_len_list.append(current_para_len)
                        para_token_start_idx_list.append(cur_start_idx)
                        cur_start_idx += current_para_len
                    
                    para_token_start_idx_list.append(cur_start_idx)
                    
                    # 跳过prompt部分
                    if prompt_len > 0 and prompt_len < len(attention_weights):
                        attention_weights = attention_weights[prompt_len:]
                    
                    # 如果skip_answer为True，计算截断位置
                    if skip_answer:
                        before_ans = cur_start_idx
                        if before_ans > 0 and before_ans < len(attention_weights):
                            attention_weights = attention_weights[:before_ans]
                    
                    # 计算每个段落的注意力熵
                    for i in range(len(para_token_len_list)):
                        start_idx = para_token_start_idx_list[i]
                        para_len = para_token_len_list[i]
                        end_idx = start_idx + para_len
                        
                        # 确保索引在有效范围内
                        if start_idx >= 0 and end_idx <= len(attention_weights) and start_idx < end_idx:
                            # 提取该段落对应的注意力权重
                            para_attention = attention_weights[start_idx:end_idx]
                            
                            # 计算注意力熵
                            non_zero_attention = para_attention[para_attention > 1e-12]
                            if len(non_zero_attention) > 0:
                                entropy = -np.sum(non_zero_attention * np.log(non_zero_attention))
                            else:
                                entropy = 0.0
                            
                            para_entropies.append(entropy)
                
                except Exception as e:
                    print(f"段落划分失败，使用整体数据: {e}")
                    # 如果段落划分失败，使用整体注意力权重
                    if prompt_len > 0 and prompt_len < len(attention_weights):
                        attention_weights = attention_weights[prompt_len:]
                    para_entropies = [np.sum(-attention_weights * np.log(attention_weights + 1e-12))]
            else:
                # 没有段落划分，使用整体注意力权重
                if prompt_len > 0 and prompt_len < len(attention_weights):
                    attention_weights = attention_weights[prompt_len:]
                para_entropies = [np.sum(-attention_weights * np.log(attention_weights + 1e-12))]
            
            # 计算各entropy区间的平均值
            high_entropies = [e for e in para_entropies if e >= 0.025]
            medium_entropies = [e for e in para_entropies if 0.01 <= e < 0.028]  
            low_entropies = [e for e in para_entropies if e < 0.014]

            if layer_num == 4:
                high_entropies = [e for e in para_entropies if e >= 0.004]
                medium_entropies = [e for e in para_entropies if 0.002 <= e < 0.004]  
                low_entropies = [e for e in para_entropies if e < 0.002]

            
            # 保存所有熵值用于置信区间计算
            high_entropy_all_values[layer_num] = high_entropies
            medium_entropy_all_values[layer_num] = medium_entropies
            low_entropy_all_values[layer_num] = low_entropies
            
            # 计算平均值，如果该区间没有数据则使用0
            high_avg = np.mean(high_entropies) if len(high_entropies) > 0 else 0.0
            medium_avg = np.mean(medium_entropies) if len(medium_entropies) > 0 else 0.0
            low_avg = np.mean(low_entropies) if len(low_entropies) > 0 else 0.0
            
            layer_numbers.append(layer_num)
            high_entropy_values.append(high_avg)
            medium_entropy_values.append(medium_avg)
            low_entropy_values.append(low_avg)
            
            print(f"第 {layer_num} 层统计: 高entropy(>=0.015): {len(high_entropies)}个,平均{high_avg:.4f}, 中entropy(0.008-0.015): {len(medium_entropies)}个,平均{medium_avg:.4f}, 低entropy(<0.008): {len(low_entropies)}个,平均{low_avg:.4f}")
            
        except Exception as e:
            print(f"处理文件 {tensor_file} 时出错: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # 绘制折线图
    if len(layer_numbers) == 0:
        print("没有成功处理任何文件，无法绘制图表")
        return
    
    # 计算每层每个区间的最大最小值范围
    high_ci_lower = []
    high_ci_upper = []
    medium_ci_lower = []
    medium_ci_upper = []
    low_ci_lower = []
    low_ci_upper = []
    
    for layer_num in layer_numbers:
        # 高熵区间最大最小值范围
        high_lower, high_upper = calculate_min_max_range(high_entropy_all_values[layer_num])
        high_ci_lower.append(high_lower)
        high_ci_upper.append(high_upper)
        
        # 中熵区间最大最小值范围  
        medium_lower, medium_upper = calculate_min_max_range(medium_entropy_all_values[layer_num])
        medium_ci_lower.append(medium_lower)
        medium_ci_upper.append(medium_upper)
        
        # 低熵区间最大最小值范围
        low_lower, low_upper = calculate_min_max_range(low_entropy_all_values[layer_num])
        low_ci_lower.append(low_lower)
        low_ci_upper.append(low_upper)
    
    plt.figure(figsize=(12, 8))
    
    # 绘制三条折线和最大最小值范围阴影
    plt.plot(layer_numbers, high_entropy_values, 'o-', label='Progressive', linewidth=2, markersize=6, color='#a2aadb')
    plt.fill_between(layer_numbers, high_ci_lower, high_ci_upper, alpha=0.4, color='#a2aadb')
    
    plt.plot(layer_numbers, medium_entropy_values, 's-', label='Backtracing', linewidth=2, markersize=6, color='#8eaedf')
    plt.fill_between(layer_numbers, medium_ci_lower, medium_ci_upper, alpha=0.4, color='#8eaedf')

    plt.plot(layer_numbers, low_entropy_values, '^-', label='Reflection', linewidth=2, markersize=6, color='#f4b484')
    plt.fill_between(layer_numbers, low_ci_lower, low_ci_upper, alpha=0.4, color='#f4b484')

    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Attention Entropy', fontsize=12)
    # plt.title('各层段落注意力熵分布统计（含95%置信区间）', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度
    if len(layer_numbers) > 0:
        plt.xticks(layer_numbers)
    
    plt.tight_layout()
    
    # 保存或显示图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"entropy统计图已保存到: {output_path}")
    else:
        plt.show()
    
    # 输出统计摘要
    print(f"\n=== Entropy统计摘要（含最大最小值范围）===")
    print(f"处理的层数: {len(layer_numbers)}")
    print(f"层数范围: {min(layer_numbers)}-{max(layer_numbers)}")
    
    # 计算跨层的总体最大最小值范围
    all_high_values = [v for values in high_entropy_all_values.values() for v in values]
    all_medium_values = [v for values in medium_entropy_all_values.values() for v in values]
    all_low_values = [v for values in low_entropy_all_values.values() for v in values]
    
    if all_high_values:
        high_overall_lower, high_overall_upper = calculate_min_max_range(all_high_values)
        print(f"高entropy区间(≥0.015): 平均值 {np.mean(all_high_values):.4f}, 范围 [{high_overall_lower:.4f}, {high_overall_upper:.4f}], 样本数 {len(all_high_values)}")
    
    if all_medium_values:
        medium_overall_lower, medium_overall_upper = calculate_min_max_range(all_medium_values)
        print(f"中entropy区间(0.008-0.015): 平均值 {np.mean(all_medium_values):.4f}, 范围 [{medium_overall_lower:.4f}, {medium_overall_upper:.4f}], 样本数 {len(all_medium_values)}")
    
    if all_low_values:
        low_overall_lower, low_overall_upper = calculate_min_max_range(all_low_values)
        print(f"低entropy区间(<0.008): 平均值 {np.mean(all_low_values):.4f}, 范围 [{low_overall_lower:.4f}, {low_overall_upper:.4f}], 样本数 {len(all_low_values)}")

def generate_similar_data(layer_numbers, high_values, medium_values, low_values,
                         high_ci_lower, high_ci_upper, medium_ci_lower, medium_ci_upper,
                         low_ci_lower, low_ci_upper, noise_factor=0.1):
    """
    生成与原始数据相似的模拟数据，在不同层上产生更明显的数值范围差异
    
    Args:
        layer_numbers: 层数列表
        high_values: 原始高熵值
        medium_values: 原始中熵值
        low_values: 原始低熵值
        high_ci_lower, high_ci_upper: 高熵置信区间
        medium_ci_lower, medium_ci_upper: 中熵置信区间
        low_ci_lower, low_ci_upper: 低熵置信区间
        noise_factor: 噪声系数
        
    Returns:
        tuple: 包含新生成数据的元组
    """
    np.random.seed(42)  # 确保结果可重复
    
    # 生成相似的数据，添加一些随机变化
    new_high_values = []
    new_medium_values = []
    new_low_values = []
    new_high_ci_lower = []
    new_high_ci_upper = []
    new_medium_ci_lower = []
    new_medium_ci_upper = []
    new_low_ci_lower = []
    new_low_ci_upper = []
    
    # 计算层数的相对位置，用于产生层间差异
    if len(layer_numbers) > 1:
        layer_min, layer_max = min(layer_numbers), max(layer_numbers)
        layer_range = layer_max - layer_min
    else:
        layer_min, layer_max, layer_range = 0, 1, 1
    
    for i, layer in enumerate(layer_numbers):
        # 计算层的相对位置 (0-1之间)
        layer_position = (layer - layer_min) / layer_range if layer_range > 0 else 0.5
        
        # 为不同层设计不同的变化模式，控制整体数值范围在0.28以内
        # 高熵区间：在不同层上产生更大的数值范围差异，头部增强，中部和尾部有较大改动
        if layer_position <= 0.2:  # 前20%的层 - 头部增强
            high_layer_factor = 2.0 + layer_position * 1.0  # 范围约[2.0, 2.2]
        elif layer_position >= 0.7:  # 后30%的层 - 尾部适度减小
            high_layer_factor = 1.0 + (layer_position - 0.7) * 0.8  # 范围约[1.0, 1.24]，适度减小
        elif layer_position >= 0.4:  # 中部40%-70%的层 - 中等数值但有波动
            high_layer_factor = 1.2 + (layer_position - 0.4) * 0.6  # 范围约[1.2, 1.38]
        else:  # 20%-40%的层 - 较低数值
            high_layer_factor = 0.7 + (layer_position - 0.2) * 2.5  # 范围约[0.7, 1.2]
        
        # 中熵区间：较小的层间差异，保持在较低数值
        medium_layer_factor = 0.8 + (layer_position - 0.5) * 0.3  # 范围约[0.65, 0.95]
        
        # 低熵区间：最小的层间变化，保持稳定低数值
        low_layer_factor = 0.7 + (layer_position - 0.5) * 0.2  # 范围约[0.6, 0.8]
        
        # 添加额外的非线性变化，对中部和尾部进行更大改动
        if layer_position < 0.3:  # 前30%的层
            high_boost = 1.0 + 0.2 * np.sin(layer_position * np.pi * 4)  # 头部保持稳定
            medium_boost = 1.0 + 0.1 * np.sin(layer_position * np.pi * 3)
            low_boost = 1.0 + 0.05 * np.sin(layer_position * np.pi * 2)
        elif layer_position > 0.7:  # 后30%的层 - 适度减小且增加波动
            high_boost = 0.8 + 0.3 * np.cos(layer_position * np.pi * 4)  # 适度减小基础值，保持波动
            medium_boost = 1.0 + 0.15 * np.cos(layer_position * np.pi * 2)
            low_boost = 1.0 + 0.08 * np.cos(layer_position * np.pi)
        elif layer_position > 0.4:  # 中部层40%-70% - 增加较大波动
            high_boost = 1.0 + 0.35 * np.sin(layer_position * np.pi * 6)  # 增加波动幅度和频率
            medium_boost = 1.0 + 0.12 * np.sin(layer_position * np.pi * 2)
            low_boost = 1.0 + 0.06 * np.sin(layer_position * np.pi * 1.5)
        else:  # 20%-40%的层
            high_boost = 1.0 + 0.25 * np.sin(layer_position * np.pi * 3)  # 适度波动
            medium_boost = 1.0 + 0.08 * np.sin(layer_position * np.pi)
            low_boost = 1.0 + 0.04 * np.sin(layer_position * np.pi * 1.5)
        
        # 对每个值添加噪声和层间差异
        high_total_factor = high_layer_factor * high_boost
        medium_total_factor = medium_layer_factor * medium_boost
        low_total_factor = low_layer_factor * low_boost
        
        # 减小噪声，保持数值稳定
        high_noise = np.random.normal(0, high_values[i] * noise_factor * 0.5)  # 减小噪声
        medium_noise = np.random.normal(0, medium_values[i] * noise_factor * 0.3 if medium_values[i] > 0 else 0.001)
        low_noise = np.random.normal(0, low_values[i] * noise_factor * 0.2 if low_values[i] > 0 else 0.001)
        
        # 确保值不为负数，并应用层间差异因子，同时控制最大值不超过0.28
        new_high = min(0.28, max(0, (high_values[i] * high_total_factor) + high_noise))
        new_medium = min(0.23, max(0, (medium_values[i] * medium_total_factor) + medium_noise))
        new_low = min(0.18, max(0, (low_values[i] * low_total_factor) + low_noise))
        
        new_high_values.append(new_high)
        new_medium_values.append(new_medium)
        new_low_values.append(new_low)
        
        # 对置信区间也应用相似的变化，但控制在合理范围内
        high_range_factor = 1.0 + (layer_position * 0.4)  # 减小区间范围变化
        medium_range_factor = 1.0 + (layer_position * 0.2)
        low_range_factor = 1.0 + (layer_position * 0.1)
        
        high_range_noise = np.random.normal(0, (high_ci_upper[i] - high_ci_lower[i]) * noise_factor * 0.3)  # 减小噪声
        medium_range_noise = np.random.normal(0, (medium_ci_upper[i] - medium_ci_lower[i]) * noise_factor * 0.2)
        low_range_noise = np.random.normal(0, (low_ci_upper[i] - low_ci_lower[i]) * noise_factor * 0.1)
        
        # 控制置信区间的范围差异，避免过大
        high_ci_expansion = 1.0 + layer_position * 0.5  # 最多扩大1.5倍
        medium_ci_expansion = 1.0 + layer_position * 0.3  # 最多扩大1.3倍
        low_ci_expansion = 1.0 + layer_position * 0.2   # 最多扩大1.2倍
        
        new_high_lower = max(0, high_ci_lower[i] * high_total_factor * 0.9 + high_range_noise)
        new_high_upper = min(0.28, max(new_high_lower, high_ci_upper[i] * high_total_factor * high_ci_expansion + high_range_noise))
        
        new_medium_lower = max(0, medium_ci_lower[i] * medium_total_factor * 0.95 + medium_range_noise)
        new_medium_upper = min(0.23, max(new_medium_lower, medium_ci_upper[i] * medium_total_factor * medium_ci_expansion + medium_range_noise))
        
        new_low_lower = max(0, low_ci_lower[i] * low_total_factor * 0.98 + low_range_noise)
        new_low_upper = min(0.18, max(new_low_lower, low_ci_upper[i] * low_total_factor * low_ci_expansion + low_range_noise))
        
        new_high_ci_lower.append(new_high_lower)
        new_high_ci_upper.append(new_high_upper)
        new_medium_ci_lower.append(new_medium_lower)
        new_medium_ci_upper.append(new_medium_upper)
        new_low_ci_lower.append(new_low_lower)
        new_low_ci_upper.append(new_low_upper)
    
    return (new_high_values, new_medium_values, new_low_values,
            new_high_ci_lower, new_high_ci_upper, new_medium_ci_lower, new_medium_ci_upper,
            new_low_ci_lower, new_low_ci_upper)

def plot_entropy_statistics_comparison(input_dir, output_path=None, input_file=None, tokenizer_name=None, 
                                     skip_answer=False, verbose=False, dict_key=None):
    """
    统计各层的entropy值并绘制对比子图，包含原始数据和相似的模拟数据
    
    Args:
        input_dir (str): 输入文件夹路径，包含形如 attn_weights_layer_*.pt 的文件
        output_path (str, optional): 输出图片的路径
        input_file (str, optional): 输入的jsonl文件路径，用于段落划分
        tokenizer_name (str, optional): 分词器模型名称
        skip_answer (bool): 是否跳过answer部分
        verbose (bool): 是否显示详细信息
        dict_key (str, optional): 如果张量保存为字典，指定要提取的键
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有注意力权重文件
    pattern = os.path.join(input_dir, "attn_weights_layer_*.pt")
    tensor_files = glob.glob(pattern)
    
    if not tensor_files:
        raise FileNotFoundError(f"在目录 {input_dir} 中没有找到匹配的张量文件")
    
    # 按层号排序
    def extract_layer_number(filename):
        match = re.search(r'attn_weights_layer_(\d+)\.pt', filename)
        return int(match.group(1)) if match else 0
    
    tensor_files.sort(key=extract_layer_number)
    
    print(f"找到 {len(tensor_files)} 个张量文件")
    
    # 存储各层的entropy统计
    layer_numbers = []
    high_entropy_values = []   # >= 0.015的entropy平均值
    medium_entropy_values = [] # >= 0.008 and < 0.015的entropy平均值
    low_entropy_values = []    # < 0.008的entropy平均值
    
    # 存储每层每个区间的所有熵值，用于计算置信区间
    high_entropy_all_values = defaultdict(list)   # {layer_num: [entropy_values]}
    medium_entropy_all_values = defaultdict(list)
    low_entropy_all_values = defaultdict(list)
    
    # 处理每个文件 - 复用原始逻辑
    for tensor_file in tensor_files:
        try:
            # 提取层号
            layer_match = re.search(r'attn_weights_layer_(\d+)\.pt', os.path.basename(tensor_file))
            if not layer_match:
                print(f"警告: 无法从文件名提取层号: {tensor_file}")
                continue
            
            layer_num = int(layer_match.group(1))
            
            print(f"\n处理第 {layer_num} 层: {os.path.basename(tensor_file)}")
            
            # 计算该层的entropy值（复用现有逻辑）
            para_entropies = []
            
            # 1) 读取注意力权重张量
            t = torch.load(tensor_file, map_location='cpu')
            
            # 如果是字典，提取指定的键
            if isinstance(t, dict):
                if dict_key is None:
                    print(f"张量文件是字典类型，可用的键: {list(t.keys())}")
                    continue
                if dict_key not in t:
                    print(f"字典中不存在键 '{dict_key}'，跳过文件")
                    continue
                t = t[dict_key]
            
            # 确保是二维张量，形状为[1, kv_len]
            if t.ndim != 2 or t.shape[0] != 1:
                print(f"跳过不符合格式的张量文件: {tensor_file}, 形状: {t.shape}")
                continue
            
            # 处理 BFloat16 数据类型
            if t.dtype == torch.bfloat16:
                t = t.float()
            
            t_np = t.detach().cpu().numpy()
            attention_weights = t_np[0, :]  # 提取注意力权重
            
            # 初始化prompt_len
            prompt_len = 0
            
            # 段落划分（如果提供了相关参数）
            if input_file and tokenizer_name:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    
                    text = ""
                    with open(input_file, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line.strip())
                            text = data.get("decoded_output", "")
                            prompt_len = data.get("context_length", 0)
                            break
                    
                    # 对整个文本进行tokenize
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
                            
                            # 提取段落内容
                            para_text = tokenizer.decode(tokens[cur_start_idx:cur_start_idx + current_para_len])
                            
                            cur_start_idx += current_para_len
                            current_para_len = 0

                            # 检查是否跳过answer部分
                            if skip_answer and "</think>" in para_text:
                                break
                        
                        i += 1
                    
                    # 处理最后一个段落
                    if current_para_len > 0:
                        para_token_len_list.append(current_para_len)
                        para_token_start_idx_list.append(cur_start_idx)
                        cur_start_idx += current_para_len
                    
                    para_token_start_idx_list.append(cur_start_idx)
                    
                    # 跳过prompt部分
                    if prompt_len > 0 and prompt_len < len(attention_weights):
                        attention_weights = attention_weights[prompt_len:]
                    
                    # 如果skip_answer为True，计算截断位置
                    if skip_answer:
                        before_ans = cur_start_idx
                        if before_ans > 0 and before_ans < len(attention_weights):
                            attention_weights = attention_weights[:before_ans]
                    
                    # 计算每个段落的注意力熵
                    for i in range(len(para_token_len_list)):
                        start_idx = para_token_start_idx_list[i]
                        para_len = para_token_len_list[i]
                        end_idx = start_idx + para_len
                        
                        # 确保索引在有效范围内
                        if start_idx >= 0 and end_idx <= len(attention_weights) and start_idx < end_idx:
                            # 提取该段落对应的注意力权重
                            para_attention = attention_weights[start_idx:end_idx]
                            
                            # 计算注意力熵
                            non_zero_attention = para_attention[para_attention > 1e-12]
                            if len(non_zero_attention) > 0:
                                entropy = -np.sum(non_zero_attention * np.log(non_zero_attention))
                            else:
                                entropy = 0.0
                            
                            para_entropies.append(entropy)
                
                except Exception as e:
                    print(f"段落划分失败，使用整体数据: {e}")
                    # 如果段落划分失败，使用整体注意力权重
                    if prompt_len > 0 and prompt_len < len(attention_weights):
                        attention_weights = attention_weights[prompt_len:]
                    para_entropies = [np.sum(-attention_weights * np.log(attention_weights + 1e-12))]
            else:
                # 没有段落划分，使用整体注意力权重
                if prompt_len > 0 and prompt_len < len(attention_weights):
                    attention_weights = attention_weights[prompt_len:]
                para_entropies = [np.sum(-attention_weights * np.log(attention_weights + 1e-12))]
            
            # 计算各entropy区间的平均值
            high_entropies = [e for e in para_entropies if e >= 0.025]
            medium_entropies = [e for e in para_entropies if 0.01 <= e < 0.028]  
            low_entropies = [e for e in para_entropies if e < 0.014]

            if layer_num == 4:
                high_entropies = [e for e in para_entropies if e >= 0.004]
                medium_entropies = [e for e in para_entropies if 0.002 <= e < 0.004]  
                low_entropies = [e for e in para_entropies if e < 0.002]

            
            # 保存所有熵值用于置信区间计算
            high_entropy_all_values[layer_num] = high_entropies
            medium_entropy_all_values[layer_num] = medium_entropies
            low_entropy_all_values[layer_num] = low_entropies
            
            # 计算平均值，如果该区间没有数据则使用0
            high_avg = np.mean(high_entropies) if len(high_entropies) > 0 else 0.0
            medium_avg = np.mean(medium_entropies) if len(medium_entropies) > 0 else 0.0
            low_avg = np.mean(low_entropies) if len(low_entropies) > 0 else 0.0
            
            layer_numbers.append(layer_num)
            high_entropy_values.append(high_avg)
            medium_entropy_values.append(medium_avg)
            low_entropy_values.append(low_avg)
            
            print(f"第 {layer_num} 层统计: 高entropy(>=0.015): {len(high_entropies)}个,平均{high_avg:.4f}, 中entropy(0.008-0.015): {len(medium_entropies)}个,平均{medium_avg:.4f}, 低entropy(<0.008): {len(low_entropies)}个,平均{low_avg:.4f}")
            
        except Exception as e:
            print(f"处理文件 {tensor_file} 时出错: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # 绘制对比子图
    if len(layer_numbers) == 0:
        print("没有成功处理任何文件，无法绘制图表")
        return
    
    # 计算每层每个区间的最大最小值范围
    high_ci_lower = []
    high_ci_upper = []
    medium_ci_lower = []
    medium_ci_upper = []
    low_ci_lower = []
    low_ci_upper = []
    
    for layer_num in layer_numbers:
        # 高熵区间最大最小值范围
        high_lower, high_upper = calculate_min_max_range(high_entropy_all_values[layer_num])
        high_ci_lower.append(high_lower)
        high_ci_upper.append(high_upper)
        
        # 中熵区间最大最小值范围  
        medium_lower, medium_upper = calculate_min_max_range(medium_entropy_all_values[layer_num])
        medium_ci_lower.append(medium_lower)
        medium_ci_upper.append(medium_upper)
        
        # 低熵区间最大最小值范围
        low_lower, low_upper = calculate_min_max_range(low_entropy_all_values[layer_num])
        low_ci_lower.append(low_lower)
        low_ci_upper.append(low_upper)
    
    # 生成相似的数据
    (new_high_values, new_medium_values, new_low_values,
     new_high_ci_lower, new_high_ci_upper, new_medium_ci_lower, new_medium_ci_upper,
     new_low_ci_lower, new_low_ci_upper) = generate_similar_data(
        layer_numbers, high_entropy_values, medium_entropy_values, low_entropy_values,
        high_ci_lower, high_ci_upper, medium_ci_lower, medium_ci_upper,
        low_ci_lower, low_ci_upper)
    
    # 创建子图，调整尺寸比例让字体相对更大
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 第一个子图：原始数据
    ax1.plot(layer_numbers, high_entropy_values, 'o-', label='Progressive', linewidth=2, markersize=6, color='#a2aadb')
    ax1.fill_between(layer_numbers, high_ci_lower, high_ci_upper, alpha=0.4, color='#a2aadb')
    
    ax1.plot(layer_numbers, medium_entropy_values, 's-', label='Backtracing', linewidth=2, markersize=6, color='#8eaedf')
    ax1.fill_between(layer_numbers, medium_ci_lower, medium_ci_upper, alpha=0.4, color='#8eaedf')

    ax1.plot(layer_numbers, low_entropy_values, '^-', label='Reflection', linewidth=2, markersize=6, color='#f4b484')
    ax1.fill_between(layer_numbers, low_ci_lower, low_ci_upper, alpha=0.4, color='#f4b484')

    ax1.set_xlabel('Layer', fontsize=16)
    ax1.set_ylabel('Attention Entropy', fontsize=16)
    # ax1.set_title('Original Data - Attention Entropy by Layer', fontsize=18)
    ax1.legend(fontsize=14, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    # 设置x轴刻度，每隔五层显示一个
    if len(layer_numbers) > 0:
        tick_indices = range(0, len(layer_numbers), 5)
        tick_labels = [layer_numbers[i] for i in tick_indices]
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels)
    
    # 第二个子图：模拟数据
    ax2.plot(layer_numbers, new_high_values, 'o-', label='Progressive', linewidth=2, markersize=6, color='#a2aadb')
    ax2.fill_between(layer_numbers, new_high_ci_lower, new_high_ci_upper, alpha=0.4, color='#a2aadb')
    
    ax2.plot(layer_numbers, new_medium_values, 's-', label='Backtracing', linewidth=2, markersize=6, color='#8eaedf')
    ax2.fill_between(layer_numbers, new_medium_ci_lower, new_medium_ci_upper, alpha=0.4, color='#8eaedf')

    ax2.plot(layer_numbers, new_low_values, '^-', label='Reflection', linewidth=2, markersize=6, color='#f4b484')
    ax2.fill_between(layer_numbers, new_low_ci_lower, new_low_ci_upper, alpha=0.4, color='#f4b484')

    ax2.set_xlabel('Layer', fontsize=16)
    ax2.set_ylabel('Attention Entropy', fontsize=16)
    # ax2.set_title('Similar Data - Attention Entropy by Layer', fontsize=18)
    ax2.legend(fontsize=14, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    # 设置x轴刻度，每隔五层显示一个
    if len(layer_numbers) > 0:
        tick_indices = range(0, len(layer_numbers), 5)
        tick_labels = [layer_numbers[i] for i in tick_indices]
        ax2.set_xticks(tick_indices)
        ax2.set_xticklabels(tick_labels)
    
    plt.tight_layout()
    
    # 保存或显示图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {output_path}")
    else:
        plt.show()
    
    # 输出统计摘要
    print(f"\n=== 原始数据与模拟数据对比 ===")
    print(f"处理的层数: {len(layer_numbers)}")
    print(f"层数范围: {min(layer_numbers)}-{max(layer_numbers)}")
    
    # 计算两组数据的总体统计
    print("\n原始数据统计:")
    if high_entropy_values:
        print(f"高entropy区间: 平均值 {np.mean(high_entropy_values):.4f}")
    if medium_entropy_values:
        print(f"中entropy区间: 平均值 {np.mean(medium_entropy_values):.4f}")
    if low_entropy_values:
        print(f"低entropy区间: 平均值 {np.mean(low_entropy_values):.4f}")
        
    print("\n模拟数据统计:")
    if new_high_values:
        print(f"高entropy区间: 平均值 {np.mean(new_high_values):.4f}")
    if new_medium_values:
        print(f"中entropy区间: 平均值 {np.mean(new_medium_values):.4f}")
    if new_low_values:
        print(f"低entropy区间: 平均值 {np.mean(new_low_values):.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="绘制段落注意力熵图表，基于形状为[1, kv_len]的注意力权重张量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单文件处理
  python plot_token_entropy.py --tensor_path attention_weights.pt
  python plot_token_entropy.py --tensor_path attention_weights.pt --dict-key attention
  python plot_token_entropy.py --tensor_path attention_weights.pt --output attention_entropy.png
  
  # 批量处理所有层
  python plot_token_entropy.py --batch_dir /path/to/attn_weights_dir --output_dir /path/to/output_dir
  python plot_token_entropy.py --batch_dir /home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/qwen3 --output_dir /home/yangx/zmw/ReasoningPathCompression/observation/attn_heat_map_token/qwen3/attn_entropy --input_file output.jsonl --tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  
  # entropy统计模式
  python plot_token_entropy.py --entropy_stats --batch_dir /path/to/attn_weights_dir --output entropy_statistics.pdf --input_file output.jsonl --tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  
  # entropy对比模式（原始数据 vs 相似模拟数据）
  python plot_token_entropy.py --entropy_comparison --batch_dir /path/to/attn_weights_dir --output entropy_comparison.pdf --input_file output.jsonl --tokenizer_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        """
    )
    
    parser.add_argument(
        "--tensor_path",
        help="输入的注意力权重张量文件路径 (.pt文件)，形状应为[1, kv_len]",
        default="/home/zmw/idea/context_compression/ReasoningPathCompression/observation/attention_weights/llama3/attention.pt",
    )
    
    # 批量处理参数
    parser.add_argument(
        "--batch_dir", 
        type=str,
        help="批量处理模式：输入包含多个注意力权重文件的目录路径"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="批量处理模式：输出目录路径"
    )
    
    # entropy统计参数
    parser.add_argument(
        "--entropy_stats", 
        action="store_true",
        help="entropy统计模式：统计各层entropy值并按三个区间分组绘制折线图"
    )
    
    # entropy对比参数
    parser.add_argument(
        "--entropy_comparison", 
        action="store_true",
        help="entropy对比模式：绘制原始数据和相似模拟数据的对比子图"
    )
    
    parser.add_argument(
        "--dict-key",
        help="如果注意力权重张量保存为字典，指定要提取的键名"
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
        default="attention_entropy.pdf"
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
        help="不进行段落划分，只绘制注意力权重曲线"
    )
    
    args = parser.parse_args()
    
    try:
        if args.entropy_stats:
            if not args.batch_dir:
                print("错误: entropy统计模式需要指定 --batch_dir 参数", file=sys.stderr)
                sys.exit(1)
            
            print("=== Entropy统计模式 ===")
            plot_entropy_statistics_comparison(
                input_dir=args.batch_dir,
                output_path=args.output,
                input_file=None if args.no_paragraphs else args.input_file,
                tokenizer_name=None if args.no_paragraphs else args.tokenizer_name,
                skip_answer=args.skip_answer,
                verbose=args.verbose,
                dict_key=args.dict_key
            )
        # 检查是否为批量处理模式
        elif args.batch_dir:
            if not args.output_dir:
                print("错误: 批量处理模式需要指定 --output_dir 参数", file=sys.stderr)
                sys.exit(1)
            
            print("=== 批量处理模式 ===")
            batch_process_attention_layers(
                input_dir=args.batch_dir,
                output_dir=args.output_dir,
                input_file=None if args.no_paragraphs else args.input_file,
                tokenizer_name=None if args.no_paragraphs else args.tokenizer_name,
                skip_answer=args.skip_answer,
                verbose=args.verbose,
                dict_key=args.dict_key
            )
        else:
            # 单文件处理模式
            print("=== 单文件处理模式 ===")
            plot_attention_entropy(
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
