import os
import torch
from torch.nn import CrossEntropyLoss
import argparse
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_from_disk
import warnings
import json

from rpc.llama_profile import LlamaModel_use_attention_matrix_grad_log
from rpc.qwen2_profile import Qwen2Model_use_attention_matrix_grad_log

from rpc.llama_config import LlamaConfig
from rpc.llama_vanilla import LlamaForCausalLM

from rpc.qwen2_config import Qwen2Config
from rpc.qwen2_vanilla import Qwen2ForCausalLM

from eval.generate_answers.utils_hf import BBH_INSTRUCTIONS

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def Grad_Collect(model, tokenizer, args, data=None, task=None):
    model = model.eval()

    for name, param in model.named_parameters():
        if param.dim() < 2:
            param.requires_grad = False
        if 'lm_head' in name or 'embed' in name:
            param.requires_grad = False

    attention_record_keys = ['sum_effect']

    grad_W_dict = {}

    print(f"Total number of data samples: {len(data)}")

    data_select_range = args.data_range
    if data_select_range is None:
        data_select_range = [0, len(data)]
    elif len(data_select_range) == 1:
        data_select_range = [0, data_select_range[0]]
    elif len(data_select_range) > 2:
        raise ValueError("data_range should be a list of 0, 1 or 2 elements")
    
    if data_select_range[1] > len(data):
        warnings.warn("data_range[1] {} is larger than the length of the dataset, set to the length of the dataset {}".format(data_select_range[1], len(data)))
        data_select_range[1] = len(data)

    num_data = data_select_range[1] - data_select_range[0]
    
    pbar = tqdm(total=num_data, desc="Collecting Grad")
    for i, data_sample in enumerate(data):
        if i >= data_select_range[1] or i < data_select_range[0]:
            break
        
        CoT = data_sample['gen'][0].find("</think>")

        CoT = CoT + len("</think>")
        ans = len(data_sample['gen'][0]) - CoT

        if task == "gpqa" or task == "gsm8k":
            Ques = tokenizer(data_sample["question"], return_tensors='pt')['input_ids'].size(-1)
            AnS = tokenizer(data_sample["gen"][0][-ans:], return_tensors='pt')['input_ids'].size(-1)
            data_sample = data_sample["question"] + data_sample["gen"][0]
        elif task == "aime" or task == "math500":
            Ques = tokenizer(data_sample["problem"], return_tensors='pt')['input_ids'].size(-1)
            AnS = tokenizer(data_sample["gen"][0][-ans:], return_tensors='pt')['input_ids'].size(-1)
            data_sample = data_sample["problem"] + data_sample["gen"][0]
        elif task == "bbh":
            question = BBH_INSTRUCTIONS[args.bbh_subset] + "\nQ: " + data_sample['input'] + "\nA: Let's think step by step."
            Ques = tokenizer(question, return_tensors='pt')['input_ids'].size(-1)
            AnS = tokenizer(data_sample["gen"][0][-ans:], return_tensors='pt')['input_ids'].size(-1)
            data_sample = question + data_sample["gen"][0]

        tokenizer_output = tokenizer(data_sample, return_tensors='pt')
        data_sample = tokenizer_output['input_ids']
        data_mask = (tokenizer_output['attention_mask']==1)

        if "1.5" in args.model_name.lower() and data_sample.size(-1) >= 8000:
            continue
        elif "7" in args.model_name.lower() and data_sample.size(-1) >= 4000:
            continue
        elif "8" in args.model_name.lower() and data_sample.size(-1) >= 4000:
            continue
            
        print(f"Current Length: {data_sample.size(-1)}!!!!!!!!!!")

        logits = model(data_sample.to(next(model.parameters()).device))[0] # cross entropy loss
        labels = data_sample.to(next(model.parameters()).device)
        data_mask = data_mask.to(next(model.parameters()).device)

        # shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_data_mask = data_mask[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_data_mask = shift_data_mask.view(-1)

        gradient_mask = torch.zeros(shift_labels.size(0), dtype=torch.bool, device=shift_labels.device)
        # use args.gradient_range to select the gradient range
        keep_index = [i for i in range(shift_labels.size(0))][Ques:-AnS] # only CoT
        gradient_mask[keep_index] = True

        gradient_mask = shift_data_mask & gradient_mask

        filtered_logits = shift_logits[gradient_mask]
        filtered_labels = shift_labels[gradient_mask]

        filtered_labels = filtered_labels.to(filtered_logits.device)

        loss = loss_fct(filtered_logits, filtered_labels)

        if args.loss_type == 'ppl':
            loss = torch.exp(loss) # use ppl as loss
        elif args.loss_type == 'cross_entropy':
            pass
        else:
            raise NotImplementedError

        loss.backward()

        # store weight grad on cpu
        if args.weight_gradient:
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    if not m.weight.requires_grad:
                        continue
                    grad_W = m.weight.grad.detach()
                    if args.gradient_abs:
                        if name in grad_W_dict:
                            grad_W_dict[name + '.weight'] += grad_W.abs().cpu()
                        else:
                            grad_W_dict[name + '.weight'] = grad_W.abs().cpu()
                    else:
                        if name in grad_W_dict:
                            grad_W_dict[name + '.weight'] += grad_W.cpu()
                        else:
                            grad_W_dict[name + '.weight'] = grad_W.cpu()

        # store attention grad on cpu
        log = model.model.get_attention_matrix_log(take_abs=True, aggregating_block_size=args.aggregating_block_size, ques_len=Ques, ans_len=AnS)
        for key in attention_record_keys:
            if args.gradient_abs:
                if key in model.model.attention_matrix_log:
                    model.model.attention_matrix_log[key][0] = torch.cat([model.model.attention_matrix_log[key][0], torch.abs(log[key])], dim=1)
                else:
                    # called the first time
                    model.model.attention_matrix_log[key].append(torch.abs(log[key]))
                    print("the shape of the logged {} is {}".format(key, model.model.attention_matrix_log[key][0].shape))
            else:
                if key in model.model.attention_matrix_log:
                    model.model.attention_matrix_log[key][0] = torch.cat([model.model.attention_matrix_log[key][0], log[key]], dim=1)
                else:
                    # called the first time
                    model.model.attention_matrix_log[key].append(log[key])
                    print("the shape of the logged {} is {}".format(key, model.model.attention_matrix_log[key][0].shape))

        pbar.update(1)
    
    pbar.close()

    key = 'sum_effect'
    # key = 'grad'
    grad_Attn_tensor = torch.mean(model.model.attention_matrix_log[key][0], dim=1)

    for key, value in grad_W_dict.items():
        grad_W_dict[key] = value / num_data

    return grad_W_dict, grad_Attn_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='lmsys/vicuna-13b-v1.5-16k', help='model name')
parser.add_argument('--tokenizer_name', type=str, default=None)
parser.add_argument('--grad_dir', type=str, help='directory to save grad')
parser.add_argument('--data_path', type=str, help='dataset directory')
parser.add_argument("--bbh_subset", type=str, required=True, help="BBH task type")
parser.add_argument('--data_range', nargs='+', type=int, default=None, help='data range')
parser.add_argument('--gradient_abs', action='store_true', help='whether to use absolute value for each gradient')
parser.add_argument('--weight_gradient', action='store_true', help='whether to collect gradient of weight')
parser.add_argument('--loss_type', choices=['cross_entropy', 'ppl'], default='cross_entropy', help='loss type')
parser.add_argument('--dtype', choices=['fp32', 'fp16', 'bf16'], default='bf16')
parser.add_argument('--aggregating_block_size', type=int, default=64, help='block size for aggregating attention matrix')

args = parser.parse_args()


if __name__ == '__main__':
    if args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError("unsupported data type")

    if "llama" in args.model_name.lower():
        config = LlamaConfig.from_pretrained(args.model_name)
        config._attn_implementation_internal = "eager"
    elif "qwen" in args.model_name.lower():
        config = Qwen2Config.from_pretrained(args.model_name)
        config._attn_implementation_internal = "eager"

    if args.tokenizer_name is None:
        args.tokenizer_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.padding_side='right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    if "llama" in args.model_name.lower():
        model = LlamaForCausalLM.from_pretrained(args.model_name, config=config, torch_dtype=dtype, device_map='auto')
    elif "qwen" in args.model_name.lower():
        model = Qwen2ForCausalLM.from_pretrained(args.model_name, config=config, torch_dtype=dtype, device_map='auto')

    if model.config.architectures[0] == "LlamaForCausalLM":
        print("LlamaForCausalLM")
        LlamaModel_use_attention_matrix_grad_log(model.model)
        # model.model.gradient_checkpointing = True
        model.gradient_checkpointing_enable()
    elif model.config.architectures[0] == "Qwen2ForCausalLM":
        print("Qwen2ForCausalLM")
        Qwen2Model_use_attention_matrix_grad_log(model.model)
        # model.model.gradient_checkpointing = True
        model.gradient_checkpointing_enable()
    else:
        raise NotImplementedError

    if "aime" in args.data_path.lower():
        task = "aime"
    elif "gsm8k" in args.data_path.lower():
        task = "gsm8k"
    elif "math500" in args.data_path.lower():
        task = "math500"
    elif "gpqa" in args.data_path.lower():
        task = "gpqa"
    elif "bbh" in args.data_path.lower():
        task = "bbh"

    # Load dataset
    with open(args.data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(l) for l in f]

    grad_W_dict, grad_Attn_tensor = Grad_Collect(model, tokenizer, args, dataset, task)
    if not os.path.exists(args.grad_dir):
        os.makedirs(args.grad_dir)
    print("Saving profile grad to {}".format(args.grad_dir))
    torch.save(grad_W_dict, os.path.join(args.grad_dir, 'grad_w_dict_{}.pt'.format(task)))
    torch.save(grad_Attn_tensor, os.path.join(args.grad_dir, 'grad_attn_tensor_{}.pt'.format(task)))