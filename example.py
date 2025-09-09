
import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TextStreamer
import transformers

from rpc.llama.llama_vanilla import LlamaForCausalLM
from rpc.qwen2.qwen2_vanilla import Qwen2ForCausalLM

from rpc import enable_rpc, set_rpc_config
from utils.apply_chat_template import apply_chat_template

import json

from rkv.monkeypatch import replace_llama, replace_qwen2

from transformers.cache_utils import DynamicCache

from copy import deepcopy
import torch.nn.functional as F

# "/home/yangx/DeepSeek-R1-Distill-Qwen-7B"
# "/home/yangx/QwQ-32B"
# "/home/yangx/DeepSeek-R1-Distill-Qwen-1.5B"
# "/home/yangx/DeepSeek-R1-Distill-Llama-8B"
# "/home/yangx/Llama-3.1-8B-Instruct"

# Heatmap Prompt

# R1-llama-8B
# A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?

# R1-qwen-7B
# Ralph is going to practice playing tennis with a tennis ball machine that shoots out tennis balls for Ralph to hit. He loads up the machine with 175 tennis balls to start with. Out of the first 100 balls, he manages to hit 2/5 of them. Of the next 75 tennis balls, he manages to hit 1/3 of them. Out of all the tennis balls, how many did Ralph not hit?

# QwQ 32B
# Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

def gen_example(model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            rpc: bool = False,
            rkv: bool = False,
            rkv_mode: str = None, # rkv, snapkv, h2o, streamingllm
            rkv_budget: int = 4096,
            max_new_tokens: int = 32768,
            # RPC arguments
            P=4096,
            R=32,
            c=4,
            selectors='recent',
            aggregation='all',
            budget_cot=4096,
            budget_ans=1024,
            cp_ratio=4.0,
            mode="none", # heatmap, entropy, confidence (if induce answer, set add "_induce_answer")
            generate_rounds: bool = False,
            observation_length: int = 1024,
            observation_topk: int = 512,
            window_size: int = 32,
            ):

    attn_implementation = 'eager'
    if 'qwq' in model_path.lower():
        top_k = 40
    else:
        top_k = None

    print(f"Using Model: {model_path}, therefore top_k={top_k}")
    
    if rpc:
        enable_rpc(mode)
    
    if rkv:
        # ====== build compression config ======
        compression_config = {
            "method": rkv_mode,
            "method_config": {
                "budget": rkv_budget,
                "window_size": 8,
                "mix_lambda": 0.07,
                "retain_ratio": 0.2,
                "retain_direction": "last",
                "first_tokens": 4,
                "mode": mode,
            },
            "compression": None,
            "update_kv": True
        }
        model_config = {
            "divide_method": "step_length",
            "divide_length": 128,
            "compression_content": "think",
            "method": rkv_mode,
            "mode": mode,
            "observation_length": observation_length,
            "observation_topk": observation_topk,
        }

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # apply monkey patch
        if "llama" in model_path.lower():
            replace_llama(compression_config)
        elif "qwen" in model_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {model_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation=attn_implementation,
            device_map="auto"
        )
        model.eval()

        model.config.update(model_config)

        model.newline_token_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]

        model.after_think_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]

    else:

        if "qwen" in model_path.lower() or "qwq" in model_path.lower():
            from rpc.qwen2.qwen2_config import Qwen2Config

            config = Qwen2Config.from_pretrained(model_path)

            config.update({'mode':mode})

            if mode == "record_indices":
                config.update({'observation_length':observation_length})
                config.update({'observation_topk':observation_topk})
                config.update({'window_size':window_size})
            elif mode == "induce_answer":
                config.update({'observation_length':observation_length})
                config.update({'observation_topk':observation_topk})
                config.update({'window_size':window_size})
                config.update({'induce_answer':True})

            model = Qwen2ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config
            )
        elif "llama" in model_path.lower():
            from rpc.llama.llama_config import LlamaConfig

            config = LlamaConfig.from_pretrained(model_path)

            config.update({'mode':mode})

            if mode == "record_indices":
                config.update({'observation_length':observation_length})
                config.update({'observation_topk':observation_topk})
                config.update({'window_size':window_size})
            elif mode == "induce_answer":
                config.update({'observation_length':observation_length})
                config.update({'observation_topk':observation_topk})
                config.update({'window_size':window_size})
                config.update({'induce_answer':True})

            model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    streamer = TextStreamer(tokenizer)

    if rpc:
        set_rpc_config(model=model,
                            P=P,
                            R=R,
                            c=4,
                            selectors=selectors,
                            aggregation=aggregation,
                            kernel_size=7,
                            pooling='avgpool',
                            budget_cot=budget_cot,
                            budget_ans=budget_ans,
                            cp_ratio=cp_ratio
                            )
    elif rkv:
        print([f"RKV Inference -- {rkv_mode} Mode"])
    else:
        print(["Full KV Cache Inference"])

    prompt = input("Prompt:")
    prompt = apply_chat_template(tokenizer, prompt)

    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    context_length = inputs.input_ids.shape[-1]

    if generate_rounds:
        
        stop_ids = [
                    tokenizer.encode("\n")[-1],
                    tokenizer.encode(".\n")[-1],
                    tokenizer.encode(")\n")[-1],
                    tokenizer.encode("\n\n")[-1],
                    tokenizer.encode(".\n\n")[-1],
                    tokenizer.encode(")\n\n")[-1],
                    tokenizer.eos_token_id
                ]

        past_key_values = DynamicCache()
        first_round = True
        
        while True:
            if first_round:
                generated_dicts = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=top_k,
                    eos_token_id=stop_ids,
                    tokenizer=tokenizer,
                    past_key_values=past_key_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    streamer=streamer
                )
                input_ids = generated_dicts.sequences
                past_key_values = generated_dicts.past_key_values
                first_round = False
            else:
                generated_dicts = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=top_k,
                    tokenizer=tokenizer,
                    eos_token_id=stop_ids,
                    past_key_values=past_key_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    streamer=streamer
                )
                input_ids = generated_dicts.sequences
                past_key_values = generated_dicts.past_key_values
            
            # Check if EOS token is generated
            if tokenizer.eos_token_id in input_ids[0]:
                break

        outputs = input_ids
        output_length = outputs[0][context_length:].shape[-1]
        decoded_output = tokenizer.decode(outputs[0][context_length:], skip_special_tokens=True)
          
    elif mode == "induce_answer":

        answer_inducer_ids = tokenizer("\n**Final Answer**\n\nThe final answer is \\boxed", add_special_tokens=False)["input_ids"]
        # print(len(answer_inducer_ids), answer_inducer_ids) 11
        past_key_values = DynamicCache()
        
        generated_dicts = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=observation_length,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=top_k,
            tokenizer=tokenizer,
            past_key_values=past_key_values,
            return_dict_in_generate=True,
            output_scores=True,
            streamer=streamer
        )
        
        input_ids = generated_dicts.sequences
        past_key_values = generated_dicts.past_key_values
        # Concatenate the last token from input_ids with answer_inducer_ids
        last_token = input_ids[:, -1:]  # Get the last token from input_ids
        answer_inducer_ids = torch.tensor([answer_inducer_ids], device=input_ids.device)
        answer_inducer_ids = torch.cat([last_token, answer_inducer_ids], dim=1)
        output_dicts = model(input_ids=answer_inducer_ids, past_key_values=past_key_values, prompt_len=context_length)

    else:

        if mode == "record_indices" and rkv:
            max_new_tokens = observation_length

        with torch.no_grad():
            outputs = model.generate(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.pad_token_id,
                                    use_cache=True,
                                    do_sample=True,
                                    temperature=0.6,
                                    top_p=0.95,
                                    top_k=top_k,
                                    streamer=streamer)

        output_length = outputs[0][context_length:].shape[-1]
        decoded_output = tokenizer.decode(outputs[0][context_length:], skip_special_tokens=True)


    if rkv_mode:
        # Create data dictionary
        data = {
            "context_length": context_length,
            "output_length": output_length,
            "decoded_output": decoded_output
        }

        # Create directory if it doesn't exist and save to JSONL file
        import os
        output_dir = "/home/yangx/ReasoningPathCompression/observation"
        os.makedirs(output_dir, exist_ok=True)
            
        with open(f"{output_dir}/output_{rkv_mode}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"\nContext Length: {context_length}")
        print(f"Output Length: {output_length}\n")

    elif mode == "record_indices" or mode == "induce_answer":
        print(f"\nContext Length: {context_length}")
        print(f"Output Length: {output_length}\n")
    else:
        # Create data dictionary
        data = {
            "context_length": context_length,
            "output_length": output_length,
            "decoded_output": decoded_output
        }

        # Create directory if it doesn't exist and save to JSONL file
        import os
        output_dir = "/home/yangx/ReasoningPathCompression/observation"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/output.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"\nContext Length: {context_length}")
        print(f"Output Length: {output_length}\n")


if __name__ == '__main__':
    
    set_seed(42)
    
    fire.Fire(gen_example)    
