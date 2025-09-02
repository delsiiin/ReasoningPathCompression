
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
            rpc: bool = True,
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
            mode=None,
            ):

    attn_implementation = 'eager'
    if 'qwq' in model_path.lower():
        top_k = 40
    else:
        top_k = None

    print(f"Using Model: {model_path}, therefore top_k={top_k}")
    
    if rpc:
        enable_rpc(mode)
    
    if "qwen" in model_path.lower() or "qwq" in model_path.lower():
        from rpc.qwen2.qwen2_config import Qwen2Config

        config = Qwen2Config.from_pretrained(model_path)

        config.update({'mode':mode})

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
    else:
        print(["Full KV Cache Inference"])

    prompt = input("Prompt:")
    prompt = apply_chat_template(tokenizer, prompt)

    inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    context_length = inputs.input_ids.shape[-1]

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

    # Create data dictionary
    data = {
        "context_length": context_length,
        "output_length": output_length,
        "decoded_output": decoded_output
    }

    # Save to JSONL file
    with open("/home/yangx/ReasoningPathCompression/observation/output.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\nContext Length: {context_length}")
    print(f"Output Length: {output_length}\n")


if __name__ == '__main__':
    
    set_seed(42)
    
    fire.Fire(gen_example)    
