
import os
import fire
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TextStreamer, StoppingCriteria, StoppingCriteriaList
import transformers


from rpc import enable_rpc, set_rpc_config
from utils.apply_chat_template import apply_chat_template

import json

from rkv.monkeypatch import replace_llama, replace_qwen2

from transformers.cache_utils import DynamicCache

from copy import deepcopy
import torch.nn.functional as F


class ModelStopCriteria(StoppingCriteria):
    """Custom stopping criteria that stops generation when model.stop is True"""
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input_ids, scores, **kwargs):
        # Check if the model has a 'early_exit' attribute and if it's True
        if hasattr(self.model, 'early_exit') and self.model.early_exit:
            return True
        return False


# "/home/yangx/DeepSeek-R1-Distill-Qwen-7B"
# "/home/yangx/QwQ-32B"
# "/home/yangx/DeepSeek-R1-Distill-Qwen-1.5B"
# "/home/yangx/DeepSeek-R1-Distill-Llama-8B"
# "/home/yangx/Llama-3.1-8B-Instruct"

# Heatmap Prompt

# R1-llama-8B 594
# A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?

# R1-qwen-7B 548. R1-qwen-14B 548
# Ralph is going to practice playing tennis with a tennis ball machine that shoots out tennis balls for Ralph to hit. He loads up the machine with 175 tennis balls to start with. Out of the first 100 balls, he manages to hit 2/5 of them. Of the next 75 tennis balls, he manages to hit 1/3 of them. Out of all the tennis balls, how many did Ralph not hit?

# QwQ 32B (838) Qwen3 30B (547) GPT-OSS 20B (456)
# Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

def gen_example(model_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            rpc: bool = False,
            rpc_mode: str = None, # all, recent, recent+first
            rkv: bool = False,
            rkv_mode: str = None, # rkv, snapkv, h2o, streamingllm
            rkv_budget: int = 4096,
            max_new_tokens: int = 32768,
            # RPC arguments
            rpc_buffer_cot: int = 128,
            P=4096,
            R=8,
            c=4,
            selectors='recent',
            aggregation='group',
            rpc_budget_cot=4096,
            rpc_budget_ans=1024,
            cp_ratio=4.0,
            mode="none", # token_heatmap, step_heatmap, entropy, confidence, record_indices, induce_answer, gen_w_inducer
            generate_rounds: bool = False,
            observation_length: int = 1024,
            observation_topk: int = 512,
            window_size: int = 32,
            ):

    attn_implementation = 'eager'
    if 'qwq' in model_path.lower():
        top_k = 40
        temperature = 0.6
        top_p = 0.95
    elif "qwen3" in model_path.lower():
        top_k = 20
        temperature = 0.8
        top_p = 0.7
    elif "gpt" in model_path.lower():
        top_k = None
        temperature = 1
        top_p = 1
    else:
        top_k = None
        temperature = 0.6
        top_p = 0.95

    print(f"Using Model: {model_path}, therefore top_k={top_k}, temperature={temperature}, top_p={top_p}")
    
    if rpc:
        enable_rpc(rpc_mode)
    
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

    elif rpc:

        if "distill-qwen" in model_path.lower() or "qwq" in model_path.lower():
            from rpc.qwen2.qwen2_config import Qwen2Config
            from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
            config = Qwen2Config.from_pretrained(model_path)
            config.update({'rpc_mode':rpc_mode})

            model = Qwen2ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map="auto"
            )
        elif "qwen3" in model_path.lower():
            from rpc.qwen3.qwen3_config import Qwen3MoeConfig
            from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
            config = Qwen3MoeConfig.from_pretrained(model_path)
            config.update({'rpc_mode':rpc_mode})

            model = Qwen3MoeForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map="auto"
            )
        elif "gpt" in model_path.lower():
            from rpc.gpt_oss.gpt_oss_config import GptOssConfig
            from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
            config = GptOssConfig.from_pretrained(model_path)
            config.update({'rpc_mode':rpc_mode})

            model = GptOssForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map="auto"
            )
        elif "llama" in model_path.lower():
            from rpc.llama.llama_config import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            config = LlamaConfig.from_pretrained(model_path)
            config.update({'rpc_mode':rpc_mode})

            model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config,
                device_map="auto"
            )
    
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model.newline_token_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]

        model.CoT_done_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]

    else:

        if "distill-qwen" in model_path.lower() or "qwq" in model_path.lower():
            from rpc.qwen2.qwen2_config import Qwen2Config
            from rpc.qwen2.qwen2_vanilla import Qwen2ForCausalLM

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

        elif "qwen3" in model_path.lower():
            from rpc.qwen3.qwen3_config import Qwen3MoeConfig
            from rpc.qwen3.qwen3_vanilla import Qwen3MoeForCausalLM

            config = Qwen3MoeConfig.from_pretrained(model_path)

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

            model = Qwen3MoeForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
                config=config
            )

        elif "llama" in model_path.lower():
            from rpc.llama.llama_config import LlamaConfig
            from rpc.llama.llama_vanilla import LlamaForCausalLM

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

        elif "gpt" in model_path.lower():
            from rpc.gpt_oss.gpt_oss_config import GptOssConfig
            from rpc.gpt_oss.gpt_oss_vanilla import GptOssForCausalLM

            config = GptOssConfig.from_pretrained(model_path)

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

            model = GptOssForCausalLM.from_pretrained(
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
                            budget_cot=rpc_budget_cot,
                            budget_ans=rpc_budget_ans,
                            cp_ratio=cp_ratio,
                            buffer_cot=rpc_buffer_cot,
                            mode=rpc_mode,
                            )
    elif rkv:
        print([f"RKV Inference -- {rkv_mode} Mode"])
    else:
        print(["Full KV Cache Inference"])

    # Create custom stopping criteria
    custom_stopping_criteria = StoppingCriteriaList([ModelStopCriteria(model)])

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
                    temperature=temperature,
                    top_p=top_p,
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
                    temperature=temperature,
                    top_p=top_p,
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
          
    elif mode == "induce_answer" or mode == "gen_w_inducer":

        if mode == "induce_answer":
            max_new_tokens = observation_length
        elif mode == "gen_w_inducer":
            max_new_tokens = 1757

        answer_inducer_ids = tokenizer("\n**Final Answer**\n\nThe final answer is \\boxed", add_special_tokens=False)["input_ids"]
        # print(len(answer_inducer_ids), answer_inducer_ids) 11
        past_key_values = DynamicCache()
        
        generated_dicts = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tokenizer=tokenizer,
            past_key_values=past_key_values,
            return_dict_in_generate=True,
            output_scores=True,
            streamer=streamer
        )
        if mode == "induce_answer":
            input_ids = generated_dicts.sequences
            past_key_values = generated_dicts.past_key_values
            # Concatenate the last token from input_ids with answer_inducer_ids
            last_token = input_ids[:, -1:]  # Get the last token from input_ids
            answer_inducer_ids = torch.tensor([answer_inducer_ids], device=input_ids.device)
            answer_inducer_ids = torch.cat([last_token, answer_inducer_ids], dim=1)
            output_dicts = model(input_ids=answer_inducer_ids, past_key_values=past_key_values, prompt_len=context_length)
            outputs = input_ids
            output_length = outputs[0][context_length:].shape[-1]

        elif mode == "gen_w_inducer":
            input_ids = generated_dicts.sequences
            past_key_values = generated_dicts.past_key_values

            outputs = input_ids
            output_length = outputs[0][context_length:].shape[-1]

            # Append </think> token to input_ids
            think_end_token = torch.tensor(tokenizer('\n</think>\n\n')['input_ids'], device=input_ids.device).unsqueeze(0)
            input_ids = torch.cat([input_ids, think_end_token], dim=1)

            generated_dicts = model.generate(
                input_ids=input_ids,
                max_new_tokens=32768,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                tokenizer=tokenizer,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
                output_scores=True,
                streamer=streamer
            )

            input_ids = generated_dicts.sequences
            past_key_values = generated_dicts.past_key_values

            outputs = input_ids
            output_length = outputs[0][context_length+output_length:].shape[-1] + output_length
            
        
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
                                    temperature=temperature,
                                    top_p=top_p,
                                    top_k=top_k,
                                    streamer=streamer,
                                    )
            
            # if rkv_mode:

            #     input_ids = outputs.sequences
            #     past_key_values = outputs.past_key_values

            #     # Append </think> token to input_ids
            #     think_end_token = torch.tensor(tokenizer('\n</think>\n\n')['input_ids'], device=input_ids.device).unsqueeze(0)
            #     input_ids = torch.cat([input_ids, think_end_token], dim=1)

            #     outputs = model.generate(
            #         input_ids=input_ids,
            #         max_new_tokens=32768,
            #         do_sample=True,
            #         temperature=0.6,
            #         top_p=0.95,
            #         top_k=top_k,
            #         tokenizer=tokenizer,
            #         past_key_values=past_key_values,
            #         streamer=streamer
            #     )

        output_length = outputs[0][context_length:].shape[-1]
        decoded_output = tokenizer.decode(outputs[0][context_length:], skip_special_tokens=True)


    if rkv_mode and mode != "record_indices" and mode != "induce_answer" and mode != "gen_w_inducer":
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

    elif rpc_mode and mode != "record_indices" and mode != "induce_answer" and mode != "gen_w_inducer":
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

        with open(f"{output_dir}/output_{rpc_mode}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"\nContext Length: {context_length}")
        print(f"Output Length: {output_length}\n")

    elif mode == "record_indices" or mode == "induce_answer" or mode == "gen_w_inducer":
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
        output_dir = "/home/yangx/zmw/ReasoningPathCompression/observation"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/output.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"\nContext Length: {context_length}")
        print(f"Output Length: {output_length}\n")


if __name__ == '__main__':
    
    set_seed(42)
    
    fire.Fire(gen_example)    
