from importlib.metadata import version
import warnings
import transformers
from typing import Callable, Optional, Union
import torch

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

from rpc.qwen2.qwen2_custom_rpc import Qwen2_RPC_init, Qwen2_RPC_forward
from rpc.llama.llama_custom_rpc import Llama_RPC_init, Llama_RPC_forward

from rpc.qwen2.qwen2_custom_window_merge_old_rkv import Qwen2_Ours_init, Qwen2_Ours_forward, Qwen2_Ours_CausalLM_forward
from rpc.llama.llama_custom_window_merge_old_rkv import Llama_Ours_init, Llama_Ours_forward, Llama_Ours_CausalLM_forward

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .flash_attn.flash_attention import flash_attention_forward

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.55']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


def enable_rpc(mode=None):
    check_version()

    if mode == "rpc":

        Qwen2Attention.__init__ = Qwen2_RPC_init
        Qwen2Attention.forward = Qwen2_RPC_forward
        
        LlamaAttention.__init__ = Llama_RPC_init
        LlamaAttention.forward = Llama_RPC_forward
    
    elif mode == "ours_window_merge_rkv":
        
        Qwen2Attention.__init__ = Qwen2_Ours_init
        Qwen2Attention.forward = Qwen2_Ours_forward
        Qwen2ForCausalLM.forward = Qwen2_Ours_CausalLM_forward
        
        LlamaAttention.__init__ = Llama_Ours_init
        LlamaAttention.forward = Llama_Ours_forward
        LlamaForCausalLM.forward = Llama_Ours_CausalLM_forward

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward


def set_rpc_config(
    model,
    P=1024,
    R=32,
    c=4,
    selectors='recent',
    aggregation='all',
    kernel_size=7, 
    pooling='avgpool',
    budget_cot=4096,
    buffer_cot=128,
    budget_ans=1024,
    cp_ratio=0.25,
    mode=None,
    ):

    layers = len(model.model.layers)

    if mode == "rpc":

        for i in range(layers):
            model.model.layers[i].self_attn.kv_cluster.P = P
            model.model.layers[i].self_attn.kv_cluster.T = int(P/c)
            model.model.layers[i].self_attn.kv_cluster.R = R
            model.model.layers[i].self_attn.kv_cluster.selectors = selectors
            model.model.layers[i].self_attn.kv_cluster.aggregation = aggregation
            model.model.layers[i].self_attn.kv_cluster.kernel_size = kernel_size
            model.model.layers[i].self_attn.kv_cluster.pooling = pooling

        print(f"[RPC Config][P={P}, R={R}, c={c}][selectors={selectors}, aggregation={aggregation}]",  flush=True)

    elif "ours" in mode:
        
        for i in range(layers):
            model.model.layers[i].self_attn.kv_cluster.budget_cot = budget_cot
            model.model.layers[i].self_attn.kv_cluster.buffer_cot = buffer_cot
            model.model.layers[i].self_attn.kv_cluster.cp_ratio = cp_ratio
            model.model.layers[i].self_attn.kv_cluster.cp_cot = int(budget_cot*cp_ratio)
            model.model.layers[i].self_attn.kv_cluster.budget_ans = budget_ans
            model.model.layers[i].self_attn.kv_cluster.cp_ans = int(budget_ans*cp_ratio)
            model.model.layers[i].self_attn.kv_cluster.R = R
            model.model.layers[i].self_attn.kv_cluster.selectors = selectors
            model.model.layers[i].self_attn.kv_cluster.aggregation = aggregation
            model.model.layers[i].self_attn.kv_cluster.kernel_size = kernel_size
            model.model.layers[i].self_attn.kv_cluster.pooling = pooling

        print(f"[RPC Config][CoT Budget={budget_cot}, CoT Buffer={buffer_cot}, Ans Budget={budget_ans}, Compression ratio={cp_ratio}][selectors={selectors}, aggregation={aggregation}]",  flush=True)
