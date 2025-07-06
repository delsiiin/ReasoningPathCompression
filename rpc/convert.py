from importlib.metadata import version
import warnings
import transformers

# from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
from rpc.llama.llama_vanilla import LLAMA_ATTENTION_CLASSES
from rpc.qwen2.qwen2_vanilla import QWEN2_ATTENTION_CLASSES
from rpc.qwen2.qwen2_vanilla import Qwen2Model, Qwen2ForCausalLM


def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.45']
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
        from rpc.llama.llama_custom import LlamaRPCAttention
        from rpc.qwen2.qwen2_custom_rpc import Qwen2RPCAttention, Qwen2RPCModel
        
        # cant get attn_weights from flash-attn
        LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaRPCAttention

        QWEN2_ATTENTION_CLASSES['flash_attention_2'] = Qwen2RPCAttention
        Qwen2Model.forward = Qwen2RPCModel.forward

    elif mode == "ours_all_step":
        from rpc.llama.llama_custom import LlamaRPCAttention
        from rpc.qwen2.qwen2_custom_all_step import Qwen2RPCAttention, Qwen2RPCModel

        # cant get attn_weights from flash-attn
        LLAMA_ATTENTION_CLASSES['eager'] = LlamaRPCAttention

        QWEN2_ATTENTION_CLASSES['eager'] = Qwen2RPCAttention
        Qwen2Model.forward = Qwen2RPCModel.forward

    elif mode == "ours_window":
        from rpc.llama.llama_custom import LlamaRPCAttention
        from rpc.qwen2.qwen2_custom_window import Qwen2RPCAttention, Qwen2RPCModel, Qwen2RPCForCausalLM

        # cant get attn_weights from flash-attn
        LLAMA_ATTENTION_CLASSES['eager'] = LlamaRPCAttention

        QWEN2_ATTENTION_CLASSES['eager'] = Qwen2RPCAttention
        Qwen2Model.forward = Qwen2RPCModel.forward
        Qwen2ForCausalLM.prepare_inputs_for_generation = Qwen2RPCForCausalLM.prepare_inputs_for_generation

    elif mode == "ours_window_merge":
        from rpc.llama.llama_custom import LlamaRPCAttention
        from rpc.qwen2.qwen2_custom_window_merge_old import Qwen2RPCAttention, Qwen2RPCModel, Qwen2RPCForCausalLM

        # cant get attn_weights from flash-attn
        LLAMA_ATTENTION_CLASSES['eager'] = LlamaRPCAttention

        QWEN2_ATTENTION_CLASSES['eager'] = Qwen2RPCAttention
        Qwen2Model.forward = Qwen2RPCModel.forward
        Qwen2ForCausalLM.prepare_inputs_for_generation = Qwen2RPCForCausalLM.prepare_inputs_for_generation

    elif mode == "ours_window_merge_rkv":
        from rpc.llama.llama_custom import LlamaRPCAttention
        from rpc.qwen2.qwen2_custom_window_merge_old_rkv import Qwen2RPCAttention, Qwen2RPCModel, Qwen2RPCForCausalLM

        # cant get attn_weights from flash-attn
        LLAMA_ATTENTION_CLASSES['eager'] = LlamaRPCAttention

        QWEN2_ATTENTION_CLASSES['eager'] = Qwen2RPCAttention
        Qwen2Model.forward = Qwen2RPCModel.forward
        Qwen2ForCausalLM.prepare_inputs_for_generation = Qwen2RPCForCausalLM.prepare_inputs_for_generation

    elif mode == "ours_window_merge_new":
        from rpc.llama.llama_custom import LlamaRPCAttention
        from rpc.qwen2.qwen2_custom_window_merge import Qwen2RPCAttention, Qwen2RPCModel, Qwen2RPCForCausalLM

        # cant get attn_weights from flash-attn
        LLAMA_ATTENTION_CLASSES['eager'] = LlamaRPCAttention

        QWEN2_ATTENTION_CLASSES['eager'] = Qwen2RPCAttention
        Qwen2Model.forward = Qwen2RPCModel.forward
        Qwen2ForCausalLM.prepare_inputs_for_generation = Qwen2RPCForCausalLM.prepare_inputs_for_generation

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
