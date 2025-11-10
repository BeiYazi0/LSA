import os
import torch
from torch import nn
import onnx
from transformers.cache_utils import DynamicCache


class LlamaForCausalLMWrapper(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.args = args

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        output_attentions=False,
        position_ids=None,
        output_hidden_states=False,
        use_cache=True,
    ):
        if position_ids is None:
            position_ids = torch.ones([1, 1], dtype=torch.int64)
        # 创建新版transformers兼容的缓存对象
        past_key_values_cache = DynamicCache()
        for i, (k, v) in enumerate(past_key_values):
            past_key_values_cache.update(
                key_states=k,
                value_states=v,
                layer_idx=i,
                cache_kwargs={"position_ids": position_ids}
            )

        inputs_embeds = self.model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # 新版transformers要求返回Cache对象
        outputs = self.model.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values_cache,
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = False

        )

        hidden_states = outputs[0]
        present_key_values = outputs[1] if use_cache else None

        hidden_states = hidden_states[:, -1, :]
        hidden_states = self.model.model.norm(hidden_states)
        lm_logits = self.model.lm_head(hidden_states)

        # 将Cache对象转换回元组格式
        kv_caches_out = []
        if present_key_values is not None:
            for i in range(len(present_key_values)):
                kv_caches_out.extend(present_key_values[i])

        topk_outputs = []
        if self.args.add_topk_warper > 0:
            topk_outputs = torch.topk(lm_logits, k=self.args.topk, dim=-1)

        return (lm_logits, *kv_caches_out, *topk_outputs)


def export_llama_to_onnx(model, config, dtype, args):
    if not os.path.exists(args.onnx_export_path):  ##目录存在，返回为真
        print('create onnx export path')
        os.makedirs(args.onnx_export_path)

    """将模型导出为内存中的ONNX缓冲区"""
    llama_model_wrapper = LlamaForCausalLMWrapper(model, config, args)

    onnx_file_name = os.path.join(args.onnx_export_path, "model-orig.onnx")

    layer_num = len(model.model.layers)

    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    hidden_size1 = hidden_size // head_num

    batch = 1
    N = 1
    sumN = 1024
    lastN = sumN - N

    input_ids_shape = [batch, N]
    input_ids = torch.ones(input_ids_shape, dtype=torch.int64)
    attention_mask = torch.randn([batch, 1, N, sumN], dtype=dtype)

    in_names = ["input_ids", "attention_mask"]
    #
    # dynamic_axes = {
    #     'input_ids': {1: 'N', },
    #     'attention_mask': {2: 'N', 3: "sumN"},
    #     "position_ids": {1: 'N', },
    # }
    # dynamic_shapes = {
    #     'input_ids': {1: 1, },
    #     'attention_mask': {2: 1, 3: 128},
    #     "position_ids": {1: 1, },
    #     "past_key_values": {1: 32, 2: 127, 3: 128},
    # }

    # kv_caches_in = []
    out_names = ["lm_logits"]

    kv_cache_in_shape = [batch, head_num, lastN, hidden_size1]
    # kv_cache_dyn_axes = {2: "lastSum"}

    # 为每层创建KV缓存
    past_key_values = []
    for _ in range(layer_num):
        past_key_in = torch.randn(kv_cache_in_shape, dtype=dtype)
        past_value_in = torch.randn(kv_cache_in_shape, dtype=dtype)
        past_key_values.append((past_key_in, past_value_in))

    # 注意：这里直接传递元组列表，包装器内会转换
    input_datas = (input_ids, attention_mask, past_key_values)

    torch.onnx.export(
        llama_model_wrapper,
        input_datas,
        onnx_file_name,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=in_names,
        output_names=out_names,
        dynamo=True,
        # dynamic_shapes=dynamic_shapes,
    )

    model = onnx.load(onnx_file_name, load_external_data=False)
    print(f"original IR_VERSION: {model.ir_version}")
    for opset_id in model.opset_import:
        print(f"original opset_version: {opset_id.version} (domain: {opset_id.domain})")

    model.ir_version = 9
    onnx_file_name = os.path.join(args.onnx_export_path, "model.onnx")

    # 3. 保存兼容版本
    onnx.save(model, onnx_file_name)
    print("export down")

    return onnx_file_name
