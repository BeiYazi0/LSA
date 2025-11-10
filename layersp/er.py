import time

import numpy as np
import torch
from torch import nn


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_layer_wise_sparsity(model, args, plus=False):
    density = 1 - args.final_s
    erk_power_scale = 1

    dense_layers = set()
    if plus:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "head" not in name and "down_proj" in name:
                dense_layers.add(name)

    is_epsilon_valid = False
    # raw_probabilities
    raw_probabilities = None
    epsilon = None

    while not is_epsilon_valid:
        divisor = 0.0
        rhs = 0.0
        raw_probabilities = {}

        # 一次遍历，计算 rhs 和 divisor
        for name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) and "head" not in name):
                continue
            n_param = module.weight.numel()
            # 分配 density
            n_ones = n_param * density
            n_zeros = n_param * (1 - density)

            if name in dense_layers:
                # 已经指定为“全密集”，则把它的全零位都算到 rhs (-n_zeros)
                rhs -= n_zeros
            else:
                # 普通层，累加可分配的“非零位”
                rhs += n_ones
                # ERK 的 raw_probability
                prob = (sum(module.weight.shape) / np.prod(module.weight.shape)) ** erk_power_scale
                raw_probabilities[name] = prob
                divisor += prob * n_param

        epsilon = rhs / divisor

        # 找到最大的 raw_prob，看它是不是 epsilon*raw_prob>1，如果是就要把它也当成 dense
        max_name, max_prob = max(raw_probabilities.items(), key=lambda x: x[1])
        if max_prob * epsilon > 1.0:
            dense_layers.add(max_name)
            # 继续下一轮重新算 epsilon
        else:
            is_epsilon_valid = True

    sparsity_dict = {}
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) and "head" not in name):
            continue
        if name in dense_layers:
            sparsity_dict[name] = 0.0
        else:
            p_one = epsilon * raw_probabilities[name]
            sparsity_dict[name] = 1.0 - p_one

    return sparsity_dict


def er_score(args, model, plus=False):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    sparsity_dict = get_layer_wise_sparsity(model, args, plus)

    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
        layer_prefix = "model.layers"
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
        layer_prefix = "model.decoder.layers"
    elif "Qwen" in args.base_model:
        layers = model.transformer.h
        layer_prefix = "transformer.h"
    else:
        layers = model.model.layers
        layer_prefix = "model.layers"  # 默认路径，可能需要根据模型调整

    start_time = time.time()

    all_layer_ratio = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layer_wmetric = []

        print(f"get layer sp layer {i}")
        for name in subset:
            # 修正：使用与get_layer_wise_sparsity一致的完整路径命名
            full_name = f"{layer_prefix}.{i}.{name}"

            if full_name in sparsity_dict:
                layer_wmetric.append(sparsity_dict[full_name])
            else:
                # 处理缺失键的情况（例如跳过或使用默认值）
                layer_wmetric.append(0)  # 示例：使用0作为默认值

        all_layer_ratio.append(layer_wmetric)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    print(all_layer_ratio)
    return all_layer_ratio


class ER:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_layer_sp(self, args):
        if args.layer == "er":
            return er_score(args, self.model)
        elif args.layer == "erk":
            return er_score(args, self.model, True)
