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


def alpha_score(args, model, s1=0.8, s2=1.2):
    if "Llama" in args.base_model or "llama" in args.base_model:
        blocks = model.model.layers
    elif "opt" in args.base_model:
        blocks = model.model.decoder.layers
    elif "Qwen" in args.base_model:
        blocks = model.transformer.h
    else:
        blocks = model.model.layers

    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    metrics = np.load(args.alpha_file)

    block_metrics = [np.mean(metrics[i:i + layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
    metrics = [i for i in block_metrics for j in range(layer_num_in_block)]

    print("metric values:", metrics)

    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)

    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.final_s / (torch.sum(prunables * layerwise_pruning_ratios))
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios

    all_layer_ratio = [torch.mean(layerwise_pruning_ratios[i:i + layer_num_in_block]).item() for i in
                       range(0, len(layerwise_pruning_ratios), layer_num_in_block)]
    print(all_layer_ratio)
    return all_layer_ratio


class AlphaPruning:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_layer_sp(self, args):
        return alpha_score(args, self.model)
