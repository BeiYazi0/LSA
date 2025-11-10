import time

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


def get_thresh(model, sp):
    all_linear_weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            all_linear_weights.append(torch.abs(module.weight.flatten()).cpu())
    all_params = torch.cat(all_linear_weights)
    kth_value = torch.kthvalue(all_params, int(all_params.numel() * sp))[0].cuda()
    return kth_value


def prune_global(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
    elif "Qwen" in args.base_model:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    thresh = get_thresh(model, args.final_s)
    torch.cuda.empty_cache()

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for j, name in enumerate(subset):
            subset[name].weight.data[torch.abs(subset[name].weight.data) <= thresh] = 0

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


class GlobalPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prune(self, args):
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        prune_global(args, self.model, self.tokenizer, device=self.model.device, prune_n=args.N, prune_m=args.M)

        after_pruning_parameters = sum(torch.sum(p != 0).item() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)