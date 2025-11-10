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


def prune_mag(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
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

    prune_time = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        print(f"mag pruning layer {i} ")
        for j, name in enumerate(subset):
            p_ratio = args.all_layer_ratio[i]
            if isinstance(p_ratio, list):
                p_ratio = p_ratio[j]
            start_time = time.time()
            W_metric = torch.abs(subset[name].weight.data).to(torch.float32)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W_metric) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.kthvalue(W_metric.flatten(), int(W_metric.numel() * p_ratio))[0]
                W_mask = (W_metric <= thresh)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero
            prune_time += time.time() - start_time
            del W_metric
            torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("time_cost: %.5f sec" % prune_time)


class MagPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prune(self, args):
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        prune_mag(args, self.model, self.tokenizer, device=self.model.device, prune_n=args.N, prune_m=args.M)

        after_pruning_parameters = sum(torch.sum(p != 0).item() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path, max_shard_size="10GB")