import torch
from torch import nn


def atp_score(args, model):
    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
    elif "Qwen" in args.base_model:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    length = len(layers)
    step = args.Lamda
    mean_value = args.final_s
    first_term = mean_value - (step * (length - 1)) / 2
    all_layer_ratio = [first_term + i * step for i in range(length)]

    if step > min(2 * args.final_s / (length - 1), 2 * (1 - args.final_s) / (length - 1)):
        raise ValueError("Illegal beta")
    print(all_layer_ratio)
    return all_layer_ratio


class ATP:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_layer_sp(self, args):
        return atp_score(args, self.model)
