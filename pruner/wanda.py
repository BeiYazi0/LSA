import time

import torch
from torch import nn

from datas import get_examples


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


def prepare_calibration_input(model, layers, dataloader, device):
    model.seqlen = 2048
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs and kwargs['position_ids'] is not None:
                cache['position_ids'] = kwargs['position_ids']
            if "position_embeddings" in kwargs and kwargs['position_embeddings'] is not None:
                cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.unsqueeze(0).to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    del cache["i"]

    return inps, outs, cache


class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def free(self):
        self.scaler_row = None


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)
    print("dataset loading complete")

    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
    elif "Qwen" in args.base_model:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    with torch.no_grad():
        inps, outs, cache = prepare_calibration_input(model, layers, dataloader, device)

    if prune_n != 0:
        all_n = prune_n - torch.round(torch.tensor(args.all_layer_ratio))
        print(f"mix_n: {all_n}")

    prune_time = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            for key, value in cache.items():
                if isinstance(value, tuple):
                    cache[key] = tuple([v.to(dev) for v in value])
                else:
                    cache[key] = value.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        for h in handles:
            h.remove()

        print(f"pruning layer {i}")
        for j, name in enumerate(subset):
            p_ratio = args.all_layer_ratio[i]
            if isinstance(p_ratio, list):
                p_ratio = p_ratio[j]
            start_time = time.time()
            W = torch.abs(subset[name].weight.data).to(dtype=torch.float32)
            X = wrapped_layers[name].scaler_row.reshape((1, -1)).to(dtype=torch.float32)
            W_metric = W * torch.sqrt(X)

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                prune_n = int(all_n[i])
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_m - prune_n, dim=1, largest=False)[1], True)
            else:
                if args.use_variant:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - p_ratio) > 0.001) and (
                            alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > p_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    num_prune = int(W_metric.shape[1] * p_ratio)
                    _, indices = torch.topk(W_metric, k=num_prune, dim=-1, largest=False)
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero
            prune_time += time.time() - start_time
            wrapped_layers[name].free()

            del W_mask, W, X, W_metric
            torch.cuda.empty_cache()

        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("time_cost: %.5f sec" % prune_time)


class WandaPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prune(self, args):
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        prune_wanda(args, self.model, self.tokenizer, device=self.model.device, prune_n=args.N, prune_m=args.M)

        after_pruning_parameters = sum(torch.sum(p != 0).item() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path, max_shard_size="10GB")