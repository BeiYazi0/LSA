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


def dlp_score(args, model, tokenizer, alpha, device=torch.device("cuda:0")):
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

    start_time = time.time()
    with torch.no_grad():
        inps, outs, cache = prepare_calibration_input(model, layers, dataloader, device)

    all_layer_ratio = []
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

        layer_wmetric = []

        print(f"get layer sp layer {i} with dlp")
        for name in subset:
            W = torch.abs(subset[name].weight.data).to(dtype=torch.float32)
            X = wrapped_layers[name].scaler_row.reshape((1, -1)).to(dtype=torch.float32)
            W_metric = W * torch.sqrt(X)

            layer_wmetric.append(W_metric)

        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        all_layer_ratio.append(torch.median(layer_wmetric.float()).abs())

        torch.cuda.empty_cache()

    all_layer_ratio = torch.tensor(all_layer_ratio)
    all_layer_ratio = 1 - all_layer_ratio / torch.sum(all_layer_ratio) # layer imp


    all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) / (all_layer_ratio.max() - all_layer_ratio.min()) * alpha * 2

    all_layer_ratio = args.final_s + torch.mean(all_layer_ratio) - all_layer_ratio

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    print(all_layer_ratio)
    return all_layer_ratio.tolist()


def dlp_score_global(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader = get_examples("c4", tokenizer, n_samples=args.num_examples, seq_len=2048)
    print("dataset loading complete")

    if "Llama" in args.base_model or "llama" in args.base_model:
        layers = model.model.layers
    elif "opt" in args.base_model:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    start_time = time.time()
    with torch.no_grad():
        inps, outs, cache = prepare_calibration_input(model, layers, dataloader, device)

    all_layer_metric = []
    all_layer_numel = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_numel = torch.tensor([subset[name].weight.numel() for name in subset])
        all_layer_numel.append(layer_numel)

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

        layer_wmetric = []

        print(f"get layer sp layer {i}")
        for name in subset:
            # print(f"get layer sp layer {i} name {name}")
            W = torch.abs(subset[name].weight.data).to(dtype=torch.float32)
            X = wrapped_layers[name].scaler_row.reshape((1, -1)).to(dtype=torch.float32)
            W_metric = W * torch.sqrt(X)

            layer_wmetric.append(W_metric)

        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps
        if args.layer == "dlpb":
            layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
            attn_numel = torch.sum(layer_numel[:4]).item()
            attn_median = torch.mean(layer_wmetric[:attn_numel])
            ffn_median = torch.mean(layer_wmetric[attn_numel:])
            all_layer_metric.append(torch.tensor([attn_median] * 4 + [ffn_median] * (len(layer_numel) - 4)))
        elif args.layer == "dlpc":
            layer_wmetric = torch.tensor([torch.median(x).cpu().item() for x in layer_wmetric])
            all_layer_metric.append(layer_wmetric)

    ## mean
    all_layer_metric = torch.stack(all_layer_metric, dim=0)
    all_layer_numel = torch.stack(all_layer_numel, dim=0)
    layer_imp = 1 - all_layer_metric / torch.sum(all_layer_metric)  # layer imp
    layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min()) * args.Lamda * 2
    print(layer_imp)
    layer_prune_numel = all_layer_numel * args.final_s + (torch.mean(layer_imp) - layer_imp) * torch.mean(
        all_layer_numel.float())
    all_layer_ratio = layer_prune_numel / all_layer_numel

    print(all_layer_ratio)
    print(torch.min(all_layer_ratio), torch.max(all_layer_ratio))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    return all_layer_ratio.tolist()


class DLP:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = [2, 0.06, 0.02, 0.04, 0.02, 0.04, 0.1, 0.15, 0.12] # zero index for n:m

    @torch.no_grad()
    def get_layer_sp(self, args):
        self.alpha[0] = args.Lamda
        if args.layer == "dlp":
            return dlp_score(args, self.model, self.tokenizer, self.alpha[int(args.final_s * 10)], device=self.model.device)
        elif args.layer == "dlpb" or args.layer == "dlpc":
            return dlp_score_global(args, self.model, self.tokenizer, device=self.model.device)