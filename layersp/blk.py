import math
import time

import torch
import transformers
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


class BlockWanda:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def blk_s(
            self, block_size=128, s=0.5
    ):
        weight = self.layer.weight.to(dtype=torch.float32)
        co, ci = weight.shape
        sx = self.H.to(dtype=torch.float32)
        # sw = weight.T @ weight
        #
        # s = sx * sw
        # print(s[1:17, 1:17])
        # input()

        if block_size == 0:
            block_size = ci

        score = (weight ** 2) * torch.diag(sx)
        scale = torch.max(score)

        re_construct = 0.
        # s = 0.5
        prune_num = int(block_size * s)
        prune_idx = []
        for i1 in range(0, ci, block_size):
            i2 = min(i1 + block_size, ci)

            w1 = weight[:, i1:i2]
            w2 = weight[:, i2:]
            score1 = score[:, i1:i2]
            score2 = score[:, i2:]
            sx1 = sx[i1:i2, i1:i2]
            sx2 = sx[i1:i2, i2:]

            err = torch.zeros_like(w1)
            for i in range(prune_num):
                idx = torch.argmin(score1, dim=1).unsqueeze(1)
                re_construct += torch.sum(score1.gather(1, idx))
                prune_idx.append(idx + i1)

                w = w1.gather(1, idx)
                change = w1 * w * sx1[idx.squeeze(1)]
                score1 += 2 * change
                score1.scatter_(dim=1, index=idx, value=torch.inf)
                err.scatter_(dim=1, index=idx, src=w)

            change = w2 * (err @ sx2)
            score2 += 2 * change

        # re_construct /= (co * ci * 0.5)
        return re_construct

    def wanda_s(self, block_size=128):
        weight = self.layer.weight
        sx = self.H

        metric = weight.abs() * (torch.diag(sx) ** 0.5)
        return torch.mean(metric)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def blk_score(args, model, tokenizer, alpha, device=torch.device("cuda:0")):
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
    all_layer_metric = []
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
            wrapped_layers[name] = BlockWanda(subset[name])

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
            W_metric = wrapped_layers[name].blk_s(block_size=args.block, s=args.resp)
            wrapped_layers[name].free()

            layer_wmetric.append(W_metric)

        for j in range(args.num_examples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        all_layer_metric.append(layer_wmetric)
        all_layer_ratio.append(torch.mean(layer_wmetric.float()).abs())

        torch.cuda.empty_cache()

    all_layer_ratio = torch.tensor(all_layer_ratio)
    # print(torch.stack(all_layer_metric, dim=0))
    print(all_layer_ratio / torch.sum(all_layer_ratio))
    layer_imp = 1 - all_layer_ratio / torch.sum(all_layer_ratio)  # layer imp
    layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min()) * alpha * 2
    # layer_imp = layer_imp / torch.sum(layer_imp)
    # layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min())
    all_layer_ratio = args.final_s + torch.mean(layer_imp) - layer_imp

    # softmax 减小方差
    # while any((all_layer_ratio < 0) | (all_layer_ratio > 1)):
    #     layer_imp = torch.softmax(layer_imp, dim=0)
    #     all_layer_ratio = args.final_s + torch.mean(layer_imp) - layer_imp

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    print(all_layer_ratio)
    return all_layer_ratio.tolist()


### block?? multiply??
def blk_score_global(args, model, tokenizer, device=torch.device("cuda:0")):
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

    all_layer_ratio = []
    all_layer_metric = []
    all_layer_numel = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_numel = [subset[name].weight.numel() for name in subset]
        all_layer_numel.append(torch.tensor(layer_numel))

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
            wrapped_layers[name] = BlockWanda(subset[name])

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
            W_metric = wrapped_layers[name].blk_s(block_size=args.block)
            wrapped_layers[name].free()

            layer_wmetric.append(W_metric)

        for j in range(args.num_examples):
            outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        all_layer_metric.append(layer_wmetric)
        all_layer_ratio.append(torch.mean(layer_wmetric.float()).abs())

        torch.cuda.empty_cache()

    ## mean
    all_layer_metric = torch.stack(all_layer_metric, dim=0)
    # print(all_layer_metric)
    if args.layer == "blkb":
        attn_metric = torch.mean(all_layer_metric[:, :4], dim=1, keepdim=True)
        ffn_metric = torch.mean(all_layer_metric[:, 4:], dim=1, keepdim=True)
        all_layer_metric[:, :4] = attn_metric @ torch.ones(1, 4)
        all_layer_metric[:, 4:] = ffn_metric @ torch.ones(1, all_layer_metric.shape[1] - 4)
        # print(all_layer_metric)

    all_layer_numel = torch.stack(all_layer_numel, dim=0)
    layer_imp = 1 - all_layer_metric / torch.sum(all_layer_metric)  # layer imp
    layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min()) * args.Lamda * 2
    # print(layer_imp)
    layer_prune_numel = all_layer_numel * args.final_s + (torch.mean(layer_imp) - layer_imp) * torch.mean(
        all_layer_numel.float())
    all_layer_ratio = layer_prune_numel / all_layer_numel
    # all_layer_ratio = args.final_s + torch.mean(layer_imp) - layer_imp

    # component wise llama2-7 37.28
    # all_layer_metric = torch.stack(all_layer_metric, dim=0)
    # all_layer_numel = torch.stack(all_layer_numel, dim=0)
    # layer_imp = 1 - all_layer_metric / torch.sum(all_layer_metric)
    # layer_imp = (layer_imp - layer_imp.min()) / (layer_imp.max() - layer_imp.min()) * args.Lamda * 2
    #
    # layer_prune_numel = torch.sum(all_layer_numel * args.final_s)
    # bias_numel = torch.sum(all_layer_numel * (torch.mean(layer_imp) - layer_imp))
    # p0 = (layer_prune_numel - bias_numel) / torch.sum(all_layer_numel)
    # all_layer_ratio = p0 + torch.mean(layer_imp) - layer_imp

    print(all_layer_ratio)
    print(torch.min(all_layer_ratio), torch.max(all_layer_ratio))

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    end_time = time.time()
    print("time_cost: %.5f sec" % (end_time - start_time))
    return all_layer_ratio.tolist()


class BLK:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = [2, 0.06, 0.02, 0.04, 0.02, 0.04, 0.1, 0.15, 0.12]  # zero index for n:m

    @torch.no_grad()
    def get_layer_sp(self, args):
        self.alpha[0] = args.Lamda
        if args.layer == 'blk':
            return blk_score(args, self.model, self.tokenizer, self.alpha[int(args.final_s * 10)],
                             device=self.model.device)
        elif args.layer == "blkb" or args.layer == "blkc":
            return blk_score_global(args, self.model, self.tokenizer,
                                    device=self.model.device)
        else:
            raise NotImplementedError
