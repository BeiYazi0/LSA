import math
import time

import torch
from torch import nn
import transformers

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
    del cache["i"]
    model.config.use_cache = use_cache

    return inps, outs, cache


def min_F_norm(weight, original_weight, inputs_T, outputs_T):
    co, ci = weight.shape
    eps = 1e-3
    I_ci = torch.eye(ci, device=weight.device)
    #
    # S = torch.empty_like(weight)
    # for i in range(co):
    #     Ei = weight[i].unsqueeze(1) * inputs_T
    #     Fi = outputs_T[i]
    #     numer = Fi @ Ei.T
    #     denom = Ei @ Ei.T + eps * I_ci
    #     S[i] = numer @ torch.linalg.inv(denom)
    # weight *= S

    # ----- 1. 计算 G = X @ X.T 并加正则 -----
    G = inputs_T @ inputs_T.t()  # (ci, ci)
    G.diagonal().add_(eps * torch.mean(torch.diag(G)))  # +eps on diag

    # ----- 2. 计算 H = B @ G -----
    # H[i] 就是 h_i
    H = original_weight @ G  # (co, ci)

    # ----- 3. Cholesky 分解 -----
    # G = L @ L.T
    L = torch.linalg.cholesky(G)  # (ci, ci)

    # ----- 4. 用 cholesky_solve 一次性解出所有 y_i -----
    #  我们要解 G Y.T = H.T  ⇒  Y.T = G^{-1} @ H.T
    Yt = torch.cholesky_solve(H.t(), L)  # (ci, co)
    weight.data[:] = Yt.t() * (weight != 0)


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

    def fasterprune(
            self, s, block_size=128, update=False, prune_n=0, prune_m=0
    ):
        weight = self.layer.weight.to(dtype=torch.float32)
        co, ci = weight.shape
        sx = self.H.to(dtype=torch.float32)

        if block_size == 0:
            block_size = ci

        score = (weight ** 2) * torch.diag(sx)

        prune_num = int(block_size * s)
        prune_idx = []
        # re_construct = 0.
        for i1 in range(0, ci, block_size):
            i2 = min(i1 + block_size, ci)

            w1 = weight[:, i1:i2]
            w2 = weight[:, i2:]
            score1 = score[:, i1:i2]
            score2 = score[:, i2:]
            sx1 = sx[i1:i2, i1:i2]
            sx2 = sx[i1:i2, i2:]

            err = torch.zeros_like(w1)
            if prune_n == 0:
                for i in range(prune_num):
                    idx = torch.argmin(score1, dim=1).unsqueeze(1)
                    prune_idx.append(idx + i1)
                    # re_construct += torch.sum(score1.gather(1, idx))

                    w = w1.gather(1, idx)
                    change = w1 * w * sx1[idx.squeeze(1)]
                    score1 += 2 * change
                    score1.scatter_(dim=1, index=idx, value=torch.inf)
                    err.scatter_(dim=1, index=idx, src=w)
            else:
                for i in range(0, block_size, prune_m):
                    for j in range(prune_m - prune_n):
                        idx = torch.argmin(score1[:, i:i + prune_m], dim=1).unsqueeze(1) + i
                        prune_idx.append(idx + i1)

                        w = w1.gather(1, idx)
                        change = w1 * w * sx1[idx.squeeze(1)]
                        score1 += 2 * change
                        score1.scatter_(dim=1, index=idx, value=torch.inf)
                        err.scatter_(dim=1, index=idx, src=w)

            change = w2 * (err @ sx2)
            score2 += 2 * change

        prune_idx = torch.cat(prune_idx, dim=1)
        self.layer.weight.scatter_(dim=1, index=prune_idx, value=0)
        # print(re_construct / 1e6)
        # if update:
            # new_out = weight @ inputs.T
            # s = torch.sum(outputs.T * new_out, dim=1) / torch.sum(new_out ** 2, dim=1)
            # weight *= s.reshape(co, 1)
            # min_F_norm(weight, origin_weight, inputs.T, outputs.T)

    def wanda_prune(self, s, block_size=128, prune_n=0, prune_m=0):
        n, m = prune_n, prune_m
        weight = self.layer.weight
        C_in = weight.shape[1]
        sx = self.H

        metric = weight.abs() * (torch.diag(sx) ** 0.5)
        if n > 0:
            blocks = C_in // m
            offset = torch.repeat_interleave(torch.arange(blocks, device=weight.device) * m, m - n)
            metric = metric.view(-1, m)
            _, sorted_idx = torch.sort(metric, dim=1)
            pruned_idx = sorted_idx[:, :m - n]
            pruned_idx = pruned_idx.reshape(weight.shape[0], blocks * (m - n)) + offset
        else:
            _, sorted_idx = torch.sort(metric, dim=1)
            pruned_idx = sorted_idx[:, :int(C_in * s)]
        weight.scatter_(dim=1, index=pruned_idx, value=0)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def prune_block_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
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
        raise NotImplementedError

    with torch.no_grad():
        inps, outs, cache = prepare_calibration_input(model, layers, dataloader, device)

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

        print(f"pruning layer {i}")
        for j, name in enumerate(subset):
            start_time = time.time()
            p_ratio = args.all_layer_ratio[i]
            if isinstance(p_ratio, list):
                p_ratio = p_ratio[j]
            if args.qkv:
                if "q_proj" in name or "v_proj" in name or "k_proj" in name:
                    wrapped_layers[name].fasterprune(p_ratio, block_size=args.block, update=args.update, prune_n=prune_n, prune_m=prune_m)
                else:
                    wrapped_layers[name].wanda_prune(p_ratio, block_size=args.block, prune_n=prune_n, prune_m=prune_m)
            else:
                wrapped_layers[name].fasterprune(p_ratio, block_size=args.block, update=args.update, prune_n=prune_n, prune_m=prune_m)
            prune_time += time.time() - start_time
            wrapped_layers[name].free()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("time_cost: %.5f sec" % prune_time)


class IterBlockWandaPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prune(self, args):
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        prune_block_wanda(args, self.model, self.tokenizer, device=self.model.device, prune_n=args.N, prune_m=args.M)

        after_pruning_parameters = sum(torch.sum(p != 0).item() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)