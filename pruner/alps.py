import time
import numpy as np

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


class WrappedALPS:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

        self.H = None

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        inp = inp.float()

        self.scaler_row += inp.matmul(inp.t())
        self.nsamples += tmp

    def ALPS_admm(self, sp, nm_n=0, nm_m=0, rho=0.1, max_iter=300, update_iter=3, switch_iter=30):
        # get dense weight
        W = self.layer.weight.data.clone().to(dtype=torch.float32)
        dev = W.device

        # W = W.float()
        # W = W.to(dev)
        self.H = self.scaler_row.cpu()

        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        # ridge term
        damp1 = 0.01 * torch.mean(torch.diag(self.H)).item()
        diag = torch.arange(self.H.shape[0], device=self.H.device)
        self.H[diag, diag] += damp1

        # normalization
        X_norm = torch.diag(self.H).sqrt() + 1e-8
        self.H = self.H / X_norm
        self.H = (self.H.T / X_norm).T

        # self.YtX = torch.zeros_like(W)
        self.YtX = torch.matmul(W.cpu() * X_norm, self.H).to(dev)

        admm_st = time.time()

        # initialization
        # XTX_inv = torch.zeros_like(self.H).float().to(dev)
        B = (W * X_norm.to(dev)).t()  # .clone()
        W = None
        B_orig = B.cpu().clone()
        V = torch.zeros_like(B)
        D = torch.zeros_like(B)
        # D_suppp = torch.zeros_like(B)
        D_supp = torch.zeros_like(B)

        totp, num_cout = B.shape
        L, Q = torch.linalg.eigh(self.H.double())
        XTX_inv = (Q @ ((1 / (L + (rho))) * Q).T).float().to(dev)

        init_rho = False
        fix_supp = False
        D_fix = torch.zeros_like(D)

        Res0 = self.YtX.T.cpu()
        Res0 = torch.sum(B_orig.cpu() * Res0)
        Res0 = torch.sum(Res0)

        params = B.shape[0] * B.shape[1]
        k_spar = int(np.round((1 - sp) * params))

        if nm_n == 0:
            D = B.clone().reshape(-1)
            _, loss_idx = torch.topk(-D ** 2, totp * num_cout - k_spar)
            D[loss_idx] = 0
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(totp, num_cout)
        else:
            new_dim = totp * num_cout / nm_m
            new_dim = int(new_dim)
            k_spar = totp * num_cout * nm_n / nm_m

            D = B.clone().t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-D ** 2, nm_m - nm_n, dim=1)
            D = D.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(num_cout, totp).t()

        D_init = D.clone()
        errorp = 1
        for i_admm in range(max_iter):

            B = XTX_inv @ (self.YtX.T - V + rho * D)

            if fix_supp:
                D = ((V + rho * B) / rho) * D_fix
            elif nm_n == 0:
                D = ((V + rho * B) / rho).reshape(-1)
                _, loss_idx = torch.topk(-D ** 2, totp * num_cout - k_spar)
                D[loss_idx] = 0
                D = D.reshape(totp, num_cout)
            else:
                D = ((V + rho * B) / rho).t().reshape((new_dim, nm_m))
                _, loss_idx = torch.topk(-D ** 2, nm_m - nm_n, dim=1)
                D = D.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
                D_supp = (D == 0).to(torch.float)
                D = D.reshape(num_cout, totp).t()

            V = V + rho * (B - D)

            if (i_admm + 1) % update_iter == 0:

                if nm_n == 0:
                    D_supp = (D.reshape(-1) == 0).to(torch.float)
                supp_change = torch.sum((D_supp - D_suppp) ** 2)

                if not fix_supp:
                    if supp_change / k_spar > 0.1:
                        init_rho = True
                        rho *= 1.3
                    elif supp_change / k_spar > 0.005:
                        init_rho = True
                        rho *= 1.2
                    elif supp_change > 0.5:
                        if init_rho:
                            rho *= 1.1
                        else:
                            rho /= 5
                            B = B_orig.clone().to(dev)
                            D = D_init.clone().to(dev)
                            V = torch.zeros_like(B).to(dev)
                    else:
                        if init_rho:
                            break
                        else:
                            rho /= 5

                D_suppp = (D_supp).clone()
                if rho > 1e6:
                    rho = 1e6

                XTX_inv = (Q @ ((1 / (L + (rho))) * Q).T).float().to(dev)

                if nm_n == 0:
                    Btest = B.reshape(-1)
                    _, loss_idx = torch.topk(-Btest ** 2, totp * num_cout - k_spar)
                    Btest[loss_idx] = 0
                    Btest = Btest.reshape(totp, num_cout)
                else:
                    Btest = B.t().reshape((new_dim, nm_m))
                    _, loss_idx = torch.topk(-Btest ** 2, nm_m - nm_n, dim=1)
                    Btest = Btest.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
                    Btest = Btest.reshape(num_cout, totp).t()

                Resc = torch.matmul(self.H.to(dev), Btest) - self.YtX.T
                Resc = torch.diag(torch.matmul((Btest - B_orig.to(dev)).t(), Resc))

                errorc = torch.sum(Resc).to("cpu") / Res0
                errorc = errorc.item()

                # print("iter {}, error {} support change {}, rho {}".format(i_admm, errorc / errorp, supp_change / k_spar, rho))

                if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                    break

        if nm_n == 0:
            B = B.reshape(-1)
            _, loss_idx = torch.topk(-B ** 2, totp * num_cout - k_spar)
            B[loss_idx] = 0
            B = B.reshape(totp, num_cout)
        else:
            B = B.t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-B ** 2, nm_m - nm_n, dim=1)
            B = B.scatter(src=torch.zeros((new_dim, nm_m - nm_n)).to(dev), dim=1, index=loss_idx)
            B = B.reshape(num_cout, totp).t()

        V = None
        D = None

        Res = torch.matmul(self.H, B.cpu()) - self.YtX.T.cpu()
        Res = torch.diag(torch.matmul((B.cpu() - B_orig).t(), Res))

        error = torch.sum(Res) / Res0
        error = error.item()

        # print("Before backsolve, error is {}".format(error))
        admm_time = time.time() - admm_st

        back_st = time.time()
        B = self.cg_batch((self.H).to(dev), self.YtX.T,
                          (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10, verbose=False)

        back_time = time.time() - back_st

        Res = torch.matmul(self.H, B.cpu()) - self.YtX.T.cpu()
        Res = torch.diag(torch.matmul((B.cpu() - B_orig).t(), Res))

        error = torch.sum(Res) / Res0
        error = error.item()

        torch.cuda.synchronize()

        if isinstance(self.layer, transformers.Conv1D):
            self.layer.weight.data = (B.t() / X_norm.to(dev)).t().reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype)
        else:
            self.layer.weight.data = (B.t() / X_norm.to(dev)).reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype)

        # print("Number of iter is {}".format(i_admm))
        # print("Final Error is {}".format(error))
        # print("Time is admm: {} back:{}".format(admm_time, back_time))

        return error

    def cg_batch(self, A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

        This function solves matrix linear systems of the form

            A X = B,

        where A is a n x n positive definite matrix and B is a n x m matrix,
        and X is the n x m matrix representing the solution for the ith system.

        Args:
            A_bmm: A callable that performs a batch matrix multiply of A and a n x m matrix.
            B: A n x m matrix representing the right hand sides.
            M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a n x m matrix. (default=identity matrix)
            X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
            rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
            atol: (optional) Absolute tolerance for norm of residual. (default=0)
            maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
            verbose: (optional) Whether or not to print status messages. (default=False)
        """
        error_list = np.zeros((maxiter,))
        n, m = B.shape

        if M_bmm is None:
            M_bmm = lambda x: x
        if X0 is None:
            X0 = M_bmm(B)
        if maxiter is None:
            maxiter = 5 * n

        assert B.shape == (n, m)
        assert X0.shape == (n, m)
        assert rtol > 0 or atol > 0
        assert isinstance(maxiter, int)

        X_k = X0

        R_k = B - A @ X_k
        R_k = R_k * A_supp

        Z_k = M_bmm(R_k)

        P_k = torch.zeros_like(Z_k)

        P_k1 = P_k
        R_k1 = R_k
        R_k2 = R_k
        X_k1 = X0
        Z_k1 = Z_k
        Z_k2 = Z_k

        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

        if verbose:
            print("%03s | %010s %06s" % ("it", "dist", "it/s"))

        optimal = False
        start = time.perf_counter()
        for k in range(1, maxiter + 1):
            start_iter = time.perf_counter()
            Z_k = M_bmm(R_k)

            if k == 1:
                P_k = Z_k
                R_k1 = R_k
                X_k1 = X_k
                Z_k1 = Z_k
            else:
                R_k2 = R_k1
                Z_k2 = Z_k1
                P_k1 = P_k
                R_k1 = R_k
                Z_k1 = Z_k
                X_k1 = X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1

            denominator = (P_k * (A @ P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A @ P_k)
            R_k = R_k * A_supp
            end_iter = time.perf_counter()

            residual_norm = torch.norm(A @ X_k - B, dim=1)

            if verbose:
                print("%03d | %8.4e" %
                      (k, torch.max(residual_norm / B_norm)))

            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break

        end = time.perf_counter()

        if verbose:
            if optimal:
                print("Terminated in %d steps (optimal). Took %.3f ms." %
                      (k, (end - start) * 1000))
            else:
                print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                      (k, (end - start) * 1000))

        info = {
            "niter": k,
            "optimal": optimal
        }

        return X_k

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.X = None
        self.Y = None
        self.XXt = None
        self.YXt = None
        self.YtX = None
        # self.XtX = None
        torch.cuda.empty_cache()


def prune_alps(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
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
            wrapped_layers[name] = WrappedALPS(subset[name])

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
        for name in subset:
            start_time = time.time()
            wrapped_layers[name].ALPS_admm(sp=args.final_s, nm_n=prune_n, nm_m=prune_m, rho=0.1)
            prune_time += time.time() - start_time
            wrapped_layers[name].free()

        for j in range(args.num_examples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **cache)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("time_cost: %.5f sec" % prune_time)


class ALPSPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def prune(self, args):
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        prune_alps(args, self.model, self.tokenizer, device=self.model.device, prune_n=args.N, prune_m=args.M)

        after_pruning_parameters = sum(torch.sum(p != 0).item() for p in self.model.parameters())
        print("After prune, #parameters: {}".format(after_pruning_parameters))

        if args.save_path is not None:
            self.tokenizer.save_pretrained(args.save_path)
            self.model.save_pretrained(args.save_path)