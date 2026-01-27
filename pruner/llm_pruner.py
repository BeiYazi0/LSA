import torch
import time

from datas import get_examples
from utils import create_llama_groups


class LLMPruner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.groups = None

    def prune(self, args):
        def register_forward_hook(module):  # hook 每层的输入
            def hook_fn(module, inputs, outputs):
                module.input = inputs[0].reshape(-1, module.weight.shape[1]).transpose(0, 1)
                module.output = outputs.reshape(-1, module.weight.shape[0]).transpose(0, 1)

            return module.register_forward_hook(hook_fn)

        self.groups = create_llama_groups(self.model, start=args.start_layer,
                                          end=args.end_layer, mode=args.mode, itype=args.imp)

        if args.imp == "none":
            for param in self.model.parameters():
                param.requires_grad_(True)
        before_pruning_parameters = sum(p.numel() for p in self.model.parameters())
        print("Before prune, #parameters: {}".format(before_pruning_parameters))

        device = next(iter(self.model.parameters())).device

        keep_config = {"start": args.start_layer, "end": args.end_layer, "mode": args.mode, "pruner": "llm"}

        for i in range(args.iters):
            if args.imp == "none" or args.imp == "err":
                example_prompts = get_examples('bookcorpus', self.tokenizer, args.num_examples, seq_len=64).to(device)

                if args.imp == "err":
                    for name, module in self.model.model.named_modules():
                        if "lm_head" not in name and isinstance(module, torch.nn.Linear):
                            register_forward_hook(module)
                start_time = time.time()
                for j in range(args.num_examples):
                    batch_input = example_prompts[j].unsqueeze(0).to(device)
                    loss = self.model(batch_input, labels=batch_input).loss
                    loss.backward()

            # p_ratio = args.all_layer_ratio[i]
            # cur_d = 1 - p_ratio * (i + 1) / args.iters
            with torch.no_grad():
                for idx, group in enumerate(self.groups):  # local
                    p_ratio = args.all_layer_ratio[idx // 2]
                    cur_d = 1 - p_ratio * (i + 1) / args.iters
                    imp = group.cal_importance()
                    keep_idx = torch.sort(torch.topk(imp, dim=0, k=int(group.o_channels * cur_d))[1])[0]
                    group.prune(keep_idx)
                    keep_config[group.name] = keep_idx.tolist()

            for layer in self.model.model.layers:
                if args.mode == "channel":
                    layer.self_attn.hidden_size = layer.self_attn.q_proj.weight.data.shape[0]
                    layer.self_attn.head_dim = layer.self_attn.hidden_size // layer.self_attn.num_heads
                    layer.self_attn.rotary_emb.dim = layer.self_attn.head_dim
                    layer.self_attn.rotary_emb.inv_freq = 1.0 / (layer.self_attn.rotary_emb.base ** (
                            torch.arange(0, layer.self_attn.rotary_emb.dim, 2, dtype=torch.int64).float().to(
                                layer.self_attn.rotary_emb.inv_freq.device) / layer.self_attn.rotary_emb.dim))
                else:
                    layer.self_attn.hidden_size = layer.self_attn.q_proj.weight.data.shape[0]
                    layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
                    layer.self_attn.num_key_value_heads = layer.self_attn.num_heads // layer.self_attn.num_key_value_groups

            after_pruning_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("After Iter {}/{}, #parameters: {}".format(i + 1, args.iters, after_pruning_parameters))
            end_time = time.time()
            print("time_cost: %.5f sec" % (end_time - start_time))

        return keep_config

#  遮掩 head
