import json
import os
import argparse
import fnmatch

import numpy as np
import torch
import random

np.random.seed(0)
torch.manual_seed(0)

from transformers import AutoModelForCausalLM, AutoTokenizer

from pruner import *
from layersp import *
from lm_eval import tasks, evaluator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def eval_zero(args, model, tokenizer, task_names):
    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model_type=args.model_type,
        model=(tokenizer, model),
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )

    if results is None:
        return

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        import os
        directory_path = os.path.dirname(args.output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model_type} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(evaluator.make_table(results))


def make_parser():
    parser = argparse.ArgumentParser("Pruner Processor")

    parser.add_argument("-p", "--pruner", default=None, type=str, help="pruner")
    parser.add_argument("-s", "--final_s", default=0., type=float, help="final sparsity")

    parser.add_argument('--base_model', type=str, help='model name')
    parser.add_argument('--save_path', type=str, default=None,
                        help='save path')

    # llm pruner
    parser.add_argument("-n_exa", "--num_examples", default=128, type=int, help="prune examples num")
    parser.add_argument("-it", "--iters", default=1, type=int, help="prune iters")
    parser.add_argument("-start", "--start_layer", default=3, type=int, help="layer start to prune")
    parser.add_argument("-end", "--end_layer", default=30, type=int, help="layer end prune")
    parser.add_argument("-m", "--mode", default="none", type=str, help="channel")
    parser.add_argument("-i", "--imp", default="none", type=str, help="imp type")
    parser.add_argument("--layer", default="uniform", type=str, help="layer sp")

    # \beta
    parser.add_argument("--lamda", default=1., type=float, help="lambda")

    # control
    parser.add_argument("--fp16", action='store_true', help='use fp16')
    parser.add_argument("--decay", action='store_true', help='use decay')
    parser.add_argument("--wide", action='store_true', help='use wide')
    parser.add_argument("--qkv", action='store_true', help='use var')
    parser.add_argument("--update", action='store_true', help='update weight after prune')
    parser.add_argument("--use_variant", action='store_true',
                        help='whether to use the wanda variant described in the appendix')
    parser.add_argument("--block", default=0, type=int, help='use block')

    parser.add_argument("-prune_n", "--N", default=0, type=int, help="prune N")
    parser.add_argument("-prune_m", "--M", default=0, type=int, help="prune M")

    # owl
    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )
    parser.add_argument("--lod", action='store_true', help='measure lod')

    # alpha pruning
    parser.add_argument('--alpha_file', type=str, help='alpha pruning file metric')

    parser.add_argument('--add_topk_warper', required=False, type=int, default=0)
    parser.add_argument('--topk', required=False, type=int, default=4)

    parser.add_argument("--all_layer_ratio", nargs='+', default=[], help='use layer_ratio')

    ## eval zero
    parser.add_argument('--model_type', type=str, default="hf-causal-experimental", help='model type')
    parser.add_argument("--model_args", default="pretrained=facebook/opt-1.3b")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    ## gptq
    parser.add_argument("--gptq", action='store_true', help='gptq')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')

    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8],
                        help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')

    ## onnx
    parser.add_argument('--onnx_export_path', type=str, default=None,
                        help='onnx export path')
    parser.add_argument("--deepsparse", action='store_true', help='measure latency')
    parser.add_argument('--opset', required=False, type=int, default=18)

    parser.add_argument("--seed", default=0, type=int, help='use block')
    parser.add_argument("--resp", default=0.5, type=float, help='resp')

    return parser


def load_model_tokenizer(args):
    ckpt_dir = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        # quantization_config=bnb_config,  # 上面本地模型的配置
        device_map="cpu" if args.deepsparse else "auto",  # 使用GPU的编号
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )
    return tokenizer, model


if __name__ == '__main__':
    args = make_parser().parse_args()

    tokenizer, model = load_model_tokenizer(args)

    pruner_dic = {"llm": LLMPruner,
                  "wanda": WandaPruner, "sgpt": SparseGPTPruner, "mag": MagPruner,
                  "global": GlobalPruner,
                  "iblock": IterBlockWandaPruner,
                  "ria": RIAPruner,
                  "alps": ALPSPruner}
    layer_sp_dic = {"uniform": Uniform, "owl": OWL, "owlb": OWL, "owlc": OWL,
                    "dlp": DLP, "dlpb": DLP, "dlpc": DLP,
                    "lsa": BLK, "lsab": BLK, "lsac": BLK,
                    "atp": ATP, "alpha": AlphaPruning,
                    "er": ER, "erk": ER}
    random.seed(args.seed)
    if args.pruner is not None:
        if len(args.all_layer_ratio) == 0:
            layer_sp = layer_sp_dic[args.layer](model, tokenizer)
            args.all_layer_ratio = layer_sp.get_layer_sp(args)
        else:
            args.all_layer_ratio = [float(ratio) for ratio in args.all_layer_ratio]
            print(args.all_layer_ratio)
        pruner = pruner_dic[args.pruner](model, tokenizer)
        keep_config = pruner.prune(args)

        if keep_config and args.save_path is not None:
            with open(args.save_path, "w", encoding="utf8") as f:
                f.write(json.dumps(keep_config, indent=4, ensure_ascii=False))

    if args.gptq:
        from gptq_test import llama_gptq
        llama_gptq(model, tokenizer, dev=torch.device("cuda:0"), args=args)

    if args.deepsparse:
        from utils import export_llama_to_onnx
        onnx_file_name = export_llama_to_onnx(model, model.config, torch.float32, args)

    if args.lod:
        from layersp import lod_cal
        lod_cal(args, model, tokenizer)

    if args.tasks is not None:
        from utils import eval_ppl

        eval_tasks = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
        print(f"Selected Tasks: {eval_tasks}")

        if "wikitext" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("wikitext"))
            eval_ppl(model, tokenizer, "wikitext")
        if "ptb" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("ptb"))
            eval_ppl(model, tokenizer, "ptb")
        if "c4" in eval_tasks:
            eval_tasks.pop(eval_tasks.index("c4"))
            eval_ppl(model, tokenizer, "c4")
        if len(eval_tasks) > 0:
            eval_zero(args, model, tokenizer, eval_tasks)
