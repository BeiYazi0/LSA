import gc
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 3142))

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29508'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'


class BaseModel:
    def __init__(self, ckpt_dir, fp):
        max_batch_size = 20
        max_seq_len = 2048
        seed = AICROWD_RUN_SEED
        model_parallel_size = None
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # assert model_args.vocab_size == tokenizer.n_words
        # if torch.cuda.is_bf16_supported():
        #     torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        # else:
        #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16 if fp else torch.float32,
            device_map='auto',
            # load_in_8bit=True
        )
        model.eval()
        print(model.hf_device_map)
        self.model = model
        self.tokenizer = tokenizer
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

    def predict(self, instruction: str, input: str, temperature: float = 0.6,
                top_p: float = 0.9, top_k: int = 50, repetition_penalty: float = 1.3,
                max_gen_len: Optional[int] = None,
                logprobs: bool = False, ) -> str:

        if max_gen_len is None:
            max_gen_len = 512

        # 得到输出
        dialog = []
        text_1 = {"role": "system", "content": instruction}
        text_2 = {"role": "user", "content": input}
        dialog.append(text_1)
        dialog.append(text_2)
        prompt_token = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to('cuda')

        prompt_tokens = []
        prompt_tokens.append(prompt_token)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        generation_tokens = self.model.generate(
            prompt_token,
            max_new_tokens=max_gen_len,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        generation_token = generation_tokens[0][prompt_token.shape[-1]:]
        response = self.tokenizer.decode(generation_token, skip_special_tokens=True)
        return response