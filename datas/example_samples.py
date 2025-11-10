import random
import time

# random.seed(time.time())
# random.seed(993)

import torch

from datasets import load_dataset


def get_arc_easy(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/ai2_arc", "ARC-Easy", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "query:" + traindata[i]["question"] + "|".join(traindata[i]["choices"]["text"])
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_arc_challenge(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "query:" + traindata[i]["question"] + "choices:" + "|".join(traindata[i]["choices"]["text"])
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_openbookqa(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/openbookqa", "main", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "query:" + traindata[i]["question_stem"] + "choices:" + "|".join(traindata[i]["choices"]["text"])
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_winogrande(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "allenai/winogrande", "winogrande_xs", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = "sentence: " + traindata[i]["sentence"] + "option1: " + traindata[i]["option1"] + "option2: "\
                   + traindata[i]["option2"] + "answer: " + traindata[i]["answer"]
            tokenized_sample = tokenizer(text, padding='max_length', max_length=seq_len, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_hellaswag(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "Rowan/hellaswag", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            ctx = traindata[i]["ctx_a"] + " " + traindata[i]["ctx_b"].capitalize()
            tokenized_sample = tokenizer(traindata[i]["activity_label"] + ": " + ctx, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_boolq(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "google/boolq", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = f"{traindata[i]['passage']}\nQuestion: {traindata[i]['question']}?"
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_piqa(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        "ybisk/piqa", split='train', download_mode="reuse_cache_if_exists"
    )
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]["goal"] + "choices:1." + traindata[i]["sol1"] + "2." + traindata[i]["sol2"]
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
        download_mode="reuse_cache_if_exists"
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train', download_mode="reuse_cache_if_exists"
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_ptb(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'ptb_text_only', 'penn_treebank', split='train', download_mode="reuse_cache_if_exists"
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['sentence'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_wikitext2(tokenizer, n_samples, seq_len):
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split='train',
                             download_mode="reuse_cache_if_exists")

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_alpaca_cleaned(tokenizer, n_samples, seq_len):
    traindata = load_dataset("yahma/alpaca-cleaned", split='train', download_mode="reuse_cache_if_exists")

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['instruction'] + traindata[i]['input'] + traindata[i]['output']
            tokenized_sample = tokenizer(text, return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_examples(dataset, tokenizer, n_samples, seq_len=128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == "wikitext":
        return get_wikitext2(tokenizer, n_samples, seq_len)
    elif dataset == "ptb":
        return get_ptb(tokenizer, n_samples, seq_len)
    elif "tune":
        return get_alpaca_cleaned(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError
