import torch
from datasets import load_dataset
from torch import nn
from tqdm import tqdm


def ppl_calculate(model, testenc):
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen): ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
                       :, (i * model.seqlen): ((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        if torch.isnan(neg_log_likelihood).item(): continue
        nlls.append(neg_log_likelihood)

    nsamples = len(nlls)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()


def eval_ppl(model, tokenizer, dataset):
    if dataset == "wikitext":
        testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    elif dataset == "ptb":
        testenc = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer("\n\n".join(testenc["sentence"]), return_tensors="pt")
    elif dataset == "c4":
        testenc = load_dataset(
            'allenai/c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    else:
        raise NotImplementedError

    ppl = ppl_calculate(model, testenc)
    print(f"{dataset} ppl: {ppl}")