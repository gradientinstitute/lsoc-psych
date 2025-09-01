# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Tokenizer doesn't like other threading?
import shutil
from tqdm import tqdm
import time
import gc
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset, concatenate_datasets
import numpy as np
import pickle
import traceback

token_cache = {}


def process(hf_name, filename, HF_KEY, dataset_pile):
    # Get the losses of a single model.
    print(f"Loading {hf_name}.")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        token=HF_KEY,
        trust_remote_code=True,
    )
    # Padding doesn't work on some models

    # if not tokenizer.pad_token:
    #     # we'll be using some padding in the batching
    #     print("Warning - modifying the pad token")
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    # Device map auto can go larger than memory but its going to be very slow
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, revision=None,
        device_map="auto",  # {"": "mps"}, 
        torch_dtype="auto", # or torch.float16 potentially
        token=HF_KEY,
        trust_remote_code=True,
    )
    model.eval()  # put in evaluate mode

    success = True
    try:

        # Compute the losses
        output = process_subsets(
            model,
            tokenizer,
            dataset_pile,
            hf_name
        )

        # Only save if successful
        print(f"Job success: saving to {filename}")
        with open(filename, "wb") as f:
            pickle.dump(output, f)

    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        success = False

    cleanup_model(model, hf_name)
    return success


def process_subsets(model, tokenizer, dataset, desc):

    output = []

    # Abandon all hopes of batching - uses too much vram
    for row in tqdm(dataset, desc=desc):

        inputs = tokenizer(
            row["text"],
            padding=False,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"]

        losses = compute_token_loss(model, inputs)[0]

        # Package output
        output.append(dict(
            tokens=tokenizer.batch_decode(inputs[0]),
            losses=losses,
            subset=row["subset"],
            idx = row["idx"],
        ))
    
    return output


def load_pile_samples(
    take=100, max_ctx=1024, ref_model="openlm-research/open_llama_3b",
    HF_KEY=None):
    ## PRELOAD DATASETS
    pile = load_dataset("timaeus/pile_subsets_mini", split="train")
    pile = pile.add_column("idx", list(range(len(pile))))
    subsets = set(pile['subset'])
    all_selections = []

    # Filter the data down to the 1300 subsets we used in the project report
    for subset in subsets:
        select = pile.filter(lambda x: x["subset"] == subset)
        select = select.take(take)
        all_selections.append(select)

    subsets = concatenate_datasets(all_selections)

    # Limit to 1024 in GPT2? in Open_llama_3b? Which should be around 512 (big expansion)
    # Now we can trim the length
    # its already 512 tokens in gpt-neox
    # Soo many are being eliminated by having a context of 2048 and an encoded length of 2676
    # Tokenize wrt a reference model
    tokenizer = AutoTokenizer.from_pretrained(ref_model, token=HF_KEY, trust_remote_code=True)
    config = AutoConfig.from_pretrained(ref_model, token=HF_KEY, trust_remote_code=True)

    # only 17 exceed 1024, mainly due to whitespace oddities
    count = 0
    
    def truncate(example):
        nonlocal count
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_ctx:
            count += 1
            return {"text": tokenizer.decode(tokens[:max_ctx])}
        else:
            return {"text": text}

    subsets = subsets.map(truncate)
    print(f"{count} sequences over length {max_ctx} (of {len(subsets)})")
    return subsets


def compute_token_loss(model, input_ids):
    """Compute token probabilities for a given input."""

    device_type = next(iter(model.parameters())).device.type
    
    with torch.inference_mode():  # no_grad():
        input_dev = input_ids.to(device_type)
        outputs = model(input_dev)

    # output.logits is [batch_size * context length * nvocab]
    # The logits are for the *next token given the tokens thus far
    logits = outputs.logits[:, :-1, :]  # Exclude the last position
    targets = input_ids[:, 1:]  # Exclude the first position

    # Get probabilities using softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Move to CPU and convert to NumPy
    log_probs_np = log_probs.to(torch.float32).cpu().numpy()
    targets_np = targets.numpy()

    # Use numpy advanced indexing
    batch_size, seq_len = targets_np.shape
    batch_indices = np.arange(batch_size)[:, None]
    seq_indices = np.arange(seq_len)[None, :]
    token_log_probs = log_probs_np[batch_indices, seq_indices, targets_np]

    # THere is no loss on the first token
    token_log_probs = np.hstack(
        (np.zeros((batch_size, 1), token_log_probs.dtype),
         token_log_probs))
    return -token_log_probs
    

def filter_subsets(dataset, include, exclude, field="subset"):
    # Filter the subsets in use
    subsets = set(dataset[field])
    for e in exclude:
        subsets.discard(e)

    if include is not None:
        subsets = subsets.intersection(include)

    return dataset.filter(lambda x: x[field] in subsets)


def cleanup_model(model, model_name):
    """Clean up model from memory and optionally from disk."""

    print(f"Cleaning up {model_name}")
    del model  # No need to "send" to CPU first
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_disk_cache(model_name):
    print(f"Clearing {model_name} from cache")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    target = "models--" + model_name.replace("/", "--")
    cache_path = os.path.join(cache_dir, target)
    
    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
        except:
            print("WARNING: cache clear error")
    else:
        print("WARNING: model cache not found.")
