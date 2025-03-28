# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.

import shutil
from tqdm import tqdm
import time
import gc
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np


def load_data()

    ## PRELOAD DATASETS
    dataset_pile = load_dataset("timaeus/pile_subsets_mini", split="train")
    dataset_pile = dataset_pile.add_column(
        "idx", list(range(len(dataset_pile))))
    dataset_pile = filter_subsets(
        dataset_pile, include=DATASETS, exclude=["dm_mathematics"])

    if debug:
        dataset_pile = dataset_pile.take(50) # really mini job :)

    dataset_dm_math = load_dataset("timaeus/dm_mathematics_mini", split="train")

    ## PRETOKENIZE (all Pythia models have the same tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-6.9b"
    )
    fn_tokenize = make_token_mapper(
            tokenizer,
            padding=False,
    )
    dataset_pile = dataset_pile.map(fn_tokenize, batched=False)
    dataset_dm_math = dataset_dm_math.map(fn_tokenize, batched=False)
    return dict(
        tokenizer=tokenizer,
        pile=dataset_pile,
        dm_math=dataset_dm_math,
    )


def compute_token_loss(model, input_ids):
    """Compute token probabilities for a given input."""
    with torch.no_grad():
        outputs = model(input_ids)

    # output.logits is [batch_size * context length * nvocab]
    # The logits are for the *next token given the tokens thus far
    logits = outputs.logits[:, :-1, :]  # Exclude the last position
    targets = input_ids[:, 1:]  # Exclude the first position

    # Get probabilities using softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Move to CPU and convert to NumPy
    log_probs_np = log_probs.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Use numpy advanced indexing
    batch_size, seq_len = targets_np.shape
    batch_indices = np.arange(batch_size)[:, None]
    seq_indices = np.arange(seq_len)[None, :]
    token_log_probs = log_probs_np[batch_indices, seq_indices, targets_np]

    # THere is no loss on the first token
    token_log_probs = np.hstack((np.zeros((batch_size, 1), np.float16), token_log_probs))
    return -token_log_probs


def process_regular_dataset(model, device, dataset,
                            batch_size=8):
    """Process regular dataset with a specific model checkpoint/revision."""

    # Initialize the result dictionary with dataset-indexed structure
    results = {}

    # Some subsets have shorter samples than others
    subsets = set(dataset['subset'])

    for subset in subsets:
        select = dataset.filter(lambda x: x["subset"] == subset)
        losses = []

        # Process the dataset in batches
        for batch in tqdm(select.batch(batch_size=batch_size),
                          total=select.num_rows // batch_size,
                          desc=subset):

            # Compute token losses batched
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            token_losses = compute_token_loss(model, input_ids)
            losses.extend(token_losses)

        # AL: convert to list of dicts to match Liam
        results[subset] = interleave_dict({
            "context_id": select["idx"],
            "tokens": select["tokens"],
            "loss": losses,
        })
    return results


def filter_subsets(dataset, include, exclude, field="subset"):
    # Filter the subsets in use
    subsets = set(dataset[field])
    for e in exclude:
        subsets.discard(e)

    if include is not None:
        subsets = subsets.intersection(include)

    return dataset.filter(lambda x: x[field] in subsets)


def process_dm_mathematics(model, device, dataset, tokenizer, batch_size=8,
                           max_context_length=512):
    """Process dm_mathematics dataset with zero-shot and few-shot approaches."""
    # categories can come from set(dataset['module'])
    print("Processing dm_mathematics...")
    results = []

    # Group samples by category (module and template)
    samples_by_category = {}
    for idx, sample in enumerate(dataset):
        module = sample.get("module", "unknown")
        template = sample.get("template", "unknown")
        category = f"{module} - {template}"

        if category not in samples_by_category:
            samples_by_category[category] = []

        samples_by_category[category].append(
            {
                "idx": idx,
                "text": sample["text"],
                "module": module,
                "template": template,
                "category": category,
            }
        )

    # Initialize the results structure with all samples
    for idx, sample in enumerate(dataset):
        # Tokenize the text
        tokens = tokenizer.encode(sample["text"]) # , add_special_tokens=False)
        tokens = [tokenizer.decode([token_id]) for token_id in tokens]

        results.append(
            {
                "context_idx": idx,
                "tokens": tokens,
                "category": f"{sample.get('module', 'unknown')} - {sample.get('template', 'unknown')}",
                "loss": {"zero-shot": None, "few-shot": None},
            }
        )

    # Process each sample individually
    for idx, sample in enumerate(tqdm(dataset, desc="DM_Mathematics zero-shot")):
        # Tokenize
        encoded = tokenizer(
            sample["text"],
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)

        # Compute token probabilities
        token_loss = compute_token_loss(model, input_ids)[0]

        # Store in results
        results[idx]["loss"]["zero-shot"] = token_loss

    # Create concatenated texts for each category
    for category, samples in tqdm(samples_by_category.items(),
                                  desc="DM_Mathematics few-shot"):
        # Skip processing if only one sample in category (no few-shot benefit)
        if len(samples) <= 1:
            continue

        concatenated_text = ""
        concatenated_tokens = []
        token_mappings = []  # To track original sample tokens in concatenated context
        token_offset = 0

        for i, sample in enumerate(samples):
            # Add newline between samples
            text = sample["text"] + "\n"  # add a newline to join
            sample_tokens = tokenizer.encode(text)

            # Store mapping information
            token_mappings.append(
                {
                    "orig_idx": sample["idx"],
                    "start_idx": token_offset,
                    "end_idx": token_offset + len(sample_tokens) - 1,
                    # -1 accounting for the added "\n"
                }
            )

            # Update concatenated text and offset
            concatenated_text += text
            concatenated_tokens.extend(sample_tokens)
            token_offset += len(sample_tokens)

        input_ids = torch.tensor([concatenated_tokens]).to(device)

        # Get token probabilities for concatenated text
        concat_token_loss = compute_token_loss(model, input_ids)[0]

        # Map concatenated token probabilities to original samples
        for mapping in token_mappings:
            orig_idx = mapping["orig_idx"]
            start_idx = mapping["start_idx"]
            end_idx = mapping["end_idx"]
            sample_probs = concat_token_loss[start_idx:end_idx]
            results[orig_idx]["loss"]["few-shot"] = sample_probs

    return {"dm_mathematics": results}


def make_token_mapper(tokenizer, field="text", **kwargs):
    """Create a function to tokenize a dataset through map."""
    def fn_tokenize(example):
        input_ids = tokenizer.encode(
            example[field],
        )

        # The sampled size is either 512 or 100
        clip = 512 if len(input_ids) > 250 else 100

        if input_ids[-1] == 535:
            # Handle the non-invertible \n\n coding
            input_ids[-1] = 187
            input_ids.append(187)

        while len(input_ids) < clip:
            input_ids.append(187)

        input_ids = input_ids[:clip]

        return {
            "input_ids": input_ids,
            "tokens": tokenizer.batch_decode(input_ids),
        }

    return fn_tokenize


def interleave_dict(x):
    y = []
    keys, values = zip(*x.items())
    for vals in zip(*values):
        y.append(dict(zip(keys, vals)))
    return y


def contiguous_dict(y):
    keys = list(y[0])
    z = {}
    for k in keys:
        z[k] = [e[k] for e in y]
    return z


# The sparse checkpoints from other experiments
SPARSE_CHK = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000,
    80000, 90000, 100000, 110000, 120000, 130000, 140000, 143000
]


def setup_model(model_name, revision, worker_id, max_retries=100, rest=10):
    """Load model and tokenizer with specified revision."""

    lock(worker_id)
    for _ in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, torch_dtype=torch.float16,
                #device_map="auto"  # contradicts manual device handling below
            )
            break  # I guess it worked
        except:
            time.sleep(rest)
    else:
        assert False, "Couldnt load model!"
    # device = next(iter(model.hf_device_map.values()))
    unlock(worker_id)

    model.eval()  # switch to evaluation mode
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{worker_id}")
    else:
        device = "cpu"

    model.to(device)
    return model, device


def cleanup_model(model, clear_disk=False, model_name=None, worker_id=0):
    """Clean up model from memory and optionally from disk."""

    print(f"Cleaning up {model_name}")
    model = model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optionally clear from disk - this only needs to succeed occasionally
    # or stochastically once before you run out of hard disk space
    if locked():
        print("Not cleaning because another worker is loading a model right now")
    elif clear_disk and model_name:
        lock(worker_id)
        print("Clearing model cache")
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        target = "models--" + model_name.replace("/", "--")
        cache_path = os.path.join(cache_dir, target)
        # WARNING - do not run multiple workers on the same machine without fixing this

        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
            except:
                print("WARNING: cache clear error")
        else:
            print("WARNING: model cache not found.")
        unlock(worker_id)


# Try to make sure you dont remove a model while its being downloaded
# TODO: If I wan't in a rush I'd use subprocess?
def lock(i):
    with open(f"lock.{i}", "w"):
        pass


MAX_LOCK = 8

def locked():
    for i in range(MAX_LOCK):
        if os.path.exists(f"lock.{i}"):
            return True
    return False

def unlock(i):
    os.remove(f"lock.{i}")


