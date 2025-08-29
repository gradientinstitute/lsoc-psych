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


def process(hf_name, filename, HF_KEY, dataset_pile, device, batch_size=16):
    # Get the losses of a single model.
    print(f"Loading {hf_name} with default settings.")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        token=HF_KEY,
        trust_remote_code=True,
    )
    if not tokenizer.pad_token:
        # we'll be using some padding in the batching
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, revision=None,
        device_map="auto",  # {"": "mps"},  # "auto",
        torch_dtype="auto", # or torch.float16 potentially
        token=HF_KEY,
        trust_remote_code=True,
    )
    model.eval()  # put in evaluate mode
    # model = model.to(device)
    device = next(model.parameters()).device
    success = True
    try:

        # Process standard dataset
        print(f"Running {hf_name}...")
        output = process_subsets(
            model,
            tokenizer,
            dataset_pile,
            device=device,
            batch_size=batch_size,
        )

        print(f"Saving {filename}")
        with open(filename, "wb") as f:
            pickle.dump(output, f)

    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
        traceback.print_exc()
        success = False    
    # model = model.cpu()  # if we havent used device_map auto
    cleanup_model(model, hf_name)
    return success


def process_subsets(model, tokenizer, dataset, device, batch_size):

    output = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch["text"],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"  # ensors
        )

        # Compute losses
        input_dev = inputs["input_ids"].to(device)
        losses = compute_token_loss(model, input_dev)

        # Package output
        inputs_np = inputs["input_ids"].numpy()
        cuts = inputs["attention_mask"].numpy().sum(axis=1)
        for input, loss, cut, subset, idx in zip(inputs_np, losses, cuts, batch['subset'], batch['idx']):
            output.append(dict(
                tokens=tokenizer.batch_decode(input)[:cut],
                losses=loss[:cut],
                subset=subset,
                idx = idx,
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


def load_data(datasets=None):

    ## PRELOAD DATASETS
    dataset_pile = load_dataset("timaeus/pile_subsets_mini", split="train")
    dataset_pile = dataset_pile.add_column(
    "idx", list(range(len(dataset_pile))))
    dataset_pile = filter_subsets(
        dataset_pile, include=datasets, exclude=[])

    dataset_dm_math = load_dataset("timaeus/dm_mathematics_mini", split="train")

    return dict(
        dataset_pile=dataset_pile,
        dataset_dm_math=dataset_dm_math,
    )


def compute_token_loss(model, input_ids):
    """Compute token probabilities for a given input."""
    with torch.inference_mode():  # no_grad():
        outputs = model(input_ids)

    # output.logits is [batch_size * context length * nvocab]
    # The logits are for the *next token given the tokens thus far
    logits = outputs.logits[:, :-1, :]  # Exclude the last position
    targets = input_ids[:, 1:]  # Exclude the first position

    # Get probabilities using softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Move to CPU and convert to NumPy
    log_probs_np = log_probs.to(torch.float32).cpu().numpy()
    targets_np = targets.cpu().numpy()

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



def process_regular_dataset(model, dataset, batch_size):
    """Process regular dataset with a specific model checkpoint/revision."""

    # Initialize the result dictionary with dataset-indexed structure
    results = {}

    # Some subsets have shorter samples than others
    subsets = set(dataset['subset'])

    for subset in subsets:
        select = dataset.filter(lambda x: x["subset"] == subset)
        select = select.take(100)  # do a fixed number if we have them
        losses = []

        # If the inputs are different lengths, we can't batch them:
        # Emulate the old batching
        for batch in tqdm(select.batch(batch_size=batch_size),
                          total=select.num_rows // batch_size,
                          desc=subset):

            for input_ids in batch["input_ids"]:
                # they're ragged now so process individually
                input_ids = torch.tensor(input_ids).to(model.device)
                token_losses = compute_token_loss(model, input_ids[None, :])[0]
                losses.append(token_losses)

        # AL: convert to list of dicts to match Liam
        results[subset] = { #interleave_dict({
            "context_id": select["idx"],
            "tokens": select["tokens"],
            "loss": losses,
        } #)
    return results


def filter_subsets(dataset, include, exclude, field="subset"):
    # Filter the subsets in use
    subsets = set(dataset[field])
    for e in exclude:
        subsets.discard(e)

    if include is not None:
        subsets = subsets.intersection(include)

    return dataset.filter(lambda x: x[field] in subsets)


def process_dm_mathematics(model, dataset, tokenizer, batch_size):
    """Process dm_mathematics dataset with zero-shot and few-shot approaches."""
    # categories can come from set(dataset['module'])
    # Batch size is unused, but kept for API compatibility... actual batch size = 1
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
        input_ids = encoded["input_ids"].to(model.device)

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

        input_ids = torch.tensor([concatenated_tokens]).to(model.device)

        # Get token probabilities for concatenated text
        concat_token_loss = compute_token_loss(model, input_ids)[0]

        # Map concatenated token probabilities to original samples
        for mapping in token_mappings:
            orig_idx = mapping["orig_idx"]
            start_idx = mapping["start_idx"]
            end_idx = mapping["end_idx"]
            sample_probs = concat_token_loss[start_idx:end_idx]
            results[orig_idx]["loss"]["few-shot"] = sample_probs

    return {"dm_math_categories": results}


def make_token_mapper(tokenizer, field="text"):
    """Create a function to tokenize a dataset through map."""
    
    def fn_tokenize(example):
        text = example[field]
        input_ids = tokenizer.encode(text)
        tokens = tokenizer.batch_decode(input_ids)
        # assert "".join(tokens) == text
        # What if it doesn't? should we skip?
        return {
            "input_ids": input_ids,
            "tokens": tokens,
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



def cleanup_model(model, model_name):
    """Clean up model from memory and optionally from disk."""

    print(f"Cleaning up {model_name}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_disk_cache(model_name):
    print(f"Clearing {model_name} from cache")
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
