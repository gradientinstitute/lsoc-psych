import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs
from datasets import load_dataset
import numpy as np
import pickle
import gc
import fire  # need to install
import os
import shutil
from tqdm import tqdm
import time


def setup_model(model_name, revision, max_retries=100, rest=10):
    """Load model and tokenizer with specified revision."""

    for _ in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, torch_dtype=torch.float16,
                device_map="auto"  # contradicts manual device handling below
            )
            break  # I guess it worked
        except:
            time.sleep(rest)
    else:
        assert False, "Couldnt load model!"

    model.eval()  # switch to evaluation mode
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = next(iter(model.hf_device_map.values()))
    # model.to(device)
    return model, device


def cleanup_model(model, clear_disk=False, model_name=None):
    """Clean up model from memory and optionally from disk."""
    print(f"Cleaning up {model_name}")
    model = model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optionally clear from disk
    if clear_disk and model_name:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        target = "models--" + model_name.replace("/", "--")
        cache_path = os.path.join(cache_dir, target)
        # WARNING - do not run multiple workers on the same machine without fixing this

        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        else:
            print("WARNING: model cache not found.")


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
        tokens = tokenizer.encode(sample["text"], add_special_tokens=False)
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
            padding=False,
            truncation=True,
            max_length=max_context_length,
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
        token_mappings = []  # To track original sample tokens in concatenated context
        token_offset = 0

        for i, sample in enumerate(samples):
            # Add newline between samples
            if i > 0:
                concatenated_text += "\n"
                token_offset += 1  # Account for newline token

            # Get tokens for this sample
            sample_tokens = tokenizer.encode(sample["text"], add_special_tokens=False)

            # Store mapping information
            token_mappings.append(
                {
                    "orig_idx": sample["idx"],
                    "start_idx": token_offset,
                    "end_idx": token_offset + len(sample_tokens),  # I don't think there sould be a -1 here
                }
            )

            # Update concatenated text and offset
            concatenated_text += sample["text"]
            token_offset += len(sample_tokens)


        # Tokenize concatenated text
        encoded = tokenizer(
            concatenated_text,
            padding=False,
            truncation=True,
            max_length=max_context_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)

        # Get token probabilities for concatenated text
        concat_token_loss = compute_token_loss(model, input_ids)[0]

        # Map concatenated token probabilities to original samples
        for mapping in token_mappings:
            orig_idx = mapping["orig_idx"]
            start_idx = mapping["start_idx"]
            end_idx = mapping["end_idx"]

            # Extract token probabilities for this sample
            # Account for shifted indices from the model output (exclude first token)
            # Actually, don't. Al is now padding
            # sample_start = max(0, start_idx - 1)  # -1 to account for logits shift
            # sample_end = min(len(concat_token_loss) - 1, end_idx - 1)
            # sample_probs = concat_token_loss[sample_start : sample_end]
            sample_probs = concat_token_loss[start_idx:end_idx]

            # Store in results
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


def filter_revisions(model_name, model_checkpoints=None):
    # Get available model checkpoint steps
    refs = list_repo_refs(model_name)
    branches = [branch.ref.replace("refs/heads/", "") for branch in refs.branches]
    revs = [s for s in branches[::-1] if "step" in s]

    if model_checkpoints is not None:
        checkpoint_str = ["step{}".format(i) for i in model_checkpoints]
        revs = [s for s in revs if s in checkpoint_str]

    return revs


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


def main(worker_id=0, batch_size=64, debug=0):

    ## CONFIGURE
    DEBUG = (debug > 0)  # Set for a minimal test configuration

    work_split = [
        ["14m", "30m", "70m", "160m", "410m"],  # 512
        ["1b", "1.4b"],  # 512
        ["2.8b"],  # 256
        ["6.9b"],  # 128 max batch size..
    ]
    default_batch = [512, 512, 256, 128, 32]  # the last one is for CPU...

    # How big is big enough?
    # trying github on 410m with batch size
    # 8: 22 seconds
    # 16: 17 seconds
    # 32: 18 seconds
    # 64: 18 seconds
    # 256: 18 seconds

    if batch_size is None:
        batch_size = default_batch[worker_id]

    if worker_id == -1:
        MODEL_SIZES = ["14m", "30m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
    else:
        MODEL_SIZES = work_split[worker_id]

    print(f"I am worker {worker_id} and will do tasks {str(MODEL_SIZES)}")

    MODEL_SIZES.reverse()  # Process largest to smallest
    MODEL_CHECKPOINTS = None  # All
    CLEAR_DISK = True  # Avoid caching Terabytes of model checkpoints
    DATASETS = None  # All
    BATCH_SIZE = batch_size  # this is distinct from subset size - tune to fit in memory
    # TODO: optimal batch size is a function of model size?
    MAX_CONTEXT_LENGTH = 512  # Maximum context length for tokenization
    EXPERIMENT_NAME = "EXP000"

    if DEBUG:
        print("I am in DEBUG MODE and will do a demonstration")
        MODEL_CHECKPOINTS = [10000, 143000]
        MODEL_SIZES = ["14m", "160m"]  # overwritten, not checked
        CLEAR_DISK = True  # False
        DATASETS = ["wikipedia_en", "enron_emails"]
        EXPERIMENT_NAME = "debug"

    # Make path for results
    os.makedirs("experiments", exist_ok=True)
    experiment_folder = os.path.join("experiments", EXPERIMENT_NAME)
    os.makedirs(experiment_folder, exist_ok=True)

    ## LOAD DATASETS

    dataset_pile = load_dataset("timaeus/pile_subsets_mini", split="train")
    dataset_pile = dataset_pile.add_column(
        "idx", list(range(len(dataset_pile))))
    dataset_pile = filter_subsets(
        dataset_pile, include=DATASETS, exclude=["dm_mathematics"])
    # if DEBUG:
    #     dataset_pile = dataset_pile.take(50) # really mini job :)

    dataset_dm_math = load_dataset("timaeus/dm_mathematics_mini", split="train")

    ## TOKENIZE (all Pythia models have the same tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-6.9b"
    )
    fn_tokenize = make_token_mapper(
            tokenizer,
            padding=False,
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH,
    )
    dataset_pile = dataset_pile.map(fn_tokenize, batched=False)

    # This one doesn't need it...
    # dataset_dm_math = dataset_dm_math.map(fn_tokenize, batched=False)

    # job_num = 0
    # Loop over models
    for model_size in MODEL_SIZES:
        model_name = f"EleutherAI/pythia-{model_size}"
        model_folder = os.path.join(experiment_folder, model_size)
        os.makedirs(model_folder, exist_ok=True)
        revs = filter_revisions(model_name, MODEL_CHECKPOINTS)

        # Loop over model checkpoints with tqdm
        for revision in revs:

            # job_num += 1
            # if job_num % n_workers != worker_id:
            #     continue

            print(f"Processing {model_size} @ {revision}")
            step = int(revision.split("step")[-1])
            # format filename to be of form 000300, 010000, etc.
            filename = os.path.join(model_folder, f"step{step:06d}.pkl")

            if os.path.exists(filename):
                print(f"Pythia-{model_size}-{revision}")
                print("How convenient, its already here! Skipping")
                continue

            # Set up model (so it can be used by both process functions)
            model, device = setup_model(model_name, revision)

            # Process standard dataset
            standard_results = process_regular_dataset(
                model,
                device,
                dataset_pile,
                batch_size=BATCH_SIZE,
            )

            # Process dm_mathematics dataset separately
            dm_math_results = process_dm_mathematics(
                model,
                device,
                dataset_dm_math,
                tokenizer,
                batch_size=BATCH_SIZE,
            )

            # dm_math_results = {}  # skip for now
            cleanup_model(model, CLEAR_DISK, model_name)

            # Combine results and save
            combined_results = {**standard_results, **dm_math_results}

            # r = combined_results['wikipedia_en']
            # list(zip(r["tokens"][5], r["loss"][5]))

            print(f"Saving {filename}")
            with open(filename, "wb") as f:
                pickle.dump(combined_results, f)


if __name__ == "__main__":
    fire.Fire(main)
