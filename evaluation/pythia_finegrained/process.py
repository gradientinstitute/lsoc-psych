# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.

import pickle
import fire  # need to install
import os
import json
from token_loss import *


def main(worker_id=-1, gpu_id=-1, debug=False, batch_size=32):
    if gpu_id < 0:
        gpu_id = worker_id  # usually they should be the same

    if not os.path.exists("jobs.json"):
        print("Run plan_jobs.py first")
        return
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        print("I will use CPU")
    else:
        print(f"I will use GPU {gpu_id} of {n_gpu}")

    if debug:
        print("I am in debug MODE and will do a demonstration")
        jobs = [['160m', 'EleutherAI/pythia-160m', 'step143000', 'experiments/debug/160m/step143000.pkl']]
        CLEAR_DISK = False  # Avoid caching Terabytes of model checkpoints
        DATASETS = ["enron_emails"]

    else:
        with open("jobs.json", "r") as f:
            all_jobs = json.load(f)

        n_workers = max([j[0] for j in all_jobs]) + 1
        range_str = f"Worker id must be in [0, {n_workers-1}]"
        assert 0 <= worker_id < n_workers, range_str
        print(f"I am worker {worker_id} of {n_workers}.")

        # filter jobs
        jobs = []
        for j in all_jobs:
            if j[0] == worker_id:
                jobs.append(j[1:])
        CLEAR_DISK = True  # Avoid caching Terabytes of model checkpoints
        DATASETS = None  # no filter or list, eg ["enron_emails"]

    print(f"Assigned jobs: {len(jobs)}. GO.")


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
    # dataset_dm_math = dataset_dm_math.map(fn_tokenize, batched=False)

    # Loop over models
    for model_size, model_name, revision, filename in jobs:

        if not debug and os.path.exists(filename):
            print(f"{filename} exists - skipping!")
            continue

        print(f"Processing {model_size} @ {revision}")

        # Make sure the folder exists
        dirname, _ = os.path.split(filename)
        os.makedirs(dirname, exist_ok=True)

        # Set up model (so it can be used by both process functions)
        model, device = setup_model(model_name, revision, gpu_id)

        # Process standard dataset
        standard_results = process_regular_dataset(
            model,
            device,
            dataset_pile,
            batch_size=batch_size,
        )

        # Process dm_mathematics dataset separately
        dm_math_results = process_dm_mathematics(
            model,
            device,
            dataset_dm_math,
            tokenizer,
            batch_size=batch_size,
        )

        # Combine results and save
        combined_results = {**standard_results, **dm_math_results}

        print(f"Saving {filename}")
        with open(filename, "wb") as f:
            pickle.dump(combined_results, f)

        cleanup_model(model, CLEAR_DISK, model_name, gpu_id)

    print("Jobs done!")


if __name__ == "__main__":
    fire.Fire(main)


