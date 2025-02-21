"""Extract representative samples from huggingface"""
import os
import numpy as np
from datasets import load_dataset
# from tqdm.notebook import tqdm
from tqdm import tqdm
from google.colab import userdata  # or ask Al for local mockup


def extract_task(task):
    """
    Extract samples from the "full" dataset.
    """
    token = userdata.get("HF_API_KEY")

    # Let' train a zstd dictionary!
    quota_mb = 25    # megabytes per dataset
    MB = 1e6  # not MiB which is 1024**2
    g = 0  # global index of sample
    np.random.seed(42)
    if task == "full":
        dataset_name = "timaeus/dsir-pile-1m-2"
        field = 'contents'
    else:
        dataset_name = "timaeus/pile-" + task
        field = "text"

    sample_dir = "samples"
    dataset = load_dataset(dataset_name, token=token)["train"]

    # Determine the size of the dataset
    batch_size = 100
    sizes = []
    n_batch = len(dataset)/batch_size
    for batch in tqdm(dataset.iter(batch_size=batch_size), total=n_batch, desc=f"Reading {task}"):
        for text in batch[field]:
            nbytes = len(text.encode('utf-8'))
            sizes.append(nbytes)

    # Shuffle and cut
    sizes = np.array(sizes)
    order = np.random.permutation(len(sizes))
    sizes_i = sizes[order]
    cut = np.argmin(np.abs(np.cumsum(sizes_i) - quota_mb * MB))
    cut = int(cut+1)
    todo = np.zeros(len(sizes), dtype=int)
    todo[order[:cut]] = 1  # save to "full)


    os.makedirs(f"{sample_dir}/{task}", exist_ok=True)

    # Now we know what to do - do it!
    i = 0  # Keep track of sample index
    for batch in tqdm(dataset.iter(batch_size=batch_size), total=len(dataset)/batch_size, desc=f"Writing {task}"):
        for text in batch[field]:
            if todo[i]:
                fname = f"{sample_dir}/{task}/{task}{i:06d}.txt"
                with open(fname, "w") as f:
                    f.write(text)
            i += 1


if __name__ == "__main__":
    tasks = [
        "arxiv",
        "pile-cc",
        "dm_mathematics",
        "enron_emails",
        "freelaw",
        "github",
        "hackernews",
        "nih_exporter",
        "pubmed_abstracts",
        "pubmed_central",
        "stackexchange",
        "uspto_backgrounds",
        "wikipedia_en",
    ]

    for task in tasks:
        extract_task(task)



def notmain():
    # effective sizes considering epochs
    # note sample size varies significantly between tasks
    pile_composition = {
        "arxiv": 120.71,
        "pile-cc": 243.87,
        "dm_mathematics": 16.63,
        "enron_emails": 1.89,
        "freelaw": 82.39,
        "github": 102.18,
        "hackernews": 8.38,
        "nih_exporter": 4.07,
        "pubmed_abstracts": 41.37,
        "pubmed_central": 193.86,
        "stackexchange": 69.14,
        "uspto_backgrounds": 49.19,
        "wikipedia_en": 20.54,
        # "(missing)books3": 162.61,
        # "(missing)openwebtext2": 134.80,
        # "(missing)gutenberg": 29.20,
        # "(missing)opensubtitles": 20.91,
        # "(missing)ubuntu_irc": 11.84,
        # "(missing)bookcorpus2": 10.15,
        # "(missing)europarl": 9.85,
        # "(missing)youtube_subtitles": 8.02,
        # "(missing)philpapers": 5.11
    }



    token = userdata.get("HF_API_KEY")

    # Let' train a zstd dictionary!
    train_size = 25  # megabytes for training set
    flat_size = 25    # megabytes per dataset
    MB = 1e6  # not MiB which is 1024**2
    norm  = sum(v for v in pile_composition.values())
    sample_dir = "samples"
    os.makedirs(sample_dir, exist_ok=True)
    g = 0  # global index of sample
    np.random.seed(42)

    os.makedirs(f"{sample_dir}/train", exist_ok=True)

    for task in pile_composition:
        os.makedirs(f"{sample_dir}/{task}", exist_ok=True)
        dataset_name = "timaeus/pile-" + task
        dataset = load_dataset(dataset_name, token=token)["train"]

        # Determine the size of the dataset
        batch_size = 100
        sizes = []
        n_batch = len(dataset)/batch_size
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=n_batch, desc=f"Reading {task}"):
            for text in batch['text']:
                nbytes = len(text.encode('utf-8'))
                sizes.append(nbytes)

        # Figure out the quota for the training set
        norm  = sum(v for v in pile_composition.values())
        quota_mb = pile_composition[task] * train_size / norm  # pile representation quota

        # Shuffle and cut
        sizes = np.array(sizes)
        order = np.random.permutation(len(sizes))
        sizes_i = sizes[order]
        cut = np.argmin(np.abs(np.cumsum(sizes_i) - quota_mb * MB))
        cut = int(cut+1)
        cut2 = np.argmin(np.abs(np.cumsum(sizes_i[cut:]) - flat_size * MB))
        cut2 = int(cut2 + cut + 1)
        todo = np.zeros(len(sizes), dtype=int)
        todo[order[:cut]] = 1  # save to train
        todo[order[cut:cut2]] = 2  # save to test

        # Now we know what to do - do it!
        i = 0  # Keep track of sample index
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=len(dataset)/batch_size, desc=f"Writing {task}"):
            for text in batch['text']:
                if todo[i]:
                    if todo[i] == 1:
                        # save to the training set
                        fname = f"{sample_dir}/train/{task}{i:06d}.txt"
                    elif todo[i] == 2:
                        # save to testing set
                        fname = f"{sample_dir}/{task}/{task}{i:06d}.txt"
                    with open(fname, "w") as f:
                        f.write(text)
                i += 1


