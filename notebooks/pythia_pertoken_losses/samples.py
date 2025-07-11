"""Extract representative samples from huggingface"""
import os
import numpy as np
from datasets import load_dataset, Dataset
# from tqdm.notebook import tqdm
from tqdm import tqdm
from google.colab import userdata  # or ask Al for local mockup
from transformers import AutoTokenizer


def extract_task(task, quota, seed=42):
    """
    Extract samples from the "full" dataset.
    """

    # megabytes per dataset
    MB = 1e6  # not MiB which is 1024**2
    g = 0  # global index of sample
    np.random.seed(seed)
    if task == "pile-1m":
        # This was for a different experiment and has its own format
        dataset_name = "timaeus/dsir-pile-1m-2"
        field = 'contents'
    else:
        dataset_name = "timaeus/pile-" + task
        field = "text"

    sample_dir = "samples"
    print(f"Loading {task}...", flush=True)
    dataset = load_dataset(dataset_name)["train"]  #, token=token)["train"]

    # Sample at random
    n = dataset.num_rows
    using = np.random.choice(n, size=quota, replace=False)
    sive = np.zeros(n, dtype=bool)
    sive[using] = True

    # lucky for us, these datasets were made with a multiple of 100
    batch_size = 100
    n_batch = len(dataset)//batch_size
    sive = sive.reshape((n_batch, 100))

    records = []

    # Traverse the dataset
    for batch, extract in tqdm(
            zip(dataset.iter(batch_size=batch_size), sive),
            total=n_batch, desc=f"Reading {task}"):

        if not extract.any():
            continue

        for use, text in zip(extract, batch[field]):
            if use:
                records.append(text)

    return records


def find_all(main_string, substring):
    start = 0
    indices = []
    n = len(substring)

    while start < len(main_string):
        pos = main_string.find(substring, start)
        if pos == -1:
            break
        indices.append(pos + n)
        start = pos + 1  # Move past the current occurrence

    return indices


def get_snippet(record, n_tokens, tokenizer):
    """From a (presumably long) record, extract a snippet."""
    # could be heavily optimised, but in a rush

    # Break on blank lines!
    break_on = "\n\n"
    if record.count(break_on) < 3:
        break_on = "\n"  # fall back to just newlines then....

    breaks = find_all(record, break_on)
    breaks.append(0)  # we *may occasionally start at the beginning too?

    while(len(breaks)):
        # choose one at random
        start = int(np.random.choice(breaks))
        breaks.remove(start)  # don't try again

        # tokenize to see how long it is
        # I know I could be smarter about this but it doesn't take long
        extract = record[start:]
        extract = extract.strip("\n")
        tokens = tokenizer.encode(extract)
        tokens = tokens[:n_tokens]
        text = tokenizer.decode(tokens)

        # Now reject if its not long enough
        if len(tokens) < n_tokens:
            continue

        break
    else:
        print(".", end="")
        return None  # No luck...

    return text


def name_to_seed(task):
    """A simple but consistent string hashing function"""
    # no need to be cryptographic
    h = 0
    for char in task:
        h += ord(char)
    return h


if __name__ == "__main__":

    tasks = [
        "wikipedia_en",
        "arxiv",
        "github",
        "pile-cc",
        "dm_mathematics",  # we already have dm_mathematics_mini
        "enron_emails",
        "freelaw",
        "hackernews",
        "nih_exporter",
        "pubmed_abstracts",
        "pubmed_central",
        "stackexchange",
        "uspto_backgrounds",
        # "pile-1m",
    ]

    extracts = {}

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    quota = 2048  # over-sample in case we end up with some duds
    keep = 512  # how many to extract

    tokens = {t: 512 for t in tasks}
    for task in ["enron_emails", "nih_exporter", "pubmed_abstracts"]:
        tokens[task] = 100  # These are short documents.

    sample_dir = "samples"
    os.makedirs(sample_dir, exist_ok=True)

    for task in tasks:
        n_tokens = tokens[task]

        fname = f"{sample_dir}/{task}.txt"
        if os.path.exists(fname):
            continue

        records = extract_task(task, quota)
        np.random.seed(name_to_seed(task))

        extract = []
        for record in records:
            snip = get_snippet(record, n_tokens, tokenizer)
            if snip is not None:
                extract.append(snip)

        if not len(extract) > keep:
            print("Needs intervention!")
            import smart_embed
            smart_embed.embed(locals(), globals())

        extract = extract[:keep]
        extracts[task] = extract

        # Log the records
        with open(fname, "w") as f:
            for e in extract:
                f.write(e + "\n\n")
        print(f"Saved {fname}")

    # Go manual from here
    import smart_embed
    smart_embed.embed(locals(), globals())


    # Build the dataset incrementally,
    # interleaving the records so we get a nice preview

    d_text = []
    d_subset = []

    # Do stratified (easier to do flat)
    for i in range(keep):
        for task in tasks:
            if i < len(extracts[task]):
                d_text.append(extracts[task][i])
                d_subset.append(task)

    # d_text.extend(extract)
    # d_subset.extend([task]*len(extract))


    data = dict(
      text=d_text,
      subset=d_subset,
    )

    HF_TOKEN = userdata.get("HF_API_KEY")  # do we need tokens

    # Now save locally
    dataset = Dataset.from_dict(data)
    repo_name = "timaeus/lsoc_subsets_mini"
    dataset.push_to_hub(
      repo_id=repo_name,
      private=False,
      token=HF_TOKEN
    )
