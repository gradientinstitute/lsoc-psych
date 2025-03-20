import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs
from datasets import load_dataset
import pickle
import gc
import os
import shutil
from tqdm import tqdm


def process(token_data, model_name, revision,
            batch_size=8, clear_disk=False, filename):
    """Process dataset with a specific model checkpoint/revision."""

    # Load model and tokenizer with the specified revision
    print(f"Loading model {model_name} with revision {revision}")
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model.eval()  # switch to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize the result dictionary
    results = {}

    # sample_inx == row of dataset

    # Process the dataset in batches
    for i in range(0, len(token_data), batch_size):
        batch = token_data[i:i+batch_size]

        for j in tqdm(list(range(len(batch["text"])))):
            # Get the current example
            text = batch["text"][j]
            subset = batch["subset"][j]
            input_ids = torch.tensor(batch["input_ids"][j]).unsqueeze(0).to(device)

            # If this subset hasn't been seen yet, initialize its entry in results
            if subset not in results:
                results[subset] = {
                    "texts": [],
                    "token_probs": []
                }

            # Add the text to the results
            # TODO: we want this to be reconstructed tokens (ie. a list)
            results[subset]["texts"].append(text) 

            # Get token probabilities
            with torch.no_grad():
                # TODO: breakpoint here recommended for checking

                outputs = model(input_ids)
                # TODO: check and make sure this is working
                logits = outputs.logits[:, :-1, :]  # Exclude the last position
                input_ids_shifted = input_ids[:, 1:]  # Exclude the first position

                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # Get the probability of each actual token
                batch_size, seq_len = input_ids_shifted.shape
                token_probs = []

                for k in range(seq_len):
                    token_id = input_ids_shifted[0, k].item()
                    token_prob = probs[0, k, token_id].item()
                    token_probs.append(token_prob)

                results[subset]["token_probs"].append(token_probs)


    # Save the results per model
    with open(filename, "wb") as f:
        pickle.dump(results, f)

    # Clear the model from memory
    model = model.cpu()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optionally clear from disk
    if clear_disk:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        if os.path.exists(os.path.join(cache_dir, "models", model_name.split("/")[-1])):
            shutil.rmtree(os.path.join(cache_dir, "models", model_name.split("/")[-1]))
            print(f"Removed model {model_name} from disk cache")

    print(f"Results saved to {output_path}")
    return results


def main():
    model_name = "EleutherAI/pythia-160m"
    refs = list_repo_refs(model_name)
    branches = [branch.ref.replace("refs/heads/", "") for branch in refs.branches]
    steps = [s for s in branches if "step" in s]
    steps = steps[::-1]
    steps = steps[::10]  # as a test, decimate
    clear_disk=False  # DOn't set for debugging, do turn on for batch
    # are we worried about running out of hard drive space on the machine

    # Load dataset (train split)
    # persists across models
    dataset = load_dataset("timaeus/lsoc_subsets_mini", split="train")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

    # Tokenize the dataset
    def fn_tokenize(examples):
        return tokenizer(examples["text"], padding=False, truncation=True)
    tokenized = dataset.map(fn_tokenize, batched=True)

    # here we will need to loop over this
    # TODO: make an output folder
    rev = refs[0]  # loop over this too
    filename = f"{model_name}-{rev}.pkl"
    process(tokenized, model_name, rev, clear_disk=clear_disk)


if __name__ == "__main__":
    main()

