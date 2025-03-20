import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs
from datasets import load_dataset
import pickle
import gc
import os
import shutil
from tqdm import tqdm


def setup_model(model_name, revision):
    """Load model and tokenizer with specified revision."""
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                revision=revision, 
                                                torch_dtype=torch.float16,
                                                device_map="auto") # TODO: check with Al whether this is fine [it was in my notebook]
    model.eval()  # switch to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def cleanup_model(model, clear_disk=False, model_name=None):
    """Clean up model from memory and optionally from disk."""
    model = model.cpu()
    del model # TODO: check with Al, I don't think we want to delete the tokenizer each time? 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Optionally clear from disk
    if clear_disk and model_name:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        if os.path.exists(os.path.join(cache_dir, "models", model_name.split("/")[-1])):
            shutil.rmtree(os.path.join(cache_dir, "models", model_name.split("/")[-1]))


def compute_token_probs(model, input_ids, device):
    """Compute token probabilities for a given input."""
    with torch.no_grad():
        outputs = model(input_ids)
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

    return token_probs


def process_regular_dataset(dataset, tokenizer, model_name, revision, batch_size=8, max_context_length=512, clear_disk=False):
    """Process regular dataset with a specific model checkpoint/revision."""
    model, device = setup_model(model_name, revision)
    # Tokenize the dataset
    def fn_tokenize(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=max_context_length)
    tokenized_data = dataset.map(fn_tokenize, batched=True)


    # Initialize the result dictionary with dataset-indexed structure
    results = {}

    # Process the dataset in batches
    for i in range(0, len(tokenized_data), batch_size):
        batch = tokenized_data[i:i+batch_size]

        for j in tqdm(list(range(len(batch["text"])))):
            # Get the current example
            text = batch["text"][j]
            subset = batch["subset"][j]
            context_idx = batch["idx"][j] if "idx" in batch else j
            input_ids = torch.tensor(batch["input_ids"][j]).unsqueeze(0).to(device)
            
            # Decode tokens for this sample
            tokens = [tokenizer.decode([token_id]) for token_id in batch["input_ids"][j]]

            # If this subset hasn't been seen yet, initialize its entry in results
            if subset not in results:
                results[subset] = []

            # Compute token probabilities
            token_probs = compute_token_probs(model, input_ids, device)
            
            # Add the structured sample data
            results[subset].append({
                "context_idx": context_idx,
                "tokens": tokens,
                "loss": token_probs
            })

    # Clean up
    cleanup_model(model, clear_disk, model_name)
    
    return results


def process_dm_mathematics(dataset, tokenizer, model_name, revision, batch_size=8, clear_disk=False, max_context_length=512):
    """Process dm_mathematics dataset with both zero-shot and few-shot approaches."""
    model, device = setup_model(model_name, revision)
    
    # Initialize results dictionary with the specified structure
    results = {
        "dm_mathematics": []
    }
    
    # Group samples by category (module and template)
    samples_by_category = {}
    for idx, sample in enumerate(dataset):
        module = sample.get("module", "unknown")
        template = sample.get("template", "unknown")
        category = f"{module} - {template}"
        
        if category not in samples_by_category:
            samples_by_category[category] = []
        
        samples_by_category[category].append({
            "idx": idx,
            "text": sample["text"],
            "module": module,
            "template": template,
            "category": category
        })
    
    # Initialize the results structure with all samples
    for idx, sample in enumerate(dataset):
        # Tokenize the text
        tokens = tokenizer.encode(sample["text"], add_special_tokens=False)
        tokens = [tokenizer.decode([token_id]) for token_id in tokens]
        
        results["dm_mathematics"].append({
            "context_idx": idx,
            "tokens": tokens,
            "category": f"{sample.get('module', 'unknown')} - {sample.get('template', 'unknown')}",
            "loss": {
                "zero-shot": None,
                "few-shot": None
            }
        })
    
    # 1. Process zero-shot samples
    print("Processing zero-shot samples...")
    
    # Process each sample individually
    for idx, sample in enumerate(tqdm(dataset)):
        # Tokenize
        encoded = tokenizer(sample["text"], padding=False, truncation=True, 
                           max_length=max_context_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        
        # Compute token probabilities
        token_probs = compute_token_probs(model, input_ids, device)
        
        # Store in results
        results["dm_mathematics"][idx]["loss"]["zero-shot"] = token_probs
    
    # 2. Process few-shot samples
    print("Processing few-shot samples...")
    
    # Create concatenated texts for each category
    for category, samples in samples_by_category.items():
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
            token_mappings.append({
                "orig_idx": sample["idx"],
                "start_idx": token_offset,
                "end_idx": token_offset + len(sample_tokens) - 1
            })
            
            # Update concatenated text and offset
            concatenated_text += sample["text"]
            token_offset += len(sample_tokens)
        
        # Tokenize concatenated text
        encoded = tokenizer(concatenated_text, padding=False, truncation=True, 
                           max_length=max_context_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        
        # Get token probabilities for concatenated text
        concat_token_probs = compute_token_probs(model, input_ids, device)
        
        # Map concatenated token probabilities to original samples
        for mapping in token_mappings:
            orig_idx = mapping["orig_idx"]
            start_idx = mapping["start_idx"]
            end_idx = mapping["end_idx"]
            
            # Extract token probabilities for this sample
            # Account for shifted indices from the model output (exclude first token)
            sample_start = max(0, start_idx - 1)  # -1 to account for logits shift
            sample_end = min(len(concat_token_probs) - 1, end_idx - 1)
            
            sample_probs = concat_token_probs[sample_start:sample_end+1]
            
            # Store in results
            results["dm_mathematics"][orig_idx]["loss"]["few-shot"] = sample_probs
    
    # Clean up
    cleanup_model(model, clear_disk, model_name)
    
    return results


def main():
    ## CONFIG
    MODEL_SIZES = ['14m', '30m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']
    MODEL_SIZES.reverse()  # Fixed the reverse() call
    MODEL_CHECKPOINTS = None # list of integers
    DATASETS = None
    BATCH_SIZE = 512
    MAX_CONTEXT_LENGTH = 512  # Maximum context length for tokenization
    TEST = False
    experiment_name = "EXP000"

    REVS_SUBSET = ['step{}'.format(i) for i in MODEL_CHECKPOINTS] if MODEL_CHECKPOINTS is not None else None

    os.makedirs("experiments", exist_ok=True)
    experiment_folder = os.path.join("experiments", experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)
        
    ## LOAD DATASETS
    # Load standard dataset (train split), persists across models
    dataset_pile = load_dataset("timaeus/pile_subsets_mini", split="train")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b") # all models have the same tokenizer
    
    # Remove all data with subset="dm_mathematics"
    dataset_pile = dataset_pile.filter(lambda x: x["subset"] != "dm_mathematics")

    # Get those subsets that are in the dataset and in DATASETS if not None
    subsets = set(dataset_pile["subset"])
    if DATASETS is not None:
        subsets = subsets.intersection(DATASETS)
    # Filter the dataset to only include those subsets
    dataset_pile = dataset_pile.filter(lambda x: x["subset"] in subsets)

    # Add index to dataset for tracking context_idx
    dataset_pile = dataset_pile.add_column("idx", list(range(len(dataset_pile))))

    # Load special dm_mathematics_mini dataset
    dataset_dm_math = load_dataset("timaeus/dm_mathematics_mini", split="train")
    
    # Loop over model sizes
    for model_size in MODEL_SIZES:
        model_name = f"EleutherAI/pythia-{model_size}"
        model_folder = os.path.join(experiment_folder, model_size)
        os.makedirs(model_folder, exist_ok=True)
        
        refs = list_repo_refs(model_name)
        branches = [branch.ref.replace("refs/heads/", "") for branch in refs.branches]
        available_revs = [s for s in branches if "step" in s]

        # get intersection of model_checkpoints and available_steps, but if None then all available
        revs = [s for s in available_revs if s in REVS_SUBSET] if REVS_SUBSET is not None else available_revs
        revs = revs[::-1]

        if TEST:
            revs = revs[::10] 
            clear_disk = False  # DOn't set for debugging, do turn on for batch
        else:
            clear_disk = True  # Clear disk in production mode
            
        # Loop over model checkpoints with tqdm
        for revision in tqdm(revs, desc=f"Processing {model_size}"):
            step = int(revision.split("step")[-1])
            # format filename to be of form 000300, 010000, etc.
            filename = os.path.join(model_folder, f"step{step:06d}.pkl")
            
            # Process standard dataset
            standard_results = process_regular_dataset(dataset_pile, tokenizer, model_name, revision, 
                                                      clear_disk=False,  # Don't clear disk yet
                                                      batch_size=BATCH_SIZE,
                                                      max_context_length=MAX_CONTEXT_LENGTH)
            
            # Process dm_mathematics dataset separately
            dm_math_results = process_dm_mathematics(dataset_dm_math, tokenizer, model_name, revision, 
                                                   clear_disk=clear_disk,  # Now clear disk if needed
                                                   batch_size=BATCH_SIZE,
                                                   max_context_length=MAX_CONTEXT_LENGTH)
            
            # Combine results and save
            combined_results = {**standard_results, **dm_math_results}
            
            with open(filename, "wb") as f:
                pickle.dump(combined_results, f)



if __name__ == "__main__":
    main()