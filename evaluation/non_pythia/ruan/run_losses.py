# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.

import threading
import os
import token_loss
from huggingface_hub import snapshot_download
import traceback
import json
import shutil
from pathlib import Path


def main():
    """Collect model losses."""

    print(os.getcwd())

    print("Loading model list")
    model_file = "model_list.json"  #sys.argv[1]
    with open(model_file, "r") as f:
        model_list = json.load(f)

    assert os.path.exists(model_file)
    HF_KEY = os.environ.get("HF_KEY")
    assert len(HF_KEY), "Set HF_KEY"

    # Configuration
    out_path = os.path.expanduser("~/data/ruan_losses")
    os.makedirs(out_path, exist_ok=True)
    
    # Load the data
    print("Loading truncated pile subsets")
    subsets = token_loss.load_pile_samples(HF_KEY=HF_KEY)

    # Check for previously finished models to skip
    active = []
    for hf_url in model_list:        
        filename = os.path.join(out_path, hf_url.replace("/", "_") + ".pkl")
        if os.path.exists(filename):
            print(f"{filename} exists - skipping!")
        else:
            active.append((hf_url, filename))
    
    # active = active[:1]  # just do the first model locally as a test

    # Truncate model list
    preload = True  # Get the next task asynchronously?

    for i in range(len(active)):

        load_ahead =  (i < len(active) - 1) & preload
        
        if load_ahead:
            next_model_name = active[i+1][0]
            download_thread = threading.Thread(
                target=pre_download, args=(next_model_name, HF_KEY),
            )
            print(f"Background download {next_model_name}")  
            download_thread.start()

        # Process current model
        hf_name, filename = active[i]
        success = token_loss.process(hf_name, filename, HF_KEY, subsets)

        # finished (successfully?)
        delete_hf_model(hf_name)
        if load_ahead:
            download_thread.join()

    print("All jobs completed!")


def pre_download(name, HF_KEY):
    # Cache the first model's files
    allow_patterns = [
        "*.json", "*.safetensors", "*.txt", "pytorch_model.bin",
        "tokenizer.json", "vocab.json", "merges.txt"
    ]
    snapshot_download(name, token=HF_KEY, allow_patterns=allow_patterns)


def delete_hf_model(model_name):
    # Keep the hard drive from not filling up...
    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    if cache_path.exists():
        print(f"Found and calling deletion of {cache_path}.")
        shutil.rmtree(cache_path)
        return True
    print(f"WARNING: couldn't find{cache_path} - deletion failed.")
    return False

if __name__ == "__main__":
    main()
