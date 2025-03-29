# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.
"""
Extract Eleuther's evals from the Pythia repo.

git clone git@github.com:EleutherAI/pythia.git
"""
pythia_path = "~/code/pythia"
output_path = "~/code/lsoc-psych/data/evals/eleuther"


import os
import re
import json
import pandas as pd


def main():
    path = os.path.expanduser(pythia_path)
    output = os.path.expanduser(output_path)

    os.makedirs(output, exist_ok=True)

    models = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1.4b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
    ]

    # Dictionary to store DataFrames
    model_dfs = {}

    for model in models:
        # Note - only loading the zero-shot evals
        model_path = f"{path}/evals/pythia-v1/{model}/zero-shot"

        # sort by step
        files = sorted(os.listdir(model_path), key=lambda x:
                       int(re.search(r'step(\d+)', x).group(1)))

        data = load_json_data(model_path, files)

        # Create DataFrame and store it in dictionary with model name as key
        metrics = extract_metrics(data, "acc")

        # Create DataFrame - note we're just getting accuracy
        # other metrics might be
        #  "likelihood_difference" (for all the crows_pairs)
        #  "ppl"  perplexity, eg for lambada_openai 
        df = pd.DataFrame(metrics, index=files)

        # Clean up column names to just show step numbers
        df.index = [re.search(r'step(\d+)', col).group(1) for col in df.index]
        df.index.name = "Step"

        # WSC is a broken decreasing metric
        df.drop(columns=["wsc"], inplace=True)

        print(f"{model} DataFrame shape: {df.shape}")
        df.to_csv(f"{output}/{model}.csv")
    print("Done.")




def load_json_data(path, files):
    """
    Load data from JSON files in the specified path.

    Args:
        path (str): Directory path containing JSON files
        files (list): List of JSON filenames to load

    Returns:
        list: List of dictionaries containing the JSON data
    """
    data = []
    for file in files:
        file_path = os.path.join(path, file)
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                data.append(json_data)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

    return data


def extract_metrics(data, metric="acc"):
    """
    Extract a specific metric from all JSON files for multiple tasks.

    Args:
        data (list): List of JSON data dictionaries
        tasks (list): List of task names
        metric (str): Metric name (e.g., "acc" or "acc_norm")

    Returns:
        dict: Dictionary of lists where each key is a task and each value is a list of metrics
    """
    tasks = sorted(list(data[0]["results"].keys()))
    values = {}
    for d in data:
        for task in tasks:
            if metric in d['results'][task]:
                if task not in values:
                    values[task] = []
                values[task].append(d['results'][task][metric])
                
            # try:
            #     # Some tasks might have 'acc', others might have 'likelihood_difference'
            #     # Let's strictly extract the accuracy metrics
                
            #     # Try to get the first available metric
            #     if metric in d['results'][task]:
            #         value = d['results'][task][metric]
            #     elif metric + ",none" in d['results'][task]:
            #         value = d['results'][task][metric + ",none"]
            #     elif "likelihood_difference" in d['results'][task]:
            #         value = d['results'][task]["likelihood_difference"]
            #     else:
            #         # Get the first metric available for this task
            #         first_metric = next(iter(d['results'][task].keys()))
            #         value = d['results'][task][first_metric]

            #     values[task].append(value)
            # except (KeyError, StopIteration) as e:
            #     print(f"Could not find metric for {task} in data: {str(e)}")
            #     values[task].append(None)

    return values

if __name__ == "__main__":
    main()


