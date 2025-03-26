"""Plan and allocate jobs"""
import fire
import os
from enum import Enum
import json
from huggingface_hub import list_repo_refs


splits = {
    "S": ["14m", "31m", "70m", "160m", "410m"],
    "M": ["1b", "1.4b"],
    "L": ["2.8b"],
    "XL": ["6.9b"],
    "ALL": ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"],
}


def main(n_workers=0, models="", inventory="auto", EXPERIMENT_NAME="EXP000"):
    if n_workers == 0:
        print("usage: plan_jobs.py n_workers models inventory")
    if models not in splits:
        print(f"Model size must be in:")
        for s in splits:
            print(f"   {s}: {splits[s]}")
        return

    model_sizes = splits[models]
    experiment_folder = os.path.join("experiments", EXPERIMENT_NAME)
    MODEL_CHECKPOINTS = None  # Not filtering

    # Compute remaining work
    if inventory == "auto":
        # Make one
        inventory = "_inventory.txt"
        os.system(f"find experiments > {inventory}")

    have = set()
    with open(inventory, "r") as f:
        for row in f.readlines():
            have.add(row[:-1])

    jobs = []
    for model_size in model_sizes:
        model_name = f"EleutherAI/pythia-{model_size}"
        model_folder = os.path.join(experiment_folder, model_size)
        revs = filter_revisions(model_name, MODEL_CHECKPOINTS)

        for revision in revs:
            step = int(revision.split("step")[-1])
            filename = os.path.join(model_folder, f"step{step:06d}.pkl")

            if filename not in have:
                worker = len(jobs) % n_workers
                job = (worker, model_size, model_name, revision, filename)
                jobs.append(job)

    with open("jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)
    print("Job allocations saved.")



def filter_revisions(model_name, model_checkpoints=None):
    # Get available model checkpoint steps
    refs = list_repo_refs(model_name)
    branches = [branch.ref.replace("refs/heads/", "") for branch in refs.branches]
    revs = [s for s in branches[::-1] if "step" in s]

    if model_checkpoints is not None:
        checkpoint_str = ["step{}".format(i) for i in model_checkpoints]
        revs = [s for s in revs if s in checkpoint_str]

    return revs



if __name__ == "__main__":
    fire.Fire(main)

