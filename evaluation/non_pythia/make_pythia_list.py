from huggingface_hub import list_repo_refs
import numpy as np


# Define the model set
models = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
]

# What revisions are available?
refs = list_repo_refs("EleutherAI/" + models[0])
branches = [branch.ref.replace("refs/heads/", "") for branch in refs.branches]
av = [s for s in branches[::-1] if "step" in s]
av = [int(s[4:]) for s in av]

# When resampling a finite list we often get collisions
n_checkpoints = 30
ask_checkpoints = n_checkpoints
while True:
    target = np.exp(np.linspace(np.log(128), np.log(143000), ask_checkpoints))

    # Snap to available
    revs = []
    for target in target:
        idx = np.argmin(np.abs(np.array(av) - target))
        revs.append(av[idx])
    revs = [0] + revs + [av[-1]]
    revs = np.unique(revs)

    # fine-tune the spacing...
    if len(revs) > n_checkpoints:
        ask_checkpoints -= 1

    elif len(revs) < n_checkpoints:
        ask_checkpoints += 1
    else:
        break

print("name,url,revision")


# Stratify over models to approximately balance the load between machines:
for step in revs:
    for model in models:
        size = model.split("-")[-1]
        code = f"Pythia_{size}_{step:06d}"
        url = f"EleutherAI/{model}"
        rev = f"step{step:d}"
        print(f"{code},{url},{rev}")

