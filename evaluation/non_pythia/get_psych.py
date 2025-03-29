# Copyright (c) Gradient Institute and Timaeus. All rights reserved.
# Licensed under the Apache 2.0 License.
"""
Aggregating losses into a token-loss based psychometrics matrix.

We've gone and run a lot of publicly available models on our [timaeus/pile-subsets-mini](https://huggingface.co/datasets/timaeus/pile_subsets_mini).

The log-files are aggregated into a feature-set for later analysis.
"""
import sys
import os
import numpy as np
import pickle
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python get_psych.py input_dir output_name n_ag(0-100)")
else:
    # Get all files in the directory and construct a psychometrics matrix
    import_dir = sys.argv[1]
    output= sys.argv[2]
    if len(sys.argv) > 3:
        take = int(sys.argv[3])
    else:
        take = 20

    files = [f for f in os.listdir(import_dir) if os.path.isfile(os.path.join(import_dir, f))]

    # Print all files
    print(f"Found {len(files)} files in {import_dir}:")

    old_cols = None
    # Now you can loop over them
    coldata = {}
    for file in files:
        file_path = os.path.join(import_dir, file)
        name = file[:-4]
        # print(f"Loading {name}...")
        try:
            with open(file_path, 'rb') as f:
                model_dict = pickle.load(f)
        except:
            # print("rm " + file, end=";")
            print("scp A100-7:/home/paperspace/pythia/output/" + file + " pythia_output/")
            print("scp A100-8:/home/paperspace/pythia/output/" + file + " pythia_output/")
            continue
        columns = []
        values = []
        for task in sorted(model_dict):
            values_dict = model_dict[task]
            if task == "dm_math_categories":
                continue
            losses = np.array([v.astype(float).sum() for v in values_dict['loss']])
            new_cols = [f"{task} {_id:03d}" for _id in values_dict['context_id']]
            columns.extend(new_cols[:take])
            values.extend(losses[:take])

        if old_cols is not None:
            assert columns == old_cols
        old_cols = columns
        coldata[name] = values

    psych = pd.DataFrame(coldata).T
    psych.columns = columns
    psych.index.name = "Model"
    fname = f"{output}-{take}.csv"
    psych.to_csv(fname)
    print(f"Saved to {fname}.")
