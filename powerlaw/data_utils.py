import pandas as pd
import pickle
import numpy as np
from collections import namedtuple


# Load these once and keep in memory
df_llc_sparse = None
df_llc_dense = None
df_loss_sparse = None
df_loss_dense = None
df_llc_nb10 = None


def reload_data(data_path):
    global df_loss_sparse, df_llc_sparse
    global df_loss_dense, df_llc_dense
    global df_llc_nb10


    # Sparse losses - contains results for 1B but doesn't contain std
    with open(f'{data_path}/pythia-test-losses-new-seed-1024.pkl',
                'rb') as file:
        df_loss_sparse = pickle.load(file)
        df_loss_sparse.set_index('Step', inplace=True)

    # Dense test losses and std dev for all models and subsets
    # (including full pile) up to 410m
    with open(f'{data_path}/pythia-test-losses-dense-new-seed.pkl', 'rb') as file:
        df_loss_dense = pickle.load(file)
        df_loss_dense.set_index('Step', inplace=True)

    # Pile subset LLCs for sparse checkpoints and models up to 1b
    with open(f'{data_path}/pythia-pile-subset-llcs.pkl', 'rb') as file:
        df_llc_sparse = pickle.load(file)
        df_llc_sparse.set_index('Step', inplace=True)

    # nbeta=10
    with open(f'{data_path}/pythia-pile-subset-llcs-nbeta-10.pkl', 'rb') as file:
        df_llc_nb10 = pickle.load(file)
        df_llc_nb10.set_index('Step', inplace=True)
        # align column order
        subset = [c for c in df_llc_sparse.columns if c in df_llc_nb10.columns]
        df_llc_nb10 = df_llc_nb10[subset]

    # Full pile LLCs for sparse checkpoints for models up to 1b
    with open(f'{data_path}/pythia-full-llc.pkl', 'rb') as file:
        df_llc_sparse_patch = pickle.load(file)
        df_llc_sparse_patch.set_index('Step', inplace=True)

        # Patch "full llc" into the results:
        renames = {c:c+"-full" for c in df_llc_sparse_patch.columns}
        df_llc_sparse_patch.rename(columns=renames, inplace=True)
        df_llc_sparse = df_llc_sparse.join(df_llc_sparse_patch)

    # Large model limited context checkp[oints for models > 1b
    # Sparse losses - contains results for 1B but doesn't contain std
    with open(f'{data_path}/variable_context_pile_subsets.pkl', 'rb') as file:
        df_big = pickle.load(file)

        keep = [c for c in df_big.columns if "_1024_" in c]
        rename = ["pythia-" + c.replace("_pile-", "-").replace("_1024_", "_").replace("_pile1m", "-full") for c in keep]
        df_big.set_index('Step', inplace=True)
        df_big = df_big[keep]
        df_big.rename(columns=dict(zip(keep, rename)), inplace=True)

        # Now split into loss and LLC columns
        loss_cols = [c for c in df_big if c.endswith("_loss")]
        base = [c[:-5] for c in loss_cols]
        llc_cols = [c + "_llc" for c in base]

        patch_loss = df_big[loss_cols].rename(columns=dict(zip(loss_cols, base)))
        patch_llc = df_big[llc_cols].rename(columns=dict(zip(llc_cols, base)))

        df_llc_sparse = df_llc_sparse.join(patch_llc)
        df_loss_sparse = df_loss_sparse.join(patch_loss)

    # Dense LLCs (currently just for Pythia410m)
    with open(f'{data_path}/pythia-410m-dense-llc.pkl', 'rb') as file:
        df_llc_dense = pickle.load(file)
        df_llc_dense.rename(columns={"_step": "Step"}, inplace=True)
        df_llc_dense.set_index('Step', inplace=True)


def load_dfs(code, data_path):
    """Data Loading.
    - updated to 28/1/2024 format with dense results specifically for pythia-410m.
    - (this shall be "410m-dense")
    - updated to 24/1/2024 format
    """

    # We are in a half-way kind of state where some models are dense
    # and others are sparse
    global df_loss_sparse, df_llc_sparse
    global df_loss_dense, df_llc_dense
    global df_llc_nb10

    if df_loss_sparse is None:
        # Load all the dataframes on first use
        reload_data(data_path)

    # Extract sub-dataframes specifically for each model
    # So we need to know where to look to find each of these
    msizes = {
        '14m': (df_loss_dense, df_llc_sparse),
        '31m': (df_loss_dense, df_llc_sparse),
        '70m': (df_loss_dense, df_llc_sparse),
        '160m': (df_loss_dense, df_llc_sparse),
        '160m-nb': (df_loss_sparse, df_llc_nb10),
        '410m-dense': (df_loss_dense, df_llc_dense),
        '410m-nb': (df_loss_sparse, df_llc_nb10),
        '410m': (df_loss_sparse, df_llc_sparse),
        '1b': (df_loss_sparse, df_llc_sparse),
        '1.4b': (df_loss_sparse, df_llc_sparse),
        '2.8b': (df_loss_sparse, df_llc_sparse),
        '6.9b': (df_loss_sparse, df_llc_sparse),
    }
    assert code in msizes
    msize = code.split("-")[0]  # get the size component if any tags

    df_loss, df_llc = msizes[code]

    # For now, we are not using the std
    columns = [c for c in df_llc.columns if msize in c]

    rename_llc = {c: c.split(msize+"-")[-1] for c in columns}
    task_llc = df_llc[columns].rename(columns=rename_llc)

    if df_loss is df_loss_dense:
        loss_columns = [c + "-loss" for c in columns]
        rename_loss = {c + "-loss": c.split(msize+"-")[-1] for c in columns}
    else:
        loss_columns = columns
        rename_loss = rename_llc
    task_loss = df_loss[loss_columns].rename(columns=rename_loss)

    # Now (often) we have way more steps in loss than in llc
    if len(task_loss) > len(task_llc):
        task_loss = task_loss.loc[task_llc.index]

    if code == "410m-dense":
        drop = ['wikipedia_en']
        print("Warning: dropping incomplete wikipedia task from 410m-dense")
        task_llc.drop(columns=drop, inplace=True)
        task_loss.drop(columns=drop, inplace=True)

    if code == "1b" and False:  # temporarily don't drop
        # we have observed dm_mathematics unusual trajectory on this
        drop = ['dm_mathematics']
        print("Warning: dropping dm_mathematics (no analysis interval 1B)")
        task_llc.drop(columns=drop, inplace=True)
        task_loss.drop(columns=drop, inplace=True)

    # Note these may have NANs now...
    return task_llc / 100., task_loss


# Trace is just a named tuple
Trace = namedtuple('Trace', ['x','y','s'])


def trim_trace(df_llc, df_loss, task, start=0, finish=1e10):
    """
    Extract and trim (by step) a task from the data.
    Returns 1D arrays for LLC, Loss and Step.
    Inclusive of start and finish step.
    """
    # assert (df_loss.columns == df_llc.columns).all() not needed to be true
    s = df_llc[task].index.values
    x = df_llc[task].values
    y = df_loss[task].values
    keep = (s >= start) & (s<=finish)
    x = x[keep]
    y = y[keep]
    s = s[keep]

    # Sometimes LLCs are missing in the tail...
    if np.isnan(x).any():
        ix = np.argmax(np.isnan(x))
        x = x[:ix]
        y = y[:ix]
        s = s[:ix]

    assert not np.isnan(x).any()
    assert not np.isnan(y).any()

    return Trace(x, y, s)


def split(trace, cutoff_step):
    """
    Simple train/test split.
    The cutoff step is part of the training split....
    """
    x, y, s = trace
    cut = np.argmax(s > cutoff_step)
    x_train, y_train, s_train = x[:cut], y[:cut], s[:cut]
    x_test, y_test, s_test = x[cut:], y[cut:], s[cut:]
    return Trace(x_train, y_train, s_train), Trace(x_test, y_test, s_test)


def old_load_dfs(msize, data_path):
    """Data loading pre 24/1/2024 fmt"""

    with open(f'{data_path}/pile_{msize}_subset_loss_df.pkl', 'rb') as file:
       df_loss = pickle.load(file)

    with open(f'{data_path}/pile_{msize}_subset_llc_df.pkl', 'rb') as file:
      df_llc = pickle.load(file)

    step = "step" if "step" in df_loss.columns else "Step"
    df_loss.set_index(step, inplace=True)
    df_llc.set_index(step, inplace=True)
    df_llc = df_llc / 100.  # adjust the scale

    return df_llc, df_loss



