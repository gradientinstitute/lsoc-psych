# scp -r paperspace@184.105.215.235:/home/paperspace/lsoc-psych/datasets/experiments/debug/1b/step143000.pkl ./

import pickle
import numpy as np


# Cross check my modified script
f1 = "experiments/debug/160m/step143000.pkl"
f2  = "experiments/debug1/160m/step143000.pkl"

with open(f1, "rb") as f:
    dat1 = pickle.load(f)

with open(f2, "rb") as f:
    dat2 = pickle.load(f)


by_id1 = {d['context_idx']: d for d in dat1['dm_mathematics']}
by_id2 = {d['context_idx']: d for d in dat2['dm_mathematics']}

n = len(by_id1)

for ix in range(n):
    if not all(by_id1[ix]["loss"]["zero-shot"] == by_id2[ix]["loss"]["zero-shot"]):
        print(f"{ix} zero shot mismatch")


def allclose(a, b, tol=0.1):
    return all( abs(i-j) < tol for i, j in zip(a, b))


for ix in range(n):
    ta = by_id1[ix]["tokens"]
    tb = by_id2[ix]["tokens"]

    a = by_id1[ix]["loss"]["few-shot"].tolist()
    b = by_id2[ix]["loss"]["few-shot"].tolist()

    if ta != tb:
        print(f"token mismatch")

    if len(a) != len(b):
        print(f"{ix} length mismatch")
        m = min(len(a), len(b)) - 1

    if not allclose(a[:-1], b[:-1]):
        print(f"{ix} core value mismatch")
        break
# The few shot is too long (an extra token in it?)
task = "enron_emails"
L1 = dat1[task]
L2 = dat2[task]

if len(L1) != len(L2):
    print("task length mismatch!")

for r1, r2 in zip(L1, L2):
    if r1["tokens"] != r2["tokens"]:
        print("task token mismatch")

    if any(r1["loss"] != r2["loss"]):
        print("task value mismatch")

# focus = by_id[127]
# list(zip(focus['tokens'], focus['loss']['few-shot']))

import smart_embed
smart_embed.embed(locals(), globals())

