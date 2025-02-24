"""Extracts a helm dataset as a toy problem and upload to huggingface."""
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, login


def main():

    response = input(
        "You've already run this - are you sure? (Y to continue)"
    ).strip().upper()

    if response != "Y":
        print("Stopping.")
        return

    with open("data/helm.txt", "r") as f:
        lines = f.readlines()
    # Gotta put the ---------------- in manually!
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l]

    # Scrape the accuracy tab
    t = 0
    while not lines[t].startswith("--------"):
        t += 1
    tasks = lines[1:t]
    # Now read in
    m = int((len(lines) -t -1)/t)
    data = np.zeros((m, t-1))
    models = []
    for i in range(m):
        start = 1+(i+1)*(t)
        models.append(lines[start])
        col = lines[start+1:start+t]
        if "-" in col:
            print
        data[i] = [np.nan if c == "-" else float(c)
            for c in col]

    df = pd.DataFrame(data, index=models, columns=tasks)

    df.dropna(inplace=True)  # its just Jamba 1.5 Mini
    assert df.isna().sum().sum() == 0

    df.drop('MMLU All Subjects - EM', axis=1, inplace=True)
    df.drop('Miscellaneous - EM', axis=1, inplace=True)

    # Fix the truncated column names:
    tasks = list(df.columns)
    replace = [
        (" - EM", ""),
        ("Scienc...", "Science"),
        ("Histor...", "History"),
        ("And ...", ""),
        (" ...", ""),
    ]

    for r in replace:
        tasks = [t.replace(*r) for t in tasks]

    df.columns = tasks

    from huggingface_hub import HfApi, login
    df.to_csv("helm.csv", index=True, header=True) 

    #login()  # This will prompt for your HF token

    # Initialize the API
    api = HfApi()

    repo = "alistaireid/helm"
    # Initialize the API
    api = HfApi()

    try:
        api.create_repo(
            repo_id=repo,
            repo_type="dataset",
            private=False  # This makes it public
        )
    except:
        pass

    api.upload_file(
        path_or_fileobj="helm.csv",
        path_in_repo="helm.csv",
        repo_id=repo,
        repo_type="dataset"
    ) 
    print("Uploaded!")


if __name__=="__main__":
    main()
