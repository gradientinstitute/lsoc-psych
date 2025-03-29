# Analysis Tools for Singular Psychometrics

# The Latent Space of Capabilities (LSoC) Project

The Latent Space of Capabilities (LSoC) project aims to advance the scientific understanding of AI capabilities through the novel combination of singular learning theory [^1] and psychometrics. The goal is to provide scientific foundations for the current use of model evaluations, as well as develop next-generation AI evaluation methods, by leveraging measurements that assess *how* models compute, not just *what* they compute.

This work is particularly relevant for AI governance and safety assessment, as current evaluation methods may fail to generalize or provide false confidence in model capabilities.

[^1]: Watanabe, S. (2009). Algebraic Geometry and Statistical Learning Theory.

This repository contains analysis code for the Latent Space of Capabilities (LSoC) project.


## Code Features

- **Factor Analysis**: Prototype factorization (including PCA, nonnegative matrix factorisation, tensor rank decomposition) using cross-validation for model selection.
- **Power Law Fitting**: Tools for fitting and analyzing power law relationships (used to analyse the relationship between loss and LLC across model scales).
- **Fine grained loss**: Analysis of fine-grained (token-level) losses of Pythia models across their training steps.
- **Data collection scripts**: Used to gathering some of the project's evaluation data such as fine-grained token losses.
- **Visualisations**: Of the associated analysis techniques

## Installation

To install the `lsoc` package, clone it from github and then in the cloned directory.
To use the sample data without configuration, install with `-e`.

```bash
pip install -e .
```


## Examples

Then try some example notebooks to see the analysis in action:

- `notebooks/powerlaw/demo.ipynb`: Demonstrate power law analysis on a Pythia model developmental trajectory.
- `notebooks/factor/tensor_rank.ipynb`: Demonstration of a Tensor Rank Decomposition (TRD)
- `notebooks/factor/Pythia70m.ipynb`: Latent developmental factors in timeseries of Pythia70m training steps.

The repository includes some sample data in the `data/` directory to run these demos.

## Modules

- `src/lsoc/`: Core Python modules
  - `factor/`: Matrix factorization implementations and accompanying visualisation
  - `powerlaw/`: Curve fitting and analysis tools and accompanying visualisation
- `notebooks/`: Jupyter notebooks demonstrating analyses
- `evaluation/`: Scripts for processing and analyzing model outputs

## Note

This repository focuses on analyzing LLC data rather than generating it. LLC estimation requires separate computation.
See [Timaeus](https://github.com/timaeus-research) for resources.

## Citation

If you use this code in your research, please cite:

```
@article{carroll2025psychometrics,
  title={Psychometrics for Pythia: Connecting Evaluations to Interpretability using Singular Learning Theory},
  author={Carroll, Liam and Reid, Alistair and Hoogland, Jesse and O'Callaghan, Simon and Wang, George and van Wingerden, Stan and Murfet, Daniel},
  year={2025},
}
```

## Licence

Copyright 2025 Gradient Institute and Timaeus

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
