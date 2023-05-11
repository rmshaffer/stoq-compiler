[![Downloads](https://pepy.tech/badge/stoqcompiler)](https://pepy.tech/project/stoqcompiler)
[![Python package](https://github.com/rmshaffer/stoq-compiler/workflows/Python%20package/badge.svg)](https://github.com/rmshaffer/stoq-compiler/actions/)
[![codecov](https://codecov.io/gh/rmshaffer/stoq-compiler/branch/main/graph/badge.svg?token=KTF1NV8X0E)](https://codecov.io/gh/rmshaffer/stoq-compiler)

# stoqcompiler: Toolset for stochastic approximate quantum compilation

The `stoqcompiler` package provides a toolset for stochastic approximate quantum compilation as introduced in [Quantum 7, 997](https://doi.org/10.22331/q-2023-05-04-997) (2023).

## Installation

The `stoqcompiler` package distribution is hosted on PyPI and can be installed via `pip`:

```
pip install stoqcompiler
```

Alternatively, the package and its requirements can be installed by cloning the repository locally:

```
git clone https://github.com/rmshaffer/stoq-compiler
cd stoq-compiler
pip install -r requirements.txt
pip install -e .
```

## Usage

For examples of using `stoqcompiler`, see the [example notebooks](./examples) and [unit tests](./tests).

## Citation

If you use or refer to this project in any publication, please cite the corresponding paper:

> Ryan Shaffer, Hang Ren, Emiliia Dyrenkova, Christopher G. Yale, Daniel S. Lobser, Ashlyn D. Burch, Matthew N. H. Chow, Melissa C. Revelle, Susan M. Clark, Hartmut Häffner. _Sample-efficient verification of continuously-parameterized quantum gates for small quantum processors._ [Quantum 7, 997](https://doi.org/10.22331/q-2023-05-04-997) (2023).

Please note that this repository [contains the data and code](./paper) used to generate each of the figures in the above-mentioned paper.
