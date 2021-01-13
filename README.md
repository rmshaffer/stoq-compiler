[![Python package](https://github.com/rmshaffer/stoq-compiler/workflows/Python%20package/badge.svg)](https://github.com/rmshaffer/stoq-compiler/actions/)
[![codecov](https://codecov.io/gh/rmshaffer/stoq-compiler/branch/main/graph/badge.svg?token=KTF1NV8X0E)](https://codecov.io/gh/rmshaffer/stoq-compiler)

# stoqcompiler: Toolset for stochastic approximate quantum compilation

The `stoqcompiler` package provides a toolset for stochastic approximate quantum compilation as introduced in [arXiv:2101.04474](https://arxiv.org/abs/2101.04474).

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

> Ryan Shaffer. _Stochastic search for approximate compilation of unitaries._ [arXiv:2101.04474](https://arxiv.org/abs/2101.04474) (2021).
