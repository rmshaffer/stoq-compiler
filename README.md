[![Python package](https://github.com/rmshaffer/stoq-compiler/workflows/Python%20package/badge.svg)](https://github.com/rmshaffer/stoq-compiler/actions/)
[![codecov](https://codecov.io/gh/rmshaffer/stoq-compiler/branch/main/graph/badge.svg?token=KTF1NV8X0E)](https://codecov.io/gh/rmshaffer/stoq-compiler)

# stoqcompiler: Toolset for stochastic approximate quantum compilation

The `stoqcompiler` package provides a toolset for stochastic approximate quantum compilation, including an implementation of the randomized analog verification (RAV) protocol as introduced in [arXiv:2003.04500](https://arxiv.org/abs/2003.04500).

## Installation

The `stoqcompiler` package and its requirements can be installed via `pip` by cloning the repository locally:

```
git clone https://github.com/rmshaffer/stoq-compiler
cd stoq-compiler
pip install -r requirements.txt
pip install -e .
```

## Usage

For examples of using `stoqcompiler`, see the [example notebooks](./examples).
