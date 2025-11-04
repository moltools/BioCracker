# BioCracker

<p align="center">
    <a href="https://github.com/moltools/biocracker/actions/workflows/tests.yml">
      <img alt="testing & quality" src="https://github.com/moltools/biocracker/actions/workflows/tests.yml/badge.svg" /></a>
    <a href="https://pypi.org/project/biocracker">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/biocracker" /></a>
    <a href="https://pypi.org/project/biocracker">
      <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/biocracker" /></a>
    <a href="https://doi.org/10.5281/zenodo.17524264">
      <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.17524264.svg" alt="DOI" /></a>
</p>

Parser for antiSMASH output GenBank files.

See the [examples folder](https://github.com/moltools/biocracker/tree/main/examples) for usage examples.

## Installation

Some BioCracker depdendencies rely on various command line tools to operate. These tools might not be available on all platforms. The `pyproject.toml` file specifies the core parser that is platform independent, but some functionality might be limited without the command line tools. BioCracker is designed to fail gracefully when some of these third party dependencies are not available.

We recommend installing BioCracker in a virtual conda environment, based on the provided `environment.yml` file to make sure all modules are available:

```bash
conda env create -f environment.yml
```

### Installing PARAS

PARAS is used by BioCracker to predict substrate specificities of NRPS adenylation domains.

PARAS has no PyPI package ans must be installed from source manually:

```bash
pip install "paras @ git+https://github.com/bthedragonmaster/parasect.git@v2.0.0"
```

### Installing HMMER2 on macOS Arm64

Use Rosetta to install the x86_64 version of HMMER2:

```bash
conda activate biocracker
conda config --env --set subdir osx-64
conda install hmmer2
```

## Development

To set up a development environment, use the provided `environment.dev.yml` file:

```bash
conda env create -f environment.dev.yml
```
