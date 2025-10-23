# BioCracker

Parser for antiSMASH output GenBank files.

## Installation

Some BioCracker depdendencies rely on various command line tools to operate. These tools might not be available on all platforms. The `pyproject.toml` file specifies the core parser that is platform independent, but some functionality might be limited without the command line tools. BioCracker is designed to fail gracefully when some of these third party dependencies are not available.

We recommend installing BioCracker in a virtual conda environment, based on the provided `environment.yml` file to make sure all modules are available:

```bash
conda env create -f environment.yml
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
