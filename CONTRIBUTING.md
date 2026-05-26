# Contributing to mcpy

Thanks for your interest in improving `mcpy`. Contributions of all kinds are
welcome: bug reports, feature requests, documentation, and code.

## Reporting bugs and requesting features

Please open an issue on the
[GitHub issue tracker](https://github.com/farrisric/mcpy/issues). For bugs,
include:

- what you expected to happen and what actually happened,
- a minimal example that reproduces the problem,
- your OS, Python version, and `mcpy` version.

## Asking for help

For usage questions that are not bugs, open a
[GitHub issue](https://github.com/farrisric/mcpy/issues) with the
`question` label.

## Development setup

```sh
git clone https://github.com/farrisric/mcpy.git
cd mcpy
pip install -e ".[test]"
```

## Running the tests

```sh
pytest tests/
```

The test suite covers the cell, move, and logging modules and does not require
`mace-torch`, `torch`, or `mpi4py`. CI runs these tests on Python 3.11–3.13.

## Linting

`mcpy` uses `flake8` (config in `.flake8`: max line length 100, single quotes):

```sh
flake8 mcpy/
```

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes, adding tests for new behaviour where practical.
3. Ensure `pytest tests/` and `flake8 mcpy/` pass.
4. Open a pull request describing the change and the motivation.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
