# Contributing

The project follows the [scverse cookiecutter template](https://github.com/scverse/cookiecutter-scverse).
The package lives in `src/scdecontx/`, tests in `tests/`, and the Sphinx documentation in `docs/`.
[Hatch](https://hatch.pypa.io/) manages the environments for tests and docs.

## Install locally

```bash
git clone https://github.com/mhbsiam/scdecontx.git
cd scdecontx
pip install -e ".[dev,test]"
```

Check which copy Python imports, in particular after installs from other paths:

```bash
python -c "import scdecontx; print(scdecontx.__file__)"
```

## Code style

[pre-commit](https://pre-commit.com/) runs [ruff](https://docs.astral.sh/ruff/) and the other checks in `.pre-commit-config.yaml` on every commit. Enable it once per clone:

```bash
pre-commit install
```

Run all checks by hand with `pre-commit run --all-files`.

## Tests and benchmarks

```bash
python -m pytest          # in the current environment
hatch test --all          # across the Python versions in pyproject.toml
```

`tests/test_celda.py` compares scDecontX with celda results in `tests/fixtures/celda/`. To make them again, install celda in R and run `Rscript tests/make_celda_fixtures.R`.

`tests/benchmarks/` compares scDecontX with R celda, NiRuff/decontx-python and omicverse/py-decontx; see `tests/benchmarks/README.md`. It needs R with celda and clones of the two Python ports:

```bash
bash tests/benchmarks/scripts/setup_others.sh others
python tests/benchmarks/scripts/compare.py --niruff others/niruff --omicverse others/omicverse --work <dir>
python tests/benchmarks/scripts/report.py
```

## Documentation

```bash
hatch run docs:build
hatch run docs:open
```

The docs are written in [MyST](https://myst-parser.readthedocs.io/) Markdown. The API page is generated from the docstrings with `autosummary`.

## Releases

Bump `version` in `pyproject.toml`, add a section to `CHANGELOG.md`, and create a GitHub release with a tag such as `v1.0.0`. scDecontX is not published to PyPI; users install a release with `pip install git+https://github.com/mhbsiam/scdecontx.git@v1.0.0`.
