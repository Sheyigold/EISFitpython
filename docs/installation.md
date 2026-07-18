# Installation

EISFitpython requires Python 3.9 or newer. Its runtime dependencies are NumPy,
SciPy, Matplotlib, and pandas.

## 📥 Install from a checkout

```bash
git clone https://github.com/Sheyigold/EISFitpython.git
cd EISFitpython
python -m pip install -e .
```

Verify the import:

```bash
python -c "import EISFitpython; print(EISFitpython.__version__)"
```

The expected version for this documentation is `0.5.1`.

## 🧪 Install test dependencies

```bash
python -m pip install -e ".[test]"
pytest
```

## 📖 Build the documentation

Install the documentation dependencies declared by the `docs` extra:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Open the local address printed by MkDocs. To create the deployable static site:

```bash
mkdocs build --strict
```

The generated HTML is written to `site/` and is not required at runtime.

## 🖥️ Non-interactive environments

Plot and report functions call `matplotlib.pyplot.show()`. On a server or in CI,
select the `Agg` backend before importing Matplotlib:

```bash
MPLBACKEND=Agg pytest
```

On Windows PowerShell, the equivalent is:

```powershell
$env:MPLBACKEND = "Agg"
pytest
```

## 📁 Working-directory advice

Simulation, report, and plotting functions write output files to the process's
current working directory. Create and enter a dedicated results directory to keep
each run organized.
