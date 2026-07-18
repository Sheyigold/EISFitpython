# EISFitpython documentation

EISFitpython is a Python package for simulating, fitting, plotting, and comparing
electrochemical impedance spectroscopy (EIS) data with equivalent-circuit models.
It includes single-spectrum fitting, sequential batch fitting, and a global-local
single chi-square workflow for temperature-dependent experiments.

This site documents version **0.5.1**. It complements the project README with
complete function signatures, parameter order, return values, generated files,
and workflow examples.

## 🚀 Start here

- [Install the package and build these docs](installation.md).
- Learn the [circuit-string grammar and parameter order](guides/circuit-models.md).
- Follow a [single-spectrum fitting workflow](guides/single-spectrum-fitting.md).
- Use [global-local fitting](guides/global-local-fitting.md) for shared and
  temperature-dependent parameters.
- Look up every public function in the [API reference](api/index.md).
- Review the [generated output files](generated-files.md) produced by simulations,
  reports, and plots.

## 🧩 Module relationships

```mermaid
flowchart LR
    accTitle: EISFitpython module relationships
    accDescr: Data extraction and circuit evaluation feed the core fitting module. Batch fitting and global-local fitting build on all three.

    data_extraction["data_extraction<br/>read, trim, split"]
    circuit_main["circuit_main<br/>evaluate circuits"]
    eisfit_main["EISFit_main<br/>fit, report, plot"]
    batchfit["EIS_Batchfit<br/>batch and Arrhenius"]
    singlechi["singlechi<br/>global-local fitting"]

    data_extraction --> eisfit_main
    circuit_main --> eisfit_main
    data_extraction --> batchfit
    circuit_main --> batchfit
    eisfit_main --> batchfit
    data_extraction --> singlechi
    circuit_main --> singlechi
    eisfit_main --> singlechi
```

## 📦 Imports

The package exposes its version at the top level. Functions remain in their
modules, so import the modules you need:

```python
import EISFitpython
from EISFitpython import circuit_main as circuits
from EISFitpython import data_extraction as data
from EISFitpython import EISFit_main as fit
from EISFitpython import EIS_Batchfit as batch
from EISFitpython import singlechi

print(EISFitpython.__version__)
```
