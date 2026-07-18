# Global-local fitting

The global-local workflow fits all stacked spectra in a single optimization. A
parameter can be shared by every dataset (global) or have one value per dataset
(local).

## 🧠 Parameter semantics

For `N` detected spectra:

- A scalar is global.
- An iterable whose length is exactly `N` is local and contributes `N` fitted
  values.
- A single-value iterable can also provide a global starting value.

Keep the structured tuple in the same slot order as the circuit parameters.

```python
circuit = "(R1|Q1)+(R2|Q2)+Q3"

params = (
    [1.8e5, 1.3e5, 9.8e4],  # R1: local
    1.3e-11,                 # Q1: global
    0.90,                    # n1: global
    [2.2e6, 1.2e6, 7.1e5],  # R2: local
    [1.48e-10] * 3,          # Q2: local
    [0.90] * 3,              # n2: local
    [1.27e-6] * 3,           # Q3: local
    [0.50] * 3,              # n3: local
)
```

Use exactly `N` values for each local parameter so every dataset receives its own
starting value.

## 🧱 Prepare stacked data

```python
import numpy as np
from EISFitpython import data_extraction as data
from EISFitpython import singlechi

files = data.get_eis_files("EIS_Data", "Example-3-4")
f, Z, lengths = data.stack_NEISYS_files(files, return_lengths=True)

# Validate the intended partition before fitting.
f_sets, Z_sets = data.split_array(f, Z, lengths=lengths)
N = len(f_sets)
temperature_c = np.array([140, 150, 160])

assert N == len(temperature_c)
```

`Single_chi_report` identifies each dataset from the sweep reset in `f`. Concatenate
the spectra in acquisition order and retain the reset at every dataset boundary.

## 🎯 Fit and report

```python
popt, perror, Z_fit = singlechi.Single_chi_report(
    f,
    Z,
    params,
    temperature_c,
    circuit,
    weight_mtd="M",
)
```

The workflow uses Levenberg-Marquardt without user-supplied bounds. It writes one
fitted-data file per temperature, a text report, individual Nyquist plots, and a
stacked Bode plot.

The flattened result groups local values together in each parameter slot. For the
example above, the order begins:

```text
R1_T1, R1_T2, R1_T3, Q1, n1,
R2_T1, R2_T2, R2_T3, Q2_T1, ...
```

Use the report labels instead of hard-coded slices when the circuit or global/local
choices may change.

## 🧰 Supporting functions

- `flatten_params` converts the structured tuple to the optimizer vector.
- `Single_chi(..., circuit_type="fit")` returns concatenated real and imaginary
  values.
- `Single_chi(..., circuit_type="predict")` returns complex impedance.
- `format_circuit_output` builds pandas tables with circuit-aware labels.
- `generate_plots` saves the global-local figures.
