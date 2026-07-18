# Fitting one spectrum

This workflow reads one spectrum, defines a circuit, fits its parameters, and
writes a report with fitted data and plots.

## 🧪 Minimal workflow

```python
from EISFitpython import data_extraction as data
from EISFitpython import EISFit_main as fit

f, Z = data.readTXT("measurement.txt")
f, Z = data.trim_data(f, Z, fmin=5.0)

circuit = "(R1|Q1)+(R2|Q2)+Q3"
initial = [4.8e4, 1.3e-10, 0.8, 9.2e4, 1.3e-9, 0.9, 5.7e-7, 0.6]

# Empty bounds represent an unconstrained fit in the project examples.
popt, perror, Z_fit = fit.full_EIS_report(
    f,
    Z,
    initial,
    circuit,
    UB=[],
    LB=[],
    weight_mtd="M",
    method="lm",
    single_chi="No",
    plot_type="both",
)
```

The call writes its outputs to the current directory. See
[Generated files](../generated-files.md).

## ⚖️ Weighting

`EISFit` passes its weight vector to SciPy as `sigma`.

| Code | Vector passed as `sigma` | Notes |
|---|---|---|
| `M` | `[abs(Z), abs(Z)]` | Modulus weighting; default |
| `P` | `[Z.real, Z.imag]` | Proportional weighting as implemented |
| `U` | `None` | Unity/unweighted fit |

Choose the weighting method that matches the analysis protocol used for the
measurement series.

## 🎯 Bounds and solver

The method is forwarded to `scipy.optimize.curve_fit`.

- Use `method="lm"` for unconstrained problems.
- Use `method="trf"` or `method="dogbox"` with finite lower and upper bounds.
- Keep `params`, `LB`, and `UB` aligned with the circuit's parameter order.

```python
lower = [0.0, 0.0, 0.0, 0.0]
upper = [1e9, 1e9, 1.0, 1e3]

fit_stats, popt, perror, corr = fit.EISFit(
    f,
    Z,
    params=[5.0, 20.0, 1e-6, 0.9],
    circuit="R1+(R2|Q1)",
    UB=upper,
    LB=lower,
    weight_mtd="M",
    method="trf",
)
```

## 📊 Fit statistics

The first return value from `EISFit` contains:

| Key | Meaning |
|---|---|
| `chi2` | Sum of squared residuals after weighting |
| `R_chi2` | `chi2 / dof` |
| `R_squared` | Coefficient calculated from the concatenated real/imaginary vector |
| `adj_R_squared` | Adjusted coefficient |
| `RMSE` | Root mean squared error |
| `SER` | Standard error of regression |
| `MAE` | Mean absolute residual |
| `F_statistic`, `p_value` | F statistic and associated p-value |
| `dof` | `2 * len(f) - len(params)` |
| `conf_intervals` | 95% half-widths based on the covariance estimate |
| `N_points`, `N_params` | Concatenated observation count and fitted parameter count |

These metrics describe the implementation's concatenated real/imaginary residual
vector. Their interpretation depends on the chosen weighting and model assumptions.

## 📍 Direct fitting calls

Use `EISFit` when you need arrays and statistics without report files. Use
`fit_report` for a printed parameter table from a standard circuit fit. Global-local
fitting has its own report function.
