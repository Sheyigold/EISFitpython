# `EISFitpython.singlechi`

Global-local fitting across stacked EIS datasets. The module supports all circuit
elements listed in [Circuit models](../guides/circuit-models.md).

```python
from EISFitpython import singlechi
```

## `ELEMENT_PARAM_COUNT`

```python
ELEMENT_PARAM_COUNT = {
    "R": 1, "C": 1, "L": 1, "W": 1,
    "Q": 2, "F": 2, "G": 2, "H": 3,
}
```

This mapping aligns structured fit parameters and report labels.

## `flatten_params`

```python
flatten_params(params, circuit_str=None, N_sub=None)
```

Flatten a structured global-local parameter collection for the optimizer.

| Parameter | Description |
|---|---|
| `params` | Circuit-ordered scalars and iterables. |
| `circuit_str` | Optional numbered circuit string used to walk parameter slots. |
| `N_sub` | Number of datasets. Iterables of exactly this length are local. |

Scalars become one value. Iterables of length `N_sub` contribute every value.
Single-value iterables provide a global starting value. Returns a Python list.

## `Single_chi`

```python
Single_chi(f, *params, circuit_str, circuit_type="fit")
```

Split a stacked frequency sweep, distribute structured parameter values to each
dataset, evaluate the circuit, and combine the result.

| Parameter | Description |
|---|---|
| `f` | Concatenated frequency array. Boundaries are detected by sweep reset. |
| `*params` | Scalars for global slots; iterables of dataset length for local slots. |
| `circuit_str` | Circuit expression. Required keyword-only argument. |
| `circuit_type` | `"fit"` for concatenated real/imaginary output; `"predict"` for complex output. |

Returns a real array of length `2 * len(f)` in fit mode or a complex array of
length `len(f)` in prediction mode.

## `format_circuit_output`

```python
format_circuit_output(
    popt,
    perror,
    CorrM,
    circuit_str,
    N_sub,
    Temp,
    param_template,
)
```

Build circuit-aware parameter and correlation tables.

`param_template` contains one Boolean per circuit parameter slot: `True` means the
slot is local and consumes `N_sub` optimized values; `False` means global and
consumes one. `Temp` values are inserted into local labels with a `°C` suffix.

Returns `(parameter_df, correlation_df)` as pandas DataFrames. The parameter table
contains `Fit_Params`, `Value`, `Error`, and `% error`. The correlation table is
limited to the number of generated labels and masks its diagonal and upper triangle.

## `Single_chi_report`

```python
Single_chi_report(f, Z, params, Temp, circuit_str, weight_mtd="M")
```

Run the complete global-local fit, format and print statistics, write per-temperature
fitted data, generate plots, and write a report.

| Parameter | Description |
|---|---|
| `f`, `Z` | Concatenated frequency and complex-impedance arrays. |
| `params` | Structured circuit parameters using scalar/global and iterable/local semantics. |
| `Temp` | Temperatures in Celsius; one per detected dataset. |
| `circuit_str` | Numbered circuit expression. |
| `weight_mtd` | `"M"`, `"P"`, or `"U"`. |

Returns `(popt, perror, Z_fit)`. The optimization uses `method="lm"`.

The function writes `Single-chi_Report.txt`, `S-chi_fit_data_<Temp>C.txt` for each
dataset, individual Nyquist SVGs, and one stacked Bode SVG.

## `generate_plots`

```python
generate_plots(f, Z, Z_fit, Temp)
```

Split stacked data by sweep reset, create one Nyquist plot per paired temperature,
create a stacked Bode plot, save the SVG files, and display all figures.

Returns `(last_nyquist_figure, bode_figure)`. Supply one temperature value for each
dataset.
