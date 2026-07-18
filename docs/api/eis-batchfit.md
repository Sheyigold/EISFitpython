# `EISFitpython.EIS_Batchfit`

Sequential fitting and derived analysis for multiple spectra.

```python
from EISFitpython import EIS_Batchfit as batch
```

These functions create plots and reports in the current working directory.

## `Batch_fit`

```python
Batch_fit(
    files,
    params,
    circuit,
    Temp,
    UB,
    LB,
    weight_mtd,
    method,
    single_chi="No",
    min_value=None,
    max_value=None,
)
```

Fit each file sequentially with one circuit. The optimized parameters for a file
become the initial parameters for the next file.

| Parameter | Description |
|---|---|
| `files` | Ordered paths. `.txt` and `.csv` are supported. |
| `params` | Initial parameter vector for the first fit. |
| `circuit` | Equivalent-circuit string. |
| `Temp` | Temperatures in Celsius for generated `Data_<K>K` labels. |
| `UB`, `LB` | Upper and lower parameter bounds. |
| `weight_mtd` | `"M"`, `"P"`, or `"U"`. |
| `method` | SciPy curve-fit method. |
| `single_chi` | Forwarded to `fit_report`; normal batch use requires `"No"`. |
| `min_value`, `max_value` | Inclusive frequency bounds for the fitting interval. |

Returns `(all_params, all_fit_perror)` as NumPy arrays with one row per completed
file. Supply one temperature value for each file.

For each `.txt` file the function supports both NEISYS and general three-column
text formats.

Outputs include a combined report, one fitted-data file and Nyquist SVG per dataset,
one stacked Bode SVG, printed results, and displayed figures.

## `Nyq_stack_plot`

```python
Nyq_stack_plot(files, Temp)
```

Read multiple spectra and draw them on one consistently scaled Nyquist plot. Each
temperature is used directly in a `<value>K` legend label; no Celsius-to-Kelvin
conversion is performed.

Returns the Matplotlib axes, writes `Nyq_stackplot.svg`, and displays the figure.

## `plot_arrhenius`

```python
plot_arrhenius(
    R_values,
    R_err=None,
    temp=None,
    diameter=None,
    thickness=None,
    labels=None,
)
```

Convert resistance series to conductivity, run activation-energy analysis with
`EISFit_main.EA`, and save a plot and text report.

| Parameter | Description |
|---|---|
| `R_values` | One resistance array or a list of component arrays, in ohms. |
| `R_err` | Matching error array or list. For multiple components, pass a list with the same length. |
| `temp` | Measurement temperatures in Celsius; the function adds 273. |
| `diameter` | Sample diameter in centimetres. |
| `thickness` | Sample thickness in centimetres. |
| `labels` | One label or a list of component labels. |

Returns `(conductivities, conductivity_errors)`, both lists of NumPy-compatible
arrays. Writes `Arrhenius_plot.svg` and `Arrhenius_Analysis_Report.txt`, displays
the figure, and prints the captured report.

## `C_eff`

```python
C_eff(
    R_arrays,
    R_err_arrays,
    Q_arrays,
    Q_err_arrays,
    n_arrays,
    n_err_arrays,
    T,
    labels=None,
)
```

Calculate effective capacitance and its implemented error estimate for multiple
`R|Q` components.

Supply aligned component collections for resistance, CPE coefficient, exponent,
and their uncertainties. `T` is interpreted as Celsius for report labels.

Returns a list of `(C, C_err)` array tuples, writes
`Effective_Capacitance_Values.txt`, and prints the report.
