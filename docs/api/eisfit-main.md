# `EISFitpython.EISFit_main`

Core simulation, nonlinear least-squares fitting, reporting, plotting, activation
energy, and modulus functions.

```python
from EISFitpython import EISFit_main as fit
```

## `logf_gen`

```python
logf_gen(start_freq, end_freq, num_points)
```

Return `numpy.logspace(log10(start_freq), log10(end_freq), num_points)`. Both
frequencies must be positive for finite real output. The order follows the supplied
endpoints, so descending endpoints produce a descending array.

## `predict_Z`

```python
predict_Z(start_freq, end_freq, no_points, params, circuit_str)
```

Generate logarithmically spaced frequencies, evaluate a circuit, save the simulated
data, and return `(f, Z)`.

| Parameter | Type | Description |
|---|---|---|
| `start_freq`, `end_freq` | `int` or `float` | Frequency endpoints in hertz. |
| `no_points` | positive `int` | Number of generated frequencies. |
| `params` | sequence of float | Circuit parameter values. |
| `circuit_str` | `str` | Equivalent-circuit expression. |

The function writes `EIS-SIM_<circuit>.txt`, replacing `|` with `p` in the file
name. Rows are saved from high to low frequency, even if the returned array is
ascending.

## `EISFit`

```python
EISFit(
    f,
    Z,
    params,
    circuit,
    UB,
    LB,
    weight_mtd="M",
    method="lm",
    single_chi="No",
)
```

Fit concatenated real and imaginary impedance with `scipy.optimize.curve_fit`.

| Parameter | Description |
|---|---|
| `f` | Frequency array in hertz. |
| `Z` | Complex impedance array aligned with `f`. |
| `params` | Initial optimizer vector. |
| `circuit` | Circuit string when `single_chi="No"`; callable when `single_chi="Yes"`. |
| `UB`, `LB` | Upper and lower bounds, passed to SciPy as `(LB, UB)`. |
| `weight_mtd` | `"M"`, `"P"`, or `"U"`; see the fitting guide. |
| `method` | SciPy curve-fit method, commonly `"lm"`, `"trf"`, or `"dogbox"`. |
| `single_chi` | `"No"` converts a circuit string with `Z_curve_fit`; `"Yes"` uses the callable directly. |

Returns:

```text
(fit_stats, popt, perror, CorrMat)
```

- `fit_stats` is a dictionary of residual and regression statistics.
- `popt` is the optimized parameter vector.
- `perror` is `sqrt(diag(pcov))`.
- `CorrMat` is the parameter correlation matrix.

The returned values provide the optimized parameters, their standard errors, and
their correlation structure.

## `circuit_components`

```python
circuit_components(circuit_str)
```

Return numbered element tokens in textual order using the pattern
`[RCLQWFGH]` followed by one or more digits.

```python
fit.circuit_components("(R1|Q1)+(R2|Q2)+Q3")
# ["R1", "Q1", "R2", "Q2", "Q3"]
```

Unnumbered elements are omitted from this report-label helper even though circuit
evaluation can still recognize their element letter.

## `fit_report`

```python
fit_report(f, Z, params, circuit, UB, LB, weight_mtd, method, single_chi)
```

Run `EISFit`, print fit statistics, a circuit-aware parameter table, and a lower
triangular correlation matrix. Returns `(popt, perror)` for a standard circuit fit.

The function changes pandas' process-wide display options while formatting the
report.

## `nyquistPlot`

```python
nyquistPlot(ax, Z, param_dict)
```

Plot `-Z.imag` against `Z.real` on a supplied Matplotlib axes. Impedance is scaled
to ohms, kilohms, megohms, or gigohms from the largest magnitude in `Z`. The axes
are labeled and set to equal aspect. Returns `ax`.

Keyword arguments in `param_dict` are forwarded to `Axes.plot`.

## `bodePlot`

```python
bodePlot(axes, f, Z, param_dict, all_Z=None, plot_types=None)
```

Plot selected frequency-domain components on supplied axes. Allowed values in
`plot_types` are `"magnitude"`, `"phase"`, `"real"`, and `"imaginary"`. The
implemented default is `['imaginary', 'phase']`.

`all_Z` may contain all batch datasets to select one common impedance scale.
Returns the input axes list.

## `plot_fit`

```python
plot_fit(f=None, Z=None, Z_fit=None, plot_type="bode")
```

Create, save, and display data/fit plots. `plot_type` is `"nyquist"`, `"bode"`, or
`"both"`.

The function may write:

- `Nyquist-plot_with_fit.svg` or `Nyquist-plot_data_only.svg`
- `Bode-magnitude_with_fit.svg` or `Bode-magnitude_data_only.svg`
- `Bode-phase_with_fit.svg` or `Bode-phase_data_only.svg`

See the [plotting guide](../guides/plotting.md) for direct plotting examples.

## `full_EIS_report`

```python
full_EIS_report(
    freq,
    Z,
    params,
    circuit,
    UB,
    LB,
    weight_mtd,
    method,
    single_chi,
    plot_type="both",
)
```

Run `fit_report`, capture a second copy of its printed output, save report and fitted
data, create plots, and return `(popt, perror, Z_fit)`.

This entry point is designed for a circuit string with `single_chi="No"`. It writes
`EIS_fit_report.txt`, `EIS_fit_data_<circuit>.txt`, and files created by `plot_fit`.
The fitted-data rows are sorted from high to low frequency.

## `EA`

```python
EA(ax, T, conductivity, plot_params)
```

Fit an Arrhenius relationship and plot it.

| Parameter | Description |
|---|---|
| `ax` | Matplotlib axes. |
| `T` | Temperature in Kelvin. |
| `conductivity` | Conductivity in S/cm. Values must be positive for the logarithm. |
| `plot_params` | Keyword arguments forwarded to `Axes.plot` for the data. |

The regression uses `x = 1000/T` and `y = log(conductivity*T)`. The function prints
R-squared, activation energy, pre-exponential factor, and extrapolated room-
temperature conductivity. It returns `ax`, not the internal results dictionary.

## `modulus_plot`

```python
modulus_plot(
    axes,
    f,
    Z,
    param_dict,
    diameter,
    thickness,
    plot_types=None,
)
```

Calculate and plot electric-modulus magnitude, real part, or imaginary part.
`diameter` and `thickness` are in centimetres. The allowed plot types are
`"magnitude"`, `"real"`, and `"imaginary"`; the implemented default is
`['imaginary']`.

Returns the supplied axes list.
