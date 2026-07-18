# Batch and Arrhenius analysis

Batch fitting applies one circuit model to multiple files in sequence. The fitted
parameters from one file become the initial values for the next file.

## 🧰 Sequential batch fitting

```python
import numpy as np
from EISFitpython import data_extraction as data
from EISFitpython import EIS_Batchfit as batch

files = data.get_eis_files("EIS_Data", "Example-3-4")
temperature_c = np.array([140, 150, 160, 170, 180, 190, 200])

circuit = "(R1|Q1)+(R2|Q2)+Q3"
initial = [1.8e6, 1.3e-11, 0.9, 8.2e6, 1.3e-9, 0.9, 5.7e-7, 0.6]

fit_params, fit_errors = batch.Batch_fit(
    files,
    initial,
    circuit,
    temperature_c,
    UB=[],
    LB=[],
    weight_mtd="M",
    method="lm",
    min_value=None,
    max_value=None,
)
```

The two return arrays have one row per processed file and one column per circuit
parameter. Supply one temperature value for each file.

The `Temp` values are treated as Celsius for generated labels. When `min_value`
and `max_value` are set, each spectrum is trimmed inclusively to that range.

## 📈 Stacked Nyquist plot

```python
ax = batch.Nyq_stack_plot(files, temperature_c + 273)
```

`Nyq_stack_plot` appends `K` directly to each supplied temperature and does no
conversion. Pass Kelvin values when the legend should display Kelvin. It reads
each supported file, selects one common impedance scale, writes
`Nyq_stackplot.svg`, displays the figure, and returns its axes.

## 🌡️ Arrhenius conductivity analysis

After fitting, extract one or more resistance series:

```python
R_bulk = fit_params[:, 0]
R_bulk_err = fit_errors[:, 0]
R_gb = fit_params[:, 3]
R_gb_err = fit_errors[:, 3]

conductivity, conductivity_error = batch.plot_arrhenius(
    R_values=[R_bulk, R_gb],
    R_err=[R_bulk_err, R_gb_err],
    temp=temperature_c,
    diameter=0.8,
    thickness=0.21,
    labels=["Bulk", "Grain boundary"],
)
```

`temp` is converted from Celsius to Kelvin by adding 273. `diameter` and
`thickness` are in centimetres, resistance is in ohms, and the resulting
conductivity is in S/cm:

\[
\sigma = \frac{1}{R}\frac{l}{A},\qquad
A = \pi\left(\frac{d}{2}\right)^2.
\]

The function uses fixed geometry uncertainties of 0.001 cm for thickness and
0.01 cm\(^2\) for area when resistance errors are supplied. It returns lists of
conductivity arrays and error arrays and writes an SVG plot plus a text report.

## 🔋 Effective capacitance

For an `R|Q` response, `C_eff` calculates:

\[
C_\mathrm{eff} = \left(QR^{1-n}\right)^{1/n}.
\]

```python
capacitance_results = batch.C_eff(
    R_arrays=[R_bulk, R_gb],
    R_err_arrays=[R_bulk_err, R_gb_err],
    Q_arrays=[fit_params[:, 1], fit_params[:, 4]],
    Q_err_arrays=[fit_errors[:, 1], fit_errors[:, 4]],
    n_arrays=[fit_params[:, 2], fit_params[:, 5]],
    n_err_arrays=[fit_errors[:, 2], fit_errors[:, 5]],
    T=temperature_c,
    labels=["Bulk", "Grain boundary"],
)
```

Each returned item is `(C, C_err)`. The implemented uncertainty approximation is
`C * (Q_err/Q + n_err/n + R_err/R)`.
