# Plotting and modulus analysis

Plotting helpers accept Matplotlib axes so callers can control figure layout and
styling. Most accept a `param_dict` that is forwarded to `Axes.plot`.

## 📊 Nyquist plots

```python
import matplotlib.pyplot as plt
from EISFitpython import EISFit_main as fit

fig, ax = plt.subplots()
fit.nyquistPlot(
    ax,
    Z,
    {"marker": "o", "ls": "", "color": "black", "label": "Data"},
)
ax.legend()
```

`nyquistPlot` plots \(-Z''\) against \(Z'\), chooses \(\Omega\), k\(\Omega\),
M\(\Omega\), or G\(\Omega\) from that dataset's maximum magnitude, and returns
the same axes. When overlaying separately scaled datasets, ensure they fall in the
same scale range or scale them yourself before plotting.

## 📉 Bode plots

```python
fig, axes = plt.subplots(2, 1, sharex=True)
fit.bodePlot(
    axes,
    f,
    Z,
    {"marker": "o", "ls": "", "color": "black"},
    plot_types=["magnitude", "phase"],
)
```

Allowed plot types are `magnitude`, `phase`, `real`, and `imaginary`. The number
of axes must equal the number of requested types. If `plot_types` is omitted, the
implemented default is `['imaginary', 'phase']`.

Pass `all_Z=[Z1, Z2, ...]` to choose one common impedance scale for a batch.

## 🖼️ Convenience plot writer

`plot_fit(f, Z, Z_fit, plot_type)` creates and saves Nyquist and/or Bode figures,
calls `plt.show()`, and returns Matplotlib objects. `plot_type` is `"nyquist"`,
`"bode"`, or `"both"`.

For complete control over titles, legends, and layout, create Matplotlib axes and
call `nyquistPlot` or `bodePlot` directly.

## 🧲 Electric modulus

```python
fig, ax = plt.subplots()
fit.modulus_plot(
    [ax],
    f,
    Z,
    {"marker": "o", "ls": "", "color": "black"},
    diameter=0.8,
    thickness=0.21,
    plot_types=["imaginary"],
)
```

Geometry is in centimetres. The function uses
\(\epsilon_0 = 8.854\times10^{-14}\) F/cm and calculates
\(C_0=\epsilon_0 A/l\), followed by \(M'=\omega C_0Z'\) and
\(M''=\omega C_0Z''\). Allowed types are `magnitude`, `real`, and
`imaginary`; the implemented default is `['imaginary']`.

## 🔥 Arrhenius helper

`EA(ax, T, conductivity, plot_params)` expects Kelvin and S/cm, fits
`log(conductivity * T)` against `1000 / T`, prints activation-energy results, plots
the data and fit, and returns the axes.
