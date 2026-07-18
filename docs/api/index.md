# API reference

The public API is organized into five function-based modules. Functions are not
re-exported from the package root.

| Module | Purpose | Public functions |
|---|---|---|
| [`circuit_main`](circuit-main.md) | Circuit evaluation and SciPy-compatible callables | `compute_impedance`, `Z_curve_fit`, `Z_gen` |
| [`data_extraction`](data-extraction.md) | File discovery, readers, stacking, trimming, splitting | `get_eis_files`, `parse_NEISYS_data`, `readNEISYS`, `readTXT`, `readCSV`, `stack_NEISYS_files`, `stack_TXT_files`, `stack_CSV_files`, `trim_data`, `split_array` |
| [`EISFit_main`](eisfit-main.md) | Simulation, nonlinear fitting, reports, plots, derived analyses | `logf_gen`, `predict_Z`, `EISFit`, `circuit_components`, `fit_report`, `nyquistPlot`, `bodePlot`, `plot_fit`, `full_EIS_report`, `EA`, `modulus_plot` |
| [`EIS_Batchfit`](eis-batchfit.md) | Sequential fits, stacked plots, Arrhenius and capacitance analysis | `Batch_fit`, `Nyq_stack_plot`, `plot_arrhenius`, `C_eff` |
| [`singlechi`](singlechi.md) | Global-local fitting across stacked datasets | `flatten_params`, `Single_chi`, `format_circuit_output`, `Single_chi_report`, `generate_plots` |

## 📦 Package metadata

```python
import EISFitpython

EISFitpython.__version__  # "0.5.1"
```

## 📐 Conventions

- Frequencies are in hertz unless stated otherwise.
- Impedance arrays are complex NumPy-compatible values in ohms.
- Circuit parameters follow the left-to-right order described in
  [Circuit models](../guides/circuit-models.md).
- `Temp` or `T` units depend on the function; each API entry states whether the
  values are Celsius or Kelvin.
- `ax` and `axes` are Matplotlib `Axes` objects supplied by the caller.
- File-writing behavior is summarized in [Generated files](../generated-files.md).
