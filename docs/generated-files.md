# Generated files

EISFitpython report and convenience-plot functions write to the current working
directory. Use a dedicated results directory to keep each analysis run organized.

Create and enter a results directory before starting an analysis:

```python
from pathlib import Path
import os

results = Path("results/run-001")
results.mkdir(parents=True, exist_ok=True)
os.chdir(results)
```

## 📋 File inventory

| Function | Files written |
|---|---|
| `EISFit_main.predict_Z` | `EIS-SIM_<circuit>.txt` |
| `EISFit_main.plot_fit` | Nyquist and/or Bode SVGs; names depend on whether `Z_fit` is supplied |
| `EISFit_main.full_EIS_report` | `EIS_fit_report.txt`, `EIS_fit_data_<circuit>.txt`, plus `plot_fit` SVGs |
| `EIS_Batchfit.Batch_fit` | `Batch_EISfit_Report.txt`, `EIS_fit_Data_<K>K.txt` and `<label>-Nyquist_plot.svg` per dataset, `stack-bode_plot.svg` |
| `EIS_Batchfit.Nyq_stack_plot` | `Nyq_stackplot.svg` |
| `EIS_Batchfit.plot_arrhenius` | `Arrhenius_plot.svg`, `Arrhenius_Analysis_Report.txt` |
| `EIS_Batchfit.C_eff` | `Effective_Capacitance_Values.txt` |
| `singlechi.Single_chi_report` | `Single-chi_Report.txt`, `S-chi_fit_data_<Temp>C.txt`, individual Nyquist SVGs, stacked Bode SVG |
| `singlechi.generate_plots` | `Data_<K>K_S-chi_nyquist_plot.svg`, `<last-label>_S-chi_stack_bode_plot.svg` |

## 📄 Data-file conventions

Simulated and fitted text outputs contain comment headers followed by:

```text
frequency [Hz]    real impedance [ohm]    imaginary impedance [ohm]
```

Rows are generally sorted from high to low frequency. Report functions return
arrays as well as writing files, so downstream code can avoid re-reading the saved
text.

## 🖼️ Figure display

Plotting entry points call `matplotlib.pyplot.show()`. In scripts, this may block
until figure windows are closed, depending on the active backend. In automated
environments, select a non-interactive Matplotlib backend before importing the
package.

## ⚠️ Name sanitization

Circuit-derived filenames replace `|` with `p`, but retain characters such as
parentheses and `+`. Those names are valid on common platforms but may need quoting
in shell commands.
