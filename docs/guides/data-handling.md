# Reading and preparing data

The data API uses a common representation throughout the package:

- `f`: one-dimensional frequency array in hertz.
- `Z`: one-dimensional complex impedance array where `Z.real` is \(Z'\) and
  `Z.imag` is \(Z''\).

The file readers preserve the sign in the third input column. Plotting functions
display `-Z.imag` for the conventional Nyquist and Bode sign convention.

## 📂 Discover files

```python
from EISFitpython import data_extraction as data

files = data.get_eis_files("EIS_Data", subfolder="Example-3-4")
```

`get_eis_files` returns every regular file directly inside the selected directory,
sorted by path. Pair the ordered file list with the matching temperature array for
batch analysis.

## 📄 Choose a reader

| Function | Expected format | Header behavior | Delimiters |
|---|---|---|---|
| `readNEISYS` | NEISYS text export | Looks for `Freq. [Hz]`; starts after it | Tab |
| `readTXT` | Three numeric columns | Detects common header words or first numeric row | Tab, comma, or whitespace |
| `readCSV` | Headerless three-column numeric file | Headerless input | Comma or semicolon |

All three return `(f, Z)` and omit rows whose frequency is zero.

```python
f, Z = data.readTXT("measurement.txt")
```

The expected column order is:

```text
frequency, real impedance, imaginary impedance
```

## ✂️ Trim a frequency interval

`trim_data` applies inclusive bounds:

```python
f_fit, Z_fit = data.trim_data(f, Z, fmin=1.0, fmax=1e6)
```

Pass `None` at either end to leave that side unbounded. The original point order
and values are preserved.

## 🧱 Stack multiple spectra

```python
f_all, Z_all, lengths = data.stack_NEISYS_files(
    files,
    return_lengths=True,
)
```

`stack_NEISYS_files` concatenates data in the same order as `files`. With
`return_lengths=True`, it also returns each file's point count. The TXT and CSV
stacking functions return `(f_all, Z_all)`.

## 🧩 Split stacked spectra

Use exact lengths whenever they are available:

```python
f_sets, Z_sets = data.split_array(f_all, Z_all, lengths=lengths)
```

`split_array` uses these strategies:

1. If `lengths` is provided, split at its cumulative boundaries and require
   `sum(lengths) == len(f)`.
2. Otherwise detect a new sweep when the direction of the frequency differences
   reverses.
3. When `split_freq` is provided, approximate frequency matching provides an
   additional splitting strategy.

For a single spectrum the result is still a list containing one frequency array.
If `Z` is omitted, the second return value is an empty list.

For global-local fitting, concatenate spectra in acquisition order so each new
sweep resets toward its starting frequency.
