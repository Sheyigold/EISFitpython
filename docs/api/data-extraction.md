# `EISFitpython.data_extraction`

File discovery, text parsing, stacking, filtering, and dataset partitioning.

```python
from EISFitpython import data_extraction as data
```

All readers return frequency and impedance arrays while preserving the sign of the
imaginary input column.

## `get_eis_files`

```python
get_eis_files(base_path=".", subfolder=None)
```

Return a sorted list of regular files directly inside `base_path/subfolder`, or
inside `base_path` when `subfolder` is omitted. The function prints the number and
names of files it finds.

## `parse_NEISYS_data`

```python
parse_NEISYS_data(lines)
```

Parse an iterable of NEISYS text lines. Data begins after the first line containing
`Freq. [Hz]`. Each tab-separated data row contributes frequency, real impedance,
and imaginary impedance.

Returns `(f, Z)` as NumPy arrays.

## `readNEISYS`

```python
readNEISYS(filename)
```

Read a NEISYS export and delegate parsing to `parse_NEISYS_data`. Returns `(f, Z)`.

## `readTXT`

```python
readTXT(filename)
```

Read a flexible three-column text file. It recognizes tab, comma, and whitespace
separators, blank lines, and headers containing words such as `freq`, `hz`, `real`,
`imag`, or `z`. Returns `(f, Z)` as NumPy arrays.

## `readCSV`

```python
readCSV(filename)
```

Read a headerless numeric file with at least three columns. Comma and semicolon
delimiters are supported. Columns are frequency, real impedance, and imaginary
impedance. Returns `(f, Z)` as NumPy arrays.

## `stack_NEISYS_files`

```python
stack_NEISYS_files(filenames, return_lengths=False)
```

Concatenate NEISYS spectra in input order.

| `return_lengths` | Return value |
|---|---|
| `False` | `(f, Z)` |
| `True` | `(f, Z, lengths)` where `lengths[i]` is the number of valid rows from `filenames[i]` |

Files are processed in the supplied order.

## `stack_TXT_files`

```python
stack_TXT_files(filenames)
```

Read each path with `readTXT`, concatenate the spectra, and return `(f, Z)`.

## `stack_CSV_files`

```python
stack_CSV_files(filenames)
```

The CSV counterpart to `stack_TXT_files`, using `readCSV` for each file.

## `trim_data`

```python
trim_data(f, Z, fmin=None, fmax=None)
```

Return `(f_trimmed, Z_trimmed)` using an inclusive Boolean mask. `fmin=None` or
`fmax=None` leaves that boundary open. Inputs must support NumPy-style elementwise
comparison and matching Boolean indexing.

## `split_array`

```python
split_array(f, Z=None, split_freq=None, lengths=None)
```

Partition a stacked sweep into lists of per-dataset arrays.

| Parameter | Type | Description |
|---|---|---|
| `f` | one-dimensional array-like | Stacked frequencies. |
| `Z` | array-like or `None` | Optional aligned impedance values. |
| `split_freq` | float or `None` | Optional frequency value for approximate boundary matching. |
| `lengths` | sequence of int or `None` | Exact point counts; preferred when available. |

Returns `(sublists_f, sublists_Z)`. `sublists_Z` is empty when `Z` is `None`.
