# Changelog

All notable changes to EISFitpython are documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-06-13

Backward-compatible minor release. Two correctness fixes for the global-local
(single chi-square) workflow, plus a proper installable package definition.

### Fixed
- **`split_array` is now robust.** The previous implementation split a stacked
  multi-temperature spectrum using exact float equality (`freq == split_freq`)
  and assumed every dataset started at the global maximum frequency. Datasets
  with slightly different maximum frequencies, floating-point noise, or unequal
  lengths were silently mis-split (often merged into one). It now detects each
  dataset by sweep-reset (the frequency reversing direction), which is robust to
  float noise and to datasets that do not share an identical maximum frequency.
- **W/F/G/H parameters are now labeled correctly in the global-local report.**
  `format_circuit_output` and the validation loop in `flatten_params` previously
  only handled R/L/C and Q. Warburg/Gerischer elements fell through without
  advancing the parameter index, so every parameter printed *after* them was
  misaligned in the report table. A single `ELEMENT_PARAM_COUNT` table now drives
  both parameter counting and labeling for all element types.

### Added
- `split_array(..., lengths=...)`: opt-in exact split by explicit per-dataset
  point counts; bypasses all heuristics and raises `ValueError` on a mismatch.
- `stack_NEISYS_files(..., return_lengths=True)`: also returns per-file point
  counts, which can be passed straight to `split_array(..., lengths=...)` for an
  exact, heuristic-free split.
- `pyproject.toml`: the project is once again pip-installable (sdist + wheel).
  The importable package `EISFitpython` is built from the `modules/` directory.
- `tests/test_eisfit_patches.py`: regression tests locking in the golden split
  (new == old partition on clean data) and the new behavior.

### Changed
- `split_freq` in `split_array` is now a fallback (matched with `np.isclose`)
  used only when no sweep reset is detected; it emits a `RuntimeWarning`
  suggesting `lengths=`. Existing calls keep working unchanged.

### Compatibility notes
- All public signatures only **gained optional keyword arguments**; every
  existing call site keeps working.
- R/C/L/Q report labels are byte-identical to before.
- For well-formed descending sweeps that reset at the same maximum frequency,
  `split_array` returns the **same partition** as previous releases.
- **Honest behavior change:** `split_array` now produces *different* (now-correct)
  results for inputs the old code mishandled — datasets with different maximum
  frequencies, and ascending sweeps. Pinned users on older releases are
  unaffected; this only matters on upgrade.
- **Known caveat (preserved):** the length "validation" inside `flatten_params`
  remains lenient by original design — an iterable whose length differs from
  `N_sub` is treated as a global parameter rather than rejected. Only the W/F/G/H
  index desync was fixed here.
