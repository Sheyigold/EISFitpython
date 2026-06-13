"""
Regression tests for the split_array and global-local report-labeling fixes.

Run from the repository root (with EISFitpython importable) via:

    pytest test_eisfit_patches.py -v

The tests lock in two guarantees:
  1. On well-formed descending sweeps, the patched split_array returns exactly
     the same partition as the original implementation (no breakage for
     existing users).
  2. The patched split_array and report labeling correctly handle cases the old
     code got wrong: unequal-length datasets, datasets with different maximum
     frequencies, and circuits containing W / F / G / H elements.
"""

import numpy as np
import pytest

# Prefer the installed package; fall back to a modules/ dir on sys.path.
try:
    from EISFitpython import data_extraction as dt
except ImportError:  # running with modules/ directly on the path
    import data_extraction as dt

try:
    from EISFitpython import singlechi as sc
except Exception:  # pragma: no cover - depends on environment
    try:
        import singlechi as sc
    except Exception:
        sc = None


# --------------------------------------------------------------------------- #
# Reference: the ORIGINAL split_array implementation (exact float equality).
# Used only to prove the new code reproduces it on well-formed data.
# --------------------------------------------------------------------------- #
def legacy_split(f, Z=None, split_freq=None):
    if split_freq is None:
        split_freq = np.max(f)
    sublists_f, sublists_Z = [], []
    current_freqs, current_Zs = [], []
    for i, freq in enumerate(f):
        if freq == split_freq and i != 0:
            sublists_f.append(np.array(current_freqs))
            if Z is not None:
                sublists_Z.append(np.array(current_Zs))
            current_freqs, current_Zs = [], []
        current_freqs.append(freq)
        if Z is not None:
            current_Zs.append(Z[i])
    if current_freqs:
        sublists_f.append(np.array(current_freqs))
        if Z is not None and current_Zs:
            sublists_Z.append(np.array(current_Zs))
    return sublists_f, sublists_Z


def descending_sweep(fmax=1e6, fmin=1e-1, n=20):
    return np.logspace(np.log10(fmax), np.log10(fmin), n)


# --------------------------------------------------------------------------- #
# 1. Golden split: identical output to the legacy code on clean data.
# --------------------------------------------------------------------------- #
def test_golden_split_matches_legacy():
    sweep = descending_sweep()
    f = np.concatenate([sweep, sweep, sweep])          # 3 identical datasets
    Z = (f + 1j * (-f)).astype(complex)

    new_f, new_Z = dt.split_array(f, Z, split_freq=np.max(f))
    old_f, old_Z = legacy_split(f, Z, split_freq=np.max(f))

    assert len(new_f) == len(old_f) == 3
    for a, b in zip(new_f, old_f):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(new_Z, old_Z):
        np.testing.assert_array_equal(a, b)


def test_single_dataset_not_split():
    f = descending_sweep()
    sub_f, _ = dt.split_array(f, split_freq=np.max(f))
    assert len(sub_f) == 1
    np.testing.assert_array_equal(sub_f[0], f)


# --------------------------------------------------------------------------- #
# 2. Hard cases the old code got wrong.
# --------------------------------------------------------------------------- #
def test_unequal_lengths_and_different_max():
    # Dataset 2 has fewer points AND a different maximum frequency.
    d1 = descending_sweep(fmax=1e6, fmin=1e-1, n=20)
    d2 = descending_sweep(fmax=9e5, fmin=1e-2, n=14)   # different max, shorter
    f = np.concatenate([d1, d2])

    # Reset detection splits correctly...
    sub_f, _ = dt.split_array(f)
    assert [len(s) for s in sub_f] == [20, 14]

    # ...whereas the legacy code (global max appears only once) failed to split.
    old_f, _ = legacy_split(f, split_freq=np.max(f))
    assert len(old_f) == 1  # documents the original bug


def test_explicit_lengths_path():
    d1 = descending_sweep(n=20)
    d2 = descending_sweep(n=14)
    f = np.concatenate([d1, d2])
    Z = (f - 1j * f).astype(complex)

    sub_f, sub_Z = dt.split_array(f, Z, lengths=[20, 14])
    assert [len(s) for s in sub_f] == [20, 14]
    assert [len(s) for s in sub_Z] == [20, 14]


def test_explicit_lengths_mismatch_raises():
    f = np.concatenate([descending_sweep(n=20), descending_sweep(n=14)])
    with pytest.raises(ValueError):
        dt.split_array(f, lengths=[20, 10])  # 30 != 34


def test_float_noise_does_not_break_split():
    # Each dataset starts at a *slightly* different max -> exact == would fail.
    sweep = descending_sweep()
    f = np.concatenate([sweep, sweep * (1 + 1e-9), sweep * (1 - 1e-9)])
    sub_f, _ = dt.split_array(f)
    assert len(sub_f) == 3


# --------------------------------------------------------------------------- #
# 3. flatten_params now counts W (and validates all element types).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(sc is None, reason="EISFitpython package not importable")
def test_flatten_params_handles_warburg():
    # (R1|Q1)+W1 : slots = R1, Q1, n1, W1 ; N_sub = 2
    params = ([10.0, 20.0], 1e-9, 0.8, [5.0, 6.0])
    flat = sc.flatten_params(params, circuit_str="(R1|Q1)+W1", N_sub=2)
    assert flat == [10.0, 20.0, 1e-9, 0.8, 5.0, 6.0]


@pytest.mark.skipif(sc is None, reason="EISFitpython package not importable")
def test_flatten_params_W_does_not_desync_following_elements():
    # W appears BEFORE other elements. Before the fix, W was not counted in the
    # validation loop, so the per-element index desynced for everything after it.
    # W1 local, R1 local, Q1/n1 global ; N_sub = 2
    params = ([5.0, 6.0], [10.0, 20.0], 1e-9, 0.8)
    flat = sc.flatten_params(params, circuit_str="W1+(R1|Q1)", N_sub=2)
    assert flat == [5.0, 6.0, 10.0, 20.0, 1e-9, 0.8]


# --------------------------------------------------------------------------- #
# 4. Report labeling: unchanged for R/C/L/Q, correct for W/F.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(sc is None, reason="EISFitpython package not importable")
def test_label_alignment_RQ_circuit_unchanged():
    circuit = "(R1|Q1)+(R2|Q2)"
    N_sub, Temp = 3, [140, 150, 160]
    # R1 local, Q1/n1 global, R2 local, Q2/n2 global -> 3+1+1+3+1+1 = 10
    template = [True, False, False, True, False, False]
    popt = np.arange(10, dtype=float) + 1.0
    perror = popt * 0.01
    CorrM = np.eye(10)

    df, _ = sc.format_circuit_output(popt, perror, CorrM, circuit, N_sub, Temp, template)

    labels = [s.replace(" = ", "") for s in df["Fit_Params"]]
    assert labels == [
        "R1_140°C", "R1_150°C", "R1_160°C", "Q1", "n1",
        "R2_140°C", "R2_150°C", "R2_160°C", "Q2", "n2",
    ]
    np.testing.assert_array_equal(df["Value"].to_numpy(dtype=float), popt)


@pytest.mark.skipif(sc is None, reason="EISFitpython package not importable")
def test_label_alignment_with_warburg():
    circuit = "(R1|Q1)+W1"
    N_sub, Temp = 2, [25, 50]
    # R1 local, Q1 global, n1 global, W1 local -> 2+1+1+2 = 6
    template = [True, False, False, True]
    popt = np.arange(6, dtype=float) + 1.0
    perror = popt * 0.01
    CorrM = np.eye(6)

    df, _ = sc.format_circuit_output(popt, perror, CorrM, circuit, N_sub, Temp, template)

    labels = [s.replace(" = ", "") for s in df["Fit_Params"]]
    assert labels == ["R1_25°C", "R1_50°C", "Q1", "n1", "W1_25°C", "W1_50°C"]
    # Every parameter is accounted for and stays aligned with its value.
    assert len(df) == len(popt)
    np.testing.assert_array_equal(df["Value"].to_numpy(dtype=float), popt)


@pytest.mark.skipif(sc is None, reason="EISFitpython package not importable")
def test_label_alignment_with_finite_warburg():
    circuit = "(R1|Q1)+F1"
    N_sub, Temp = 2, [25, 50]
    # R1 local, Q1 global, n1 global, F1 global, F1_n global -> 2+1+1+1+1 = 6
    template = [True, False, False, False, False]
    popt = np.arange(6, dtype=float) + 1.0
    perror = popt * 0.01
    CorrM = np.eye(6)

    df, _ = sc.format_circuit_output(popt, perror, CorrM, circuit, N_sub, Temp, template)

    labels = [s.replace(" = ", "") for s in df["Fit_Params"]]
    assert labels == ["R1_25°C", "R1_50°C", "Q1", "n1", "F1", "F1_n"]
    np.testing.assert_array_equal(df["Value"].to_numpy(dtype=float), popt)
