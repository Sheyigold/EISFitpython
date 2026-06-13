"""
Data Extraction and Processing Functions for EIS (Electrochemical Impedance Spectroscopy) Data

This module provides functions to read and process EIS data from various file formats
including NEISYS, ZAHNER_IM6, CSV, and dynamic EIS data. It includes utilities for:
- Reading EIS files from specified directories
- Parsing data from different spectrometer formats
- Trimming frequency ranges
- Splitting data arrays
- Processing both single and multiple EIS measurements

Author: Sheyi Clement Adediwura

'Please cite the following paper if you use this code:
S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
https://doi.org/10.1039/d3ta06237f

"""

#External libraries
import numpy as np
import os
import glob
import pandas as pd
import re
import warnings


def get_eis_files(base_path='.', subfolder=None):
    """
    Get all data files from specified location.

    Args:
        base_path (str): Base path to data folder (default: current directory)
        subfolder (str, optional): Name of subfolder containing data files.
            If None, searches base_path directly.

    Returns:
        list: Sorted list of file paths
    """
    # Build search path
    search_path = os.path.join(base_path, subfolder) if subfolder else base_path

    # Get all files (use * to match everything)
    file_location = os.path.join(search_path, '*')
    filenames = sorted([f for f in glob.glob(file_location) if os.path.isfile(f)])


    # Print summary
    print(f'\nFound {len(filenames)} total files:')
    if filenames:
              for i, file in enumerate(filenames, 1):
                print(f'  {i}. {os.path.basename(file)}')
    else:
        print(f'\nNo files found in {os.path.abspath(search_path)}')

    return filenames


def parse_NEISYS_data(lines):
    """Helper function to parse NEISYS data from a list of lines."""

    # Find the line containing frequency header
    start_index = 0
    for i, line in enumerate(lines):
        if 'Freq. [Hz]' in line:  # Changed from 'Freq.[Hz]' to match exact header
            start_index = i + 1
            break

    exp_data = lines[start_index:]
    f, Z = [], []
    for line in exp_data:
        try:
            each = line.strip().split('\t')  # Split on tabs since data is tab-separated
            if len(each) >= 3:  # Ensure we have all three columns
                freq = float(each[0])
                real = float(each[1])
                imag = float(each[2])
                if freq != 0:
                    f.append(freq)
                    Z.append(complex(real, imag))
        except (ValueError, IndexError):
            continue  # Skip lines that can't be converted or are incomplete

    return np.array(f), np.array(Z)

def readNEISYS(filename):
    """Function to extract .txt file from NEISYS spectrometer."""
    try:
        with open(filename, 'r') as data:
            lines = data.readlines()
        return parse_NEISYS_data(lines)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error reading file {filename}: {str(e)}")

def readTXT(filename):
    """
    Function to extract data from .txt files with flexible header handling.

    Supports:
    - Files with or without headers
    - Various header formats (with/without #)
    - Common header terms like 'freq', 'hz', 'real', 'imag'
    - Tab, comma, or space-separated values

    Args:
        filename (str): Path to .txt file

    Returns:
        tuple: (frequencies array, complex impedances array)
    """
    f, Z = [], []

    try:
        with open(filename, 'r') as data:
            lines = data.readlines()

        # Common header keywords
        header_keywords = ['freq', 'hz', 'real', 'imag', 'z']

        # Find where data starts
        start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip().lower()

            # Skip empty lines
            if not line:
                continue

            # Check if line contains header keywords
            if any(keyword in line.lower() for keyword in header_keywords):
                start_idx = i + 1
                continue

            # Try to parse as data
            try:
                values = re.split(r'[\t,\s]+', line)
                if len(values) >= 3:
                    float(values[0])
                    float(values[1])
                    float(values[2])
                    start_idx = i
                    break
            except (ValueError, IndexError):
                continue

        # Process data lines
        for line in lines[start_idx:]:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                values = re.split(r'[\t,\s]+', line)
                if len(values) >= 3:
                    freq = float(values[0])
                    real = float(values[1])
                    imag = float(values[2])

                    if freq != 0:  # Skip zero frequency
                        f.append(freq)
                        Z.append(complex(real, imag))

            except (ValueError, IndexError):
                print(f"Warning: Skipping invalid line: {line}")
                continue

        if not f:
            raise ValueError(f"No valid data found in {filename}")

        return np.array(f), np.array(Z)

    except Exception as e:
        raise Exception(f"Error reading {filename}: {str(e)}")

def readCSV(filename):
    """
    Function to extract data from CSV files without headers.

    Expects:
    - 3 columns: frequency, real Z, imaginary Z
    - No headers or comments
    - Comma or semicolon delimited values

    Args:
        filename (str): Path to .csv file

    Returns:
        tuple: (frequencies array, complex impedances array)
    """
    try:
        # Try numpy's loadtxt first - fastest for simple numeric data
        try:
            data = np.loadtxt(filename, delimiter=',')
            if data.shape[1] >= 3:
                f = data[:, 0]  # First column for frequency
                Z = data[:, 1] + 1j * data[:, 2]  # Real + imaginary parts
                # Filter zero frequencies
                mask = f != 0
                return f[mask], Z[mask]
        except Exception:
            # If comma delimiter failed, try semicolon
            data = np.loadtxt(filename, delimiter=';')
            if data.shape[1] >= 3:
                f = data[:, 0]
                Z = data[:, 1] + 1j * data[:, 2]
                mask = f != 0
                return f[mask], Z[mask]

    except Exception as e:
        raise Exception(f"Error reading {filename}: {str(e)}")

    raise ValueError(f"No valid data found in {filename}")

def stack_NEISYS_files(filenames, return_lengths=False):
    """
    Function to extract multiple .txt files from NEISYS spectrometer and store them in a single array.

    Input
    ----------
    filenames: list of strings
        List of filenames of .txt files to extract impedance data from
    return_lengths: bool, optional
        If True, also return a list with the number of points contributed by
        each file. These lengths can be passed straight to ``split_array``
        (``lengths=`` argument) for an exact, unambiguous split. Default False
        preserves the original ``(f, Z)`` return signature.

    Ouput
    -------
    combined_frequencies, combined_impedances : np.ndarray
        Arrays of frequencies and complex impedances
    (optionally) lengths : list of int
        Per-file point counts, returned only when ``return_lengths=True``.
    """
    all_frequencies = []
    all_impedances = []
    lengths = []

    for filename in filenames:
        with open(filename, 'r') as data:
            lines = data.readlines()
        f, Z = parse_NEISYS_data(lines)
        all_frequencies.extend(f)
        all_impedances.extend(Z)
        lengths.append(len(f))

    f_arr = np.array(all_frequencies)
    Z_arr = np.array(all_impedances)

    if return_lengths:
        return f_arr, Z_arr, lengths
    return f_arr, Z_arr


def stack_TXT_files(filenames):
    """
    Function to extract multiple .txt files and store them in a single array.
    Uses improved readTXT with flexible header handling.

    Parameters
    ----------
    filenames: list of strings
        List of filenames of .txt files to extract impedance data from

    Returns
    -------
    combined_frequencies, combined_impedances : np.ndarray
        Arrays of frequencies and complex impedances from all files
    """
    all_frequencies = []
    all_impedances = []

    for filename in filenames:
        try:
            f, Z = readTXT(filename)  # Use the improved readTXT function
            all_frequencies.extend(f)
            all_impedances.extend(Z)
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue

    if not all_frequencies:
        raise ValueError("No valid data found in any of the files")

    return np.array(all_frequencies), np.array(all_impedances)

def stack_CSV_files(filenames):
    """
    Function to extract multiple .csv files and store them in a single array.
    Uses improved readCSV with flexible header handling.

    Parameters
    ----------
    filenames: list of strings
        List of filenames of .csv files to extract impedance data from

    Returns
    -------
    combined_frequencies, combined_impedances : np.ndarray
        Arrays of frequencies and complex impedances from all files
    """
    all_frequencies = []
    all_impedances = []

    for filename in filenames:
        try:
            f, Z = readCSV(filename)  # Use the improved readCSV function
            all_frequencies.extend(f)
            all_impedances.extend(Z)
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue

    if not all_frequencies:
        raise ValueError("No valid data found in any of the files")

    return np.array(all_frequencies), np.array(all_impedances)



def trim_data(f, Z, fmin=None, fmax=None):
    """
    Trim frequency and impedance data based on specified minimum and maximum frequencies.

    Parameters:
    - f (1D array): A 1D numpy array containing frequency values.
    - Z (1D array): A 1D numpy array containing impedance values.
    - fmin (float, optional): Minimum frequency value for trimming.
    - fmax (float, optional): Maximum frequency value for trimming.

    Returns:
    - f_trimmed (1D array): Trimmed frequency values.
    - Z_trimmed (1D array): Corresponding trimmed impedance values.
    """

    # Initialize mask with all True values
    mask = np.ones_like(f, dtype=bool)

    # Update mask based on specified fmin and fmax
    if fmin is not None:
        mask &= (f >= fmin)
    if fmax is not None:
        mask &= (f <= fmax)
    f_trimmed = f[mask]
    Z_trimmed = Z[mask]

    return f_trimmed, Z_trimmed

def split_array(f, Z=None, split_freq=None, lengths=None):
    """
    Split a stacked frequency/impedance array into per-dataset sublists.

    This is the routine that separates a stacked variable-temperature spectrum
    (e.g. produced by ``stack_NEISYS_files``) back into one sublist per dataset.
    It is the backbone of the global-local (single chi-square) workflow.

    Three splitting strategies are available, in order of robustness:

    1. ``lengths`` given  -> exact split by explicit per-dataset point counts
       (recommended; bypasses all heuristics).
    2. default            -> sweep-reset detection. A new dataset begins wherever
       the frequency reverses direction back toward its starting extreme. This is
       robust to floating-point noise and to datasets that do not share an
       identical maximum frequency.
    3. ``split_freq`` given but no reset detected -> legacy fallback that starts a
       new sublist whenever the frequency is close to ``split_freq``
       (``np.isclose`` instead of exact ``==``).

    Backward compatibility
    ----------------------
    The signature only gains optional keyword arguments, so every existing call
    keeps working. For well-formed descending sweeps that reset at the same
    maximum frequency, this function returns exactly the same partition as the
    original implementation (see test_eisfit_patches.py::test_golden_split).

    Parameters
    ----------
    f : 1D array
        Stacked frequency values.
    Z : 1D array, optional
        Stacked impedance values corresponding to ``f``.
    split_freq : float, optional
        Deprecated. Retained for backward compatibility; only used as a fallback
        when no sweep reset can be detected.
    lengths : sequence of int, optional
        Explicit number of points per dataset. When provided, the split is exact
        and all heuristics are bypassed.

    Returns
    -------
    sublists_f : list of 1D numpy arrays
    sublists_Z : list of 1D numpy arrays (empty list if Z is None)
    """
    f = np.asarray(f)
    n = len(f)

    if lengths is not None:
        # --- Strategy 1: exact, explicit split (bulletproof) ---
        total = int(np.sum(lengths))
        if total != n:
            raise ValueError(
                f"sum(lengths)={total} does not match len(f)={n}. "
                f"Pass the per-dataset point counts in order."
            )
        bounds = np.cumsum(np.asarray(lengths, dtype=int))[:-1]

    elif n < 2:
        # Nothing to split.
        bounds = np.array([], dtype=int)

    else:
        # --- Strategy 2: sweep-reset detection (robust default) ---
        df = np.diff(f)
        descending = np.median(df) < 0  # dominant sweep direction
        if descending:
            bounds = np.where(df > 0)[0] + 1   # frequency jumps back up -> new sweep
        else:
            bounds = np.where(df < 0)[0] + 1   # frequency jumps back down -> new sweep

        # --- Strategy 3: legacy fallback (only if no reset found) ---
        if bounds.size == 0 and split_freq is not None:
            warnings.warn(
                "split_array: no sweep reset detected; falling back to "
                "split_freq matching. Consider passing lengths= for an exact "
                "split.", RuntimeWarning, stacklevel=2,
            )
            bounds = np.array(
                [i for i in range(1, n) if np.isclose(f[i], split_freq)],
                dtype=int,
            )

    indices = np.split(np.arange(n), bounds)
    sublists_f = [f[idx] for idx in indices]

    if Z is not None:
        Z = np.asarray(Z)
        sublists_Z = [Z[idx] for idx in indices]
    else:
        sublists_Z = []

    return sublists_f, sublists_Z
