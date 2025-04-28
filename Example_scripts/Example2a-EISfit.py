# -*- coding: utf-8 -*-

"""
EIS Data Analysis Script
-----------------------
This script performs electrochemical impedance spectroscopy (EIS) data analysis.
Citations: S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
DOI: https://doi.org/10.1039/d3ta06237f
"""

#%% Import Libraries
# Standard libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Set up path for custom EIS modules
current_dir = os.path.dirname(os.path.abspath(__file__))
EISmodules_path = os.path.join(current_dir, '..', 'EIS_Modules')
sys.path.insert(0, EISmodules_path)

# Import custom EIS modules
import data_extraction as dt      # type: ignore # For data file handling
import EIS_Batchfit as ebf      # type: ignore # For batch fitting
import EISFit_main as em     # type: ignore # For batch fitting


# import EIS data files
filenames=dt.get_eis_files(base_path='../EIS_Data', subfolder='Example-2')

# %%
#Case 1
# Extract and process EIS data for the first file in the dataset

# Read frequency and impedance data from file
f, Z = dt.readNEISYS(filenames[0])
# f: frequency points in Hz (numpy array)
# Z: complex impedance values (numpy array)

# Optional: Trim frequency range of the data
# Set fmin/fmax to numeric values to trim, or None to keep full range
f, Z = dt.trim_data(f, Z, fmin=None, fmax=None)

# Generate Nyquist plot of the raw data
# Options for plot_type: 'nyquist', 'bode', or 'both'
# - Nyquist: -Z'' vs Z' 
# - Bode: |Z| and phase angle vs frequency
a = em.plot_fit(f, Z, Z_fit=None, plot_type='nyquist')

#%%
# Case 1 EIS data fitting 

# Define equivalent circuit model
# (R1|Q1)+Q3 represents:
# - R1 and Q1 in parallel, followed by 
# - Q3 in series
circuit = "(R1|Q1)+Q2"

# Initial parameter guesses:
# [R1, Q1, n1, Q2, n2] where:
# - R1: Resistance (ohms)
# - Q1, Q2: Constant phase element parameters (F/s^(n-1))
# - n1, n2: CPE exponents (dimensionless, 0-1)
params = [1.1e+05, 1.09e-9, 8.02314862e-01, 1.2e-07, 5.3e-01] 

# Select weighting method for the fitting:
# - 'U': Unity (uniform) weighting
# - 'P': Proportional weighting
# - 'M': Modulus weighting
weight_mtd='M'

# Choose optimization algorithm:
# - 'lm': Levenberg-Marquardt (for unconstrained problems)
# - 'trf': Trust Region Reflective
# - 'dogbox': Dogleg algorithm
method='lm'

# Define parameter bounds for constrained optimization
# Empty lists mean unconstrained optimization
UB = []  # Upper bounds
LB = []  # Lower bounds

# Frequency range limits for data trimming
# None means no trimming at that end
min=None  # Minimum frequency (Hz)
max=None  # Maximum frequency (Hz)

# Perform EIS fitting and generate comprehensive report
# Returns fitted parameters and their estimated errors
fit_params, fit_perror = em.full_EIS_report(f,Z,params,circuit,UB,LB,weight_mtd,method,single_chi='No',plot_type='both')

# %%
