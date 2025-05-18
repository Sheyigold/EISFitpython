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


# Import custom EIS modules
from EISFitpython import data_extraction as dt     
from EISFitpython import EISFit_main as em    

# import EIS data files
# The get_eis_files function is used to retrieve EIS data files from a specified base path and subfolder.
# The base path is set to '../../EIS_Data' and the subfolder is 'Example-2'.
filenames=dt.get_eis_files(base_path='../../EIS_Data', subfolder='Example-2')

# %%
# Extract and process EIS data 
f, Z = dt.readNEISYS(filenames[1])
# f: frequency points in Hz (numpy array)
# Z: complex impedance values (numpy array)

# Optional: Trim frequency range of the data
# Set fmin/fmax to numeric values to trim, or None to keep full range
# The trim_data function is used to trim the frequency and impedance data based on specified minimum and maximum frequencies.
f,Z = dt.trim_data(f, Z, fmin=5, fmax=None)

# Generate Nyquist plot of the raw data
# Options for plot_type: 'nyquist', 'bode', or 'both'
# - Nyquist: -Z'' vs Z' 
# - Bode: |Z| and phase angle vs frequency
a = em.plot_fit(f, Z, Z_fit=None, plot_type='both')

#%%
# EIS data fitting 

# Define equivalent circuit model
# (R1|Q1)+Q3 represents:
# - R1 and Q1 in parallel, followed by 
# - Q3 in series
circuit = "(R1|Q1)+(R2|Q2)+Q3"

# Initial parameter guesses:
# [R1, Q1, n1, Q2, n2] where:
# - R1: Resistance (ohms)
# - Q1, Q2: Constant phase element parameters (F/s^(n-1))
# - n1, n2: CPE exponents (dimensionless, 0-1)
params = [4.8e4, 1.3e-10, 0.8, 9.2e4, 1.3e-9,0.9, 5.7e-7, 0.6]

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
fit_params, fit_perror, Z_fit = em.full_EIS_report(f,Z,params,circuit,UB,LB,weight_mtd,method,single_chi='No',plot_type='nyquist')

# %%
