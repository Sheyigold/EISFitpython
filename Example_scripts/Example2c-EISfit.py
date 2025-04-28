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


#%%# 
# Case 3 

# Extract single frequency data
f, Z = dt.readNEISYS(filenames[2])

#%%
# Case 3 EIS data fitting 


weight_mtd='M'

method='lm'

UB = []  # Upper bounds
LB = []  # Lower bounds

min=None  # Minimum frequency (Hz)
max=None  # Maximum frequency (Hz)

circuit = "(R1|Q1)+(R2|Q1)+Q2"

params = [1.8e6, 1.3e-10, 0.8, 8.2e6, 1.3e-9, 0.9, 5.7e-7, 0.6]
# The params variable is defined as the initial guess for the parameters of the equivalent circuit model.

fit_params, fit_perror = em.full_EIS_report(f,Z,params,circuit,UB,LB,weight_mtd,method,single_chi='No',plot_type='both')


# %%
