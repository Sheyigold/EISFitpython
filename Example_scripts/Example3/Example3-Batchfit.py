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
from EISFitpython import EIS_Batchfit as ebf  

# import EIS data files
filenames=dt.get_eis_files(base_path='../../EIS_Data', subfolder='Example-3-4')

#%%
 # Stackplot of EIS data
# Temperature points corresponding to the files in the batch
Temp = np.array([140, 150, 160, 170, 180, 190, 200])

# Create Nyquist plot for the selected files
a=ebf.Nyq_stack_plot(filenames, Temp)

#%%
# Define the equivalent circuit model for fitting
circuit = "(R1|Q1)+(R2|Q1)+Q2"
# Circuit components:
# - R1|Q1: Parallel combination of resistance and constant phase element  (bulk response)
# - R2|Q1: Parallel combination of resistance and constant phase element (grain boundary)
# - Q2: Series constant phase element (electrode response)

# Initial parameter guesses for the circuit model
params = [1.8e6, 1.3e-11, 0.9,8.2e6, 1.3e-9,0.9, 5.7e-7, 0.6]
# Parameters order: [R1, Q1,n1, R2, Q1, n1, Q2, n2]

# Select the weighting method for the fitting algorithm
weight_mtd='M'
# 'M': Modulus weighting - weights each point by |Z|

# Choose optimization algorithm
method='lm'
# Levenberg-Marquardt algorithm: efficient for unconstrained nonlinear least squares problems
# Alternative options: 'trf' (Trust Region Reflective) or 'dogbox'

# Define parameter bounds for constrained optimization
# Empty lists indicate unconstrained optimization
UB = []  # Upper bounds
LB = []  # Lower bounds

# Perform batch fitting of all files
fit_params, fit_perror = ebf.Batch_fit(filenames, params, circuit,  Temp, UB, LB, weight_mtd, method, 
                                      single_chi='No', min_value=None, max_value=None)

# %%
# Extract fitted resistance parameters for different components
Rb = fit_params[:,0]  # Bulk resistance from first column
Rgb = fit_params[:,3]  # Grain boundary resistance from third column
Rb_err = fit_perror[:,0]  # Error in bulk resistance from first column
Rgb_err = fit_perror[:,3]  # Error in grain boundary resistance from third column
# Sample geometry parameters for conductivity calculations
D = 0.8  # Pellet diameter in cm 
l = 0.21  # Pellet thickness in cm

# Calculate total resistance
Rt = Rb + Rgb  # Sum of bulk and grain boundary resistances
Rt_err = np.sqrt(Rb_err**2 + Rgb_err**2)  # Error in total resistance
# Temperature points for Arrhenius plot (in Celsius)
T = np.array([140, 150, 160, 170, 180, 190, 200])
# For multiple components:
R_values = [Rb, Rgb, Rt]
labels = ['Bulk', 'Gb', 'Total']
R_errors = [Rb_err, Rgb_err, Rt_err]
conductivities, conductivity_errors = ebf.plot_arrhenius(R_values,R_errors, T, D, l, labels=labels)
# For single component:
#ax, conductivities = ebf.plot_arrhenius(Rb, T, D, l, labels='Bulk')

# %%

#Effective Capacitance Calculation
# For CPE elements, calculate the effective capacitance using:
#   C_eff = (R^(1-n) * Q)^(1/n)
Rb = fit_params[:, 0]
Rb_err = fit_perror[:, 0]
Qb = fit_params[:, 1]
Qb_err = fit_perror[:, 1]
nb = fit_params[:, 2]
nb_err = fit_perror[:, 2]
Rgb = fit_params[:, 3]
Rgb_err = fit_perror[:, 3]
Qgb = fit_params[:, 4]
Qgb_err = fit_perror[:, 4]
ngb = fit_params[:, 5]
ngb_err = fit_perror[:, 5]

# List of all resistance values
R_values = [Rb, Rgb, Rt]
R_errors = [Rb_err, Rgb_err, Rt_err]

# List of all Q (CPE) values
Q_values = [Qb, Qgb]
Q_errors = [Qb_err, Qgb_err]

# List of all n (CPE exponent) values 
n_values = [nb, ngb]
n_errors = [nb_err, ngb_err]


# Example usage with your data:
labels = ['Bulk', 'Grain Boundary']
results = ebf.C_eff(R_values, R_errors, Q_values, Q_errors, 
                n_values, n_errors, T, labels=labels)

# Access results
#bulk_C, bulk_C_err = results[0]
#gb_C, gb_C_err = results[1]

# %%
