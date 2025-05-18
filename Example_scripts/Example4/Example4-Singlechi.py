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
from EISFitpython import singlechi as sc           
from EISFitpython import EISFit_main as em      
from EISFitpython import EIS_Batchfit as ebf    

#%% Data Processing
# Get EIS data files
filenames=dt.get_eis_files(base_path='../../EIS_Data', subfolder='Example-3-4')

# Extract frequency and impedance data from NEISYS spectrometer files
f, Z = dt.stack_NEISYS_files(filenames)

# Split data at 1MHz frequency point
sublist, _ = dt.split_array(f, Z=None, split_freq=1e6)
N_sub = len(sublist)

# Temperature points for measurements (in Celsius)
Temp = np.array([140, 150, 160, 170, 180, 190, 200])

# Generate plot for the selected files
# Plotting options: 'nyquist', 'bode', or 'both'
a=em.plot_fit(f, Z,plot_type='both')
#%% Fitting Parameters
# Equivalent circuit model
circuit_str = "(R1|Q1)+(R2|Q2)+Q3"  

# Initial parameter values for circuit elements
# Bulk resistance values for different temperatures
R1 = [178808.66948441, 134937.40033909, 98229.05551748, 71382.87890687,
          53085.29295227, 31963.07882123, 19634.78343412]
Q1 = 1.3e-11    # Bulk capacitance
n1 = 0.9        # Bulk CPE exponent

# Grain boundary parameters
R2 = [2181790.13666849, 1231143.00488342, 711584.00646881,
          421614.55472459, 252776.26680737, 159093.45155081,
          102970.19913917]  # Grain boundary resistance
Q2 = [1.48e-10 for _ in range(N_sub)]  # Grain boundary capacitance
n2 = [0.9 for _ in range(N_sub)]       # Grain boundary CPE exponent
Q3 = [1.27389324e-06 for _ in range(N_sub)]
n3 = [0.5 for _ in range(N_sub)]       # CPE exponent

# Combine parameters for fitting
params = (R1, Q1, n1, R2, Q2, n2, Q3, n3)

# Perform fitting and generate reports
fit_params, fit_perror, Z_fit = sc.Single_chi_report(f, Z, params, Temp, circuit_str,weight_mtd='M')

#%% Arrhenius Plot Analysis
# Extract resistance components
Rb = np.abs(fit_params[0:7])   # Bulk resistance
Rb_err = np.abs(fit_perror[0:7])   # Error in bulk resistance from first column
Rgb = np.abs(fit_params[9:16])   # Grain boundary resistance
Rgb_err = np.abs(fit_perror[9:16])  # Error in grain boundary resistance from third column

# Sample dimensions
D = 0.8    # Pellet diameter in cm
l = 0.21   # Pellet thickness in cm

# Calculate total resistance
Rt = Rb + Rgb
Rt_err = np.sqrt(Rb_err**2 + Rgb_err**2)  # Error in total resistance

# Temperature points for analysis (in Celsius)
T = np.array([140, 150, 160, 170, 180, 190, 200])

# For multiple components:
R_values = [Rb, Rgb, Rt]
R_errors = [Rb_err, Rgb_err, Rt_err]
labels = ['Bulk', 'Gb', 'Total']
conductivities, conductivity_errors = ebf.plot_arrhenius(R_values,R_errors, T, D, l, labels=labels)

#%% Effective Capacitance Calculation
# Extract parameters for capacitance calculation

# List of all resistance values
R_values = [Rb, Rgb, Rt]
R_errors = [Rb_err, Rgb_err, Rt_err]

# List of all Q (CPE) values
Q_values = [fit_params[7], fit_params[16:23]]
Q_errors = [fit_perror[7], fit_perror[16:23]]

# List of all n (CPE exponent) values 
n_values = [fit_params[8], fit_params[23:30]]
n_errors = [fit_perror[8], fit_perror[23:30]]


# Example usage with your data:
labels = ['Bulk', 'Grain Boundary']
results = ebf.C_eff(R_values, R_errors, Q_values, Q_errors, 
                n_values, n_errors, T, labels=labels)

# Access results
bulk_C, bulk_C_err = results[0]
gb_C, gb_C_err = results[1]


# %%

# %%
