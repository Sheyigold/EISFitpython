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


# Set up path for custom EIS modules
current_dir = os.path.dirname(os.path.abspath(__file__))
EISmodules_path = os.path.join(current_dir, '..', 'EIS_Modules')
sys.path.insert(0, EISmodules_path)

# Import custom EIS modules
import data_extraction as dt      # type: ignore # For data file handling
import singlechi as sc           # type: ignore # For single chi analysis
import EIS_Batchfit as ebf      # type: ignore # For batch fitting

#%% Data Processing
# Get EIS data files
filenames=dt.get_eis_files(base_path='../EIS_Data', subfolder='Example-3-4')

# Extract frequency and impedance data from NEISYS spectrometer files
f, Z = dt.full_readNEISYS(filenames)

# Split data at 1MHz frequency point
sublist, _ = dt.split_array(f, Z=None, split_freq=1e6)
N_sub = len(sublist)

# Temperature points for measurements (in Celsius)
Temp = np.array([140, 150, 160, 170, 180, 190, 200])

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
fit_params, fit_perror, Z_fit, figN, figB = sc.Single_chi_report(f, Z, params, Temp, circuit_str)

#%% Arrhenius Plot Analysis
# Extract resistance components
Rb = np.abs(fit_params[0:7])     # Bulk resistance
Rgb = np.abs(fit_params[9:16])   # Grain boundary resistance

# Sample dimensions
D = 0.8    # Pellet diameter in cm
l = 0.21   # Pellet thickness in cm

# Calculate total resistance
Rt = Rb + Rgb

# Temperature points for analysis (in Celsius)
T = np.array([140, 150, 160, 170, 180, 190, 200])

# For multiple components:
R_values = [Rb, Rgb, Rt]
labels = ['Bulk', 'Gb', 'Total']
conductivities = ebf.plot_arrhenius(R_values, T, D, l, labels=labels)

con1 = conductivities[0]
con2 = conductivities[1]
con3 = conductivities[2]
#%% Conductivity Analysis
# Calculate conductivity with error analysis
Rb_err = fit_perror[0:7]
ebf.sigma_report(Rb, Rb_err, l, con1, T)

Rgb_err = fit_perror[8:15]
ebf.sigma_report(Rgb, Rgb_err, l, con2, T)

#%% Effective Capacitance Calculation
# Extract parameters for capacitance calculation


# Effective capacitance calculation
cpe_params = {
     'bulk': {
          'R': fit_params[0:7],    'R_err': fit_perror[0:7],
          'Q': fit_params[7],      'Q_err': fit_perror[7],
          'n': fit_params[8],      'n_err': fit_perror[8]
     },
     'gb': {
          'R': fit_params[9:16],   'R_err': fit_perror[9:16],
          'Q': fit_params[16:23],  'Q_err': fit_perror[16:23],
          'n': fit_params[23:30],  'n_err': fit_perror[23:30]
     }
}
Cb_eff = ebf.C_eff(cpe_params['bulk']['R'], cpe_params['bulk']['R_err'],
                         cpe_params['bulk']['Q'], cpe_params['bulk']['Q_err'],
                         cpe_params['bulk']['n'], cpe_params['bulk']['n_err'], T)
Cgb_eff = ebf.C_eff(cpe_params['gb']['R'], cpe_params['gb']['R_err'],
                          cpe_params['gb']['Q'], cpe_params['gb']['Q_err'],
                          cpe_params['gb']['n'], cpe_params['gb']['n_err'], T)

# %%
