 # -*- coding: utf-8 -*-
"""
EIS Data Analysis Script
-----------------------
This script performs electrochemical impedance spectroscopy (EIS) data analysis.
Citations: S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
DOI: https://doi.org/10.1039/d3ta06237f
"""

#%% Library Imports
# System and File Operations
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


# Custom EIS Module Imports
from EISFitpython import EISFit_main as em       

#%% EIS Data Simulation
# This section simulates EIS data using a defined equivalent circuit model.

# Define equivalent circuit model and parameters
# Circuit model options:

#Example 1
#circuit = "(R1|C1)"  # Simple parallel RC circuit
#params = [20, 100e-6]  # [R1 (Ω), C1 (F)]

#Example 2
#circuit = "(R1|C1)+(R2|C2)"  # Two parallel RC circuits
#params = [20, 100e-6, 10, 0.1]  # For "(R1|C1)+(R2|C2)"

#Example 3  
#circuit = "R1+(R2|Q1)"           # RQC circuit with a resistor in series
# Note: Q1 is a constant phase element (CPE) with parameters [Q, n]
# Q is the capacitance and n is the exponent (0 < n < 1)
# Q1 = [Q, n] = [10e-7, 0.9]
#params = [5, 20, 10e-6, 0.9]  # For "R1+(R2|Q1)"


#Example 4
#circuit = "R1+(R2|Q1|C1)"          # RQC circuit with a resistor in series
# Note: Q1 is a constant phase element (CPE) with parameters [Q, n]
# Q is the capacitance and n is the exponent (0 < n < 1)
# Q1 = [Q, n] = [10e-7, 0.9]
#params = [10e6, 60e6, 12e-10, 0.85, 1e-9]  # For "R1+(R2|Q1|C1)"

#Example 5
#circuit = "(R1|Q1)+(R2|Q2)"    # Two RQ circuits in series with resistor
# Note: Q1 is a constant phase element (CPE) with parameters [Q, n]
# Q is the capacitance and n is the exponent (0 < n < 1)
# Q1 = [Q, n] = [2e-12, 0.85]
# Q2 = [Q, n] = [4e-10, 0.75]
#params = [50e6, 2e-12, 0.85, 80e6, 4e-10,0.75]  # For "(R1|C1)+(R2|C2)+R3"

#Example 6
circuit = "(R1|Q1)+(R2|Q2)+Q3"    # Two RQ circuits in series with a constant phase element (CPE)
# Note: Q1 is a constant phase element (CPE) with parameters [Q, n]
# Q is the capacitance and n is the exponent (0 < n < 1)
# Q1 = [Q, n] = [2e-12, 0.85]
# Q2 = [Q, n] = [4e-10, 0.75]
# Q3 = [Q, n] = [1e-7, 0.7]
params = [50e6, 2e-12, 0.85, 80e6, 4e-10,0.75, 1e-7,0.7]  # For "(R1|C1)+(R2|C2)+R3"

#Example 7
#circuit = "(R1|Q1)+(R2|Q2)+R3+L1"    # Two RQ circuits in series with a resistor and an inductor
# Note: Q1 is a constant phase element (CPE) with parameters [Q, n]
# Q is the capacitance and n is the exponent (0 < n < 1)
# Q1 = [Q, n] = [2e-12, 0.85]
# Q2 = [Q, n] = [4e-10, 0.75]
# Q3 = [Q, n] = [1e-7, 0.7]
# L1 is the inductance with a value of 1e-4 H

#params = [50e6, 2e-12, 0.85, 80e6, 4e-10,0.75, 1e7, 1e-4]  # For "(R1|C1)+(R2|C2)+R3"

#Example 8
#circuit = "(R1|Q1)+(R2|Q2)+L1+W1"    # Two RQ circuits in series with an inductor and a Warburg element
# Note: Q1 is a constant phase element (CPE) with parameters [Q, n]
# Q is the capacitance and n is the exponent (0 < n < 1)
# Q1 = [Q, n] = [2e-12, 0.85]
# Q2 = [Q, n] = [4e-10, 0.75]
# L1 is the inductance with a value of 1e-4 H   
# W1 is the Warburg element with a value of 1e-3 
#params = [50e6, 2e-12, 0.85, 80e6, 4e-10,0.75, 2.5e-3, 1e7] # For "(R1|Q1)+(R2|Q2)+L1+W1"



# Frequency range settings
# Define the frequency range for simulation
start_freq = 3e-3 # Starting frequency in Hz
end_freq = 1e7     # Ending frequency in Hz
no_points = 50       # Number of frequency points to simulate


# Generate simulated impedance data and plot it
# Options for plot_type: 'nyquist', 'bode', or 'both'
# - Nyquist: -Z'' vs Z' 
# - Bode: |Z| and phase angle vs frequency

F_sim, Z_sim = em.predict_Z(start_freq, end_freq, no_points, params, circuit)
# Returns complex impedance array with real and imaginary components

#%%
# Plotting the simulated data
# Set up the figure for Nyquist plot
param_dict= {'marker': 'o', 'color': 'k', 'ms': '6', 'ls': '', 
                  'mfc': 'grey'}

# Nyquist plot
# Set up the figure for Nyquist plot
fig_nyquist = plt.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
ax_nyquist = fig_nyquist.add_subplot(111)

# generate Nyquist plot
# Note: Z_sim is a complex array with real and imaginary components
plot_nyq=em.nyquistPlot(ax_nyquist, Z_sim, param_dict)

# Save the figure in current directory
fig_nyquist.savefig('nyquist_plot.svg', dpi=300, bbox_inches='tight')
plt.show()
#%%
# Bode plot
# Set up the figure for Bode plot
fig1B, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Plot imaginary and phase only (traditional Bode plot)
bd=em.bodePlot([ax1, ax2], F_sim, Z_sim, param_dict)

# Save the figure in current directory
fig1B.savefig('bode_plot.svg', dpi=300, bbox_inches='tight')
plt.show()
#%%
# Bode plot variations
# this section generates a Bode plot for the imaginary part of the impedance data.

# Set up the figure for Bode plot
fig2B, ax = plt.subplots(figsize=(6, 6), facecolor='w', edgecolor='k')
param_dict = {'marker': 'o', 'color': 'blue', 'ms': '6', 'ls': '', 
              'mfc': 'grey'}

# Generate Bode plot for imaginary part of impedance data
em.bodePlot([ax], F_sim, Z_sim, param_dict, plot_types=['imaginary'])

# Save the figure in current directory
#fig2B.savefig('bode_plot.svg', dpi=300, bbox_inches='tight')
plt.show()
#%%
# Bode plot variations

# This section generates a Bode plot for the real part of the impedance data.

# Set up the figure for Bode plot
fig3B, ax = plt.subplots(figsize=(6, 6), facecolor='w', edgecolor='k')
param_dict = {'marker': 's', 'color': 'r', 'ms': '6', 'ls': '', 
              'label': 'simulated data', 'mfc': 'green'}

# Generate Bode plot for real part of impedance data
em.bodePlot([ax], F_sim, Z_sim, param_dict, plot_types=['real'])

# Save the figure in current directory
#fig2B.savefig('bode_plot.svg', dpi=300, bbox_inches='tight')
plt.show()
#%%
# Bode plot variations
# This section generates a Bode plot for the magnitude of the impedance data.

# Set up the figure for Bode plot
fig4B, ax = plt.subplots(figsize=(6, 6), facecolor='w', edgecolor='k')
param_dict = {'marker': 'h', 'color': 'cyan', 'ms': '6', 'ls': '', 
              'label': 'simulated data', 'mfc': 'k'}

# Generate Bode plot for magnitude of impedance data
em.bodePlot([ax], F_sim, Z_sim, param_dict, plot_types=['magnitude'])

# Save the figure in current directory
#fig2B.savefig('bode_plot.svg', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Plot all Bode plot components
# This section generates a Bode plot for all components of the impedance data.
# Set up the figure for Bode plot
param_dict = {'marker': 'o', 'color': 'k', 'ms': '6', 'ls': '', 
              'label': 'simulated data', 'mfc': 'grey'}
fig5B, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

# Generate Bode plot for all components of impedance data
bp=em.bodePlot([ax1, ax2, ax3, ax4], F_sim, Z_sim, param_dict,
         plot_types=['magnitude', 'phase', 'real', 'imaginary'])

# Save the figure in current directory
fig5B.savefig('bode_plot_all.svg', dpi=300, bbox_inches='tight')
plt.show()

# %%

#Modulus plot
# This section generates modulus plots for the simulated data.
# Define parameters for modulus plot
diameter = 0.1  # Diameter in cm
thickness = 0.01  # Thickness in cm

# Set up the figure for Bode plot
param_dict= {'marker': 'o', 'color': 'k', 'ms': '6', 'ls': '', 
                 'label': 'simulated data', 'mfc': 'grey'}


# Magnitude of modulus plot
fig_mag = plt.figure(figsize=(6, 6))
ax_mag = fig_mag.add_subplot(111)
em.modulus_plot([ax_mag], F_sim, Z_sim, param_dict, diameter, thickness, 
             )
plt.show()
#%%
# Set up the figure for Bode plot
param_dict= {'marker': 'o', 'color': 'brown', 'ms': '6', 'ls': '', 
                 'label': 'simulated data', 'mfc': 'brown'}

# Real part of modulus plot
fig_real = plt.figure(figsize=(6, 6))
ax_real = fig_real.add_subplot(111)
em.modulus_plot([ax_real], F_sim, Z_sim, param_dict, diameter, thickness, 
             plot_types=['real'])

# Imaginary part of modulus plot
fig_imag = plt.figure(figsize=(6, 6))
ax_imag = fig_imag.add_subplot(111)
em.modulus_plot([ax_imag], F_sim, Z_sim, param_dict, diameter, thickness, 
             plot_types=['imaginary'])
plt.show()
#%%
# Combined plots (all three separate figures)
figs = []
param_dict= {'marker': '*', 'color': 'k', 'ms': '6', 'ls': '', 
                 'label': 'simulated data', 'mfc': 'none'}

plot_types = ['magnitude', 'real', 'imaginary']
for plot_type in plot_types:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    em.modulus_plot([ax], F_sim, Z_sim, param_dict, diameter, thickness, 
                 plot_types=[plot_type])
    figs.append(fig)

plt.show()
# %%
