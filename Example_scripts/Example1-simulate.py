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

# Set up path for custom EIS modules
current_dir = os.path.dirname(os.path.abspath(__file__))
EISmodules_path = os.path.join(current_dir, '..', 'EIS_Modules')
sys.path.insert(0, EISmodules_path)

# Custom EIS Module Imports
import EISFit_main as em       # type: ignore

#%% EIS Data Simulation Parameters
# Define equivalent circuit model
circuit = "(R1|C1)"  # Simple parallel RC circuit
# Alternative circuit models (commented out):
#circuit = "(R1|C1)+(R2|C2)+R3"    # Two RC circuits in series with resistor
#circuit = "(R1|C1)+(R2|C2)"       # Two RC circuits in series
#circuit = "R1+(R1|C1)"            # Resistor in series with RC circuit

# Frequency range settings
start_freq = 1e-5    # Starting frequency in Hz
end_freq = 1e7       # Ending frequency in Hz
no_points = 50       # Number of frequency points to simulate

# Circuit component parameters
params = [20, 100e-6]  # [R1 (Ω), C1 (F)]
# Alternative parameter sets (commented out):
#params = [20, 100e-6, 10, 0.1, 10]  # For "(R1|C1)+(R2|C2)+R3"
#params = [20, 100e-6, 10, 0.1]      # For "(R1|C1)+(R2|C2)"
#params = [20, 100, 0.1]             # For "R1+(R1|C1)"

# Generate simulated impedance data
Simulate = em.predict_Z(start_freq, end_freq, no_points, params, circuit)
# Returns complex impedance array with real and imaginary components

# %%
