# EISFitpython
EISFitpython is a Python package for Electrochemical Impedance Spectroscopy (EIS) analysis, optimized for temperature-dependent experiments and complex circuit models. It applies a unified chi-square optimization to fit multiple impedance datasets at once, preserving inter-parameter relationships and propagating uncertainties across the full temperature range.

## Key Features

Accurate modeling of temperature-dependent impedance demands consistent parameter evolution. EISFitpython delivers:

### Unified Optimization Framework

Standard EIS workflows fit each temperature in isolation, which can produce erratic parameter trends, biased activation-energy estimates, and incomplete error analysis. EISFitpython replaces this with a single global-local optimization:

- **Concurrent Dataset Fitting**  
  Reduces parameter inconsistency by optimizing all temperatures together.  
- **Global Parameters**  
  Keeps temperature-independent circuit elements constant across datasets.  
- **Local Parameters**  
  Allows temperature-dependent elements to adapt within physical constraints.  
- **Error Propagation**  
  Tracks uncertainty through every temperature step.
  
### Advanced Analysis Capabilities

- **Global-Local Parameter Classification**  
  Automatically labels each circuit element as temperature-independent or temperature-dependent.  
- **Unified Chi-Square Objective**  
  Optimizes all datasets against one cost function for consistency.  
- **Automated Classification**  
  Detects and assigns global vs. local roles without manual intervention.  
- **Statistical Validation**  
  Computes confidence intervals and parameter correlations.

## Citation

If you use this code in your research, please cite:
```
S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
https://doi.org/10.1039/d3ta06237f
```

# I. Theoretical Background

## Fundamentals of Electrochemical Impedance Spectroscopy

Electrochemical Impedance Spectroscopy (EIS) is a powerful analytical technique that characterizes the electrical properties of materials and their interfaces with electronically conducting electrodes. The technique involves:

1. **Signal Application**: 
   $$E(t) = E_0\sin(\omega t)$$

2. **System Response**: 
   $$I(t) = I_0\sin(\omega t + \phi)$$

3. **Impedance Calculation**: 
   $$Z(\omega) = \frac{E(t)}{I(t)} = |Z|e^{j\phi} = Z' + jZ''$$

Where:
- $\omega = 2\pi f$ is the angular frequency
- $\phi$ is the phase shift
- $|Z|$ is the impedance magnitude
- $Z'$ and $Z''$ are real and imaginary components

## Complex Nonlinear Least-Squares Fitting

Electrochemical Impedance Spectroscopy (EIS) represents a non-linear system where impedance data must be fitted using complex non-linear least-squares (CNLS) analysis to extract meaningful physical parameters. The CNLS technique operates by minimizing a sum of squares function S, which quantifies the difference between experimental data and the theoretical model.

### Mathematical Framework

The sum of squares function for model fitting:

$$S = \sum_{i=1}^{m} w_i[f_i^\mathrm{EXP}(x_i) - f_i^\mathrm{TH}(x, P)]^2$$

For EIS data over frequency range $\omega_i$:

$$S = \sum_{i=1}^{N} \{w_i[\mathrm{Re}Z_i^\mathrm{EXP}(\omega_i) - \mathrm{Re}Z_i^\mathrm{TH}(\omega_i, P)]^2 + w_i[\mathrm{Im}Z_i^\mathrm{EXP}(\omega_i) - \mathrm{Im}Z_i^\mathrm{TH}(\omega_i, P)]^2\}$$

Where:
- $Z_i^\mathrm{EXP}$ and $Z_i^\mathrm{TH}$ represent experimental and theoretical impedance respectively
- Re and Im denote real and imaginary components
- $w_i$ and $w_i$ are weighting factors for real and imaginary components

### Weighting Methods

The choice of weighting method significantly impacts the fitting quality. EISFitpython implements three established approaches:

1. **Unit Weighting**
   $$w_i^\mathrm{RE} = w_i^\mathrm{IM} = 1$$
   - Treats all data points equally
   - May overemphasize larger impedance values
   - Suitable for data with uniform uncertainties

2. **Proportional Weighting**
   $$w_i^\mathrm{RE} = \frac{1}{(\mathrm{Re}Z_i)^2}, \quad w_i^\mathrm{IM} = \frac{1}{(\mathrm{Im}Z_i)^2}$$
   - Assumes constant relative errors
   - May overemphasize high and low-frequency regions
   - Useful when uncertainties scale with impedance magnitude

3. **Modulus Weighting**
   $$w_i= \frac{1}{|Z_i|^2}$$
   - Balances contributions from small and large impedances
   - Provides equal statistical weight to real and imaginary components
   - Recommended for typical EIS measurements

# II. Getting Started

## Installation

### System Requirements
- Python 3.9 or higher
- NumPy ≥ 1.19.0
- SciPy ≥ 1.6.0
- Matplotlib ≥ 3.3.0
- Pandas ≥ 1.2.0

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Sheyigold/EISFitpython.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Core Modules Overview

### 1. Circuit Module (circuit_main.py)

Core module for defining and evaluating equivalent circuit models.

#### Supported Circuit Elements

- `R`: Resistor (Z = R)
- `C`: Capacitor (Z = 1/jωC)
- `L`: Inductor (Z = jωL)
- `Q`: Constant Phase Element (Z = 1/[Q(jω)ᵅ])
- `W`: Warburg Element (Z = σω^(-1/2)(1-j))
- `F`: Finite-length Warburg (Z = σ·tanh(√(jωτ))/√(jωτ))

#### Circuit String Syntax
- Series connections: `+`
- Parallel connections: `|`
- Nested circuits: `()`

Example: `"(R1|Q1)+(R2|Q2)"` represents a series combination of two parallel R-Q circuits.

#### Key Functions

1. `compute_impedance(params, circuit_str, freqs)`
   - Calculates complex impedance for a given circuit
   - Parameters: element values, circuit string, frequency points

2. `Z_curve_fit(circuit_str)` 
   - Generates fitting function for scipy.optimize.curve_fit
   - Used internally by fitting routines

### 2. Data Extraction Module (data_extraction.py)

Handles importing and processing raw EIS data files with sophisticated data validation and preprocessing capabilities.

#### Key Functions

1. `readNEISYS(filename)`
   - Reads NEISYS format spectrometer files
   
2. `readTXT(filename)` / `readCSV(filename)`
   - Supports multiple standard data formats:
     * Three-column format: f, Z', Z"

3. `trim_data(f, Z, fmin, fmax)`
   - Advanced frequency range filtering:
     * Linear or logarithmic range selection
     * Automatic endpoint adjustment
     * Data density preservation
   - Optional noise filtering
   - Returns filtered f, Z arrays
   - Example:
     ```python
     # Trim to specific frequency range
     f_trim, Z_trim = dt.trim_data(f, Z, 
                                  fmin=1e-1,  # 0.1 Hz
                                  fmax=1e6)   # 1 MHz
     ```

4. `get_eis_files(base_path, subfolder)`
   - Smart file handling:
     * Recursive directory search
     * Pattern matching for EIS data
     * Automatic sorting by temperature
   - Supports multiple file formats
   - Temperature extraction from filenames
   - Example:
     ```python
     # Get temperature-series files
     files = dt.get_eis_files(
         base_path='EIS_Data',
         subfolder='temp_series'
     )
     ```

### 3. EIS Fitting Module (EISFit_main.py)

Core fitting and analysis functionality.

#### Key Functions

1. `EISFit(f, Z, params, circuit, UB, LB, weight_mtd='M', method='lm', single_chi='No')`
   - Main fitting routine
   - Supports multiple weighting methods:
     * 'M': Modulus weighting (|Z|)
     * 'P': Proportional weighting
     * 'U': Unity (no weighting)
   - Returns optimized parameters, errors, and fit statistics

2. `full_EIS_report(f, Z, params, circuit, UB, LB, weight_mtd, method, single_chi, plot_type)`
   - Comprehensive analysis with parameter values and plots
   - Generates Nyquist and/or Bode plots
   - Calculates statistical metrics (χ², R², AICc)

3. `plot_fit(f, Z, Z_fit, plot_type)`
   - Visualization of experimental and fitted data
   - Supports Nyquist, Bode, or both plot types
   - Automatic scaling of impedance units (Ω, kΩ, MΩ, GΩ)

### 4. Batch Analysis Module (EIS_Batchfit.py)

Tools for analyzing multiple datasets, especially temperature-dependent studies.

#### Key Functions

1. `Batch_fit(files, params, circuit, Temp, UB, LB, weight_mtd, method)`
   - Fits multiple datasets with same circuit model
   - Handles temperature series data
   - Generates combined plots and reports

2. `plot_arrhenius(R_values, temp, diameter, thickness)`
   - Arrhenius plot generation
   - Calculates activation energies
   - Handles multiple components (bulk, grain boundary, etc.)

3. `sigma_report(R, Rerr, l, con, T)`
   - Calculates conductivity and its uncertainty
   - Accounts for geometric factors and measurement errors
   - Reports temperature-dependent conductivity values with:
     * Absolute conductivity (S/cm)
     * Standard error
     * Percent error
   - Handles both single and multiple temperature points

4. `C_eff(R, R_err, Q, Q_err, n, n_err, T)`
   - Calculates effective capacitance from CPE parameters
   - Includes error propagation
   - Useful for analyzing non-ideal capacitive behavior

### 5. Advanced Analysis Module (singlechi.py)

Specialized tools for complex systems with shared parameters across multiple temperatures.

#### Key Functions

1. `Single_chi_report(f, Z, params, Temp, circuit_str)`
   - Advanced fitting with global-local parameter handling:
     * Global parameters remain constant across temperatures
     * Local parameters vary with temperature
   - Temperature-dependent parameter tracking
   - Comprehensive error analysis including:
     * Parameter correlations
     * Standard errors
     * Chi-square statistics
   
2. `flatten_params(params, circuit_str, N_sub)`
   - parameter management for complex fits
   - Handles mixed global/local parameters:
     * For global parameters: single value used across all temperatures
     * For local parameters: array of values [T₁, T₂, ..., Tₙ]
   - Automatic parameter validation and structure verification
   - Example usage:
     ```python
     # Mixed global-local parameter structure
     params = (
         [R1_T1, R1_T2, ..., R1_Tn],  # Local R1 values
         Q1,                           # Global Q1 value
         n1,                           # Global n1 value
         [R2_T1, R2_T2, ..., R2_Tn]   # Local R2 values
     )
     ```

3. `Single_chi(f, *params, circuit_str, circuit_type)`
   - Core computation engine for impedance calculation
   - Handles parameter distribution across temperature points
   - Supports both fitting and prediction modes
   - Automatically manages parameter structure based on temperature points

## Quick Start Guide

### 1. Basic Circuit Simulation
```python
import EISFit_main as em

# Define a simple RC circuit
circuit = "(R1|C1)"
params = [100, 1e-6]  # R = 100 Ω, C = 1 µF
freqs = np.logspace(-2, 6, 50)  # 0.01 Hz to 1 MHz

# Generate simulated data
Z = em.predict_Z(freqs[0], freqs[-1], len(freqs), params, circuit)
```

### 2. Loading and Fitting Real Data
```python
import data_extraction as dt

# Load EIS data
f, Z = dt.readNEISYS('my_data.txt')

# Define circuit and initial parameters
circuit = "(R1|Q1)+R2"
params = [1e5, 1e-9, 0.8, 50]  # R1, Q1, n1, R2
UB = [1e6, 1e-6, 1, 100]      # Upper bounds
LB = [1e4, 1e-12, 0.5, 10]    # Lower bounds

# Perform fitting
popt, perror = em.full_EIS_report(
    f, Z, params, circuit, 
    UB=UB, LB=LB,
    weight_mtd='M',      # Modulus weighting
    method='lm',         # Levenberg-Marquardt algorithm
    single_chi='No',
    plot_type='both'     # Generate both Nyquist and Bode plots
)
```

### 3. Temperature-Dependent Analysis
```python
import EIS_Batchfit as ebf

# Load multiple temperature datasets
files = dt.get_eis_files(base_path='EIS_Data', subfolder='temp_series')
temps = np.array([25, 50, 75, 100])  # °C

# Equivalent circuit model
circuit_str = "(R1|Q1)+(R2|Q2)"

# Initial parameter guesses for the circuit model
params = [1.8e6, 1.3e-11, 0.9,8.2e6, 1.3e-9,0.9, 5.7e-7, 0.6]
# Parameters order: [R1, Q1,n1, R2, Q1, n1, Q2, n2]

# Perform batch fitting
fit_params, fit_errors = ebf.Batch_fit(
    files, params, circuit, temps,
    UB, LB, weight_mtd='M', method='lm'
)

# Generate Arrhenius plot
D = 1.3    # Sample diameter (cm)
L = 0.2    # Sample thickness (cm)
ebf.plot_arrhenius(fit_params[:,0], temps, D, L, labels='Bulk')
```

### 4. Temperature-Dependent Analysis using Global-Local Optimization Strategy
```python
import data_extraction as dt     # For data file handling
import singlechi as sc          # For single chi analysis
import EIS_Batchfit as ebf      # For batch fitting
import numpy as np

#%% Data Processing
# Get EIS data files
filenames=dt.get_eis_files(base_path='../EIS_Data', subfolder='Example-3-4')

# Extract frequency and impedance data from NEISYS spectrometer files
f, Z = dt.full_readNEISYS(filenames)

# Split data at 1MHz frequency point
sublist, _ = dt.split_array(f, Z=None, split_freq=1e6)
N_sub = len(sublist)

# Temperature points (in Celsius)
temps = np.array([140, 150, 160, 170, 180])

# Define parameters with temperature dependence
circuit = "(R1|Q1)+(R2|Q2)"
params = (
    R1_values,  # Array of R1 values for each temperature
    Q1,         # Global Q1 (temperature-independent) / local (temperature-dependent, i.e., ferroelectrics )
    n1,         # Global n1 (temperature-independent) / local (temperature-dependent, i.e., ferroelectrics )
    R2_values,  # Array of R2 values for each temperature
    Q2,         # Global/local  Q2
    n2          # Global/local  n2
)

# Perform analysis
results = sc.Single_chi_report(f, Z, params, temps, circuit)
```

The module handles complex parameter relationships:
- Resistance values (R) typically show temperature dependence
- CPE parameters (Q, n) often remain constant across temperatures
- Automatic handling of parameter interdependencies
- Built-in correlation analysis between parameters


## Common Error Messages

1. Invalid Circuit String:
   - Check element numbering (R1, R2, etc.)
   - Verify parentheses matching
   - Ensure valid operators (+, |)

2. Parameter Count Mismatch:
   - Count elements in circuit string
   - Remember CPE needs 2 parameters (Q, n)
   - Check temperature point count for batch analysis

3. Convergence Failures:
   - Try different initial parameters
   - Check parameter bounds
   - Verify data quality
   - Consider simpler circuit model

4. File Format Errors:
   - Verify column structure
   - Check for header presence/format
   - Ensure valid numeric data

## Recommendation 

1. Data Preparation:
   - Use consistent file naming
   - Maintain temperature series structure
   - Use trim_data() for focused analysis
   - Check frequency range coverage

2. Circuit Model Selection:
   - Start simple, add complexity as needed
   - Validate with physical meaning
   - Check parameter uncertainties

3. Fitting Strategy:
   - Use reasonable initial guesses
   - Set appropriate parameter bounds
   - Try different weighting methods

4. Temperature Analysis:
   - Ensure consistent sample geometry
   - Use proper temperature units
   - Validate Arrhenius behavior
   - Check activation energy

## Example Usage

See Example_scripts/ directory for detailed examples:
- Example1-simulate.py: Circuit simulation
- Example2a/b/c-EISfit.py: Single temperature fitting
- Example3-Batchfit.py: Temperature series analysis
- Example4-Singlechi.py: Advanced parameter sharing


---

## License
EISFitpython is distributed under the MIT License.

---

## Author

Sheyi Clement Adediwura (c) 2024

---

