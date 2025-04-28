"""
A comprehensive toolkit for analyzing and fitting EIS data, specifically designed for solid-state ionic conductors
and electrochemical systems. This module provides advanced functionality for impedance analysis, equivalent
circuit modeling, and activation energy calculations.

Key Features:
-------------
* Equivalent circuit fitting with statistical analysis
* Nyquist and Bode plot visualization 
* Arrhenius activation energy analysis
* Complex impedance and modulus calculations
* Automated data processing and visualization
* Error analysis and uncertainty propagation
* Multiple weighting methods for fitting

Requires: numpy, scipy, matplotlib, pandas

Author: Sheyi Clement Adediwura 

'Please cite the following paper if you use this code:
S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
https://doi.org/10.1039/d3ta06237f

"""

#External libraries
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
import re
import os
import datetime


#Custom libraries 
import circuit_main as Eqc
import data_extraction as dt




def logf_gen(start_freq, end_freq, num_points):
    """Generate logarithmically spaced frequency values.
    
    Args:
        start_freq (float): Starting frequency value
        end_freq (float): Ending frequency value 
        num_points (int): Number of frequency points to generate
        
    Returns:
        array: Logarithmically spaced frequency values
    """
    start_exponent = np.log10(start_freq)
    end_exponent = np.log10(end_freq) 
    return np.logspace(start_exponent, end_exponent, num_points)

def predict_Z(start_freq, end_freq, no_points, params, circuit_str):
    """Compute and plot impedance for a specified circuit model and parameters.
    
    Args:
        start_freq (float): Starting frequency
        end_freq (float): Ending frequency
        no_points (int): Number of frequency points
        params (array): Circuit model parameters
        circuit_str (str): Circuit model specification string
        
    Returns:
        array: Complex impedance values
        
    Plots:
        - Nyquist plot of predicted impedance
        - Bode plots of magnitude and phase
    """
    # Generate frequency array
    f = logf_gen(start_freq, end_freq, no_points)
    
    # Calculate impedance values
    Z = Eqc.compute_impedance(params, circuit_str, f)
    
    # Create Nyquist plot
    figN, ax = plt.subplots(figsize=(5, 5), dpi=150, facecolor='w', edgecolor='b')
    params_dict = {'marker': 'o', 'color':'k', 'ms':'6', 'ls':'', 'mfc':'grey'}
    a = nyquistPlot(ax, Z, params_dict)
    
    # Create Bode plots
    figB, ax = plt.subplots(nrows=2, figsize=(5,6), dpi=150, facecolor='w', edgecolor='k')
    figB.subplots_adjust(hspace=0.3)
    b = bodePlot(ax, f, Z, params_dict)
    plt.show()
    
    
    mypath = '../Figures'
    # Create directory for figures if it doesn't exist
    if not os.path.exists(mypath):
            os.makedirs(mypath)
            
    figN.savefig(os.path.join(mypath, 'Nyquist-sim.svg'), bbox_inches='tight')
    figB.savefig(os.path.join(mypath, 'Bode-sim.svg'), bbox_inches='tight')
    return Z

def EISFit(f, Z, params, circuit, UB, LB, weight_mtd='M', method='lm',single_chi='No'):
    """
    Perform curve fitting of measured EIS data to a specified equivalent circuit model.
    
    Parameters:
    -----------
    f : array-like
        Frequency values
    Z : array-like
        Complex impedance measurements
    params : array-like
        Initial parameter guesses
    circuit : str
        Circuit model string
    UB, LB : array-like
        Upper and lower bounds for parameters
    weight_mtd : str, optional
        Weighting method ('M'=modulus, 'P'=proportional, 'U'=unity)
    method : str, optional
        Optimization method for curve_fit
    single_chi : str, optional
        Whether to use single chi calculation ('Yes'/'No')
        
    Returns:
    --------
    chi2 : float
        Chi-squared value
    R_chi2 : float
        Reduced chi-squared value
    AICc : float
        Corrected Akaike Information Criterion
    popt : array
        Optimized parameters
    perror : array
        Standard errors of parameters
    R_squared : float
        R-squared value of fit
    CorrMat : array
        Correlation matrix
    pcov : array
        Covariance matrix
    
    Notes:
    ------
    AIC calculation adapted from:
    Ingdal et al. (2019) Electrochimica Acta 317, 648-653
    """
    
    # Set up weighting based on specified method
    if weight_mtd == 'M':
        weight = np.hstack([np.abs(Z), np.abs(Z)])  # Modulus weighting
    elif weight_mtd == 'P':
        weight = np.hstack([Z.real, Z.imag])        # Proportional weighting
    elif weight_mtd == 'U':
        weight = None                               # Unity (no) weighting
    else:
        raise ValueError('Please specify weighting method using M, P, or U')
    
    # Process circuit model based on single_chi parameter
    if single_chi == 'No':
        circuit = Eqc.Z_curve_fit(circuit)
    elif single_chi == 'Yes':
        circuit = circuit

    # Perform curve fitting with or without weighting
    if weight is not None:
        popt, pcov = curve_fit(circuit, f, np.hstack([Z.real, Z.imag]), 
                              p0=params, sigma=weight, bounds=(LB,UB), 
                              method=method, maxfev=1000000, ftol=1E-15)
    else:
        popt, pcov = curve_fit(circuit, f, np.hstack([Z.real, Z.imag]), 
                              p0=params, bounds=(LB,UB), 
                              method=method, maxfev=1000000, ftol=1E-15)

    # Calculate fit statistics
    Zexp = np.hstack([Z.real, Z.imag])
    residuals = Zexp - circuit(f, *popt)
    if weight is not None:
        residuals /= weight
        
    # Calculate goodness of fit parameters
    chi2 = np.sum(residuals**2)
    dof = 2 * len(f) - len(params)  # Degrees of freedom
    R_chi2 = chi2 / dof             # Reduced chi-squared
    SSR = np.sum(residuals**2)      # Sum of squared residuals
    SST = np.sum((Zexp - np.mean(Zexp))**2)  # Total sum of squares
    R_squared = 1 - (SSR / SST)     # R-squared value

    # Calculate AIC and AICc
    N = len(Zexp)
    pn = len(popt)
    AIC = N * np.log(2 * np.pi) + N * np.log(chi2) - N * np.log(N) + N + 2 * (pn + 1)
    if weight is not None:
        AIC -= np.sum(np.log(np.abs(weight)))
    AICc = AIC + 2 * pn * ((pn + 1) / (N - pn - 1))  # Corrected AIC

    # Calculate parameter uncertainties and correlation matrix
    perror = np.sqrt(np.diag(pcov))  # Standard errors
    CorrMat = np.zeros_like(pcov, dtype=float)
    for i in range(pcov.shape[0]):
        for j in range(pcov.shape[1]):
            CorrMat[i, j] = pcov[i, j] / np.sqrt(pcov[i, i] * pcov[j, j])

    return chi2, R_chi2, AICc, popt, perror, R_squared, CorrMat, pcov


def rmse(Z, Z_fit):
    t=np.abs(np.sqrt(np.sum(((Z-Z_fit))**2)))
    return t

# Extract component names from the circuit string
def circuit_components(circuit_str):
        """Extract component names from the circuit string in the order they appear.
        """
        return re.findall(r'[RCLQE]\d+', circuit_str)

def fit_report(f,Z,params,circuit,UB,LB,weight_mtd,method,single_chi):
    """Generate a report of the fit results including standard errors, correlation matrix, and plots.
    """
    chi2, R_chi2, AICc, popt, perror,R_squared,CM,pcov= EISFit(f, Z, params, circuit, UB, LB, weight_mtd, method,single_chi)
    if single_chi== 'No':
        #if circuit_str is not None:
        component_names = circuit_components(circuit)
        # Create an expanded list of component parameters
        All_component_names = []
        n_counter = 1 
        for name in component_names:
            if name.startswith('Q'):
                #expanded_component_names.extend([f"{name}", "n"])
                All_component_names.extend([f"{name}", f"n{n_counter}"])
                n_counter += 1
            elif name.startswith('F'):
                All_component_names.extend([f"{name}", f"{name}_n"])
            else:
                All_component_names.append(name)
            # Create the DataFrame using the expanded component names
        index = [f'{name} =' for name in All_component_names]
        cols = ['Fit Params', 'Stderr', '% error']
        df = pd.DataFrame(index=index, columns=cols)
        df['Fit Params'] = np.array(popt)
        df['Stderr'] = np.array(perror)
        df['% error'] = (df['Stderr'] / df['Fit Params']) * 100
        Fit = df
        # Generate column and row labels
        labels= [f'{name}' for name in All_component_names]
        row_labels = [label + ' :' for label in labels]
    else:
        raise ValueError ('fit_report is only meant for Single_chi = NO')
    # Create a DataFrame with the desired labels
    CorrM = pd.DataFrame(CM, columns=labels, index=row_labels)
    mask_upper = np.triu(np.ones(CorrM.shape), k=0).astype(np.bool_) 
    CorrM = CorrM.mask(mask_upper).round(3)
    CorrM=CorrM.fillna('')
    # Create the matrix DataFrame
    matrix_df = pd.DataFrame(CorrM)
    

    
    # Print single fit results header
    print("┌" + "─" * 70 + "┐")  # Lighter border for subheader
    print("│" + " EIS Fitting Results ".center(70) + "│")
    print("└" + "─" * 70 + "┘")
    print("\nParameter Values and Standard Errors:")
    print("-" * 72)
    print(Fit.to_string())
    
    print("\nCorrelation Matrix:")
    print("-" * 72)
    print(np.round(matrix_df).to_string())
    print("\n")
    
    
    return  popt, perror

def nyquistPlot(ax, Z, param_dict):
    """ 
    Nyquist plot function using matplotlib with automatic scaling and shared axes
    """
    # Input validation
    if Z is None or len(Z) == 0:
        raise ValueError("Empty or invalid impedance data")
    
    # Ensure Z is numpy array
    Z = np.asarray(Z, dtype=complex)
    # Determine the appropriate impedance unit (Ω, kΩ, MΩ, or GΩ)
    Z_magnitude = np.abs(Z)  # Get the magnitude of the impedance
    
    # Check for the largest impedance value and decide on the appropriate scale
    max_impedance = np.max(Z_magnitude)

    if max_impedance >= 1e9:  # If impedance is in GΩ (Gigaohms)
        Z_scaled = Z / 1e9  # Convert to GΩ
        unit_label = r'\mathrm{G\Omega}'
    elif max_impedance >= 1e6:  # If impedance is in MΩ (Megaohms)
        Z_scaled = Z / 1e6  # Convert to MΩ
        unit_label = r'\mathrm{M\Omega}'
    elif max_impedance >= 1e3:  # If impedance is in kΩ (Kiloohms)
        Z_scaled = Z / 1e3  # Convert to kΩ
        unit_label = r'\mathrm{k\Omega}'
    else:  # If impedance is in Ω (Ohms)
        Z_scaled = Z  # Leave it as Ω
        unit_label = r'\mathrm{\Omega}'
    
    # Plot the Nyquist plot with scaled impedance values
    ax.plot(Z_scaled.real, -Z_scaled.imag, **param_dict)

    # Set the labels to -imaginary vs real, and include the unit label
    ax.set_xlabel(r'$ \mathit{Z^{\prime}} \,/\, ' + unit_label + r'$', fontsize=15)
    ax.set_ylabel(r'$ \mathit{-Z^{\prime\prime}} \,/\, ' + unit_label + r'$', fontsize=15)
    
    # Ensure the x and y axes have the same scale
    ax.set_aspect('equal')  # This ensures the axes are scaled equally

    # Extract the current upper limits for both x and y axes
    x_lower, x_upper = ax.get_xlim()  # Get x-axis limits as (min, max)
    y_lower, y_upper = ax.get_ylim()  # Get y-axis limits as (min, max)

    # Find the larger of the two upper limits
    upper_limit = max(x_upper, y_upper)

    # Set both axis limits to the larger value
    ax.set_xlim([0, upper_limit])  # Set x-axis limits
    ax.set_ylim([0, upper_limit])  # Set y-axis limits

    # Customize tick parameters and labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust the font size of the axis offset labels
    y_offset = ax.yaxis.get_offset_text()
    y_offset.set_size(8)
    x_offset = ax.xaxis.get_offset_text()
    x_offset.set_size(8)

    return ax




def bodePlot(axes, f, Z, param_dict, all_Z=None):   
    """Bode plot function that plots impedance magnitude and phase vs frequency with automatic unit scaling
    
    Parameters
    ----------
    axes : tuple
        Tuple of matplotlib axes for magnitude and phase plots
    f : array-like
        Frequency values
    Z : array-like
        Complex impedance values for current dataset
    param_dict : dict
        Dictionary of plotting parameters
    all_Z : list, optional
        List of all impedance datasets for batch scaling
    """
    # Input validation
    if Z is None or len(Z) == 0:
        raise ValueError("Empty or invalid impedance data")
    
    # Ensure Z is numpy array
    Z = np.asarray(Z, dtype=complex)
    
    # Determine scaling based on all datasets if provided
    if all_Z is not None:
        # Get maximum impedance across all datasets
        max_impedance = max(np.max(np.abs(z)) for z in all_Z)
    else:
        # Single dataset scaling
        max_impedance = np.max(np.abs(Z))

    # Scale impedance based on maximum value
    if max_impedance >= 1e9:  # GΩ range
        Z_scaled = Z / 1e9
        unit_label = r'\mathrm{G\Omega}'
    elif max_impedance >= 1e6:  # MΩ range
        Z_scaled = Z / 1e6
        unit_label = r'\mathrm{M\Omega}'
    elif max_impedance >= 1e3:  # kΩ range
        Z_scaled = Z / 1e3
        unit_label = r'\mathrm{k\Omega}'
    else:  # Ω range
        Z_scaled = Z
        unit_label = r'\mathrm{\Omega}'
    
    # Plot magnitude and phase
    ax1, ax2 = axes
    ax1.plot(f, np.abs(Z_scaled), **param_dict)
    ax2.plot(f, -np.angle(Z, deg=True), **param_dict)
    
    # Set labels with proper LaTeX formatting
    ax1.set_ylabel(r'$|Z| \,/\, ' + unit_label + '$', fontsize=15, color="black")
    ax2.set_ylabel(r'$-\phi_Z \,/\, ^{\circ}$', fontsize=15, color="black")
    
    # Configure axes
    for ax in axes:    
        ax.set_xlabel(r'$ \it\nu \it$/Hz ', fontsize=15)
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    return axes

def plot_fit(f=None, Z=None, Z_fit=None, plot_type='bode'):
    """Plot EIS data as Nyquist and/or Bode plots.
    Allows plotting experimental data (Z) and fitted data (Z_fit)."""
    
    mypath = '../Figures'
    timestamp = datetime.datetime.now().strftime("%m-%d_%S")
    # Create directory for figures if it doesn't exist
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    if plot_type not in ['nyquist', 'bode', 'both']:
        raise ValueError("Please specify type of plot i.e. 'nyquist', 'bode' or both.")
    
    if f is not None:
        f_dat = f

    # Define common plot parameters
    data_params = {'marker': 'o', 'color': 'k', 'ms': '6', 'ls': '', 
                  'label': 'Data', 'mfc': 'none'}
    fit_params = {'marker': '*', 'color': 'darkred', 'ms': '6', 'ls': '', 
                 'label': 'Fit', 'mfc': 'none'}

    if plot_type == 'nyquist' or plot_type == 'both':
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150, facecolor='w', edgecolor='k')
        
        if Z is not None:
            ax = nyquistPlot(ax, Z, data_params)
        
        if Z_fit is not None:
            ax = nyquistPlot(ax, Z_fit, fit_params)
           
        
        ax.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        fig.savefig(os.path.join(mypath, f'Nyquist-plot_{timestamp}.svg'), 
                       bbox_inches='tight', dpi=300)
        
        if plot_type == 'nyquist':
            plt.show()
            return ax
    
    if plot_type == 'bode' or plot_type == 'both':
        fig, ax = plt.subplots(nrows=2, figsize=(5, 6), dpi=150, facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.12)
        
        if Z is not None:
            ax = bodePlot(ax, f_dat, Z, data_params)
        
        if Z_fit is not None:
            ax = bodePlot(ax, f_dat, Z_fit, fit_params)
        
        
        # Add legends to both magnitude and phase plots
        ax[0].legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        ax[1].legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        
        
        fig.savefig(os.path.join(mypath, f'Bode-plot_{timestamp}.svg'), 
                       bbox_inches='tight', dpi=300)
        
        plt.show()
        return ax

    return None

def full_EIS_report(f,Z,params,circuit,UB,LB,weight_mtd,method,single_chi,plot_type='both'):
    """
    Generate a complete EIS fitting report including parameter values and plots.
    
    Parameters:
    -----------
    f : array-like
        Frequency values
    Z : array-like
        Complex impedance values
    params : array-like 
        Initial parameter guesses
    circuit : str
        Circuit model string
    UB, LB : array-like
        Upper and lower bounds for parameters
    weight_mtd : str
        Weighting method ('M','P' or 'U')
    method : str
        Optimization method
    single_chi : str
        Whether to use single chi calculation
    plot_type : str
        Type of plot to generate ('nyquist', 'bode' or 'both')
        
    Returns:
    --------
    popt : array
        Optimized parameter values
    perror : array  
        Parameter standard errors
    """
    # Get fit parameters and errors
    popt, perror= fit_report(f,Z,params,circuit,UB,LB,weight_mtd,method,single_chi)
    
    # Get impedance function from circuit
    Z_function  = Eqc.Z_curve_fit(circuit)
    
    # Calculate fitted impedance values
    Z_values = Z_function(f, *popt)
    
    # Convert concatenated impedance back to complex form
    real_values = Z_values[:len(Z_values)//2]
    imag_values = Z_values[len(Z_values)//2:]
    Z_fit = real_values + 1j * imag_values
    
    # Generate plots
    plot_fit(f, Z, Z_fit, plot_type)

    
    return popt, perror



def EA(ax, T, conductivity, plot_params):
    """
    Calculate and plot Arrhenius activation energy analysis for conductivity data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on
    T : array_like
        Temperature values in Kelvin
    conductivity : array_like  
        Conductivity values in S/cm
    plot_params : dict
        Dictionary of plotting parameters
    
    Returns
    -------
    dict
        Dictionary containing calculated values (Ea, sigma_RT, uncertainties)
    """
    # Constants
    KB = 8.617333262145e-5  # Boltzmann constant in eV/K
    T_ROOM = 298  # Room temperature in K
    T_UNCERTAINTY = 0.01  # Temperature measurement uncertainty

    def calculate_sigma(a0, Ea, T):
        """Calculate conductivity using Arrhenius equation"""
        return (a0/T) * np.exp(-Ea / (KB * T))

    def propagate_error(a0, delta_a0, Ea, delta_Ea, T, delta_T):
        """
        Calculate error propagation for Arrhenius conductivity using partial derivatives.
        
        σ = (a0/T) * exp(-Ea/(kB*T))
        
        Partial derivatives:
        ∂σ/∂a0 = (1/T) * exp(-Ea/(kB*T))
        ∂σ/∂Ea = -(a0/T) * exp(-Ea/(kB*T)) * 1/(kB*T)
        ∂σ/∂T = -(a0/T²) * exp(-Ea/(kB*T)) + (a0/T) * exp(-Ea/(kB*T)) * Ea/(kB*T²)
        """
        # Base conductivity
        sigma = calculate_sigma(a0, Ea, T)
        
        # Partial derivatives
        dsigma_da0 = (1/T) * np.exp(-Ea/(KB * T))
        dsigma_dEa = -(sigma)/(KB * T)
        dsigma_dT = sigma * ((Ea/(KB * T**2)) - (1/T))
        
        # Total uncertainty using error propagation formula
        # δσ = sqrt[(∂σ/∂a0 * δa0)² + (∂σ/∂Ea * δEa)² + (∂σ/∂T * δT)²]
        delta_sigma = np.sqrt(
            (dsigma_da0 * delta_a0)**2 +
            (dsigma_dEa * delta_Ea)**2 +
            (dsigma_dT * delta_T)**2
        )
        
        return delta_sigma, {
            'relative_error_a0': abs((dsigma_da0 * delta_a0)/sigma),
            'relative_error_Ea': abs((dsigma_dEa * delta_Ea)/sigma),
            'relative_error_T': abs((dsigma_dT * delta_T)/sigma)
        }

    # Prepare data for Arrhenius plot
    Tk = 1000 / T
    logcon = np.log(conductivity * T)
    conductivity_T = np.exp(logcon)

    # Perform linear regression
    regression = stats.linregress(Tk, logcon)
    
    # Calculate parameters and uncertainties
    results = {
        'Ea': abs(regression.slope) * 1000 * KB,
        'delta_Ea': abs(regression.stderr) * 1000 * KB,
        'a0': np.exp(regression.intercept),
        'delta_a0': np.exp(regression.intercept_stderr),
        'R_squared': regression.rvalue**2
    }
    
    # Calculate sigma at room temperature and uncertainty with error contributions
    results['sigma_RT'] = calculate_sigma(results['a0'], results['Ea'], T_ROOM)
    delta_sigma, error_contributions = propagate_error(
        results['a0'], results['delta_a0'],
        results['Ea'], results['delta_Ea'],
        T_ROOM, T_UNCERTAINTY
    )
    results['delta_sigma'] = delta_sigma
    results['error_contributions'] = error_contributions

    # Print detailed results
    print(f"R-squared: {results['R_squared']:.4f}")
    print(f"Activation Energy (Ea): {results['Ea']:.4f} ± {results['delta_Ea']:.4f} eV")
    print(f"Pre-exponential (σ∞): {results['a0']:.7f} ± {results['delta_a0']:.7f}")
    print(f"RT Conductivity (σRT): {results['sigma_RT']:.5e} ± {results['delta_sigma']:.5e} S/cm")
    print("=" * 53)

    # Plot experimental data
    ax.plot(Tk, conductivity_T, **plot_params)
    
    # Plot fit line
    fit_params = {'marker': '', 'color': 'k', 'lw': 0.5, 'ls': '--', 'mfc': 'none'}
    yfit = np.exp(regression.intercept + regression.slope * Tk)
    ax.plot(Tk, yfit, **fit_params)

    # Configure plot
    ax.set_xlabel(r'$1000·\it{T}^{-1}\it$ / K$^{-1}$', fontsize=15)
    ax.set_ylabel(r'$\it\sigma·T\it$ / K·S·cm$^{-1}$', fontsize=15)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=13)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
    ax.legend(prop={'size': 10})
    plt.tight_layout()
    return ax

def mod_plot(filename, diameter, thickness, modulus_scale):
    """
    Plot impedance and modulus data from EIS measurements.
    
    Args:
        filename (str): Path to the EIS data file
        diameter (float): Sample diameter in cm 
        thickness (float): Sample thickness in cm
        modulus_scale (float): Scaling factor for modulus plot
        
    Returns:
        matplotlib.axes.Axes: Axes object containing the plot
    """
    # Read impedance data
    data = dt.readNEISYS(filename)
    frequency, impedance = data
    
    # Calculate electrical parameters
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    radius = diameter/2
    area = np.pi * radius**2
    capacitance_0 = (epsilon_0 * area)/thickness
    
    # Calculate angular frequency and complex modulus
    angular_freq = 2 * np.pi * np.array(frequency)
    modulus = angular_freq * capacitance_0 * impedance
    modulus_real = angular_freq * capacitance_0 * impedance.real 
    modulus_imag = angular_freq * capacitance_0 * impedance.imag

    # Plot 1: Impedance vs frequency and scaled modulus
    fig1, ax1 = plt.subplots(figsize=(5,5), dpi=150, facecolor='w', edgecolor='k')
    
    # Plot impedance data
    params_imped = {'marker': 'o', 'color':'k', 'ms':'3', 'ls':'--', 'mfc':'none'}
    ax1.plot(frequency, impedance.imag, **params_imped)
    
    # Plot scaled modulus data  
    params_mod = {'marker': 'h', 'color':'r', 'ms':'3', 'ls':'-', 'mfc':'r'}
    ax1.plot(frequency, modulus.imag * modulus_scale, **params_mod)
    
    # Configure first plot
    ax1.set_ylabel(r'$ \it Zimag\it$ /Ω', fontsize=8)
    ax1.set_xlabel(r'$ log_{10} \it(f) \it$/Hz', fontsize=8)
    ax1.set_xscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=8)

    # Plot 2: Complex modulus plane
    fig2, ax2 = plt.subplots(figsize=(7,5), dpi=150, facecolor='w', edgecolor='k')
    
    # Plot Nyquist-style modulus data
    params_nyquist = {'marker': 'h', 'color':'k', 'ms':'4', 'ls':'', 'mfc':'none'}
    ax2.plot(modulus_real, -modulus_imag, **params_nyquist)

    return ax2


