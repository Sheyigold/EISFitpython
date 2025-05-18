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
import sys
from io import StringIO


#Custom libraries 
from EISFitpython import circuit_main as Eqc
from EISFitpython import data_extraction as dt




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
    """Compute impedance for a specified circuit model and parameters.
    
    Parameters:
    -----------
    start_freq : float
        Starting frequency in Hz
    end_freq : float
        Ending frequency in Hz
    no_points : int
        Number of frequency points to generate
    params : array-like
        Circuit model parameters
    circuit_str : str
        Circuit model specification string (e.g. "(R1|Q1)+(R2|Q2)")
        
    Returns:
    --------
    tuple
        (f, Z) where:
        f : array-like
            Generated frequency points
        Z : array-like
            Complex impedance values
    """
    # Input validation
    if not all(isinstance(x, (int, float)) for x in [start_freq, end_freq]):
        raise ValueError("Frequencies must be numeric values")
    if not isinstance(no_points, int) or no_points <= 0:
        raise ValueError("Number of points must be a positive integer")
        
    try:
        # Generate frequency array
        f = logf_gen(start_freq, end_freq, no_points)
        
        # Calculate impedance values
        Z = Eqc.compute_impedance(params, circuit_str, f)
        
        # If impedance calculation was successful
        if Z is not None:
            # Create output filename using safe circuit string
            safe_circuit = circuit_str.replace('|', 'p')
            output_filename = f'EIS-SIM_{safe_circuit}.txt'
                  	     
            # Write data to file
            with open(output_filename, 'w') as file:
                # Write header information
                file.write(f"# Start Frequency (Hz): {start_freq}\n")
                file.write(f"# End Frequency (Hz): {end_freq}\n")
                file.write(f"# Number of Points: {no_points}\n")
                file.write(f"# Circuit Model: {circuit_str}\n")
                file.write(f"# Parameters: {', '.join(map(str, params))}\n")
                file.write("\nFreq.[Hz]\t\t Zs'[ohm] \t\t Zs''[ohm]\n")
                
                # Sort data from high to low frequency
                sorted_indices = np.argsort(f)[::-1]
                f_sorted = f[sorted_indices]
                Z_sorted = Z[sorted_indices]
                
                # Write data from high to low frequency
                for freq, z_val in zip(f_sorted, Z_sorted):
                    file.write(f"{freq:.6e}\t{z_val.real:.6e}\t{z_val.imag:.6e}\n")
                                
            print(f"Data saved to: {output_filename}")
            
            return f, Z
        
        else:
            print("Error: Impedance calculation failed")
            return None, None
        
    except Exception as e:
        print(f"Error in predict_Z: {str(e)}")
        return None, None

def EISFit(f, Z, params, circuit, UB, LB, weight_mtd='M', method='lm', single_chi='No'):
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
    fit_stats : dict
        Dictionary containing all fit statistics
    popt : array
        Optimized parameters
    perror : array
        Standard errors of parameters
    CorrMat : array
        Correlation matrix
    pcov : array
        Covariance matrix
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
        
    # Calculate basic statistics
    chi2 = np.sum(residuals**2)
    dof = 2 * len(f) - len(params)  # Degrees of freedom
    R_chi2 = chi2 / dof             # Reduced chi-squared
    
    # Sum of squares calculations
    SSR = np.sum(residuals**2)      # Sum of squared residuals
    SST = np.sum((Zexp - np.mean(Zexp))**2)  # Total sum of squares
    R_squared = 1 - (SSR / SST)     # R-squared value
    
    # Calculate advanced statistics
    N = len(Zexp)
    pn = len(popt)
    
    # Adjusted R-squared
    adj_R_squared = 1 - ((1 - R_squared) * (N - 1) / (N - pn - 1))
    
    # Root mean squared error
    RMSE = np.sqrt(SSR / N)
    
    # Standard error of regression
    SER = np.sqrt(SSR / dof)
    
    # Mean absolute error
    MAE = np.mean(np.abs(residuals))
    
    # F-statistic and p-value
    F_stat = (SST - SSR) / (pn - 1) / (SSR / (N - pn))
    p_value = 1 - stats.f.cdf(F_stat, pn - 1, N - pn)
    

    # Calculate parameter uncertainties and correlation matrix
    perror = np.sqrt(np.diag(pcov))  # Standard errors
    
    # Calculate confidence intervals (95%)
    conf_level = 0.95
    t_value = stats.t.ppf((1 + conf_level) / 2, dof)
    conf_intervals = t_value * perror
    
    # Calculate correlation matrix
    CorrMat = np.zeros_like(pcov, dtype=float)
    for i in range(pcov.shape[0]):
        for j in range(pcov.shape[1]):
            CorrMat[i, j] = pcov[i, j] / np.sqrt(pcov[i, i] * pcov[j, j])

    # Store all statistics in a dictionary
    fit_stats = {
        'chi2': chi2,
        'R_chi2': R_chi2,
        'R_squared': R_squared,
        'adj_R_squared': adj_R_squared,
        'RMSE': RMSE,
        'SER': SER,
        'MAE': MAE,
        'F_statistic': F_stat,
        'p_value': p_value,
        'dof': dof,
        'conf_intervals': conf_intervals,
        'N_points': N,
        'N_params': pn
    }

    return fit_stats, popt, perror, CorrMat


# Extract component names from the circuit string
def circuit_components(circuit_str):
        """Extract component names from the circuit string in the order they appear.
        """
        return re.findall(r'[RCLQWF]\d+', circuit_str)

def fit_report(f, Z, params, circuit, UB, LB, weight_mtd, method, single_chi):
    """
    Generate a comprehensive report of the EIS fitting results.
    
    Parameters:
    -----------
    f, Z, params, circuit, UB, LB, weight_mtd, method, single_chi : 
        Same parameters as EISFit function
        
    Returns:
    --------
    popt : array
        Optimized parameters
    perror : array
        Standard errors of parameters
    """
    # Get fit results
    fit_stats, popt, perror, CM= EISFit(f, Z, params, circuit, UB, LB, weight_mtd, method, single_chi)
    
    if single_chi == 'No':
        component_names = circuit_components(circuit)
        # Create expanded list of component parameters
        All_component_names = []
        n_counter = 1 
        for name in component_names:
            if name.startswith('Q'):
                All_component_names.extend([f"{name}", f"n{n_counter}"])
                n_counter += 1
            elif name.startswith('F'):
                All_component_names.extend([f"{name}", f"{name}_n"])
            else:
                All_component_names.append(name)
                
        # Create parameter DataFrame
        index = [f'{name} =' for name in All_component_names]
        cols = ['Fit Params', 'Stderr', '% error']
        df = pd.DataFrame(index=index, columns=cols)
        df['Fit Params'] = np.array(popt)
        df['Stderr'] = np.array(perror)
        df['% error'] = (df['Stderr'] / df['Fit Params']) * 100
        Fit = df
        
        # Generate correlation matrix labels
        labels = [f'{name}' for name in All_component_names]
        row_labels = [label + ' :' for label in labels]
        
       # Create correlation matrix DataFrame with proper diagonal values
        CorrM = pd.DataFrame(CM, columns=labels, index=row_labels)
        CorrM = CorrM.round(3)  # Round all values first
        np.fill_diagonal(CorrM.values, 1.000)  # Set diagonal to 1.000 (3 decimal places)
        mask_upper = np.triu(np.ones(CorrM.shape), k=1).astype(bool)  # Mask above diagonal only
        CorrM = CorrM.mask(mask_upper)
        CorrM = CorrM.fillna('')  # Replace NaN with empty string

        # Format the DataFrame to ensure consistent decimal places
        pd.options.display.float_format = '{:.3f}'.format  # Force 3 decimal places globally
        matrix_df = pd.DataFrame(CorrM)
                        
        # Print results header with box drawing characters

        print( " EIS Fit Report ".center(70))
        print( "=" * 70)
        
        # Print fit statistics
        print("\nFit Statistics:")
        print("-" * 70)
        print(f"Chi-squared: {fit_stats['chi2']:.6e}")
        print(f"Reduced Chi-squared: {fit_stats['R_chi2']:.6e}")
        print(f"R-squared: {fit_stats['R_squared']:.6f}")
        print(f"Adjusted R-squared: {fit_stats['adj_R_squared']:.6f}")
        print(f"RMSE: {fit_stats['RMSE']:.6e}")
        print(f"Degrees of Freedom: {fit_stats['dof']}")
        
        # Print parameter values and errors
        print("\nParameter Values and Standard Errors:")
        print("-" * 72)
        # Use scientific notation for 'Fit Params' and 'Stderr' columns only
        pd.set_option('display.float_format', None)  # Reset global format
        Fit['Fit Params'] = Fit['Fit Params'].map('{:.3e}'.format)
        Fit['Stderr'] = Fit['Stderr'].map('{:.3e}'.format)
        # Keep percentage as regular float with 3 decimal places
        Fit['% error'] = Fit['% error'].map('{:.3f}'.format)
        
        print(Fit.to_string())
        
        # Print correlation matrix
        print("\nCorrelation Matrix:")
        print("-" * 72)
        print(matrix_df.to_string())
        print("\n")
    else:
        raise ValueError('fit_report is only meant for Single_chi = NO')
        
    return popt, perror

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
    ax.set_xlim([-0.1, upper_limit])  # Set x-axis limits
    ax.set_ylim([0, upper_limit])  # Set y-axis limits

    # Customize tick parameters and labels
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust the font size of the axis offset labels
    y_offset = ax.yaxis.get_offset_text()
    y_offset.set_size(8)
    x_offset = ax.xaxis.get_offset_text()
    x_offset.set_size(8)

    return ax


def bodePlot(axes, f, Z, param_dict, all_Z=None, plot_types=None):
    """Bode plot function that plots selected impedance components vs frequency
    
    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        List of axes objects for plotting
    f : array-like
        Frequency values
    Z : array-like
        Complex impedance values for current dataset
    param_dict : dict
        Dictionary of plotting parameters
    all_Z : list, optional
        List of all impedance datasets for batch scaling
    plot_types : list of str, optional
        List of plots to generate. Options: ['magnitude', 'phase', 'real', 'imaginary']
        Default is ['magnitude', 'phase']
    
    Returns
    -------
    list
        List of axes with the plotted data
    """
    # Input validation
    if Z is None or len(Z) == 0:
        raise ValueError("Empty or invalid impedance data")
    
    # Set default plot types if none provided
    if plot_types is None:
        plot_types = ['imaginary', 'phase']
    
    # Validate plot types
    valid_types = ['magnitude', 'phase', 'real', 'imaginary']
    if not all(ptype in valid_types for ptype in plot_types):
        raise ValueError(f"Plot types must be one or more of {valid_types}")
    
    # Validate number of axes matches plot types
    if len(axes) != len(plot_types):
        raise ValueError(f"Number of axes ({len(axes)}) must match number of plot types ({len(plot_types)})")
    
    # Ensure Z is numpy array
    Z = np.asarray(Z, dtype=complex)
    
    # Determine scaling based on all datasets if provided
    if all_Z is not None:
        max_impedance = max(np.max(np.abs(z)) for z in all_Z)
    else:
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
    
    # Plot requested components
    for ax, plot_type in zip(axes, plot_types):
        if plot_type == 'magnitude':
            ax.plot(f, np.abs(Z_scaled), **param_dict)
            ax.set_ylabel(r'$|Z| \,/\, ' + unit_label + '$', fontsize=15)
            #ax.set_yscale('log')
            #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
        elif plot_type == 'phase':
            ax.plot(f, -np.angle(Z, deg=True), **param_dict)
            ax.set_ylabel(r'$-\phi_Z \,/\, ^{\circ}$', fontsize=15)
        elif plot_type == 'real':
            ax.plot(f, Z_scaled.real, **param_dict)
            ax.set_ylabel(r'$Z^{\prime} \,/\, ' + unit_label + '$', fontsize=15)
        elif plot_type == 'imaginary':
            ax.plot(f, -Z_scaled.imag, **param_dict)
            ax.set_ylabel(r'$-Z^{\prime\prime} \,/\, ' + unit_label + '$', fontsize=15)
        
        # Common axis properties
        ax.set_xlabel(r'$ \it\nu \it$/Hz ', fontsize=15)
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        
    
    return axes

def plot_fit(f=None, Z=None, Z_fit=None, plot_type='bode'):
    """Plot EIS data as Nyquist and/or Bode plots.
    Allows plotting experimental data (Z) and fitted data (Z_fit) individually.
    
    Parameters:
    -----------
    f : array-like, optional
        Frequency values
    Z : array-like, optional
        Measured impedance values
    Z_fit : array-like, optional
        Fitted impedance values
    plot_type : str, optional
        Type of plot ('nyquist', 'bode', or 'both')
        
    Returns:
    --------
    tuple or Axes
        Figure and axes objects for the plots
    """
    if plot_type not in ['nyquist', 'bode', 'both']:
        raise ValueError("Please specify type of plot i.e. 'nyquist', 'bode' or both.")
    
    if f is not None:
        f_dat = f

    # Define common plot parameters
    data_params = {'marker': 'o', 'color': 'k', 'ms': '6', 'ls': '', 
                  'label': 'Data', 'mfc': 'none'}
    fit_params = {'marker': '*', 'color': 'darkred', 'ms': '6', 'ls': '', 
                 'label': 'Fit', 'mfc': 'none'}

    figs = []
    axes = []

    # Nyquist plot
    if plot_type in ['nyquist', 'both']:
        fig_nyq = plt.figure(figsize=(5, 5), facecolor='w', edgecolor='k')
        ax_nyq = fig_nyq.add_subplot(111)
        
        if Z is not None:
            ax_nyq = nyquistPlot(ax_nyq, Z, data_params)
        
        if Z_fit is not None:
            ax_nyq = nyquistPlot(ax_nyq, Z_fit, fit_params)
            nyquist_filename = f'Nyquist-plot_with_fit.svg'
        else:
            nyquist_filename = f'Nyquist-plot_data_only.svg'
        
        ax_nyq.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        fig_nyq.savefig(nyquist_filename, bbox_inches='tight', dpi=300)
        figs.append(fig_nyq)
        axes.append(ax_nyq)

    # Bode plots
    if plot_type in ['bode', 'both']:
        # Magnitude plot
        fig_mag = plt.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
        ax_mag = fig_mag.add_subplot(111)
        
        # Phase plot
        fig_phase = plt.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
        ax_phase = fig_phase.add_subplot(111)
        
        if Z is not None:
            ax_mag, ax_phase = bodePlot([ax_mag, ax_phase], f_dat, Z, data_params)
        
        if Z_fit is not None:
            ax_mag, ax_phase = bodePlot([ax_mag, ax_phase], f_dat, Z_fit, fit_params)
            bode_filename_mag = f'Bode-magnitude_with_fit.svg'
            bode_filename_phase = f'Bode-phase_with_fit.svg'
        else:
            bode_filename_mag = f'Bode-magnitude_data_only.svg'
            bode_filename_phase = f'Bode-phase_data_only.svg'
        
        # Add legends
        ax_mag.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        ax_phase.legend(loc='best', fontsize=10, frameon=True, framealpha=0.9)
        
        # Save plots
        fig_mag.savefig(bode_filename_mag, bbox_inches='tight', dpi=300)
        fig_phase.savefig(bode_filename_phase, bbox_inches='tight', dpi=300)
        
        figs.extend([fig_mag, fig_phase])
        axes.extend([fig_mag, fig_phase])

    plt.show()
    
    if plot_type == 'both':
        return figs, axes
    else:
        return figs[0], axes[0]
    
def full_EIS_report(freq, Z, params, circuit, UB, LB, weight_mtd, method, single_chi, plot_type='both'):
    """
    Generate complete EIS report and save to file.
    """
    # First run fit_report normally to display on screen
    popt, perror = fit_report(freq, Z, params, circuit, UB, LB, weight_mtd, method, single_chi)
    
    # Create string buffer and redirect stdout
    output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    
    # Run fit_report again to capture its output
    fit_report(freq, Z, params, circuit, UB, LB, weight_mtd, method, single_chi)
    
    # Get the captured content and restore stdout
    report_content = output.getvalue()
    sys.stdout = old_stdout
    output.close()
    
    # Save fit report in current directory
    report_file = f'EIS_fit_report.txt'
    
    # Write the captured output to file using UTF-8 encoding
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Generate plots and fitted data
    if single_chi == 'No':
        Z_function = Eqc.Z_curve_fit(circuit)
    else:
        Z_function = circuit
    
    Z_values = Z_function(freq, *popt)
    real_values = Z_values[:len(Z_values)//2]
    imag_values = Z_values[len(Z_values)//2:]
    Z_fit = real_values + 1j * imag_values
    
    # Save fitted data to file
    safe_circuit = circuit.replace('|', 'p')
    fit_data_file = f'EIS_fit_data_{safe_circuit}.txt'
    with open(fit_data_file, 'w') as file:
        # Write header information
        file.write(f"# Fitted impedance data\n")
        file.write(f"# Circuit Model: {circuit}\n")
        file.write(f"# Optimized Parameters: {', '.join(map(str, popt))}\n")
        file.write(f"# Frequency Range: {min(freq):.2e} - {max(freq):.2e} Hz\n")
        file.write("Freq.[Hz]\t\t Zs'[ohm] \t\t Zs''[ohm]\n")  # Column headers
        
        # Sort data from high to low frequency
        sorted_indices = np.argsort(freq)[::-1]  # Sort in descending order
        freq_sorted = freq[sorted_indices]
        Z_fit_sorted = Z_fit[sorted_indices]
        
        # Write data from high to low frequency
        for f, z in zip(freq_sorted, Z_fit_sorted):
            file.write(f"{f:.6e}\t{z.real:.6e}\t{z.imag:.6e}\n")
    
    axes = plot_fit(freq, Z, Z_fit, plot_type)
    
    print(f"\nFit results have been saved to: {os.path.abspath(report_file)}")
    print(f"Fitted data have been saved to: {os.path.abspath(fit_data_file)}")
    
    return popt, perror, Z_fit

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
    KB = 8.617333e-5  # Boltzmann constant in eV/K
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
    print(f"Activation Energy (EA): {results['Ea']:.4f} ± {results['delta_Ea']:.4f} eV")
    print(f"Pre-exponential (a0): {results['a0']:.7f} ± {results['delta_a0']:.7f} S·K/cm")
    print(f"RT Conductivity (σRT): {results['sigma_RT']:.5e} S/cm")
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


def modulus_plot(axes, f, Z, param_dict, diameter, thickness, plot_types=None):
    """Modulus plot function that plots selected modulus components vs frequency
    
    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        List of axes objects for plotting
    f : array-like
        Frequency values
    Z : array-like
        Complex impedance values for current dataset
    param_dict : dict
        Dictionary of plotting parameters
    diameter : float
        Sample diameter in cm
    thickness : float
        Sample thickness in cm
    plot_types : list of str, optional
        List of plots to generate. Options: ['magnitude', 'real', 'imaginary']
        Default is ['magnitude']
    
    Returns
    -------
    list
        List of axes with the plotted data
    """
    # Input validation
    if Z is None or len(Z) == 0:
        raise ValueError("Empty or invalid impedance data")
    
    # Set default plot types if none provided
    if plot_types is None:
        plot_types = ['imaginary']
    
    # Validate plot types
    valid_types = ['magnitude', 'real', 'imaginary']
    if not all(ptype in valid_types for ptype in plot_types):
        raise ValueError(f"Plot types must be one or more of {valid_types}")
    
    # Validate number of axes matches plot types
    if len(axes) != len(plot_types):
        raise ValueError(f"Number of axes ({len(axes)}) must match number of plot types ({len(plot_types)})")
    
    # Calculate modulus
    Z = np.asarray(Z, dtype=complex)
    epsilon_0 = 8.854e-14  # Vacuum permittivity in F/cm
    radius = diameter/2
    area = np.pi * radius**2
    capacitance_0 = (epsilon_0 * area)/thickness
    
    # Calculate angular frequency and complex modulus
    angular_freq = 2 * np.pi * np.array(f)
    Mreal = angular_freq * capacitance_0 * Z.real
    Mimag = angular_freq * capacitance_0 * Z.imag
    M = Mreal + 1j * Mimag

    # Plot requested components
    for ax, plot_type in zip(axes, plot_types):
        if plot_type == 'magnitude':
            ax.plot(f, np.abs(M), **param_dict)
            ax.set_ylabel(r'$|M| ' + r'$', fontsize=15)
            ax.set_yscale('log')
            #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
        elif plot_type == 'real':
            ax.plot(f, Mreal, **param_dict)
            ax.set_yscale('log')
            ax.set_ylabel(r'$M^{\prime} ' + r'$', fontsize=15)
        
        elif plot_type == 'imaginary':
            ax.plot(f, -Mimag, **param_dict)
            ax.set_ylabel(r'$-M^{\prime\prime} ' + r'$', fontsize=15)
            ax.set_yscale('log')
            #ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
        
        # Common axis properties
        ax.set_xlabel(r'$ \it\nu \it$/Hz ', fontsize=15)
        ax.set_xscale('log')
        ax.tick_params(axis='both', which='major', labelsize=10)    
    return axes


