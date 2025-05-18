"""
EIS Fitting Module for Single and Multiple Impedance Datasets

This module provides functions for fitting and analyzing Electrochemical Impedance 
Spectroscopy (EIS) data using equivalent circuit models. Core functionalities include:

- Flexible parameter handling supporting both global and local fitting:
    - Global parameters: shared across all datasets (e.g. for temperature-independent elements)
    - Local parameters: unique for each dataset (e.g. for temperature-dependent elements)
- Parameter flattening and structuring for circuit elements
- Impedance calculation for various circuit elements
- Fitting report generation with parameter statistics
- Nyquist and Bode plot visualization

Particularly useful for variable temperature (VT) EIS analysis where some circuit
elements may have temperature dependence while others remain constant.

Author: Sheyi Clement Adediwura 

'Please cite the following paper if you use this code:
S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
https://doi.org/10.1039/d3ta06237f


"""


#External libraries
import numpy as np 
import re
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys
import datetime
from io import StringIO

#Custom libraries 
from EISFitpython import EISFit_main as fit 
from EISFitpython import data_extraction as dt 
from EISFitpython import circuit_main as ct

# Function to flatten parameters for circuit fitting
def flatten_params(params, circuit_str=None, N_sub=None):
    """
    Flatten parameters with awareness of global vs per-sublist parameters.
    
    Parameters:
    ----------
    params : tuple or list
        Parameters in order matching circuit elements:
        - scalar/single value = global parameter
        - list/array of length N_sub = per-sublist parameter
    circuit_str : str
        Circuit string (e.g. "(R1|Q1)+(R2|W1)+L1")
    N_sub : int, optional
        Number of sublists for validating per-sublist parameters
        
    Returns:
    -------
    list
        Flattened parameter list preserving global/per-sublist structure
    """
    flat = []
    
    def is_global_param(p):
        if not hasattr(p, '__iter__') or isinstance(p, (str, bytes)):
            return True
        return len(p) != N_sub
    
    def validate_param(p, elem_type, elem_num):
        if N_sub is not None and not is_global_param(p):
            if len(p) != N_sub:
                raise ValueError(
                    f"Parameter for {elem_type}{elem_num} has length {len(p)}, "
                    f"expected {N_sub} for per-sublist parameter"
                )
    
    if circuit_str and N_sub:
        # Parse circuit elements including L, W, and F
        elements = re.findall(r'([RQCLWF])(\d+)', circuit_str)
        param_idx = 0
        
        for elem_type, elem_num in elements:
            if elem_type in ['R', 'L', 'C']:
                # R, L, C elements need 1 parameter
                validate_param(params[param_idx], elem_type, elem_num)
                param_idx += 1
            elif elem_type == 'Q':
                # Q elements need 2 parameters (Q, n)
                validate_param(params[param_idx], elem_type, elem_num)
                validate_param(params[param_idx + 1], f'n(Q{elem_num})', '')
                param_idx += 2
            elif elem_type == 'W':
                # Warburg short needs 2 parameters (W, n)
                validate_param(params[param_idx], elem_type, elem_num)
                validate_param(params[param_idx + 1], f'n(W{elem_num})', '')
                param_idx += 2
            elif elem_type == 'F':
                # Finite length Warburg needs 3 parameters (F, n, δ)
                validate_param(params[param_idx], elem_type, elem_num)
                validate_param(params[param_idx + 1], f'n(F{elem_num})', '')
                validate_param(params[param_idx + 2], f'δ(F{elem_num})', '')
                param_idx += 3
    
    # Flatten while preserving parameter structure
    for p in params:
        if is_global_param(p):
            # Global parameter - add single value
            flat.append(p if not hasattr(p, '__iter__') else p[0])
        else:
            # Per-sublist parameter - extend with all values
            flat.extend(p)
    
    return flat

def Single_chi(f, *params, circuit_str, circuit_type='fit'):
    """
    Compute impedance dynamically for sublists based on a flexible circuit structure.
    """
    sublist, _ = dt.split_array(f, Z=None, split_freq=1e6)
    N_sub = len(sublist)
    
    Zs = []
    
    def build_params_for_sublist(i):
        """Build parameter list for sublist i"""
        new_params = []
        for p in params:
            if hasattr(p, '__iter__') and not isinstance(p, (str, bytes)):
                val = p[i] if len(p) == N_sub else p[0]
                new_params.append(val)
            else:
                new_params.append(p)
        return new_params
    
    try:
        # Process each sublist
        for i, fi in enumerate(sublist):
            params_i = build_params_for_sublist(i)
            Zi = ct.compute_impedance(params_i, circuit_str, fi)
            Zs.append(Zi)
        
        # Combine results
        Z = np.hstack(Zs)
        
        # Return appropriate format
        if circuit_type == 'predict':
            return Z
        elif circuit_type == 'fit':
            return np.hstack([np.real(Z), np.imag(Z)])
        else:
            raise ValueError('circuit_type must be "fit" or "predict"')
            
    except Exception as e:
        print(f"Error in impedance calculation: {e}")
        return np.zeros(len(f) * 2)  # Return zeros instead of NaN
    

def format_circuit_output(popt, perror, CorrM, circuit_str, N_sub,Temp, param_template):
    """
    Format circuit-aware parameter output and correlation matrix.
    
    Parameters:
    ----------
    popt : array-like
        Optimized parameters from fitting
    perror : array-like
        Parameter errors
    CorrM : array-like
        Correlation matrix
    circuit_str : str
        Circuit string (e.g. "(R1|Q1)+(R2|Q2)+Q3")
    N_sub : int
        Number of sublists
    param_template : list
        Template indicating which parameters are per-sublist
    """
    elements = re.findall(r'([RQCLWF])(\d+)', circuit_str)
    param_labels = []
    param_values = []
    param_errors = []
    template_idx = 0
    flat_idx = 0
    
    for elem_type, elem_num in elements:
        if template_idx >= len(param_template):
            break
            
        try:
            if elem_type in ['R', 'L', 'C']:
                if param_template[template_idx]:
                    # Per-sublist parameter with temperature
                    for i in range(N_sub):
                        param_labels.append(f"{elem_type}{elem_num}_{Temp[i]}°C")
                        param_values.append(popt[flat_idx + i])
                        param_errors.append(perror[flat_idx + i])
                    flat_idx += N_sub
                else:
                    # Global parameter
                    param_labels.append(f"{elem_type}{elem_num}")
                    param_values.append(popt[flat_idx])
                    param_errors.append(perror[flat_idx])
                    flat_idx += 1
                template_idx += 1
                    
            elif elem_type == 'Q':
                # Q parameter
                if param_template[template_idx]:
                    for i in range(N_sub):
                        param_labels.append(f"Q{elem_num}_{Temp[i]}°C")
                        param_values.append(popt[flat_idx + i])
                        param_errors.append(perror[flat_idx + i])
                    flat_idx += N_sub
                else:
                    param_labels.append(f"Q{elem_num}")
                    param_values.append(popt[flat_idx])
                    param_errors.append(perror[flat_idx])
                    flat_idx += 1
                template_idx += 1
                
                # n parameter
                if template_idx < len(param_template):
                    if param_template[template_idx]:
                        for i in range(N_sub):
                            param_labels.append(f"n{elem_num}_{Temp[i]}°C")
                            param_values.append(popt[flat_idx + i])
                            param_errors.append(perror[flat_idx + i])
                        flat_idx += N_sub
                    else:
                        param_labels.append(f"n{elem_num}")
                        param_values.append(popt[flat_idx])
                        param_errors.append(perror[flat_idx])
                        flat_idx += 1
                    template_idx += 1
                    
        except IndexError as e:
            print(f"Error processing {elem_type}{elem_num}: {e}")
            print(f"Template index: {template_idx}, Flat index: {flat_idx}")
            break

    # Create parameter DataFrame
    df = pd.DataFrame({
        'Fit_Params': [f"{label} = " for label in param_labels],
        'Value': param_values,
        'Error': param_errors,
        '% error': np.array(param_errors) / np.array(param_values) * 100
    })
    
    # Format correlation matrix
    matrix_size = min(len(param_labels), len(CorrM))
    matrix_labels = param_labels[:matrix_size]
    CorrM_subset = CorrM[:matrix_size, :matrix_size]
    
    CorrM_df = pd.DataFrame(
        CorrM_subset,
        columns=matrix_labels,
        index=[f"{label} :" for label in matrix_labels]
    )
    
    mask_upper = np.triu(np.ones(CorrM_df.shape), k=0).astype(np.bool_)
    CorrM_df = CorrM_df.mask(mask_upper).round(3)
    CorrM_df = CorrM_df.fillna('')
    
    return df, CorrM_df



def Single_chi_report(f, Z, params, Temp, circuit_str, weight_mtd='M'):
    """
    Run fitting routine with circuit-aware parameter printing.
    
    Parameters:
    ----------
    f : array-like
        Frequency data
    Z : array-like
        Impedance data
    params : tuple
        Original structured parameters
    Temp : array-like
        Temperature values in Celsius
    circuit_str : str
        Circuit string (e.g. "(R1|Q1)+(R2|Q2)+Q3")
    weight_mtd : str, optional
        Weighting method for fitting (default: 'M')
    
    Returns:
    -------
    tuple
        (popt, perror, Z_fit, figN, figB) - Optimized parameters, errors, fitted impedance, 
        and figure handles for Nyquist and Bode plots
    """
    report_content = []
    
    try:
        # Split data into sublists
        sublist, _ = dt.split_array(f, Z=None, split_freq=1e6)
        N_sub = len(sublist)
        
        # Create parameter template
        param_template = []
        for p in params:
            if hasattr(p, '__iter__') and not isinstance(p, (str, bytes)):
                param_template.append(len(p) == N_sub)
            else:
                param_template.append(False)
        
        # Flatten parameters
        flat_params = flatten_params(params, circuit_str=circuit_str, N_sub=N_sub)
        
        def circuit_callable(f, *flat_params):
            idx = 0
            structured_params = []
            for is_per_sublist in param_template:
                if is_per_sublist:
                    structured_params.append(flat_params[idx:idx+N_sub])
                    idx += N_sub
                else:
                    structured_params.append(flat_params[idx])
                    idx += 1
            return Single_chi(f, *structured_params, circuit_str=circuit_str, circuit_type='fit')
        
        # Run EISFit
        fit_stats, popt, perror, CorrM = fit.EISFit(
            f, Z, flat_params, circuit_callable, 
            UB=None, LB=None, 
            weight_mtd=weight_mtd, method='lm',
            single_chi='Yes'
        )
        
        # Format output with circuit awareness
        param_df, corr_df = format_circuit_output(np.abs(popt), np.abs(perror), CorrM, circuit_str, N_sub,Temp, param_template)
        
        # Build report content
        report_content.extend([
            "\n Global-Local EIS Analysis Report ".center(70),
            "═" * 70,
            f"\nAnalysis Type: Single Chi-Square Minimization",
            f"Temperature Range: {min(Temp)}-{max(Temp)}°C",
            f"Number of Datasets: {N_sub}",
            f"Number of Parameters: {len(popt)}",
            "\nFit Statistics:",
            "-" * 70,
            f"Chi-squared: {fit_stats['chi2']:.6e}",
            f"Reduced Chi-squared: {fit_stats['R_chi2']:.6e}",
            f"R-squared: {fit_stats['R_squared']:.6f}",
            f"Adjusted R-squared: {fit_stats['adj_R_squared']:.6f}",
            f"RMSE: {fit_stats['RMSE']:.6e}",
            f"Degrees of Freedom: {fit_stats['dof']}",
            "\nParameter Values and Standard Errors:",
            "-" * 72
        ])
        
        # Format parameter DataFrame with improved spacing
        param_df['Value'] = param_df['Value'].map('{:.3e}'.format)
        param_df['Error'] = param_df['Error'].map('{:.3e}'.format)
        param_df['% error'] = param_df['% error'].map('{:.3f}'.format)
        
        # Add spacing and alignment
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.colheader_justify', 'center')
        
        # Format the DataFrame string with custom spacing
        param_str = param_df.to_string(
            index=False,
            justify='center',
            col_space={
                'Fit_Params': 12,
                'Value': 15,
                'Error': 15,
                '% error': 15
            },
            formatters={
                'Value': lambda x: f"{x:>12}",
                'Error': lambda x: f"{x:>12}",
                '% error': lambda x: f"{x:>8}"
            }
        )
        
        report_content.append(param_str)
        
        # Add correlation matrix
        report_content.extend([
            "\nCorrelation Matrix:",
            "-" * 72,
            corr_df.to_string(),
            "\n"
        ])
        
        # Join and display report
        full_report = "\n".join(report_content)
        print(full_report)
        
        # Calculate fitted impedance
        def predict_callable(f, *flat_params):
            idx = 0
            structured_params = []
            for is_per_sublist in param_template:
                if is_per_sublist:
                    structured_params.append(flat_params[idx:idx+N_sub])
                    idx += N_sub
                else:
                    structured_params.append(flat_params[idx])
                    idx += 1
            return Single_chi(f, *structured_params, circuit_str=circuit_str, circuit_type='predict')
        
        Z_fit = predict_callable(f, *popt)

        # Split data into sublists with proper indexing
        sublist_indices = []
        current_idx = 0
        for i in range(N_sub):
            # Get the frequency subset for this temperature
            subset, _ = dt.split_array(f[current_idx:], Z=None, split_freq=np.max(f))
            if not subset:
                raise ValueError(f"No data found for temperature {Temp[i]}°C")
            
            points_in_subset = len(subset[0])  # Get number of points in this subset
            sublist_indices.append((current_idx, current_idx + points_in_subset))
            current_idx += points_in_subset
        
        # Save fitted data - one file per temperature
        for i in range(N_sub):
            # Get data for current subset using the stored indices
            start_idx, end_idx = sublist_indices[i]
            
            f_sub = f[start_idx:end_idx]
            Z_fit_sub = Z_fit[start_idx:end_idx]

            # Create filename with temperature in working directory
            fit_data_file = f'S-chi_fit_data_{Temp[i]}C.txt'
            

            # Sort from high to low frequency
            sorted_indices = np.argsort(f_sub)[::-1]
            f_sorted = f_sub[sorted_indices]
            Z_fit_sorted = Z_fit_sub[sorted_indices]
            
            # Get relevant parameters for this temperature
            temp_params = []
            param_idx = 0
            for is_per_sublist in param_template:
                if is_per_sublist:
                    temp_params.append(popt[param_idx + i])
                    param_idx += N_sub
                else:
                    temp_params.append(popt[param_idx])
                    param_idx += 1
            
            with open(fit_data_file, 'w') as file:
                # Write header information
                file.write(f"# Single Chi-Square Fitted Data\n")
                file.write(f"# Temperature: {Temp[i]}°C\n")
                file.write(f"# Frequency Range: {min(f_sub):.2e} - {max(f_sub):.2e} Hz\n")
               
                # Write column headers
                file.write("\nFreq.[Hz]\t Zs'[ohm] \t Zs''[ohm]\n")
                
                # Write frequency and impedance data
                for freq, z_val in zip(f_sorted, Z_fit_sorted):
                    file.write(f"{freq:.6e}\t{z_val.real:.6e}\t{z_val.imag:.6e}\n")
        
        print("\nFitted data files saved:")
        for i in range(N_sub):
            print(f"Temperature {Temp[i]}°C: {os.path.abspath(f'S-chi_fit_data_{Temp[i]}C.txt')}")
     
        # Generate and save plots
        figN, figB = generate_plots(f, Z, Z_fit, Temp)
        
        # Save report
        report_file = f'Single-chi_Report.txt'
        report_path = os.path.join(os.getcwd(), report_file)
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
            
        print(f"\nFit report has been saved to: {os.path.abspath(report_path)}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None, None, None, None, None
    
    return popt, perror, Z_fit

def generate_plots(f, Z, Z_fit, Temp):
    """Generate Nyquist and Bode plots and save them."""
    # Split frequency and impedance data for plotting
    f_s, Z_data_s = dt.split_array(f, Z, split_freq=1e6)
    f_s, Z_fit_s = dt.split_array(f, Z_fit, split_freq=1e6)
    
   # Save plot in current directory
    cwd = os.getcwd()
    
    T = Temp
    markers = ["o", "h", "^", "s", "p", "d", "v", "^", "<", ">", "1", "2", "3", "4", "8", "H", "D", "P", "X", "|", "_"]
    labels = [f"Data_{x+273}K" for x in T]
    # Create Nyquist plots and save each one
    nyquist_figs = []
    for i, (fi, Zi, Zf, marker, label) in enumerate(zip(f_s, Z_data_s, Z_fit_s, markers, labels)):
        figN, ax = plt.subplots(figsize=(5, 5))
        # Data points
        params_dict = {'marker': marker, 'color': 'k', 'ms': 6, 
                      'ls': '', 'label': label, 'mfc': 'grey'}
        fit.nyquistPlot(ax, Zi, params_dict)
        
        # Fitted line
        params_dict = {'marker': '*', 'color': 'darkred', 
                      'ms': 6, 'ls': '', 'label':'Fit','mfc': 'none'}
        fit.nyquistPlot(ax, Zf, params_dict)
    
        ax.legend(prop={'size': 7})
        
        
        # Save Nyquist plot
        figname = os.path.join(cwd, label)
        plt.savefig(figname + '_S-chi_nyquist_plot.svg', bbox_inches='tight', dpi=300)
        nyquist_figs.append(figN)
    
    # Create and save Bode plot
    figB, ax = plt.subplots(nrows=2, figsize=(5, 6))
    figB.subplots_adjust(hspace=0.12)
    
    for fi, Zi, Zf, marker, label in zip(f_s, Z_data_s, Z_fit_s, markers, labels):
        # Data points
        params_dict = {'marker': marker, 'color': 'k', 'ms': 6, 
                      'ls': '', 'label': label, 'mfc': 'grey'}
        #fit.bodePlot(ax, fi, Zi, params_dict)
        fit.bodePlot(ax, fi, Zi, params_dict, Z_data_s)
        
        # Fitted line
        params_dict = {'marker': '*', 'color': 'brown', 
                      'ms': 6, 'ls': '', 'mfc': 'none'}
        fit.bodePlot(ax, fi, Zf, params_dict,Z_data_s)
    
    ax[0].legend(prop={'size': 7.5})
    ax[1].legend(prop={'size': 7})
    
    # Save Bode plot
    figname = os.path.join(cwd, label)
    plt.savefig(figname + '_S-chi_stack_bode_plot.svg', bbox_inches='tight', dpi=300)
    
    plt.show()
    return nyquist_figs[-1], figB  # Return the last Nyquist figure and Bode figure for display