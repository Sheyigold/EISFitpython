"""
This module provides functions for batch analysis of EIS (Electrochemical Impedance Spectroscopy) data.

Key functionality includes:
- Batch fitting of EIS data with equivalent circuit models (Batch_fit)
- Plotting Nyquist and Bode plots for multiple datasets (Nyq_stack_plot, Bode_stack_plot)  
- Arrhenius analysis and conductivity calculations (EAplot, sigma)
- Calculation of effective capacitance from constant phase elements (C_eff, Cap)


Author: Sheyi Clement Adediwura 

'Please cite the following paper if you use this code:
S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
https://doi.org/10.1039/d3ta06237f

"""

#External libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from io import StringIO
import datetime


#Custom libraries 
from EISFitpython import EISFit_main as fit
from EISFitpython import circuit_main as Eqc
from EISFitpython import data_extraction as dt



def Batch_fit(files, params, circuit, Temp, UB, LB, weight_mtd, method, single_chi='No', min_value=None, max_value=None):
    """
    Perform batch fitting of EIS data for multiple files with equivalent circuit models.
    """
    # Initialize main report storage
    full_report = StringIO()
    
    try:
        # Initialize storage lists
        results = []
        all_params = []
        all_fit_perror = []
        all_freq = []
        all_Z = []
        all_Z_fit = []
        
        T = Temp
        labels = [f"Data_{i+273}K" for i in T]
            
        # Print and store batch analysis header
        header = "\n\n" + " BATCH EIS ANALYSIS REPORT ".center(70) + "\n"
        header += "═" * 70 + f"\n\nCircuit Model: {circuit}\n" + "=" * 72
        print(header)
        full_report.write(header + "\n")

        for filename, label in zip(files, labels):
            # Create individual report buffer
            individual_output = StringIO()
            sys.stdout = individual_output
            
            # Print individual analysis header
            print(f"\n{label.center(70)}\n{'='*70}")
            
            # Read data file
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension == '.txt':
                try:
                    data = dt.readNEISYS(filename)
                except:
                    data = dt.readTXT(filename)
            elif file_extension == '.csv':
                data = dt.readCSV(filename)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
            
            # Trim data if needed
            if min_value is not None and max_value is not None:
                f, Z = dt.trim_data(data[0], data[1], fmin=min_value, fmax=max_value)
            else:
                f, Z = data
            
            # Perform fitting
            params, fit_perror = fit.fit_report(f, Z, params, circuit, UB, LB, weight_mtd, method, single_chi)
            
            # Get individual report and restore stdout
            individual_report = individual_output.getvalue()
            sys.stdout = sys.__stdout__
            individual_output.close()
            
            # Print individual report immediately
            print(individual_report)
            
            # Add to full report
            full_report.write(individual_report)
            
            # Calculate fitted impedance
            Z_function = Eqc.Z_curve_fit(circuit)
            Z_values = Z_function(f, *params)
            real_values = Z_values[:len(Z_values)//2]
            imag_values = Z_values[len(Z_values)//2:]
            Z_fit = real_values + 1j * imag_values
            
            # Save individual fit data
            fit_data_file = f'EIS_fit_{label}.txt'
            with open(fit_data_file, 'w') as file:
                # Write header information
                file.write(f"# Fitted impedance data for {label}\n")
                file.write(f"# Temperature: {T[files.index(filename)]}°C\n")
                file.write(f"# Frequency Range: {min(f):.2e} - {max(f):.2e} Hz\n")
                file.write(f"# Circuit Model: {circuit}\n")
                file.write("Freq.[Hz]\t\t Zs'[ohm] \t\t Zs''[ohm]\n")  # Column headers
                
                # Sort data from high to low frequency
                sorted_indices = np.argsort(f)[::-1]  # Sort in descending order
                f_sorted = f[sorted_indices]
                Z_fit_sorted = Z_fit[sorted_indices]
                
                # Write experimental and fitted data from high to low frequency                    
                for freq, z_val in zip(f_sorted, Z_fit_sorted):
                    file.write(f"{freq:.6e}\t{z_val.real:.6e}\t{z_val.imag:.6e}\n")
            
            # Store results
            results.append((params, fit_perror))
            all_params.append(params)
            all_fit_perror.append(fit_perror)
            all_freq.append(f)
            all_Z.append(Z)
            all_Z_fit.append(Z_fit)
            
            # Create and save Nyquist plot
            figN, ax = plt.subplots(figsize=(5, 5), facecolor='w', edgecolor='b')
            params_dict = {'marker': 'o', 'color': 'k', 'ms': 6, 'ls': '', 'label': label, 'mfc': 'grey'}
            fit.nyquistPlot(ax, Z, params_dict)
            params_dict = {'marker': '*', 'color': 'darkred', 'ms': 6, 'lw': 2.5, 'ls': '', 'label': 'Fit', 'mfc': 'none'}
            fit.nyquistPlot(ax, Z_fit, params_dict)
            ax.legend(prop={'size': 6})
            
            # Save plot in current directory
            cwd = os.getcwd()
            figname = os.path.join(cwd, label)
            figN.savefig(figname + '-Nyquist_plot.svg', bbox_inches='tight')
            plt.show()
            plt.close(figN)
            
        # Create and save Bode plot
        figB, ax = plt.subplots(nrows=2, figsize=(5, 6))
        figB.subplots_adjust(hspace=0.12)
        
        markers = ["o","h","s","p","d","v","^","<",">","x","1","2","3","4","8","H","D","P","X","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
        
        for freq, Z, Z_fit, marker, label in zip(all_freq, all_Z, all_Z_fit, markers, labels):
            params_dict = {'marker': marker, 'color': 'k', 'ms': 6, 'ls': '', 'label': label, 'mfc': 'grey'}
            fit.bodePlot(ax, freq, Z, params_dict, all_Z)
            params_dict = {'marker': '*', 'color': 'brown', 'ms': 6, 'ls': '', 'mfc': 'none'}
            fit.bodePlot(ax, freq, Z_fit, params_dict, all_Z)
        
        ax[0].legend(prop={'size': 7.5})
        ax[1].legend(prop={'size': 7})
        
        # Save Bode plot
        bode_filename = os.path.join(cwd, f'stack-bode_plot.svg')
        plt.savefig(bode_filename, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(figB)
        
    finally:
        # Get full report content
        full_report.write("\n\nAnalysis Complete\n")
        report_content = full_report.getvalue()
        full_report.close()
        
        # Save complete report
        report_file = f'Batch_EISfit_Report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"\nComplete batch analysis results have been saved to: {os.path.abspath(report_file)}")
    
    return np.array(all_params), np.array(all_fit_perror)

def Nyq_stack_plot(files, Temp):
    """Creates a stacked Nyquist plot of multiple impedance spectra."""
    
    # Create plot
    figN, ax = plt.subplots(figsize=(5, 5), facecolor='w', edgecolor='b')
    
    # Track maximum impedance for consistent scaling
    max_z = 0
    all_impedances = []
    
    
    # First pass: collect all impedance data and find maximum
    for filename in files:
        try:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension == '.txt':
                try:
                    _, impedance = dt.readNEISYS(filename)
                except:
                    _, impedance = dt.readTXT(filename)
            elif file_extension == '.csv':
                _, impedance = dt.readCSV(filename)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
                
            all_impedances.append(impedance)
            max_z = max(max_z, np.max(np.abs(impedance)))
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
            continue
    
    # Determine global scale and unit
    if max_z >= 1e9:
        scale = 1e9
        unit_label = r'\mathrm{G\Omega}'
    elif max_z >= 1e6:
        scale = 1e6
        unit_label = r'\mathrm{M\Omega}'
    elif max_z >= 1e3:
        scale = 1e3
        unit_label = r'\mathrm{k\Omega}'
    else:
        scale = 1
        unit_label = r'\mathrm{\Omega}'
    
    # Plot markers
    markers = ["o","*","h","s","p","d","v","^","<",">","x","1","2","3","4","8","H","D","P","X","|","_",0,1,2,3,4,5,6,7,8,9,10,11]

    
    # Second pass: plot with consistent scaling
    for i, (impedance, temp) in enumerate(zip(all_impedances, Temp)):
        # Scale impedance
        Z_scaled = impedance / scale
        
        # Plot parameters
        params_dict = {
            'marker': markers[i % len(markers)],
            'color': 'k',
            'ms': 5,
            'ls': '',
            'label': f'{temp}K',
            'mfc': 'none'
        }
        
        # Plot data
        ax.plot(Z_scaled.real, -Z_scaled.imag, **params_dict)
    
    # Configure plot
    ax.set_xlabel(r'$ \mathit{Z^{\prime}} \,/\, ' + unit_label + r'$', fontsize=15)
    ax.set_ylabel(r'$ \mathit{-Z^{\prime\prime}} \,/\, ' + unit_label + r'$', fontsize=15)
    ax.set_aspect('equal')
    
    # Calculate plot limits using scaled data
    x_max = max(np.max(imp.real / scale) for imp in all_impedances)
    y_max = max(np.max(-imp.imag / scale) for imp in all_impedances)
    
    # Add margins
    margin = 0.1
    ax.set_xlim([0, x_max * (1 + margin)])
    ax.set_ylim([0, y_max * (1 + margin)])
    
    # Format ticks and legend
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(prop={'size': 6})
    
    # Save plot in current directory
    plt.savefig(f'Nyq_stackplot.svg', bbox_inches='tight', dpi=300)
    
    plt.show()
    return ax


def plot_arrhenius(R_values, R_err=None, temp=None, diameter=None, thickness=None, labels=None):
    """
    Create Arrhenius plots and analyze conductivity with error propagation.
    
    Parameters:
    -----------
    R_values : list or array
        Resistance values in ohms
    R_err : list or array, optional
        Resistance error values
    temp : array
        Temperature values in Celsius
    diameter : float
        Sample diameter in cm
    thickness : float
        Sample thickness in cm
    labels : list, optional
        Labels for different components
    """
    output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150, facecolor='w', edgecolor='k')
        
        # Handle single value inputs
        if not isinstance(R_values, list):
            R_values = [R_values]
            R_err = [R_err] if R_err is not None else [None]
            labels = [labels] if labels else ['--']
        
        # Set default labels
        if not labels:
            labels = [f'Component {i+1}' for i in range(len(R_values))]
        
        # Setup markers
        default_markers = ['o', 'h', 's', '^', 'v', '<', '>', 'p', '*']
        markers = default_markers[:len(R_values)]
        
        # Calculate sample area and measurement errors
        A = np.pi * (diameter/2)**2
        l_err = 0.001  # Length measurement error in cm
        A_err = 0.01   # Area measurement error in cm^2
        T_kelvin = temp + 273
        
        # Initialize storage
        conductivities = []
        conductivity_errors = []
        
        print("\nARRHENIUS ANALYSIS REPORT")
        print("═" * 70)
        
        # Process each component
        for R_vals, R_errs, label, marker in zip(R_values, R_err, labels, markers):
            print(f"\n{label} Analysis:")
            print("-" * 70)
            
            # Calculate conductivity and error propagation
            conductivity = (1/R_vals) * (thickness/A)
            
            # Error propagation if R_err is provided
            if R_errs is not None:
                # Relative errors
                rel_R_err = R_errs / R_vals
                rel_l_err = l_err / thickness
                rel_A_err = A_err / A
                
                # Total relative error using quadrature
                total_rel_err = np.sqrt(rel_R_err**2 + rel_l_err**2 + rel_A_err**2)
                
                # Absolute conductivity error
                cond_err = conductivity * total_rel_err
            else:
                cond_err = np.zeros_like(conductivity)
            
            # Store results
            conductivities.append(conductivity)
            conductivity_errors.append(cond_err)
            
            # Plot settings
            param_dict = {
                'marker': marker,
                'color': 'k',
                'ms': '6',
                'ls': '',
                'label': label,
                'mfc': 'grey'
            }
            
            # Plot and get activation energy
            a = fit.EA(ax, T_kelvin, conductivity, param_dict)
            
            # Print conductivity values at each temperature
            print("\nConductivity values:")
            print("-" * 70)
            print("Temperature (K)    Conductivity (S/cm)    Error (S/cm)     % Error")
            print("=" * 70)
            for t, c, e in zip(T_kelvin, conductivity, cond_err):
                percent_error = (e/c) * 100 if c != 0 else float('inf')
                print(f"{t:8.1f}         {c:10.3e}          {e:10.3e}     {percent_error:8.2f}")
            print("-" * 70 + "\n")
        
        # Save figure
        figname = f'Arrhenius_plot.svg'
        plt.savefig(figname, bbox_inches='tight', dpi=500)
        plt.show()
        
    finally:
        # Get report content and restore stdout
        report_content = output.getvalue()
        sys.stdout = old_stdout
        output.close()
        
        # Save report with

        report_file = f'Arrhenius_Analysis_Report.txt'
        
        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        # Print report and file location
        print(report_content)
        print(f"\nArrhenius analysis results have been saved to: {os.path.abspath(report_file)}")
    
    return conductivities, conductivity_errors

def C_eff(R_arrays, R_err_arrays, Q_arrays, Q_err_arrays, n_arrays, n_err_arrays, T, labels=None):
    """
    Calculate effective capacitance from multiple CPE components.
    
    Parameters:
    -----------
    R_arrays : list of arrays
        List of resistance arrays for different components
    R_err_arrays : list of arrays
        List of resistance error arrays
    Q_arrays : list of arrays
        List of CPE coefficient arrays
    Q_err_arrays : list of arrays
        List of CPE coefficient error arrays
    n_arrays : list of arrays
        List of CPE exponent arrays
    n_err_arrays : list of arrays
        List of CPE exponent error arrays
    T : array
        Temperature values in Celsius
    labels : list, optional
        Labels for different components (default: ['Component 1', 'Component 2', ...])
    
    Returns:
    --------
    list of tuples
        List of (C, C_err) pairs for each component
    """
    output = StringIO()
    old_stdout = sys.stdout
    sys.stdout = output
    
    try:
        # Set default labels if none provided
        if labels is None:
            labels = [f'Component {i+1}' for i in range(len(R_arrays))]
        
        print("\nEFFECTIVE CAPACITANCE ANALYSIS REPORT")
        print("═" * 70)
        
        # Store results
        all_results = []
        
        # Process each component
        for R, R_err, Q, Q_err, n, n_err, label in zip(
            R_arrays, R_err_arrays, Q_arrays, Q_err_arrays, 
            n_arrays, n_err_arrays, labels):
            
            print(f"\n{label} Analysis:")
            print("-" * 70)
            
            # Ensure inputs are numpy arrays
            R = np.atleast_1d(R)
            R_err = np.atleast_1d(R_err)
            Q = np.atleast_1d(Q)
            Q_err = np.atleast_1d(Q_err)
            n = np.atleast_1d(n)
            n_err = np.atleast_1d(n_err)
            T_kelvin = np.atleast_1d(T) + 273
            
            # Compute effective capacitance and its error
            with np.errstate(divide='ignore', invalid='ignore'):
                C = np.where(n != 0, (Q * R**(1 - n))**(1 / n), 0)
                C_err = np.where((Q != 0) & (n != 0) & (R != 0),
                                C * ((Q_err / Q) + (n_err / n) + (R_err / R)),
                                np.inf)
            
            # Print results for this component
            print(f"\nCapacitance values for {label}:")
            print("Temperature (K)    Capacitance (F)    Error (F)      % Error")
            print("=" * 70)
            
            for ti, Ci, err in zip(T_kelvin, C, C_err):
                percent_error = (err / Ci) * 100 if Ci != 0 else float('inf')
                print(f"{ti:8.1f}         {Ci:.3e}       {err:.3e}    {percent_error:8.2f}")
            print("-" * 70)
            
            all_results.append((C, C_err))
        
    finally:
        # Get report content and restore stdout
        report_content = output.getvalue()
        sys.stdout = old_stdout
        output.close()
        
        # Save report with
        report_file = f'Effective_Capacitance_Values.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Print report to screen
        print(report_content)
        print(f"\nEffective capacitance results saved to: {os.path.abspath(report_file)}")
    
    return all_results

