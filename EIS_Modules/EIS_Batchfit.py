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
import datetime
import numpy as np
import matplotlib.pyplot as plt


#Custom libraries 
import EISFit_main as fit
import circuit_main as Eqc
import data_extraction as dt



def Batch_fit(files, params, circuit, Temp, UB, LB, weight_mtd, method, single_chi='No', min_value=None, max_value=None):
    """
    Perform batch fitting of EIS data for multiple files.

    Parameters:
    -----------
    files : list
        List of file paths containing EIS data
    params : list
        Initial parameters for fitting
    circuit : str
        Equivalent circuit model
    UB : list
        Upper bounds for parameters
    LB : list
        Lower bounds for parameters
    weight_mtd : str
        Weighting method for fitting
    method : str
        Optimization method
    single_chi : str, optional
        Use single chi calculation (default 'No')
    min_value : float, optional
        Minimum frequency cutoff
    max_value : float, optional
        Maximum frequency cutoff

    Returns:
    --------
    tuple
        (all_params, all_fit_perror) containing fitted parameters and errors
    """
    # Initialize lists to store results
    results = []
    all_params = []
    all_fit_perror = []
    
    # Store frequency and impedance data for Bode plot
    all_freq = []
    all_Z = []
    all_Z_fit = []
    
    T= Temp
    labels = [f"Data_{i+273}K" for i in T]
    # Process each file

    mypath = '../Figures'
    # Create directory for saving figures if it doesn't exist
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    for filename,label in zip(files,labels):
        
        # Read EIS data based on file extension
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension == '.txt':
            # Try NEISYS format first, if it fails, try plain TXT format
            try:
                data = dt.readNEISYS(filename)
            except:
                data = dt.readTXT(filename)
        elif file_extension == '.csv':
            data = dt.readCSV(filename)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Trim data if frequency limits provided
        if min_value is not None and max_value is not None:
            f, Z = dt.trim_data(data[0], data[1], fmin=min_value, fmax=max_value)
        else:
            f, Z = data 
        
      # Print batch analysis header
        print("\n")
        print("╔" + "═" * 70 + "╗")  # Top border with corners
        print("║" + f" {label} EIS ANALYSIS REPORT ".center(70) + "║")  # Uppercase for main header
        print("╚" + "═" * 70 + "╝")  # Bottom border with corners
              # Fit the data using provided parameters
        params, fit_perror = fit.fit_report(f, Z, params, circuit, UB, LB, weight_mtd, method, single_chi)

        # Calculate fitted impedance values
        Z_function = Eqc.Z_curve_fit(circuit)
        Z_values = Z_function(f, *params)

        # Convert to complex impedance
        real_values = Z_values[:len(Z_values)//2]
        imag_values = Z_values[len(Z_values)//2:]
        Z_fit = real_values + 1j * imag_values
        
        
        # Store data for Bode plot
        all_freq.append(f)
        all_Z.append(Z)
        all_Z_fit.append(Z_fit)
        

        # Create Nyquist plot
        figN, ax = plt.subplots(figsize=(5, 5), dpi=150, facecolor='w', edgecolor='b')

        # Plot measured data
        params_dict = {'marker': 'o', 'color': 'k', 'ms': 6, 'ls': '', 'label': label,'mfc': 'grey'}
        a = fit.nyquistPlot(ax, Z, params_dict)

        # Plot fitted data
        params_dict = {'marker': '*', 'color': 'darkred', 'ms': 6, 'lw': 2.5, 'ls': '', 'label': 'Fit', 'mfc': 'none'}
        a = fit.nyquistPlot(ax, Z_fit, params_dict)

        
        # Store results
        results.append((params, fit_perror))
        all_params.append(params)
        all_fit_perror.append(fit_perror)
        ax.legend(prop={'size': 6})
        
        # Save plot
        myfile = label 
        figname = os.path.join(mypath, myfile)
        figN.savefig(figname +'-Nyquist_plot.svg', bbox_inches='tight')
        
        plt.show()
    
    # Create and save Bode plot
    figB, ax = plt.subplots(nrows=2, figsize=(5, 6), dpi=150)
    figB.subplots_adjust(hspace=0.12)
    
    markers = ["o","h","s","p","d","v","^","<",">","x","1","2","3","4","8","H","D","P","X","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
    
    for freq, Z, Z_fit, marker, label in zip(all_freq, all_Z, all_Z_fit, markers, labels):
        # Data points
        params_dict = {'marker': marker, 'color': 'k', 'ms': 6, 
                      'ls': '', 'label': label, 'mfc': 'grey'}
        fit.bodePlot(ax, freq, Z, params_dict, all_Z)
        
        # Fitted line
        params_dict = {'marker':'*', 'color': 'brown', 
                      'ms': 6, 'ls': '', 'mfc': 'none'}
        fit.bodePlot(ax, freq, Z_fit, params_dict, all_Z)
    
    ax[0].legend(prop={'size': 7.5})
    ax[1].legend(prop={'size': 7})
    
    # Save Bode plot
    timestamp = datetime.datetime.now().strftime("%m-%d_%S")
    plt.savefig(f'{mypath}/stack-bode_plot_{timestamp}.svg', bbox_inches='tight', dpi=300)
    plt.show()

    return np.array(all_params), np.array(all_fit_perror)

def Nyq_stack_plot(files, Temp):
    """Creates a stacked Nyquist plot of multiple impedance spectra."""
    
    # Create plot
    figN, ax = plt.subplots(figsize=(5, 5), dpi=150, facecolor='w', edgecolor='b')
    
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
    
    # Save plot
    mypath = '../Figures'
    if not os.path.exists(mypath):
        os.makedirs(mypath)
    timestamp = datetime.datetime.now().strftime("%m-%d_%S")
    plt.savefig(f'{mypath}/Nyq_stackplot_{timestamp}.svg', bbox_inches='tight', dpi=300)
    
    plt.show()
    return ax


def plot_arrhenius(R_values, temp, diameter, thickness, labels=None):
    """
    Create Arrhenius plots for resistance components.
    
    Parameters:
    -----------
    R_values : list of arrays or single array
        List of resistance arrays [Rb, Rgb, Rt] or single resistance array
    temp : array-like
        Temperature points in Celsius
    diameter : float
        Sample diameter in cm
    thickness : float
        Length/thickness of the sample in cm
    labels : list or str, optional
        List of labels ['Bulk', 'Gb', 'Total'] or single label
    markers : list or str, optional
        List of markers ['o', 'h', 's'] or single marker
    save_path : str, optional
        Path to save the figure
    filename : str, optional
        Base filename for saving
    dpi : int, optional
        Figure resolution (default: 1000)
        
    Returns:
    --------
    tuple
        (ax, conductivities) containing the plot axis and list of conductivities
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150, facecolor='w', edgecolor='k')
    
    # Convert single resistance to list for consistent handling
    if not isinstance(R_values, list):
        R_values = [R_values]
        labels = [labels] if labels else ['--']
    
    # Set default labels and markers if not provided
    if not labels:
        labels = [f'-- {i+1}' for i in range(len(R_values))]
  
    default_markers = ['o', 'h', 's', '^', 'v', '<', '>', 'p', '*']
    markers = default_markers[:len(R_values)]
    
    # Calculate sample area
    A = np.pi * (diameter/2)**2
    
    # Convert temperature to Kelvin
    T_kelvin = temp + 273
    
    conductivities = []
    
    # Plot each component with unique colors
    for R_values, label, marker in zip(R_values, labels, markers):
        # Calculate conductivity
        conductivity = (1/R_values) * (thickness/A)
        
        # Set plotting parameters
        param_dict = {
            'marker': marker,
            'color': 'k',
            'ms': '6',
            'ls': '',
            'label': label,
            'mfc': 'grey'  # Empty markers
        }
        
        # Print analysis results
        print(f"\n{label} Arrhenius Analysis Results:")
        print("=" * 53)
        
        # Plot using EA function
        a = fit.EA(ax, T_kelvin, conductivity, param_dict)
        conductivities.append(conductivity)

    # Save figure if path provided
    save_path= '../Figures'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = 'Arrhenius_plot'
    
    timestamp = datetime.datetime.now().strftime("%m-%d_%S")
    figname = os.path.join(save_path, filename + f'_{timestamp}.svg')
    plt.savefig(figname, bbox_inches='tight', dpi=500)
    
    plt.show()
    
    return conductivities

def sigma_report(R, Rerr, l, con, T):
        """
        Calculate conductivity and its error for given resistance measurements.
        
        Parameters:
        -----------
        R : array-like
            Resistance values in ohms
        Rerr : array-like 
            Resistance error values
        l : float
            Sample length/thickness in cm
        con : array-like 
            Conductivity values
        T : array-like
            Temperature values in Celsius
            
        Returns:
        --------
        None. Prints conductivity values with errors.
        """
        l_err = 0.001  # Length measurement error in cm
        A_err = 0.01   # Area measurement error in cm^2
        A = np.pi * (1.3 / 2)**2  # Sample area in cm^2
        factor = (l_err / l) + (A_err / A) + (Rerr / R)  # Total error factor
        
        # Convert inputs to numpy arrays
        T_arr = np.atleast_1d(T)
        con_arr = np.atleast_1d(con)
        factor_arr = np.atleast_1d(factor)
        C_err = con_arr * factor_arr  # Calculate conductivity error
        
        print('                    Conductivities               Stderr               % errors')
        print('===============================================================================')
        
        # Print conductivity values with errors for each temperature
        for temp, sigma_val, err in zip(T_arr, con_arr, C_err):
            percent_error = (err / sigma_val) * 100 if sigma_val != 0 else float('inf')
            print(f" Sigma_@ {temp}ºC = {float(sigma_val)}   +/-  {float(err)}    {percent_error:.4f}  ")
    
def C_eff(R, dR, Q, dQ, n, dn, T):
    """
    Calculate effective capacitance from constant phase element parameters.

    Parameters:
    -----------
    R : float or array-like
        Resistance values in ohms
    dR : float or array-like
        Resistance error values
    Q : float or array-like
        CPE coefficient values
    dQ : float or array-like
        CPE coefficient error values 
    n : float or array-like
        CPE exponent values
    dn : float or array-like
        CPE exponent error values
    T : float or array-like
        Temperature values in Celsius

    Returns:
    --------
    tuple
        (C, C_err) where:
        C : float or array-like
            Effective capacitance values
        C_err : float or array-like 
            Effective capacitance error values
    """
    # Ensure inputs are numpy arrays for element-wise operations
    R = np.atleast_1d(R)
    dR = np.atleast_1d(dR)
    Q = np.atleast_1d(Q)
    dQ = np.atleast_1d(dQ)
    n = np.atleast_1d(n)
    dn = np.atleast_1d(dn)
    T = np.atleast_1d(T)
    
    # Compute effective capacitance and its error
    C = (Q * R**(1 - n))**(1 / n)

    #Propagate uncertainties via the power-law log-based formula:
    # Terms for relative contributions
    term_Q = (1.0 / n) * (dQ / Q)
    term_R = ((1.0 - n) / n) * (dR / R)
    term_n = (np.log(Q * R**(1 - n)) / n**2) * dn
    
    # Combined relative error
    rel_error = np.sqrt(term_Q**2 + term_R**2 + term_n**2)

    C_err= C * rel_error
    
    print('                    Effective capacitance               Stderr               % errors')
    print('========================================================================================')
    for ti, Ci, err in zip(T, C, C_err):
        percent_error = (err / Ci) * 100 if Ci != 0 else float('inf')
        print(f" C(F)_@ {ti}ºC = {Ci}   +/-  {err}    {percent_error:.4f}  ")
    
    # Return scalar if input was scalar, else return arrays
    if C.size == 1:
        return C[0], C_err[0]
    return C, C_err

   

def Cap(C, C_err, T):
    """
    Print capacitance values with their errors and percentage errors.

    Parameters:
    -----------
    C : array-like
        Capacitance values in Farads
    C_err : array-like
        Capacitance error values
    T : array-like
        Temperature values in Celsius

    Returns:
    --------
    tuple
        (C, C_err) Returns the input capacitance and error arrays
    """
    print('                    Effective capacitance               Stderr               % errors')
    print('========================================================================================')
    for i,j,k in zip(T,C,C_err):
        print(f" C(F)_@ {i}ºC = {j}   +/-  {k}    {(k/j)*100:.4f}  ")
    
    return C, C_err

