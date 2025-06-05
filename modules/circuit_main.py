"""
EIS Circuit Module

This module provides functionality for computing and fitting electrical impedance spectroscopy (EIS) data
using equivalent circuit models. It includes tools for calculating complex impedance of various circuit
elements and their combinations in series and parallel configurations.

The module supports the following circuit elements:
- R: Resistor 
- C: Capacitor
- L: Inductor
- W: Warburg element (short)
- Q: Constant Phase Element (CPE)
- F: Finite-length Warburg element

Circuit strings can be constructed using:
- '+' for series connections 
- '|' for parallel connections
- '()' for nested circuits

Example circuit string: "R+(C|R)"  # Represents a resistor in series with a parallel RC circuit

Functions:
    compute_impedance(p, circuit_str, freqs): Computes complex impedance for a given circuit
    Z_curve_fit(circuit_str): Generates impedance fitting function for curve_fit
    Z_gen(circuit_string, return_type): Generates impedance fitting function with specified return type

Dependencies:
    - numpy
    - re
    
Author: Sheyi Clement Adediwura 

'Please cite the following paper if you use this code:
S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
https://doi.org/10.1039/d3ta06237f

"""

#External libraries
import numpy as np
import re



def compute_impedance(p, circuit_str, freqs):
    """
    Compute the complex impedance of a given circuit string based on the provided equations.
    
    Parameters:
    - params (list): List of parameters corresponding to circuit elements.
    - circuit_str (str): String representation of the circuit.
    - freqs (array): Frequencies at which impedance should be calculated.
    
    Returns:
    - array: Impedances for each frequency.
    """
    
    w = 2 * np.pi * freqs # Angular frequency # Angular frequency
    
    # Map of circuit elements to their impedance calculations
    def impedance(element, params, freq):
        if element == 'R':#Resistor
            return params[0]
        elif element == 'C':#Capacitor
            return 1 / (1j * w * params[0])
        elif element == 'L':#Inductor
            return 1j * w * params[0]
        elif element == 'W':#Warburg elemen short
            return params[0]*(1-1j)/np.sqrt(w)
        elif element == 'Q':#Constant phase element
            return 1.0 / (params[0] * (1j * w) ** params[1])
        elif element == 'F':#Warburg elemen finite lenght
            return params[0]*np.tanh(np.sqrt(1j*w*params[1]))/np.sqrt(1j*w*params[1])
        
        else:
            raise ValueError(f"Unsupported element: {element}")

    # Recursive function to evaluate circuit impedance
    def evaluate_circuit(circuit_str, param_offset):
        if '+' in circuit_str:  # Series components
            components = circuit_str.split('+')
            impedances = []
            for comp in components:
                imp, param_offset = evaluate_circuit(comp, param_offset)
                impedances.append(imp)
            total_impedance = sum(impedances)
        elif '|' in circuit_str:  # Parallel components
            components = circuit_str.split('|')
            impedances = []
            for comp in components:
                imp, param_offset = evaluate_circuit(comp, param_offset)
                impedances.append(imp)
            # Parallel impedance calculation
            total_impedance = 1 / sum([1 / imp for imp in impedances])
        elif '(' in circuit_str:  # Nested circuits
            inner_circuit = circuit_str[1:-1]
            total_impedance, param_offset = evaluate_circuit(inner_circuit, param_offset)
        else:  # Individual components        
            # Match against R, C, L, Q, W, F elements
            element = re.match(r'[RCLQWF]', circuit_str).group()
    
            # Check if the element is Q or F to extract parameters accordingly
            if element in ['Q', 'F']:
                params = tuple(p[param_offset:param_offset + 2])
                param_offset += 2
            else:
                params = (p[param_offset],)
                param_offset += 1

            total_impedance = impedance(element, params, freqs)

        return total_impedance, param_offset

    return evaluate_circuit(circuit_str, 0)[0]


def Z_curve_fit(circuit_str):
    """
    Generate an impedance fitting function based on a given circuit string.
    
    Parameters:
    - circuit_string (str): String representation of the circuit.
    
    Returns:
    - function: Fitting function to be used with curve_fit.
    """
    
    def Zcomplex(f, *params):
        Z = compute_impedance(params, circuit_str, f)
        return np.hstack([Z.real, Z.imag])
    return Zcomplex


def Z_gen(circuit_string, return_type="concatenated"):
    """
    Generate an impedance fitting function based on a given circuit string.
    
    Parameters:
    - circuit_string (str): String representation of the circuit.
    - return_type (str): "concatenated" to return concatenated impedance, "complex" to return complex impedance.
    
    Returns:
    - function: Fitting function to be used with curve_fit.
    """
    
    def impedance_fit_concatenated(f, *params):
        Z = compute_impedance(params, circuit_string, f)
        return np.concatenate([Z.real, Z.imag])
    
    def impedance_fit_complex(f, *params):
        return compute_impedance(params, circuit_string, f)
    
    if return_type == "concatenated":
        return impedance_fit_concatenated
    elif return_type == "complex":
        return impedance_fit_complex
    else:
        raise ValueError("Invalid return_type. Choose either 'concatenated' or 'complex'.")


