"""
Data Extraction and Processing Functions for EIS (Electrochemical Impedance Spectroscopy) Data

This module provides functions to read and process EIS data from various file formats 
including NEISYS, ZAHNER_IM6, CSV, and dynamic EIS data. It includes utilities for:
- Reading EIS files from specified directories
- Parsing data from different spectrometer formats
- Trimming frequency ranges
- Splitting data arrays
- Processing both single and multiple EIS measurements

Author: Sheyi Clement Adediwura 

'Please cite the following paper if you use this code:
S. C. Adediwura, N. Mathew, J. Schmedt auf der Günne, J. Mater. Chem. A 12 (2024) 15847–15857.
https://doi.org/10.1039/d3ta06237f

"""

#External libraries
import numpy as np
import os
import glob



def get_eis_files(base_path='{}', subfolder='{}'):
    """
    Get EIS data files from specified location.
    
    Args:
        base_path (str): Base path to EIS data folder
        subfolder (str): Name of subfolder containing data files
            
    Returns:
        list: Sorted list of .txt file paths
    """
    # Set the file location path and get all .txt files
    file_location = os.path.join(base_path, subfolder, '*.txt')
    filenames = sorted(glob.glob(file_location))
    
    print('Number of files:', len(filenames))
    print('Files:')
    for i, file in enumerate(filenames, 1):
        print(f'{i}. {file}')
            
    return filenames


def parse_NEISYS_data(lines):
    """Helper function to parse NEISYS data from a list of lines."""
    
    # Find the line containing frequency header
    start_index = 0
    for i, line in enumerate(lines):
        if 'Freq. [Hz]' in line:  # Changed from 'Freq.[Hz]' to match exact header
            start_index = i + 1
            break

    exp_data = lines[start_index:]
    f, Z = [], []
    for line in exp_data:
        try:
            each = line.strip().split('\t')  # Split on tabs since data is tab-separated
            if len(each) >= 3:  # Ensure we have all three columns
                freq = float(each[0])
                real = float(each[1])
                imag = float(each[2])
                if freq != 0:
                    f.append(freq)
                    Z.append(complex(real, imag))
        except (ValueError, IndexError):
            continue  # Skip lines that can't be converted or are incomplete
            
    return np.array(f), np.array(Z)

def readNEISYS(filename):
    """Function to extract .txt file from NEISYS spectrometer."""
    try:
        with open(filename, 'r') as data:
            lines = data.readlines()
        return parse_NEISYS_data(lines)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except Exception as e:
        raise Exception(f"Error reading file {filename}: {str(e)}")

def readTXT(filename): 
    """ 
     function to extract .txt file without headers
     the file must have only 3 column data i.e f, reZ and imgZ
    """
    with open(filename, 'r') as data:
        lines = data.readlines()
    exp_data = lines[0:]#start data extraction from first line
    f, Z = [], []
    for line in exp_data:
        each = line.split() #lines splitted with tab
        if float(each[0]) != 0:
            f.append(float(each[0])) #assign first column to frequency 
            Z.append(complex(float(each[1]), float(each[2]))) #assign second & third colum to real and imag

    return np.array(f), np.array(Z)


def readCSV(filename): 
    """ 
    Function to extract .csv file without headers.
    The file must have only 3 column data i.e f, reZ and imgZ.
    """
    with open(filename, 'r') as data:
        lines = data.readlines()

    f, Z = [], []
    for line in lines:
        each = [item.strip() for item in line.split(',')] #lines splitted with comma
        if float(each[0]) != 0:
            f.append(float(each[0])) #assign first column to frequency 
            Z.append(complex(float(each[1]), float(each[2]))) #assign second & third column to real and imag

    return np.array(f), np.array(Z)


def full_readNEISYS(filenames):
    """
    Function to extract multiple .txt files from NEISYS spectrometer and store them in a single file.

    Input
    ----------
    filenames: list of strings
        List of filenames of .txt files to extract impedance data from

    Ouput
    -------
    combined_frequencies, combined_impedances : np.ndarray
        Arrays of frequencies and complex impedances
    """
    all_frequencies = []
    all_impedances = []
    
    for filename in filenames:
        with open(filename, 'r') as data:
            lines = data.readlines()
        f, Z = parse_NEISYS_data(lines)
        all_frequencies.extend(f)
        all_impedances.extend(Z)

    return np.array(all_frequencies), np.array(all_impedances)

    
def readDynamic(filename):
    """ 
    function to extract .txt file from dynamic EIS data from NEISYS spectrometer
    """
    with open(filename, 'r') as data:
        lines = data.readlines()
    exp_data = lines[3:] #start data extraction from line 4
    f, Z, tm, T = [], [], [], []
    for line in exp_data:
        each = line.split('\t') #lines splitted with tab
        if float(each[0]) != 0:
            f.append(float(each[0])) #assign first column to frequency 
            Z.append(complex(float(each[1]), float(each[2]))) #assign second & third colum to real and imag
            tm.append(float(each[3]))
            T.append(float(each[4]))

    return np.array(f), np.array(Z), np.array(tm), np.array(T)

def full_readTXT(filenames):
    """
    Function to extract multiple .txt files without headers and store them in a single file.

    Parameters
    ----------
    filenames: list of strings
        List of filenames of .txt files to extract impedance data from

    Returns
    -------
    combined_frequencies, combined_impedances : np.ndarray
        Arrays of frequencies and complex impedances from all files
    """
    all_frequencies = []
    all_impedances = []
    
    for filename in filenames:
        try:
            with open(filename, 'r') as data:
                lines = data.readlines()
            
            for line in lines:
                each = line.split()
                if float(each[0]) != 0:
                    all_frequencies.append(float(each[0]))
                    all_impedances.append(complex(float(each[1]), float(each[2])))
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

    return np.array(all_frequencies), np.array(all_impedances)


def full_readCSV(filenames):
    """
    Function to extract multiple .csv files without headers and store them in a single file.

    Parameters
    ----------
    filenames: list of strings
        List of filenames of .csv files to extract impedance data from

    Returns
    -------
    combined_frequencies, combined_impedances : np.ndarray
        Arrays of frequencies and complex impedances from all files
    """
    all_frequencies = []
    all_impedances = []
    
    for filename in filenames:
        try:
            with open(filename, 'r') as data:
                lines = data.readlines()
            
            for line in lines:
                each = [item.strip() for item in line.split(',')]
                if float(each[0]) != 0:
                    all_frequencies.append(float(each[0]))
                    all_impedances.append(complex(float(each[1]), float(each[2])))
        except FileNotFoundError:
            print(f"Warning: File {filename} not found, skipping...")
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

    return np.array(all_frequencies), np.array(all_impedances)


def trim_data(f, Z, fmin=None, fmax=None):
    """
    Trim frequency and impedance data based on specified minimum and maximum frequencies.
    
    Parameters:
    - f (1D array): A 1D numpy array containing frequency values.
    - Z (1D array): A 1D numpy array containing impedance values.
    - fmin (float, optional): Minimum frequency value for trimming.
    - fmax (float, optional): Maximum frequency value for trimming.
    
    Returns:
    - f_trimmed (1D array): Trimmed frequency values.
    - Z_trimmed (1D array): Corresponding trimmed impedance values.
    """
    
    # Initialize mask with all True values
    mask = np.ones_like(f, dtype=bool)
    
    # Update mask based on specified fmin and fmax
    if fmin is not None:
        mask &= (f >= fmin)
    if fmax is not None:
        mask &= (f <= fmax)
    f_trimmed = f[mask]
    Z_trimmed = Z[mask]
    
    return f_trimmed, Z_trimmed

def split_array(f, Z=None, split_freq=None):
    """
    Split the provided 2D data array into sublists based on specific end frequencies in the first column.
    
    Parameters:
    - f (1D array): A 1D numpy array containing frequency values.
    - Z (1D array, optional): A 1D numpy array corresponding to f.
    - split_freq (float, optional): Frequency at which to split. Default is maximum frequency in `f`.
    
    Returns:
    - List of 1D numpy arrays containing the split frequency data.
    - List of 1D numpy arrays containing the split Z data (empty if Z is None).
    """
    if split_freq is None:
        split_freq = np.max(f)

    sublists_f = []
    sublists_Z = []
    current_freqs = []
    current_Zs = []

    for i, freq in enumerate(f):
        if freq == split_freq and i != 0:
            sublists_f.append(np.array(current_freqs))
            if Z is not None:
                sublists_Z.append(np.array(current_Zs))
            current_freqs = []
            current_Zs = []

        current_freqs.append(freq)
        if Z is not None:
            current_Zs.append(Z[i])

    # Adding any remaining data to the sublists
    if current_freqs:
        sublists_f.append(np.array(current_freqs))
        if Z is not None and current_Zs:
            sublists_Z.append(np.array(current_Zs))

    return sublists_f, sublists_Z
