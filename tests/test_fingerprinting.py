# tests/test_fingerprinting.py
import numpy as np
import sys, os

# Adjust path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.fingerprinting.fp_calculator import calculate_morgan_fingerprint

def test_morgan_calc_success():
    """Test successful Morgan fingerprint calculation."""
    smi = "CCO" # Ethanol
    radius = 2
    nbits = 256
    fp = calculate_morgan_fingerprint(smi, radius, nbits)
    assert fp is not None
    assert isinstance(fp, np.ndarray)
    assert fp.shape == (nbits,)
    assert fp.dtype == np.int8
    # Check if it contains 0s and 1s (might not always have 1s for simple mol/small bits)
    assert np.all((fp == 0) | (fp == 1))
    # Could check specific bits if known, but length/type is a good start
    assert fp.sum() > 0 # Ethanol should set some bits

def test_morgan_calc_invalid_smiles():
    """Test Morgan fingerprint calculation with invalid SMILES."""
    smi = "invalid-smiles"
    radius = 2
    nbits = 256
    fp = calculate_morgan_fingerprint(smi, radius, nbits)
    assert fp is None

def test_morgan_calc_empty_smiles():
    """Test Morgan fingerprint calculation with empty SMILES."""
    smi = ""
    radius = 2
    nbits = 256
    fp = calculate_morgan_fingerprint(smi, radius, nbits)
    assert fp is None
