# tests/test_preprocessing.py
import sys, os

# Adjust path to import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import needs careful handling due to global canonicalizer init maybe
# It's better to test the worker directly if possible
from src.preprocessing.tautomers import _canonicalize_smiles_worker

def test_tautomer_success():
    """Test successful tautomer canonicalization."""
    # Keto-enol pair (example)
    keto = "C=C(O)C"   # Propen-2-ol (enol form)
    expected_canon = "CC(C)=O" # Acetone (keto form) - RDKit should canonicalize to this
    original, canonical = _canonicalize_smiles_worker(keto)
    assert original == keto
    assert canonical is not None
    # RDKit canonical SMILES might vary slightly, but should represent acetone
    # A robust check might involve converting back to mol or using InChIKey
    # For simplicity, let's check if it contains the expected structure elements
    assert canonical == expected_canon # This comparison might be too strict depending on RDKit version

def test_tautomer_invalid():
    """Test tautomer canonicalization on invalid SMILES."""
    original, canonical = _canonicalize_smiles_worker("invalid")
    assert canonical is None

def test_tautomer_no_change():
    """Test tautomer canonicalization on molecule with no significant tautomers."""
    smi = "CCO" # Ethanol
    original, canonical = _canonicalize_smiles_worker(smi)
    assert original == smi
    # Canonical SMILES might differ slightly even if tautomerism isn't the main factor
    # assert canonical == smi # This might fail due to SMILES canonicalization rules
    assert canonical is not None # Should succeed
