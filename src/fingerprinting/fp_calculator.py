# src/fingerprinting/fp_calculator.py
import logging
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import RDLogger # To manage RDKit logging

logger = logging.getLogger(__name__)
rdk_logger = RDLogger.logger()

# Cache RDKit functions locally for minor performance improvement in loops/map
_MolFromSmiles = Chem.MolFromSmiles
_GetMorganFingerprintAsBitVect = rdmd.GetMorganFingerprintAsBitVect
_ConvertToNumpyArray = DataStructs.ConvertToNumpyArray

def calculate_morgan_fingerprint(smiles: str, radius: int, n_bits: int, use_features: bool = False) -> np.ndarray | None:
    """
    Calculates a Morgan fingerprint for a single SMILES string.

    Args:
        smiles (str): Input SMILES string.
        radius (int): Morgan fingerprint radius.
        n_bits (int): Number of bits in the fingerprint vector.
        use_features (bool): Whether to use feature invariants (FCFP). Defaults to False (ECFP).

    Returns:
        np.ndarray | None: A NumPy array (np.int8) of the fingerprint, or None if calculation fails.
    """
    if not isinstance(smiles, str) or not smiles:
        return None

    # Suppress RDKit errors/warnings during calculation if desired
    # rdk_logger.setLevel(RDLogger.CRITICAL)

    mol = _MolFromSmiles(smiles)
    if mol is None:
        # rdk_logger.setLevel(RDLogger.ERROR) # Restore level
        return None

    try:
        fp = _GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useFeatures=use_features)
        # Convert to NumPy array of specified type
        # Initialize with zeros, dtype int8 seems appropriate for bit vectors (0 or 1)
        fp_array = np.zeros((n_bits,), dtype=np.int8)
        _ConvertToNumpyArray(fp, fp_array)
        # rdk_logger.setLevel(RDLogger.ERROR) # Restore level
        return fp_array
    except Exception as e:
        # Log specific calculation errors if they occur, although less common for Morgan
        logger.debug(f"Fingerprint calculation failed for SMILES '{smiles}': {e}")
        # rdk_logger.setLevel(RDLogger.ERROR) # Restore level
        return None


# --- Processing function combining parsing and calculation ---
def process_line_for_fp(
    line: str,
    delimiter: str = ' ',
    smiles_index: int = 0,
    name_index: int = 1,
    fp_type: str = 'morgan',
    radius: int = 2,
    n_bits: int = 256,
    use_features: bool = False
    ) -> tuple[np.ndarray | None, bytes | None, bytes | None] | None:
    """
    Parses a line containing SMILES and Name, calculates a fingerprint.

    Args:
        line (str): Input line from file.
        delimiter (str): Delimiter separating columns.
        smiles_index (int): Index of the SMILES column (0-based).
        name_index (int): Index of the Name column (0-based).
        fp_type (str): Type of fingerprint ('morgan'). Add more types later.
        radius (int): Morgan radius.
        n_bits (int): Morgan nBits.
        use_features (bool): Morgan useFeatures.

    Returns:
        tuple[np.ndarray | None, bytes | None, bytes | None] | None:
            Returns (fingerprint_array, smiles_bytes, name_bytes) on success,
            or None if parsing or calculation fails.
            Fingerprint is None within the tuple if only parsing succeeded but FP failed.
    """
    try:
        parts = line.strip().split(delimiter)
        if len(parts) > max(smiles_index, name_index):
            smiles = parts[smiles_index]
            name = parts[name_index]
            # Basic validation
            if not smiles or not name:
                # logger.debug(f"Empty SMILES or Name in line: {line.strip()}")
                return None
        else:
            # logger.debug(f"Could not parse line with delimiter '{delimiter}': {line.strip()}")
            return None
    except Exception as e:
        # logger.debug(f"Error parsing line '{line.strip()}': {e}")
        return None

    # Calculate fingerprint based on type
    fp_array = None
    if fp_type.lower() == 'morgan':
        fp_array = calculate_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits, use_features=use_features)
    # Add elif blocks here for other fingerprint types in the future
    # elif fp_type.lower() == 'rdkit':
    #    fp_array = calculate_rdkit_fingerprint(...)
    else:
        logger.warning(f"Unsupported fingerprint type requested: {fp_type}")
        # Decide if this is a line failure (return None) or just FP failure (return fp_array=None)
        # Let's return None for now as the user requested this specific type
        return None

    # If fingerprint calculation failed, we might still want to keep track
    # but original code filtered these out. We follow that for now.
    if fp_array is None:
        return None # Indicate failure for this line

    try:
        # Encode strings to bytes using utf-8 (more standard than ascii ignore)
        smiles_bytes = smiles.encode('utf-8')
        name_bytes = name.encode('utf-8')
        return fp_array, smiles_bytes, name_bytes
    except Exception as e:
        # logger.debug(f"Error encoding SMILES/Name for line '{line.strip()}': {e}")
        return None # Treat encoding error as failure
