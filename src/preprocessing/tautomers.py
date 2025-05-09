# src/preprocessing/tautomers.py
import logging
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize
from multiprocessing import Pool, cpu_count

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Initialize Tautomer Canonicalizer ---
# Suppress RDKit logs during initialization if desired
rdk_logger = RDLogger.logger()
rdk_logger.setLevel(RDLogger.CRITICAL) # Options: INFO, WARNING, ERROR, CRITICAL

lta = None
try:
    lta = MolStandardize.tautomer.TautomerCanonicalizer()
    logger.debug("MolStandardize.tautomer.TautomerCanonicalizer initialized successfully.")
    rdk_logger.setLevel(RDLogger.ERROR) # Restore RDKit logging level after init
except NameError:
    logger.error("MolStandardize.tautomer.TautomerCanonicalizer not found. Ensure RDKit is installed correctly.")
    rdk_logger.setLevel(RDLogger.ERROR) # Restore log level even on failure
except Exception as e:
    logger.error(f"Error initializing TautomerCanonicalizer: {e}")
    rdk_logger.setLevel(RDLogger.ERROR) # Restore log level

# --- Worker Function ---
def _canonicalize_smiles_worker(smiles: str) -> tuple[str | None, str | None]:
    """
    Worker function to canonicalize tautomer for a single SMILES string.

    Returns:
        tuple[str | None, str | None]: (original_smiles, canonical_smiles or None if failed)
    """
    if lta is None:
        # Logged once per process ideally, but might log per call if init failed badly
        # logger.error("TautomerCanonicalizer not available in worker.")
        return smiles, None
    if not isinstance(smiles, str) or not smiles:
        # logger.debug(f"Input is not a valid string, skipping: '{smiles}'")
        return smiles, None

    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            # Canonicalize tautomer
            can_mol = lta.canonicalize(mol)
            # Generate canonical SMILES, keeping stereochemistry info
            can_smiles = Chem.MolToSmiles(can_mol, isomericSmiles=True)
            return smiles, can_smiles
        except Exception as e:
            # logger.warning(f"Canonicalization failed for SMILES '{smiles}': {e}")
            # Log fewer warnings by checking this in the main process
            return smiles, None # Return original smiles and None for canonical
    else:
        # logger.debug(f"Could not parse SMILES, skipping: '{smiles}'")
        return smiles, None

# --- Main Processing Function ---
def process_smiles_file(input_file: str, output_file: str, smiles_col: str = 'smiles', id_col: str = 'zinc_id', delimiter: str = ' ', n_jobs: int = -1):
    """
    Reads a SMILES file, canonicalizes tautomers in parallel, removes duplicates based
    on the canonical form, and writes the unique canonical SMILES and IDs to a new file.

    Args:
        input_file (str): Path to the input SMILES file.
        output_file (str): Path to the output SMILES file.
        smiles_col (str): Name of the column containing SMILES strings. Defaults to 'smiles'.
        id_col (str): Name of the column containing molecule IDs. Defaults to 'zinc_id'.
        delimiter (str): Delimiter used in the input and output files. Defaults to ' '.
        n_jobs (int): Number of CPU cores to use (-1 for all available). Defaults to -1.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    if lta is None:
        logger.error("Tautomer processing cannot proceed: TautomerCanonicalizer failed to initialize.")
        return False

    logger.info(f"Starting tautomer processing for: {input_file}")
    try:
        # Use low_memory=False for potentially better parsing of mixed types
        df = pd.read_csv(input_file, sep=delimiter, low_memory=False)
        if smiles_col not in df.columns or id_col not in df.columns:
            logger.error(f"Required columns '{smiles_col}' or '{id_col}' not found in {input_file}. Available: {df.columns.tolist()}")
            return False
        # Ensure smiles column is string type
        df[smiles_col] = df[smiles_col].astype(str)

    except Exception as e:
        logger.error(f"Failed to read or validate input file {input_file}: {e}")
        return False

    if n_jobs == -1:
        n_cpus = cpu_count()
    else:
        n_cpus = max(1, min(n_jobs, cpu_count())) # Ensure at least 1, max available
    logger.info(f"Using {n_cpus} cores for parallel canonicalization.")

    original_smiles = df[smiles_col].tolist()
    num_mols = len(original_smiles)
    logger.info(f"Read {num_mols} molecules from input file.")

    processed_results = []
    parse_failures = 0
    canon_failures = 0

    try:
        # Use multiprocessing Pool with map for parallel execution
        # map preserves order which simplifies re-integration with DataFrame
        with Pool(processes=n_cpus) as pool:
            # Process chunks to potentially reduce memory overhead for very large files
            chunksize = max(1, num_mols // (n_cpus * 4)) # Heuristic chunk size
            logger.debug(f"Using chunksize: {chunksize}")
            # Using map directly on the list
            processed_results = pool.map(_canonicalize_smiles_worker, original_smiles, chunksize=chunksize)

        # Analyze results
        canonical_smiles_map = {} # Store original index -> canonical smiles
        temp_canonical_list = []
        for i, (original_smi, canonical_smi) in enumerate(processed_results):
            if canonical_smi is None:
                # Could distinguish between parse failure (original was bad) vs canon failure
                if Chem.MolFromSmiles(original_smi) is None:
                    parse_failures += 1
                else:
                    canon_failures += 1
                temp_canonical_list.append(None) # Keep placeholder for length matching if needed, or skip
            else:
                temp_canonical_list.append(canonical_smi)
                canonical_smiles_map[i] = canonical_smi # Store successful results

        num_processed_successfully = len(canonical_smiles_map)
        logger.info(f"Successfully canonicalized {num_processed_successfully} molecules.")
        if parse_failures > 0:
            logger.warning(f"Could not parse SMILES for {parse_failures} input molecules.")
        if canon_failures > 0:
            logger.warning(f"Canonicalization failed for {canon_failures} valid input molecules.")

        if num_processed_successfully == 0:
             logger.warning(f"No molecules could be successfully processed and canonicalized in {input_file}.")
             # Still write empty file maybe? Or just return False.
             # Let's write empty file for consistency.
             pd.DataFrame(columns=[smiles_col, id_col]).to_csv(output_file, sep=delimiter, index=False)
             logger.info(f"Empty output file written: {output_file}")
             return True # Indicate process completed, even if no valid output

        # Add canonical smiles as a new column to the original DataFrame using the map
        df['canonical_smiles'] = df.index.map(canonical_smiles_map)

        # Filter out rows where canonicalization failed
        df_processed = df.dropna(subset=['canonical_smiles']).copy()

        # Drop duplicates based on the 'canonical_smiles' column
        original_valid_count = len(df_processed)
        df_processed.drop_duplicates(subset=['canonical_smiles'], keep='first', inplace=True)
        deduplicated_count = len(df_processed)
        logger.info(f"Removed {original_valid_count - deduplicated_count} duplicate tautomers from successfully processed molecules.")

        # --- Prepare and Write Output ---
        # Select the canonical smiles and the original ID column
        df_output = df_processed[['canonical_smiles', id_col]]
        # Rename 'canonical_smiles' back to the original smiles column name for output
        df_output.rename(columns={'canonical_smiles': smiles_col}, inplace=True)

        # Write the output file
        df_output.to_csv(output_file, sep=delimiter, index=False)
        logger.info(f"Saved {deduplicated_count} unique canonical tautomers to: {output_file}")
        return True

    except Exception as e:
        logger.error(f"An error occurred during parallel processing or writing for {input_file}: {e}", exc_info=True)
        return False
