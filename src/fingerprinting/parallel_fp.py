# src/fingerprinting/parallel_fp.py
import logging
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import time

from .fp_calculator import process_line_for_fp

logger = logging.getLogger(__name__)

def calculate_fingerprints_for_file(
    input_filename: str,
    n_jobs: int = -1,
    delimiter: str = ' ',
    smiles_index: int = 0,
    name_index: int = 1,
    fp_type: str = 'morgan',
    radius: int = 2,
    n_bits: int = 256,
    use_features: bool = False,
    skip_header: bool = True,
    chunksize_multiplier: int = 4 # Heuristic for pool.imap chunksize
    ) -> tuple[list[np.ndarray], list[bytes], list[bytes]]:
    """
    Reads a SMILES/Name file and calculates fingerprints in parallel.

    Args:
        input_filename (str): Path to the input file.
        n_jobs (int): Number of CPU cores (-1 for all).
        delimiter (str): Column delimiter.
        smiles_index (int): Index of SMILES column.
        name_index (int): Index of Name column.
        fp_type (str): Fingerprint type ('morgan').
        radius (int): Morgan radius.
        n_bits (int): Morgan nBits.
        use_features (bool): Morgan useFeatures.
        skip_header (bool): Whether to skip the first line of the file.
        chunksize_multiplier (int): Multiplier for calculating Pool chunksize (larger can be faster for short tasks).

    Returns:
        tuple[list[np.ndarray], list[bytes], list[bytes]]:
            Lists of fingerprint arrays, smiles bytes, and name bytes for successfully processed lines.
            Returns empty lists if the file cannot be read or no lines succeed.
    """
    logger.info(f"Starting fingerprint generation for file: {input_filename}")

    lines_read = 0
    fp_list = []
    smiles_list = []
    name_list = []
    failed_lines = 0

    try:
        # --- Determine number of workers ---
        if n_jobs == -1:
            n_workers = cpu_count()
        else:
            n_workers = max(1, min(n_jobs, cpu_count()))
        logger.info(f"Using {n_workers} processes for parallel calculation.")

        # --- Prepare partial function for worker ---
        # Curry the fixed arguments for the worker function
        worker_func = partial(
            process_line_for_fp,
            delimiter=delimiter,
            smiles_index=smiles_index,
            name_index=name_index,
            fp_type=fp_type,
            radius=radius,
            n_bits=n_bits,
            use_features=use_features
        )

        # --- Read lines (consider memory for huge files) ---
        # Reading all lines at once might be bad for extremely large files.
        # If necessary, implement reading/processing in chunks, but Pool.imap handles iterators well.
        with open(input_filename, 'r', encoding='utf-8') as ifs:
            if skip_header:
                try:
                    next(ifs) # Skip the header line
                    logger.debug("Skipped header line.")
                except StopIteration:
                    logger.warning(f"File {input_filename} seems empty or has only a header.")
                    return [], [], [] # Return empty lists for empty file

            # Use Pool with imap_unordered for potential efficiency and progress bar
            with Pool(processes=n_workers) as pool:
                # Estimate total lines for TQDM (can be inaccurate/slow for huge files)
                # We might skip total count for very large files or do it less accurately
                # For now, let's assume we read lines or use a quick estimate if possible
                lines_for_processing = list(ifs) # Read remaining lines; USE WITH CAUTION ON MEMORY!
                lines_read = len(lines_for_processing) + (1 if skip_header else 0)
                logger.info(f"Read {lines_read} total lines (including header: {skip_header}). Processing {len(lines_for_processing)}.")

                if not lines_for_processing:
                     logger.warning(f"No data lines found in {input_filename} after header skip.")
                     return [], [], []

                # Calculate chunksize heuristic
                chunksize = max(1, len(lines_for_processing) // (n_workers * chunksize_multiplier))
                logger.debug(f"Using chunksize {chunksize} for imap_unordered.")

                # Process lines in parallel with progress bar
                results_iterator = pool.imap_unordered(worker_func, lines_for_processing, chunksize=chunksize)

                start_time = time.perf_counter()
                # Wrap iterator with tqdm for progress
                with tqdm(total=len(lines_for_processing), desc=f"Processing {os.path.basename(input_filename)}", unit="mol") as pbar:
                    for result in results_iterator:
                        if result is not None:
                            fp_array, smiles_bytes, name_bytes = result
                            fp_list.append(fp_array)
                            smiles_list.append(smiles_bytes)
                            name_list.append(name_bytes)
                        else:
                            failed_lines += 1
                        pbar.update(1) # Update progress bar for each result processed

                end_time = time.perf_counter()
                logger.info(f"Parallel processing step finished in {end_time - start_time:.4f} seconds.")

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_filename}")
        return [], [], []
    except Exception as e:
        logger.error(f"An error occurred during fingerprint generation for {input_filename}: {e}", exc_info=True)
        return [], [], [] # Return empty lists on error

    # --- Log Summary ---
    successful_count = len(fp_list)
    logger.info(f"Finished processing {input_filename}. Successfully generated {successful_count} fingerprints.")
    if failed_lines > 0:
        logger.warning(f"{failed_lines} lines failed parsing or fingerprint calculation.")

    return fp_list, smiles_list, name_list
