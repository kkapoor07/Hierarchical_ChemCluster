# scripts/generate_fingerprints.py
import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from glob import glob
import time
import numpy as np
from tqdm import tqdm # Ensure tqdm is imported

# --- Setup sys.path ---
# Add the src directory to the Python path
# Assumes the script is run from the project root or scripts/ folder
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # Import necessary functions from refactored modules
    from src.common.logging_utils import setup_logging
    from src.common.timing_utils import timing
    from src.fingerprinting.parallel_fp import calculate_fingerprints_for_file
    from src.io.writers import save_fingerprints_h5
    from src.io.hdf5_utils import aggregate_hdf5_datasets
except ImportError as e:
    print(f"Error importing project modules: {e}. Check script location or PYTHONPATH.")
    # Decide if aggregation is critical; if so, exit
    if "aggregate_hdf5_datasets" in str(e):
        print("Aggregation function import failed. Aggregation step will not be available.")
        aggregate_hdf5_datasets = None # Mark as unavailable
    else:
        # Allow running without aggregation if other imports work? Or just exit?
        sys.exit(1) # Exit if core modules fail

# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def find_files(input_dir: str, pattern: str = '*_taut.smi', exclude_pattern: str | None = None) -> list[str]:
    """
    Finds files matching a glob pattern recursively, with optional exclusion.

    Args:
        input_dir (str): Root directory to search within.
        pattern (str): Glob pattern for files to include (e.g., '*.smi', '*.csv').
        exclude_pattern (str | None): Substring to exclude from filenames. If None, no exclusion.

    Returns:
        list[str]: A list of matching file paths.
    """
    if not os.path.isdir(input_dir):
        logger.warning(f"Input directory not found: {input_dir}")
        return []

    # Use os.path.join for robust path construction
    search_pattern = os.path.join(input_dir, '**', pattern)
    try:
        # Set recursive=True to search subdirectories
        files = glob(search_pattern, recursive=True)
        logger.info(f"Found {len(files)} files matching '{pattern}' in '{input_dir}' recursively (before exclusion).")

        if exclude_pattern:
            original_count = len(files)
            # Ensure exclude_pattern is treated correctly if empty string
            if exclude_pattern:
                 files = [f for f in files if exclude_pattern not in os.path.basename(f)]
                 excluded_count = original_count - len(files)
                 if excluded_count > 0:
                     logger.info(f"Excluded {excluded_count} files containing '{exclude_pattern}'.")
            else:
                 logger.info("No exclusion pattern specified.")


        logger.info(f"Found {len(files)} files to process.")
        return files
    except Exception as e:
        logger.error(f"Error during file search in '{input_dir}' with pattern '{pattern}': {e}")
        return []


def process_and_save_fp_wrapper(
    input_file: str,
    output_dir: str,
    h5_filename_pattern: str = "{basename}.h5", # Allows flexible output naming
    fp_args: dict # Dictionary containing args for calculate_fingerprints_for_file
    ) -> tuple[str, bool, int, str]: # Return output path as well
    """
    Wrapper for ProcessPoolExecutor: calls fingerprint calculation and saves the result.

    Args:
        input_file (str): Path to the input SMILES file.
        output_dir (str): Directory to save the output HDF5 file.
        h5_filename_pattern (str): F-string pattern for output filename, e.g., "{basename}.h5".
        fp_args (dict): Dictionary of arguments for calculate_fingerprints_for_file
                        (excluding input_filename).

    Returns:
        tuple[str, bool, int, str]: (input_filename, success_status, number_of_fingerprints_saved, output_filepath)
    """
    output_filepath = "" # Initialize in case of early exit
    try:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        try:
             # Format output filename using the base name
            output_filename = h5_filename_pattern.format(basename=base_name)
        except KeyError:
            logger.warning(f"Invalid h5_filename_pattern: {h5_filename_pattern}. Using default.")
            output_filename = f"{base_name}.h5" # Fallback

        output_filepath = os.path.join(output_dir, output_filename)
        logger.debug(f"Dispatching task: {input_file} -> {output_filepath}")

        # --- Call the parallel fingerprint calculation ---
        # Pass internal n_jobs argument correctly
        internal_n_jobs = fp_args.get('n_jobs', 1) # Default to 1 if controlled externally
        fp_calc_args = fp_args.copy() # Avoid modifying original dict
        fp_calc_args['n_jobs'] = internal_n_jobs

        fp_list, smiles_list, name_list = calculate_fingerprints_for_file(
            input_filename=input_file,
            **fp_calc_args # Pass all other FP calculation arguments
        )

        num_fps = len(fp_list)
        if num_fps == 0:
            logger.warning(f"No fingerprints generated for {input_file}. Skipping save.")
            # Return True indicating task finished, but 0 FPs saved.
            return input_file, True, 0, "" # Return empty path on skipped save

        # --- Save the results ---
        # Convert fp_list (list of arrays) to a single numpy array
        fp_dtype_np = fp_args.get('fp_dtype', np.int8)
        fp_array = np.array(fp_list, dtype=fp_dtype_np)

        save_success = save_fingerprints_h5(
            output_filename=output_filepath,
            fp_array=fp_array,
            smiles_list=smiles_list,
            name_list=name_list,
            fp_dtype=fp_dtype_np, # Pass numpy dtype object
            compression=fp_args.get('compression'),
            compression_opts=fp_args.get('compression_opts')
        )

        return input_file, save_success, num_fps, output_filepath if save_success else ""

    except Exception as e:
        logger.error(f"Exception in process_and_save_fp_wrapper for {input_file}: {e}", exc_info=True)
        return input_file, False, 0, "" # Return empty path on error

# --- Main Execution Block ---
@timing # Keep timing decorator if desired
def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Generate fingerprints in parallel for SMILES files and optionally aggregate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add config file argument FIRST
    parser.add_argument("-c", "--config", help="Path to YAML configuration file.")

    # Group for Input/Output settings
    io_group = parser.add_argument_group('Input/Output Settings')
    io_group.add_argument("--input_dir", help="Root directory containing input SMILES files (e.g., *_taut.smi).")
    io_group.add_argument("--output_dir", help="Directory to save the output HDF5 fingerprint files (individual files per input).")
    io_group.add_argument("--pattern", default="*_taut.smi", help="Glob pattern to find input SMILES files (searches recursively).")
    io_group.add_argument("--exclude", default=None, help="Substring in filename to exclude (e.g., '_bad'). Set to '' for no exclusion.")
    io_group.add_argument("--output_pattern", default="{basename}.h5", help="Pattern for individual output HDF5 filenames. Use {basename} for input filename without extension.")

    # Group for File Parsing settings
    parsing_group = parser.add_argument_group('File Parsing Settings')
    parsing_group.add_argument("--delimiter", default=" ", help="Delimiter in input SMILES files.")
    parsing_group.add_argument("--smiles_col_idx", type=int, default=0, help="0-based index of the SMILES column.")
    parsing_group.add_argument("--name_col_idx", type=int, default=1, help="0-based index of the Name/ID column.")
    parsing_group.add_argument("--skip_header", action='store_true', default=None, help="Skip the first line of input files (CLI overrides config).")

    # Group for Fingerprint settings
    fp_group = parser.add_argument_group('Fingerprint Settings')
    fp_group.add_argument("--fp_type", default="morgan", choices=['morgan'], help="Type of fingerprint to generate.")
    fp_group.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    fp_group.add_argument("--nbits", type=int, default=256, help="Number of bits for Morgan fingerprint.")
    fp_group.add_argument("--use_features", action='store_true', default=None, help="Use features (FCFP) instead of connectivity (ECFP) for Morgan.")
    fp_group.add_argument("--fp_dtype", default="int8", choices=['int8', 'int16', 'int32', 'int64'], help="Numpy dtype for storing fingerprints (use int8 for binary).")

    # Group for Parallelism settings
    parallel_group = parser.add_argument_group('Parallelism Settings')
    parallel_group.add_argument("--n_workers", type=int, default=-1, help="Number of worker processes for FILE-level parallelism (-1 for all cores, 1 for serial).")

    # Group for HDF5 Output settings
    hdf5_group = parser.add_argument_group('HDF5 Output Settings')
    hdf5_group.add_argument("--compression", default="gzip", choices=['gzip', 'lzf', 'none'], help="Compression for HDF5 output ('none' for no compression).")
    hdf5_group.add_argument("--compress_level", type=int, default=4, help="Compression level (1-9 for gzip).")

    # Group for Aggregation settings
    agg_group = parser.add_argument_group('Aggregation Settings')
    agg_group.add_argument("--aggregate", action='store_true', default=None, help="If set, aggregate individual HDF5 files after generation.")
    agg_group.add_argument("--aggregated_output_file", default="fingerprints_aggregated.h5", help="Filename for the aggregated HDF5 file (saved relative to output_dir).")
    agg_group.add_argument("--aggregate_datasets", nargs='+', default=['fp_list', 'smiles_list', 'name_list'], help="List of datasets to include in aggregation.")
    agg_group.add_argument("--aggregate_pattern", default="*.h5", help="Glob pattern of files to aggregate within the output directory (default: *.h5).")

    # Group for Logging settings
    log_group = parser.add_argument_group('Logging Settings')
    log_group.add_argument("--log_file", default="logs/generate_fingerprints.log", help="Path to log file.")
    log_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    # --- Initial Parse & Config Loading ---
    args_initial, _ = parser.parse_known_args()
    args = load_config_and_reparse(parser, args_initial) # Load config and re-parse CLI

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file # Use final path from effective args
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        try: os.makedirs(log_dir, exist_ok=True)
        except OSError as e: print(f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr)
    setup_logging(log_file, level=log_level)

    # --- Final Argument Validation ---
    required_args_list = ['input_dir', 'output_dir']
    missing_args_list = [arg for arg in required_args_list if getattr(args, arg, None) is None]
    if missing_args_list:
        parser.error(f"Missing required arguments: {', '.join(missing_args_list)}")

    if not os.path.isdir(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {args.output_dir}")
    except OSError as e:
        parser.error(f"Could not create output directory {args.output_dir}: {e}")

    # Use final `args` values for the rest of the script
    # Get defaults here if needed, although set_defaults in load_config_and_reparse should handle it
    pattern = args.pattern
    exclude_pattern = args.exclude if args.exclude else None
    output_pattern = args.output_pattern
    delimiter = args.delimiter
    smiles_col_idx = args.smiles_col_idx
    name_col_idx = args.name_col_idx
    skip_header = args.skip_header
    fp_type = args.fp_type
    radius = args.radius
    nbits = args.nbits
    use_features = args.use_features
    fp_dtype_str = args.fp_dtype
    n_workers = args.n_workers
    compression = args.compression
    compress_level = args.compress_level
    aggregate = args.aggregate
    aggregated_output_file = args.aggregated_output_file
    aggregate_datasets = args.aggregate_datasets
    aggregate_pattern = args.aggregate_pattern

    # --- Find Files ---
    input_files = find_files(args.input_dir, pattern, exclude_pattern=exclude_pattern)
    # ... (rest of the main function remains unchanged from the previous version) ...
    # --- Determine Parallelism Strategy ---
    if n_workers == -1: max_workers = cpu_count()
    else: max_workers = max(1, n_workers)
    internal_n_jobs = 1 if max_workers > 1 else -1
    logger.info(f"Processing {len(input_files)} files using up to {max_workers} worker processes.")
    if max_workers == 1: logger.info(f"Internal calculation per file will use all available cores.")
    else: logger.info(f"Internal calculation per file will use 1 core.")

    # --- Prepare Arguments for Fingerprint Generation ---
    fp_args_dict = {
        "n_jobs": internal_n_jobs, "delimiter": delimiter, "smiles_index": smiles_col_idx,
        "name_index": name_col_idx, "fp_type": fp_type, "radius": radius,
        "n_bits": nbits, "use_features": use_features, "skip_header": skip_header,
        "fp_dtype": np.dtype(fp_dtype_str),
        "compression": compression if compression != 'none' else None,
        "compression_opts": compress_level,
    }
    tasks = [(f, args.output_dir, output_pattern, fp_args_dict) for f in input_files]

    # --- Execute Fingerprint Generation Tasks ---
    logger.info("--- Starting Fingerprint Generation ---")
    start_time_fp = time.perf_counter()
    success_count, fail_count, total_fps_saved = 0, 0, 0
    tasks_submitted = 0
    generated_output_files = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = { executor.submit(process_and_save_fp_wrapper, *task_args): task_args for task_args in tasks }
        tasks_submitted = len(future_to_task)
        logger.info(f"Submitted {tasks_submitted} fingerprint generation tasks.")

        with tqdm(total=tasks_submitted, desc="Generating Fingerprints", unit="file") as pbar:
            for future in as_completed(future_to_task):
                input_file, _, _, _ = future_to_task[future] # Get input file from original task args
                try:
                    _, success, num_saved, output_filepath = future.result()
                    if success:
                        success_count += 1; total_fps_saved += num_saved
                        if output_filepath: generated_output_files.append(output_filepath)
                    else: fail_count += 1
                except Exception as exc:
                    fail_count += 1
                    logger.error(f"Task for {input_file} generated exception: {exc}", exc_info=True)
                pbar.update(1)

    end_time_fp = time.perf_counter()
    logger.info(f"--- Fingerprint Generation Finished in {end_time_fp - start_time_fp:.2f} seconds ---")
    logger.info(f"Successfully processed/saved: {success_count} files.")
    logger.info(f"Total fingerprints saved in individual files: {total_fps_saved}.")
    logger.info(f"Failed tasks: {fail_count} files.")

    if fail_count > 0: logger.warning("Some generation tasks failed. Check logs.")

    # --- Optional Aggregation Step ---
    if aggregate:
        if aggregate_hdf5_datasets is None:
             logger.error("Aggregation function not available (import failed). Skipping.")
        elif not generated_output_files:
            logger.warning("No individual files to aggregate. Skipping.")
        else:
            logger.info("--- Starting Fingerprint Aggregation ---")
            aggregate_output_filepath = os.path.join(args.output_dir, aggregated_output_file)
            logger.info(f"Aggregating HDF5 files from {args.output_dir} matching '{aggregate_pattern}' into {aggregate_output_filepath}")
            logger.info(f"Datasets: {aggregate_datasets}")

            agg_success = aggregate_hdf5_datasets(
                input_dir=args.output_dir, output_file=aggregate_output_filepath,
                file_pattern=aggregate_pattern, recursive=False, # Assume flat output dir
                datasets_to_aggregate=aggregate_datasets,
                compression=compression if compression != 'none' else None,
                compression_opts=compress_level
            )
            if not agg_success:
                logger.error("Fingerprint aggregation failed.")
                sys.exit(1)
            logger.info("--- Fingerprint Aggregation Finished ---")
    else:
        logger.info("--- Skipping Fingerprint Aggregation ---")

    if fail_count > 0 :
         logger.error("Exiting with error code due to failures in fingerprint generation.")
         sys.exit(1)

if __name__ == "__main__":
    main()
