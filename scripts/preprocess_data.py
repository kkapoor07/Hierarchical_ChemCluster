# scripts/preprocess_data.py
import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from glob import glob
import time # For simple overall timing

# --- Setup sys.path ---
# Add the src directory to the Python path
# This assumes the script is run from the project root or scripts/ folder
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.common.logging_utils import setup_logging
    from src.common.timing_utils import timing
    from src.preprocessing.tautomers import process_smiles_file
except ImportError as e:
    print(f"Error importing project modules: {e}. Ensure script is run from the correct directory or PYTHONPATH is set.")
    sys.exit(1)

# --- Get Logger ---
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def find_files(input_dir: str, pattern: str = '*.smi', exclude_pattern: str | None = "_taut") -> list[str]:
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

    search_pattern = os.path.join(input_dir, '**', pattern)
    try:
        files = glob(search_pattern, recursive=True)
        logger.info(f"Found {len(files)} files matching '{pattern}' in '{input_dir}' (before exclusion).")

        if exclude_pattern:
            original_count = len(files)
            files = [f for f in files if exclude_pattern not in os.path.basename(f)]
            excluded_count = original_count - len(files)
            if excluded_count > 0:
                logger.info(f"Excluded {excluded_count} files containing '{exclude_pattern}'.")

        logger.info(f"Found {len(files)} files to process.")
        return files
    except Exception as e:
        logger.error(f"Error during file search in '{input_dir}' with pattern '{pattern}': {e}")
        return []


def process_file_wrapper(input_file: str, output_dir: str, smiles_col: str, id_col: str, delimiter: str) -> tuple[str, bool]:
    """
    Wrapper function for use with concurrent.futures. Handles calling the core
    processing function and managing output paths.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Base directory to save the output file.
        smiles_col (str): Name of the SMILES column.
        id_col (str): Name of the ID column.
        delimiter (str): File delimiter.

    Returns:
        tuple[str, bool]: The input filename and a boolean indicating success.
    """
    # Define output filename based on input filename
    base_name = os.path.basename(input_file)
    output_name = os.path.splitext(base_name)[0] + "_taut.smi" # Standardized output name

    # Place output file directly in the output directory (simpler)
    # If preserving structure is needed, uncomment and adapt the relative path logic
    output_file = os.path.join(output_dir, output_name)

    # Optional: Preserve directory structure
    # try:
    #     # Assumes input_dir is accessible here or passed somehow if needed
    #     # This example assumes output_dir is the base for the mirrored structure
    #     relative_path = os.path.relpath(os.path.dirname(input_file), start=global_input_dir) # Requires global_input_dir
    #     output_subdir = os.path.join(output_dir, relative_path)
    #     os.makedirs(output_subdir, exist_ok=True)
    #     output_file = os.path.join(output_subdir, output_name)
    # except NameError: # Fallback if global_input_dir is not defined
    #     output_file = os.path.join(output_dir, output_name)
    # except ValueError: # Handle case where input files might be outside input_dir (unlikely)
    #     output_file = os.path.join(output_dir, output_name)


    logger.debug(f"Dispatching task: {input_file} -> {output_file}")
    try:
        # We run the file processing itself serially within this worker,
        # as the parallelism happens *inside* process_smiles_file (n_jobs > 1)
        # OR the parallelism happens *across files* (n_workers > 1 in main).
        # Avoid nested parallelism unless carefully managed.
        # Let's assume outer parallelism across files: set internal n_jobs=1
        success = process_smiles_file(input_file, output_file, smiles_col, id_col, delimiter, n_jobs=1)
        if not success:
             logger.warning(f"Processing function returned False for: {input_file}")
        return input_file, success
    except Exception as e:
        # Log exceptions occurring within the wrapper/process itself
        logger.error(f"Exception in process_file_wrapper for {input_file}: {e}", exc_info=True)
        return input_file, False


# --- Main Execution Block ---
@timing # Keep timing decorator if desired
def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Preprocess SMILES files in parallel: Canonicalize tautomers and remove duplicates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add config file argument FIRST
    parser.add_argument("-c", "--config", help="Path to YAML configuration file.")

    # Group for Input/Output paths
    path_group = parser.add_argument_group('Input/Output Paths')
    path_group.add_argument("--input_dir", help="Root directory containing input SMILES files.")
    path_group.add_argument("--output_dir", help="Directory to save processed '_taut.smi' files.")
    path_group.add_argument("--pattern", default="*.smi", help="Glob pattern to find input SMILES files (searches recursively).")
    path_group.add_argument("--exclude", default="_taut", help="Substring in filename to exclude (e.g., prevent reprocessing '_taut.smi' files). Set to '' for no exclusion.")

    # Group for File Parsing settings
    parsing_group = parser.add_argument_group('File Parsing Settings')
    parsing_group.add_argument("--smiles_col", default="smiles", help="Name of the SMILES column.")
    parsing_group.add_argument("--id_col", default="zinc_id", help="Name of the molecule ID column.")
    parsing_group.add_argument("--delimiter", default=" ", help="Delimiter for input/output files (e.g., ' ' or ',').")

    # Group for Parallelism settings
    parallel_group = parser.add_argument_group('Parallelism Settings')
    parallel_group.add_argument("--n_workers", type=int, default=-1, help="Number of worker processes for file-level parallelism (-1 for all cores, 1 for serial).")

    # Group for Logging settings
    log_group = parser.add_argument_group('Logging Settings')
    log_group.add_argument("--log_file", default="logs/preprocess_data.log", help="Path to the log file.")
    log_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    # --- Initial Parse for Config Path ---
    args_initial, _ = parser.parse_known_args()

    # --- Load Config, Update Defaults, Re-parse ---
    args = load_config_and_reparse(parser, args_initial)

    # --- Setup Logging ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file # Use final path from args
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

    # --- Find Files ---
    # Use final args values after potential config override
    input_files = find_files(args.input_dir, args.pattern, exclude_pattern=args.exclude)
    if not input_files:
        logger.warning("No input files found matching criteria. Exiting.")
        sys.exit(0)

    # --- Determine Parallelism ---
    if args.n_workers == -1: max_workers = cpu_count()
    else: max_workers = max(1, args.n_workers)
    logger.info(f"Processing {len(input_files)} files using up to {max_workers} worker processes.")

    # --- Prepare and Execute Tasks ---
    start_time_overall = time.perf_counter()
    success_count, fail_count, tasks_submitted = 0, 0, 0

    tasks = [(f, args.output_dir, args.smiles_col, args.id_col, args.delimiter) for f in input_files]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = { executor.submit(process_file_wrapper, *task_args): task_args[0] for task_args in tasks }
        tasks_submitted = len(future_to_file)
        logger.info(f"Submitted {tasks_submitted} preprocessing tasks.")

        with tqdm(total=tasks_submitted, desc="Preprocessing Files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                input_file = future_to_file[future]
                try:
                    _, success = future.result()
                    if success: success_count += 1
                    else: fail_count += 1
                except Exception as exc:
                    fail_count += 1
                    logger.error(f"Task for {input_file} generated exception: {exc}", exc_info=True)
                pbar.update(1)

    end_time_overall = time.perf_counter()
    logger.info(f"Preprocessing finished in {end_time_overall - start_time_overall:.2f} seconds.")
    logger.info(f"Successfully processed: {success_count} files.")
    logger.info(f"Failed to process: {fail_count} files.")

    if fail_count > 0:
        logger.warning("Some files failed preprocessing. Check logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()
