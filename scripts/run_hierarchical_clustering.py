# scripts/run_hierarchical_clustering.py
import argparse
import logging
import os
import sys
import time
import glob
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import numpy as np # For dtype conversion

# --- Setup sys.path ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from src.common.logging_utils import setup_logging
    from src.common.timing_utils import timing
    # Import necessary functions from refactored modules
    from src.io.hdf5_utils import aggregate_hdf5_datasets
    from src.clustering.cluster_stage1 import run_clustering_stage1
    from src.clustering.cluster_stage2 import run_clustering_stage2_single_file
except ImportError as e:
    print(f"Error importing project modules: {e}. Ensure script is run from correct directory or PYTHONPATH is set.")
    sys.exit(1)

logger = logging.getLogger(__name__)

# --- Main Pipeline Function ---
@timing
def run_pipeline(args):
    """Executes the hierarchical clustering pipeline stages."""

    # --- Stage 0: Aggregate Initial Fingerprints (Optional) ---
    # Assumes fingerprints were saved individually by generate_fingerprints.py
    # If generate_fingerprints.py already created one large file, this can be skipped.
    aggregated_fp_file = args.aggregated_fp_input_file # Use this as the input for Stage 1

    if args.run_fp_aggregation:
        logger.info("--- Running Initial Fingerprint Aggregation ---")
        # Define pattern for individual FP files, assume they are in args.fp_dir
        if not args.fp_dir or not os.path.isdir(args.fp_dir):
            logger.error(f"Fingerprint directory (--fp_dir) must be specified and exist for aggregation.")
            return False
        fp_pattern = args.fp_file_pattern

        # Define output path for the aggregated file
        aggregated_fp_file = os.path.join(args.work_dir, "fingerprints_aggregated.h5")
        logger.info(f"Aggregating '{fp_pattern}' from {args.fp_dir} into {aggregated_fp_file}")

        agg_success = aggregate_hdf5_datasets(
            input_dir=args.fp_dir,
            output_file=aggregated_fp_file,
            file_pattern=fp_pattern,
            recursive=True, # Assuming fingerprints might be in subdirs
            datasets_to_aggregate=['fp_list', 'smiles_list', 'name_list'], # Specify datasets
            compression=args.compression,
            compression_opts=args.compress_level
        )
        if not agg_success:
            logger.error("Initial fingerprint aggregation failed. Aborting pipeline.")
            return False
        logger.info("--- Finished Initial Fingerprint Aggregation ---")
    elif not os.path.isfile(aggregated_fp_file):
         logger.error(f"Aggregated fingerprint input file not found: {aggregated_fp_file}. Run with --run_fp_aggregation or provide correct path via --aggregated_fp_input_file.")
         return False
    else:
         logger.info(f"--- Skipping Initial Fingerprint Aggregation (using existing file: {aggregated_fp_file}) ---")


    # --- Stage 1: Initial Clustering ---
    stage1_output_dir = os.path.join(args.work_dir, "stage1_clusters")
    logger.info(f"--- Running Clustering Stage 1 ---")
    logger.warning(f"Ensure sufficient RAM is available on this node for Stage 1 processing of {aggregated_fp_file}!")

    stage1_success = run_clustering_stage1(
        input_h5_path=aggregated_fp_file,
        output_dir=stage1_output_dir,
        n_clusters_k1=args.k1,
        faiss_n_init=args.s1_n_init,
        faiss_max_iter=args.s1_max_iter,
        faiss_seed=args.seed,
        faiss_verbose=args.verbose, # Use general verbosity flag?
        output_fp_dtype=np.dtype(args.fp_dtype),
        compression=args.compression,
        compression_opts=args.compress_level
    )
    if not stage1_success:
        logger.error("Clustering Stage 1 failed. Aborting pipeline.")
        return False
    logger.info(f"--- Finished Clustering Stage 1 (Output clusters in: {stage1_output_dir}) ---")

    # --- Stage 2: Sub-Clustering ---
    stage2_output_dir = os.path.join(args.work_dir, "stage2_results")
    logger.info(f"--- Running Clustering Stage 2 ---")

    # Find cluster files from stage 1
    stage1_cluster_files = glob.glob(os.path.join(stage1_output_dir, 'cluster_*.h5'))
    if not stage1_cluster_files:
        logger.warning("No cluster files found from Stage 1. Skipping Stage 2.")
        return True # Pipeline finished, just no Stage 2 work needed

    logger.info(f"Found {len(stage1_cluster_files)} cluster files from Stage 1 to process for Stage 2.")

    # Determine parallelism for Stage 2 (processing files in parallel)
    if args.n_workers_stage2 == -1:
        max_workers_s2 = cpu_count()
    else:
        max_workers_s2 = max(1, args.n_workers_stage2)
    logger.info(f"Using up to {max_workers_s2} workers for Stage 2 file processing.")

    success_count_s2 = 0
    fail_count_s2 = 0
    tasks_submitted_s2 = 0

    # Use ProcessPoolExecutor to run stage 2 on multiple cluster files concurrently
    with ProcessPoolExecutor(max_workers=max_workers_s2) as executor:
        future_to_file_s2 = {}
        for cluster_file in stage1_cluster_files:
            future = executor.submit(
                run_clustering_stage2_single_file,
                input_cluster_h5_path=cluster_file,
                output_results_dir=stage2_output_dir,
                k2_determination_method=args.k2_method,
                k2_value=args.k2_value,
                base_k=args.k2_base_k, # Pass base_k
                faiss_n_init=args.s2_n_init,
                faiss_max_iter=args.s2_max_iter,
                faiss_seed=args.seed, # Use same seed? Or vary? Currently same.
                faiss_verbose=False, # Usually less verbose here
                output_fp_dtype=np.dtype(args.fp_dtype),
                compression=args.compression,
                compression_opts=args.compress_level
            )
            future_to_file_s2[future] = cluster_file

        tasks_submitted_s2 = len(future_to_file_s2)
        logger.info(f"Submitted {tasks_submitted_s2} Stage 2 clustering tasks.")

        # Process completed futures with progress
        with tqdm(total=tasks_submitted_s2, desc="Running Stage 2 Clustering", unit="file") as pbar:
            for future in as_completed(future_to_file_s2):
                input_cfile = future_to_file_s2[future]
                try:
                    success = future.result()
                    if success:
                        success_count_s2 += 1
                    else:
                        fail_count_s2 += 1
                        logger.warning(f"Stage 2 task failed for input file: {input_cfile}")
                except Exception as exc:
                    fail_count_s2 += 1
                    logger.error(f"Stage 2 task for {input_cfile} generated an exception: {exc}", exc_info=True)
                pbar.update(1)

    logger.info(f"Stage 2 processing finished. Success: {success_count_s2}, Failed: {fail_count_s2}")
    if fail_count_s2 > 0:
        logger.error("One or more Stage 2 sub-clustering tasks failed. Check logs. Cannot proceed to final aggregation.")
        return False # Abort if stage 2 had errors

    logger.info(f"--- Finished Clustering Stage 2 (Results in: {stage2_output_dir}) ---")


    # --- Stage 3: Aggregate Final Representatives ---
    logger.info(f"--- Running Final Representative Aggregation ---")
    final_centers_file = os.path.join(args.work_dir, "representatives_final.h5")

    # Aggregate the 'centers_cluster_*.h5' files from the stage2 output directory
    agg_centers_success = aggregate_hdf5_datasets(
        input_dir=stage2_output_dir, # Directory containing centers_cluster_*.h5 files
        output_file=final_centers_file,
        file_pattern="centers_cluster_*.h5", # Pattern for center files
        recursive=False, # Assume they are directly in stage2_output_dir
        datasets_to_aggregate=['fp_list', 'smiles_list', 'name_list'], # Datasets in centers files
        compression=args.compression,
        compression_opts=args.compress_level
    )

    if not agg_centers_success:
        logger.error("Final representative aggregation failed. Check logs.")
        return False

    logger.info(f"Final aggregated representatives saved to: {final_centers_file}")
    logger.info(f"--- Finished Final Representative Aggregation ---")
    return True # Overall pipeline success


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Run the full Hierarchical Cheminformatics Clustering pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file argument - added first
    parser.add_argument("-c", "--config", help="Path to YAML configuration file.")

    # --- Paths Group ---
    path_group = parser.add_argument_group('Input/Output Paths')
    path_group.add_argument("--aggregated_fp_input_file", help="Path to the input HDF5 file containing all fingerprints.")
    path_group.add_argument("--work_dir", help="Working directory for intermediate and final results.")
    # Arguments below are more relevant to generate_fingerprints.py but might be in config
    path_group.add_argument("--fp_dir", help="[Config Only/Informational] Directory containing individual fingerprint HDF5 files.")
    path_group.add_argument("--fp_file_pattern", help="[Config Only/Informational] Pattern for individual fingerprint files.")

    # --- Stage 1 Settings Group ---
    s1_group = parser.add_argument_group('Stage 1 Clustering Settings')
    s1_group.add_argument("--k1", type=int, help="Number of initial clusters for Stage 1.")
    s1_group.add_argument("--s1_n_init", type=int, default=1, help="FAISS n_init/nredo for Stage 1 K-Means.")
    s1_group.add_argument("--s1_max_iter", type=int, default=25, help="FAISS max_iter for Stage 1 K-Means.")

    # --- Stage 2 Settings Group ---
    s2_group = parser.add_argument_group('Stage 2 Clustering Settings')
    s2_group.add_argument("--k2_method", choices=['ratio', 'sqrt', 'fixed'], default='ratio', help="Method to determine number of sub-clusters (k2) in Stage 2.")
    s2_group.add_argument("--k2_value", type=float, default=8.0, help="Value used for k2 determination (ratio divisor or fixed k).")
    s2_group.add_argument("--k2_base_k", type=int, default=2, help="Minimum number of sub-clusters per Stage 1 cluster.")
    s2_group.add_argument("--s2_n_init", type=int, default=1, help="FAISS n_init/nredo for Stage 2 K-Means.")
    s2_group.add_argument("--s2_max_iter", type=int, default=25, help="FAISS max_iter for Stage 2 K-Means.")
    s2_group.add_argument("--n_workers_stage2", type=int, default=-1, help="Number of workers for parallel processing of Stage 1 cluster files in Stage 2 (-1 for all cores).")

    # --- General Settings Group ---
    general_group = parser.add_argument_group('General Settings')
    general_group.add_argument("--seed", type=int, default=None, help="Random seed for FAISS K-Means for reproducibility.")
    general_group.add_argument("--fp_dtype", choices=['int8', 'int16', 'int32', 'int64'], default="int8", help="Numpy dtype for output fingerprints.")
    general_group.add_argument("--compression", choices=['gzip', 'lzf', 'none'], default="gzip", help="Compression for HDF5 output files.")
    general_group.add_argument("--compress_level", type=int, default=4, help="Compression level (1-9 for gzip).")

    # --- Logging Group ---
    log_group = parser.add_argument_group('Logging Settings')
    log_group.add_argument("--log_file", default="logs/pipeline.log", help="Path to the main pipeline log file.")
    log_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    # --- Initial Parse for Config Path ---
    args_initial, _ = parser.parse_known_args()

    # --- Load Config, Update Defaults, Re-parse ---
    args = load_config_and_reparse(parser, args_initial) # This function now defined in src/common/utils.py

    # --- Setup Logging (using final args) ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file # Use final log_file path from args
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        try: os.makedirs(log_dir, exist_ok=True)
        except OSError as e: print(f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr)
    setup_logging(log_file, level=log_level) # Function defined in src/common/logging_utils.py

    # --- Final Argument Validation ---
    required_args_list = ['aggregated_fp_input_file', 'work_dir', 'k1']
    missing_args_list = [arg for arg in required_args_list if getattr(args, arg, None) is None]
    if missing_args_list:
        parser.error(f"Missing required arguments: {', '.join(missing_args_list)}")

    if not os.path.isfile(args.aggregated_fp_input_file):
         parser.error(f"Aggregated fingerprint input file not found: {args.aggregated_fp_input_file}")

    # --- Start Pipeline ---
    logger.info("Starting Hierarchical Clustering Pipeline...")
    # Log final effective arguments (load_config_and_reparse already does this if logging is set up)
    # logger.info("Effective Arguments:")
    # for k, v in sorted(vars(args).items()):
    #    logger.info(f"  {k}: {v}")

    overall_start_time = time.perf_counter()

    # Ensure working directory exists
    try:
        os.makedirs(args.work_dir, exist_ok=True)
        logger.info(f"Using working directory: {args.work_dir}")
    except OSError as e:
         logger.error(f"Cannot create or access working directory {args.work_dir}: {e}")
         sys.exit(1)

    # --- Execute main pipeline function ---
    try:
        pipeline_success = run_pipeline(args) # Pass the final 'args'
    except Exception as e:
         logger.critical(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
         pipeline_success = False

    overall_end_time = time.perf_counter()
    logger.info(f"Pipeline finished in {overall_end_time - overall_start_time:.2f} seconds.")

    if not pipeline_success:
        logger.error("Pipeline execution failed.")
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully.")
        sys.exit(0)
