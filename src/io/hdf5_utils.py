# src/io/hdf5_utils.py
import h5py
import os
import glob
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def aggregate_hdf5_datasets(
    input_dir: str,
    output_file: str,
    file_pattern: str = "*.h5",
    recursive: bool = True,
    datasets_to_aggregate: list[str] | None = None, # Specify datasets or aggregate all
    compression: str | None = "gzip",
    compression_opts: int = 4
    ) -> bool:
    """
    Aggregates specified datasets from multiple HDF5 files into a single output HDF5 file.

    Finds files matching the pattern in the input directory (recursively or not)
    and appends data from the specified datasets. Datasets must have compatible
    shapes (except for the first dimension) and dtypes across files.

    Args:
        input_dir (str): Directory containing input HDF5 files.
        output_file (str): Path to the aggregated output HDF5 file.
        file_pattern (str): Glob pattern to find input files. Defaults to "*.h5".
        recursive (bool): Whether to search directories recursively. Defaults to True.
        datasets_to_aggregate (list[str] | None): List of dataset names to aggregate.
                                                  If None, attempts to aggregate all datasets found
                                                  in the first file. Defaults to None.
        compression (str | None): Compression for output datasets. Defaults to "gzip".
        compression_opts (int): Compression level. Defaults to 4.

    Returns:
        bool: True if aggregation completes successfully, False otherwise.
    """
    logger.info(f"--- Starting HDF5 Aggregation ---")
    logger.info(f"Input Directory: {input_dir}")
    logger.info(f"File Pattern: {file_pattern}")
    logger.info(f"Output File: {output_file}")
    logger.info(f"Recursive Search: {recursive}")

    # --- Find Input Files ---
    search_path = os.path.join(input_dir, '**', file_pattern) if recursive else os.path.join(input_dir, file_pattern)
    try:
        input_files = glob(search_path, recursive=recursive)
        if not input_files:
            logger.warning(f"No files found matching '{search_path}'. Aborting aggregation.")
            return False
        logger.info(f"Found {len(input_files)} files to potentially aggregate.")
    except Exception as e:
        logger.error(f"Error finding input files: {e}", exc_info=True)
        return False

    # --- Prepare Output File and Datasets ---
    first_file = True
    output_datasets = {} # Store handles to output datasets

    try:
        # Ensure output directory exists
        output_dir_path = os.path.dirname(output_file)
        if output_dir_path and not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
            logger.info(f"Created output directory: {output_dir_path}")

        with h5py.File(output_file, 'w') as out_h5f:
            logger.info(f"Opened output file for writing: {output_file}")

            # --- Process Input Files ---
            with tqdm(total=len(input_files), desc="Aggregating HDF5", unit="file") as pbar:
                for file_path in input_files:
                    logger.debug(f"Processing file: {file_path}")
                    try:
                        with h5py.File(file_path, 'r') as in_h5f:
                            # Determine datasets to process
                            if first_file:
                                if datasets_to_aggregate is None:
                                    # Use all datasets from the first file if not specified
                                    datasets_to_aggregate = list(in_h5f.keys())
                                    logger.info(f"Aggregating all datasets found in first file: {datasets_to_aggregate}")
                                else:
                                    # Validate specified datasets exist in the first file
                                    valid_datasets = [ds for ds in datasets_to_aggregate if ds in in_h5f]
                                    if len(valid_datasets) != len(datasets_to_aggregate):
                                        missing = set(datasets_to_aggregate) - set(valid_datasets)
                                        logger.warning(f"Specified datasets not found in first file: {missing}. Skipping them.")
                                    datasets_to_aggregate = valid_datasets
                                    logger.info(f"Aggregating specified datasets: {datasets_to_aggregate}")

                                if not datasets_to_aggregate:
                                     logger.error("No valid datasets specified or found to aggregate. Aborting.")
                                     return False # Abort if no datasets to aggregate

                                # Create datasets in the output file based on the first file
                                for ds_name in datasets_to_aggregate:
                                    in_ds = in_h5f[ds_name]
                                    shape = in_ds.shape
                                    # Allow first dimension to be resizable (None)
                                    maxshape = (None,) + shape[1:] if len(shape) > 0 else (None,)
                                    out_ds = out_h5f.create_dataset(
                                        ds_name,
                                        shape=shape,
                                        maxshape=maxshape,
                                        dtype=in_ds.dtype,
                                        compression=compression,
                                        compression_opts=compression_opts
                                        # Add chunking? Auto-chunking might be okay.
                                    )
                                    output_datasets[ds_name] = out_ds
                                    logger.debug(f"Created output dataset '{ds_name}' with shape {shape}, maxshape {maxshape}, dtype {in_ds.dtype}")
                                first_file = False # Mark first file as processed

                            # Append data from current file
                            for ds_name in datasets_to_aggregate:
                                if ds_name in in_h5f and ds_name in output_datasets:
                                    in_ds = in_h5f[ds_name]
                                    out_ds = output_datasets[ds_name]

                                    # Check compatibility (dtype checked implicitly by HDF5 on write, shapes mostly here)
                                    if len(in_ds.shape) != len(out_ds.shape) or (len(in_ds.shape) > 1 and in_ds.shape[1:] != out_ds.shape[1:]):
                                         logger.warning(f"Incompatible shape for dataset '{ds_name}' in {file_path} ({in_ds.shape}) vs output ({out_ds.shape}). Skipping append.")
                                         continue # Skip appending this dataset from this file

                                    if in_ds.shape[0] > 0: # Only append if there's data
                                        try:
                                            current_len = out_ds.shape[0]
                                            append_len = in_ds.shape[0]
                                            # Resize the output dataset
                                            out_ds.resize((current_len + append_len,) + out_ds.shape[1:])
                                            # Append data directly using slicing
                                            # Read source data in chunks if it's huge? For now, direct read/write.
                                            out_ds[current_len:] = in_ds[...] # Read all data from input dataset
                                            logger.debug(f"Appended {append_len} items to dataset '{ds_name}'. New shape: {out_ds.shape}")
                                        except Exception as append_err:
                                             logger.error(f"Failed to append data for dataset '{ds_name}' from {file_path}: {append_err}", exc_info=True)
                                             # Decide: continue or abort? Continue for now.
                                else:
                                     logger.warning(f"Dataset '{ds_name}' not found or empty in {file_path}. Skipping append.")
                    except Exception as file_err:
                        logger.error(f"Failed to read or process file {file_path}: {file_err}", exc_info=True)
                        # Decide: continue or abort? Continue for now.
                    pbar.update(1) # Update progress bar

        logger.info(f"--- HDF5 Aggregation Finished ---")
        logger.info(f"Aggregated data saved to: {output_file}")
        final_shapes = {name: ds.shape for name, ds in output_datasets.items()}
        logger.info(f"Final dataset shapes: {final_shapes}")
        return True

    except Exception as e:
        logger.error(f"Failed to create or write to output file {output_file}: {e}", exc_info=True)
        # Clean up potentially partially written file
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                logger.warning(f"Removed partially written file: {output_file}")
            except OSError as rm_err:
                logger.error(f"Could not remove partially written file {output_file}: {rm_err}")
        return False
